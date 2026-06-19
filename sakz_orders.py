"""sakz_orders.py - Execution safety layer (PHASE 2, TESTNET-ONLY by default).

The disciplined order path between a signal and the exchange. Conservative by
design: every order passes ALL pre-trade gates, is written to a durable ledger
BEFORE it is sent (idempotency), and the live backend REFUSES mainnet unless
explicitly opted in.

    signal -> PreTradeGate -> OrderRouter -> { PaperBackend | LiveBackend }
                                  |
                              live_orders ledger (DB = source of truth)

Risk math is reused from sakz_risk (fractional Kelly, drawdown gate,
correlation-aware cap, win-rate from outcomes); nothing is re-implemented here.
The live transport (`signed_request`), key cipher, instrument metadata and
server clock are all INJECTABLE, so the whole module is unit-testable offline
with fakes - no network, no real keys.

Features
--------
* Order ledger + control + per-user key vault + realized-PnL ledger tables.
* Kill switch (manual + auto on drawdown / daily-loss breach).
* PreTradeGate: kill switch -> fields/side -> SL side -> daily-loss limit ->
  drawdown budget -> max open -> size (fractional Kelly, win-rate from outcomes)
  clamped to per-order / per-symbol / total notional caps.
* Per-user encrypted API keys (Fernet via sakz_execution, cipher injectable).
* Idempotent LiveBackend (Bybit V5), tick/lot rounding, validated retCode,
  testnet-only unless SAKZ_ALLOW_MAINNET, plus a reconciliation pass.
* Server-time sync helper to avoid recv_window drift.

Env (all optional; safe defaults):
  SAKZ_MAX_OPEN_POSITIONS=5  SAKZ_MAX_NOTIONAL_PER_ORDER=50
  SAKZ_MAX_NOTIONAL_PER_SYMBOL=100  SAKZ_MAX_TOTAL_NOTIONAL=250
  SAKZ_DAILY_LOSS_LIMIT_PCT=0.05  SAKZ_DRAWDOWN_BUDGET=0.15
  SAKZ_KELLY_FRACTION=0.25  SAKZ_KELLY_CAP=0.25
  SAKZ_WINRATE_MIN_SAMPLE=10  SAKZ_MIN_NOTIONAL=1.0
  SAKZ_ALLOW_MAINNET=(unset -> mainnet sends refused)
"""
import os
import math
import time
import uuid
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional

import sakz_risk as risk

logger = logging.getLogger(__name__)


class SakzExecError(Exception):
    """Raised when an exchange response is missing/!=retCode 0, or a send fails."""


# --------------------------------------------------------------------------- #
# Config helpers                                                               #
# --------------------------------------------------------------------------- #
def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _envi(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def live_trading_allowed() -> bool:
    """Mainnet (real-money) sends require an explicit, loud opt-in."""
    return os.environ.get("SAKZ_ALLOW_MAINNET", "").strip().lower() in ("1", "true", "yes", "on")


OPEN_STATUSES = ("pending", "submitted", "open")


# --------------------------------------------------------------------------- #
# Schema (this module owns its own tables, like sakz_paper)                    #
# --------------------------------------------------------------------------- #
_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_orders (
    order_link_id TEXT PRIMARY KEY,
    chat_id       INTEGER,
    symbol        TEXT NOT NULL,
    exchange      TEXT NOT NULL DEFAULT 'BYBIT',
    side          TEXT NOT NULL,
    qty           REAL NOT NULL,
    notional      REAL,
    entry_price   REAL,
    stop_loss     REAL,
    take_profit   REAL,
    status        TEXT NOT NULL DEFAULT 'pending',
    exch_order_id TEXT,
    filled_qty    REAL DEFAULT 0,
    avg_price     REAL,
    testnet       INTEGER NOT NULL DEFAULT 1,
    signal_id     TEXT,
    reason        TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT
);
"""

_CONTROL_SCHEMA = """
CREATE TABLE IF NOT EXISTS trading_control (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);
"""

_KEYS_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_api_keys (
    chat_id        INTEGER NOT NULL,
    exchange       TEXT NOT NULL DEFAULT 'BYBIT',
    api_key_enc    TEXT NOT NULL,
    api_secret_enc TEXT NOT NULL,
    testnet        INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    updated_at     TEXT,
    PRIMARY KEY (chat_id, exchange)
);
"""

_PNL_SCHEMA = """
CREATE TABLE IF NOT EXISTS pnl_ledger (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id  INTEGER,
    symbol   TEXT,
    realized REAL NOT NULL,
    ts       TEXT NOT NULL
);
"""


def orders_init_db(db_connect: Callable) -> None:
    """Create ledger / control / key-vault / pnl tables + indexes. Idempotent."""
    conn = db_connect()
    try:
        conn.execute(_LEDGER_SCHEMA)
        conn.execute(_CONTROL_SCHEMA)
        conn.execute(_KEYS_SCHEMA)
        conn.execute(_PNL_SCHEMA)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lo_status ON live_orders(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lo_symbol ON live_orders(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pnl_ts ON pnl_ledger(ts)")
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Kill switch                                                                  #
# --------------------------------------------------------------------------- #
def set_kill_switch(db_connect: Callable, halted: bool, reason: str = "") -> None:
    conn = db_connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO trading_control (key, value, updated_at) VALUES ('halted', ?, ?)",
            ("1" if halted else "0", _now()),
        )
        conn.execute(
            "INSERT OR REPLACE INTO trading_control (key, value, updated_at) VALUES ('halt_reason', ?, ?)",
            (reason, _now()),
        )
        conn.commit()
        logger.warning("[orders] kill switch %s%s", "ENGAGED" if halted else "released",
                       f" ({reason})" if reason else "")
    finally:
        conn.close()


def is_halted(db_connect: Callable) -> bool:
    conn = db_connect()
    try:
        c = conn.cursor()
        c.execute("SELECT value FROM trading_control WHERE key='halted'")
        row = c.fetchone()
        return bool(row) and str(row[0]) == "1"
    finally:
        conn.close()


def halt_reason(db_connect: Callable) -> str:
    conn = db_connect()
    try:
        c = conn.cursor()
        c.execute("SELECT value FROM trading_control WHERE key='halt_reason'")
        row = c.fetchone()
        return (row[0] if row and row[0] else "") or ""
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Realized PnL ledger + daily-loss limit                                       #
# --------------------------------------------------------------------------- #
def record_realized_pnl(db_connect: Callable, realized: float, symbol=None, chat_id=None) -> None:
    conn = db_connect()
    try:
        conn.execute(
            "INSERT INTO pnl_ledger (chat_id, symbol, realized, ts) VALUES (?,?,?,?)",
            (chat_id, (symbol or "").upper() or None, float(realized), _now()),
        )
        conn.commit()
    finally:
        conn.close()


def daily_realized_pnl(db_connect: Callable, day: Optional[str] = None) -> float:
    day = day or _today()
    conn = db_connect()
    try:
        c = conn.cursor()
        c.execute("SELECT COALESCE(SUM(realized), 0) FROM pnl_ledger WHERE substr(ts,1,10)=?", (day,))
        row = c.fetchone()
        return float(row[0] or 0.0)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Win rate from historical outcomes (signal_outcomes)                          #
# --------------------------------------------------------------------------- #
def win_rate_for_symbol(db_connect: Callable, symbol: str, exchange: Optional[str] = None,
                        min_sample: Optional[int] = None, default: float = 0.55):
    """Return (win_rate, sample_size). Falls back to `default` below min sample."""
    min_sample = _envi("SAKZ_WINRATE_MIN_SAMPLE", 10) if min_sample is None else min_sample
    conn = db_connect()
    try:
        c = conn.cursor()
        if exchange:
            c.execute("SELECT outcome FROM signal_outcomes WHERE symbol=? AND exchange=?",
                      (symbol.upper(), exchange.upper()))
        else:
            c.execute("SELECT outcome FROM signal_outcomes WHERE symbol=?", (symbol.upper(),))
        outcomes = [r[0] for r in c.fetchall() if r[0] and r[0] != "pending"]
    except Exception:
        return default, 0
    finally:
        conn.close()
    if len(outcomes) < min_sample:
        return default, len(outcomes)
    return risk.win_rate_from_outcomes(outcomes), len(outcomes)


# --------------------------------------------------------------------------- #
# Per-user encrypted API key vault                                             #
# --------------------------------------------------------------------------- #
def _default_encrypt(plaintext: str) -> str:
    from sakz_execution import encrypt_secret  # lazy import (needs Fernet + vault key)
    return encrypt_secret(plaintext)


def _default_decrypt(token: str) -> str:
    from sakz_execution import decrypt_secret
    return decrypt_secret(token)


def store_user_keys(db_connect: Callable, chat_id: int, api_key: str, api_secret: str,
                    *, exchange: str = "BYBIT", testnet: bool = True,
                    encrypt: Optional[Callable] = None) -> None:
    enc = encrypt or _default_encrypt
    conn = db_connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO user_api_keys
               (chat_id, exchange, api_key_enc, api_secret_enc, testnet, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (chat_id, exchange.upper(), enc(api_key), enc(api_secret),
             1 if testnet else 0, _now(), _now()),
        )
        conn.commit()
    finally:
        conn.close()


def load_user_keys(db_connect: Callable, chat_id: int, *, exchange: str = "BYBIT",
                   decrypt: Optional[Callable] = None):
    dec = decrypt or _default_decrypt
    conn = db_connect()
    try:
        c = conn.cursor()
        c.execute("SELECT api_key_enc, api_secret_enc FROM user_api_keys WHERE chat_id=? AND exchange=?",
                  (chat_id, exchange.upper()))
        row = c.fetchone()
    finally:
        conn.close()
    if not row:
        raise SakzExecError(f"no API keys stored for chat_id={chat_id} on {exchange}")
    return dec(row[0]), dec(row[1])


def make_key_getter(db_connect: Callable, *, exchange: str = "BYBIT",
                    decrypt: Optional[Callable] = None) -> Callable:
    """Build a `get_keys(chat_id) -> (api_key, api_secret)` for LiveBackend."""
    def _getter(chat_id):
        return load_user_keys(db_connect, chat_id, exchange=exchange, decrypt=decrypt)
    return _getter


# --------------------------------------------------------------------------- #
# Server-time sync (avoids recv_window drift)                                  #
# --------------------------------------------------------------------------- #
def compute_time_offset(server_ms, local_ms) -> int:
    return int(server_ms) - int(local_ms)


class ServerClock:
    """Tracks the offset between exchange server time and local time.

    `get_server_ms` is injectable so this is testable offline.
    """

    def __init__(self, get_server_ms: Callable, local_ms: Optional[Callable] = None):
        self.get_server_ms = get_server_ms
        self.local_ms = local_ms or (lambda: int(time.time() * 1000))
        self.offset_ms = 0

    def sync(self) -> int:
        self.offset_ms = compute_time_offset(self.get_server_ms(), self.local_ms())
        return self.offset_ms

    def now_ms(self) -> int:
        return int(self.local_ms()) + self.offset_ms


def sync_server_time(testnet: Optional[bool] = None) -> int:
    """Best-effort: query Bybit public server time and push the offset into
    sakz_execution so signed requests use a corrected timestamp. Returns the
    offset in ms (0 on failure). Network call - never raises."""
    try:
        import requests
        from sakz_execution import _base_url, set_time_offset_ms
        base = _base_url(testnet)
        local = int(time.time() * 1000)
        r = requests.get(f"{base}/v5/market/time", timeout=10)
        data = r.json()
        server = int(data["result"]["timeNano"]) // 1_000_000
        offset = compute_time_offset(server, local)
        set_time_offset_ms(offset)
        logger.info("[orders] server-time offset synced: %d ms", offset)
        return offset
    except Exception as e:
        logger.warning("[orders] server-time sync failed (non-fatal): %s", e)
        return 0


# --------------------------------------------------------------------------- #
# Rounding helpers                                                             #
# --------------------------------------------------------------------------- #
def round_down_step(value: float, step: Optional[float]) -> float:
    """Floor `value` to a multiple of `step` (lot size / tick size)."""
    if not step or step <= 0:
        return value
    # add a tiny epsilon to counter float representation error before floor
    return math.floor((value + 1e-12) / step) * step


# --------------------------------------------------------------------------- #
# Side normalisation + open-exposure snapshot                                  #
# --------------------------------------------------------------------------- #
_BUY_WORDS = ("long", "buy", "bull")
_SELL_WORDS = ("short", "sell", "bear")


def normalize_side(bias: str) -> Optional[str]:
    b = (bias or "").strip().lower()
    if b in _BUY_WORDS:
        return "Buy"
    if b in _SELL_WORDS:
        return "Sell"
    return None


def _open_exposure(db_connect: Callable):
    """Return (open_count, total_notional, per_symbol dict, open_symbols list)."""
    conn = db_connect()
    try:
        c = conn.cursor()
        marks = ",".join("?" for _ in OPEN_STATUSES)
        c.execute(f"SELECT symbol, notional FROM live_orders WHERE status IN ({marks})",
                  tuple(OPEN_STATUSES))
        rows = c.fetchall()
    finally:
        conn.close()
    per_symbol = {}
    total = 0.0
    for r in rows:
        sym = r[0]
        notional = float(r[1] or 0.0)
        per_symbol[sym] = per_symbol.get(sym, 0.0) + notional
        total += notional
    return len(rows), total, per_symbol, list(per_symbol.keys())


# --------------------------------------------------------------------------- #
# PreTradeGate                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class GateResult:
    ok: bool
    reason: str = ""
    side: str = ""
    qty: float = 0.0
    notional: float = 0.0
    reward_risk: float = 0.0
    win_rate: float = 0.0


class PreTradeGate:
    """Every order must pass ALL checks (each independently testable):

      1. kill switch engaged            -> reject
      2. required fields / valid side   -> reject
      3. stop-loss on the correct side  -> reject
      4. daily-loss limit exceeded      -> reject (+ auto-halt)
      5. drawdown budget exceeded       -> reject (+ auto-halt)
      6. max concurrent positions       -> reject
      7. position size (fractional Kelly, win-rate from outcomes) clamped to
         per-order / per-symbol / total notional caps -> reject if ~0
    """

    def __init__(self, db_connect: Callable, *, use_outcome_winrate: bool = True):
        self.db_connect = db_connect
        self.use_outcome_winrate = use_outcome_winrate
        self.max_open = _envi("SAKZ_MAX_OPEN_POSITIONS", 5)
        self.max_per_order = _envf("SAKZ_MAX_NOTIONAL_PER_ORDER", 50.0)
        self.max_per_symbol = _envf("SAKZ_MAX_NOTIONAL_PER_SYMBOL", 100.0)
        self.max_total = _envf("SAKZ_MAX_TOTAL_NOTIONAL", 250.0)
        self.kelly_fraction = _envf("SAKZ_KELLY_FRACTION", 0.25)
        self.kelly_cap = _envf("SAKZ_KELLY_CAP", 0.25)
        self.drawdown_budget = _envf("SAKZ_DRAWDOWN_BUDGET", 0.15)
        self.daily_loss_limit = _envf("SAKZ_DAILY_LOSS_LIMIT_PCT", 0.05)
        self.min_notional = _envf("SAKZ_MIN_NOTIONAL", 1.0)

    def evaluate(self, signal: dict, equity: float, *, peak_equity: Optional[float] = None,
                 win_rate: Optional[float] = None,
                 correlation: Optional[Mapping[tuple, float]] = None) -> GateResult:
        # 1. kill switch
        if is_halted(self.db_connect):
            return GateResult(False, "trading halted (kill switch)")

        # 2. required fields + valid side
        side = normalize_side(signal.get("bias", ""))
        if side is None:
            return GateResult(False, f"unknown/invalid bias: {signal.get('bias')!r}")
        symbol = (signal.get("symbol") or "").upper()
        entry = float(signal.get("entry_price") or signal.get("price") or 0)
        sl = float(signal.get("stop_loss") or 0)
        t1 = float(signal.get("t1") or 0)
        if not symbol or entry <= 0 or sl <= 0 or t1 <= 0:
            return GateResult(False, "incomplete signal (symbol/entry/sl/t1 required)")

        # 3. stop-loss on the correct side of entry
        if side == "Buy" and not (sl < entry):
            return GateResult(False, "stop-loss must be BELOW entry for a long")
        if side == "Sell" and not (sl > entry):
            return GateResult(False, "stop-loss must be ABOVE entry for a short")

        # 4. daily-loss limit
        if equity > 0 and self.daily_loss_limit > 0:
            today = daily_realized_pnl(self.db_connect)
            if today < 0 and (-today) >= self.daily_loss_limit * equity:
                self._auto_halt("daily loss limit exceeded")
                return GateResult(False, "daily loss limit exceeded -> trading halted")

        # 5. drawdown budget
        gate = risk.DrawdownGate(threshold=self.drawdown_budget, peak_equity=float(peak_equity or 0.0))
        if not gate.can_open(equity):
            self._auto_halt("drawdown budget exceeded")
            return GateResult(False, "drawdown budget exceeded -> trading halted")

        # 6. max concurrent positions
        open_count, total_notional, per_symbol, open_symbols = _open_exposure(self.db_connect)
        if open_count >= self.max_open:
            return GateResult(False, f"max open positions reached ({self.max_open})")

        # 7. position sizing via fractional Kelly
        risk_per_unit = abs(entry - sl)
        reward = abs(t1 - entry)
        reward_risk = (reward / risk_per_unit) if risk_per_unit > 0 else 0.0
        if win_rate is None:
            if self.use_outcome_winrate:
                wr, _n = win_rate_for_symbol(self.db_connect, symbol,
                                             exchange=signal.get("exchange"))
            else:
                wr = 0.55
        else:
            wr = float(win_rate)
        notional = risk.kelly_position_size(equity, wr, reward_risk,
                                            self.kelly_fraction, self.kelly_cap)
        if correlation is not None and open_symbols:
            notional = risk.correlated_exposure_cap(symbol, open_symbols, correlation, notional)

        remaining_symbol = max(0.0, self.max_per_symbol - per_symbol.get(symbol, 0.0))
        remaining_total = max(0.0, self.max_total - total_notional)
        notional = min(notional, self.max_per_order, remaining_symbol, remaining_total)

        if notional < self.min_notional:
            return GateResult(False, f"size below minimum after caps (notional={notional:.4f})",
                              side=side, reward_risk=reward_risk, win_rate=wr)

        qty = notional / entry
        return GateResult(True, "ok", side=side, qty=qty, notional=notional,
                          reward_risk=reward_risk, win_rate=wr)

    def _auto_halt(self, reason: str) -> None:
        try:
            set_kill_switch(self.db_connect, True, reason)
        except Exception as e:
            logger.error("[orders] auto-halt failed: %s", e)


# --------------------------------------------------------------------------- #
# Hardened transport wrapper                                                   #
# --------------------------------------------------------------------------- #
def validated_request(signed_request: Callable, *args, **kwargs) -> dict:
    """Call a Bybit signed-request transport and validate the envelope.

    Raises SakzExecError unless the response is a dict with retCode == 0.
    (The raw bybit_signed_request returns r.json() WITHOUT this check.)
    """
    resp = signed_request(*args, **kwargs)
    if not isinstance(resp, dict):
        raise SakzExecError(f"non-dict exchange response: {resp!r}")
    ret = resp.get("retCode")
    if ret != 0:
        raise SakzExecError(f"bybit retCode={ret}: {resp.get('retMsg', 'unknown')}")
    return resp


def _default_signed_request(*args, **kwargs):
    from sakz_execution import bybit_signed_request  # lazy import
    return bybit_signed_request(*args, **kwargs)


def _default_testnet() -> bool:
    try:
        from sakz_execution import BYBIT_TESTNET
        return bool(BYBIT_TESTNET)
    except Exception:
        return True  # fail safe


# Bybit order-status -> ledger-status mapping (for reconciliation)
_STATUS_MAP = {
    "Filled": "open",
    "PartiallyFilled": "submitted",
    "New": "submitted",
    "Created": "submitted",
    "Untriggered": "submitted",
    "Triggered": "submitted",
    "Cancelled": "canceled",
    "Deactivated": "canceled",
    "Rejected": "failed",
}


# --------------------------------------------------------------------------- #
# Backends                                                                     #
# --------------------------------------------------------------------------- #
class PaperBackend:
    """Delegates to the existing paper layer so paper + live share one path."""

    name = "paper"

    def __init__(self, db_connect: Callable, paper_open: Optional[Callable] = None):
        self.db_connect = db_connect
        self._paper_open = paper_open

    def place(self, signal: dict, gate: GateResult, chat_id=None, order_link_id=None) -> dict:
        open_fn = self._paper_open
        if open_fn is None:
            from sakz_paper import paper_maybe_open  # lazy import
            open_fn = paper_maybe_open
        opened = open_fn(signal, self.db_connect)
        return {"status": "open" if opened else "skipped", "backend": "paper"}


class LiveBackend:
    """Bybit V5 order backend. TESTNET by default; REFUSES mainnet unless opted in.

    `get_keys(chat_id) -> (api_key, api_secret)`, `signed_request`, and
    `get_instrument(symbol) -> {qtyStep, tickSize, minOrderQty}` are injectable
    so the backend is fully unit-testable offline.
    """

    name = "live"

    def __init__(self, db_connect: Callable, get_keys: Callable,
                 signed_request: Optional[Callable] = None, testnet: Optional[bool] = None,
                 category: str = "linear", get_instrument: Optional[Callable] = None):
        self.db_connect = db_connect
        self.get_keys = get_keys
        self.signed_request = signed_request or _default_signed_request
        self.testnet = _default_testnet() if testnet is None else bool(testnet)
        self.category = category
        self.get_instrument = get_instrument

    def _mainnet_ok(self) -> bool:
        return True if self.testnet else live_trading_allowed()

    def _get_order(self, order_link_id: str) -> Optional[dict]:
        conn = self.db_connect()
        try:
            c = conn.cursor()
            c.execute("SELECT order_link_id, status, exch_order_id, symbol FROM live_orders WHERE order_link_id=?",
                      (order_link_id,))
            row = c.fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return {"order_link_id": row[0], "status": row[1], "exch_order_id": row[2], "symbol": row[3]}

    def _insert_pending(self, olid, signal, gate, chat_id) -> None:
        conn = self.db_connect()
        try:
            conn.execute(
                """INSERT INTO live_orders
                   (order_link_id, chat_id, symbol, exchange, side, qty, notional,
                    entry_price, stop_loss, take_profit, status, testnet, signal_id,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,'pending',?,?,?,?)""",
                (olid, chat_id, (signal.get("symbol") or "").upper(),
                 (signal.get("exchange") or "BYBIT").upper(), gate.side, gate.qty, gate.notional,
                 float(signal.get("entry_price") or signal.get("price") or 0),
                 float(signal.get("stop_loss") or 0), float(signal.get("t1") or 0),
                 1 if self.testnet else 0, str(signal.get("signal_id") or ""), _now(), _now()),
            )
            conn.commit()
        finally:
            conn.close()

    def _update_status(self, olid, status, exch_order_id=None, reason=None,
                       filled_qty=None, avg_price=None) -> None:
        conn = self.db_connect()
        try:
            conn.execute(
                "UPDATE live_orders SET status=?, "
                "exch_order_id=COALESCE(?, exch_order_id), reason=COALESCE(?, reason), "
                "filled_qty=COALESCE(?, filled_qty), avg_price=COALESCE(?, avg_price), "
                "updated_at=? WHERE order_link_id=?",
                (status, exch_order_id, reason, filled_qty, avg_price, _now(), olid),
            )
            conn.commit()
        finally:
            conn.close()

    def place(self, signal: dict, gate: GateResult, chat_id=None, order_link_id=None) -> dict:
        if not self._mainnet_ok():
            raise SakzExecError("mainnet trading disabled (set SAKZ_ALLOW_MAINNET to enable)")
        if not gate.ok:
            raise SakzExecError(f"refusing to place a rejected order: {gate.reason}")

        symbol = (signal.get("symbol") or "").upper()
        qty = gate.qty
        sl = float(signal.get("stop_loss") or 0)
        t1 = float(signal.get("t1") or 0)

        # tick/lot rounding via injected instrument metadata
        if self.get_instrument:
            info = self.get_instrument(symbol) or {}
            qty_step = info.get("qtyStep")
            tick = info.get("tickSize")
            min_qty = info.get("minOrderQty")
            qty = round_down_step(qty, qty_step)
            if tick:
                sl = round_down_step(sl, tick) if sl > 0 else sl
                t1 = round_down_step(t1, tick) if t1 > 0 else t1
            if min_qty is not None and qty < float(min_qty):
                raise SakzExecError(f"qty {qty} below exchange minimum {min_qty} for {symbol}")
        if qty <= 0:
            raise SakzExecError(f"rounded qty collapsed to {qty} for {symbol}")

        olid = order_link_id or ("sakz-" + uuid.uuid4().hex[:20])

        # Idempotency: never re-send a link id that is already live.
        existing = self._get_order(olid)
        if existing and existing["status"] in ("submitted", "open", "closed"):
            return {"status": existing["status"], "order_link_id": olid,
                    "exch_order_id": existing["exch_order_id"], "recovered": True, "backend": "live"}
        if not existing:
            self._insert_pending(olid, signal, gate, chat_id)

        params = {
            "category": self.category, "symbol": symbol, "side": gate.side,
            "orderType": "Market", "qty": str(qty), "orderLinkId": olid,
        }
        if sl > 0:
            params["stopLoss"] = str(sl)
        if t1 > 0:
            params["takeProfit"] = str(t1)

        api_key, api_secret = self.get_keys(chat_id)
        try:
            resp = validated_request(self.signed_request, api_key, api_secret, "POST",
                                     "/v5/order/create", params, testnet=self.testnet)
        except Exception as e:
            self._update_status(olid, "failed", reason=str(e))
            raise

        exch_id = None
        try:
            exch_id = resp.get("result", {}).get("orderId")
        except Exception:
            pass
        self._update_status(olid, "submitted", exch_order_id=exch_id)
        return {"status": "submitted", "order_link_id": olid, "exch_order_id": exch_id,
                "backend": "live", "testnet": self.testnet, "qty": qty}

    def reconcile(self, chat_id=None) -> dict:
        """Query the exchange for every non-terminal ledger order and update its
        status/fills. Call on startup and periodically. Returns a summary."""
        conn = self.db_connect()
        try:
            c = conn.cursor()
            marks = ",".join("?" for _ in OPEN_STATUSES)
            c.execute(f"SELECT order_link_id, chat_id FROM live_orders WHERE status IN ({marks})",
                      tuple(OPEN_STATUSES))
            rows = c.fetchall()
        finally:
            conn.close()

        updated, errors = 0, 0
        for olid, row_chat in rows:
            try:
                api_key, api_secret = self.get_keys(row_chat if row_chat is not None else chat_id)
                resp = validated_request(self.signed_request, api_key, api_secret, "GET",
                                         "/v5/order/realtime",
                                         {"category": self.category, "orderLinkId": olid},
                                         testnet=self.testnet)
                lst = (resp.get("result", {}) or {}).get("list", []) or []
                if not lst:
                    continue
                o = lst[0]
                new_status = _STATUS_MAP.get(o.get("orderStatus"), None)
                if new_status:
                    self._update_status(
                        olid, new_status, exch_order_id=o.get("orderId"),
                        filled_qty=float(o.get("cumExecQty") or 0) or None,
                        avg_price=float(o.get("avgPrice") or 0) or None,
                    )
                    updated += 1
            except Exception as e:
                errors += 1
                logger.warning("[orders] reconcile %s failed: %s", olid, e)
        return {"checked": len(rows), "updated": updated, "errors": errors}


# --------------------------------------------------------------------------- #
# Router                                                                       #
# --------------------------------------------------------------------------- #
class OrderRouter:
    """Runs the gate, then hands the order to whichever backend is configured."""

    def __init__(self, db_connect: Callable, backend):
        self.db_connect = db_connect
        self.backend = backend
        self.gate = PreTradeGate(db_connect)

    def submit(self, signal: dict, *, equity: float, chat_id=None,
               peak_equity: Optional[float] = None, win_rate: Optional[float] = None,
               correlation: Optional[Mapping[tuple, float]] = None,
               order_link_id: Optional[str] = None) -> dict:
        result = self.gate.evaluate(signal, equity, peak_equity=peak_equity,
                                    win_rate=win_rate, correlation=correlation)
        if not result.ok:
            logger.info("[orders] rejected %s: %s", signal.get("symbol"), result.reason)
            return {"status": "rejected", "reason": result.reason, "backend": self.backend.name}
        return self.backend.place(signal, result, chat_id=chat_id, order_link_id=order_link_id)


# --------------------------------------------------------------------------- #
# Admin status text (for the /orders command)                                  #
# --------------------------------------------------------------------------- #
def orders_status_text(db_connect: Callable) -> str:
    halted = is_halted(db_connect)
    open_count, total_notional, per_symbol, _ = _open_exposure(db_connect)
    today = daily_realized_pnl(db_connect)
    lines = [
        ("\U0001F6D1 Trading: HALTED" if halted else "\u2705 Trading: ACTIVE"),
    ]
    if halted:
        reason = halt_reason(db_connect)
        if reason:
            lines.append("  reason: " + reason)
    lines.append(f"Open positions: {open_count}")
    lines.append(f"Open notional: {total_notional:.2f}")
    lines.append(f"Realized PnL today: {today:+.2f}")
    if per_symbol:
        lines.append("By symbol: " + ", ".join(f"{s} {n:.1f}" for s, n in per_symbol.items()))
    return "\n".join(lines)
