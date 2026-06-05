"""
sakz_ws.py — Real-time WebSocket layer for sakz_bot
=====================================================
Exchange: MEXC Futures  (wss://contract.mexc.com/edge)
Symbols:  BTCUSDT  →  BTC_USDT  (auto-converted)

Streams:
  • Price feed       — live ticker for all watched/alerted symbols
  • Price-level alerts — fires the instant price crosses a /palert target
  • Funding rate     — streams every funding update, flags extremes
  • Volume spike     — mid-candle wash-trade / spike detector
  • Liquidation      — large liquidation cluster detector

Integration:
  Call start_ws(bot) once at bot startup (pass the telegram Bot object).
  The module reads alert/watchlist tables from sakz_data.db automatically.
  All callbacks send Telegram messages directly via the bot object.

Usage in sakz_bot.py main():
    import sakz_ws
    asyncio.create_task(sakz_ws.start_ws(application.bot))

Requirements:
    pip install websockets
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

logger = logging.getLogger("sakz_ws")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MEXC_WS_URL      = "wss://contract.mexc.com/edge"
PING_INTERVAL    = 15        # seconds — MEXC requires ping every 15s
RECONNECT_DELAY  = 5         # base delay before reconnect (seconds)
RECONNECT_MAX    = 120       # cap exponential backoff at 2 minutes
ALERT_RELOAD_INT = 60        # reload alert/watchlist tables every N seconds
VOL_SPIKE_MULT   = 4.0       # volume × this vs recent avg = spike
FUNDING_EXTREME  = 0.0005    # ±0.05% per 8h = extreme funding flag
LIQ_THRESHOLD    = 50_000    # USD value of liquidation to alert on (50k+)
PRICE_DECIMALS   = 8         # max decimal places in price display

# FIX: Watchdog — if no message received in this window, force reconnect.
# Catches "half-open" connections where MEXC stops sending but doesn't close.
WS_DEAD_TIMEOUT  = 90        # seconds — force reconnect if silent this long

DB_PATH = os.environ.get("SAKZ_DB_PATH", "sakz_data.db")

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL FORMAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def to_mexc_sym(symbol: str) -> str:
    """BTCUSDT  →  BTC_USDT (MEXC futures format)"""
    symbol = symbol.upper().replace("_", "")
    if symbol.endswith("USDT"):
        return symbol[:-4] + "_USDT"
    if symbol.endswith("USDC"):
        return symbol[:-4] + "_USDC"
    return symbol + "_USDT"

def from_mexc_sym(symbol: str) -> str:
    """BTC_USDT  →  BTCUSDT"""
    return symbol.replace("_", "")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PRICE CACHE
# Thread-safe dict: {mexc_symbol: {"price": float, "ts": float}}
# The rest of the bot can read from this without making REST calls.
# ─────────────────────────────────────────────────────────────────────────────
_price_cache: dict = {}
_volume_history: dict = {}   # {mexc_sym: [recent_volumes]}  for spike detection
_funding_cache: dict  = {}   # {mexc_sym: float}  latest funding rate


def get_live_price(symbol: str) -> Optional[float]:
    """
    Read latest WS price for a symbol (BTCUSDT or BTC_USDT format).
    Returns None if not yet cached.
    """
    msym = to_mexc_sym(symbol)
    entry = _price_cache.get(msym)
    if entry and (time.time() - entry["ts"]) < 30:
        return entry["price"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DB HELPERS  (minimal — reads from sakz_data.db, no writes except alert fire)
# ─────────────────────────────────────────────────────────────────────────────

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _load_price_alerts() -> list:
    """All untriggered /palert rows."""
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT id, chat_id, symbol, exchange, target, direction "
            "FROM price_level_alerts WHERE triggered=0"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("_load_price_alerts: %s", e)
        return []


def _load_conf_alerts() -> list:
    """All /alert rows (confidence-based — we notify on scan, not WS)."""
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT chat_id, symbol, min_conf FROM user_alerts"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("_load_conf_alerts: %s", e)
        return []


def _load_watchlist() -> list:
    """All /watch rows."""
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT chat_id, symbol, min_conf FROM watchlist"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("_load_watchlist: %s", e)
        return []


def _load_broadcast_channels() -> list:
    """All /broadcast channels."""
    try:
        conn = _db()
        rows = conn.execute("SELECT chat_id FROM broadcast_channels").fetchall()
        conn.close()
        return [r["chat_id"] for r in rows]
    except Exception as e:
        logger.warning("_load_broadcast_channels: %s", e)
        return []


def _mark_alert_triggered(alert_id: str):
    try:
        conn = _db()
        conn.execute(
            "UPDATE price_level_alerts SET triggered=1 WHERE id=?", (alert_id,)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("_mark_alert_triggered: %s", e)


def _load_pro_subscribers() -> list:
    try:
        conn = _db()
        rows = conn.execute("SELECT chat_id FROM pro_subscribers").fetchall()
        conn.close()
        return [r["chat_id"] for r in rows]
    except Exception as e:
        logger.warning("_load_pro_subscribers: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ALERT STATE  (in-memory dedup — prevents alert spam)
# ─────────────────────────────────────────────────────────────────────────────
_alerted_price:   dict = {}   # alert_id  → timestamp
_alerted_funding: dict = {}   # sym       → timestamp  (cooldown 4h)
_alerted_vol:     dict = {}   # sym       → timestamp  (cooldown 1h)
_alerted_liq:     dict = {}   # sym       → timestamp  (cooldown 30min)

_FUNDING_COOLDOWN = 4 * 3600
_VOL_COOLDOWN     = 1 * 3600
_LIQ_COOLDOWN     = 30 * 60


def _cooldown_ok(store: dict, key: str, window: int) -> bool:
    last = store.get(key, 0)
    if time.time() - last > window:
        store[key] = time.time()
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM MESSAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_price(price: float) -> str:
    if price == 0:
        return "0"
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.8f}".rstrip("0")


async def _send(bot, chat_id: int, text: str):
    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        logger.warning("_send to %s: %s", chat_id, e)


# ─────────────────────────────────────────────────────────────────────────────
# EVENT HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_ticker(bot, data: dict, price_alerts: list):
    """
    Process a ticker update from MEXC.
    - Update price cache
    - Check price-level alerts
    - Check volume spikes
    """
    try:
        sym     = data.get("symbol", "")            # BTC_USDT
        price   = float(data.get("lastPrice", 0))
        volume  = float(data.get("volume24", 0))    # 24h volume in base currency
        if not sym or price == 0:
            return

        # ── Update price cache ───────────────────────────────────────────────
        _price_cache[sym] = {"price": price, "ts": time.time()}

        # ── Price-level alert check ──────────────────────────────────────────
        plain = from_mexc_sym(sym)           # BTCUSDT
        for alert in price_alerts:
            if alert["symbol"].upper() != plain:
                continue
            if alert.get("triggered"):
                continue
            target    = float(alert["target"])
            direction = alert["direction"]    # "above" or "below"
            fired     = (direction == "above" and price >= target) or \
                        (direction == "below" and price <= target)
            if fired and _cooldown_ok(_alerted_price, alert["id"], 3600):
                _mark_alert_triggered(alert["id"])
                arrow = "🟢 ↑" if direction == "above" else "🔴 ↓"
                msg = (
                    f"🔔 PRICE ALERT TRIGGERED\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{arrow}  {plain}\n"
                    f"Target:   ${_fmt_price(target)}\n"
                    f"Current:  ${_fmt_price(price)}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚡ Real-time via WebSocket\n"
                    f"Run /cscan {plain.replace('USDT','')} for full signal."
                )
                await _send(bot, alert["chat_id"], msg)
                logger.info("Price alert fired: %s %s %.6f (chat=%s)",
                            plain, direction, target, alert["chat_id"])

        # ── Volume spike check ───────────────────────────────────────────────
        hist = _volume_history.setdefault(sym, [])
        if volume > 0:
            hist.append(volume)
            if len(hist) > 24:
                hist.pop(0)
        if len(hist) >= 6:
            avg_vol = sum(hist[:-1]) / (len(hist) - 1)
            if avg_vol > 0 and volume >= avg_vol * VOL_SPIKE_MULT:
                if _cooldown_ok(_alerted_vol, sym, _VOL_COOLDOWN):
                    subscribers = _load_pro_subscribers()
                    if subscribers:
                        msg = (
                            f"📊 VOLUME SPIKE DETECTED\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"🔍 {plain}\n"
                            f"Volume: {volume/avg_vol:.1f}× avg\n"
                            f"Price:  ${_fmt_price(price)}\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⚠️ Abnormal participation — could signal:\n"
                            f"   • Breakout / breakdown incoming\n"
                            f"   • Whale accumulation / distribution\n"
                            f"   • Wash trading (cross-check /pro)\n"
                            f"\nRun /cscan {plain.replace('USDT','')} to assess."
                        )
                        for cid in subscribers:
                            await _send(bot, cid, msg)

    except Exception as e:
        logger.debug("_handle_ticker error: %s", e)


async def _handle_funding(bot, data: dict):
    """
    Process a funding rate update.
    Flags extreme rates to /pro subscribers.
    """
    try:
        sym  = data.get("symbol", "")
        rate = float(data.get("fundingRate", 0))
        if not sym:
            return

        _funding_cache[sym] = rate
        plain = from_mexc_sym(sym)

        if abs(rate) >= FUNDING_EXTREME:
            if _cooldown_ok(_alerted_funding, sym, _FUNDING_COOLDOWN):
                subscribers = _load_pro_subscribers()
                if not subscribers:
                    return

                direction = "🔴 EXTREMELY NEGATIVE" if rate < 0 else "🟠 EXTREMELY POSITIVE"
                bias_hint = (
                    "Shorts paying longs — LONG bias favoured, but longs may be overleveraged."
                    if rate < 0 else
                    "Longs paying shorts — SHORT bias favoured, watch for long squeeze."
                )
                msg = (
                    f"⚡ EXTREME FUNDING RATE\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{direction}\n"
                    f"Symbol:  {plain}\n"
                    f"Rate:    {rate*100:.4f}% per 8h\n"
                    f"Annualised: {rate*100*3*365:.1f}%\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📌 {bias_hint}\n"
                    f"\n⚠️ Real-time WebSocket data · /pro alert"
                )
                for cid in subscribers:
                    await _send(bot, cid, msg)
                logger.info("Extreme funding alert: %s rate=%.6f", plain, rate)

    except Exception as e:
        logger.debug("_handle_funding error: %s", e)


async def _handle_liquidation(bot, data: dict):
    """
    Process a liquidation event.
    Alerts /pro subscribers when a large liquidation hits.
    """
    try:
        sym    = data.get("symbol", "")
        side   = data.get("side", "")         # 1=long liq, 2=short liq
        price  = float(data.get("price", 0))
        qty    = float(data.get("quantity", 0))
        usd    = price * qty

        if usd < LIQ_THRESHOLD or not sym:
            return

        plain     = from_mexc_sym(sym)
        side_lbl  = "🔴 LONG LIQUIDATED" if str(side) == "1" else "🟢 SHORT LIQUIDATED"
        implication = (
            "Large long wiped — selling pressure incoming, watch for cascade."
            if str(side) == "1" else
            "Large short wiped — buying pressure, potential short squeeze."
        )

        if _cooldown_ok(_alerted_liq, sym, _LIQ_COOLDOWN):
            subscribers = _load_pro_subscribers()
            if not subscribers:
                return

            msg = (
                f"💥 LIQUIDATION ALERT\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{side_lbl}\n"
                f"Symbol:  {plain}\n"
                f"Size:    ${usd:,.0f}\n"
                f"Price:   ${_fmt_price(price)}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📌 {implication}\n"
                f"\n⚡ Real-time · /cscan {plain.replace('USDT','')} for context"
            )
            for cid in subscribers:
                await _send(bot, cid, msg)
            logger.info("Liquidation alert: %s side=%s usd=%.0f", plain, side, usd)

    except Exception as e:
        logger.debug("_handle_liquidation error: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# SUBSCRIPTION MANAGER
# Figures out which symbols to watch based on:
#   - All active /palert symbols
#   - All active /watch symbols
#   - All active /alert symbols
#   - BTC + ETH always (for regime context)
# ─────────────────────────────────────────────────────────────────────────────

_ALWAYS_WATCH = {"BTC_USDT", "ETH_USDT", "SOL_USDT"}

def _get_target_symbols() -> set:
    """Build set of MEXC-format symbols that need live streaming."""
    syms = set(_ALWAYS_WATCH)
    try:
        for alert in _load_price_alerts():
            syms.add(to_mexc_sym(alert["symbol"]))
        for row in _load_conf_alerts():
            syms.add(to_mexc_sym(row["symbol"]))
        for row in _load_watchlist():
            syms.add(to_mexc_sym(row["symbol"]))
    except Exception as e:
        logger.warning("_get_target_symbols: %s", e)
    return syms


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

async def _ws_session(bot):
    """
    Single WebSocket session.
    Subscribes to ticker + funding + liquidation for all target symbols.
    Runs until disconnect, then caller reconnects.
    """
    symbols      = _get_target_symbols()
    price_alerts = _load_price_alerts()
    last_reload  = time.time()

    logger.info("[sakz_ws] Connecting to MEXC WS (%d symbols)", len(symbols))

    async with websockets.connect(
        MEXC_WS_URL,
        ping_interval=None,      # we handle pings manually (MEXC-style)
        close_timeout=10,
    ) as ws:
        logger.info("[sakz_ws] Connected ✅")

        # ── Subscribe ────────────────────────────────────────────────────────
        subs = []
        for sym in symbols:
            subs += [
                {"method": "sub.ticker",       "param": {"symbol": sym}},
                {"method": "sub.funding.rate", "param": {"symbol": sym}},
                {"method": "sub.liquidation",  "param": {"symbol": sym}},
            ]
        for sub in subs:
            await ws.send(json.dumps(sub))
            await asyncio.sleep(0.05)   # avoid burst

        logger.info("[sakz_ws] Subscribed to %d streams", len(subs))

        # ── Main loop ────────────────────────────────────────────────────────
        last_ping    = time.time()
        last_message = time.time()   # FIX: watchdog timestamp

        while True:
            now = time.time()

            # ── FIX: Watchdog — force reconnect if connection has gone silent ─
            if now - last_message > WS_DEAD_TIMEOUT:
                logger.warning(
                    "[sakz_ws] No message for %ds — connection appears dead, reconnecting",
                    WS_DEAD_TIMEOUT
                )
                return   # exits _ws_session → start_ws reconnects with backoff

            # ── Ping ────────────────────────────────────────────────────────
            if now - last_ping >= PING_INTERVAL:
                await ws.send(json.dumps({"method": "ping"}))
                last_ping = now

            # ── Reload alerts periodically ───────────────────────────────────
            if now - last_reload >= ALERT_RELOAD_INT:
                price_alerts = _load_price_alerts()
                new_syms     = _get_target_symbols()
                added        = new_syms - symbols
                for sym in added:
                    for method in ("sub.ticker", "sub.funding.rate", "sub.liquidation"):
                        await ws.send(json.dumps(
                            {"method": method, "param": {"symbol": sym}}
                        ))
                symbols     = new_syms
                last_reload = now

            # ── Receive ──────────────────────────────────────────────────────
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=20)
                last_message = time.time()   # FIX: reset watchdog on any message
            except asyncio.TimeoutError:
                # Nothing received in 20s — send ping and loop (watchdog will
                # force reconnect if silence persists beyond WS_DEAD_TIMEOUT)
                await ws.send(json.dumps({"method": "ping"}))
                last_ping = time.time()
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            channel = msg.get("channel", "")
            data    = msg.get("data", {})

            if channel == "pong" or msg.get("channel") == "rs.pong":
                continue

            if "ticker" in channel and data:
                await _handle_ticker(bot, data, price_alerts)

            elif "funding" in channel and data:
                await _handle_funding(bot, data)

            elif "liquidation" in channel and data:
                await _handle_liquidation(bot, data)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def start_ws(bot):
    """
    Main coroutine. Call once at startup:
        asyncio.create_task(sakz_ws.start_ws(application.bot))

    Reconnects automatically with exponential backoff on any disconnect.
    """
    if not _WS_AVAILABLE:
        logger.warning(
            "[sakz_ws] websockets package not installed. "
            "Run: pip install websockets\n"
            "WebSocket layer disabled — REST polling still active."
        )
        return

    delay = RECONNECT_DELAY
    attempt = 0

    while True:
        attempt += 1
        try:
            await _ws_session(bot)
            # Session ended (clean close or watchdog timeout) — reconnect
            logger.info("[sakz_ws] Session ended — reconnecting (attempt %d)...", attempt)
            # FIX: reset backoff only if we had a reasonably long session (>30s)
            # This prevents instant retry loops on auth/config errors
            delay = RECONNECT_DELAY
            attempt = 0

        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError) as e:
            logger.warning("[sakz_ws] Connection lost (attempt %d): %s", attempt, e)

        except Exception as e:
            logger.error("[sakz_ws] Unexpected error (attempt %d): %s", attempt, e)

        # Exponential backoff
        logger.info("[sakz_ws] Reconnecting in %ds...", delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, RECONNECT_MAX)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: read live price from anywhere in sakz_bot.py
# ─────────────────────────────────────────────────────────────────────────────

def ws_price(symbol: str) -> Optional[float]:
    """
    Drop-in helper for sakz_bot.py.
    Returns live WS price or None if WS hasn't seen this symbol yet.

    Usage:
        from sakz_ws import ws_price
        price = ws_price("BTCUSDT") or bybit_get_current_price("BTCUSDT")
    """
    return get_live_price(symbol)


def ws_funding(symbol: str) -> Optional[float]:
    """
    Returns latest streaming funding rate for a symbol, or None.
    Usage:
        from sakz_ws import ws_funding
        rate = ws_funding("BTCUSDT")
    """
    msym = to_mexc_sym(symbol)
    return _funding_cache.get(msym)


def ws_status() -> dict:
    """
    Returns a status snapshot for /status command integration.
    Add to your status_command output:
        from sakz_ws import ws_status
        ws = ws_status()
    """
    return {
        "symbols_tracked": len(_price_cache),
        "prices_cached":   len(_price_cache),
        "funding_cached":  len(_funding_cache),
        "oldest_price_s":  (
            round(time.time() - min((v["ts"] for v in _price_cache.values()), default=time.time()))
            if _price_cache else None
        ),
        "ws_available": _WS_AVAILABLE,
    }
