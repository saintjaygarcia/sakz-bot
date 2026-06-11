"""
sakz_paper.py — Auto Paper Trading Layer
==========================================
Automatically opens, tracks and closes paper positions whenever a
high-confidence signal fires.  Unlike /pick (which requires the user
to manually activate a signal), paper mode runs silently in the
background and gives you unbiased, automated performance data.

INSTALL
───────
    No new dependencies — uses the existing SQLite db_connect() and the
    current-price fetchers already in sakz_bot.

INTEGRATE into sakz_bot.py
───────────────────────────

STEP 1 — Import at the top of sakz_bot.py:
    from sakz_paper import (
        paper_init_db, paper_maybe_open, paper_mark_all,
        paper_close_expired, paper_get_open, paper_summary,
        format_paper_open, format_paper_summary,
        PAPER_CONF_MIN, PAPER_MAX_OPEN,
    )

STEP 2 — After db_connect() is first called (e.g. in main() or startup):
    paper_init_db(db_connect)

STEP 3 — After every signal fires in scan / cscan / run_full_scan, add:
    opened = paper_maybe_open(result, db_connect)
    if opened:
        logger.info("Paper position auto-opened: %s %s", result['bias'], result['symbol'])

STEP 4 — Add a background job (in the job_queue registration block):
    app.job_queue.run_repeating(paper_job, interval=900, first=120,
                                name='paper_mtm')

    Then add this function:

    async def paper_job(context: ContextTypes.DEFAULT_TYPE):
        \"\"\"Background job: mark paper positions to market every 15 min.\"\"\"
        def _price(exchange, symbol):
            try:
                if exchange == 'BYBIT':   return bybit_get_current_price(symbol)
                if exchange == 'BINANCE': return binance_get_current_price(symbol)
                return mexc_get_current_price(symbol)
            except Exception:
                return None

        closed = paper_mark_all(db_connect, _price)
        expired = paper_close_expired(db_connect, _price)
        for pos in closed + expired:
            logger.info("Paper closed: %s %s → %s  PnL=%.3f%%",
                        pos['bias'], pos['symbol'], pos['outcome'],
                        pos.get('pnl_pct', 0))

STEP 5 — Add commands (in handler registration block):
    app.add_handler(CommandHandler("paper", paper_command))

    Then add this function:

    async def paper_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        \"\"\"
        /paper          — show open positions + 7-day summary
        /paper history  — last 20 closed positions
        \"\"\"
        _track(update)
        args = context.args or []

        if args and args[0].lower() == 'history':
            rows = paper_get_closed(db_connect, limit=20)
            if not rows:
                await update.message.reply_text("📋 No closed paper positions yet.")
                return
            lines = ["📋 *Paper Trading — Last 20 Closed*", ""]
            for r in rows:
                oc_e = '✅' if r['outcome'] not in ('SL','EXPIRED') else '❌'
                lines.append(
                    f"{oc_e} {r['symbol']} {r['bias']} → {r['outcome']}  "
                    f"{(r['pnl_pct'] or 0):+.2f}%  (conf {r['confidence']})"
                )
            await update.message.reply_text(
                "\\n".join(lines), parse_mode='Markdown'
            )
            return

        open_msg = format_paper_open(paper_get_open(db_connect))
        summ_msg = format_paper_summary(paper_summary(db_connect))
        await update.message.reply_text(
            open_msg + "\\n\\n" + summ_msg, parse_mode='Markdown'
        )
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PAPER_CONF_MIN    = 8      # baseline minimum confidence to auto-open a position
PAPER_MAX_OPEN    = 12     # maximum concurrent open positions
PAPER_FEE_PCT     = 0.00075  # 0.075% per side (taker)
PAPER_MAX_HOLD_H  = 72     # force-expire after this many hours

# ── Adaptive confidence floor (Flaw 5) ─────────────────────────────────────────
# PAPER_CONF_MIN above is only the *baseline*. The floor actually used to open
# positions is recalibrated from realised paper performance: the summary already
# tracks conf>=9 win-rate (hi_conf_wr) vs conf-8 win-rate (lo_conf_wr). When the
# conf-8 bucket clearly underperforms (a bear-regime tell), raise the floor so we
# stop opening weak conf-8 trades; when conf-8 is performing strongly, allow a
# slightly lower floor to capture more edge. Always bounded + sample-gated.
PAPER_CONF_FLOOR_MIN   = 7      # never require less than this
PAPER_CONF_FLOOR_MAX   = 10     # never require more than this
PAPER_ADAPT_LOOKBACK_H = 336    # 14 days of closed trades for a stable read
PAPER_ADAPT_MIN_SAMPLE = 10     # need this many trades in a bucket to trust it
PAPER_ADAPT_WR_WEAK    = 45.0   # win-rate below this = weak bucket
PAPER_ADAPT_WR_STRONG  = 58.0   # win-rate above this = strong bucket
PAPER_ADAPT_DELTA      = 12.0   # hi-vs-lo win-rate gap that justifies raising
_ADAPTIVE_TTL          = timedelta(minutes=30)  # recompute the floor at most this often

_adaptive_conf_min     = PAPER_CONF_MIN   # cached effective floor
_adaptive_conf_min_at  = None             # datetime of last recompute (UTC)

# ═══════════════════════════════════════════════════════════════════════════════
# DB SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_positions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    exchange     TEXT    NOT NULL,
    bias         TEXT    NOT NULL,
    confidence   INTEGER NOT NULL,
    entry_price  REAL    NOT NULL,
    stop_loss    REAL    NOT NULL,
    t1           REAL    NOT NULL,
    t2           REAL    NOT NULL,
    t3           REAL    NOT NULL,
    opened_at    TEXT    NOT NULL,
    closed_at    TEXT,
    exit_price   REAL,
    outcome      TEXT    DEFAULT 'OPEN',
    pnl_pct      REAL,
    rr           REAL,
    signal_tf    TEXT,
    signal_json  TEXT
);
"""


def paper_init_db(db_connect: Callable):
    """
    Create the paper_positions table if it doesn't exist.
    Call once at startup after db_connect() is available.
    """
    try:
        conn = db_connect()
        conn.execute(_SCHEMA)
        conn.commit()
        conn.close()
        logger.info("paper_positions table ready")
    except Exception as e:
        logger.error("paper_init_db: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# OPEN POSITION
# ═══════════════════════════════════════════════════════════════════════════════

def paper_adaptive_conf_min(db_connect: Callable) -> int:
    """
    Derive the effective minimum confidence from realised paper performance.

    Uses the hi_conf_wr (conf >= 9) vs lo_conf_wr (conf 8) delta that
    paper_summary already computes. Result is bounded to
    [PAPER_CONF_FLOOR_MIN, PAPER_CONF_FLOOR_MAX] and falls back to the static
    PAPER_CONF_MIN whenever there isn't enough data to judge.
    """
    try:
        s = paper_summary(db_connect, lookback_hours=PAPER_ADAPT_LOOKBACK_H)
    except Exception as e:
        logger.warning("paper_adaptive_conf_min: summary failed: %s", e)
        return PAPER_CONF_MIN

    if s.get('error') or s.get('trades', 0) == 0:
        return PAPER_CONF_MIN

    lo_n  = s.get('lo_conf_n', 0)
    hi_n  = s.get('hi_conf_n', 0)
    lo_wr = s.get('lo_conf_wr', 0.0)
    hi_wr = s.get('hi_conf_wr', 0.0)

    # Not enough conf-8 trades to judge - keep the baseline.
    if lo_n < PAPER_ADAPT_MIN_SAMPLE:
        return PAPER_CONF_MIN

    floor = PAPER_CONF_MIN  # 8

    # Conf-8 bucket is weak AND clearly worse than conf>=9 -> stop opening conf-8.
    if (lo_wr < PAPER_ADAPT_WR_WEAK
            and (hi_wr - lo_wr) >= PAPER_ADAPT_DELTA
            and hi_n >= PAPER_ADAPT_MIN_SAMPLE):
        floor = 9
        # Even the conf>=9 bucket is poor -> demand only the strongest signals.
        if hi_wr < PAPER_ADAPT_WR_WEAK:
            floor = 10
    # Conf-8 bucket is performing strongly -> capture a little more edge.
    elif lo_wr >= PAPER_ADAPT_WR_STRONG:
        floor = 7

    return max(PAPER_CONF_FLOOR_MIN, min(PAPER_CONF_FLOOR_MAX, floor))


def _effective_conf_min(db_connect: Callable) -> int:
    """Cached wrapper around paper_adaptive_conf_min (recomputes every 30 min)."""
    global _adaptive_conf_min, _adaptive_conf_min_at
    now = datetime.utcnow()
    if _adaptive_conf_min_at is not None and (now - _adaptive_conf_min_at) < _ADAPTIVE_TTL:
        return _adaptive_conf_min
    try:
        new_floor = paper_adaptive_conf_min(db_connect)
    except Exception as e:
        logger.warning("_effective_conf_min: %s", e)
        new_floor = PAPER_CONF_MIN
    if new_floor != _adaptive_conf_min:
        logger.info("\U0001F4C8 Paper adaptive confidence floor: %d \u2192 %d",
                    _adaptive_conf_min, new_floor)
    _adaptive_conf_min    = new_floor
    _adaptive_conf_min_at = now
    return _adaptive_conf_min


def paper_maybe_open(signal: dict, db_connect: Callable) -> bool:
    """
    Conditionally open a paper position from a scan result dict.

    Guards:
     ─ confidence < PAPER_CONF_MIN  → skip
     ─ symbol already open          → skip (no stacking same pair)
     ─ open count >= PAPER_MAX_OPEN → skip
     ─ missing key price levels     → skip

    Returns True if a position was opened.
    """
    conf = int(signal.get('confidence', 0))
    conf_min = _effective_conf_min(db_connect)
    if conf < conf_min:
        return False

    symbol   = signal.get('symbol', '').upper()
    exchange = signal.get('exchange', 'BYBIT').upper()
    bias     = signal.get('bias', '')

    # Prefer explicit entry_price; fall back to 'price'
    entry = float(signal.get('entry_price') or signal.get('price') or 0)
    sl    = float(signal.get('stop_loss') or 0)
    t1    = float(signal.get('t1') or 0)
    t2    = float(signal.get('t2') or 0)
    t3    = float(signal.get('t3') or 0)
    tf    = signal.get('signal_tf', signal.get('timeframe', '4h'))

    if not all([symbol, bias, entry, sl, t1]):
        logger.debug("paper_maybe_open: incomplete signal for %s — skipping", symbol)
        return False

    try:
        conn = db_connect()
        c = conn.cursor()

        # Already open for this symbol?
        c.execute("SELECT COUNT(*) FROM paper_positions WHERE symbol=? AND outcome='OPEN'",
                  (symbol,))
        if c.fetchone()[0] > 0:
            conn.close()
            return False

        # At cap?
        c.execute("SELECT COUNT(*) FROM paper_positions WHERE outcome='OPEN'")
        if c.fetchone()[0] >= PAPER_MAX_OPEN:
            conn.close()
            return False

        c.execute(
            """INSERT INTO paper_positions
               (symbol, exchange, bias, confidence, entry_price, stop_loss,
                t1, t2, t3, opened_at, outcome, signal_tf, signal_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)""",
            (symbol, exchange, bias, conf, entry, sl, t1, t2, t3,
             datetime.utcnow().isoformat(), tf, json.dumps(signal, default=str))
        )
        conn.commit()
        conn.close()
        logger.info("📋 Paper opened: %s %s conf=%d entry=%.6f SL=%.6f T1=%.6f",
                    bias, symbol, conf, entry, sl, t1)
        return True

    except Exception as e:
        logger.warning("paper_maybe_open %s: %s", symbol, e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MARK-TO-MARKET  (run every 15 min from a background job)
# ═══════════════════════════════════════════════════════════════════════════════

def paper_mark_all(db_connect: Callable,
                   price_fn: Callable[[str, str], Optional[float]]) -> List[dict]:
    """
    Check all open positions against current market prices.
    Closes any where SL or a target has been hit.

    price_fn(exchange: str, symbol: str) → float | None

    Returns list of positions that were closed this call.
    """
    closed = []
    try:
        conn = db_connect()
        c    = conn.cursor()
        c.execute("SELECT * FROM paper_positions WHERE outcome='OPEN'")
        rows = c.fetchall()
        cols = [d[0] for d in c.description]

        for row in rows:
            pos   = dict(zip(cols, row))
            price = price_fn(pos['exchange'], pos['symbol'])
            if price is None:
                continue

            outcome = _check_targets(price, pos['bias'],
                                      pos['stop_loss'], pos['t1'],
                                      pos['t2'], pos['t3'])
            if outcome:
                _write_close(conn, pos['id'], price, outcome,
                             pos['entry_price'], pos['bias'], pos['stop_loss'])
                closed.append({**pos, 'exit_price': price, 'outcome': outcome})

        conn.commit()
        conn.close()

    except Exception as e:
        logger.warning("paper_mark_all: %s", e)

    return closed


def paper_close_expired(db_connect: Callable,
                        price_fn: Callable[[str, str], Optional[float]]) -> List[dict]:
    """Force-close positions open longer than PAPER_MAX_HOLD_H hours."""
    expired  = []
    cutoff   = (datetime.utcnow() - timedelta(hours=PAPER_MAX_HOLD_H)).isoformat()
    try:
        conn = db_connect()
        c    = conn.cursor()
        c.execute(
            "SELECT * FROM paper_positions WHERE outcome='OPEN' AND opened_at < ?",
            (cutoff,)
        )
        rows = c.fetchall()
        cols = [d[0] for d in c.description]

        for row in rows:
            pos   = dict(zip(cols, row))
            price = price_fn(pos['exchange'], pos['symbol']) or pos['entry_price']
            _write_close(conn, pos['id'], price, 'EXPIRED',
                         pos['entry_price'], pos['bias'], pos['stop_loss'])
            expired.append({**pos, 'exit_price': price, 'outcome': 'EXPIRED'})

        conn.commit()
        conn.close()

    except Exception as e:
        logger.warning("paper_close_expired: %s", e)

    return expired


def _check_targets(price: float, bias: str,
                   sl: float, t1: float, t2: float, t3: float) -> Optional[str]:
    """Return the outcome string if a level has been breached, else None."""
    if bias == 'LONG':
        if price <= sl:  return 'SL'
        if t3 and price >= t3: return 'T3'
        if t2 and price >= t2: return 'T2'
        if price >= t1:  return 'T1'
    else:
        if price >= sl:  return 'SL'
        if t3 and price <= t3: return 'T3'
        if t2 and price <= t2: return 'T2'
        if price <= t1:  return 'T1'
    return None


def _write_close(conn, pos_id: int, exit_price: float, outcome: str,
                 entry: float, bias: str, sl: float = 0.0):
    """Calculate PnL + R:R and write the close record.

    R:R is realised-reward / planned-risk, where planned-risk is the distance
    from entry to the stop-loss (NOT the exit distance). Using the exit
    distance made every trade look like ~1R and corrupted the paper stats."""
    direction = 1.0 if bias == 'LONG' else -1.0
    gross_pct = (exit_price - entry) / entry * direction * 100
    net_pct   = gross_pct - PAPER_FEE_PCT * 2 * 100   # round-trip

    # Planned risk = entry → stop-loss distance. Fall back to exit distance
    # only if SL is missing, to avoid divide-by-zero.
    if sl and entry:
        risk_pct = abs(entry - sl) / entry * 100
    else:
        risk_pct = abs(exit_price - entry) / entry * 100
    rr        = net_pct / risk_pct if risk_pct else 0.0

    conn.execute(
        """UPDATE paper_positions
           SET closed_at=?, exit_price=?, outcome=?, pnl_pct=?, rr=?
           WHERE id=?""",
        (datetime.utcnow().isoformat(), exit_price, outcome,
         round(net_pct, 4), round(rr, 3), pos_id)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QUERIES
# ═══════════════════════════════════════════════════════════════════════════════

def paper_get_open(db_connect: Callable) -> List[dict]:
    """Return all currently open paper positions."""
    try:
        conn = db_connect()
        c    = conn.cursor()
        c.execute("SELECT * FROM paper_positions WHERE outcome='OPEN' ORDER BY opened_at DESC")
        cols = [d[0] for d in c.description]
        rows = [dict(zip(cols, r)) for r in c.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.warning("paper_get_open: %s", e)
        return []


def paper_get_closed(db_connect: Callable, limit: int = 20) -> List[dict]:
    """Return the most recent closed paper positions."""
    try:
        conn = db_connect()
        c    = conn.cursor()
        c.execute(
            "SELECT * FROM paper_positions WHERE outcome!='OPEN' ORDER BY closed_at DESC LIMIT ?",
            (limit,)
        )
        cols = [d[0] for d in c.description]
        rows = [dict(zip(cols, r)) for r in c.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.warning("paper_get_closed: %s", e)
        return []


def paper_summary(db_connect: Callable, lookback_hours: int = 168) -> dict:
    """
    Performance stats for all closed positions in the last lookback_hours.
    Default: 7 days (168h).
    """
    since = (datetime.utcnow() - timedelta(hours=lookback_hours)).isoformat()
    try:
        conn = db_connect()
        c    = conn.cursor()
        c.execute(
            """SELECT * FROM paper_positions
               WHERE outcome != 'OPEN' AND closed_at >= ?
               ORDER BY closed_at DESC""",
            (since,)
        )
        cols = [d[0] for d in c.description]
        rows = [dict(zip(cols, r)) for r in c.fetchall()]
        conn.close()

        if not rows:
            return {'trades': 0, 'lookback_h': lookback_hours,
                    'error': 'No closed positions in window'}

        n      = len(rows)
        pnls   = [r['pnl_pct'] for r in rows if r['pnl_pct'] is not None]
        wins   = [r for r in rows if r['outcome'] not in ('SL', 'EXPIRED')]
        losses = [r for r in rows if r['outcome'] == 'SL']
        exps   = [r for r in rows if r['outcome'] == 'EXPIRED']

        total_pnl  = sum(pnls)
        gross_w    = sum(p for p in pnls if p > 0)
        gross_l    = abs(sum(p for p in pnls if p < 0))
        pf         = round(gross_w / gross_l, 2) if gross_l else float('inf')

        # Conf breakdown
        hi_conf = [r for r in rows if r['confidence'] >= 9]
        lo_conf = [r for r in rows if r['confidence'] < 9]
        hi_wr   = (len([r for r in hi_conf if r['outcome'] not in ('SL','EXPIRED')])
                   / len(hi_conf) * 100) if hi_conf else 0
        lo_wr   = (len([r for r in lo_conf if r['outcome'] not in ('SL','EXPIRED')])
                   / len(lo_conf) * 100) if lo_conf else 0

        return {
            'trades':     n,
            'wins':       len(wins),
            'losses':     len(losses),
            'expired':    len(exps),
            'win_rate':   round(len(wins) / n * 100, 1),
            'total_pnl':  round(total_pnl, 2),
            'avg_pnl':    round(total_pnl / n, 3),
            'best':       round(max(pnls), 2) if pnls else 0,
            'worst':      round(min(pnls), 2) if pnls else 0,
            'pf':         pf if pf != float('inf') else 'inf',
            'hi_conf_wr': round(hi_wr, 1),
            'lo_conf_wr': round(lo_wr, 1),
            'hi_conf_n':  len(hi_conf),
            'lo_conf_n':  len(lo_conf),
            'lookback_h': lookback_hours,
            'recent':     rows[:5],
            'error':      None,
        }
    except Exception as e:
        logger.warning("paper_summary: %s", e)
        return {'trades': 0, 'lookback_h': lookback_hours, 'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════

def format_paper_open(positions: List[dict]) -> str:
    """Format the list of currently open paper positions for Telegram."""
    if not positions:
        return "📋 *Paper Positions*\nNo open positions."

    lines = [f"📋 *Paper Positions ({len(positions)} open)*", ""]
    for p in positions:
        opened  = p.get('opened_at', '')[:16].replace('T', ' ')
        bias_e  = '🟢' if p['bias'] == 'LONG' else '🔴'
        lines.append(
            f"{bias_e} *{p['symbol']}* {p['bias']}  conf {p['confidence']}/10  [{p['exchange']}]\n"
            f"  Entry: `{p['entry_price']:.6g}`  SL: `{p['stop_loss']:.6g}`\n"
            f"  T1: `{p['t1']:.6g}`  T2: `{p['t2']:.6g}`  T3: `{p['t3']:.6g}`\n"
            f"  Opened: {opened} UTC"
        )
    return "\n".join(lines)


def format_paper_summary(s: dict) -> str:
    """Format the 7-day performance summary for Telegram."""
    if s.get('error') or s.get('trades', 0) == 0:
        return (
            f"📊 *Paper Summary — {s.get('lookback_h', 168)}h*\n"
            f"No data: {s.get('error', 'no closed positions')}"
        )

    wr_e  = '🟢' if s['win_rate'] >= 55 else ('🟡' if s['win_rate'] >= 45 else '🔴')
    pnl_e = '📈' if s['total_pnl'] >= 0 else '📉'
    pf_str = str(s['pf']) if s['pf'] != float('inf') else '∞'

    lines = [
        f"📊 *Paper Summary — Last {s['lookback_h']}h*",
        f"",
        f"  Trades:   {s['trades']}  ({s['wins']}W / {s['losses']}L / {s['expired']} expired)",
        f"  {wr_e} Win rate:  {s['win_rate']}%",
        f"  {pnl_e} Total PnL: {s['total_pnl']:+.2f}%",
        f"  📐 Avg / trade: {s['avg_pnl']:+.3f}%",
        f"  ⚖️  Profit factor: {pf_str}",
        f"  🏆 Best:  {s['best']:+.2f}%   💀 Worst: {s['worst']:+.2f}%",
        f"",
        f"  *Conf breakdown:*",
        f"  Conf ≥9: {s['hi_conf_n']} trades → {s['hi_conf_wr']:.0f}% win",
        f"  Conf  8: {s['lo_conf_n']} trades → {s['lo_conf_wr']:.0f}% win",
    ]

    if s.get('recent'):
        lines += ["", "  *Recent closes:*"]
        for r in s['recent']:
            oc_e = '✅' if r['outcome'] not in ('SL', 'EXPIRED') else '❌'
            lines.append(
                f"  {oc_e} {r['symbol']} {r['bias']} → {r['outcome']}  "
                f"{(r['pnl_pct'] or 0):+.2f}%"
            )

    return "\n".join(lines)
