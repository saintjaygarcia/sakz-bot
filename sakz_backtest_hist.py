"""
sakz_backtest_hist.py — Historical Walk-Forward Backtest
=========================================================
Fetches extended OHLCV history and replays score_pair() across rolling
windows to measure how the scoring engine performs on historical data.

This is the REAL backtest go-trader has that sakz_bot was missing.
/backtest in sakz_bot only shows outcomes of signals that fired while
the bot was running.  This module tests the strategy on history.

INSTALL
───────
    No new dependencies — uses whatever OHLCV fetcher is available
    (sakz_ccxt if installed, raw exchange functions otherwise).

INTEGRATE into sakz_bot.py
───────────────────────────
1.  At the top of sakz_bot.py add:
        from sakz_backtest_hist import run_hist_backtest, format_hist_result

2.  Add command handler in the handler registration block:
        app.add_handler(CommandHandler("btfull", btfull_command))

3.  Add this async function anywhere in sakz_bot.py:

    async def btfull_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        \"\"\"
        /btfull SYMBOL [tf] [bars]
        Example: /btfull POWER 4h 500
        \"\"\"
        _track(update)
        args   = context.args or []
        if not args:
            await update.message.reply_text(
                "Usage: /btfull SYMBOL [tf] [bars]\\n"
                "Example: /btfull POWER 4h 500"
            )
            return

        raw    = args[0].upper()
        symbol = raw if raw.endswith('USDT') else raw + 'USDT'
        tf     = args[1].lower() if len(args) > 1 else '4h'
        bars   = int(args[2]) if len(args) > 2 and args[2].isdigit() else 500

        await update.message.reply_text(
            f"🔬 Running historical backtest\\n"
            f"📊 {symbol} · {tf.upper()} · {bars} bars\\n"
            f"⏳ This takes ~60s…"
        )
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            SCAN_EXECUTOR,
            lambda: run_hist_backtest(symbol, tf_key=tf, total_bars=bars)
        )
        await update.message.reply_text(
            format_hist_result(result),
            parse_mode='Markdown'
        )

DESIGN NOTES
────────────
Walk-forward means we never look ahead.  At each step the model only
sees data up to bar i — exactly what the live bot sees.  This avoids
lookahead bias that would inflate win-rate figures.

Fee model: 0.075% per side (taker).  Deducted on entry and on exit.
Targets: directly from score_pair() — same as live signals, so results
are comparable to /stats output.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
FEE_PCT        = 0.00075   # 0.075% per side (standard taker perp fee)
MIN_BARS       = 120       # minimum history bars needed before first test
WINDOW         = 120       # bars fed into score_pair each step
STEP           = 4         # bars to advance per step (≈ 1 candle)
MAX_HOLD_BARS  = 24        # force-close after N bars if no target hit

# Bybit interval string from unified tf key
_TF_TO_BYBIT = {
    '15m': '15', '30m': '30', '1h': '60',
    '2h': '120', '4h': '240', '6h': '360', '1d': 'D',
}

# Confirmation-timeframe for each primary TF (mirrors TF_CONFIGS)
_CONFIRM_TF = {
    '15m': '1h', '30m': '4h', '1h': '4h',
    '2h': '4h',  '4h': '1d', '6h': '1d', '1d': '1d',
}


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def run_hist_backtest(
    symbol:     str,
    tf_key:     str  = '4h',
    total_bars: int  = 500,
    min_conf:   int  = 5,
    exchange:   str  = 'MEXC',
    fee_pct:    float = FEE_PCT,
) -> dict:
    """
    Walk-forward backtest for symbol on tf_key.

    Algorithm
    ─────────
    1.  Fetch `total_bars` of primary OHLCV + confirmation TF OHLCV
    2.  Slide a window of WINDOW bars (step = STEP bars)
    3.  Run score_pair() on each window
    4.  If conf ≥ min_conf, simulate a trade using the signal's own
        entry / stop_loss / t1 / t2 / t3 levels
    5.  Walk forward bar by bar to find which level is hit first
        (or force-close after MAX_HOLD_BARS)
    6.  Deduct round-trip fees and accumulate equity curve

    Returns a dict suitable for format_hist_result().
    """
    # ── Lazy imports to avoid circular dependency ─────────────────────────────
    try:
        from sakz_bot import add_indicators, score_pair
    except ImportError as e:
        return _empty(symbol, exchange, tf_key, f"Cannot import sakz_bot: {e}")

    con_tf = _CONFIRM_TF.get(tf_key, '1d')

    # ── Fetch history ─────────────────────────────────────────────────────────
    pri_df = _fetch_hist(exchange, symbol, tf_key, total_bars + 20)
    con_df = _fetch_hist(exchange, symbol, con_tf,
                          max(total_bars // 4, 200))

    if pri_df is None or len(pri_df) < MIN_BARS + 20:
        got = len(pri_df) if pri_df is not None else 0
        return _empty(symbol, exchange, tf_key,
                      f"Insufficient history ({got} bars, need {MIN_BARS + 20})")
    if con_df is None or len(con_df) < 30:
        return _empty(symbol, exchange, tf_key, "No confirmation-TF data")

    logger.info("btfull %s %s: pri=%d bars  con=%d bars",
                symbol, tf_key, len(pri_df), len(con_df))

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    trades:     list  = []
    equity:     float = 1.0
    peak:       float = 1.0
    max_dd:     float = 0.0
    i:          int   = WINDOW          # start after enough warm-up bars
    skip_until: int   = 0              # bar to resume after a trade

    while i < len(pri_df) - MAX_HOLD_BARS - 1:

        if i < skip_until:
            i += STEP
            continue

        # Time-align confirmation slice (roughly)
        con_ratio = len(con_df) / len(pri_df)
        con_end   = max(30, int(i * con_ratio))
        pri_slice = pri_df.iloc[i - WINDOW : i].copy()
        con_slice = con_df.iloc[:con_end].copy()

        try:
            pri_ind = add_indicators(pri_slice, timeframe=tf_key)
            con_ind = add_indicators(con_slice, timeframe='1d')
        except Exception as e:
            logger.debug("btfull add_indicators at bar %d: %s", i, e)
            i += STEP
            continue

        if pri_ind is None or con_ind is None:
            i += STEP
            continue
        if len(pri_ind) < 5 or len(con_ind) < 5:
            i += STEP
            continue

        # Run scorer — pass user_requested=True so flip-cooldown
        # doesn't silently suppress signals during replay
        try:
            sig = score_pair(pri_ind, con_ind, 0.0, symbol,
                             user_requested=True)
        except TypeError:
            # Older sakz_bot without user_requested param
            sig = score_pair(pri_ind, con_ind, 0.0, symbol)
        except Exception as e:
            logger.debug("btfull score_pair bar %d: %s", i, e)
            i += STEP
            continue

        # ScanFailure or None → no signal this bar
        if sig is None or hasattr(sig, 'reason'):
            i += STEP
            continue

        conf = sig.get('confidence', 0)
        if conf < min_conf:
            i += STEP
            continue

        bias = sig.get('bias', '')
        sl   = sig.get('stop_loss', 0.0)
        t1   = sig.get('t1', 0.0)
        t2   = sig.get('t2', 0.0)
        t3   = sig.get('t3', 0.0)

        # Entry = close of the current bar (next-bar execution)
        entry = float(pri_df['close'].iloc[i])
        risk  = abs(entry - sl)

        if not all([bias, sl, t1]) or risk == 0:
            i += STEP
            continue

        # ── Simulate trade ────────────────────────────────────────────────────
        outcome, exit_px, exit_bar = _simulate(
            pri_df, i, entry, sl, t1, t2, t3, bias, MAX_HOLD_BARS
        )

        # Directional PnL before fees
        direction = 1.0 if bias == 'LONG' else -1.0
        gross_pct = (exit_px - entry) / entry * direction
        net_pct   = gross_pct - fee_pct * 2     # entry + exit fees

        # R:R (positive = profit, negative = loss)
        rr = net_pct / (risk / entry) if risk else 0.0

        equity *= (1.0 + net_pct)
        dd      = (peak - equity) / peak * 100.0
        if equity > peak:
            peak = equity
        if dd > max_dd:
            max_dd = dd

        trades.append({
            'bar':     i,
            'bias':    bias,
            'conf':    conf,
            'outcome': outcome,
            'entry':   round(entry, 6),
            'exit':    round(exit_px, 6),
            'rr':      round(rr, 3),
            'pnl_pct': round(net_pct * 100, 4),
        })

        # Skip bars consumed by the trade to avoid overlapping signals
        skip_until = exit_bar + STEP
        i = exit_bar + STEP

    return _aggregate(trades, equity, max_dd, symbol, exchange, tf_key,
                      len(pri_df), min_conf, fee_pct)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate(df: pd.DataFrame, entry_bar: int, entry: float,
              sl: float, t1: float, t2: float, t3: float,
              bias: str, max_bars: int) -> tuple:
    """
    Walk forward from entry_bar+1, checking each candle's high/low.
    Returns (outcome, exit_price, exit_bar).
    outcome: 'T1' | 'T2' | 'T3' | 'SL' | 'TIMEOUT'

    Priority per candle:  SL checked before targets on the same bar
    to avoid favourably assuming the target filled first in a bad candle.
    """
    end_bar = min(entry_bar + max_bars + 1, len(df))

    for j in range(entry_bar + 1, end_bar):
        high = float(df['high'].iloc[j])
        low  = float(df['low'].iloc[j])

        if bias == 'LONG':
            if low  <= sl: return ('SL',      sl, j)
            if high >= t3: return ('T3',      t3, j)
            if high >= t2: return ('T2',      t2, j)
            if high >= t1: return ('T1',      t1, j)
        else:
            if high >= sl: return ('SL',      sl, j)
            if low  <= t3: return ('T3',      t3, j)
            if low  <= t2: return ('T2',      t2, j)
            if low  <= t1: return ('T1',      t1, j)

    # Max bars elapsed — exit at last bar's close
    last_bar   = min(entry_bar + max_bars, len(df) - 1)
    exit_price = float(df['close'].iloc[last_bar])
    return ('TIMEOUT', exit_price, last_bar)


# ═══════════════════════════════════════════════════════════════════════════════
# OHLCV FETCHER  (ccxt preferred, falls back to raw sakz_bot functions)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_hist(exchange: str, symbol: str, tf: str,
                limit: int) -> Optional[pd.DataFrame]:
    """
    Try ccxt first (supports large limit values natively via pagination).
    Falls back to the raw exchange functions in sakz_bot if ccxt is absent.
    """
    # ── ccxt path ─────────────────────────────────────────────────────────────
    try:
        from sakz_ccxt import (bybit_fetch_ohlcv, binance_fetch_ohlcv,
                                mexc_fetch_ohlcv)
        if exchange == 'BYBIT':
            bybit_tf = _TF_TO_BYBIT.get(tf, '240')
            return bybit_fetch_ohlcv(symbol, bybit_tf, limit)
        elif exchange == 'BINANCE':
            return binance_fetch_ohlcv(symbol, tf, limit)
        else:
            return mexc_fetch_ohlcv(symbol, tf, limit)
    except ImportError:
        pass

    # ── Raw fallback ──────────────────────────────────────────────────────────
    try:
        from sakz_bot import (bybit_fetch_ohlcv, binance_fetch_ohlcv,
                               mexc_fetch_ohlcv)
        if exchange == 'BYBIT':
            bybit_tf = _TF_TO_BYBIT.get(tf, '240')
            return bybit_fetch_ohlcv(symbol, bybit_tf, limit)
        elif exchange == 'BINANCE':
            return binance_fetch_ohlcv(symbol, tf, limit)
        else:
            return mexc_fetch_ohlcv(symbol, tf, limit)
    except Exception as e:
        logger.warning("_fetch_hist %s %s %s: %s", exchange, symbol, tf, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate(trades: list, equity: float, max_dd: float,
               symbol: str, exchange: str, tf: str,
               total_bars: int, min_conf: int, fee_pct: float) -> dict:

    if not trades:
        return _empty(symbol, exchange, tf,
                      f"No trades fired (conf ≥ {min_conf}) in {total_bars} bars")

    n       = len(trades)
    pnls    = [t['pnl_pct'] for t in trades]
    wins    = [t for t in trades if t['outcome'] != 'SL']
    t1_hits = [t for t in trades if t['outcome'] in ('T1', 'T2', 'T3')]
    t2_hits = [t for t in trades if t['outcome'] in ('T2', 'T3')]
    t3_hits = [t for t in trades if t['outcome'] == 'T3']
    timeouts= [t for t in trades if t['outcome'] == 'TIMEOUT']

    win_rate  = len(wins) / n * 100
    total_pnl = (equity - 1.0) * 100

    avg_rr = float(np.mean([t['rr'] for t in trades]))

    # Sharpe-like (annualised, assuming each bar = 4h → 6 bars/day → 2190/year)
    bars_per_year = {'15m': 35040, '1h': 8760, '4h': 2190, '1d': 365}.get(tf, 2190)
    std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 0.0001
    sharpe  = (float(np.mean(pnls)) / std_pnl) * (bars_per_year ** 0.5) / STEP

    # Profit factor
    gross_wins   = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    pf = gross_wins / gross_losses if gross_losses else float('inf')

    # Expectancy per trade (% terms)
    expectancy = float(np.mean(pnls))

    return {
        'symbol':     symbol,
        'exchange':   exchange,
        'tf':         tf,
        'total_bars': total_bars,
        'min_conf':   min_conf,
        'fee_pct':    fee_pct,
        'trades':     n,
        'wins':       len(wins),
        'losses':     n - len(wins),
        'timeouts':   len(timeouts),
        'win_rate':   round(win_rate, 1),
        't1_rate':    round(len(t1_hits) / n * 100, 1),
        't2_rate':    round(len(t2_hits) / n * 100, 1),
        't3_rate':    round(len(t3_hits) / n * 100, 1),
        'avg_rr':     round(avg_rr, 3),
        'expectancy': round(expectancy, 4),
        'pf':         round(pf, 2) if pf != float('inf') else 'inf',
        'total_pnl':  round(total_pnl, 2),
        'max_dd':     round(max_dd, 2),
        'sharpe':     round(sharpe, 2),
        'best_trade': round(max(pnls), 2),
        'worst_trade':round(min(pnls), 2),
        'all_trades': trades,
        'error':      None,
    }


def _empty(symbol, exchange, tf, error) -> dict:
    return {
        'symbol': symbol, 'exchange': exchange, 'tf': tf,
        'trades': 0, 'wins': 0, 'losses': 0, 'timeouts': 0,
        'win_rate': 0, 't1_rate': 0, 't2_rate': 0, 't3_rate': 0,
        'avg_rr': 0, 'expectancy': 0, 'pf': 0,
        'total_pnl': 0, 'max_dd': 0, 'sharpe': 0,
        'best_trade': 0, 'worst_trade': 0,
        'total_bars': 0, 'min_conf': 0, 'fee_pct': 0,
        'all_trades': [], 'error': error,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def format_hist_result(r: dict) -> str:
    if r.get('error'):
        return (
            f"🔬 *Historical Backtest — {r['symbol']}*\n"
            f"❌ {r['error']}"
        )

    n         = r['trades']
    win_rate  = r['win_rate']
    total_pnl = r['total_pnl']

    # Traffic-light emojis
    wr_e  = '🟢' if win_rate >= 55 else ('🟡' if win_rate >= 45 else '🔴')
    pnl_e = '📈' if total_pnl >= 0 else '📉'
    pf_e  = '✅' if (r['pf'] != 'inf' and r['pf'] >= 1.3) or r['pf'] == 'inf' else '⚠️'
    sh_e  = '✅' if r['sharpe'] >= 1.0 else ('🟡' if r['sharpe'] >= 0.5 else '🔴')

    # Long/short split
    longs  = [t for t in r['all_trades'] if t['bias'] == 'LONG']
    shorts = [t for t in r['all_trades'] if t['bias'] == 'SHORT']
    l_wr   = round(len([t for t in longs if t['outcome'] != 'SL']) / len(longs) * 100, 0) if longs else 0
    s_wr   = round(len([t for t in shorts if t['outcome'] != 'SL']) / len(shorts) * 100, 0) if shorts else 0

    lines = [
        f"🔬 *Historical Backtest — {r['symbol']}*",
        f"📊 {r['tf'].upper()} · {r['total_bars']} bars · conf ≥ {r['min_conf']} · fee {r['fee_pct']*100:.3f}% per side",
        f"",
        f"*Overview*",
        f"  Trades:      {n}  ({r['wins']}W / {r['losses']}L / {r['timeouts']} timeout)",
        f"  {wr_e} Win rate:   {win_rate}%",
        f"  📐 Avg R:R:   {r['avg_rr']:+.3f}R",
        f"  💰 Expectancy:{r['expectancy']:+.4f}% / trade",
        f"",
        f"*Target Hit Rates*",
        f"  🎯 T1:  {r['t1_rate']}%",
        f"  🎯 T2:  {r['t2_rate']}%",
        f"  🎯 T3:  {r['t3_rate']}%",
        f"",
        f"*Risk Metrics*",
        f"  {pnl_e} Cumulative PnL:  {total_pnl:+.2f}%",
        f"  📉 Max drawdown:   {r['max_dd']:.2f}%",
        f"  {pf_e} Profit factor:   {r['pf']}",
        f"  {sh_e} Sharpe-like:     {r['sharpe']:.2f}",
        f"  🏆 Best trade:   {r['best_trade']:+.2f}%",
        f"  💀 Worst trade:  {r['worst_trade']:+.2f}%",
        f"",
        f"*Direction Split*",
        f"  LONG  {len(longs)} trades → {l_wr:.0f}% win",
        f"  SHORT {len(shorts)} trades → {s_wr:.0f}% win",
        f"",
        f"⚠️ Fees deducted: {r['fee_pct']*200:.2f}% round-trip · No slippage model",
        f"📌 This replays score\\_pair() on historical bars — comparable to /stats",
    ]
    return "\n".join(lines)
