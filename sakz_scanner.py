"""sakz_scanner.py - pure signal engine (LEAF module).

Extracted from sakz_bot.py (Stage 2 of the monolith refactor): technical
indicators, pair scoring, BTC regime gate, and the R:R gate.

Dependency rule: this module imports ONLY sakz_state, third-party libraries,
and the already-clean satellite modules (sakz_exchanges / sakz_db / sakz_conviction
/ xgboost_train / rf_train). It must NEVER import sakz_bot, so the call graph
stays acyclic:  sakz_bot -> sakz_scanner -> (state / satellites).
"""
import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import ta

import sakz_state as state
import sakz_exchanges
from sakz_exchanges import (
    bybit_fetch_ohlcv,
    mexc_fetch_ohlcv,
    binance_fetch_ohlcv,
)
from sakz_db import db_save_signal_bias

logger = logging.getLogger(__name__)

# ---- optional conviction / ML layers (same guards as the monolith) ----
try:
    from sakz_conviction import conviction_scores, CONVICTION_DISPLAY
    _CONVICTION_AVAILABLE = True
except ImportError:
    _CONVICTION_AVAILABLE = False

try:
    from xgboost_train import predict_signal as xgb_predict_signal, train as xgb_train, model_meta as xgb_model_meta
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost_train not found - XGBoost ML scoring disabled")

try:
    from rf_train import (
        predict_signal as rf_predict_signal,
        train          as rf_train_model,
        model_meta     as rf_model_meta,
        consensus_verdict,
    )
    _RF_AVAILABLE = True
except ImportError:
    _RF_AVAILABLE = False
    logger.warning("rf_train not found - RF ML scoring disabled")

# ---- engine constants ----
_MIN_SIGNAL_RR = max(0.1, float(os.getenv("MIN_SIGNAL_RR", "1.5")))

# ---- ML influence thresholds (win-probability → signal shaping) ----
# Tunable via env. Untrained models return 0.5, so these defaults keep ML inert
# until it has learned from real outcomes. STRONG/WEAK drive the ±1 confidence
# nudge; HIDE_BELOW removes only the very-weakest signals from curated /scan &
# /autoscan (NEVER from user-requested /analyse, /cscan, /chart).
_ML_STRONG_EDGE = float(os.getenv("ML_STRONG_EDGE", "0.68"))
_ML_WEAK_EDGE   = float(os.getenv("ML_WEAK_EDGE",   "0.40"))
_ML_HIDE_BELOW  = float(os.getenv("ML_HIDE_BELOW",  "0.35"))

# ---- signal SURFACING thresholds (display only — NOT the paper execution floor) ----
SCAN_DISPLAY_CONF_MIN = max(0.0, float(os.getenv("SCAN_DISPLAY_CONF_MIN", "7")))
# Autoscan pushes are unsolicited; allow a stricter floor (defaults to same).
AUTOSCAN_DISPLAY_CONF_MIN = max(
    SCAN_DISPLAY_CONF_MIN,
    float(os.getenv("AUTOSCAN_DISPLAY_CONF_MIN", str(SCAN_DISPLAY_CONF_MIN))),
)

_SCAN_SHOW_ALL_FLAGS = {"all", "low", "everything", "full"}
_btc_regime_cache_ttl = 900
_BTCD_TTL = 1800
_BTC_PRICE_TTL = 60
_FIB_RATIOS = (0.236, 0.382, 0.500, 0.618, 0.786)

# ---- scan-failure reason tags used by the engine (value-compared strings) ----
REASON_GAP_BLOCK      = "GAP_BLOCK"
REASON_FLIP_BLOCK     = "FLIP_BLOCK"
REASON_COUNTER_TREND  = "COUNTER_TREND"


@dataclass
class ScanFailure:
    """Carries the reason a symbol produced no signal."""
    reason: str
    exchange: str = ""
    tf: str       = ""
    detail: str   = ""


# ============================================================================
# Extracted engine functions (verbatim from sakz_bot.py, source order)
# ============================================================================


def _get_btc_dominance() -> dict:
    """
    Fetch BTC dominance trend from available sources.
    Returns {'btcd': float_pct, 'trend': 'rising'|'falling'|'flat'}.
    Falls back to {'btcd': 0, 'trend': 'flat'} if unavailable.
    Uses the last 10 daily closes of BTC.D to determine trend direction.
    """

    now = datetime.now()
    if state._btcd_cache and (now - state._btcd_cache['time']).total_seconds() < _BTCD_TTL:
        return state._btcd_cache

    result = {'btcd': 0.0, 'trend': 'flat', 'time': now}
    try:
        # Binance BTCDOMUSDT is a dominance index (not a futures pair — spot only)
        # We approximate BTC dominance from CoinGecko global API if available,
        # otherwise derive a simple proxy: BTC market cap vs ETH+BTC
        # Practical approach: try binance spot 1d BTCDOMUSDT if present
        for fetch_fn, sym, interval in [
            (bybit_fetch_ohlcv,   'BTCDOMUSDT', 'D'),
            (binance_fetch_ohlcv, 'BTCDOMUSDT', '1d'),
        ]:
            try:
                df_d = fetch_fn(sym, interval, 15)
                if df_d is not None and len(df_d) >= 5:
                    closes = df_d['close'].dropna().values
                    current = float(closes[-1])
                    # Trend: compare 3-period SMA now vs 3-period SMA 5 bars ago
                    sma_now  = float(closes[-3:].mean())
                    sma_prev = float(closes[-8:-5].mean()) if len(closes) >= 8 else sma_now
                    delta    = sma_now - sma_prev
                    trend    = 'rising' if delta > 0.2 else 'falling' if delta < -0.2 else 'flat'
                    result   = {'btcd': current, 'trend': trend, 'time': now}
                    logger.debug("BTC.D: %.2f%% trend=%s (delta=%.3f)", current, trend, delta)
                    break
            except Exception:
                continue
    except Exception as e:
        logger.debug("BTC.D fetch error: %s", e)

    state._btcd_cache = result
    return result


def calculate_leverage(price, entry_low, entry_high, stop_loss, atr, confidence, bias):
    try:
        entry_mid = (entry_low + entry_high) / 2
        sl_dist   = abs(entry_mid - stop_loss) / entry_mid * 100
        if sl_dist == 0:
            return None
        raw_max = 100 / (sl_dist * 1.4)
        atr_pct = (atr / price) * 100
        # Volatility factor + hard ceiling per regime — keeps suggestions realistic
        # High-ATR coins liquidate fast; capping leverage prevents reckless numbers.
        if atr_pct < 1.0:
            vf = 1.0;  vl = "Very Low";  regime_cap = 50
        elif atr_pct < 2.0:
            vf = 0.85; vl = "Low";       regime_cap = 30
        elif atr_pct < 3.5:
            vf = 0.70; vl = "Medium";    regime_cap = 20
        elif atr_pct < 5.5:
            vf = 0.55; vl = "High";      regime_cap = 10
        elif atr_pct < 7.0:
            vf = 0.40; vl = "Very High"; regime_cap = 5
        else:
            vf = 0.25; vl = "Extreme";   regime_cap = 3
        cf        = {10: 1.0, 9: 0.85, 8: 0.70}.get(confidence, 0.50)
        suggested = min(raw_max * vf * cf, regime_cap)
        max_safe  = min(raw_max * vf,      regime_cap)
        def clean(x):
            if x >= 50:   return int(x // 10) * 10
            elif x >= 20: return int(x // 5) * 5
            elif x >= 10: return int(x // 2) * 2
            else:         return max(1, int(x))
        suggested = clean(suggested)
        max_safe  = clean(max_safe)
        liq_dist  = (1 / suggested) * 100 * 0.9 if suggested > 0 else 100
        fluct     = max(0, liq_dist - sl_dist)
        return {'suggested': suggested, 'max_safe': max_safe,
                'sl_dist': sl_dist, 'liq_dist': liq_dist,
                'fluct': fluct, 'atr_pct': atr_pct, 'vol_label': vl}
    except Exception:
        return None


def find_pivot_support(df, lookback=60, swing=3):
    """
    FIX #2 — Real structural support via swing-low detection.
    A swing low is a candle whose low is lower than the `swing`
    candles on each side.

    FIX #SL — Previously returned max(pivots), which is the HIGHEST swing low
    in the window.  That is correct for the "nearest support below price" use
    case ONLY when all pivots are below the current price.  But when the market
    has sold off since those pivots formed, max(pivots) can be ABOVE the current
    price, producing a stop-loss above entry — caught by the direction-safety
    guard but wasteful.

    The fix: filter to only swing-lows that are strictly below the current close,
    then return the highest of those (nearest meaningful floor).  If none qualify
    (e.g. price is already at all-time lows in the window), fall back to the
    rolling-min of the lookback window which is guaranteed to be ≤ current price.
    """
    lows        = df['low'].values
    closes      = df['close'].values
    current_price = closes[-1]
    n           = len(lows)
    pivots      = []
    start       = max(swing, n - lookback)
    for i in range(start, n - swing):
        if all(lows[i] <= lows[i - j] for j in range(1, swing + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, swing + 1)):
            pivots.append(lows[i])

    # Keep only pivots that are genuinely below the current price.
    valid = [p for p in pivots if p < current_price]
    if valid:
        return max(valid)   # nearest support floor (highest pivot below price)
    # Fallback: rolling-min is always ≤ current price by definition
    return df['low'].iloc[-lookback:].min()


def find_pivot_resistance(df, lookback=60, swing=3):
    """
    FIX #2 — Real structural resistance via swing-high detection.
    Returns the lowest (closest relevant ceiling) swing high.
    Falls back to 50-period rolling max.
    """
    highs  = df['high'].values
    n      = len(highs)
    pivots = []
    start  = max(swing, n - lookback)
    for i in range(start, n - swing):
        if all(highs[i] >= highs[i - j] for j in range(1, swing + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, swing + 1)):
            pivots.append(highs[i])
    if pivots:
        return min(pivots)
    return df['high'].iloc[-lookback:].max()


def add_indicators(df, timeframe='4h'):
    """
    FIX #2  — Pivot-based S/R instead of rolling min/max
    FIX #5  — Volume MA shifted by 1 so current bar is excluded
    FIX #9  — Stochastic window=5 for 4H crypto sensitivity
               (daily charts keep default window=14)
    """
    try:
        df  = df.copy()
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']

        df['rsi']       = ta.momentum.RSIIndicator(c, window=14).rsi()
        macd            = ta.trend.MACD(c)
        df['macd_diff'] = macd.macd_diff()
        df['macd_line'] = macd.macd()
        df['macd']      = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        bb              = ta.volatility.BollingerBands(c)
        df['bb_upper']  = bb.bollinger_hband()
        df['bb_lower']  = bb.bollinger_lband()
        df['bb_mid']    = bb.bollinger_mavg()
        # FIX #BB — bandwidth = (upper - lower) / mid; squeeze = bw at 20-period low
        df['bb_bw']     = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_bw_min'] = df['bb_bw'].rolling(20).min().shift(1)  # shift so current bar not included
        df['ema20']     = ta.trend.EMAIndicator(c, window=20).ema_indicator()
        df['ema50']     = ta.trend.EMAIndicator(c, window=50).ema_indicator()
        df['atr']       = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

        # FIX #5 — shift(1) excludes the current forming candle from the baseline
        df['volume_ma'] = v.rolling(20).mean().shift(1)

        # FIX #COR — Close Location Value (CLV): genuinely independent of RSI/MACD/EMA.
        # Measures where within the candle range the close settled each bar.
        #   CLV = (2*close - high - low) / (high - low)
        #   +1.0 = close at the very top (buyers fully absorbed the candle)
        #   -1.0 = close at the very bottom (sellers fully absorbed)
        # Rolling 5-bar mean catches sustained absorption without noise.
        hl_range     = (h - l).replace(0, float('nan'))
        df['clv']    = ((2 * c - h - l) / hl_range).fillna(0)
        df['clv_ma'] = df['clv'].rolling(5).mean().shift(1)   # shift: exclude current bar

        # FIX #9 �� use window=5 for 4H (crypto moves fast); daily stays at 14
        stoch_window = 5 if timeframe == '4h' else 14
        stoch        = ta.momentum.StochasticOscillator(h, l, c, window=stoch_window, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        df = df.dropna()

        # FIX #2 — Pivot S/R (computed after dropna so indices are clean)
        if len(df) >= 10:
            df['support']    = find_pivot_support(df)
            df['resistance'] = find_pivot_resistance(df)
        else:
            df['support']    = l.rolling(20).min()
            df['resistance'] = h.rolling(20).max()

        # FIX #DIV — RSI and MACD divergence detection
        # Bullish divergence: price makes lower low but RSI makes higher low → trend exhaustion reversal signal
        # Bearish divergence: price makes higher high but RSI makes lower high → hidden weakness
        # Uses a 5-bar lookback to find the prior swing low/high
        if len(df) >= 6:
            close_s = df['close']
            rsi_s   = df['rsi']
            macd_s  = df['macd_diff']
            # Rolling 5-bar prior swing for comparison
            price_lo5 = close_s.shift(1).rolling(5).min()
            price_hi5 = close_s.shift(1).rolling(5).max()
            rsi_lo5   = rsi_s.shift(1).rolling(5).min()
            rsi_hi5   = rsi_s.shift(1).rolling(5).max()
            macd_lo5  = macd_s.shift(1).rolling(5).min()
            macd_hi5  = macd_s.shift(1).rolling(5).max()

            # Bullish RSI divergence: price lower low, RSI higher low
            df['rsi_bull_div'] = (close_s < price_lo5) & (rsi_s > rsi_lo5) & (rsi_s < 50)
            # Bearish RSI divergence: price higher high, RSI lower high
            df['rsi_bear_div'] = (close_s > price_hi5) & (rsi_s < rsi_hi5) & (rsi_s > 50)
            # Bullish MACD divergence: price lower low, MACD histogram higher low
            df['macd_bull_div'] = (close_s < price_lo5) & (macd_s > macd_lo5) & (macd_s < 0)
            # Bearish MACD divergence: price higher high, MACD histogram lower high
            df['macd_bear_div'] = (close_s > price_hi5) & (macd_s < macd_hi5) & (macd_s > 0)
        else:
            for col in ['rsi_bull_div', 'rsi_bear_div', 'macd_bull_div', 'macd_bear_div']:
                df[col] = False

        return df
    except Exception as e:
        logger.warning("add_indicators error (timeframe=%s): %s", timeframe, e)
        return None


def _get_btc_price_cached() -> float:
    """Return cached BTC/USDT price, refreshing every 60 s."""

    now = datetime.now()
    if state._btc_price_cache and (now - state._btc_price_cache['time']).total_seconds() < _BTC_PRICE_TTL:
        return state._btc_price_cache['price']
    price = 0.0
    try:
        for fn, sym in [
            (lambda s: bybit_fetch_ohlcv(s, '1', 2),  'BTCUSDT'),
            (lambda s: binance_fetch_ohlcv(s, '1m', 2), 'BTCUSDT'),
            (lambda s: mexc_fetch_ohlcv(s, '1m', 2),    'BTCUSDT'),
        ]:
            df = fn(sym)
            if df is not None and len(df) >= 1:
                price = float(df['close'].iloc[-1])
                break
    except Exception as e:
        logger.warning("btc price fetch failed for %s: %s", sym, e)
    state._btc_price_cache = {'price': price, 'time': now}
    return price


def get_btc_regime():
    """
    Determine the current BTC market regime using 4H OHLCV data.
    Tries Bybit BTCUSDT first, falls back to Binance, then MEXC.
    Returns one of: 'STRONG_BULL', 'BULL', 'NEUTRAL', 'BEAR', 'STRONG_BEAR'.
    Caches the result for _btc_regime_cache_ttl seconds.
    """


    # Serve cache if still fresh
    if state._btc_regime_cache is not None:
        age = (datetime.now() - state._btc_regime_cache['time']).total_seconds()
        if age < _btc_regime_cache_ttl:
            return state._btc_regime_cache['regime']

    df = None
    for fetch_fn, symbol in [
        (lambda s: bybit_fetch_ohlcv(s, '240', 150),   'BTCUSDT'),
        (lambda s: binance_fetch_ohlcv(s, '4h', 150),  'BTCUSDT'),
        (lambda s: mexc_fetch_ohlcv(s, '4h', 150),     'BTCUSDT'),
    ]:
        try:
            df = fetch_fn(symbol)
            if df is not None and len(df) >= 80:
                break
            df = None
        except Exception:
            df = None

    if df is None:
        logger.warning("BTC regime: could not fetch OHLCV — defaulting NEUTRAL")
        state._btc_regime_cache = {'regime': 'NEUTRAL', 'time': datetime.now()}
        return 'NEUTRAL'

    df = add_indicators(df, timeframe='4h')
    if df is None or len(df) < 20:
        state._btc_regime_cache = {'regime': 'NEUTRAL', 'time': datetime.now()}
        return 'NEUTRAL'

    # ── Factor 1 — EMA alignment ───────────────────────────────────────
    last  = df.iloc[-1]
    ema20 = last['ema20']
    ema50 = last['ema50']
    rsi   = last['rsi']
    bull_ema = ema20 > ema50
    bear_ema = ema20 < ema50

    # ── Factor 2 — ADX trend strength ─────────────────────────────────
    # ADX measures trend strength regardless of direction.
    # < 20: no trend (ranging/choppy)
    # 20-25: weak trend forming
    # > 25: established trend
    try:
        adx_ind = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        adx = float(adx_ind.adx().iloc[-1])
    except Exception:
        adx = 0.0
    trend_present  = adx > 20
    trend_strong   = adx > 25

    # ── Factor 3 — EMA gap slope (trend accelerating or dying?) ────────
    # Compare the EMA20-EMA50 gap now vs 3 candles ago.
    # Widening gap = trend strengthening. Narrowing = trend exhausting.
    try:
        gap_now  = abs(df['ema20'].iloc[-1] - df['ema50'].iloc[-1])
        gap_prev = abs(df['ema20'].iloc[-4] - df['ema50'].iloc[-4])
        gap_widening  = gap_now > gap_prev * 1.02   # gap grew by >2%
        gap_narrowing = gap_now < gap_prev * 0.98   # gap shrank by >2%
    except Exception:
        gap_widening = gap_narrowing = False

    # ── Factor 4 — EMA crossover recency ───────────────────────────────
    # Walk back up to 50 candles to find the last time EMA20 crossed EMA50.
    # fresh_cross  (< 8 candles): early in trend, higher false-signal risk
    # established  (8–40 candles): most reliable regime zone
    # old_cross   (> 40 candles): trend may be maturing / reversal risk
    cross_age = 99  # default: no cross found in lookback
    try:
        ema20_s = df['ema20']
        ema50_s = df['ema50']
        for i in range(1, min(51, len(df))):
            curr_above = ema20_s.iloc[-i]   > ema50_s.iloc[-i]
            prev_above = ema20_s.iloc[-i-1] > ema50_s.iloc[-i-1]
            if curr_above != prev_above:
                cross_age = i
                break
    except Exception:
        cross_age = 99
    fresh_cross      = cross_age <= 4    # very new — cautious
    established_cross = 4 < cross_age <= 40  # sweet spot
    # cross_age > 40: old trend, may be late

    # ── Factor 5 — RSI (tightened, non-overlapping bands) ──────────────
    # Old: BULL >45, BEAR <55 → overlap 45–55 = ambiguous 25% of the time.
    # New: BULL >52, BEAR <48 → clean gap, no overlap.
    rsi_bull = rsi > 52
    rsi_bear = rsi < 48
    rsi_neutral = not rsi_bull and not rsi_bear  # 48–52: genuinely ambiguous

    # ── Classify regime ────────────────────────────────────────────────
    if bull_ema:
        if trend_strong and gap_widening and rsi_bull:
            regime = 'STRONG_BULL'
        elif trend_present and rsi_bull and not gap_narrowing:
            regime = 'BULL'
        elif (fresh_cross or rsi_neutral or gap_narrowing or not trend_present):
            # EMA says bull but evidence is weak or mixed — could still reverse
            regime = 'NEUTRAL'
        else:
            regime = 'BULL'
    elif bear_ema:
        if trend_strong and gap_widening and rsi_bear:
            regime = 'STRONG_BEAR'
        elif trend_present and rsi_bear and not gap_narrowing:
            regime = 'BEAR'
        elif (fresh_cross or rsi_neutral or gap_narrowing or not trend_present):
            regime = 'NEUTRAL'
        else:
            regime = 'BEAR'
    else:
        regime = 'NEUTRAL'

    state._btc_regime_cache = {'regime': regime, 'time': datetime.now()}
    logger.info(
        "BTC regime: %s (EMA20=%.0f EMA50=%.0f RSI=%.1f ADX=%.1f "
        "gap_trend=%s cross_age=%d candles)",
        regime, ema20, ema50, rsi, adx,
        'widening' if gap_widening else ('narrowing' if gap_narrowing else 'flat'),
        cross_age
    )
    return regime


def session_context() -> dict:
    """
    Returns the current trading session and a score modifier for the signal bias.

    Returns:
        {
          'session':  str    — 'ASIAN' | 'LONDON' | 'OVERLAP' | 'NY' | 'DEAD'
          'modifier': float  — quality bonus/penalty fed into ig_l/ig_s (+1 to -1)
          'note':     str    — human-readable context for signal card
          'bias_warning': bool — True if the session typically fakes in the signal direction
        }

    Score modifiers:
        OVERLAP  +1.0  — NY+London dual liquidity; real trend entries
        NY       +0.5  — strong continuation session
        LONDON   +0.25 — trend initiating but manipulation-wick risk
        ASIAN    -0.5  — consolidation; breakouts often reversed at London open
        DEAD     -1.0  — extremely thin; signals here have lowest follow-through
    """
    hour = datetime.utcnow().hour   # 0–23

    if 12 <= hour < 17:
        session, modifier = 'OVERLAP', 1.0
        note = "NY+London overlap — highest real liquidity window ✅"
        bias_warning = False
    elif 13 <= hour < 21:
        session, modifier = 'NY', 0.5
        note = "NY session — trend continuation favoured ✅"
        bias_warning = False
    elif 7 <= hour < 12:
        session, modifier = 'LONDON', 0.25
        note = "London open — trend initiation, watch for manipulation wicks ⚠️"
        bias_warning = True   # London open wicks can flush before trend direction
    elif 0 <= hour < 8:
        session, modifier = 'ASIAN', -0.5
        note = "Asian session — low volatility, breakouts frequently reversed at London open ⚠️"
        bias_warning = True
    else:   # 21–23
        session, modifier = 'DEAD', -1.0
        note = "Dead zone (21–00 UTC) — thin market, low follow-through ❌"
        bias_warning = True

    return {
        'session':      session,
        'modifier':     modifier,
        'note':         note,
        'bias_warning': bias_warning,
    }


def fib_confluence_score(df, price: float, support: float, resistance: float,
                          atr: float, bias: str) -> tuple:
    """
    FIX #FIB — Fibonacci Retracement Confluence

    Finds the last significant swing high and swing low in the lookback window,
    computes the standard Fibonacci retracement levels between them, then checks
    whether the pivot S/R level used by score_pair coincides with any Fib level.

    A confluence is defined as: |pivot_level - fib_level| / price < tolerance.

    Returns (score: int, fib_levels: list[float], confluence_note: str)
        score          — 0 (no confluence) or +1 (confluence) or +2 (tight confluence)
        fib_levels     — list of computed fib prices for signal card display
        confluence_note — human-readable description for reasons list
    """
    _FIB_TOLERANCE = 0.005   # 0.5% — tighter than ATR-based check; genuinely confluent

    try:
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n      = len(highs)

        if n < 20:
            return 0, [], ""

        # Look back over last 60 bars for the significant swing high and low.
        # Use a 3-bar swing definition consistent with find_pivot_support/resistance.
        lookback   = min(60, n - 1)
        swing      = 3
        start      = max(swing, n - lookback)

        swing_highs = []
        swing_lows  = []
        for i in range(start, n - swing):
            if all(highs[i] >= highs[i - j] for j in range(1, swing + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, swing + 1)):
                swing_highs.append((i, highs[i]))
            if all(lows[i] <= lows[i - j] for j in range(1, swing + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, swing + 1)):
                swing_lows.append((i, lows[i]))

        if not swing_highs and not swing_lows:
            return 0, [], ""

        # Use the most recent significant swing high and swing low.
        # Fall back to rolling window extremes when a swing side has no pivots
        # (e.g. strong trending market with no opposing swing).
        if swing_highs:
            sh_idx, sh_price = max(swing_highs, key=lambda x: x[1])
        else:
            sh_idx  = n - 1
            sh_price = float(df['high'].iloc[max(0, n - lookback):].max())

        if swing_lows:
            sl_idx, sl_price = min(swing_lows,  key=lambda x: x[1])
        else:
            sl_idx  = n - 1
            sl_price = float(df['low'].iloc[max(0, n - lookback):].min())

        swing_range = sh_price - sl_price
        if swing_range < atr * 0.5:
            # Swing range too small to be meaningful �� skip
            return 0, [], ""

        # Compute Fibonacci levels (retracement from the swing extremes)
        # Standard retracements measured from the bottom of the move (sl_price)
        fib_levels = {
            ratio: sl_price + ratio * swing_range
            for ratio in _FIB_RATIOS
        }

        # Check pivot level against each fib level
        # For LONG: relevant pivot is support (floor), should be near 0.382/0.5/0.618
        # For SHORT: relevant pivot is resistance (ceiling), should be near 0.236/0.382/0.5
        pivot_level  = support  if bias == 'LONG' else resistance
        fib_priority = (
            [0.618, 0.500, 0.382, 0.786, 0.236] if bias == 'LONG'
            else [0.236, 0.382, 0.500, 0.618, 0.786]
        )

        best_score = 0
        best_note  = ""
        for ratio in fib_priority:
            fib_price = fib_levels[ratio]
            deviation = abs(pivot_level - fib_price) / price if price > 0 else 1.0
            if deviation < _FIB_TOLERANCE / 2:          # < 0.25% — very tight
                best_score = 2
                best_note  = (f"🎯 Fib {ratio:.3f} confluence ({deviation*100:.2f}% deviation) "
                              f"at ${fib_price:.4f} — high-probability structural zone")
                break
            elif deviation < _FIB_TOLERANCE:             # < 0.5% — confluent
                if best_score < 1:
                    best_score = 1
                    best_note  = (f"Fib {ratio:.3f} near pivot ({deviation*100:.2f}% deviation) "
                                  f"at ${fib_price:.4f} — structural confluence")

        return best_score, list(fib_levels.values()), best_note

    except Exception as _fib_e:
        logger.debug("fib_confluence_score error: %s", _fib_e)
        return 0, [], ""


def candle_quality_score(candle, bias: str) -> tuple:
    """
    FIX #CQ — Candle Quality Scoring
    ───��─────────────────────────────
    Scores the signal candle on three independent structural properties.
    A doji and a full-body engulfing candle are NOT the same signal —
    this function quantifies that difference and feeds it into confidence.

    Returns (score: float, label: str, detail: str)

    score > 0.0  → quality bonus  (max +1.0 fed to quality_bonus)
    score < 0.0  → quality penalty (min -1.0 applied as confidence demotion)
    score = 0.0  → neutral / indeterminate candle

    Three sub-scores (each ���1 to +1), averaged then clamped to [–1, +1]:

    1. BODY RATIO  — body / total range
       A candle whose body fills > 60 % of its range committed to a direction.
       A body < 25 % (doji zone) is structural indecision.

    2. CLOSE POSITION — where close sits in the range [0 = low, 1 = high]
       For LONG:  close near the top (> 0.65) is bullish conviction.
                  close near the bottom (< 0.35) negates the signal.
       For SHORT: mirrored.

    3. WICK ASYMMETRY — upper_wick vs lower_wick ratio in the signal direction
       For LONG:  lower wick > upper wick → rejection of lows, bullish.
                  upper wick > 2× lower wick → sellers absorbed the move.
       For SHORT: upper wick > lower wick → rejection of highs, bearish.
    """
    try:
        o, h, l, c = float(candle['open']), float(candle['high']), float(candle['low']), float(candle['close'])
        total_range = h - l
        if total_range < 1e-12:
            return 0.0, "NEUTRAL", "zero-range candle — skip"

        body       = abs(c - o)
        body_ratio = body / total_range                          # [0, 1]

        close_pos  = (c - l) / total_range                      # [0, 1]; 0=closed at low, 1=at high

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        wick_sum   = upper_wick + lower_wick

        # ── sub-score 1: body ratio ─────────────────────────────────────
        if body_ratio >= 0.70:     body_sub =  1.0
        elif body_ratio >= 0.50:   body_sub =  0.5
        elif body_ratio >= 0.35:   body_sub =  0.0
        elif body_ratio >= 0.20:   body_sub = -0.5
        else:                      body_sub = -1.0   # doji

        # Doji override: body < 10% is structural indecision — close position and
        # wick asymmetry are meaningless noise on a candle with no body conviction.
        # Short-circuit and return POOR immediately.
        if body_ratio < 0.10:
            return -0.80, "POOR", (f"Doji / pin bar — body={body_ratio:.0%}, no directional conviction")

        # ── sub-score 2: close position (direction-aware) ────────────────
        if bias == 'LONG':
            if close_pos >= 0.70:    close_sub =  1.0
            elif close_pos >= 0.50:  close_sub =  0.5
            elif close_pos >= 0.35:  close_sub =  0.0
            elif close_pos >= 0.20:  close_sub = -0.5
            else:                    close_sub = -1.0
        else:  # SHORT
            if close_pos <= 0.30:    close_sub =  1.0
            elif close_pos <= 0.50:  close_sub =  0.5
            elif close_pos <= 0.65:  close_sub =  0.0
            elif close_pos <= 0.80:  close_sub = -0.5
            else:                    close_sub = -1.0

        # ── sub-score 3: wick asymmetry (direction-aware) ───────────────
        if wick_sum < 1e-12:
            wick_sub = 0.0          # no wicks at all — body-only candle, neutral here
        elif bias == 'LONG':
            # Bullish: lower wick absorbs selling, upper wick is healthy extension
            # Penalise: large upper wick means the rally was rejected hard
            ratio = lower_wick / (upper_wick + 1e-12)
            if ratio >= 2.0:       wick_sub =  1.0   # strong lower wick / rejection
            elif ratio >= 1.0:     wick_sub =  0.5
            elif ratio >= 0.5:     wick_sub =  0.0
            elif ratio >= 0.25:    wick_sub = -0.5
            else:                  wick_sub = -1.0   # upper wick >> lower wick, bullish move rejected
        else:  # SHORT
            ratio = upper_wick / (lower_wick + 1e-12)
            if ratio >= 2.0:       wick_sub =  1.0
            elif ratio >= 1.0:     wick_sub =  0.5
            elif ratio >= 0.5:     wick_sub =  0.0
            elif ratio >= 0.25:    wick_sub = -0.5
            else:                  wick_sub = -1.0

        raw   = (body_sub + close_sub + wick_sub) / 3.0
        score = max(-1.0, min(1.0, raw))

        # Human-readable label and detail
        if score >= 0.55:
            label  = "STRONG"
            detail = (f"body={body_ratio:.0%}, close_pos={close_pos:.0%}, "
                      f"body_sub={body_sub:+.1f} close_sub={close_sub:+.1f} wick_sub={wick_sub:+.1f}")
        elif score >= 0.15:
            label  = "GOOD"
            detail = (f"body={body_ratio:.0%}, close_pos={close_pos:.0%}")
        elif score >= -0.15:
            label  = "NEUTRAL"
            detail = (f"body={body_ratio:.0%} — mid-range close or equal wicks")
        elif score >= -0.55:
            label  = "WEAK"
            detail = (f"body={body_ratio:.0%}, close_pos={close_pos:.0%} — indecisive candle")
        else:
            label  = "POOR"
            detail = (f"body={body_ratio:.0%} doji/spinning top, close_pos={close_pos:.0%} — "
                      "signal candle shows no directional conviction")

        return score, label, detail

    except Exception as _cq_e:
        logger.debug("candle_quality_score error: %s", _cq_e)
        return 0.0, "NEUTRAL", "calc error"


def confirm_rr(entry, sl, t1, t2, t3, min_rr=1.5):
    """Pre-trade gate: True only if the entry->T1 reward clears `min_rr`x the
    entry->SL risk. R:R is measured from the actual fill price (`entry`), not
    the signal candle, so the enforced floor matches the displayed R:R
    (_rr_ratio). T1 is the binding target every signal must justify; t2/t3 are
    accepted for completeness/future use. Returns False on invalid geometry.
    """
    try:
        entry = float(entry); sl = float(sl); t1 = float(t1)
        risk = abs(entry - sl)
        if risk <= 0:
            return False
        return (abs(t1 - entry) / risk) >= float(min_rr)
    except (TypeError, ValueError):
        return False


def score_pair(df4h, df1d, funding, symbol, user_requested: bool = False):
    """
    Scoring engine — all bias/accuracy fixes applied:
    FIX #1  — Confidence formula: pure ratio-based, no raw-score inflation
    FIX #3  — Daily reads use iloc[-2] (last CLOSED candle, not forming one)
    FIX #4  — MACD crossover confirmed over 2 bars + histogram growth required
    FIX #6  — Funding thresholds raised to meaningful levels (±0.03% / ±0.07%)
    FIX #7  — T1/T2 capped against nearest pivot S/R level
    FIX #8  — Counter-trend veto: opposing daily EMA raises qualification bar
    FIX #RG — BTC market regime gate: suppresses contra-regime signals below conf 9
    FIX #MS — Minimum absolute score gate: downgrade conf when winning score too thin
               (prevents RSI alone from producing a 10/10 signal on 4:0 ratio)
    FIX #EZ — Entry zone freshness flag: tags signal if price was outside zone at
               scan time so format_signal can warn the user before they enter
    FIX #VA — Volatility-aware targets: T1/T2/T3 multipliers and entry/SL width
               scale with ATR% regime (RANGING/LOW/MEDIUM/HIGH/EXTREME) so targets
               are realistic for both stagnant and explosive coins
    FIX #TP — Guaranteed T1 < T2 < T3 ordering (LONG) / T1 > T2 > T3 (SHORT)
               via sequential clamping — prior approach could produce inversions
               when pivot resistance cap pushed T1 past T2_raw
    FIX #BB — BB squeeze breakout scoring: bandwidth expanding from 20-bar low
               adds +2 score in the breakout direction — consolidation breakouts
               are high-probability entries not captured by RSI/MACD alone
    FIX #OI — Outcome resolver: same-candle SL+target now correctly preserves
               prior confirmed target as sl_after_target=1 instead of always
               overwriting with sl_hit, eliminating a systematic win undercount
    FIX #SR — Scan sort tiebreaker: confidence+score ties resolved by vol regime
               (MEDIUM > HIGH > LOW > EXTREME > RANGING) for better signal ordering
    FIX #CQ — Candle Quality Scoring: signal candle scored on body ratio, close
               position within range, and wick asymmetry. STRONG/GOOD candles get
               a confidence bonus (max +1); WEAK/POOR candles get a penalty (−1
               or −2) that can drop confidence below the gate and suppress the
               signal entirely.  Filters out dojis and spinning tops that the
               existing logic accepted as valid breakout candles.
    FIX #SESSION — Session Awareness: UTC session tagged and scored. OVERLAP and NY
               sessions add to ig (uncapped). ASIAN and DEAD sessions penalise ig —
               signals in thin/consolidation windows require the rest of the stack
               to be stronger to compensate.  Session note surfaced in signal card.
    FIX #FIB — Fibonacci Confluence: 0.236/0.382/0.5/0.618/0.786 Fibonacci
               retracement levels computed from last major swing H/L in the 60-bar
               lookback. When pivot S/R coincides with a Fib level within 0.5%,
               the zone scores +1 to vg (structure); within 0.25% scores +2.
               Displayed in signal card details. Purely additive — never penalises.
    """
    try:
        # ── CRYPTO-ONLY GATE ── reject tokenised stocks/metals/oil/FX outright
        # so crypto indicators and the BTC-regime gate are never applied to
        # non-crypto instruments. (/analyse uses a separate raw path and is
        # intentionally unaffected.)
        if not sakz_exchanges.is_crypto_symbol(symbol):
            logger.debug("NON-CRYPTO skipped at scoring: %s", symbol)
            return None
        # Warning tags — set below if signal passes with caveats.
        # Always initialized so score_pair always attaches them to result.
        low_conf_warning = None
        regime_warning   = None
        regime_blocked   = False
        regime_block_detail = None
        btc_regime       = 'UNKNOWN'

        L   = df4h.iloc[-1]   # current closed 4H candle
        P   = df4h.iloc[-2]   # previous 4H candle
        P2  = df4h.iloc[-3]   # two 4H candles ago (2-bar MACD confirmation)

        # FIX #3 — use the last CLOSED daily candle, never the forming one
        # iloc[-1] is still forming mid-day; its MACD/RSI values are noise.
        LD  = df1d.iloc[-2]   # last closed daily candle
        PD  = df1d.iloc[-3]   # previous closed daily candle

        price      = L['close']
        rsi4       = L['rsi']
        macd_diff  = L['macd_diff']; prev_diff  = P['macd_diff']; prev2_diff = P2['macd_diff']
        bb_upper   = L['bb_upper'];  bb_lower   = L['bb_lower']
        ema20      = L['ema20'];     ema50      = L['ema50']
        support    = float(L['support']); resistance = float(L['resistance'])
        atr        = L['atr']
        vol        = L['volume'];    vol_ma     = L['volume_ma']
        stoch_k    = L['stoch_k'];   stoch_d    = L['stoch_d']

        # FIX #3 — daily indicators from the closed candle
        rsi_d   = LD['rsi']
        macd_d  = LD['macd_diff']; prev_d = PD['macd_diff']
        ema20_d = LD['ema20'];     ema50_d = LD['ema50']

        # ── FIX #COR / FIX #SI — Grouped scoring accumulators ─────────
        # Six independent buckets instead of four, addressing ceiling #1:
        # the old mg (momentum) and tg (trend) buckets each contained two
        # sub-families that fire simultaneously from the same price event.
        #
        # SPLIT: momentum → osc_g + mtf_g
        #   osc_g = Oscillator group  : RSI 4H + Stochastic
        #           Both react to 4H price, but via different formulas.
        #           Cap = 3 so max contribution from "price is low" = 3.
        #   mtf_g = Multi-TF momentum : RSI Daily only
        #           Daily RSI is structurally the same formula as 4H RSI —
        #           they share the same underlying price stream, so they are
        #           NOT independent.  Treated as a cross-timeframe confirmation
        #           multiplier (max +2) rather than an additive peer.
        #
        # SPLIT: trend → cross_g + pos_g
        #   cross_g = Crossover group : MACD 4H + MACD Daily
        #             MACD is derived from price, but a *crossover event* on
        #             two different timeframes is two different signals — a
        #             4H cross is a 16-bar event; a daily cross is a 365-bar
        #             event.  Cap = 3 so a single timeframe MACD cross tops out.
        #   pos_g   = Position group  : EMA 4H + EMA Daily
        #             "Price above EMA" is structural positioning, not momentum.
        #             Cap = 3.  Aligns with cross_g but is a separate question
        #             (where you are vs where you're going).
        #
        #   vg = Structure group: BB band, BB squeeze, Pivot S/R  (cap 4)
        #   ig = Independent    : Volume, Funding, CLV (uncapped)
        #
        # Net effect: a single strong 4H candle can max osc_g (cap 3) OR
        # cross_g (cap 3), but not both to their old combined cap of 4+5=9.
        # Reaching high total score now genuinely requires 4+ distinct signals.
        osc_g_l,  osc_g_s  = 0, 0   # oscillators  (RSI 4H, Stochastic)
        mtf_g_l,  mtf_g_s  = 0, 0   # cross-TF     (RSI Daily)
        cross_g_l, cross_g_s = 0, 0 # MACD crossovers (4H + Daily)
        pos_g_l,  pos_g_s  = 0, 0   # EMA position  (4H + Daily)
        vg_l, vg_s = 0, 0
        ig_l, ig_s = 0, 0
        lr, sr = [], []

        # ── RSI 4H → oscillator bucket ─────────────────────────────────
        if rsi4 < 30:    osc_g_l+=3; lr.append(f"RSI 4H extremely oversold ({rsi4:.1f})")
        elif rsi4 < 40:  osc_g_l+=2; lr.append(f"RSI 4H oversold ({rsi4:.1f})")
        elif rsi4 < 48:  osc_g_l+=1; lr.append(f"RSI 4H leaning oversold ({rsi4:.1f})")
        elif rsi4 > 70:  osc_g_s+=3; sr.append(f"RSI 4H extremely overbought ({rsi4:.1f})")
        elif rsi4 > 60:  osc_g_s+=2; sr.append(f"RSI 4H overbought ({rsi4:.1f})")
        elif rsi4 > 52:  osc_g_s+=1; sr.append(f"RSI 4H leaning overbought ({rsi4:.1f})")
        rsi4_prev = P['rsi'] if 'rsi' in P.index else rsi4
        if 48 <= rsi4 <= 52:
            if rsi4 > rsi4_prev:   osc_g_l+=1; lr.append(f"RSI 4H crossing 50 upward — bullish momentum ({rsi4:.1f})")
            elif rsi4 < rsi4_prev: osc_g_s+=1; sr.append(f"RSI 4H crossing 50 downward — bearish momentum ({rsi4:.1f})")

        # Daily RSI → cross-TF confirmation bucket (structurally same formula,
        # different timeframe — treated as confirming multiplier, cap=2)
        if rsi_d < 35:   mtf_g_l+=2; lr.append(f"RSI Daily oversold ({rsi_d:.1f}) [closed candle]")
        elif rsi_d < 45: mtf_g_l+=1; lr.append(f"RSI Daily weak ({rsi_d:.1f}) [closed candle]")
        elif rsi_d > 65: mtf_g_s+=2; sr.append(f"RSI Daily overbought ({rsi_d:.1f}) [closed candle]")
        elif rsi_d > 55: mtf_g_s+=1; sr.append(f"RSI Daily elevated ({rsi_d:.1f}) [closed candle]")

        # ── FIX #4 — MACD → crossover bucket ──────────────────────────
        # MACD crossovers go to cross_g (cap=3). Sustained momentum without
        # a fresh cross goes to cross_g at reduced weight (+1) — still a
        # crossover-family signal but weaker evidence.
        macd_cross_bull_4h = (macd_diff > 0 and prev_diff <= 0 and
                               abs(macd_diff) > abs(prev_diff))
        macd_cross_bear_4h = (macd_diff < 0 and prev_diff >= 0 and
                               abs(macd_diff) > abs(prev_diff))
        macd_sust_bull_4h  = (macd_diff > 0 and prev_diff > 0 and
                               prev2_diff <= 0 and macd_diff > prev_diff)
        macd_sust_bear_4h  = (macd_diff < 0 and prev_diff < 0 and
                               prev2_diff >= 0 and macd_diff < prev_diff)

        if macd_sust_bull_4h:
            cross_g_l+=3; lr.append("MACD confirmed bullish crossover 4H ⚡ (2-candle hold)")
        elif macd_cross_bull_4h:
            cross_g_l+=2; lr.append("MACD tentative bullish crossover 4H (1 candle — unconfirmed)")
        elif macd_sust_bear_4h:
            cross_g_s+=3; sr.append("MACD confirmed bearish crossover 4H ⚡ (2-candle hold)")
        elif macd_cross_bear_4h:
            cross_g_s+=2; sr.append("MACD tentative bearish crossover 4H (1 candle — unconfirmed)")
        elif macd_diff > 0:
            cross_g_l+=1; lr.append("MACD bullish momentum 4H")
        elif macd_diff < 0:
            cross_g_s+=1; sr.append("MACD bearish momentum 4H")

        # FIX #4 + FIX #3 — daily MACD from closed candle, confirmed crossover
        macd_cross_bull_1d = (macd_d > 0 and prev_d <= 0 and abs(macd_d) > abs(prev_d))
        macd_cross_bear_1d = (macd_d < 0 and prev_d >= 0 and abs(macd_d) > abs(prev_d))

        if macd_cross_bull_1d:
            cross_g_l+=3; lr.append("MACD confirmed bullish crossover Daily ⚡ [closed candle]")
        elif macd_cross_bear_1d:
            cross_g_s+=3; sr.append("MACD confirmed bearish crossover Daily ⚡ [closed candle]")
        elif macd_d > 0:
            cross_g_l+=1; lr.append("MACD bullish Daily [closed candle]")
        elif macd_d < 0:
            cross_g_s+=1; sr.append("MACD bearish Daily [closed candle]")

        # ── EMA → position bucket ─────────────────────────────��─��───��───
        if price > ema20 > ema50:   pos_g_l+=2; lr.append("Bullish EMA stack 4H")
        elif price < ema20 < ema50: pos_g_s+=2; sr.append("Bearish EMA stack 4H")
        elif price > ema20:         pos_g_l+=1; lr.append("Price above EMA20 4H")
        elif price < ema20:         pos_g_s+=1; sr.append("Price below EMA20 4H")

        # Daily EMA from closed candle (FIX #3) — same pos_g bucket
        if price > ema20_d > ema50_d:   pos_g_l+=2; lr.append("Bullish EMA stack Daily [closed]")
        elif price < ema20_d < ema50_d: pos_g_s+=2; sr.append("Bearish EMA stack Daily [closed]")

        # ── Bollinger Bands ────────────────────────────────────────────
        if price <= bb_lower:   vg_l+=2; lr.append("Price at/below lower Bollinger Band")
        elif price >= bb_upper: vg_s+=2; sr.append("Price at/above upper Bollinger Band")

        # FIX #BB — BB squeeze breakout
        bb_bw     = L.get('bb_bw',     None) if hasattr(L, 'get') else L['bb_bw']     if 'bb_bw'     in L.index else None
        bb_bw_min = L.get('bb_bw_min', None) if hasattr(L, 'get') else L['bb_bw_min'] if 'bb_bw_min' in L.index else None
        if bb_bw is not None and bb_bw_min is not None and bb_bw_min > 0:
            squeeze_expanding = (bb_bw > bb_bw_min * 1.05)
            if squeeze_expanding:
                if price > P['close']:  vg_l+=2; lr.append(f"BB squeeze breakout BULLISH (bw expanding from floor)")
                else:                   vg_s+=2; sr.append(f"BB squeeze breakout BEARISH (bw expanding from floor)")

        # ── Stochastic → oscillator bucket (same osc_g as RSI 4H) ────────
        # Stochastic and RSI both measure price momentum on the 4H timeframe.
        # Both fire on the same candle, so they share osc_g's cap.
        if stoch_k < 20 and stoch_d < 20:   osc_g_l+=2; lr.append(f"Stochastic oversold (K:{stoch_k:.1f})")
        elif stoch_k > 80 and stoch_d > 80: osc_g_s+=2; sr.append(f"Stochastic overbought (K:{stoch_k:.1f})")
        if stoch_k > stoch_d and stoch_k < 45:   osc_g_l+=1; lr.append(f"Stochastic bullish cross low zone (K:{stoch_k:.1f})")
        elif stoch_k < stoch_d and stoch_k > 55: osc_g_s+=1; sr.append(f"Stochastic bearish cross high zone (K:{stoch_k:.1f})")

        # ── FIX #DIV — RSI/MACD Divergence Detection ──────────────────
        # Divergence is independent of the oscillator level (RSI<30 and
        # RSI-divergence are different signals from the same indicator family
        # but carry different information — level vs structural reversal).
        # Placed in vg (structure group) since divergence is a structural
        # break in indicator behaviour, not a momentum threshold crossing.
        _rsi_bull_div  = bool(L.get('rsi_bull_div',  False) if hasattr(L, 'get') else (L['rsi_bull_div']  if 'rsi_bull_div'  in L.index else False))
        _rsi_bear_div  = bool(L.get('rsi_bear_div',  False) if hasattr(L, 'get') else (L['rsi_bear_div']  if 'rsi_bear_div'  in L.index else False))
        _macd_bull_div = bool(L.get('macd_bull_div', False) if hasattr(L, 'get') else (L['macd_bull_div'] if 'macd_bull_div' in L.index else False))
        _macd_bear_div = bool(L.get('macd_bear_div', False) if hasattr(L, 'get') else (L['macd_bear_div'] if 'macd_bear_div' in L.index else False))

        if _rsi_bull_div:
            vg_l += 2; lr.append(f"🔀 Bullish RSI divergence — price lower low, RSI higher low (reversal signal)")
        if _macd_bull_div:
            vg_l += 2; lr.append(f"🔀 Bullish MACD divergence — price lower low, MACD histogram recovering")
        if _rsi_bear_div:
            vg_s += 2; sr.append(f"🔀 Bearish RSI divergence — price higher high, RSI lower high (exhaustion)")
        if _macd_bear_div:
            vg_s += 2; sr.append(f"🔀 Bearish MACD divergence — price higher high, MACD histogram weakening")

        # ── Volume (vol_ma already shifted by 1 — FIX #5) ─────────────
        if vol_ma and vol_ma > 0:
            ratio = vol / vol_ma
            if ratio > 1.8:
                if price > P['close']: ig_l+=2; lr.append(f"Strong volume bullish candle ({ratio:.1f}x avg)")
                else:                  ig_s+=2; sr.append(f"Strong volume bearish candle ({ratio:.1f}x avg)")
            elif ratio > 1.3:
                if price > P['close']: ig_l+=1; lr.append(f"Above avg volume bullish ({ratio:.1f}x)")
                else:                  ig_s+=1; sr.append(f"Above avg volume bearish ({ratio:.1f}x)")

        # ── Pivot S/R proximity (FIX #2 — real pivot levels) ──────────
        if atr > 0:
            if (price - support) < atr * 0.5:    vg_l+=2; lr.append(f"Near pivot support (${support:.4f})")
            if (resistance - price) < atr * 0.5: vg_s+=2; sr.append(f"Near pivot resistance (${resistance:.4f})")

        # ── FIX #FIB — Fibonacci retracement confluence ────────────────
        # Compute for BOTH directions now; apply the directional score to
        # vg_l / vg_s, and store levels for the result dict regardless.
        # Purely additive — never penalises, never blocks.
        _fib_l_score, _fib_levels, _fib_l_note = fib_confluence_score(df4h, price, support, resistance, atr, 'LONG')
        _fib_s_score, _fib_levels_s, _fib_s_note = fib_confluence_score(df4h, price, support, resistance, atr, 'SHORT')
        if _fib_l_score > 0 and _fib_l_note:
            vg_l += _fib_l_score; lr.append(_fib_l_note)
        if _fib_s_score > 0 and _fib_s_note:
            vg_s += _fib_s_score; sr.append(_fib_s_note)
        # Use LONG levels for display (they span the full swing range — same prices)
        _fib_display_levels = _fib_levels if _fib_levels else _fib_levels_s

        # ── FIX #6 — Funding thresholds at meaningful levels ───────────
        if funding != 0:
            fp = funding * 100
            if funding < -0.0007:   ig_l+=3; lr.append(f"Extreme negative funding ({fp:.4f}%) — strong short squeeze risk")
            elif funding < -0.0003: ig_l+=2; lr.append(f"Negative funding ({fp:.4f}%) — shorts paying longs")
            elif funding < 0:       ig_l+=1; lr.append(f"Slightly negative funding ({fp:.4f}%)")
            elif funding > 0.0007:  ig_s+=3; sr.append(f"Extreme positive funding ({fp:.4f}%) — strong long squeeze risk")
            elif funding > 0.0003:  ig_s+=2; sr.append(f"Positive funding ({fp:.4f}%) — longs paying shorts")
            elif funding > 0:       ig_s+=1; sr.append(f"Slightly positive funding ({fp:.4f}%)")

        # ── FIX #COR — Close Location Value (CLV): independent pressure signal ─
        # Where the candle closed within its own range is genuinely uncorrelated
        # with RSI/MACD/EMA because those are derived from close prices across
        # many bars, while CLV measures buyer/seller dominance within a single
        # candle's range.  A sustained rising CLV (5-bar avg) while price falls
        # is real absorption evidence — a different data source entirely.
        clv    = L.get('clv',    None) if hasattr(L, 'get') else (L['clv']    if 'clv'    in L.index else None)
        clv_ma = L.get('clv_ma', None) if hasattr(L, 'get') else (L['clv_ma'] if 'clv_ma' in L.index else None)
        if clv is not None and clv_ma is not None and not (clv != clv):   # not NaN
            if clv > 0.15 and clv > clv_ma:
                ig_l+=2; lr.append(f"CLV bullish: buyers absorbing within candle range ({clv:+.2f})")
            elif clv < -0.15 and clv < clv_ma:
                ig_s+=2; sr.append(f"CLV bearish: sellers dominating within candle range ({clv:+.2f})")

        # ── CONVICTION LAYER — CVD / OI / VWAP ─────────────────────────
        # Three genuinely independent data sources that are NOT correlated
        # with the existing RSI/MACD/EMA/BB buckets.  They go into ig (the
        # uncapped independent group) so their contribution is bounded only
        # by the quality of evidence, not by an artificial cap.
        #
        # Called AFTER bias is decided (CVD/OI/VWAP scoring is directional).
        # Called BEFORE the final ig sum so the points are included in ls/ss.
        #
        # At this point `bias` is already determined from the ls/ss comparison
        # above, but ig_l and ig_s are still being accumulated — the final
        # sum happens below at "Apply group caps, then sum".
        _conv = {}
        if _CONVICTION_AVAILABLE:
            try:
                _binance_ok = sakz_exchanges.BINANCE_AVAILABLE if sakz_exchanges.BINANCE_AVAILABLE is not None else False
                _conv = conviction_scores(symbol, df4h, bias, _binance_ok)
                ig_l += _conv.get('ig_long_bonus',  0)
                ig_s += _conv.get('ig_short_bonus', 0)
                # Surface conviction notes in the relevant reason list
                for _cn in _conv.get('display_lines', []):
                    if bias == 'LONG':
                        lr.append(_cn)
                    else:
                        sr.append(_cn)
            except Exception as _conv_e:
                logger.debug("conviction_scores error for %s: %s", symbol, _conv_e)

        # ── FIX #SESSION — Session awareness ───────────────────────────
        # Session modifier goes into ig (uncapped independent group).
        # Positive: OVERLAP (+1) and NY (+0.5) �� add in the signal direction.
        # Negative: ASIAN (-0.5) and DEAD (-1) — subtract from the signal direction.
        #   (Penalising the direction bucket reduces winning score, which can widen
        #    or narrow the gap, eventually hitting the GAP_BLOCK or LOW_CONF gates.)
        # LONDON (+0.25) adds a small bonus but is flagged as manipulation-wick risk.
        _sess = session_context()
        _sess_mod = _sess['modifier']
        if _sess_mod > 0:
            ig_l += _sess_mod; lr.append(f"📅 Session: {_sess['note']}")
            ig_s += _sess_mod; sr.append(f"📅 Session: {_sess['note']}")
        elif _sess_mod < 0:
            # Penalise by reducing the winning direction's ig score
            # (floor at 0 — never drive ig negative from session alone)
            ig_l = max(0.0, ig_l + _sess_mod)
            ig_s = max(0.0, ig_s + _sess_mod)
            lr.append(f"📅 Session: {_sess['note']}")
            sr.append(f"📅 Session: {_sess['note']}")

        # ── FIX #COR / FIX #SI — Apply group caps, then sum ───────────
        # Six buckets, tighter caps per bucket:
        #   osc_g   cap=3  : RSI 4H + Stochastic (same 4H price, different formula)
        #   mtf_g   cap=2  : RSI Daily (same formula, different TF — confirmation only)
        #   cross_g cap=3  : MACD crossovers 4H + Daily (event-based, two TFs can stack)
        #   pos_g   cap=3  : EMA position 4H + Daily (structural, two TFs can stack)
        #   vg      cap=4  : BB + S/R structure (unchanged)
        #   ig      uncapped: Volume, Funding, CLV (genuinely different sources)
        #
        # Old max from momentum-derivatives alone: mg_cap(4) + tg_cap(5) = 9 points
        # New max from momentum-derivatives alone: osc(3) + mtf(2) + cross(3) + pos(3) = 11 raw
        # but only a market that has BOTH a fresh MACD cross AND EMA alignment AND
        # oversold RSI AND daily confirmation can approach that — four distinct facts,
        # not one strong candle firing all oscillators simultaneously.
        OSC_CAP   = 3   # RSI 4H + Stochastic
        MTF_CAP   = 2   # RSI Daily (confirmation multiplier)
        CROSS_CAP = 3   # MACD crossovers (4H + Daily)
        POS_CAP   = 3   # EMA position (4H + Daily)
        STRUCTURE_CAP = 4   # BB + S/R (unchanged)

        ls = (min(osc_g_l,  OSC_CAP)   + min(mtf_g_l,  MTF_CAP) +
              min(cross_g_l, CROSS_CAP) + min(pos_g_l,  POS_CAP) +
              min(vg_l, STRUCTURE_CAP)  + ig_l)
        ss = (min(osc_g_s,  OSC_CAP)   + min(mtf_g_s,  MTF_CAP) +
              min(cross_g_s, CROSS_CAP) + min(pos_g_s,  POS_CAP) +
              min(vg_s, STRUCTURE_CAP)  + ig_s)

        # ── Bias determination ─────────────────────────────────────────
        # NEUTRAL / very-weak evidence handling.
        #   • Auto-scan (user_requested=False): drop directionless noise so the
        #     alert feed only carries real signals.
        #   • Manual /scan or /chart (user_requested=True): NEVER drop. The user
        #     explicitly asked about this pair, so always return a best-effort
        #     read. The gap / counter-trend / regime / abs-score gates are all
        #     soft for user_requested, so it surfaces as a low-confidence
        #     "weak signal" (e.g. 2/10-4/10) with warnings instead of nothing.
        if ls == ss or (ls < 3 and ss < 3):
            if not user_requested:
                return None
            logger.debug("WEAK SOFT-WARN (user_requested): %s ls=%d ss=%d - "
                         "showing weak signal instead of dropping", symbol, ls, ss)
        if ls >= ss:
            bias, score, reasons = "LONG",  ls, lr
            winning, losing      = ls, ss
        else:
            bias, score, reasons = "SHORT", ss, sr
            winning, losing      = ss, ls

        # ── FIX #GAP — Minimum score gap ───────────────────────────────
        # Winning side must beat losing side by at least 2 points.
        # A 1-point edge (e.g. 5 vs 4) in a ranging market is noise —
        # the next candle can easily flip it, producing alternating signals.
        # When user_requested=True, convert to soft warning so analysis shows.
        if (winning - losing) < 2:
            gap_detail = f"gap={winning-losing} (winning={winning} losing={losing})"
            if user_requested:
                logger.debug("GAP SOFT-WARN (user_requested): %s %s %s", symbol, bias, gap_detail)
                ig_l.append(f"⚠️ Narrow gap: {gap_detail} — choppy market, trade carefully")
            else:
                logger.debug("GAP BLOCK: %s %s gap=%d (winning=%d losing=%d) — too close",
                             symbol, bias, winning - losing, winning, losing)
                return ScanFailure(REASON_GAP_BLOCK, detail=gap_detail)

        # ── FIX #FLIP — Direction flip cooldown ────────────────────────
        # If the last fired signal for this symbol was the OPPOSITE direction
        # within the cooldown window, require confidence ≥ _FLIP_MIN_CONF
        # to override it.  Prevents rapid SHORT→LONG→SHORT churn on auto-scan.
        #
        # When user_requested=True (manual /scan SYMBOL), the hard block is
        # converted to a soft warning so the full analysis is always shown.
        # The regime gate never fired for the user because the flip block was
        # returning before it — this is why it appeared the ADX gate was
        # "too strong".  It wasn't — it never ran.
        _FLIP_WINDOW_H = 6          # was 8; 6h = 1.5 × 4H candles
        _FLIP_MIN_CONF = 9
        last = state._last_signal_bias.get(symbol)
        if last and last['bias'] != bias:
            hours_since = (datetime.now() - last['time']).total_seconds() / 3600
            if hours_since < _FLIP_WINDOW_H:
                tentative_ratio = (winning / (winning + losing)) * 10
                if round(tentative_ratio) < _FLIP_MIN_CONF:
                    flip_detail = (
                        f"tried {last['bias']}→{bias} after {hours_since:.1f}h, "
                        f"conf≈{round(tentative_ratio)} (need {_FLIP_MIN_CONF} to auto-flip)"
                    )
                    if user_requested:
                        # ── SOFT WARN: show analysis anyway for manual queries ──
                        # The user explicitly asked — they want the current picture.
                        logger.debug("FLIP SOFT-WARN (user_requested): %s %s", symbol, flip_detail)
                        ig_l.append(f"⚠️ Flip cooldown: {flip_detail} — showing analysis anyway")
                    else:
                        # ── HARD BLOCK: auto-scan / background jobs ────────────
                        logger.debug(
                            "FLIP BLOCK: %s tried to flip %s→%s after %.1fh (conf≈%d < %d)",
                            symbol, last['bias'], bias, hours_since,
                            round(tentative_ratio), _FLIP_MIN_CONF
                        )
                        return ScanFailure(REASON_FLIP_BLOCK, detail=flip_detail)

        # ── FIX #8 — Counter-trend veto ────────────────────────────────
        # If daily EMA stack strongly opposes the bias, require higher
        # minimum score — prevents weak counter-trend signals qualifying.
        daily_bearish = ema20_d < ema50_d
        daily_bullish = ema20_d > ema50_d
        counter_trend = (bias == 'LONG' and daily_bearish) or \
                        (bias == 'SHORT' and daily_bullish)
        min_score_req = 6 if counter_trend else 3

        if winning < min_score_req:
            ct_detail = f"winning={winning} < min_score_req={min_score_req} ({bias} vs daily EMA)"
            if user_requested:
                logger.debug("COUNTER_TREND SOFT-WARN (user_requested): %s %s", symbol, ct_detail)
                ig_l.append(f"⚠️ Counter-trend: {ct_detail} — signal opposes daily EMA, higher risk")
            else:
                return ScanFailure(REASON_COUNTER_TREND, detail=ct_detail)

        # ── FIX #1 — Confidence: pure ratio, capped quality bonus ──────
        # Old: raw_conf + score * 0.12  → inflated by up to +3 points
        # New: ratio × 10, plus max +1.0 quality bonus for timeframe/volume agreement
        ratio_conf    = (winning / (winning + losing)) * 10
        quality_bonus = 0.0
        if (bias == 'LONG' and daily_bullish) or (bias == 'SHORT' and daily_bearish):
            quality_bonus += 0.5   # both timeframes agree
        vol_ratio_check = (vol / vol_ma) if (vol_ma and vol_ma > 0) else 1.0
        if vol_ratio_check > 1.3 and ((bias == 'LONG' and price > P['close']) or
                                       (bias == 'SHORT' and price < P['close'])):
            quality_bonus += 0.5   # volume confirms direction

        # ── FIX #CQ — Candle Quality gate ──────────────────────────────
        # Score the signal candle on body ratio, close position, and wick
        # asymmetry.  A doji and a full-body engulfing candle are NOT the
        # same signal — this quantifies that difference.
        #
        # quality_score range: [–1.0, +1.0]
        #   STRONG (≥ 0.55) → +0.5 quality_bonus (stacks with TF/vol bonuses)
        #   GOOD   (≥ 0.15) → +0.25 quality_bonus
        #   NEUTRAL          → no change
        #   WEAK   (≥ −0.55) → −1 confidence demotion (post-round, min 4 to pass)
        #   POOR   (< −0.55) → −2 confidence demotion; most false breakouts land here
        #
        # Note: the demotion is applied AFTER rounding so the gate is always
        # a whole-number comparison against the existing LOW_CONF threshold.
        cq_score, cq_label, cq_detail = candle_quality_score(L, bias)
        if cq_score >= 0.55:
            quality_bonus += 0.5
            reasons.append(f"✅ Strong signal candle ({cq_label}): {cq_detail}")
        elif cq_score >= 0.15:
            quality_bonus += 0.25
            reasons.append(f"✔ Good signal candle ({cq_label}): {cq_detail}")
        elif cq_score <= -0.55:
            reasons.append(f"⚠️ Poor signal candle ({cq_label}): {cq_detail} — confidence penalised")
        elif cq_score <= -0.15:
            reasons.append(f"⚠️ Weak signal candle ({cq_label}): {cq_detail} — confidence penalised")

        confidence = min(10, round(ratio_conf + quality_bonus))

        # Apply candle quality demotion AFTER rounding
        if cq_score <= -0.55:
            confidence -= 2   # POOR candle: doji / spinning top / close mid-range + bad wicks
        elif cq_score <= -0.15:
            confidence -= 1   # WEAK candle: indecisive but not a clear doji

        # Low confidence: tag warning but continue — user requested this pair explicitly
        low_conf_warning = None
        if confidence < 4:
            low_conf_warning = f"Very weak signal: conf={confidence}/10 (winning={winning} losing={losing})"
            logger.debug("LOW CONF WARN: %s conf=%d — passing with warning", symbol, confidence)

        # ── FIX #MS — Minimum absolute score gate ──────────────────────
        # A ratio of 4:0 gives confidence=10 but ls=4 is weak evidence —
        # only 4 indicator points touched the entire stack.  Enforce a
        # minimum raw winning score so high confidence requires substance:
        #   conf 9–10 → winning ≥ 9  (very strong absolute signal)
        #   conf 7–8  → winning ≥ 7
        #   conf 4–6  → winning ≥ 5  (existing min_score_req already ≥ 3/6)
        # This prevents a coin with one extreme RSI reading from scoring 10/10.
        # FIX #CR — Confidence Recalibration: raise the absolute score bar for
        # high-conf signals. Previously conf=8 only needed winning≥7, which
        # let moderate-evidence signals inflate to 8/10.
        # New floors: conf 9-10 → winning≥10, conf 8 → winning≥9
        # This reduces the number of high-conf signals but improves their accuracy.
        abs_score_floor = {10: 10, 9: 10, 8: 9, 7: 7, 6: 5, 5: 5, 4: 5}.get(confidence, 5)
        if winning < abs_score_floor:
            # Downgrade confidence rather than drop the signal — it may still
            # be valid but at a more honest conviction level.
            old_conf   = confidence
            confidence = confidence - 1
            if confidence < 4:
                low_conf_warning = (low_conf_warning or "") + f" Abs-score gate: conf {old_conf}→{confidence}"
                logger.debug("ABS SCORE GATE WARN: %s conf %d→%d (winning=%d floor=%d) — passing with warning",
                             symbol, old_conf, confidence, winning, abs_score_floor)
            else:
                logger.debug("ABS SCORE GATE: %s conf %d→%d (winning=%d floor=%d)",
                             symbol, old_conf, confidence, winning, abs_score_floor)

        # ── FIX #RG2 — BTC Market Regime Gate (5-level) ───────────────
        # New get_btc_regime() returns: STRONG_BULL, BULL, NEUTRAL, BEAR, STRONG_BEAR
        #
        # Gate logic — what each regime blocks:
        #
        #   STRONG_BULL → SHORTs blocked below conf 9 (extreme vol: 7)
        #   BULL        → SHORTs blocked below conf 8 (extreme vol: 7)
        #   NEUTRAL     → ALL signals blocked below conf 8
        #                 (previously: -1 conf penalty only — far too weak)
        #                 Rationale: choppy markets are where false signals
        #                 cluster. If the regime is truly neutral, there is
        #                 no directional edge and low-conf signals should not fire.
        #   BEAR        → LONGs  blocked below conf 8 (extreme vol: 7)
        #   STRONG_BEAR → LONGs  blocked below conf 9 (extreme vol: 7)
        #
        # Exception: EXTREME vol regime (ATR% >= 5.5%) lowers all floors by 1.
        # In panic/euphoria conditions reversals are genuinely more likely.
        btc_regime     = get_btc_regime()
        is_btc         = symbol.upper().replace('_USDT', 'USDT') in ('BTCUSDT', 'BTCPERP')
        is_extreme_vol = (((atr / price) * 100) >= 5.5)

        # regime_warning is set when the signal is below the recommended floor
        # for the current BTC regime.  When regime_blocked is also set, the
        # signal is HARD-BLOCKED (user directive): autoscan / /scan drop it and
        # single-pair /scan redirects the user to /analyse.
        regime_warning = None
        regime_blocked = False
        regime_block_detail = None

        # ���─ FIX #BTCD — BTC Dominance Filter ──────────────────────────
        # BTC.D rising = capital rotating from alts to BTC → alt LONGs face headwind.
        # Apply a confidence penalty of −1 for altcoin LONGs when BTC.D is rising.
        # Attach a warning note; never hard-block (user may know why they're trading).
        _btcd_info = _get_btc_dominance()
        _btcd_note = None
        if not is_btc and _btcd_info['trend'] != 'flat' and _btcd_info['btcd'] > 0:
            if _btcd_info['trend'] == 'rising' and bias == 'LONG':
                confidence = max(1, confidence - 1)
                _btcd_note = (f"⚠️ BTC.D rising ({_btcd_info['btcd']:.1f}%) — alt LONG headwind "
                              f"(conf docked −1 → {confidence})")
                reasons.append(_btcd_note)
                logger.debug("BTCD: %s LONG penalised -1 conf (BTC.D rising %.1f%%)",
                             symbol, _btcd_info['btcd'])
            elif _btcd_info['trend'] == 'falling' and bias == 'SHORT':
                _btcd_note = (f"ℹ️ BTC.D falling ({_btcd_info['btcd']:.1f}%) — alt SHORT headwind "
                              f"(capital returning to alts)")
                reasons.append(_btcd_note)

        if not is_btc:
            if btc_regime in ('STRONG_BULL', 'BULL') and bias == 'SHORT':
                floor = (9 if btc_regime == 'STRONG_BULL' else 8)
                if is_extreme_vol: floor = max(floor - 1, 7)
                if confidence < floor:
                    regime_warning = f"Counter-regime SHORT: conf={confidence} below {btc_regime} floor ({floor})"
                    regime_blocked = True
                    regime_block_detail = regime_warning
                    logger.debug("REGIME BLOCK: %s SHORT conf=%d below floor=%d (%s)",
                                 symbol, confidence, floor, btc_regime)
                else:
                    reasons.append(f"⚠️ Counter-regime SHORT in BTC {btc_regime} — high-conf only ({confidence}/10)")

            elif btc_regime in ('STRONG_BEAR', 'BEAR') and bias == 'LONG':
                floor = (9 if btc_regime == 'STRONG_BEAR' else 8)
                if is_extreme_vol: floor = max(floor - 1, 7)
                if confidence < floor:
                    regime_warning = f"Counter-regime LONG: conf={confidence} below {btc_regime} floor ({floor})"
                    regime_blocked = True
                    regime_block_detail = regime_warning
                    logger.debug("REGIME BLOCK: %s LONG conf=%d below floor=%d (%s)",
                                 symbol, confidence, floor, btc_regime)
                else:
                    reasons.append(f"⚠️ Counter-regime LONG in BTC {btc_regime} — high-conf only ({confidence}/10)")

            elif btc_regime == 'NEUTRAL':
                neutral_floor = 7 if is_extreme_vol else 8
                if confidence < neutral_floor:
                    regime_warning = f"{bias} conf={confidence} below NEUTRAL floor ({neutral_floor})"
                    regime_blocked = True
                    regime_block_detail = regime_warning
                    logger.debug("REGIME BLOCK: %s %s conf=%d below neutral floor=%d",
                                 symbol, bias, confidence, neutral_floor)
                else:
                    reasons.append(f"ℹ️ BTC NEUTRAL regime — only high-conf signals pass ({confidence}/10)")


        # ── FIX #VA — Volatility-aware target multipliers ──────────────
        # FIX RR — Risk:Reward restructure.
        # OLD structure had SL > T1 distance in every regime (e.g. MEDIUM:
        # T1=1.5*ATR, SL=2.5*ATR → R:R=0.6). This means you need a >63%
        # win rate just to break even at T1 — mathematically impossible for
        # most technical systems. A 4.3% win rate against this structure
        # produces catastrophic negative EV.
        #
        # NEW structure: SL is always TIGHTER than T1, giving R:R >= 1.3:1
        # at T1. This means a system with even 45% win rate has positive EV.
        # T2 and T3 extend further to reward letting winners run.
        #
        # Regime    | OLD T1/SL      | NEW T1/SL      | R:R (T1)
        # RANGING   | 0.8 / 1.5 ATR  | 1.2 / 0.7 ATR  | 1.71
        # LOW       | 1.2 / 2.0 ATR  | 1.6 / 1.0 ATR  | 1.60
        # MEDIUM    | 1.5 / 2.5 ATR  | 2.0 / 1.3 ATR  | 1.54
        # HIGH      | 1.8 / 3.0 ATR  | 2.5 / 1.6 ATR  | 1.56
        # EXTREME   | 2.2 / 3.5 ATR  | 3.0 / 2.0 ATR  | 1.50
        #
        # Note: SL tightening is safe because Fix SL1 now requires a CLOSE
        # below the SL level (not a wick), so tighter SLs no longer fire on
        # normal candle noise.
        atr_pct_va = (atr / price) * 100

        # FIX #EW — Wider entry zones (+30%) to reduce missed entries.
        # entry_w_lo = lower pullback bound, entry_w_hi = upper chase allowance.
        # Wider zones mean price is more likely to trade through the zone
        # before the signal expires, reducing missed-entry count significantly.
        if atr_pct_va < 1.0:
            vol_regime = 'RANGING'
            t1_mult, t2_mult, t3_mult = 1.2,  1.9,  3.0
            sl_mult                   = 0.7
            entry_w_lo, entry_w_hi    = 0.22, 0.15   # was 0.15 / 0.10
        elif atr_pct_va < 2.0:
            vol_regime = 'LOW'
            t1_mult, t2_mult, t3_mult = 1.6,  2.5,  3.8
            sl_mult                   = 1.0
            entry_w_lo, entry_w_hi    = 0.35, 0.18   # was 0.25 / 0.12
        elif atr_pct_va < 3.5:
            vol_regime = 'MEDIUM'
            t1_mult, t2_mult, t3_mult = 2.0,  3.0,  4.8
            sl_mult                   = 1.3
            entry_w_lo, entry_w_hi    = 0.50, 0.25   # was 0.35 / 0.18
        elif atr_pct_va < 5.5:
            vol_regime = 'HIGH'
            t1_mult, t2_mult, t3_mult = 2.5,  3.8,  5.8
            sl_mult                   = 1.6
            entry_w_lo, entry_w_hi    = 0.65, 0.32   # was 0.45 / 0.22
        else:
            vol_regime = 'EXTREME'
            t1_mult, t2_mult, t3_mult = 3.0,  4.5,  7.0
            sl_mult                   = 2.0
            entry_w_lo, entry_w_hi    = 0.80, 0.42   # was 0.55 / 0.28

        # ── REALISTIC TARGET / STOP CAPS (per vol regime) ──────────────────
        # Raw ATR multiples produce fantasy targets on high-ATR coins (T3 at
        # +39%) and suicidal stops (SL at -27%). Cap the absolute move from
        # entry as a % of price, scaled by vol regime, so T2/T3 are realistic
        # for the swing horizon and SL stays survivable at the leverage used.
        _T2_PCT_CAP = {'RANGING':0.04,'LOW':0.06,'MEDIUM':0.09,'HIGH':0.12,'EXTREME':0.15}[vol_regime]
        _T3_PCT_CAP = {'RANGING':0.07,'LOW':0.10,'MEDIUM':0.15,'HIGH':0.20,'EXTREME':0.25}[vol_regime]
        _SL_PCT_CAP = {'RANGING':0.025,'LOW':0.04,'MEDIUM':0.06,'HIGH':0.08,'EXTREME':0.10}[vol_regime]

        if bias == "LONG":
            # ── FIX #TL (Timing Lag) — Pullback-anchored entry zone ────────
            # The 4H signal fires at candle CLOSE, meaning price has already
            # travelled the full move that triggered the signal.  Setting
            # entry_high = price + X ATR means we're asking the user to chase
            # above the close — the worst possible entry.
            #
            # Instead, anchor the zone BELOW the close price:
            #   entry_high = close - 0.05 ATR  (just under close — don't chase)
            #   entry_low  = close - entry_w_lo ATR  (deeper pullback, still valid)
            #
            # The zone now represents a realistic retracement entry:
            # "I want to buy the dip after the signal candle" not
            # "I want to buy at whatever price I see when I open Telegram".
            #
            # Targets are still calculated from the signal price (close) so
            # R:R is correctly measured from the breakout level, not the entry.
            entry_high = price - atr * 0.05                  # just under close — minimum pullback
            entry_low  = price - atr * (entry_w_lo + entry_w_hi)  # max pullback still in structure
            # Safety floor: entry_low must stay above stop_loss
            stop_loss  = max(support - atr * 0.3, price - atr * sl_mult)
            # Direction safety: SL must always be BELOW price for a LONG.
            # find_pivot_support can return a swing-low that is above the current
            # price (if the market has fallen since that pivot), which pushes SL
            # above entry — nonsensical and dangerous.  Fall back to ATR-based SL.
            if stop_loss >= price:
                stop_loss = price - atr * sl_mult

            # ── FIX #TP — Guaranteed T1 < T2 < T3 ordering ─────────────
            # Step 1: compute raw targets from price using vol-regime multipliers
            t1_raw = price + atr * t1_mult
            t2_raw = price + atr * t2_mult
            t3_raw = price + atr * t3_mult

            # Step 2: anchor T1 below nearest resistance (or use raw if no wall)
            if resistance > entry_high:
                t1 = min(t1_raw, resistance * 0.998)
                # Floor: T1 must always be above entry zone (tradeable signal)
                t1 = max(t1, entry_high * 1.001)
            else:
                t1 = t1_raw

            # Step 3+4: REALISTIC T2/T3 — proportional to the T1 move (so they
            # scale with structure), then hard-capped as a % of entry so
            # high-ATR coins don't get fantasy targets. Ordering is preserved.
            _d1      = abs(t1 - price)
            _t2_dist = min(max(_d1 * 1.7, atr * (t2_mult - t1_mult)), price * _T2_PCT_CAP)
            _t2_dist = max(_t2_dist, _d1 * 1.25)            # strict ordering T2 > T1
            _t3_dist = min(max(_d1 * 2.6, _t2_dist * 1.3),   price * _T3_PCT_CAP)
            _t3_dist = max(_t3_dist, _t2_dist * 1.25)        # strict ordering T3 > T2
            t2 = price + _t2_dist
            t3 = price + _t3_dist

        else:
            # ── FIX #TL — SHORT pullback-anchored entry zone ──────��─────
            # Mirror of the LONG fix: price fell to create the SHORT signal,
            # so entering at the close or below it is chasing the dump.
            # Set the zone to a realistic dead-cat bounce level:
            #   entry_low  = close + 0.05 ATR  (just above close — minimum bounce)
            #   entry_high = close + (entry_w_lo + entry_w_hi) ATR  (fuller bounce still valid)
            entry_low  = price + atr * 0.05
            entry_high = price + atr * (entry_w_lo + entry_w_hi)
            stop_loss  = min(resistance + atr * 0.3, price + atr * sl_mult)
            # Direction safety: SL must always be ABOVE price for a SHORT.
            if stop_loss <= price:
                stop_loss = price + atr * sl_mult

            # ── FIX #TP — Guaranteed T1 > T2 > T3 ordering (SHORT) ─────
            t1_raw = price - atr * t1_mult
            t2_raw = price - atr * t2_mult
            t3_raw = price - atr * t3_mult

            # Anchor T1 above nearest support
            if support < entry_low:
                t1 = max(t1_raw, support * 1.002)
                # Ceiling: T1 must always be below entry zone
                t1 = min(t1, entry_low * 0.999)
            else:
                t1 = t1_raw

            # REALISTIC T2/T3 (SHORT) — R-multiples of the T1 move, %-capped.
            _d1      = abs(price - t1)
            _t2_dist = min(max(_d1 * 1.7, atr * (t2_mult - t1_mult)), price * _T2_PCT_CAP)
            _t2_dist = max(_t2_dist, _d1 * 1.25)
            _t3_dist = min(max(_d1 * 2.6, _t2_dist * 1.3),   price * _T3_PCT_CAP)
            _t3_dist = max(_t3_dist, _t2_dist * 1.25)
            t2 = price - _t2_dist
            t3 = price - _t3_dist

        # ── DURATION ENGINE ─────────────────────────────────────────────
        # FIX #3 applied throughout — daily EMA/MACD from closed candle.
        # FIX #8 applied — counter-trend trades get shorter holds.

        # ── FIX #ATR-SL — ATR Stop Distance Validation ─────────────────
        # A stop that is less than 1.0× ATR from entry will be stopped out
        # on normal candle noise — the stop is not at a meaningful level.
        # If the computed stop is too tight, widen it to 1.2× ATR minimum.
        # Widen silently (don't fail — the signal can still be valid).
        _atr_sl_min_mult = 1.0   # stop must be at least 1.0× ATR from price
        _sl_distance = abs(price - stop_loss)
        if atr > 0 and _sl_distance < atr * _atr_sl_min_mult:
            _old_sl = stop_loss
            if bias == 'LONG':
                stop_loss = price - atr * _atr_sl_min_mult
                reasons.append(f"⚠️ SL widened to {_atr_sl_min_mult}× ATR — original was {(_sl_distance/atr):.2f}× ATR (noise-stop risk)")
            else:
                stop_loss = price + atr * _atr_sl_min_mult
                reasons.append(f"⚠️ SL widened to {_atr_sl_min_mult}× ATR — original was {(_sl_distance/atr):.2f}× ATR (noise-stop risk)")
            logger.debug("ATR-SL FIX: %s %s SL widened from %.6f to %.6f (ATR=%.6f)",
                         symbol, bias, _old_sl, stop_loss, atr)

        # ── SL ABSOLUTE-DISTANCE CAP ───────────────────────────────────────
        # A stop further than the vol-regime cap from entry is a wipeout at the
        # leverage the bot uses (e.g. -22% SL at L8 = liquidation). Pull it in
        # so the worst-case loss stays survivable. Keeps R:R honest too.
        _sl_cap_dist = price * _SL_PCT_CAP
        if _sl_cap_dist > 0 and abs(price - stop_loss) > _sl_cap_dist:
            _old_sl_cap = stop_loss
            if bias == 'LONG':
                stop_loss = price - _sl_cap_dist
            else:
                stop_loss = price + _sl_cap_dist
            reasons.append(
                f"⚠️ SL capped to {_SL_PCT_CAP*100:.1f}% of entry "
                f"(was {abs(price-_old_sl_cap)/price*100:.1f}%) — survivable at leverage")
            logger.debug("SL-CAP: %s %s SL tightened from %.6f to %.6f (cap %.1f%%)",
                         symbol, bias, _old_sl_cap, stop_loss, _SL_PCT_CAP*100)

        dur_score = 0
        dur_notes = []

        if bias == "LONG":
            tf_aligned = (price > ema20 > ema50) and (ema20_d > ema50_d)
        else:
            tf_aligned = (price < ema20 < ema50) and (ema20_d < ema50_d)

        if tf_aligned:
            dur_score += 3; dur_notes.append("Both 4H & Daily EMAs aligned — trend has legs")
        elif counter_trend:
            dur_score -= 2; dur_notes.append("Counter-trend — daily opposes, shorter hold recommended")
        else:
            dur_notes.append("EMA structure mixed — trend may be short-lived")

        # FIX #4 — confirmed crossovers only for duration scoring
        if macd_cross_bull_1d or macd_cross_bear_1d:
            dur_score += 3; dur_notes.append("Daily MACD confirmed cross [closed] — early in move, hold longer")
        elif macd_cross_bull_4h or macd_cross_bear_4h or macd_sust_bull_4h or macd_sust_bear_4h:
            dur_score += 2; dur_notes.append("4H MACD confirmed cross — move is fresh")
        elif (macd_diff > 0 and macd_d > 0) or (macd_diff < 0 and macd_d < 0):
            dur_score += 1; dur_notes.append("MACD momentum sustained both timeframes")
        else:
            dur_notes.append("MACD diverging across timeframes — move may be late")

        atr_pct = (atr / price) * 100
        if atr_pct >= 5.0:
            dur_score -= 2; dur_notes.append(f"ATR {atr_pct:.1f}% — very high volatility, exit quickly")
        elif atr_pct >= 3.0:
            dur_score -= 1; dur_notes.append(f"ATR {atr_pct:.1f}% — elevated volatility, targets hit faster")
        elif atr_pct >= 1.5:
            dur_score += 1; dur_notes.append(f"ATR {atr_pct:.1f}% — moderate volatility, medium hold suits")
        else:
            dur_score += 2; dur_notes.append(f"ATR {atr_pct:.1f}% — low volatility, needs more time to move")

        rsi_extreme = (rsi4 < 25 or rsi4 > 75) or (rsi_d < 30 or rsi_d > 70)
        if rsi_extreme:
            dur_score -= 1; dur_notes.append("RSI at extreme — snap-back likely, not a sustained trend")
        else:
            dur_score += 1; dur_notes.append("RSI in trend zone — sustainable directional momentum")

        vol_ratio = (vol / vol_ma) if (vol_ma and vol_ma > 0) else 1.0
        if vol_ratio > 2.5:
            dur_score -= 2; dur_notes.append(f"Volume {vol_ratio:.1f}x avg — possible climax candle, exit sooner")
        elif vol_ratio > 2.0:
            dur_score -= 1; dur_notes.append(f"Volume {vol_ratio:.1f}x avg — watch for exhaustion")
        elif vol_ratio > 1.3:
            dur_score += 1; dur_notes.append(f"Volume {vol_ratio:.1f}x avg — healthy participation")
        else:
            dur_notes.append("Volume below average — weak follow-through expected")

        if 40 <= rsi_d <= 60:
            dur_score += 2; dur_notes.append("Daily RSI in 40–60 range — trend has room to run [closed candle]")

        # Hard cap: very volatile coins cannot get multi-day holds
        if atr_pct >= 5.0 and dur_score > 5:
            dur_score = 5

        if dur_score >= 9:   hold_hours, hold = 72, "~3 days";   tf_note = f"Strong structural trend — hold ~3 days. {dur_notes[0]}"
        elif dur_score >= 7: hold_hours, hold = 48, "~2 days";   tf_note = f"Confirmed multi-TF trend — hold ~2 days. {dur_notes[0]}"
        elif dur_score >= 5: hold_hours, hold = 24, "~1 day";    tf_note = f"4H trend intact — hold ~24 hours. {dur_notes[0]}"
        elif dur_score >= 3: hold_hours, hold = 10, "~10 hours"; tf_note = f"Short-lived setup — exit within 10 hours. {dur_notes[0]}"
        elif dur_score >= 1: hold_hours, hold = 4,  "~4 hours";  tf_note = f"Quick move expected — target within 4 hours. {dur_notes[0]}"
        else:                hold_hours, hold = 1,  "~1 hour";   tf_note = f"Weak setup — scalp only. {dur_notes[0]}"

        # FIX #SE — Signal Expiry: cap hold_hours based on vol_regime so
        # high-volatility signals expire faster and don't stay open 3 days.
        # A 3-day EXTREME-vol signal is almost always stale within hours.
        _hold_caps = {'RANGING': 72, 'LOW': 48, 'MEDIUM': 24, 'HIGH': 12, 'EXTREME': 6}
        _cap = _hold_caps.get(vol_regime, 24)
        if hold_hours > _cap:
            hold_hours = _cap
            hold = f"~{_cap}h (vol-capped)"
            tf_note = tf_note + f" [hold capped at {_cap}h for {vol_regime} volatility]" 

        lev = calculate_leverage(price, entry_low, entry_high, stop_loss, atr, confidence, bias)

        # ── FIX #LS — Leverage-Safe Spread Cap ─────────────────────────
        # At high leverage a wide entry zone becomes dangerous: entering at
        # the bottom vs the top of the zone can shift liquidation price by
        # (spread% * leverage)%.  We cap the zone so that the spread never
        # exceeds (20 / suggested_leverage)% of price, with a floor of 0.4%
        # and ceiling of 3.0% (same as old fixed cap but now leverage-aware).
        if lev and lev['suggested'] and lev['suggested'] > 0:
            _sugg_lev = lev['suggested']
            _max_spread_pct = max(0.4, min(3.0, 20.0 / _sugg_lev)) / 100.0
            _max_spread_abs = price * _max_spread_pct
            _current_spread = abs(entry_high - entry_low)
            if _current_spread > _max_spread_abs and _max_spread_abs > 0:
                _entry_mid = (entry_low + entry_high) / 2
                entry_low  = _entry_mid - _max_spread_abs / 2
                entry_high = _entry_mid + _max_spread_abs / 2
                # Re-enforce direction safety after tightening
                if bias == 'LONG':
                    entry_low = max(entry_low, stop_loss * 1.001)
                else:
                    entry_high = min(entry_high, stop_loss * 0.999)
                # Recompute leverage from the tightened zone
                lev = calculate_leverage(price, entry_low, entry_high, stop_loss, atr, confidence, bias)

        # ── FIX #EZ — Entry zone freshness flag ────────────────────────
        # The entry zone is computed at scan time.  By the time a user
        # reads the signal, price may have already broken through the zone.
        # Tag the signal so format_signal can warn the user.
        #   entry_in_zone=True  → price still within entry_low..entry_high
        #   entry_in_zone=False → price has moved out of the zone
        entry_in_zone = (entry_low <= price <= entry_high)

        # ── FIX #FLIP — record this signal for cooldown tracking ───────
        state._last_signal_bias[symbol] = {'bias': bias, 'time': datetime.now()}
        db_save_signal_bias(symbol, bias)  # FIX #PERSIST-BIAS — write to DB so cooldown survives restarts

        # ── FIX #RR-GATE — enforce minimum entry R:R BEFORE the signal fires ──
        # All entry/SL/target adjustments (spread-tighten, ATR-SL widen,
        # resistance/support anchoring) are final by this point. Anchor R:R to
        # the worst realistic fill — the edge of the entry zone nearest price
        # (entry_high for LONG, entry_low for SHORT) — identical to _rr_ratio,
        # so a signal is gated on exactly the ratio the user sees on the card.
        _rr_entry = entry_high if bias == 'LONG' else entry_low
        if not confirm_rr(_rr_entry, stop_loss, t1, t2, t3, min_rr=_MIN_SIGNAL_RR):
            _rr_risk = abs(_rr_entry - stop_loss)
            _rr_dbg  = (abs(t1 - _rr_entry) / _rr_risk) if _rr_risk > 0 else 0.0
            logger.debug(
                "RR-GATE: %s %s rejected — entry->T1 R:R %.2f < %.2f "
                "(entry=%.6g sl=%.6g t1=%.6g)",
                symbol, bias, _rr_dbg, _MIN_SIGNAL_RR, _rr_entry, stop_loss, t1)
            return None

        result = {
            'symbol': symbol, 'bias': bias, 'confidence': confidence,
            'score': score, 'price': price,
            'entry_low': entry_low, 'entry_high': entry_high,
            'stop_loss': stop_loss,
            't1': t1, 't2': t2, 't3': t3,
            'hold': hold, 'tf_note': tf_note,
            'hold_hours': hold_hours, 'dur_score': dur_score, 'dur_reasons': dur_notes,
            'funding': funding * 100,
            'rsi4': rsi4, 'rsi_d': rsi_d,
            'stoch_k': stoch_k, 'atr': atr,
            'reasons': reasons, 'leverage': lev,
            'counter_trend': counter_trend,
            'btc_regime': btc_regime,
            'vol_regime': vol_regime,
            'entry_in_zone': entry_in_zone,
            # FIX #CQ — candle quality metadata (surfaced in signal cards + ML features)
            'candle_quality': cq_label,
            'candle_quality_score': round(cq_score, 3),
            # FIX #SESSION — session context at signal time
            'session':          _sess['session'],
            'session_note':     _sess['note'],
            'session_warning':  _sess['bias_warning'],
            # FIX #FIB — Fibonacci retracement levels (for signal card details)
            'fib_levels':       _fib_display_levels,
            'fib_confluence':   max(_fib_l_score, _fib_s_score),
            'scan_time': datetime.now(),
            # FIX #RI — store BTC price at scan time for regime invalidation
            'btc_price_at_scan': _get_btc_price_cached(),
            # CONVICTION LAYER — CVD / OI / VWAP payload (empty dict if disabled)
            'conviction': _conv,
            # PRIME — absolute evidence depth (winning side raw score). Used only
            # as a light tiebreaker in the Prime composite; no legacy gate reads it.
            'winning_score': winning,
            'losing_score': losing,
        }

        # ── ML SCORING — inject win-probability scores if models are loaded ──
        if _XGB_AVAILABLE:
            try:
                result['ml_score'] = xgb_predict_signal(result)
            except Exception as _ml_e:
                logger.debug("XGB predict failed for %s: %s", symbol, _ml_e)
        if _RF_AVAILABLE:
            try:
                result['rf_score'] = rf_predict_signal(result)
                rf_meta = rf_model_meta()
                if result.get('ml_score') is not None and result.get('rf_score') is not None:
                    consensus = (result['ml_score'] + result['rf_score']) / 2
                    result['consensus_score']   = consensus
                    result['consensus_verdict'] = (
                        "Strong edge" if consensus >= 0.70 else
                        "Moderate edge" if consensus >= 0.55 else
                        "Weak edge"
                    )
            except Exception as _ml_e:
                logger.debug("RF predict failed for %s: %s", symbol, _ml_e)

        # ── ML INFLUENCE — let win-probability actually shape the signal ──────
        # (1) NUDGE the headline confidence ±1 (applies to everyone, so /analyse
        #     confidence reflects the full analysis incl. ML).
        # (2) HIDE only the very-weakest signals from /scan & /autoscan (curated
        #     "best pairs"). User-requested pairs (/analyse, /cscan, /chart) are
        #     NEVER hidden. Untrained models return 0.5, so nothing changes until
        #     the models have learned from real outcomes.
        _edge = result.get('consensus_score')
        if _edge is None:
            _edge = result.get('ml_score')
        if _edge is None:
            _edge = result.get('rf_score')
        if _edge is not None:
            if   _edge >= _ML_STRONG_EDGE: _ml_nudge = +1
            elif _edge <= _ML_WEAK_EDGE:   _ml_nudge = -1
            else:                          _ml_nudge = 0
            if _ml_nudge:
                _old_c     = confidence
                confidence = max(1, min(10, confidence + _ml_nudge))
                result['ml_conf_adjust'] = confidence - _old_c
                if confidence != _old_c:
                    logger.debug("ML NUDGE: %s conf %d→%d (edge=%.2f)",
                                 symbol, _old_c, confidence, _edge)
            if (not user_requested) and _edge < _ML_HIDE_BELOW:
                logger.debug("ML HIDE: %s dropped from scan (edge=%.2f < %.2f)",
                             symbol, _edge, _ML_HIDE_BELOW)
                return None

        # Attach warning tags so display layer can rate the risk
        result['regime_warning']      = regime_warning    # set above if regime floor missed
        result['regime_blocked']      = regime_blocked    # HARD block: drop from autoscan/scan
        result['regime_block_detail'] = regime_block_detail
        result['low_conf_warning']    = low_conf_warning  # set above if conf < 4
        result['btc_regime']          = btc_regime
        result['confidence']          = confidence        # may have been ML-nudged / downgraded

        # ── PRIME — continuous 1-decimal confidence ───────────────────────────
        # Recovers the sub-integer margin that min(10, round(...)) throws away,
        # so a "strong 8" (raw 8.4) is distinguishable from a "weak 8" (raw 7.6).
        # We take the pre-round base (ratio_conf + quality_bonus) and re-apply the
        # NET integer delta that landed on `confidence` (candle demotion, BTC.D,
        # abs-score gate, ML nudge). This never double-counts: conviction is
        # already inside ratio_conf via the ig group.
        try:
            _base_precise = ratio_conf + quality_bonus
            _init_round   = min(10, round(_base_precise))
            _conf_precise = _base_precise + (confidence - _init_round)
            result['confidence_precise'] = round(max(0.0, min(10.0, _conf_precise)), 1)
        except Exception:
            result['confidence_precise'] = float(confidence)

        return result
    except Exception as e:
        logger.warning("score_pair error for %s: %s", symbol, e)
        return None


def _rr_ratio(r):
    """Compute R:R ratio for the trade the user ACTUALLY takes.

    FIX RR-ENTRY — risk/reward must be measured from the entry zone, not the
    signal candle close. FIX #TL deliberately offsets the entry zone away from
    `price` (a pullback below close for LONGs, a bounce above close for SHORTs),
    so measuring from `price` described a trade nobody enters and systematically
    understated the real R:R.

    We anchor to the *worst realistic fill* — the side of the entry zone closest
    to price (entry_high for LONG, entry_low for SHORT) — so the displayed R:R
    is conservative and never overstates. Falls back to `price` only if the
    entry-zone fields are missing (e.g. legacy/edge signal payloads).
    Returns float or None.
    """
    try:
        price     = r['price']
        stop_loss = r['stop_loss']
        t1        = r['t1']
        bias      = str(r.get('bias', 'LONG')).upper()
        # Worst realistic fill = the edge of the entry zone nearest to price.
        if bias == 'LONG':
            entry = r.get('entry_high', price)
        else:
            entry = r.get('entry_low', price)
        try:
            entry = float(entry)
        except (TypeError, ValueError):
            entry = price
        if not entry or entry <= 0:
            entry = price
        risk      = abs(entry - stop_loss)
        reward    = abs(t1 - entry)
        if risk > 0:
            return reward / risk
    except Exception:
        pass
    return None


# ---- Phase 0: display-floor helpers ----

def _conf_of(r) -> float:
    """Safe confidence extractor — returns 0.0 on any failure."""
    try:
        return float(r.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def passes_display_floor(r, floor: float = None) -> bool:
    """True when r's confidence meets the display floor."""
    if floor is None:
        floor = SCAN_DISPLAY_CONF_MIN
    return _conf_of(r) >= floor


def risk_reasons(r) -> list:
    """Data-driven reasons a signal is low quality.
    Returns a non-empty list — at minimum a generic caution line.
    Reused by the on-demand scan path and (Phase 1) the geometry validator.
    """
    reasons = []
    conf = _conf_of(r)
    if conf < SCAN_DISPLAY_CONF_MIN:
        reasons.append(f"Low confidence ({conf:.0f}/10, below {SCAN_DISPLAY_CONF_MIN:.0f})")
    try:
        rr = _rr_ratio(r)
        if rr is not None and rr < _MIN_SIGNAL_RR:
            reasons.append(f"Reward:risk {rr:.2f} is below the {_MIN_SIGNAL_RR:.1f} minimum")
    except Exception:
        pass
    try:
        entry = r.get("entry_low") or r.get("price")
        sl = r.get("stop_loss")
        if entry and sl:
            dist = abs(float(entry) - float(sl)) / float(entry) * 100
            if dist < 0.3:
                reasons.append(f"Stop very tight ({dist:.2f}%) — likely noise-stopped")
            elif dist > 12:
                reasons.append(f"Stop very wide ({dist:.1f}%) — outsized risk")
    except Exception:
        pass
    vol = r.get("quote_volume") or r.get("volume_usd") or r.get("volume")
    try:
        if vol is not None and float(vol) < 1_000_000:
            reasons.append("Thin 24h liquidity — slippage / manipulation risk")
    except (TypeError, ValueError):
        pass
    regime = (r.get("btc_regime") or "").lower()
    bias = (r.get("bias") or "").upper()
    if regime and bias:
        if bias == "LONG" and "bear" in regime:
            reasons.append("BTC regime is bearish — headwind for a LONG")
        elif bias == "SHORT" and "bull" in regime:
            reasons.append("BTC regime is bullish — headwind for a SHORT")
    if not reasons:
        reasons.append("Setup quality is marginal; size carefully")
    return reasons


