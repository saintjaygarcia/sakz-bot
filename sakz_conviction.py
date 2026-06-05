"""
sakz_conviction.py  ─  Conviction Layer Patch for sakz_bot
═══════════════════════════════════════════════════════════
Adds three new independent evidence sources to the scoring engine:

  • CVD (Cumulative Volume Delta)   — buy vs sell pressure from candle data
  • Open Interest direction         — rising/falling OI as conviction filter
  • VWAP position                   — institutional reference price bias

HOW TO INTEGRATE
────────────────
Step 1 — Import at the top of sakz_bot.py (after existing imports):

    from sakz_conviction import fetch_cvd_signal, fetch_oi_signal, fetch_vwap_signal, \
                                 conviction_scores, CONVICTION_DISPLAY

Step 2 — Hook into score_pair(), just before the final `result = { ... }` dict
         (search for the comment "── FIX #FLIP — record this signal"):

    # ── CONVICTION LAYER — CVD / OI / VWAP ──────────────────────────────
    _conv = conviction_scores(symbol, df4h, bias, BINANCE_AVAILABLE)
    ig_l += _conv['ig_long_bonus']
    ig_s += _conv['ig_short_bonus']
    # Re-sum after adding conviction points (before GAP/CONF gates already passed,
    # so this only affects result enrichment — not blocking logic)

Step 3 — Add conviction data to the result dict (inside the result = { } block):

    'conviction':        _conv,

Step 4 — Surface in format_signal() (optional but recommended).
         Search for where 'funding' is displayed and add below it:

    conv = sig.get('conviction', {})
    if conv.get('display_lines'):
        lines.append("📊 CONVICTION:")
        for dl in conv['display_lines']:
            lines.append(f"   {dl}")

═══════════════════════════════════════════════════════════════════════════════

SCORING CONTRIBUTION  (goes into ig — the uncapped independent group)
──────────────────────────────────────────────────────────────────────
  CVD   : +2 (strong alignment) / +1 (mild) per direction
  OI    : +2 (rising OI confirms bias) / −1 (falling OI, soft veto)
  VWAP  : +1 (price on correct side of VWAP for the bias)
  Max additive:  +5 (strong CVD + rising OI + VWAP aligned)
  Max subtractive: −1 (falling OI against bias — soft, not a hard block)

All three sources are genuinely uncorrelated:
  CVD   ← per-candle buy/sell volume split (different from RSI/MACD/EMA)
  OI    ← total open contracts (positional data, not price-derived)
  VWAP  ← volume-weighted average price (different from SMA/EMA weighting)
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept':     'application/json',
}

# ── Module-level caches ───────────────────────────────────────────────────────
# Keyed by symbol. TTL avoids redundant API calls within the same scan cycle.
_cvd_cache: dict  = {}   # { symbol: { 'value': float, 'time': datetime } }
_oi_cache:  dict  = {}   # { symbol: { 'direction': str, 'pct': float, 'time': datetime } }
_CVD_TTL  = 300          # 5 min
_OI_TTL   = 300          # 5 min


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CVD — Cumulative Volume Delta
# ═══════════════════════════════════════════════════════════════════════════════
def compute_cvd_from_ohlcv(df: pd.DataFrame, window: int = 20) -> float:
    """
    Approximate CVD from OHLCV candles.

    True CVD requires tick data (which is expensive). The free approximation:
      buy_vol  ≈ volume × close_position_in_range   where cp = (close−low)/(high−low)
      sell_vol ≈ volume × (1 − close_position)

    delta_per_bar = buy_vol − sell_vol  =  volume × (2*cp − 1)
                  = volume × (2*close − high − low) / (high − low)

    This is identical to the CLV formula already in add_indicators(), just
    accumulated as a running sum rather than a moving average. We look at the
    rolling sum over the last `window` bars to capture recent momentum.

    Returns a float:
      > 0  buying pressure dominated over the window
      < 0  selling pressure dominated
      0    neutral / zero-range candles
    """
    try:
        df  = df.tail(window + 5).copy()
        hl  = (df['high'] - df['low']).replace(0, float('nan'))
        cp  = (2 * df['close'] - df['high'] - df['low']) / hl
        cp  = cp.fillna(0).clip(-1, 1)
        delta = df['volume'] * cp
        return float(delta.tail(window).sum())
    except Exception as e:
        logger.debug("compute_cvd_from_ohlcv error: %s", e)
        return 0.0


def fetch_cvd_signal(symbol: str, df4h: pd.DataFrame) -> dict:
    """
    Derive CVD signal from the df4h OHLCV data already in memory.
    No extra API call needed — uses the same candles score_pair has.

    Returns:
        {
          'cvd_value':    float,   raw cumulative delta (last 20 bars)
          'cvd_pct':      float,   cvd as % of total volume (normalised)
          'signal':       str,     'BULL' | 'BEAR' | 'NEUTRAL'
          'strength':     str,     'STRONG' | 'MILD' | 'WEAK'
          'display':      str,     one-line human label
        }
    """
    cached = _cvd_cache.get(symbol)
    if cached and (datetime.now() - cached['time']).total_seconds() < _CVD_TTL:
        return cached['data']

    result = {'cvd_value': 0.0, 'cvd_pct': 0.0, 'signal': 'NEUTRAL',
              'strength': 'WEAK', 'display': 'CVD: neutral'}
    try:
        window  = 20
        cvd_val = compute_cvd_from_ohlcv(df4h, window=window)

        # Normalise by total volume in the window to get a percentage
        total_vol = float(df4h['volume'].tail(window).sum())
        cvd_pct   = (cvd_val / total_vol * 100) if total_vol > 0 else 0.0

        # Classify
        if cvd_pct > 15:
            signal, strength = 'BULL', 'STRONG'
        elif cvd_pct > 5:
            signal, strength = 'BULL', 'MILD'
        elif cvd_pct < -15:
            signal, strength = 'BEAR', 'STRONG'
        elif cvd_pct < -5:
            signal, strength = 'BEAR', 'MILD'
        else:
            signal, strength = 'NEUTRAL', 'WEAK'

        arrow = '📈' if signal == 'BULL' else ('📉' if signal == 'BEAR' else '➡️')
        display = f"CVD: {arrow} {signal} ({strength}) | delta {cvd_pct:+.1f}% of vol"

        result = {
            'cvd_value': round(cvd_val, 4),
            'cvd_pct':   round(cvd_pct,  2),
            'signal':    signal,
            'strength':  strength,
            'display':   display,
        }
    except Exception as e:
        logger.debug("fetch_cvd_signal %s: %s", symbol, e)

    _cvd_cache[symbol] = {'data': result, 'time': datetime.now()}
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  OPEN INTEREST — Binance Futures (free public endpoint)
# ═══════════════════════════════════════════════════════════════════════════════
def _binance_oi_hist(symbol: str, period: str = '4h', limit: int = 5) -> list:
    """
    Fetch OI history from Binance /futures/data/openInterestHist.
    Returns list of {'sumOpenInterest': str, 'timestamp': int} dicts, or [].
    Period options: '5m','15m','30m','1h','2h','4h','6h','12h','1d'
    """
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={'symbol': symbol, 'period': period, 'limit': limit},
            headers=HEADERS, timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                return data
        return []
    except Exception as e:
        logger.debug("_binance_oi_hist %s: %s", symbol, e)
        return []


def _bybit_oi(symbol: str) -> dict:
    """
    Fallback: Bybit /v5/market/open-interest (recent + older reading).
    Returns {'current': float, 'prior': float} or {}.
    """
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/open-interest",
            params={'category': 'linear', 'symbol': symbol,
                    'intervalTime': '4h', 'limit': 3},
            headers=HEADERS, timeout=10
        )
        if r.status_code != 200:
            return {}
        rows = r.json().get('result', {}).get('list', [])
        if len(rows) < 2:
            return {}
        # rows[0] = most recent, rows[-1] = oldest in the small window
        current = float(rows[0].get('openInterest', 0) or 0)
        prior   = float(rows[-1].get('openInterest', 0) or 0)
        return {'current': current, 'prior': prior}
    except Exception as e:
        logger.debug("_bybit_oi %s: %s", symbol, e)
        return {}


def fetch_oi_signal(symbol: str, binance_available: bool = True) -> dict:
    """
    Fetch Open Interest direction for a symbol.
    Tries Binance first (more granular), falls back to Bybit.

    Returns:
        {
          'current':    float,   latest OI value
          'prior':      float,   OI 4h ago (or 2 readings back)
          'pct_change': float,   % change between prior → current
          'direction':  str,     'RISING' | 'FALLING' | 'FLAT'
          'display':    str,     one-line human label
        }
    """
    cached = _oi_cache.get(symbol)
    if cached and (datetime.now() - cached['time']).total_seconds() < _OI_TTL:
        return cached['data']

    result = {'current': 0.0, 'prior': 0.0, 'pct_change': 0.0,
              'direction': 'FLAT', 'display': 'OI: data unavailable'}

    try:
        current, prior = 0.0, 0.0

        # ── Binance path ──────────────────────────────────────────────────
        if binance_available:
            rows = _binance_oi_hist(symbol, period='4h', limit=3)
            if len(rows) >= 2:
                # rows are oldest→newest from this endpoint
                current = float(rows[-1].get('sumOpenInterest', 0) or 0)
                prior   = float(rows[0].get('sumOpenInterest', 0) or 0)

        # ── Bybit fallback ────────────────────────────────────────────────
        if current == 0.0:
            bybit_data = _bybit_oi(symbol)
            if bybit_data:
                current = bybit_data.get('current', 0.0)
                prior   = bybit_data.get('prior',   0.0)

        if current == 0.0 or prior == 0.0:
            _oi_cache[symbol] = {'data': result, 'time': datetime.now()}
            return result

        pct = ((current - prior) / prior * 100) if prior > 0 else 0.0

        if pct > 2.0:
            direction = 'RISING'
        elif pct < -2.0:
            direction = 'FALLING'
        else:
            direction = 'FLAT'

        arrow = '🔺' if direction == 'RISING' else ('🔻' if direction == 'FALLING' else '➡️')
        display = f"OI: {arrow} {direction} ({pct:+.1f}%)"

        result = {
            'current':    round(current, 2),
            'prior':      round(prior,   2),
            'pct_change': round(pct,     2),
            'direction':  direction,
            'display':    display,
        }
    except Exception as e:
        logger.debug("fetch_oi_signal %s: %s", symbol, e)

    _oi_cache[symbol] = {'data': result, 'time': datetime.now()}
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  VWAP — Intraday institutional reference price
# ═══════════════════════════════════════════════════════════════════════════════
def compute_vwap(df: pd.DataFrame) -> float:
    """
    Standard VWAP: cumulative (typical_price × volume) / cumulative volume.
    typical_price = (high + low + close) / 3.

    Uses the last 24 bars of the provided dataframe (i.e. 24×4H = 1 trading day
    on the 4H chart — a sensible intraday VWAP window for crypto).

    Returns the VWAP price, or 0.0 on failure.
    """
    try:
        df_w  = df.tail(24).copy()
        tp    = (df_w['high'] + df_w['low'] + df_w['close']) / 3
        vwap  = (tp * df_w['volume']).cumsum() / df_w['volume'].cumsum()
        return float(vwap.iloc[-1])
    except Exception as e:
        logger.debug("compute_vwap error: %s", e)
        return 0.0


def fetch_vwap_signal(symbol: str, df4h: pd.DataFrame, price: float) -> dict:
    """
    Determine whether current price is above or below VWAP.

    Returns:
        {
          'vwap':       float,
          'position':   str,    'ABOVE' | 'BELOW' | 'AT'
          'pct_diff':   float,  (price − vwap) / vwap × 100
          'display':    str,
        }
    """
    result = {'vwap': 0.0, 'position': 'AT', 'pct_diff': 0.0,
              'display': 'VWAP: unavailable'}
    try:
        vwap = compute_vwap(df4h)
        if vwap <= 0:
            return result

        pct_diff = (price - vwap) / vwap * 100

        if pct_diff > 0.3:
            position = 'ABOVE'
            arrow    = '🟢'
        elif pct_diff < -0.3:
            position = 'BELOW'
            arrow    = '🔴'
        else:
            position = 'AT'
            arrow    = '🟡'

        display = f"VWAP: {arrow} {position} (${vwap:.4f}, diff {pct_diff:+.2f}%)"

        result = {
            'vwap':     round(vwap,     6),
            'position': position,
            'pct_diff': round(pct_diff, 3),
            'display':  display,
        }
    except Exception as e:
        logger.debug("fetch_vwap_signal %s: %s", symbol, e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MASTER SCORER — feeds directly into score_pair ig bucket
# ═══════════════════════════════════════════════════════════════════════════════
def conviction_scores(
    symbol: str,
    df4h:   pd.DataFrame,
    bias:   str,               # 'LONG' or 'SHORT'
    binance_available: bool = True,
) -> dict:
    """
    Compute CVD / OI / VWAP conviction signals and translate them into
    ig_long_bonus and ig_short_bonus point adjustments for score_pair.

    Called once per symbol inside score_pair AFTER the bias is determined.
    The bias parameter lets OI and VWAP give directional bonuses.

    Scoring rules:
    ─────────────────────────────────────────────────────────────────────
    CVD:
      BULL + STRONG  → +2 to ig_long, +0 to ig_short
      BULL + MILD    → +1 to ig_long
      BEAR + STRONG  → +2 to ig_short, +0 to ig_long
      BEAR + MILD    → +1 to ig_short
      NEUTRAL        → no change

    OI:
      RISING  → +2 to the signal direction's ig (conviction behind the move)
      FALLING → −1 to the signal direction's ig (money leaving the trade)
      FLAT    → no change

    VWAP:
      LONG  bias and price ABOVE VWAP → +1 to ig_long
      SHORT bias and price BELOW VWAP → +1 to ig_short
      Opposing alignment              → no change (never penalises)

    Returns dict with all three signal payloads + ig adjustments + display lines.
    """
    price = float(df4h['close'].iloc[-1])

    cvd  = fetch_cvd_signal(symbol, df4h)
    oi   = fetch_oi_signal(symbol, binance_available)
    vwap = fetch_vwap_signal(symbol, df4h, price)

    ig_long_bonus  = 0.0
    ig_short_bonus = 0.0
    notes_l, notes_s = [], []

    # ── CVD contribution ─────────────────────────────────────────────────
    cvd_sig = cvd['signal']
    cvd_str = cvd['strength']
    if cvd_sig == 'BULL':
        pts = 2 if cvd_str == 'STRONG' else 1
        ig_long_bonus += pts
        notes_l.append(f"✅ {cvd['display']}")
    elif cvd_sig == 'BEAR':
        pts = 2 if cvd_str == 'STRONG' else 1
        ig_short_bonus += pts
        notes_s.append(f"✅ {cvd['display']}")
    # NEUTRAL → no notes, no points

    # ── OI contribution (directional — only credited to the bias direction) ──
    oi_dir = oi['direction']
    if oi_dir == 'RISING':
        if bias == 'LONG':
            ig_long_bonus  += 2
            notes_l.append(f"✅ {oi['display']} — rising OI confirms LONG conviction")
        else:
            ig_short_bonus += 2
            notes_s.append(f"✅ {oi['display']} — rising OI confirms SHORT conviction")
    elif oi_dir == 'FALLING':
        if bias == 'LONG':
            ig_long_bonus  = max(0, ig_long_bonus  - 1)
            notes_l.append(f"⚠️ {oi['display']} — money leaving, weakens LONG conviction")
        else:
            ig_short_bonus = max(0, ig_short_bonus - 1)
            notes_s.append(f"⚠️ {oi['display']} — money leaving, weakens SHORT conviction")
    # FLAT → no notes, no points

    # ── VWAP contribution ────────────────────────────────────────────────
    vwap_pos = vwap['position']
    if bias == 'LONG' and vwap_pos == 'ABOVE':
        ig_long_bonus += 1
        notes_l.append(f"✅ {vwap['display']} — price above VWAP, institutional bias bullish")
    elif bias == 'SHORT' and vwap_pos == 'BELOW':
        ig_short_bonus += 1
        notes_s.append(f"✅ {vwap['display']} — price below VWAP, institutional bias bearish")
    elif vwap_pos != 'AT':
        # Opposing (LONG but below VWAP, or SHORT but above VWAP) — surface as note only
        if bias == 'LONG':
            notes_l.append(f"ℹ️ {vwap['display']} — below VWAP, watch for institutional pressure")
        else:
            notes_s.append(f"ℹ️ {vwap['display']} — above VWAP, watch for institutional pressure")

    # Pick the display lines for the active bias direction
    display_lines = notes_l if bias == 'LONG' else notes_s

    return {
        'ig_long_bonus':   ig_long_bonus,
        'ig_short_bonus':  ig_short_bonus,
        'cvd':             cvd,
        'oi':              oi,
        'vwap':            vwap,
        'display_lines':   display_lines,   # for format_signal
    }


# ── Convenience constant for integration check ────────────────────────────────
CONVICTION_DISPLAY = "sakz_conviction v1.0 loaded"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION REFERENCE  (copy-paste snippets)
# ═══════════════════════════════════════════════════════════════════════════════
_INTEGRATION_GUIDE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: Import (top of sakz_bot.py, after existing imports)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from sakz_conviction import conviction_scores, CONVICTION_DISPLAY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: Hook in score_pair() — paste AFTER the bias is decided
        (right before the ig bucket caps and final sum)
        Search: "── FIX #SESSION — Session awareness"
        Paste ABOVE the line "ig_l = max(0.0, ig_l + _sess_mod)"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ── CONVICTION LAYER ───────────────────────────────────────────
        _conv = conviction_scores(symbol, df4h, bias,
                                  BINANCE_AVAILABLE if BINANCE_AVAILABLE is not None else False)
        ig_l += _conv['ig_long_bonus']
        ig_s += _conv['ig_short_bonus']
        # Append conviction notes into the relevant reason list
        for _cn in _conv.get('display_lines', []):
            if bias == 'LONG':
                lr.append(_cn)
            else:
                sr.append(_cn)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: Add to result dict in score_pair() (inside result = { } block)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        'conviction':  _conv,

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4: Display in format_signal() (optional but recommended)
        Find where funding is displayed, add below it:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        conv = sig.get('conviction', {})
        if conv.get('display_lines'):
            lines.append("")
            lines.append("📊 CONVICTION LAYER:")
            for dl in conv['display_lines']:
                lines.append(f"   {dl}")
"""


if __name__ == "__main__":
    # Quick smoke-test — does not require a running bot
    import sys
    import numpy as np

    print(CONVICTION_DISPLAY)
    print("\nRunning smoke test with synthetic OHLCV...\n")

    # Generate synthetic rising candles with strong buy-side delta
    n   = 60
    rng = np.random.default_rng(42)
    closes = np.cumsum(rng.normal(0.002, 0.01, n)) + 100
    highs  = closes + rng.uniform(0.001, 0.005, n)
    lows   = closes - rng.uniform(0.001, 0.005, n)
    vols   = rng.uniform(1000, 5000, n)
    df_test = pd.DataFrame({
        'open':   closes - rng.uniform(0, 0.002, n),
        'high':   highs,
        'low':    lows,
        'close':  closes,
        'volume': vols,
    })

    price  = float(closes[-1])
    symbol = "TESTUSDT"

    cvd  = fetch_cvd_signal(symbol, df_test)
    vwap = fetch_vwap_signal(symbol, df_test, price)

    print(f"CVD:  {cvd['display']}")
    print(f"VWAP: {vwap['display']}")
    print(f"OI:   (skipped — requires live API)")
    print(f"\nConviction scores (LONG bias):")
    conv = conviction_scores(symbol, df_test, 'LONG', binance_available=False)
    print(f"  ig_long_bonus  = {conv['ig_long_bonus']}")
    print(f"  ig_short_bonus = {conv['ig_short_bonus']}")
    for dl in conv['display_lines']:
        print(f"  {dl}")

    print("\n" + "─" * 60)
    print("INTEGRATION GUIDE:")
    print(_INTEGRATION_GUIDE)
