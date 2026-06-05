"""sakz_correlation.py — cross-asset correlation engine (TODO section 8).

Pure numpy helpers, no I/O. Intended to be called once per scan cycle from
sakz_bot.py with the close-price arrays it already fetches.

Provides:
  * rolling return-correlation matrix across scanned symbols
  * high-correlation pair detection + BTC-move independence flag
  * ETH/BTC ratio altcoin risk regime
  * sector clustering tags + follow-on confidence damping
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _returns(closes: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(closes, dtype=np.float64)
    if arr.size < 2:
        return np.empty(0)
    rets = np.diff(arr) / arr[:-1]
    if window and rets.size > window:
        rets = rets[-window:]
    return rets


def rolling_correlation_matrix(
    price_frames: Mapping[str, Sequence[float]],
    window: int = 20,
) -> dict:
    """Pairwise Pearson correlation of returns over the last `window` bars.

    price_frames maps symbol -> close-price sequence (oldest..newest).
    Returns dict[(sym_a, sym_b)] -> correlation, for a<b (sorted), plus the
    diagonal as 1.0. Symbols with too little data are skipped.
    """
    series = {}
    for sym, closes in price_frames.items():
        r = _returns(closes, window)
        if r.size >= 2:
            series[sym] = r
    out = {}
    syms = sorted(series)
    for i, a in enumerate(syms):
        out[(a, a)] = 1.0
        for b in syms[i + 1:]:
            ra, rb = series[a], series[b]
            n = min(ra.size, rb.size)
            if n < 2:
                continue
            c = np.corrcoef(ra[-n:], rb[-n:])[0, 1]
            if np.isnan(c):
                c = 0.0
            out[(a, b)] = float(c)
    return out


def get_corr(matrix: Mapping[tuple, float], a: str, b: str) -> float:
    """Order-independent correlation lookup."""
    if a == b:
        return 1.0
    return float(matrix.get((a, b), matrix.get((b, a), 0.0)))


def high_correlation_pairs(matrix: Mapping[tuple, float], threshold: float = 0.85) -> list:
    """All (a,b) pairs (a!=b) with |correlation| >= threshold."""
    return [
        (a, b, c)
        for (a, b), c in matrix.items()
        if a != b and abs(c) >= threshold
    ]


def is_btc_dependent(
    symbol: str,
    matrix: Mapping[tuple, float],
    btc_symbol: str = "BTCUSDT",
    threshold: float = 0.85,
) -> bool:
    """True if a 'breakout' is really just a BTC move (corr >= threshold)."""
    return get_corr(matrix, symbol, btc_symbol) >= threshold


# ── ETH/BTC altcoin risk regime ──────────────────────────────────────
def eth_btc_ratio_regime(
    eth_closes: Sequence[float],
    btc_closes: Sequence[float],
    window: int = 20,
) -> str:
    """Altcoin risk tag from the ETH/BTC ratio trend.

    Falling ETH/BTC = risk-off for alts regardless of BTC's own direction.
    Returns 'ALT_RISK_OFF' | 'ALT_RISK_ON' | 'ALT_NEUTRAL'.
    """
    eth = np.asarray(eth_closes, dtype=np.float64)
    btc = np.asarray(btc_closes, dtype=np.float64)
    n = min(eth.size, btc.size)
    if n < window + 1:
        return "ALT_NEUTRAL"
    ratio = eth[-window:] / btc[-window:]
    sma = float(ratio.mean())
    last = float(ratio[-1])
    slope = float(ratio[-1] - ratio[0])
    if last < sma and slope < 0:
        return "ALT_RISK_OFF"
    if last > sma and slope > 0:
        return "ALT_RISK_ON"
    return "ALT_NEUTRAL"


# ── Sector clustering ──────────────────────────────────────────────
# Lightweight, extend freely. Symbols are matched on their base (strip USDT).
SECTOR_MAP = {
    "BTC": "L1", "ETH": "L1", "SOL": "L1", "AVAX": "L1", "ADA": "L1",
    "BNB": "L1", "NEAR": "L1", "APT": "L1", "SUI": "L1", "SEI": "L1",
    "ARB": "L2", "OP": "L2", "MATIC": "L2", "STRK": "L2", "MANTA": "L2",
    "UNI": "DeFi", "AAVE": "DeFi", "MKR": "DeFi", "CRV": "DeFi",
    "LDO": "DeFi", "SNX": "DeFi", "COMP": "DeFi", "DYDX": "DeFi",
    "DOGE": "meme", "SHIB": "meme", "PEPE": "meme", "WIF": "meme",
    "BONK": "meme", "FLOKI": "meme",
    "FET": "AI", "RNDR": "AI", "TAO": "AI", "AGIX": "AI", "WLD": "AI",
}


def base_symbol(symbol: str) -> str:
    s = symbol.upper().replace("_", "")
    for quote in ("USDT", "USDC", "USD", "PERP"):
        if s.endswith(quote):
            s = s[: -len(quote)]
            break
    return s


def sector_of(symbol: str) -> str:
    """Sector tag for a symbol, or 'other' if unknown."""
    return SECTOR_MAP.get(base_symbol(symbol), "other")


def follow_on_confidence(
    confidence: float,
    symbol: str,
    already_signalled: Sequence[str],
    damp: float = 0.8,
) -> float:
    """Damp confidence when a same-sector signal already fired this cycle.

    The first signal in a sector keeps full confidence; each follow-on in the
    same sector is multiplied by `damp` (default 0.8) to reflect correlation.
    """
    sec = sector_of(symbol)
    if sec == "other":
        return confidence
    prior_same = sum(1 for s in already_signalled if sector_of(s) == sec)
    if prior_same <= 0:
        return confidence
    return confidence * (damp ** prior_same)
