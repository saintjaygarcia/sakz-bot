"""sakz_orderflow.py — order-flow upgrades (TODO section 5).

Pure detectors meant to run on data sakz_conviction.py already gathers (CVD,
volume) plus pivot levels from score_pair. No network here; the live
liquidation/CVD feeds are fetched elsewhere (see liquidation_cluster_risk note).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

NO_DIVERGENCE = "NONE"
BEARISH_DIVERGENCE = "BEARISH_DIVERGENCE"  # price up, CVD not -> hidden selling
BULLISH_DIVERGENCE = "BULLISH_DIVERGENCE"  # price down, CVD not -> hidden buying


def delta_divergence(closes: Sequence[float], cvd: Sequence[float], lookback: int = 10) -> str:
    """Detect price/CVD (cumulative volume delta) divergence.

    * Price makes a new high over `lookback` but CVD does not -> BEARISH (hidden
      selling into the high).
    * Price makes a new low but CVD does not -> BULLISH (hidden buying).
    Returns one of NONE / BEARISH_DIVERGENCE / BULLISH_DIVERGENCE.
    """
    c = np.asarray(closes, dtype=np.float64)
    d = np.asarray(cvd, dtype=np.float64)
    if c.size < lookback + 1 or d.size < lookback + 1:
        return NO_DIVERGENCE
    prior_c, last_c = c[-lookback - 1:-1], c[-1]
    prior_d, last_d = d[-lookback - 1:-1], d[-1]
    price_new_high = last_c > prior_c.max()
    price_new_low = last_c < prior_c.min()
    cvd_new_high = last_d > prior_d.max()
    cvd_new_low = last_d < prior_d.min()
    if price_new_high and not cvd_new_high:
        return BEARISH_DIVERGENCE
    if price_new_low and not cvd_new_low:
        return BULLISH_DIVERGENCE
    return NO_DIVERGENCE


def is_absorption(
    close_prev: float,
    close_now: float,
    volume: float,
    avg_volume: float,
    level: float,
    atr: float,
    vol_mult: float = 2.0,
    proximity_atr: float = 0.25,
    move_atr: float = 0.25,
) -> bool:
    """Large volume near a S/R level with minimal price movement = absorption.

    True when, near `level` (within proximity_atr*ATR), volume >= vol_mult*avg
    yet price moved less than move_atr*ATR. Caller can score +1 in the volume
    bucket (vg) when this fires near a pivot.
    """
    if atr <= 0 or avg_volume <= 0:
        return False
    near_level = abs(close_now - level) <= proximity_atr * atr
    heavy_volume = volume >= vol_mult * avg_volume
    small_move = abs(close_now - close_prev) <= move_atr * atr
    return bool(near_level and heavy_volume and small_move)


def absorption_score(
    close_prev: float,
    close_now: float,
    volume: float,
    avg_volume: float,
    levels: Sequence[float],
    atr: float,
    **kwargs,
) -> int:
    """+1 if absorption is detected near any provided pivot level, else 0."""
    for lvl in levels or []:
        if is_absorption(close_prev, close_now, volume, avg_volume, lvl, atr, **kwargs):
            return 1
    return 0


def liquidation_cluster_risk(
    sl_price: float,
    liquidation_levels: Sequence[float],
    atr: float,
    proximity_atr: float = 0.5,
    min_cluster: int = 3,
) -> bool:
    """Flag a signal-quality red flag when liquidation levels cluster near the SL.

    `liquidation_levels` is a list of price levels of recent large liquidations
    (fetched live from Bybit/Binance liquidation streams in sakz_ws.py /
    sakz_ccxt.py — not fetched here). Returns True when >= min_cluster levels sit
    within proximity_atr*ATR of the stop, i.e. the stop sits in a liquidation
    magnet zone.
    """
    if atr <= 0 or not liquidation_levels:
        return False
    near = [lvl for lvl in liquidation_levels if abs(lvl - sl_price) <= proximity_atr * atr]
    return len(near) >= min_cluster
