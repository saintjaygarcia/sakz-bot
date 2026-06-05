"""sakz_trade_mgmt.py — active trade management (TODO section 3).

Pure helpers for managing an open trade after T1 is reached. No I/O; designed to
be called from the trade-tracking job in sakz_bot.py (the loop around line ~4450
that resolves signal_outcomes) and from sakz_paper.py.

Conventions:
  * bias is "LONG" or "SHORT".
  * stop levels move only in the protective direction (never loosen).
"""
from __future__ import annotations

LOW_VOL_REGIMES = ("LOW", "RANGING")


def move_sl_to_breakeven(entry: float, current_sl: float, bias: str, buffer: float = 0.0) -> float:
    """Once T1 is confirmed, pull the stop to entry (optionally + small buffer).

    For LONG the stop can only move up; for SHORT only down. Never loosens.
    `buffer` is a fraction of entry locked beyond breakeven (e.g. 0.001 = +0.1%).
    """
    bias = bias.upper()
    if bias == "LONG":
        be = entry * (1.0 + buffer)
        return max(current_sl, be)
    if bias == "SHORT":
        be = entry * (1.0 - buffer)
        return min(current_sl, be)
    return current_sl


def atr_trailing_stop(
    bias: str,
    current_price: float,
    atr: float,
    prev_sl: float,
    mult: float = 1.0,
) -> float:
    """Trail the stop by `mult` x ATR behind price; tighten-only.

    LONG  -> price - mult*ATR, but never below prev_sl.
    SHORT -> price + mult*ATR, but never above prev_sl.
    """
    bias = bias.upper()
    if atr <= 0:
        return prev_sl
    if bias == "LONG":
        candidate = current_price - mult * atr
        return max(prev_sl, candidate)
    if bias == "SHORT":
        candidate = current_price + mult * atr
        return min(prev_sl, candidate)
    return prev_sl


def cap_targets_for_regime(
    entry: float,
    atr: float,
    bias: str,
    t2: float,
    t3: float,
    vol_regime: str,
    t2_atr_cap: float = 2.0,
    t3_atr_cap: float = 3.5,
) -> tuple:
    """In LOW/RANGING volatility, pull unrealistically far T2/T3 closer to entry.

    Returns (t2_capped, t3_capped). In other regimes the inputs pass through.
    Caps are distances from entry: T2 <= t2_atr_cap*ATR, T3 <= t3_atr_cap*ATR.
    """
    bias = bias.upper()
    if vol_regime not in LOW_VOL_REGIMES or atr <= 0:
        return t2, t3
    if bias == "LONG":
        return min(t2, entry + t2_atr_cap * atr), min(t3, entry + t3_atr_cap * atr)
    if bias == "SHORT":
        return max(t2, entry - t2_atr_cap * atr), max(t3, entry - t3_atr_cap * atr)
    return t2, t3


def manage_after_t1(
    bias: str,
    entry: float,
    current_price: float,
    atr: float,
    current_sl: float,
    trail_mult: float = 1.0,
    breakeven_buffer: float = 0.0,
) -> float:
    """Combined post-T1 stop logic: breakeven first, then ATR trail; tighten-only.

    Returns the new protective stop. Calling this every candle after T1 locks in
    profit while leaving room for T2/T3 continuation.
    """
    sl = move_sl_to_breakeven(entry, current_sl, bias, buffer=breakeven_buffer)
    return atr_trailing_stop(bias, current_price, atr, sl, mult=trail_mult)
