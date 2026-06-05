"""sakz_risk.py — portfolio & position-sizing helpers (TODO section 6).

Pure, dependency-free functions so they can be unit-tested and reused from
sakz_bot.py (calculate_leverage) and sakz_paper.py without touching the event
loop. Nothing here performs I/O.

Implements:
  * fractional Kelly position sizing (capped)
  * per-symbol win-rate / reward-risk estimation from signal_outcomes rows
  * correlation-aware combined-exposure cap
  * max-drawdown budget gate
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

WIN_OUTCOMES = ("t1_hit", "t2_hit", "t3_hit")
LOSS_OUTCOMES = ("sl_hit",)


# ── Kelly sizing ──────────────────────────────────────────────────────────
def full_kelly_fraction(win_rate: float, reward_risk: float) -> float:
    """Full-Kelly fraction f* = (b*p - q) / b.

    win_rate    p ∈ [0,1]
    reward_risk b = expected reward / risk (e.g. avg win / avg loss, or RR1).
    Returns 0.0 when the edge is non-positive or inputs are degenerate.
    """
    p = max(0.0, min(1.0, float(win_rate)))
    b = float(reward_risk)
    if b <= 0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return f if f > 0 else 0.0


def fractional_kelly(
    win_rate: float,
    reward_risk: float,
    fraction: float = 0.25,
    cap: float = 0.25,
) -> float:
    """Quarter-Kelly by default, hard-capped as a fraction of equity.

    fraction  multiplier on full Kelly (0.25 = quarter Kelly).
    cap       absolute ceiling on the returned equity fraction.
    """
    f = full_kelly_fraction(win_rate, reward_risk) * max(0.0, fraction)
    return max(0.0, min(f, cap))


def kelly_position_size(
    equity: float,
    win_rate: float,
    reward_risk: float,
    fraction: float = 0.25,
    cap: float = 0.25,
) -> float:
    """Notional to allocate = equity * fractional_kelly(...)."""
    if equity <= 0:
        return 0.0
    return equity * fractional_kelly(win_rate, reward_risk, fraction, cap)


def win_rate_from_outcomes(outcomes: Iterable[str]) -> float:
    """Win rate from a sequence of signal_outcomes.outcome strings.

    Wins = t1/t2/t3_hit, losses = sl_hit; unresolved values are ignored.
    Returns 0.0 if there are no resolved outcomes.
    """
    wins = losses = 0
    for o in outcomes:
        if o in WIN_OUTCOMES:
            wins += 1
        elif o in LOSS_OUTCOMES:
            losses += 1
    resolved = wins + losses
    return wins / resolved if resolved else 0.0


# ── Correlation-aware exposure cap ──────────────────────────────────────
def correlated_exposure_cap(
    candidate_symbol: str,
    open_symbols: Sequence[str],
    correlation: Mapping[tuple, float],
    base_size: float,
    corr_threshold: float = 0.85,
    max_cluster_size: int = 3,
) -> float:
    """Scale down a new position's size if it is highly correlated with open ones.

    correlation maps (symbol_a, symbol_b) -> rolling correlation in [-1,1]
    (order-independent lookups are handled here).

    If the candidate is >= corr_threshold correlated with N already-open symbols,
    once the correlated cluster would exceed max_cluster_size the size is divided
    by the cluster size so combined exposure to one driver stays bounded.
    """
    def corr(a: str, b: str) -> float:
        if a == b:
            return 1.0
        return float(correlation.get((a, b), correlation.get((b, a), 0.0)))

    correlated = [s for s in open_symbols if corr(candidate_symbol, s) >= corr_threshold]
    cluster = len(correlated) + 1  # include the candidate
    if cluster <= max_cluster_size:
        return base_size
    return base_size / cluster


# ── Drawdown budget gate ─────────────────────────────────────────────
def portfolio_drawdown(peak_equity: float, current_equity: float) -> float:
    """Fractional drawdown from peak, in [0,1]. 0 when at/above peak."""
    if peak_equity <= 0:
        return 0.0
    dd = (peak_equity - current_equity) / peak_equity
    return dd if dd > 0 else 0.0


@dataclass
class DrawdownGate:
    """Tracks peak equity and blocks new entries past a drawdown budget.

    Default budget 15% (matches the TODO). Call update(equity) each time live
    P&L changes; can_open() reports whether new positions are allowed.
    """
    threshold: float = 0.15
    peak_equity: float = 0.0

    def update(self, equity: float) -> None:
        if equity > self.peak_equity:
            self.peak_equity = equity

    def drawdown(self, equity: float) -> float:
        return portfolio_drawdown(self.peak_equity, equity)

    def can_open(self, equity: float) -> bool:
        # Update peak first so a fresh high never reads as a drawdown.
        self.update(equity)
        return self.drawdown(equity) < self.threshold
