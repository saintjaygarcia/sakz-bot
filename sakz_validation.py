"""sakz_validation.py — statistical validation for backtesting (TODO section 7).

Pure helpers to extend sakz_backtest_hist.py with walk-forward evaluation,
regime-separated reporting, and a Monte Carlo edge (permutation) test.
Dependencies: numpy only.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

# FIX #6 — single source of truth for outcome tuples (previously duplicated
# here and in sakz_risk.py, which risked silent drift). sakz_risk is pure
# stdlib so importing it here adds no heavy dependency.
from sakz_risk import WIN_OUTCOMES, LOSS_OUTCOMES


def walk_forward_splits(
    periods: Sequence,
    train_span: int = 1,
    test_span: int = 1,
    expanding: bool = False,
) -> list:
    """Rolling/anchored walk-forward splits over ordered periods (e.g. years).

    train 2022 -> test 2023 -> train 2023 -> test 2024 -> ...
    Returns a list of (train_periods, test_periods) tuples.
      * expanding=False: rolling window of `train_span` periods (default).
      * expanding=True : anchored window growing from the start.
    """
    uniq = sorted(set(periods))
    splits = []
    pos = train_span
    while pos + test_span <= len(uniq):
        train_start = 0 if expanding else pos - train_span
        train = uniq[train_start:pos]
        test = uniq[pos:pos + test_span]
        splits.append((train, test))
        pos += test_span
    return splits


def regime_split_report(
    trades: Sequence[dict],
    regime_key: str = "btc_regime",
    outcome_key: str = "outcome",
    pnl_key: str = "pnl",
) -> dict:
    """Group backtest trades by BTC regime and report win rate / pnl per regime.

    Surfaces whether the strategy only works in one regime (e.g. BULL).
    """
    groups = defaultdict(list)
    for t in trades:
        groups[str(t.get(regime_key, "UNKNOWN"))].append(t)

    report = {}
    for regime, ts in groups.items():
        wins = sum(1 for t in ts if t.get(outcome_key) in WIN_OUTCOMES)
        losses = sum(1 for t in ts if t.get(outcome_key) in LOSS_OUTCOMES)
        resolved = wins + losses
        pnl = sum(float(t.get(pnl_key, 0) or 0) for t in ts)
        report[regime] = {
            "trades": len(ts),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / resolved) if resolved else 0.0,
            "total_pnl": pnl,
        }
    return report


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Largest peak-to-trough drop of an equity curve, as a positive fraction."""
    arr = np.asarray(equity_curve, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(arr)
    # guard against zero/negative peaks
    safe_peak = np.where(running_peak == 0, np.nan, running_peak)
    dd = (running_peak - arr) / safe_peak
    dd = np.nan_to_num(dd, nan=0.0)
    return float(dd.max()) if dd.size else 0.0


def monte_carlo_edge_test(
    trade_returns: Sequence[float],
    n_iter: int = 10000,
    seed: int = 42,
) -> dict:
    """Permutation test for genuine edge.

    Each iteration randomly assigns a long/short (sign) to every trade's
    magnitude and totals the result, simulating a strategy with no directional
    skill. The p-value is the fraction of random runs whose total >= the real
    total. A small p-value (default edge threshold 0.05) means the real edge is
    unlikely to be luck.
    """
    arr = np.asarray(trade_returns, dtype=np.float64)
    if arr.size == 0:
        return {"real_total": 0.0, "p_value": 1.0, "has_edge": False, "n_iter": 0}
    real_total = float(arr.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_iter, arr.size))
    randomized = (signs * np.abs(arr)).sum(axis=1)
    p_value = float((randomized >= real_total).mean())
    return {
        "real_total": real_total,
        "p_value": p_value,
        "mean_random": float(randomized.mean()),
        "pct95_random": float(np.percentile(randomized, 95)),
        "n_iter": int(n_iter),
        "has_edge": p_value < 0.05,
    }
