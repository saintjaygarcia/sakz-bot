"""sakz_explain.py — signal explanation layer (TODO section 4).

Pure formatting helpers for format_signal() in sakz_bot.py. They turn the score
buckets, BTC regime and reason lists that score_pair already computes into a
compact, human-readable explanation. No I/O.
"""
from __future__ import annotations

from typing import Mapping, Sequence

# Display order + labels for the score buckets score_pair produces.
# Keys are the bucket variable names; only present buckets are shown.
BUCKET_LABELS = [
    ("osc_g", "Osc"),
    ("mtf_g", "MTF"),
    ("cross_g", "MACD"),
    ("pos_g", "EMA"),
    ("vg", "Vol"),
    ("ig", "Struct"),
]

REGIME_EMOJI = {
    "STRONG_BULL": "\U0001F680",  # rocket
    "BULL": "\u26A1",            # high voltage
    "NEUTRAL": "\u2696\uFE0F",   # balance
    "BEAR": "\U0001F43B",        # bear
    "STRONG_BEAR": "\u2744\uFE0F",  # snowflake
}


def bucket_breakdown(scores: Mapping[str, float]) -> str:
    """Compact one-line bucket breakdown, e.g. 'Osc:3 MTF:2 MACD:2 EMA:1 Struct:2'.

    Only buckets present in `scores` are shown; zero values are kept (they are
    informative) but None/missing buckets are skipped.
    """
    parts = []
    for key, label in BUCKET_LABELS:
        if key in scores and scores[key] is not None:
            val = scores[key]
            num = int(val) if float(val).is_integer() else round(float(val), 1)
            parts.append(f"{label}:{num}")
    return " ".join(parts)


def regime_tag(btc_regime: str, counter_trend_active: bool = False) -> str:
    """One-line BTC regime context tag for the signal card.

    e.g. '\u26A1 BULL regime — counter-trend filter active'.
    """
    regime = (btc_regime or "NEUTRAL").upper()
    emoji = REGIME_EMOJI.get(regime, "\u2696\uFE0F")
    tag = f"{emoji} {regime.replace('_', ' ')} regime"
    if counter_trend_active:
        tag += " — counter-trend filter active"
    return tag


def dominant_reasons(
    reasons: Sequence,
    top_n: int = 2,
) -> list:
    """Pick the top `top_n` reasons by weight.

    Accepts either:
      * a sequence of (reason_text, weight) pairs, or
      * a plain sequence of reason strings (treated as equal weight, order kept).
    Returns a list of reason strings, de-duplicated, preserving best-first order.
    """
    if not reasons:
        return []
    weighted = []
    for item in reasons:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            weighted.append((str(item[0]), float(item[1])))
        else:
            weighted.append((str(item), 0.0))
    has_weights = any(w for _, w in weighted)
    if has_weights:
        weighted.sort(key=lambda kv: kv[1], reverse=True)
    seen = set()
    out = []
    for text, _ in weighted:
        if text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) >= top_n:
            break
    return out


def reason_summary(reasons: Sequence, top_n: int = 2, joiner: str = " + ") -> str:
    """Human 'why' string, e.g. 'RSI 4H oversold + MACD 4H crossover confirmed'."""
    return joiner.join(dominant_reasons(reasons, top_n=top_n))


def explanation_block(
    scores: Mapping[str, float],
    btc_regime: str,
    reasons: Sequence,
    counter_trend_active: bool = False,
    top_n: int = 2,
) -> str:
    """Assemble the full multi-line explanation appended to a signal card."""
    lines = []
    why = reason_summary(reasons, top_n=top_n)
    if why:
        lines.append(f"Why: {why}")
    bd = bucket_breakdown(scores)
    if bd:
        lines.append(f"Score: {bd}")
    lines.append(regime_tag(btc_regime, counter_trend_active))
    return "\n".join(lines)
