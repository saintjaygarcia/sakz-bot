"""
sakz_prime.py  ─  PRIME signal layer (pure, testable core)
===========================================================
The PRIME feature surfaces only the highest-conviction trades ("too good to be
true") to subscribers, on a per-user schedule, with a 15-minute live cache.

This module holds the *pure* logic only (no Telegram, no network, no DB) so it
can be unit-tested in isolation:

  • prime_score()            -> continuous 0-10 composite for ranking
  • entry_decay_multiplier() -> down-rank calls whose entry has already run
  • rank_prime()             -> top-N that clear the PRIME bar
  • scheduling helpers       -> per-user GMT alert-slot math
  • format helpers + constants (bar, top-N, disclaimer)

Design note (important, verified against sakz_scanner.py):
  Confidence ALREADY contains the CVD/OI/VWAP conviction layer (it is added to
  the `ig` group -> winning -> ratio_conf -> confidence). So the Prime composite
  does NOT re-add conviction (that would double-count). The backbone is the
  precise (1-decimal) confidence; ML edge and evidence depth are light
  tiebreakers; entry-distance is a decay multiplier applied last.
"""

import os
from datetime import timedelta

# ── Tunables (env-overridable) ────────────────────────────────────────────────
PRIME_BAR    = float(os.environ.get("PRIME_BAR", "8.5"))     # composite floor
PRIME_TOP_N  = int(os.environ.get("PRIME_TOP_N", "3"))        # picks per session
PRIME_CACHE_TTL_MIN = int(os.environ.get("PRIME_CACHE_TTL_MIN", "15"))

# Entry-distance decay shape (direction-aware, in % move past entry)
_DECAY_FREE_PCT = 0.3    # full credit until price has moved this far in-trade
_DECAY_FLOOR_PCT = 3.0   # at/after this, decay bottoms out
_DECAY_MIN_MULT = 0.5    # never drop a call below half its score on distance alone

PRIME_DISCLAIMER = (
    "\u26a0\ufe0f Not financial advice. PRIME calls are informational signals "
    "based on probability, never a guarantee. Markets are unpredictable \u2014 "
    "trade your own risk and size responsibly."
)

PRIME_EMPTY_MESSAGE = "Nothing here yet."


# ── Score components ───────────────────────────────────────────────────────────
def _precise_confidence(signal):
    """Backbone: 1-decimal confidence if present, else fall back to the int."""
    cp = signal.get("confidence_precise")
    if cp is None:
        cp = signal.get("confidence")
    try:
        return float(cp) if cp is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _ml_edge(signal):
    """Return ML win-probability in [0,1] or None. Prefers consensus, then xgb, rf."""
    for key in ("consensus_score", "ml_score", "rf_score"):
        v = signal.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _edge_adjust(signal):
    """ML edge tiebreaker: maps [0,1] (0.5 neutral) to +/- 0.4 confidence points."""
    edge = _ml_edge(signal)
    if edge is None:
        return 0.0
    return (edge - 0.5) * 0.8  # 1.0 -> +0.4, 0.0 -> -0.4


def _depth_adjust(signal):
    """Evidence-depth tiebreaker: reward winning score beyond the conf-8 floor (9),
    capped and with diminishing returns so correlated indicators can't run away."""
    depth = signal.get("winning_score")
    if depth is None:
        return 0.0
    try:
        depth = float(depth)
    except (TypeError, ValueError):
        return 0.0
    return min(max(depth - 9.0, 0.0) * 0.05, 0.3)  # up to +0.3


def entry_decay_multiplier(signal, current_price):
    """Return a multiplier in [_DECAY_MIN_MULT, 1.0].

    A call whose entry has already run in the trade direction is less
    actionable (you missed the entry), so its score decays. If price is at or
    better than entry, no decay. Direction-aware.
    """
    try:
        entry = float(signal.get("price") or 0)
        cur = float(current_price or 0)
    except (TypeError, ValueError):
        return 1.0
    if entry <= 0 or cur <= 0:
        return 1.0
    bias = str(signal.get("bias", "LONG")).upper()
    move_pct = ((cur - entry) / entry * 100.0) if bias == "LONG" else ((entry - cur) / entry * 100.0)
    if move_pct <= _DECAY_FREE_PCT:
        return 1.0  # entry still reachable / price better than entry
    if move_pct >= _DECAY_FLOOR_PCT:
        return _DECAY_MIN_MULT
    frac = (move_pct - _DECAY_FREE_PCT) / (_DECAY_FLOOR_PCT - _DECAY_FREE_PCT)
    return 1.0 - (1.0 - _DECAY_MIN_MULT) * frac


def prime_score(signal, current_price=None):
    """Continuous 0-10 Prime composite used for ranking.

    backbone (precise confidence, already includes conviction)
      + ML edge tiebreaker (+/-0.4)
      + evidence-depth tiebreaker (+0.3 max)
      , clamped to [0,10], then * entry-distance decay multiplier.
    """
    base = _precise_confidence(signal) + _edge_adjust(signal) + _depth_adjust(signal)
    base = max(0.0, min(10.0, base))
    cp = current_price if current_price is not None else signal.get("current_price")
    decay = entry_decay_multiplier(signal, cp) if cp else 1.0
    return round(base * decay, 2)


def passes_regime_gate(signal):
    """Prime never surfaces a hard regime-blocked signal."""
    return not bool(signal.get("regime_blocked"))


def rank_prime(signals, current_prices=None, bar=None, top_n=None):
    """Rank candidate signals and return the top-N that clear the bar.

    signals: iterable of signal dicts.
    current_prices: optional {symbol: price} for live decay.
    Returns a list of dicts: {signal, score, symbol} sorted desc by score.
    May be shorter than top_n (or empty -> caller shows PRIME_EMPTY_MESSAGE).
    """
    bar = PRIME_BAR if bar is None else bar
    top_n = PRIME_TOP_N if top_n is None else top_n
    current_prices = current_prices or {}
    scored = []
    for s in signals:
        if not passes_regime_gate(s):
            continue
        sym = s.get("symbol")
        cp = current_prices.get(sym) if sym else None
        score = prime_score(s, cp)
        if score >= bar:
            scored.append({"signal": s, "score": score, "symbol": sym})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_n]


# ── Per-user GMT alert scheduling (pure math) ──────────────────────────────────
def normalize_offset(gmt_offset):
    """Clamp a GMT offset (hours, may be fractional like 5.5) to [-12, 14]."""
    try:
        off = float(gmt_offset)
    except (TypeError, ValueError):
        return 0.0
    return max(-12.0, min(14.0, off))


def local_now(now_utc, gmt_offset):
    """Return the user's local datetime given a UTC time and GMT offset (hours)."""
    return now_utc + timedelta(hours=normalize_offset(gmt_offset))


def is_slot_due(now_utc, gmt_offset, slot_hour):
    """True when the user's LOCAL time is exactly slot_hour:00 (minute 0).

    The scheduler tick runs once per minute; this fires a single minute per slot.
    """
    loc = local_now(now_utc, gmt_offset)
    return loc.hour == int(slot_hour) and loc.minute == 0


def due_slots(now_utc, gmt_offset, slot_hours):
    """Return the subset of slot_hours that are due right now for this user."""
    return [h for h in (slot_hours or []) if is_slot_due(now_utc, gmt_offset, h)]


def slot_dedup_key(chat_id, now_utc, gmt_offset, slot_hour):
    """Stable key to ensure a user is alerted at most once per local slot per day."""
    loc = local_now(now_utc, gmt_offset)
    return f"{chat_id}:{loc.date().isoformat()}:{int(slot_hour)}"


# ── Display helpers ────────────────────────────────────────────────────────────
def format_offset(gmt_offset):
    """Render a GMT offset like 'GMT+1', 'GMT-5', 'GMT+5:30', 'GMT' (0)."""
    off = normalize_offset(gmt_offset)
    if off == 0:
        return "GMT"
    sign = "+" if off > 0 else "-"
    a = abs(off)
    hours = int(a)
    mins = int(round((a - hours) * 60))
    if mins:
        return f"GMT{sign}{hours}:{mins:02d}"
    return f"GMT{sign}{hours}"


def cache_is_fresh(last_refresh_utc, now_utc, ttl_min=None):
    """True if the cache was refreshed within the TTL window."""
    if last_refresh_utc is None:
        return False
    ttl = PRIME_CACHE_TTL_MIN if ttl_min is None else ttl_min
    return (now_utc - last_refresh_utc) < timedelta(minutes=ttl)
