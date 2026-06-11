"""
sakz_signal_logic.py — pure, side-effect-free signal lifecycle logic.

This module is intentionally free of any Telegram / network / DB imports so it
can be unit-tested in isolation (see test_signal_behavior.py). sakz_bot.py and
sakz_db.py call into these helpers; all I/O (live price, DB rows) is passed in
as plain values so the rules stay deterministic and testable.

Behaviours implemented here (mirrors the product spec):

  1. Autoscan dedup ("always" subscribers):
       Do NOT re-push the same call (same exchange+symbol+bias+timeframe)
       consecutively UNLESS the coin has gained a HIGHER confidence than the
       last time it was pushed. Genuinely new/different calls still flow.

  2. Per-user timeframe cadence:
       A subscriber who picked a specific timeframe is only alerted on that
       timeframe's cadence (e.g. 4h => at most once per 4h), so they are
       notified "when they want". "Always" subscribers are not throttled by
       cadence (only by the confidence dedup above).

  3. /pnl resolves to the OLDEST (first) recorded signal for a pair.

  4. If a pair moved counter-direction and hit its stop-loss, /pnl renders a
       LOSS card pinned to the bot's stop-loss at the bot's suggested leverage.

  5. While a coin keeps moving in the profiting direction (even if re-called),
       the FIRST signal is the one that stays recorded.

  6. Signals that are dormant / not in motion for 15 minutes are auto-cleared.

  7. A days-old winning trade that reverses >= 20% from its peak price is
       evicted from memory. A later /pnl must verify the pair was scanned AFTER
       it was removed, otherwise the bot reports it "wasn't scanned recently".
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# How many seconds each timeframe "cadence" lasts. A subscriber who picked a
# timeframe is alerted at most once per this window.
TF_CADENCE_SECS: Dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
}

# Dormancy window: a signal that shows no favourable motion for this long is
# auto-cleared.
DORMANT_SECS = 15 * 60  # 15 minutes

# A favourable move smaller than this (in %) is treated as "no motion".
MOTION_EPS_PCT = 0.05

# Peak-reversal eviction threshold: a winning trade that retraces this fraction
# of its peak price is deleted from memory.
PEAK_REVERSAL_FRAC = 0.20  # 20%


def _as_dt(value: Any) -> Optional[datetime]:
    """Coerce a datetime | ISO-string | None into a naive datetime (or None)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    try:
        return datetime.fromisoformat(str(value)).replace(tzinfo=None)
    except Exception:
        return None


def dedup_key(exchange: str, symbol: str, bias: str, timeframe: str) -> str:
    """Stable identity for a single 'call' (one direction on one timeframe)."""
    return f"{exchange}_{symbol}_{bias}_{timeframe}".upper()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Autoscan dedup — confidence-aware
# ──────────────────────────────────────────────────────────────────────────────

def autoscan_decide_send(
    prev: Optional[Dict[str, Any]],
    confidence: float,
    now: Optional[datetime] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Decide whether to push a call to an "always" subscriber.

    Rules:
      • Never pushed before               -> SEND.
      • Confidence strictly higher than    -> SEND (coin gained confidence).
        the last pushed confidence
      • Same or lower confidence           -> SUPPRESS (don't repeat the call).

    Args:
      prev: the last record for this dedup key, shape {"confidence": float,
            "sent_at": datetime|iso}, or None if never sent.
      confidence: the current call's confidence.
      now: timestamp to stamp the record with (defaults to datetime.now()).

    Returns:
      (should_send, new_record). new_record should be persisted ONLY when
      should_send is True (so a suppressed lower-confidence re-detect never
      lowers the high-water mark).
    """
    now = now or datetime.now()
    conf = float(confidence or 0)

    if not prev:
        return True, {"confidence": conf, "sent_at": now}

    prev_conf = float(prev.get("confidence", 0) or 0)
    if conf > prev_conf:
        return True, {"confidence": conf, "sent_at": now}

    # Same call, not stronger -> do not repeat consecutively.
    return False, dict(prev)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Per-user timeframe cadence gating
# ──────────────────────────────────────────────────────────────────────────────

def timeframe_cadence_secs(tf_pref: Optional[str]) -> int:
    """Seconds between allowed alerts for a timeframe preference (0 = no gate)."""
    if not tf_pref:
        return 0  # "always" — no cadence gate
    return TF_CADENCE_SECS.get(str(tf_pref).lower(), 0)


def timeframe_due(
    last_notify: Any,
    tf_pref: Optional[str],
    now: Optional[datetime] = None,
) -> bool:
    """Return True if a timeframe subscriber is due for a new alert.

    • "always" (tf_pref falsy) -> always due; the confidence dedup is the only
      gate for those users.
    • A specific timeframe -> due only once its cadence window has elapsed since
      the last alert, so the user is alerted on the cadence they picked.
    """
    now = now or datetime.now()
    secs = timeframe_cadence_secs(tf_pref)
    if secs <= 0:
        return True
    last = _as_dt(last_notify)
    if last is None:
        return True
    return (now - last).total_seconds() >= secs


# ──────────────────────────────────────────────────────────────────────────────
# 3. Oldest-signal resolution for /pnl
# ──────────────────────────────────────────────────────────────────────────────

def _scan_dt(sig: Dict[str, Any]) -> datetime:
    dt = _as_dt(sig.get("scan_time") or sig.get("first_scan_time"))
    return dt if dt is not None else datetime.max


def pick_oldest_signal(signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the earliest-scanned signal (the original/first call)."""
    if not signals:
        return None
    return min(signals, key=_scan_dt)


def collapse_to_oldest_per_direction(
    signals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse many re-detected rows to ONE per direction, keeping the oldest.

    Used when /pnl finds several historical rows for a symbol — we anchor each
    direction to its first (oldest) call and return them newest-first-call last.
    """
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in signals or []:
        gk = (
            str(r.get("exchange", "")).upper(),
            str(r.get("symbol", "")).upper().replace("/", "").replace("_", ""),
            str(r.get("bias", "")).upper(),
        )
        keep = grouped.get(gk)
        if keep is None or _scan_dt(r) < _scan_dt(keep):
            grouped[gk] = r
    out = list(grouped.values())
    out.sort(key=_scan_dt)  # oldest first
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4. PnL card mode (loss-at-SL vs normal) + 7. eviction guard
# ──────────────────────────────────────────────────────────────────────────────

def pnl_card_mode(
    record: Optional[Dict[str, Any]],
    latest_scan_time: Any = None,
    eviction_removed_at: Any = None,
) -> str:
    """Decide how /pnl should render for a pair.

    Returns one of:
      • "not_recent" : the pair was evicted from memory and has NOT been
                        re-scanned since removal -> bot says it wasn't scanned
                        recently.
      • "loss_at_sl" : the tracked call hit its stop-loss -> render the realized
                        loss at the bot's SL and suggested leverage.
      • "normal"     : render the standard PnL card.
    """
    removed = _as_dt(eviction_removed_at)
    scanned = _as_dt(latest_scan_time) or _scan_dt(record or {})
    if removed is not None:
        # Only honor a fresh scan that happened AFTER the eviction.
        if scanned is None or scanned <= removed:
            return "not_recent"

    if record and str(record.get("status", "")).lower() == "sl_hit":
        return "loss_at_sl"

    return "normal"


def loss_at_sl_pct(bias: str, entry: float, stop_loss: float, leverage: float) -> float:
    """Leveraged % loss when a position is closed at the stop-loss.

    Always returns a non-positive number (a loss), capped at -100%.
    """
    entry = float(entry or 0)
    sl = float(stop_loss or 0)
    lev = float(leverage or 1)
    if entry <= 0:
        return 0.0
    if str(bias).upper() == "LONG":
        raw = (sl - entry) / entry * 100.0
    else:
        raw = (entry - sl) / entry * 100.0
    lev_pct = raw * lev
    # SL is adverse; clamp any tiny positive rounding to <= 0 and floor at -100%.
    if lev_pct > 0:
        lev_pct = 0.0
    return max(lev_pct, -100.0)


# ──────────────────────────────────────────────────────────────────────────────
# 5. First-signal retention helper
# ──────────────────────────────────────────────────────────────────────────────

def merge_keep_first(
    existing: Optional[Dict[str, Any]],
    incoming: Dict[str, Any],
    current_price: Optional[float] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Merge a re-detected call into the stored record, KEEPING the first call.

    • If there is no existing record, OR the direction flipped, the incoming
      call becomes the new anchor (a flip is a genuinely new trade).
    • Otherwise the original entry / stop / targets / scan_time / confidence are
      preserved (the FIRST signal stays recorded) and only the live peak and
      motion bookkeeping are advanced.
    """
    now = now or datetime.now()

    if existing and str(existing.get("bias", "")).upper() == str(
        incoming.get("bias", "")
    ).upper():
        merged = dict(existing)
    else:
        # New pair or a direction flip → start a fresh anchor from incoming.
        merged = dict(incoming)
        merged.setdefault("first_scan_time", incoming.get("scan_time") or now)
        merged["peak_price"] = current_price or incoming.get("price") or incoming.get("entry")
        merged["last_motion_time"] = now
        merged["status"] = "active"
        return merged

    # Advance peak + motion only.
    bias = str(merged.get("bias", "LONG")).upper()
    entry = float(merged.get("price") or merged.get("entry") or 0)
    prev_peak = merged.get("peak_price")
    prev_peak = float(prev_peak) if prev_peak not in (None, "") else entry
    new_peak = prev_peak
    moved = False
    if current_price and current_price > 0 and entry > 0:
        if bias == "LONG":
            if current_price > prev_peak:
                new_peak = current_price
            fav_pct = (current_price - entry) / entry * 100.0
        else:
            if current_price < prev_peak:
                new_peak = current_price
            fav_pct = (entry - current_price) / entry * 100.0
        # Motion = a NEW favourable peak beyond the epsilon.
        peak_fav_pct = (
            (new_peak - entry) / entry * 100.0
            if bias == "LONG"
            else (entry - new_peak) / entry * 100.0
        )
        if new_peak != prev_peak and peak_fav_pct > MOTION_EPS_PCT:
            moved = True

    merged["peak_price"] = new_peak
    if moved:
        merged["last_motion_time"] = now
    merged.setdefault("last_motion_time", existing.get("last_motion_time") or now)
    merged.setdefault("first_scan_time", existing.get("first_scan_time") or existing.get("scan_time") or now)
    merged.setdefault("status", existing.get("status") or "active")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 6. Dormancy clear
# ──────────────────────────────────────────────────────────────────────────────

def should_clear_dormant(
    record: Dict[str, Any],
    now: Optional[datetime] = None,
    threshold_secs: int = DORMANT_SECS,
) -> bool:
    """True if a signal has shown no favourable motion for >= threshold_secs.

    A stopped-out signal is never "dormant" (it has a terminal status and is
    handled by the SL flow instead).
    """
    if not record:
        return False
    if str(record.get("status", "")).lower() == "sl_hit":
        return False
    now = now or datetime.now()
    anchor = _as_dt(record.get("last_motion_time")) or _as_dt(
        record.get("first_scan_time")
    ) or _as_dt(record.get("scan_time"))
    if anchor is None:
        return False
    return (now - anchor).total_seconds() >= threshold_secs


# ──────────────────────────────────────────────────────────────────────────────
# 7. Peak-reversal eviction
# ──────────────────────────────────────────────────────────────────────────────

def peak_favorable_pct(bias: str, entry: float, peak_price: float) -> float:
    """Favourable % the peak reached vs entry (positive when in profit)."""
    entry = float(entry or 0)
    peak = float(peak_price or 0)
    if entry <= 0 or peak <= 0:
        return 0.0
    if str(bias).upper() == "LONG":
        return (peak - entry) / entry * 100.0
    return (entry - peak) / entry * 100.0


def peak_reversal_frac(bias: str, peak_price: float, current_price: float) -> float:
    """Fraction the price has retraced from its peak (0..1+, adverse only)."""
    peak = float(peak_price or 0)
    cur = float(current_price or 0)
    if peak <= 0 or cur <= 0:
        return 0.0
    if str(bias).upper() == "LONG":
        frac = (peak - cur) / peak
    else:
        frac = (cur - peak) / peak
    return max(frac, 0.0)


def should_evict_peak_reversal(
    bias: str,
    entry: float,
    peak_price: float,
    current_price: float,
    threshold_frac: float = PEAK_REVERSAL_FRAC,
    require_profit: bool = True,
) -> bool:
    """True if a winning trade has reversed >= threshold_frac from its peak.

    require_profit ensures we only evict trades that actually went into profit
    ("a trade ... on profit"), matching the spec.
    """
    if require_profit and peak_favorable_pct(bias, entry, peak_price) <= 0:
        return False
    return peak_reversal_frac(bias, peak_price, current_price) >= threshold_frac
