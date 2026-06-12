"""Unit tests for sakz_prime pure core. Run: python3 test_prime.py ; echo EXIT=$?"""
import sys
from datetime import datetime, timedelta

import sakz_prime as P

_checks = 0
_fails = []


def check(name, cond):
    global _checks
    _checks += 1
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        _fails.append(name)


def approx(a, b, eps=1e-6):
    return abs(a - b) <= eps


# ── precise confidence backbone ────────────────────────────────────────────────
check("precise conf preferred over int",
      approx(P._precise_confidence({"confidence_precise": 8.4, "confidence": 8}), 8.4))
check("falls back to int conf",
      approx(P._precise_confidence({"confidence": 7}), 7.0))
check("missing conf -> 0",
      approx(P._precise_confidence({}), 0.0))

# ── ML edge tiebreaker ──────────────────────────────────────────────────────────
check("edge 1.0 -> +0.4", approx(P._edge_adjust({"consensus_score": 1.0}), 0.4))
check("edge 0.0 -> -0.4", approx(P._edge_adjust({"ml_score": 0.0}), -0.4))
check("edge 0.5 -> 0", approx(P._edge_adjust({"rf_score": 0.5}), 0.0))
check("no edge -> 0", approx(P._edge_adjust({}), 0.0))
check("consensus preferred over ml",
      approx(P._edge_adjust({"consensus_score": 1.0, "ml_score": 0.0}), 0.4))

# ── depth tiebreaker ─────────────────────────────────────────────────────────────
check("depth 9 -> 0", approx(P._depth_adjust({"winning_score": 9}), 0.0))
check("depth 12 -> +0.15", approx(P._depth_adjust({"winning_score": 12}), 0.15))
check("depth capped at +0.3", approx(P._depth_adjust({"winning_score": 100}), 0.3))
check("depth below floor -> 0", approx(P._depth_adjust({"winning_score": 5}), 0.0))

# ── entry decay ──────────────────────────────────────────────────────────────────
long_sig = {"price": 100.0, "bias": "LONG"}
check("no decay at entry", approx(P.entry_decay_multiplier(long_sig, 100.0), 1.0))
check("no decay below entry (long)", approx(P.entry_decay_multiplier(long_sig, 99.0), 1.0))
check("full decay floor at +3%", approx(P.entry_decay_multiplier(long_sig, 103.0), 0.5))
check("decay floor beyond +3%", approx(P.entry_decay_multiplier(long_sig, 110.0), 0.5))
check("partial decay mid-range < 1",
      0.5 < P.entry_decay_multiplier(long_sig, 101.5) < 1.0)
short_sig = {"price": 100.0, "bias": "SHORT"}
check("short: no decay above entry", approx(P.entry_decay_multiplier(short_sig, 101.0), 1.0))
check("short: decay when price drops past entry",
      approx(P.entry_decay_multiplier(short_sig, 97.0), 0.5))
check("bad entry -> no decay", approx(P.entry_decay_multiplier({"price": 0}, 100.0), 1.0))

# ── prime_score composite ────────────────────────────────────────────────────────
# strong 8 with good ML + depth beats weak 8 with poor ML
strong8 = {"symbol": "AAAUSDT", "confidence_precise": 8.4, "bias": "LONG", "price": 100.0,
           "consensus_score": 0.8, "winning_score": 12}
weak8 = {"symbol": "BBBUSDT", "confidence_precise": 7.6, "bias": "LONG", "price": 100.0,
         "consensus_score": 0.45, "winning_score": 9}
ss = P.prime_score(strong8, 100.0)
ws = P.prime_score(weak8, 100.0)
check("strong8 scores higher than weak8", ss > ws)
check("strong8 score in range", 8.0 <= ss <= 10.0)
check("score clamped to <=10",
      P.prime_score({"confidence_precise": 10, "consensus_score": 1.0, "winning_score": 50}, None) <= 10.0)
check("decay lowers score",
      P.prime_score(strong8, 103.0) < P.prime_score(strong8, 100.0))

# ── regime gate ───────────────────────────────────────────────────────────────────
check("regime blocked excluded", P.passes_regime_gate({"regime_blocked": True}) is False)
check("regime ok included", P.passes_regime_gate({"regime_blocked": False}) is True)

# ── ranking / threshold / topN ──────────────────────────────────────────────────
cands = [
    {"symbol": "AAAUSDT", "confidence_precise": 9.2, "bias": "LONG", "price": 100.0, "winning_score": 11},
    {"symbol": "BBBUSDT", "confidence_precise": 8.7, "bias": "LONG", "price": 100.0, "winning_score": 10},
    {"symbol": "CCCUSDT", "confidence_precise": 8.6, "bias": "LONG", "price": 100.0, "winning_score": 10},
    {"symbol": "DDDUSDT", "confidence_precise": 8.0, "bias": "LONG", "price": 100.0, "winning_score": 9},  # below 8.5
    {"symbol": "EEEUSDT", "confidence_precise": 9.9, "bias": "LONG", "price": 100.0, "winning_score": 11,
     "regime_blocked": True},  # blocked
]
ranked = P.rank_prime(cands, bar=8.5, top_n=3)
check("ranked returns top 3", len(ranked) == 3)
check("ranked sorted desc", ranked[0]["score"] >= ranked[1]["score"] >= ranked[2]["score"])
check("top pick is AAA", ranked[0]["symbol"] == "AAAUSDT")
check("below-bar DDD excluded", all(r["symbol"] != "DDDUSDT" for r in ranked))
check("regime-blocked EEE excluded", all(r["symbol"] != "EEEUSDT" for r in ranked))
check("empty when nothing clears bar", P.rank_prime(cands, bar=9.95, top_n=3) == [])

# ── scheduling math ───────────────────────────────────────────────────────────────
# 07:00 UTC, GMT+1 -> local 08:00 -> slot 8 due
u = datetime(2026, 6, 12, 7, 0)
check("slot 8 due at 07:00 UTC GMT+1", P.is_slot_due(u, 1, 8) is True)
check("slot 14 not due", P.is_slot_due(u, 1, 14) is False)
check("minute != 0 not due", P.is_slot_due(datetime(2026, 6, 12, 7, 1), 1, 8) is False)
check("due_slots filters", P.due_slots(u, 1, [8, 14, 20]) == [8])
# half-hour offset India +5:30 : 02:30 UTC -> 08:00 local
check("half-hour offset slot due",
      P.is_slot_due(datetime(2026, 6, 12, 2, 30), 5.5, 8) is True)
# wrap-around midnight: 22:00 UTC GMT+3 -> 01:00 local
check("wraparound local hour", P.local_now(datetime(2026, 6, 12, 22, 0), 3).hour == 1)
check("dedup key stable per day/slot",
      P.slot_dedup_key(42, u, 1, 8) == P.slot_dedup_key(42, datetime(2026, 6, 12, 7, 30), 1, 8))
check("dedup key differs by slot",
      P.slot_dedup_key(42, u, 1, 8) != P.slot_dedup_key(42, u, 1, 14))

# ── offset formatting / clamping ──────────────────────────────────────────────────
check("format GMT+1", P.format_offset(1) == "GMT+1")
check("format GMT", P.format_offset(0) == "GMT")
check("format GMT-5", P.format_offset(-5) == "GMT-5")
check("format half hour", P.format_offset(5.5) == "GMT+5:30")
check("offset clamp high", P.normalize_offset(99) == 14.0)
check("offset clamp low", P.normalize_offset(-99) == -12.0)

# ── cache freshness ────────────────────────────────────────────────────────────────
now = datetime(2026, 6, 12, 12, 0)
check("fresh within ttl", P.cache_is_fresh(now - timedelta(minutes=10), now, 15) is True)
check("stale beyond ttl", P.cache_is_fresh(now - timedelta(minutes=20), now, 15) is False)
check("none never fresh", P.cache_is_fresh(None, now, 15) is False)

# ── disclaimer / constants present ──────────────────────────────────────────────
check("disclaimer non-empty", isinstance(P.PRIME_DISCLAIMER, str) and len(P.PRIME_DISCLAIMER) > 20)
check("empty message set", P.PRIME_EMPTY_MESSAGE == "Nothing here yet.")

print(f"\n{_checks - len(_fails)}/{_checks} checks passed")
if _fails:
    print("FAILURES:")
    for f in _fails:
        print("  -", f)
    sys.exit(1)
print("ALL PRIME CORE CHECKS PASSED")
sys.exit(0)
