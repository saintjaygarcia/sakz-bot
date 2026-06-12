"""Integration smoke test: PRIME DB persistence + ranking + dashboard payload.
Exercises the real sakz_db helpers against a temp DB (no Telegram/network).
Run: python3 test_prime_integration.py ; echo EXIT=$?
"""
import os, sys, tempfile, json

# Point the DB at a throwaway file BEFORE importing sakz_db.
_tmp = tempfile.mkdtemp()
os.environ["SAKZ_DB_PATH"] = os.path.join(_tmp, "prime_test.db")

import sakz_db as db
import sakz_prime as prime

_fails = []
def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        _fails.append(name)

db.db_init()

CHAT = 123456

# ── subscribe / offset / slots ─────────────────────────────────────────
check("not subscribed initially", db.db_prime_is_subscribed(CHAT) is False)
db.db_prime_subscribe(CHAT)
check("subscribed after subscribe", db.db_prime_is_subscribed(CHAT) is True)
db.db_prime_set_offset(CHAT, 1.0)
check("offset persisted", db.db_prime_get_user(CHAT)["gmt_offset"] == 1.0)
# idempotent subscribe must NOT wipe the offset
db.db_prime_subscribe(CHAT)
check("offset preserved on re-subscribe", db.db_prime_get_user(CHAT)["gmt_offset"] == 1.0)

db.db_prime_set_slots(CHAT, [8, 14, 20])
check("slots persisted", db.db_prime_get_slots(CHAT) == [8, 14, 20])
remaining = db.db_prime_toggle_slot(CHAT, 14)
check("toggle removes slot", remaining == [8, 20])
remaining = db.db_prime_toggle_slot(CHAT, 14)
check("toggle re-adds slot", remaining == [8, 14, 20])

check("get_all_users returns the user",
      any(u["chat_id"] == CHAT for u in db.db_prime_get_all_users()))

# ── cache round-trip with real signal payloads ─────────────────────────────
fake_signals = [
    {"symbol": "AAAUSDT", "bias": "LONG", "exchange": "bybit", "price": 100.0,
     "entry_low": 99.5, "entry_high": 100.0, "t1": 102.0, "t2": 104.0, "t3": 107.0,
     "confidence": 9, "confidence_precise": 9.2, "winning_score": 12, "consensus_score": 0.8},
    {"symbol": "BBBUSDT", "bias": "SHORT", "exchange": "bybit", "price": 50.0,
     "entry_low": 50.0, "entry_high": 50.3, "t1": 49.0, "t2": 48.0, "t3": 46.0,
     "confidence": 8, "confidence_precise": 8.6, "winning_score": 10, "consensus_score": 0.6},
    {"symbol": "CCCUSDT", "bias": "LONG", "exchange": "mexc", "price": 10.0,
     "entry_low": 9.9, "entry_high": 10.0, "t1": 10.2, "t2": 10.5, "t3": 11.0,
     "confidence": 7, "confidence_precise": 7.4, "winning_score": 8},  # below 8.5 bar
]
ranked = prime.rank_prime(fake_signals, current_prices=None, bar=8.5, top_n=3)
check("ranking keeps 2 above bar", len(ranked) == 2)
check("AAA ranked first", ranked[0]["symbol"] == "AAAUSDT")

payload = [{"symbol": x["signal"]["symbol"], "bias": x["signal"]["bias"],
            "entry_low": x["signal"]["entry_low"], "entry_high": x["signal"]["entry_high"],
            "t1": x["signal"]["t1"], "t2": x["signal"]["t2"], "t3": x["signal"]["t3"],
            "confidence": x["signal"]["confidence"],
            "confidence_precise": x["signal"]["confidence_precise"],
            "score": x["score"]} for x in ranked]
db.db_prime_cache_put(json.dumps(payload))
row = db.db_prime_cache_get()
check("cache row stored", row is not None and row["payload"])
loaded = json.loads(row["payload"])
check("cache round-trips picks", len(loaded) == 2 and loaded[0]["symbol"] == "AAAUSDT")

# ── alert dedup ledger ────────────────────────────────────────────
from datetime import datetime
key = prime.slot_dedup_key(CHAT, datetime(2026, 6, 12, 7, 0), 1, 8)
check("alert not sent initially", db.db_prime_alert_already_sent(key) is False)
db.db_prime_mark_alert_sent(key)
check("alert marked sent", db.db_prime_alert_already_sent(key) is True)

# ── unsubscribe wipes user + prefs ───────────────────────────────────
db.db_prime_unsubscribe(CHAT)
check("unsubscribed", db.db_prime_is_subscribed(CHAT) is False)
check("slots cleared on unsubscribe", db.db_prime_get_slots(CHAT) == [])

print()
if _fails:
    print("FAILURES:", _fails)
    sys.exit(1)
print("ALL PRIME INTEGRATION CHECKS PASSED")
sys.exit(0)
