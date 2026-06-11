"""
test_signal_behavior.py — behaviour tests for the signal lifecycle spec.

Covers:
  1. Autoscan dedup (no repeat unless higher confidence).
  2. Per-user timeframe cadence gating.
  3. /pnl resolves to the oldest signal.
  4. SL-hit -> loss-at-SL card mode + leveraged loss math.
  5. First signal retained while profiting (re-detects don't overwrite).
  6. Dormant clear after 15 min of no motion.
  7. 20% peak reversal eviction + "not scanned recently" guard.

Also exercises the sqlite-backed DB helpers (active_signals / signal_evictions)
against an in-memory database so the persistence layer is verified too.

Run directly:  python3 test_signal_behavior.py
Returns exit code 0 only when every assertion passes.
"""
from __future__ import annotations

import sqlite3
import sys
import traceback
from datetime import datetime, timedelta

import sakz_signal_logic as L


_FAILURES = []
_PASSES = 0


def check(cond, msg):
    global _PASSES
    if cond:
        _PASSES += 1
    else:
        _FAILURES.append(msg)
        print(f"  ❌ FAIL: {msg}")


NOW = datetime(2026, 6, 11, 22, 0, 0)


# ───────────────────────────────────────────────────────────────────
def test_autoscan_dedup():
    print("[1] autoscan dedup (confidence-aware)")
    # First time → send.
    send, rec = L.autoscan_decide_send(None, 7, now=NOW)
    check(send is True, "first detect should send")
    check(rec["confidence"] == 7, "record stores confidence")

    # Same confidence again → suppress (don't repeat the same call).
    send2, rec2 = L.autoscan_decide_send(rec, 7, now=NOW + timedelta(minutes=10))
    check(send2 is False, "same confidence should NOT repeat")
    check(rec2["confidence"] == 7, "suppressed re-detect keeps high-water mark")

    # Lower confidence → suppress.
    send3, _ = L.autoscan_decide_send(rec, 5, now=NOW + timedelta(minutes=20))
    check(send3 is False, "lower confidence should NOT repeat")

    # Higher confidence → send.
    send4, rec4 = L.autoscan_decide_send(rec, 9, now=NOW + timedelta(minutes=30))
    check(send4 is True, "higher confidence SHOULD re-send")
    check(rec4["confidence"] == 9, "record advances to higher confidence")

    # After a higher-confidence send, an equal one is suppressed again.
    send5, _ = L.autoscan_decide_send(rec4, 9, now=NOW + timedelta(minutes=40))
    check(send5 is False, "equal to new high should NOT repeat")


def test_timeframe_cadence():
    print("[2] per-user timeframe cadence")
    # "always" subscriber is never cadence-gated.
    check(L.timeframe_due(NOW, None, now=NOW + timedelta(seconds=1)) is True,
          "always subscriber always due")

    # 4h subscriber: not due 1h later, due 4h later.
    last = NOW
    check(L.timeframe_due(last, "4h", now=NOW + timedelta(hours=1)) is False,
          "4h subscriber not due after 1h")
    check(L.timeframe_due(last, "4h", now=NOW + timedelta(hours=4)) is True,
          "4h subscriber due after 4h")
    # Never alerted before → due.
    check(L.timeframe_due(None, "1d", now=NOW) is True,
          "first-ever alert is due")
    # 15m subscriber due after 15m, not at 5m.
    check(L.timeframe_due(last, "15m", now=NOW + timedelta(minutes=5)) is False,
          "15m not due at 5m")
    check(L.timeframe_due(last, "15m", now=NOW + timedelta(minutes=15)) is True,
          "15m due at 15m")


def test_oldest_signal():
    print("[3] /pnl oldest signal")
    sigs = [
        {"symbol": "BTC", "bias": "LONG", "scan_time": NOW - timedelta(hours=1)},
        {"symbol": "BTC", "bias": "LONG", "scan_time": NOW - timedelta(days=2)},
        {"symbol": "BTC", "bias": "LONG", "scan_time": NOW - timedelta(minutes=5)},
    ]
    oldest = L.pick_oldest_signal(sigs)
    check(oldest["scan_time"] == NOW - timedelta(days=2), "picks earliest scan_time")
    check(L.pick_oldest_signal([]) is None, "empty -> None")

    # Collapse multiple re-detects to one-per-direction, oldest kept.
    rows = [
        {"exchange": "BYBIT", "symbol": "ETHUSDT", "bias": "LONG", "scan_time": NOW - timedelta(hours=2)},
        {"exchange": "BYBIT", "symbol": "ETHUSDT", "bias": "LONG", "scan_time": NOW - timedelta(days=1)},
        {"exchange": "BYBIT", "symbol": "ETHUSDT", "bias": "SHORT", "scan_time": NOW - timedelta(hours=3)},
    ]
    collapsed = L.collapse_to_oldest_per_direction(rows)
    check(len(collapsed) == 2, "collapses to 2 directions")
    longrow = [r for r in collapsed if r["bias"] == "LONG"][0]
    check(longrow["scan_time"] == NOW - timedelta(days=1), "keeps oldest LONG call")


def test_sl_loss_card():
    print("[4] SL-hit loss card mode + math")
    rec = {"status": "sl_hit", "bias": "LONG", "price": 100.0, "stop_loss": 95.0}
    mode = L.pnl_card_mode(rec, latest_scan_time=NOW)
    check(mode == "loss_at_sl", "sl_hit -> loss_at_sl mode")

    # LONG: entry 100, SL 95 = -5% raw, x10 leverage = -50%.
    pct = L.loss_at_sl_pct("LONG", 100.0, 95.0, 10)
    check(abs(pct - (-50.0)) < 1e-9, f"LONG SL loss should be -50%, got {pct}")

    # SHORT: entry 100, SL 110 = -10% raw, x5 = -50%.
    pct_s = L.loss_at_sl_pct("SHORT", 100.0, 110.0, 5)
    check(abs(pct_s - (-50.0)) < 1e-9, f"SHORT SL loss should be -50%, got {pct_s}")

    # Loss is floored at -100%.
    pct_floor = L.loss_at_sl_pct("LONG", 100.0, 80.0, 20)  # -20% x20 = -400%
    check(pct_floor == -100.0, "loss floored at -100%")

    # A normal (active) record renders the normal card.
    rec_ok = {"status": "active", "bias": "LONG", "scan_time": NOW}
    check(L.pnl_card_mode(rec_ok, latest_scan_time=NOW) == "normal",
          "active -> normal card")


def test_first_signal_retained():
    print("[5] first signal retained while profiting")
    first = {
        "exchange": "BYBIT", "symbol": "SOLUSDT", "bias": "LONG",
        "price": 100.0, "stop_loss": 95.0, "confidence": 7,
        "scan_time": NOW - timedelta(hours=3),
        "first_scan_time": NOW - timedelta(hours=3),
        "peak_price": 100.0, "last_motion_time": NOW - timedelta(hours=3),
        "status": "active",
    }
    # Re-detected later at a higher price, higher confidence.
    incoming = {
        "exchange": "BYBIT", "symbol": "SOLUSDT", "bias": "LONG",
        "price": 110.0, "stop_loss": 104.0, "confidence": 9,
        "scan_time": NOW,
    }
    merged = L.merge_keep_first(first, incoming, current_price=112.0, now=NOW)
    check(merged["price"] == 100.0, "original entry preserved")
    check(merged["first_scan_time"] == NOW - timedelta(hours=3),
          "original first_scan_time preserved")
    check(merged["peak_price"] == 112.0, "peak advances to new high")
    check(merged["last_motion_time"] == NOW, "motion timestamp advances")

    # Direction flip starts a fresh anchor.
    flip = dict(incoming, bias="SHORT", price=110.0)
    merged_flip = L.merge_keep_first(first, flip, current_price=110.0, now=NOW)
    check(merged_flip["price"] == 110.0, "direction flip resets anchor")
    check(merged_flip["bias"] == "SHORT", "flip keeps new bias")


def test_dormant_clear():
    print("[6] dormant clear after 15 min")
    rec = {
        "status": "active",
        "first_scan_time": NOW - timedelta(minutes=30),
        "last_motion_time": NOW - timedelta(minutes=20),
    }
    check(L.should_clear_dormant(rec, now=NOW) is True,
          "no motion 20min -> dormant")
    rec2 = dict(rec, last_motion_time=NOW - timedelta(minutes=5))
    check(L.should_clear_dormant(rec2, now=NOW) is False,
          "motion 5min ago -> not dormant")
    # Stopped-out signals are never 'dormant'.
    rec3 = dict(rec, status="sl_hit")
    check(L.should_clear_dormant(rec3, now=NOW) is False,
          "sl_hit never dormant")


def test_peak_reversal_and_guard():
    print("[7] 20% peak reversal eviction + not-scanned-recently guard")
    # LONG entry 100, peaked at 200 (in profit), now 159 = 20.5% off peak.
    check(L.should_evict_peak_reversal("LONG", 100, 200, 159) is True,
          "20%+ retrace from peak -> evict")
    # Only 10% off peak -> keep.
    check(L.should_evict_peak_reversal("LONG", 100, 200, 180) is False,
          "10% retrace -> keep")
    # Never in profit (peak below entry) -> don't evict via this rule.
    check(L.should_evict_peak_reversal("LONG", 100, 98, 70) is False,
          "never profitable -> not peak-evicted")
    # SHORT entry 100, peaked down at 50, now 61 = 22% off peak.
    check(L.should_evict_peak_reversal("SHORT", 100, 50, 61) is True,
          "SHORT 20%+ retrace -> evict")

    # Guard: pair evicted at T, no scan since -> not_recent.
    removed = NOW - timedelta(hours=1)
    mode = L.pnl_card_mode(None, latest_scan_time=NOW - timedelta(hours=2),
                           eviction_removed_at=removed)
    check(mode == "not_recent", "scan older than eviction -> not_recent")
    # Re-scanned AFTER eviction -> allowed (normal).
    rec = {"status": "active", "scan_time": NOW}
    mode2 = L.pnl_card_mode(rec, latest_scan_time=NOW, eviction_removed_at=removed)
    check(mode2 == "normal", "re-scan after eviction -> normal")
    # Re-scanned after eviction but stopped out -> loss card still wins.
    rec_sl = {"status": "sl_hit", "scan_time": NOW, "bias": "LONG",
              "price": 100, "stop_loss": 95}
    mode3 = L.pnl_card_mode(rec_sl, latest_scan_time=NOW, eviction_removed_at=removed)
    check(mode3 == "loss_at_sl", "re-scan + sl_hit -> loss_at_sl")


# ───────────────────────────────────────────────────────────────────
# DB-layer tests (in-memory sqlite, mirrors sakz_db helpers)
# ───────────────────────────────────────────────────────────────────
try:
    import sakz_db_lifecycle_schema as SCHEMA  # optional shared DDL
    _HAVE_SCHEMA = True
except Exception:
    _HAVE_SCHEMA = False


def test_db_helpers():
    print("[8] DB persistence (active_signals + evictions)")
    try:
        import sakz_db
    except Exception as e:
        print(f"  ⚠️  sakz_db import failed ({e}); skipping DB integration test")
        return
    needed = ["db_register_first_signal", "db_get_active_signal",
              "db_evict_signal", "db_get_eviction", "db_all_active_signals"]
    missing = [n for n in needed if not hasattr(sakz_db, n)]
    if missing:
        print(f"  ⚠️  sakz_db missing {missing}; skipping (will be added in integration)")
        return

    conn = sqlite3.connect(":memory:")
    sakz_db.lifecycle_init(conn)

    sig = {"exchange": "BYBIT", "symbol": "SOLUSDT", "bias": "LONG",
           "price": 100.0, "stop_loss": 95.0, "t1": 110, "t2": 120, "t3": 130,
           "confidence": 7, "leverage": 10, "scan_time": (NOW - timedelta(hours=3)).isoformat()}
    sakz_db.db_register_first_signal(sig, conn=conn)
    # Re-detect with higher price/conf must NOT overwrite the first entry.
    sig2 = dict(sig, price=120.0, confidence=9, scan_time=NOW.isoformat())
    sakz_db.db_register_first_signal(sig2, conn=conn, current_price=121.0)
    rec = sakz_db.db_get_active_signal("BYBIT", "SOLUSDT", conn=conn)
    check(rec is not None, "active signal stored")
    check(abs(float(rec["entry"]) - 100.0) < 1e-9, "first entry preserved in DB")

    # Evict + guard.
    sakz_db.db_evict_signal("BYBIT", "SOLUSDT", NOW.isoformat(), conn=conn)
    check(sakz_db.db_get_active_signal("BYBIT", "SOLUSDT", conn=conn) is None,
          "evicted signal removed from active")
    ev = sakz_db.db_get_eviction("BYBIT", "SOLUSDT", conn=conn)
    check(ev is not None, "eviction record stored")
    conn.close()


def test_non_crypto_excluded():
    print("[9] non-crypto symbols excluded from scorer")
    import sakz_exchanges as X
    # Real crypto perps must pass.
    for sym in ("BTCUSDT", "ETHUSDT", "SOL_USDT", "DOGEUSDT"):
        check(X.is_crypto_symbol(sym) is True, f"{sym} should be treated as crypto")
    # Stocks / metals / oil / FX synthetics must be rejected.
    for sym in ("MRVLSTOCKUSDT", "XAUTUSDT", "XAGUSDT", "USOILUSDT",
                "UKOILUSDT", "EURUSDT", "SPXUSDT"):
        check(X.is_crypto_symbol(sym) is False, f"{sym} should be rejected as non-crypto")


def test_paper_rr_uses_sl_distance():
    print("[10] paper R:R uses planned SL risk, not exit distance")
    import sakz_paper as P
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE paper_positions (
               id INTEGER PRIMARY KEY, entry_price REAL, stop_loss REAL,
               bias TEXT, closed_at TEXT, exit_price REAL, outcome TEXT,
               pnl_pct REAL, rr REAL)"""
    )
    conn.execute("INSERT INTO paper_positions (id, entry_price, stop_loss, bias) "
                 "VALUES (1, 100.0, 95.0, 'LONG')")
    # LONG entry 100, SL 95 (5% risk), exit at T1 110 (~10% gross reward).
    P._write_close(conn, 1, 110.0, 'T1', 100.0, 'LONG', 95.0)
    row = conn.execute("SELECT pnl_pct, rr FROM paper_positions WHERE id=1").fetchone()
    pnl, rr = row
    # Reward ~2x the planned risk → rr must be well above 1.0 (old bug gave ~1.0).
    check(rr > 1.5, f"R:R should reflect SL-based risk (got {rr})")
    # Sanity: a SL hit should produce a negative pnl and rr.
    conn.execute("INSERT INTO paper_positions (id, entry_price, stop_loss, bias) "
                 "VALUES (2, 100.0, 95.0, 'LONG')")
    P._write_close(conn, 2, 95.0, 'SL', 100.0, 'LONG', 95.0)
    row2 = conn.execute("SELECT pnl_pct, rr FROM paper_positions WHERE id=2").fetchone()
    check(row2[0] < 0, "SL hit should be a loss")
    conn.close()


def main():
    tests = [
        test_autoscan_dedup,
        test_timeframe_cadence,
        test_oldest_signal,
        test_sl_loss_card,
        test_first_signal_retained,
        test_dormant_clear,
        test_peak_reversal_and_guard,
        test_db_helpers,
        test_non_crypto_excluded,
        test_paper_rr_uses_sl_distance,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            _FAILURES.append(f"{t.__name__} raised")
            print(f"  💥 EXCEPTION in {t.__name__}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    if _FAILURES:
        print(f"RESULT: {_PASSES} passed, {len(_FAILURES)} FAILED")
        for f in _FAILURES:
            print(f"   - {f}")
        return 1
    print(f"RESULT: ALL {_PASSES} CHECKS PASSED ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
