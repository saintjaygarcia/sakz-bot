"""Tests for the autoscan reminder + T1/T2/T3/SL milestone logic."""
from datetime import datetime, timedelta
import sakz_signal_logic as S

ok = 0
fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}")


# ── Feature 1: reminders via autoscan_decide_send(reminder_secs=...) ──────────
now = datetime(2026, 6, 12, 12, 0, 0)

# Never sent -> SEND (unchanged)
send, rec = S.autoscan_decide_send(None, 8.0, now=now, reminder_secs=6 * 3600)
check("new call sends", send is True and rec["confidence"] == 8.0)

# Same conf, within reminder window -> SUPPRESS
prev = {"confidence": 8.0, "sent_at": now - timedelta(hours=2)}
send, rec = S.autoscan_decide_send(prev, 8.0, now=now, reminder_secs=6 * 3600)
check("same conf within window suppressed", send is False)

# Same conf, AFTER reminder window -> SEND (reminder)
prev = {"confidence": 8.0, "sent_at": now - timedelta(hours=7)}
send, rec = S.autoscan_decide_send(prev, 8.0, now=now, reminder_secs=6 * 3600)
check("reminder fires after window", send is True and rec["sent_at"] == now)

# Lower conf after window still reminds, but high-water preserved
prev = {"confidence": 9.0, "sent_at": now - timedelta(hours=8)}
send, rec = S.autoscan_decide_send(prev, 7.0, now=now, reminder_secs=6 * 3600)
check("reminder preserves high-water conf", send is True and rec["confidence"] == 9.0)

# Higher conf within window -> SEND (upgrade, not reminder)
prev = {"confidence": 7.0, "sent_at": now - timedelta(hours=1)}
send, rec = S.autoscan_decide_send(prev, 9.0, now=now, reminder_secs=6 * 3600)
check("upgrade still sends within window", send is True and rec["confidence"] == 9.0)

# Backward-compat: no reminder_secs -> old behavior (suppress same conf)
prev = {"confidence": 8.0, "sent_at": now - timedelta(hours=99)}
send, rec = S.autoscan_decide_send(prev, 8.0, now=now)
check("no reminder_secs -> legacy suppress", send is False)

# ── Feature 2: lifecycle_milestones ──────────────────────────────────────────
none_hit = {}

# LONG targets
m = S.lifecycle_milestones("LONG", 105, 90, 104, 110, 120, none_hit)
check("LONG t1 reached", m == ["t1"])
m = S.lifecycle_milestones("LONG", 115, 90, 104, 110, 120, none_hit)
check("LONG t1+t2 reached", m == ["t1", "t2"])
m = S.lifecycle_milestones("LONG", 125, 90, 104, 110, 120, none_hit)
check("LONG all targets reached", m == ["t1", "t2", "t3"])

# already-alerted suppressed
m = S.lifecycle_milestones("LONG", 115, 90, 104, 110, 120, {"t1": True})
check("LONG skips already-alerted t1", m == ["t2"])

# LONG stop-loss terminal
m = S.lifecycle_milestones("LONG", 89, 90, 104, 110, 120, none_hit)
check("LONG sl terminal", m == ["sl"])
m = S.lifecycle_milestones("LONG", 89, 90, 104, 110, 120, {"sl": True})
check("LONG sl not repeated", m == [])

# SHORT targets (price falls)
m = S.lifecycle_milestones("SHORT", 96, 110, 96, 90, 80, none_hit)
check("SHORT t1 reached", m == ["t1"])
m = S.lifecycle_milestones("SHORT", 79, 110, 96, 90, 80, none_hit)
check("SHORT all targets reached", m == ["t1", "t2", "t3"])
m = S.lifecycle_milestones("SHORT", 111, 110, 96, 90, 80, none_hit)
check("SHORT sl terminal", m == ["sl"])

# no price -> nothing
m = S.lifecycle_milestones("LONG", 0, 90, 104, 110, 120, none_hit)
check("no price -> no hits", m == [])

# not yet reached
m = S.lifecycle_milestones("LONG", 100, 90, 104, 110, 120, none_hit)
check("LONG nothing reached yet", m == [])

print(f"\n{ok}/{ok + fail} checks passed")
if fail:
    raise SystemExit(1)
print("ALL AUTOSCAN LIFECYCLE CHECKS PASSED")
