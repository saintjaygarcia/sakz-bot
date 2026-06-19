#!/usr/bin/env python3
"""Scale-hardening regression test for sakz_db (local SQLite path).

Proves the concurrency hardening actually works:
  1. WAL + busy_timeout + synchronous pragmas are applied on connect.
  2. db_init creates the performance indexes.
  3. Hot queries (signal_outcomes by outcome / symbol+exchange+bias) use an
     index instead of a full SCAN.
  4. Many concurrent writers/readers (simulating ~100 users) complete with
     zero 'database is locked' errors.

Runs against a throwaway temp DB so it never touches real data.
"""
import os, sys, tempfile, threading, importlib, traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Point the DB at a temp file BEFORE importing config/sakz_db.
_tmp = tempfile.mkdtemp()
DB = os.path.join(_tmp, "scale_test.db")
os.environ["SAKZ_DB_PATH"] = DB
os.environ.pop("TURSO_URL", None)
os.environ.pop("TURSO_TOKEN", None)
os.environ.pop("TURSO_DATABASE_URL", None)
os.environ.pop("TURSO_AUTH_TOKEN", None)

import config
importlib.reload(config)
import sakz_db as db
importlib.reload(db)

failures = []
def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        failures.append(name)

print("DB_PATH =", db.DB_PATH, "| USE_TURSO =", db._USE_TURSO)
db.db_init()

# 1. pragmas
conn = db.db_connect()
jm = conn.execute("PRAGMA journal_mode").fetchone()[0]
bt = conn.execute("PRAGMA busy_timeout").fetchone()[0]
sy = conn.execute("PRAGMA synchronous").fetchone()[0]
check("journal_mode is WAL", str(jm).lower() == "wal")
check("busy_timeout >= 30000ms", int(bt) >= 30000)
check("synchronous == NORMAL(1)", int(sy) == 1)

# 2. indexes created
idx = {r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'").fetchall()}
for want in ("idx_so_outcome", "idx_so_sym_exch_bias", "idx_ph_key_ts",
             "idx_sr_scan_time", "idx_pro_up_sym_exch"):
    check("index exists: " + want, want in idx)

# 3. query plans use an index (not full SCAN)
def plan(sql, params=()):
    return " ".join(str(r[-1]) for r in
                    conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall())
p1 = plan("SELECT id FROM signal_outcomes WHERE outcome='pending'")
p2 = plan("SELECT id FROM signal_outcomes WHERE exchange=? AND symbol=? AND bias=? AND outcome='pending'", ("BYBIT","BTCUSDT","LONG"))
p3 = plan("SELECT price FROM price_history WHERE key=? ORDER BY ts", ("k",))
print("   plan1:", p1)
print("   plan2:", p2)
print("   plan3:", p3)
def uses_index(p):
    u = p.upper()
    return "INDEX" in u and "SCAN" not in u.replace("SCAN SUBQUERY", "")
check("signal_outcomes(outcome) uses index", uses_index(p1))
check("signal_outcomes(exch,sym,bias) uses index", uses_index(p2))
check("price_history(key) uses index", uses_index(p3))
conn.close()

# 4. concurrency stress: 100 worker threads each do mixed read/write directly
#    through db_connect() (bypassing the global lock) to prove the *SQLite*
#    layer itself no longer throws 'database is locked' under contention.
N_WORKERS = 100
OPS = 20
errors = []
barrier = threading.Barrier(N_WORKERS)

def worker(wid):
    try:
        barrier.wait()
        for i in range(OPS):
            c = db.db_connect()
            try:
                c.execute(
                    "INSERT INTO signal_outcomes "
                    "(signal_id,exchange,symbol,bias,confidence,entry_price,stop_loss,t1,t2,t3,scan_time) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (wid, "BYBIT", f"SYM{wid}USDT", "LONG", 7,
                     100.0, 95.0, 105.0, 110.0, 115.0, "2026-06-17T00:00:00"))
                c.commit()
                c.execute("SELECT COUNT(*) FROM signal_outcomes WHERE symbol=?",
                          (f"SYM{wid}USDT",)).fetchone()
            finally:
                c.close()
    except Exception as e:
        errors.append((wid, repr(e)))

threads = [threading.Thread(target=worker, args=(w,)) for w in range(N_WORKERS)]
for t in threads: t.start()
for t in threads: t.join()

locked = [e for e in errors if "locked" in e[1].lower()]
check(f"{N_WORKERS}x{OPS} concurrent ops, 0 errors", not errors)
check("0 'database is locked' errors under load", not locked)
if errors:
    print("   first errors:", errors[:5])

# verify all rows landed
conn = db.db_connect()
total = conn.execute("SELECT COUNT(*) FROM signal_outcomes").fetchone()[0]
conn.close()
check(f"all {N_WORKERS*OPS} inserts persisted", total == N_WORKERS*OPS)

print()
if failures:
    print(f"RESULT: {len(failures)} CHECK(S) FAILED")
    sys.exit(1)
print("RESULT: ALL SCALE-HARDENING CHECKS PASSED")
