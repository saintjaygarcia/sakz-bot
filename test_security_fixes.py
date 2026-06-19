#!/usr/bin/env python3
"""Functional test for the merged build's security fixes."""
import os, sys, tempfile, importlib, hashlib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
_tmp = tempfile.mkdtemp()
os.environ["SAKZ_DB_PATH"] = os.path.join(_tmp, "sec.db")
for k in ("TURSO_URL","TURSO_TOKEN","TURSO_DATABASE_URL","TURSO_AUTH_TOKEN"):
    os.environ.pop(k, None)

import config; importlib.reload(config)
import sakz_db as db
db.db_init()

fails = []

# 1) no hardcoded admin password anywhere in .py source
import glob
leak = [f for f in glob.glob(os.path.join(ROOT, "**/*.py"), recursive=True)
        if os.path.basename(f) != "test_security_fixes.py" and "Sakazuki01" in open(f, encoding="utf-8", errors="surrogateescape").read()]
print("PASS" if not leak else "FAIL", "- no hardcoded password in .py", leak or "")
if leak: fails.append("hardcoded password")

# 2) admin session TTL: fresh auth is valid, stale auth expires
CHAT = 12345
db.db_admin_set_auth(CHAT)
ok_fresh = db.db_admin_is_authed(CHAT)
print("PASS" if ok_fresh else "FAIL", "- fresh admin session is authed")
if not ok_fresh: fails.append("fresh session")

os.environ["SAKZ_ADMIN_SESSION_TTL_HOURS"] = "0"  # everything is instantly stale
importlib.reload  # no-op; is_authed reads env live
ok_stale = db.db_admin_is_authed(CHAT)
print("PASS" if not ok_stale else "FAIL", "- stale admin session expires (TTL=0)")
if ok_stale: fails.append("stale session not expiring")
os.environ.pop("SAKZ_ADMIN_SESSION_TTL_HOURS", None)

# 3) unknown chat is never authed
print("PASS" if not db.db_admin_is_authed(99999) else "FAIL", "- unknown chat not authed")
if db.db_admin_is_authed(99999): fails.append("unknown authed")

# 4) WAL actually applied on the local path
conn = db.db_connect(); jm = conn.execute("PRAGMA journal_mode").fetchone()[0]; conn.close()
print("PASS" if str(jm).lower()=="wal" else "FAIL", f"- journal_mode is WAL (got {jm})")
if str(jm).lower()!="wal": fails.append("WAL not applied")

if fails:
    print("\nRESULT: SECURITY TEST FAILURES:", fails); sys.exit(1)
print("\nRESULT: ALL SECURITY-FIX CHECKS PASSED")
