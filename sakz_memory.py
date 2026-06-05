"""
sakz_memory.py — Persistent Memory Layer for sakz_bot
═══════════════════════════════════════════════════════
Keeps all critical user data safe across bot reuploads / redeployments.

HOW IT WORKS
────────────
1. On every startup, restore_on_startup() reads sakz_memory.json and
   re-inserts any data that is missing from the SQLite DB.
2. Every N minutes (default 10), a background job snapshots all critical
   tables into sakz_memory.json.
3. When you reupload sakz_bot.py, the SQLite DB may be wiped — but
   sakz_memory.json survives (store it on a persistent volume / commit
   it to your repo as a last resort).

DATA THAT SURVIVES REUPLOADS
─────────────────────────────
  ✅ tracked_trades        — active trade reminders per user
  ✅ user_alerts           — /alert symbol registrations
  ✅ price_level_alerts    — /palert price-level triggers
  ✅ watchlist             — /watch symbols per user
  ✅ broadcast_channels    — /broadcast channel list
  ✅ snail_unlocked        — 🐌 snail-mode unlocked users
  ✅ snail_sessions        — 🐌 active snail sessions
  ✅ snail_signals         — 🐌 snail signal history
  ✅ user_activity         — user seen/command stats
  ✅ signal_outcomes (pending only) — open trade outcomes

DATA THAT REFRESHES NORMALLY (not backed up)
─────────────────────────────────────────────
  🔄 scan_results         — refreshed on /scan
  🔄 price_history        — live price data, always fresh
  🔄 btc_price_snapshots  — live BTC price
  🔄 signal_bias          — recalculated per scan
  🔄 signal_card_cache    — rebuilt on each Details click

SETUP (add these 3 lines to sakz_bot.py)
─────────────────────────────────────────
  # At the very top of sakz_bot.py, after your imports:
  import sakz_memory

  # Inside main(), BEFORE db_init() is called:
  sakz_memory.restore_on_startup()

  # Inside main(), AFTER app is created (after Application.builder()...build()):
  sakz_memory.start_background_backup(app)

That's it. sakz_memory handles the rest automatically.

PERSISTENT VOLUME TIP (Render / Railway)
─────────────────────────────────────────
  Set the env var SAKZ_MEMORY_PATH to a path on your persistent disk:
    SAKZ_MEMORY_PATH=/data/sakz_memory.json
  Then mount /data as a persistent volume in your platform settings.
  This makes both sakz_data.db AND sakz_memory.json survive deploys.
"""

import os
import json
import sqlite3

# ── Turso support ─────────────────────────────────────────────────────────────
TURSO_URL   = os.environ.get("TURSO_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_TOKEN", "")
_USE_TURSO  = bool(TURSO_URL and TURSO_TOKEN)
if _USE_TURSO:
    try:
        import libsql_experimental as libsql
    except ImportError:
        libsql     = None
        _USE_TURSO = False
import logging
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Path to the SQLite DB — matches sakz_bot.py default
DB_PATH = os.environ.get("SAKZ_DB_PATH", "sakz_data.db")

# Path to the JSON memory snapshot — override with env var for persistent volume
MEMORY_PATH = os.environ.get("SAKZ_MEMORY_PATH", "sakz_memory.json")

# How often to auto-backup (seconds). 600 = every 10 minutes.
BACKUP_INTERVAL = int(os.environ.get("SAKZ_BACKUP_INTERVAL", "600"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _db():
    """Open the DB — Turso if configured, else local SQLite."""
    if _USE_TURSO and libsql:
        conn = libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)
        conn.row_factory = sqlite3.Row
        return conn
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
    return conn


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _col_exists(conn, table: str, col: str) -> bool:
    try:
        conn.execute(f"SELECT {col} FROM {table} LIMIT 1")
        return True
    except Exception:
        return False


def _now() -> str:
    return datetime.now().isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BACKUP
# ══════════════════════════════════════════════════════════════════════════════

def backup() -> bool:
    """
    Snapshot all critical user tables into MEMORY_PATH (JSON).
    Returns True on success, False on failure.
    """
    try:
        conn = _db()
        snapshot = {
            "_meta": {
                "version":    2,
                "created_at": _now(),
                "db_path":    DB_PATH,
            }
        }

        # ── tracked_trades ────────────────────────────────────────────────────
        if _table_exists(conn, "tracked_trades"):
            rows = conn.execute("SELECT * FROM tracked_trades").fetchall()
            snapshot["tracked_trades"] = [dict(r) for r in rows]
        else:
            snapshot["tracked_trades"] = []

        # ── user_alerts ───────────────────────────────────────────────────────
        if _table_exists(conn, "user_alerts"):
            rows = conn.execute("SELECT * FROM user_alerts").fetchall()
            snapshot["user_alerts"] = [dict(r) for r in rows]
        else:
            snapshot["user_alerts"] = []

        # ── price_level_alerts (un-triggered only) ────────────────────────────
        if _table_exists(conn, "price_level_alerts"):
            rows = conn.execute(
                "SELECT * FROM price_level_alerts WHERE triggered=0"
            ).fetchall()
            snapshot["price_level_alerts"] = [dict(r) for r in rows]
        else:
            snapshot["price_level_alerts"] = []

        # ── watchlist ─────────────────────────────────────────────────────────
        if _table_exists(conn, "watchlist"):
            rows = conn.execute("SELECT * FROM watchlist").fetchall()
            snapshot["watchlist"] = [dict(r) for r in rows]
        else:
            snapshot["watchlist"] = []

        # ── broadcast_channels ────────────────────────────────────────────────
        if _table_exists(conn, "broadcast_channels"):
            rows = conn.execute("SELECT * FROM broadcast_channels").fetchall()
            snapshot["broadcast_channels"] = [dict(r) for r in rows]
        else:
            snapshot["broadcast_channels"] = []

        # ── snail_unlocked ────────────────────────────────────────────────────
        if _table_exists(conn, "snail_unlocked"):
            rows = conn.execute("SELECT * FROM snail_unlocked").fetchall()
            snapshot["snail_unlocked"] = [dict(r) for r in rows]
        else:
            snapshot["snail_unlocked"] = []

        # ── snail_sessions (active only) ──────────────────────────────────────
        if _table_exists(conn, "snail_sessions"):
            rows = conn.execute(
                "SELECT * FROM snail_sessions WHERE active=1"
            ).fetchall()
            snapshot["snail_sessions"] = [dict(r) for r in rows]
        else:
            snapshot["snail_sessions"] = []

        # ── snail_signals (last 7 days) ───────────────────────────────────────
        if _table_exists(conn, "snail_signals"):
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            rows = conn.execute(
                "SELECT * FROM snail_signals WHERE sent_at >= ?", (cutoff,)
            ).fetchall()
            snapshot["snail_signals"] = [dict(r) for r in rows]
        else:
            snapshot["snail_signals"] = []

        # ── user_activity ─────────────────────────────────────────────────────
        if _table_exists(conn, "user_activity"):
            rows = conn.execute("SELECT * FROM user_activity").fetchall()
            snapshot["user_activity"] = [dict(r) for r in rows]
        else:
            snapshot["user_activity"] = []

        # ── signal_outcomes (pending only — open trades) ──────────────────────
        if _table_exists(conn, "signal_outcomes"):
            rows = conn.execute(
                "SELECT * FROM signal_outcomes WHERE outcome='pending'"
            ).fetchall()
            snapshot["signal_outcomes_pending"] = [dict(r) for r in rows]
        else:
            snapshot["signal_outcomes_pending"] = []

        # ── admin_sessions ────────────────────────────────────────────────────
        if _table_exists(conn, "admin_sessions"):
            rows = conn.execute("SELECT * FROM admin_sessions").fetchall()
            snapshot["admin_sessions"] = [dict(r) for r in rows]
        else:
            snapshot["admin_sessions"] = []

        conn.close()

        # Write atomically (temp file → rename)
        tmp = MEMORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)
        os.replace(tmp, MEMORY_PATH)

        counts = {
            k: len(v) for k, v in snapshot.items() if k != "_meta"
        }
        logger.info(
            "🧠 sakz_memory: backup saved → %s | %s",
            MEMORY_PATH,
            " | ".join(f"{k}={v}" for k, v in counts.items() if v),
        )
        return True

    except Exception as e:
        logger.error("🧠 sakz_memory: backup FAILED — %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RESTORE
# ══════════════════════════════════════════════════════════════════════════════

def restore_on_startup() -> bool:
    """
    Called once at bot startup (before db_init in sakz_bot.py).
    Reads sakz_memory.json and re-inserts any rows missing from the DB.
    Safe to call on a fresh DB or an existing populated DB — duplicates
    are skipped via INSERT OR IGNORE / INSERT OR REPLACE.
    Returns True if a backup was found and processed, False otherwise.
    """
    if not os.path.exists(MEMORY_PATH):
        logger.info("🧠 sakz_memory: no backup found at %s — clean start", MEMORY_PATH)
        return False

    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
    except Exception as e:
        logger.error("🧠 sakz_memory: could not read backup — %s", e)
        return False

    version = snapshot.get("_meta", {}).get("version", 1)
    created = snapshot.get("_meta", {}).get("created_at", "unknown")
    logger.info("🧠 sakz_memory: restoring backup v%s from %s", version, created)

    # We need the DB tables to exist first — call db_init from sakz_bot if needed.
    # But since restore_on_startup() is called BEFORE db_init(), we ensure tables
    # exist ourselves by running a minimal CREATE IF NOT EXISTS block.
    _ensure_tables_exist()

    conn = _db()
    restored = {}

    try:
        # ── tracked_trades ────────────────────────────────────────────────────
        trades = snapshot.get("tracked_trades", [])
        n = 0
        for row in trades:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO tracked_trades "
                    "(trade_id, chat_id, signal_json, entry_price, start_time, interval_min) "
                    "VALUES (?,?,?,?,?,?)",
                    (row["trade_id"], row["chat_id"], row["signal_json"],
                     row["entry_price"], row["start_time"], row["interval_min"])
                )
                n += conn.execute(
                    "SELECT changes() AS c"
                ).fetchone()["c"]
            except Exception as e:
                logger.debug("tracked_trades restore skip: %s", e)
        conn.commit()
        restored["tracked_trades"] = n

        # ── user_alerts ───────────────────────────────────────────────────────
        alerts = snapshot.get("user_alerts", [])
        n = 0
        for row in alerts:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO user_alerts (chat_id, symbol, min_conf, created_at) "
                    "VALUES (?,?,?,?)",
                    (row["chat_id"], row["symbol"],
                     row.get("min_conf", 7), row.get("created_at", _now()))
                )
                n += conn.execute("SELECT changes() AS c").fetchone()["c"]
            except Exception as e:
                logger.debug("user_alerts restore skip: %s", e)
        conn.commit()
        restored["user_alerts"] = n

        # ── price_level_alerts ────────────────────────────────────────────────
        palerts = snapshot.get("price_level_alerts", [])
        n = 0
        for row in palerts:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO price_level_alerts "
                    "(id, chat_id, symbol, exchange, target, direction, created_at, triggered) "
                    "VALUES (?,?,?,?,?,?,?,0)",
                    (row["id"], row["chat_id"], row["symbol"],
                     row.get("exchange", "BYBIT"), row["target"],
                     row["direction"], row.get("created_at", _now()))
                )
                n += conn.execute("SELECT changes() AS c").fetchone()["c"]
            except Exception as e:
                logger.debug("price_level_alerts restore skip: %s", e)
        conn.commit()
        restored["price_level_alerts"] = n

        # ── watchlist ─────────────────────────────────────────────────────────
        watchlist = snapshot.get("watchlist", [])
        n = 0
        for row in watchlist:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO watchlist (chat_id, symbol, min_conf, created_at) "
                    "VALUES (?,?,?,?)",
                    (row["chat_id"], row["symbol"],
                     row.get("min_conf", 6), row.get("created_at", _now()))
                )
                n += conn.execute("SELECT changes() AS c").fetchone()["c"]
            except Exception as e:
                logger.debug("watchlist restore skip: %s", e)
        conn.commit()
        restored["watchlist"] = n

        # ── broadcast_channels ────────────────────────────────────────────────
        channels = snapshot.get("broadcast_channels", [])
        n = 0
        for row in channels:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO broadcast_channels (chat_id, added_at) "
                    "VALUES (?,?)",
                    (row["chat_id"], row.get("added_at", _now()))
                )
                n += conn.execute("SELECT changes() AS c").fetchone()["c"]
            except Exception as e:
                logger.debug("broadcast_channels restore skip: %s", e)
        conn.commit()
        restored["broadcast_channels"] = n

        # ── snail_unlocked ────────────────────────────────────────────────────
        unlocked = snapshot.get("snail_unlocked", [])
        n = 0
        for row in unlocked:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO snail_unlocked (chat_id, unlocked_at) "
                    "VALUES (?,?)",
                    (row["chat_id"], row.get("unlocked_at", _now()))
                )
                n += conn.execute("SELECT changes() AS c").fetchone()["c"]
            except Exception as e:
                logger.debug("snail_unlocked restore skip: %s", e)
        conn.commit()
        restored["snail_unlocked"] = n

        # ── snail_sessions ────────────────────────────────────────────────────
        sessions = snapshot.get("snail_sessions", [])
        n = 0
        for row in sessions:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO snail_sessions "
                    "(chat_id, activated_at, expires_at, signals_sent, day_wins, active) "
                    "VALUES (?,?,?,?,?,?)",
                    (row["chat_id"], row["activated_at"], row["expires_at"],
                     row.get("signals_sent", 0), row.get("day_wins", "[]"),
                     row.get("active", 1))
                )
                n += 1
            except Exception as e:
                logger.debug("snail_sessions restore skip: %s", e)
        conn.commit()
        restored["snail_sessions"] = n

        # ── snail_signals ─────────────────────────────────────────────────────
        signals = snapshot.get("snail_signals", [])
        n = 0
        for row in signals:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO snail_signals "
                    "(id, chat_id, symbol, exchange, bias, entry_price, t2, "
                    "stop_loss, sent_at, outcome, closed_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (row.get("id"), row["chat_id"], row["symbol"], row["exchange"],
                     row["bias"], row["entry_price"], row["t2"], row["stop_loss"],
                     row["sent_at"], row.get("outcome", "pending"), row.get("closed_at"))
                )
                n += conn.execute("SELECT changes() AS c").fetchone()["c"]
            except Exception as e:
                logger.debug("snail_signals restore skip: %s", e)
        conn.commit()
        restored["snail_signals"] = n

        # ── user_activity ─────────────────────────────────────────────────────
        activity = snapshot.get("user_activity", [])
        n = 0
        for row in activity:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO user_activity "
                    "(chat_id, username, first_name, first_seen, last_seen, command_count) "
                    "VALUES (?,?,?,?,?,?)",
                    (row["chat_id"], row.get("username"), row.get("first_name"),
                     row.get("first_seen", _now()), row.get("last_seen", _now()),
                     row.get("command_count", 1))
                )
                n += conn.execute("SELECT changes() AS c").fetchone()["c"]
            except Exception as e:
                logger.debug("user_activity restore skip: %s", e)
        conn.commit()
        restored["user_activity"] = n

        # ── signal_outcomes (pending) ─────────────────────────────────────────
        outcomes = snapshot.get("signal_outcomes_pending", [])
        n = 0
        for row in outcomes:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO signal_outcomes "
                    "(signal_id, exchange, symbol, bias, confidence, entry_price, "
                    "entry_low, entry_high, stop_loss, t1, t2, t3, scan_time, "
                    "best_target_hit, sl_after_target, entry_confirmed, outcome) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row.get("signal_id", 0), row["exchange"], row["symbol"],
                     row["bias"], row["confidence"], row["entry_price"],
                     row.get("entry_low", 0), row.get("entry_high", 0),
                     row["stop_loss"], row["t1"], row["t2"], row["t3"],
                     row.get("scan_time", _now()), row.get("best_target_hit"),
                     row.get("sl_after_target", 0),
                     row.get("entry_confirmed", -1), "pending")
                )
                n += conn.execute("SELECT changes() AS c").fetchone()["c"]
            except Exception as e:
                logger.debug("signal_outcomes restore skip: %s", e)
        conn.commit()
        restored["signal_outcomes"] = n

        # ── admin_sessions ────────────────────────────────────────────────────
        admin = snapshot.get("admin_sessions", [])
        n = 0
        for row in admin:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO admin_sessions (chat_id, authenticated_at) "
                    "VALUES (?,?)",
                    (row["chat_id"], row.get("authenticated_at", _now()))
                )
                n += conn.execute("SELECT changes() AS c").fetchone()["c"]
            except Exception as e:
                logger.debug("admin_sessions restore skip: %s", e)
        conn.commit()
        restored["admin_sessions"] = n

    finally:
        conn.close()

    total = sum(restored.values())
    logger.info(
        "🧠 sakz_memory: restore complete — %d rows recovered | %s",
        total,
        " | ".join(f"{k}={v}" for k, v in restored.items() if v),
    )
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BACKGROUND AUTO-BACKUP JOB
# ══════════════════════════════════════════════════════════════════════════════

async def _auto_backup_job(context):
    """PTB JobQueue callback — fires every BACKUP_INTERVAL seconds."""
    backup()


def start_background_backup(app) -> None:
    """
    Register the auto-backup job with the PTB Application's JobQueue.
    Call this in main() after app = Application.builder()...build()
    """
    app.job_queue.run_repeating(
        _auto_backup_job,
        interval=BACKUP_INTERVAL,
        first=60,          # first backup 60s after startup
        name="sakz_memory_auto_backup",
    )
    logger.info(
        "🧠 sakz_memory: auto-backup scheduled every %ds → %s",
        BACKUP_INTERVAL, MEMORY_PATH,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ENSURE TABLES EXIST (minimal bootstrap)
# Mirrors the CREATE TABLE IF NOT EXISTS blocks from sakz_bot.py db_init().
# Called by restore_on_startup() so we can restore BEFORE the bot's own
# db_init() runs. Safe to call multiple times — all statements use IF NOT EXISTS.
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_tables_exist():
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tracked_trades (
            trade_id     TEXT PRIMARY KEY,
            chat_id      INTEGER NOT NULL,
            signal_json  TEXT NOT NULL,
            entry_price  REAL NOT NULL,
            start_time   TEXT NOT NULL,
            interval_min INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_alerts (
            chat_id  INTEGER NOT NULL,
            symbol   TEXT NOT NULL,
            min_conf INTEGER NOT NULL DEFAULT 7,
            created_at TEXT NOT NULL,
            PRIMARY KEY (chat_id, symbol)
        );
        CREATE TABLE IF NOT EXISTS price_level_alerts (
            id         TEXT PRIMARY KEY,
            chat_id    INTEGER NOT NULL,
            symbol     TEXT NOT NULL,
            exchange   TEXT NOT NULL DEFAULT 'BYBIT',
            target     REAL NOT NULL,
            direction  TEXT NOT NULL,
            created_at TEXT NOT NULL,
            triggered  INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS watchlist (
            chat_id    INTEGER NOT NULL,
            symbol     TEXT NOT NULL,
            min_conf   INTEGER NOT NULL DEFAULT 6,
            created_at TEXT NOT NULL,
            PRIMARY KEY (chat_id, symbol)
        );
        CREATE TABLE IF NOT EXISTS broadcast_channels (
            chat_id  INTEGER PRIMARY KEY,
            added_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snail_unlocked (
            chat_id     INTEGER PRIMARY KEY,
            unlocked_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snail_sessions (
            chat_id      INTEGER PRIMARY KEY,
            activated_at TEXT NOT NULL,
            expires_at   TEXT NOT NULL,
            signals_sent INTEGER NOT NULL DEFAULT 0,
            day_wins     TEXT NOT NULL DEFAULT '[]',
            active       INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS snail_signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     INTEGER NOT NULL,
            symbol      TEXT NOT NULL,
            exchange    TEXT NOT NULL,
            bias        TEXT NOT NULL,
            entry_price REAL NOT NULL,
            t2          REAL NOT NULL,
            stop_loss   REAL NOT NULL,
            sent_at     TEXT NOT NULL,
            outcome     TEXT DEFAULT 'pending',
            closed_at   TEXT
        );
        CREATE TABLE IF NOT EXISTS user_activity (
            chat_id       INTEGER PRIMARY KEY,
            username      TEXT,
            first_name    TEXT,
            first_seen    TEXT NOT NULL,
            last_seen     TEXT NOT NULL,
            command_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS signal_outcomes (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id        INTEGER NOT NULL,
            exchange         TEXT NOT NULL,
            symbol           TEXT NOT NULL,
            bias             TEXT NOT NULL,
            confidence       INTEGER NOT NULL,
            entry_price      REAL NOT NULL,
            entry_low        REAL NOT NULL DEFAULT 0,
            entry_high       REAL NOT NULL DEFAULT 0,
            stop_loss        REAL NOT NULL,
            t1               REAL NOT NULL,
            t2               REAL NOT NULL,
            t3               REAL NOT NULL,
            scan_time        TEXT NOT NULL,
            check_4h         TEXT,
            check_8h         TEXT,
            check_24h        TEXT,
            check_48h        TEXT,
            outcome          TEXT DEFAULT 'pending',
            best_target_hit  TEXT DEFAULT NULL,
            sl_after_target  INTEGER DEFAULT 0,
            entry_confirmed  INTEGER NOT NULL DEFAULT -1
        );
        CREATE TABLE IF NOT EXISTS admin_sessions (
            chat_id          INTEGER PRIMARY KEY,
            authenticated_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MANUAL BACKUP / RESTORE CLI
# Run this file directly to force a backup or restore:
#   python sakz_memory.py backup
#   python sakz_memory.py restore
#   python sakz_memory.py status
# ══════════════════════════════════════════════════════════════════════════════

def _cli_status():
    if not os.path.exists(MEMORY_PATH):
        print(f"❌ No backup found at: {MEMORY_PATH}")
        return
    with open(MEMORY_PATH, "r") as f:
        snap = json.load(f)
    meta = snap.get("_meta", {})
    print(f"\n🧠 sakz_memory backup status")
    print(f"   File    : {MEMORY_PATH}")
    print(f"   Version : {meta.get('version', '?')}")
    print(f"   Saved   : {meta.get('created_at', '?')}")
    print()
    tables = [k for k in snap if k != "_meta"]
    for t in tables:
        count = len(snap[t]) if isinstance(snap[t], list) else "?"
        emoji = "✅" if count else "⬜"
        print(f"   {emoji} {t:<35} {count} rows")
    print()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "backup":
        ok = backup()
        print("✅ Backup complete." if ok else "❌ Backup failed — check logs.")
    elif cmd == "restore":
        ok = restore_on_startup()
        print("✅ Restore complete." if ok else "⚠️  No backup found or restore failed.")
    elif cmd == "status":
        _cli_status()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python sakz_memory.py [backup|restore|status]")
