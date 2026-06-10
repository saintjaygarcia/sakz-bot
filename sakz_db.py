"""sakz_db.py - Data-access layer extracted from sakz_bot.py.

Connection helper, schema init and all 73 db_* functions. Self-contained
(standard library only). Behaviour identical to the original in-line code.
"""
import os
import json
import uuid
import sqlite3
import logging
from datetime import datetime, timedelta

# NOTE: dotenv is loaded once in config.py (single source of truth); importing
# from config below is sufficient, so no redundant load_dotenv() call here.
from config import (  # centralised configuration (single source of truth)
    TURSO_URL, TURSO_TOKEN, _USE_TURSO, DB_PATH, ACTIVE_WINDOW_MIN,
)
from sakz_errors import SakzDBError  # typed DB failure (vs silent empty result)

logger = logging.getLogger(__name__)

# -- Turso / libsql support (moved verbatim from sakz_bot.py) --

if _USE_TURSO:
    try:
        import libsql_experimental as libsql
    except ImportError:
        libsql     = None
        _USE_TURSO = False


# FIX M1 — libsql_experimental (Turso) ignores conn.row_factory, so fetched rows
# come back as plain tuples and every row['col'] access (400+ across the codebase)
# would raise. These thin wrappers convert rows to a dict-like object supporting
# BOTH row['col'] and row[0], with zero changes to call sites. Only the Turso
# path is wrapped; the local-sqlite path keeps native sqlite3.Row.
class _DictRow:
    __slots__ = ("_map", "_vals", "_norm")
    @staticmethod
    def _norm_name(c):
        # libsql/Turso can report column names table-qualified
        # ("price_history.key"), quoted/bracketed ('"key"', '`key`', '[key]')
        # or padded — especially for reserved words like `key`. Normalise so
        # row['key'] still resolves no matter how the backend names the column.
        if not isinstance(c, str):
            return None
        n = c.strip()
        if "." in n:
            n = n.split(".")[-1]
        n = n.strip().strip('"').strip("`")
        if n.startswith("[") and n.endswith("]"):
            n = n[1:-1]
        return n.strip()
    def __init__(self, cols, vals):
        self._vals = tuple(vals)
        m = {}
        norm = {}
        for i, c in enumerate(cols):
            v = self._vals[i] if i < len(self._vals) else None
            m[c] = v
            n = self._norm_name(c)
            if n and n not in norm:
                norm[n] = v
        self._map = m
        self._norm = norm
    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._vals[key]
        if key in self._map:
            return self._map[key]
        if key in self._norm:
            return self._norm[key]
        n = self._norm_name(key)
        if n is not None and n in self._norm:
            return self._norm[n]
        raise KeyError(key)
    def get(self, key, default=None):
        if key in self._map:
            return self._map[key]
        if key in self._norm:
            return self._norm[key]
        n = self._norm_name(key)
        if n is not None and n in self._norm:
            return self._norm[n]
        return default
    def keys(self):
        return list(self._map.keys())
    def __contains__(self, key):
        if key in self._map or key in self._norm:
            return True
        n = self._norm_name(key)
        return n is not None and n in self._norm
    def __iter__(self):
        return iter(self._vals)
    def __len__(self):
        return len(self._vals)
    def __repr__(self):
        return "_DictRow(%r)" % (self._map,)


class _CursorWrapper:
    def __init__(self, cur):
        self._cur = cur
    def _cols(self):
        desc = getattr(self._cur, "description", None)
        if not desc:
            return []
        cols = []
        for d in desc:
            # sqlite3 returns 7-tuples (name, None, ...) so the name is d[0].
            # libsql_experimental (Turso) returns the column name as a plain
            # STRING, in which case d[0] would wrongly be just the first letter.
            # Support both shapes so row['col'] works on either backend.
            if isinstance(d, str):
                cols.append(d)
            else:
                try:
                    cols.append(d[0])
                except (TypeError, IndexError):
                    cols.append(str(d))
        return cols
    def execute(self, *a, **k):
        self._cur.execute(*a, **k)
        return self
    def executemany(self, *a, **k):
        self._cur.executemany(*a, **k)
        return self
    def executescript(self, *a, **k):
        return self._cur.executescript(*a, **k)
    def fetchone(self):
        r = self._cur.fetchone()
        return None if r is None else _DictRow(self._cols(), r)
    def fetchall(self):
        cols = self._cols()
        return [_DictRow(cols, r) for r in self._cur.fetchall()]
    def fetchmany(self, *a, **k):
        cols = self._cols()
        return [_DictRow(cols, r) for r in self._cur.fetchmany(*a, **k)]
    def __iter__(self):
        cols = self._cols()
        for r in self._cur:
            yield _DictRow(cols, r)
    def __getattr__(self, name):
        return getattr(self._cur, name)


class _ConnWrapper:
    def __init__(self, conn):
        self._conn = conn
    def execute(self, *a, **k):
        return _CursorWrapper(self._conn.execute(*a, **k))
    def executescript(self, *a, **k):
        return self._conn.executescript(*a, **k)
    def cursor(self):
        return _CursorWrapper(self._conn.cursor())
    def __enter__(self):
        self._conn.__enter__()
        return self
    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)
    def __getattr__(self, name):
        return getattr(self._conn, name)


def db_connect():
    try:
        if _USE_TURSO and libsql:
            conn = libsql.connect(
                database=TURSO_URL,
                auth_token=TURSO_TOKEN,
            )
            try:
                conn.row_factory = sqlite3.Row  # harmless if libsql ever honors it
            except Exception:
                pass
            return _ConnWrapper(conn)   # FIX M1 — dict-row adapter for Turso path
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except SakzDBError:
        raise
    except Exception as e:
        # A dropped Turso/SQLite connection must NOT masquerade as "no data".
        backend = "turso" if (_USE_TURSO and libsql) else "sqlite"
        logger.error("db_connect failed (%s backend): %s", backend, e)
        raise SakzDBError(f"database connection failed: {e}") from e

def db_init():
    conn = db_connect()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS scan_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_time   TEXT NOT NULL,
            exchange    TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            bias        TEXT NOT NULL,
            confidence  INTEGER NOT NULL,
            score       INTEGER NOT NULL,
            data_json   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS price_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            key         TEXT NOT NULL,
            exchange    TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            ts          TEXT NOT NULL,
            price       REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS signal_outcomes (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id      INTEGER NOT NULL,
            exchange       TEXT NOT NULL,
            symbol         TEXT NOT NULL,
            bias           TEXT NOT NULL,
            confidence     INTEGER NOT NULL,
            entry_price    REAL NOT NULL,
            entry_low      REAL NOT NULL DEFAULT 0,
            entry_high     REAL NOT NULL DEFAULT 0,
            stop_loss      REAL NOT NULL,
            t1             REAL NOT NULL,
            t2             REAL NOT NULL,
            t3             REAL NOT NULL,
            scan_time      TEXT NOT NULL,
            check_4h       TEXT,
            check_8h       TEXT,
            check_24h      TEXT,
            check_48h      TEXT,
            outcome        TEXT DEFAULT 'pending',
            best_target_hit TEXT DEFAULT NULL,
            sl_after_target INTEGER DEFAULT 0,
            entry_confirmed INTEGER NOT NULL DEFAULT -1
        );

        -- FIX A1/A2: migrate existing DBs that lack the new columns
        -- SQLite ignores "duplicate column" errors so this is safe to run every boot
        -- (errors are swallowed by executescript's implicit transaction handling)
        ;

        CREATE TABLE IF NOT EXISTS user_alerts (
            chat_id     INTEGER NOT NULL,
            symbol      TEXT NOT NULL,
            min_conf    INTEGER NOT NULL DEFAULT 7,
            created_at  TEXT NOT NULL,
            PRIMARY KEY (chat_id, symbol)
        );

        CREATE TABLE IF NOT EXISTS user_tracking (
            chat_id      INTEGER PRIMARY KEY,
            signal_json  TEXT NOT NULL,
            entry_price  REAL NOT NULL,
            start_time   TEXT NOT NULL,
            interval_min INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS watchlist (
            chat_id     INTEGER NOT NULL,
            symbol      TEXT NOT NULL,
            min_conf    INTEGER NOT NULL DEFAULT 6,
            created_at  TEXT NOT NULL,
            PRIMARY KEY (chat_id, symbol)
        );

        CREATE TABLE IF NOT EXISTS broadcast_channels (
            chat_id     INTEGER PRIMARY KEY,
            added_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS btc_price_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            price       REAL NOT NULL,
            ts          TEXT NOT NULL
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
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id      INTEGER NOT NULL,
            symbol       TEXT NOT NULL,
            exchange     TEXT NOT NULL,
            bias         TEXT NOT NULL,
            entry_price  REAL NOT NULL,
            t2           REAL NOT NULL,
            stop_loss    REAL NOT NULL,
            sent_at      TEXT NOT NULL,
            outcome      TEXT DEFAULT 'pending',
            closed_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS user_activity (
            chat_id       INTEGER PRIMARY KEY,
            username      TEXT,
            first_name    TEXT,
            first_seen    TEXT NOT NULL,
            last_seen     TEXT NOT NULL,
            command_count INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS admin_sessions (
            chat_id          INTEGER PRIMARY KEY,
            authenticated_at TEXT NOT NULL
        );

        -- FIX #PERSIST-BIAS — flip-cooldown state survives restarts
        CREATE TABLE IF NOT EXISTS signal_bias (
            symbol      TEXT PRIMARY KEY,
            bias        TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        );

        -- PHASE 1 EXECUTION LAYER - per-user encrypted Bybit API keys.
        -- Secrets are stored ENCRYPTED (Fernet); plaintext never touches the DB.
        CREATE TABLE IF NOT EXISTS user_api_keys (
            chat_id        INTEGER PRIMARY KEY,
            api_key_enc    TEXT NOT NULL,
            api_secret_enc TEXT NOT NULL,
            testnet        INTEGER NOT NULL DEFAULT 1,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL
        );

        -- FIX #PERSIST-CACHE — signal card cache survives restarts
        -- Only cards written within the last 24h are restored (older ones are stale)
        CREATE TABLE IF NOT EXISTS signal_card_cache (
            cache_key   TEXT PRIMARY KEY,
            data_json   TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        );

        -- FEATURE: Multi-trade tracking — replaces single-row user_tracking per user
        CREATE TABLE IF NOT EXISTS tracked_trades (
            trade_id    TEXT PRIMARY KEY,
            chat_id     INTEGER NOT NULL,
            signal_json TEXT NOT NULL,
            entry_price REAL NOT NULL,
            start_time  TEXT NOT NULL,
            interval_min INTEGER NOT NULL
        );

        -- FEATURE: Price-level alerts — fires when price crosses a target level
        CREATE TABLE IF NOT EXISTS price_level_alerts (
            id          TEXT PRIMARY KEY,
            chat_id     INTEGER NOT NULL,
            symbol      TEXT NOT NULL,
            exchange    TEXT NOT NULL DEFAULT 'BYBIT',
            target      REAL NOT NULL,
            direction   TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            triggered   INTEGER NOT NULL DEFAULT 0
        );

        -- FEATURE: Safe Mode — users who want automatic dying-trend alerts
        -- for ANY signal they receive (not just tracked trades)
        CREATE TABLE IF NOT EXISTS safemode_users (
            chat_id    INTEGER PRIMARY KEY,
            enabled_at TEXT NOT NULL
        );

        -- PERSIST: /autoscan subscriptions survive redeploys / Railway restarts.
        -- tf_pref '' means "always / all timeframes" (maps to None in memory).
        CREATE TABLE IF NOT EXISTS autoscan_subs (
            chat_id    INTEGER PRIMARY KEY,
            tf_pref    TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );

        -- /pro feature: users subscribed to the PRO alert suite
        CREATE TABLE IF NOT EXISTS pro_subscribers (
            chat_id     INTEGER PRIMARY KEY,
            enabled_at  TEXT NOT NULL
        );

        -- /pro feature: per-token sustained uptrend streak tracking
        CREATE TABLE IF NOT EXISTS pro_uptrend_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT NOT NULL,
            exchange      TEXT NOT NULL,
            first_seen    TEXT NOT NULL,
            last_checked  TEXT NOT NULL,
            daily_gains   TEXT NOT NULL DEFAULT '[]',
            alert_sent    INTEGER NOT NULL DEFAULT 0,
            UNIQUE(symbol, exchange)
        );

        -- /pro feature: top-10 gainer persistence history
        CREATE TABLE IF NOT EXISTS pro_gainers_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol       TEXT NOT NULL UNIQUE,
            first_seen   TEXT NOT NULL,
            last_seen    TEXT NOT NULL,
            times_top10  INTEGER NOT NULL DEFAULT 1,
            alert_sent   INTEGER NOT NULL DEFAULT 0
        );

        -- /pro feature: detected manipulation / scam pump log
        CREATE TABLE IF NOT EXISTS pro_manip_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol       TEXT NOT NULL,
            exchange     TEXT NOT NULL,
            detected_at  TEXT NOT NULL,
            manip_score  INTEGER NOT NULL,
            reasons_json TEXT NOT NULL DEFAULT '[]',
            alert_sent   INTEGER NOT NULL DEFAULT 0,
            UNIQUE(symbol, exchange)
        );
    """)
    conn.commit()

    # FIX A1/A2/P3/P5 — migrate existing signal_outcomes tables that predate these columns
    # ALTER TABLE IF NOT EXISTS col is not supported in older SQLite, so we try/except each
    for _col, _def in [
        ('best_target_hit',  'TEXT DEFAULT NULL'),
        ('sl_after_target',  'INTEGER DEFAULT 0'),
        # P5 — entry confirmation: -1=unset/legacy, 0=missed, 1=confirmed
        ('entry_confirmed',  'INTEGER NOT NULL DEFAULT -1'),
        # P5 — store full entry zone so we can check confirmation from candles
        ('entry_low',        'REAL NOT NULL DEFAULT 0'),
        ('entry_high',       'REAL NOT NULL DEFAULT 0'),
    ]:
        try:
            conn.execute(f"ALTER TABLE signal_outcomes ADD COLUMN {_col} {_def}")
            conn.commit()
            logger.info("DB migration: added column signal_outcomes.%s", _col)
        except Exception:
            pass  # column already exists — safe to ignore

    conn.close()
    logger.info("Database initialised at %s", DB_PATH)

def db_save_scan(results):
    if not results:
        return
    conn = db_connect()
    c    = conn.cursor()
    now  = datetime.now().isoformat()
    # keep only last 500 signals total to avoid bloat
    c.execute("DELETE FROM scan_results WHERE id NOT IN "
              "(SELECT id FROM scan_results ORDER BY id DESC LIMIT 500)")
    for r in results:
        data = {k: v for k, v in r.items()
                if k not in ('scan_time', 'leverage') and not callable(v)}
        if 'leverage' in r and r['leverage']:
            data['leverage'] = r['leverage']
        try:
            data['scan_time_str'] = r['scan_time'].isoformat() if isinstance(r.get('scan_time'), datetime) else now
            c.execute(
                "INSERT INTO scan_results (scan_time,exchange,symbol,bias,confidence,score,data_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (now, r['exchange'], r['symbol'], r['bias'],
                 r['confidence'], r['score'], json.dumps(data, default=str))
            )
        except Exception as e:
            logger.warning("db save scan_result failed for %s: %s", r.get('symbol', '?'), e)
    conn.commit()
    conn.close()

def db_load_last_scan():
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT MAX(scan_time) as mt FROM scan_results")
    row = c.fetchone()
    if not row or not row['mt']:
        conn.close()
        return [], None
    last_ts = row['mt']
    c.execute("SELECT data_json FROM scan_results WHERE scan_time=? ORDER BY confidence DESC, score DESC",
              (last_ts,))
    rows    = c.fetchall()
    conn.close()
    results = []
    for r in rows:
        try:
            d = json.loads(r['data_json'])
            if 'scan_time_str' in d:
                d['scan_time'] = datetime.fromisoformat(d['scan_time_str'])
            results.append(d)
        except Exception as e:
            logger.debug("skipping unparseable scan_result row: %s", e)
    scan_time = datetime.fromisoformat(last_ts) if results else None
    return results, scan_time

def db_find_signals_by_symbol(norm_query, limit=40):
    """
    Find all persisted scans whose symbol matches norm_query (a normalised
    base or full symbol, e.g. 'BTC' or 'BTCUSDT'), newest first. Lets users pull
    a PnL for any past call that was ever scanned, even reversed/expired ones.
    Returns a list of signal dicts with scan_time parsed to datetime.
    """
    try:
        conn = db_connect()
        c    = conn.cursor()
        pattern = (norm_query or '').upper() + '%'
        c.execute(
            "SELECT data_json FROM scan_results "
            "WHERE REPLACE(REPLACE(UPPER(symbol),'/',''),'_','') LIKE ? "
            "ORDER BY scan_time DESC LIMIT ?",
            (pattern, limit)
        )
        rows = c.fetchall()
        conn.close()
    except Exception as e:
        logger.warning("db_find_signals_by_symbol error: %s", e)
        return []
    out = []
    for r in rows:
        try:
            d = json.loads(r['data_json'])
            if 'scan_time_str' in d:
                d['scan_time'] = datetime.fromisoformat(d['scan_time_str'])
            out.append(d)
        except Exception as e:
            logger.debug("skipping unparseable scan_result row: %s", e)
    return out

def db_append_price_history(key, exchange, symbol, price):
    conn = db_connect()
    c    = conn.cursor()
    now  = datetime.now().isoformat()
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    c.execute("DELETE FROM price_history WHERE key=? AND ts < ?", (key, cutoff))
    c.execute("INSERT INTO price_history (key,exchange,symbol,ts,price) VALUES (?,?,?,?,?)",
              (key, exchange, symbol, now, price))
    conn.commit()
    conn.close()

def db_load_price_history():
    conn = db_connect()
    c    = conn.cursor()
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    c.execute("DELETE FROM price_history WHERE ts < ?", (cutoff,))
    conn.commit()
    c.execute("SELECT key, exchange, symbol, ts, price FROM price_history ORDER BY ts ASC")
    rows = c.fetchall()
    conn.close()
    history = {}
    # Positional access (SELECT key, exchange, symbol, ts, price = 0..4). The
    # boot-critical restore must never depend on how the backend reports the
    # reserved-word column name `key` in cursor.description.
    for r in rows:
        k        = r[0]
        exchange = r[1]
        symbol   = r[2]
        ts       = r[3]
        price    = r[4]
        if k not in history:
            history[k] = []
        history[k].append({
            'time':     datetime.fromisoformat(ts),
            'price':    price,
            'exchange': exchange,
            'symbol':   symbol
        })
    return history

def db_save_signal_bias(symbol: str, bias: str):
    """Persist the last-fired bias for a symbol so flip-cooldown survives restarts."""
    conn = db_connect()
    conn.execute(
        "INSERT OR REPLACE INTO signal_bias (symbol, bias, recorded_at) VALUES (?,?,?)",
        (symbol, bias, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def db_load_signal_bias() -> dict:
    """
    Restore _last_signal_bias from DB.
    Only entries recorded within the flip-cooldown window (8 h) are meaningful;
    older ones are discarded so they don't permanently block direction changes.
    """
    cutoff = (datetime.now() - timedelta(hours=8)).isoformat()
    conn   = db_connect()
    rows   = conn.execute(
        "SELECT symbol, bias, recorded_at FROM signal_bias WHERE recorded_at >= ?",
        (cutoff,)
    ).fetchall()
    conn.close()
    return {
        r["symbol"]: {"bias": r["bias"], "time": datetime.fromisoformat(r["recorded_at"])}
        for r in rows
    }

def db_save_card_cache(key: str, entry: dict):
    """Persist a signal card cache entry so Details/Back buttons survive restarts."""
    try:
        conn = db_connect()
        conn.execute(
            "INSERT OR REPLACE INTO signal_card_cache (cache_key, data_json, recorded_at) VALUES (?,?,?)",
            (key, json.dumps(entry, default=str), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("db_save_card_cache error: %s", e)

def db_load_card_cache() -> dict:
    """
    Restore _signal_card_cache from DB.
    Only entries from the last 24 h are loaded — older signal cards are stale
    (the inline keyboard message they belong to is gone anyway).
    """
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    conn   = db_connect()
    conn.execute("DELETE FROM signal_card_cache WHERE recorded_at < ?", (cutoff,))
    conn.commit()
    rows = conn.execute(
        "SELECT cache_key, data_json FROM signal_card_cache WHERE recorded_at >= ?",
        (cutoff,)
    ).fetchall()
    conn.close()
    result = {}
    for r in rows:
        try:
            result[r["cache_key"]] = json.loads(r["data_json"])
        except Exception as e:
            logger.debug("skipping unparseable cache row: %s", e)
    return result

def db_register_outcome(signal, scan_row_id=None):
    """
    Store a new signal for outcome tracking.
    FIX #10 — use entry_high as the registered entry price.
    FIX A6  — DEDUPLICATION: skip insert if the same exchange+symbol+bias
               already has a pending outcome row. One live trade per pair,
               not one per scan cycle. This prevents repeated 4h auto-scans
               from inflating win/loss counts with duplicate signals.
    FIX P5  — store entry_low/entry_high and set entry_confirmed=0.
               Confirmed only once price actually trades within that zone.
    """
    if signal['bias'] == 'LONG':
        tracked_entry = signal.get('entry_high', signal.get('price', 0))
    else:
        tracked_entry = signal.get('entry_low', signal.get('price', 0))

    entry_low  = signal.get('entry_low',  tracked_entry)
    entry_high = signal.get('entry_high', tracked_entry)

    conn = db_connect()
    c    = conn.cursor()

    # FIX A6 — check for an existing pending row for this exact setup
    c.execute(
        "SELECT id FROM signal_outcomes "
        "WHERE exchange=? AND symbol=? AND bias=? AND outcome='pending'",
        (signal['exchange'], signal['symbol'], signal['bias'])
    )
    existing = c.fetchone()
    if existing:
        # Already tracking this signal — skip to avoid duplicate stats
        conn.close()
        logger.debug("DEDUP: skipping duplicate pending signal %s %s %s",
                     signal['exchange'], signal['symbol'], signal['bias'])
        return

    c.execute(
        "INSERT INTO signal_outcomes "
        "(signal_id,exchange,symbol,bias,confidence,entry_price,entry_low,entry_high,"
        "stop_loss,t1,t2,t3,scan_time,best_target_hit,sl_after_target,entry_confirmed) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (scan_row_id or 0, signal['exchange'], signal['symbol'], signal['bias'],
         signal['confidence'], tracked_entry, entry_low, entry_high,
         signal['stop_loss'], signal['t1'], signal['t2'], signal['t3'],
         datetime.now().isoformat(), None, 0,
         0)   # entry_confirmed=0: unconfirmed until price trades in the zone
    )
    conn.commit()
    conn.close()

def db_save_user_alert(chat_id, symbol, min_conf):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("INSERT OR REPLACE INTO user_alerts (chat_id,symbol,min_conf,created_at) VALUES (?,?,?,?)",
              (chat_id, symbol.upper(), min_conf, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def db_remove_user_alert(chat_id, symbol):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("DELETE FROM user_alerts WHERE chat_id=? AND symbol=?", (chat_id, symbol.upper()))
    conn.commit()
    conn.close()

def db_get_user_alerts(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT symbol, min_conf FROM user_alerts WHERE chat_id=?", (chat_id,))
    rows = c.fetchall()
    conn.close()
    return [(r['symbol'], r['min_conf']) for r in rows]

def db_get_all_alerts():
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT chat_id, symbol, min_conf FROM user_alerts")
    rows = c.fetchall()
    conn.close()
    return [(r['chat_id'], r['symbol'], r['min_conf']) for r in rows]

def db_add_watch(chat_id, symbol, min_conf=6):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("INSERT OR REPLACE INTO watchlist (chat_id,symbol,min_conf,created_at) VALUES (?,?,?,?)",
              (chat_id, symbol.upper(), min_conf, datetime.now().isoformat()))
    conn.commit(); conn.close()

def db_remove_watch(chat_id, symbol):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE chat_id=? AND symbol=?", (chat_id, symbol.upper()))
    conn.commit(); conn.close()

def db_get_watchlist(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT symbol, min_conf FROM watchlist WHERE chat_id=?", (chat_id,))
    rows = c.fetchall(); conn.close()
    return [(r['symbol'], r['min_conf']) for r in rows]

def db_get_all_watchlist():
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT chat_id, symbol, min_conf FROM watchlist")
    rows = c.fetchall(); conn.close()
    return [(r['chat_id'], r['symbol'], r['min_conf']) for r in rows]

def db_add_broadcast(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("INSERT OR REPLACE INTO broadcast_channels (chat_id, added_at) VALUES (?,?)",
              (chat_id, datetime.now().isoformat()))
    conn.commit(); conn.close()

def db_remove_broadcast(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("DELETE FROM broadcast_channels WHERE chat_id=?", (chat_id,))
    conn.commit(); conn.close()

def db_get_broadcast_channels():
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT chat_id FROM broadcast_channels")
    rows = c.fetchall(); conn.close()
    return [r['chat_id'] for r in rows]

def db_save_btc_price(price):
    conn = db_connect()
    c    = conn.cursor()
    cutoff = (datetime.now() - timedelta(hours=2)).isoformat()
    c.execute("DELETE FROM btc_price_snapshots WHERE ts < ?", (cutoff,))
    c.execute("INSERT INTO btc_price_snapshots (price, ts) VALUES (?,?)",
              (price, datetime.now().isoformat()))
    conn.commit(); conn.close()

def db_get_btc_price_1h_ago():
    """Returns the BTC price from ~1 hour ago, or None."""
    conn = db_connect()
    c    = conn.cursor()
    cutoff = (datetime.now() - timedelta(minutes=75)).isoformat()
    floor  = (datetime.now() - timedelta(minutes=45)).isoformat()
    c.execute("SELECT price FROM btc_price_snapshots WHERE ts BETWEEN ? AND ? ORDER BY ts ASC LIMIT 1",
              (cutoff, floor))
    row = c.fetchone(); conn.close()
    return row['price'] if row else None

def db_save_tracking(chat_id, signal, entry_price, start_time, interval_min):
    conn = db_connect()
    c    = conn.cursor()
    data = {k: v for k, v in signal.items() if not callable(v)}
    c.execute("INSERT OR REPLACE INTO user_tracking (chat_id,signal_json,entry_price,start_time,interval_min) "
              "VALUES (?,?,?,?,?)",
              (chat_id, json.dumps(data, default=str),
               entry_price, start_time.isoformat(), interval_min))
    conn.commit()
    conn.close()

def db_remove_tracking(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("DELETE FROM user_tracking WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()

def db_load_all_tracking():
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT * FROM user_tracking")
    rows = c.fetchall()
    conn.close()
    result = {}
    for r in rows:
        try:
            sig = json.loads(r['signal_json'])
            result[r['chat_id']] = {
                'signal':      sig,
                'entry_price': r['entry_price'],
                'start_time':  datetime.fromisoformat(r['start_time']),
                'interval':    r['interval_min'],
                'job':         None
            }
        except Exception as e:
            logger.debug("skipping unparseable tracked-trade row: %s", e)
    return result

def db_save_trade(chat_id: int, trade_id: str, signal: dict, entry_price: float,
                  start_time: datetime, interval_min: int):
    """Persist a single tracked trade. Multiple trades per chat_id are supported."""
    conn = db_connect()
    data = {k: v for k, v in signal.items() if not callable(v)}
    conn.execute(
        "INSERT OR REPLACE INTO tracked_trades "
        "(trade_id, chat_id, signal_json, entry_price, start_time, interval_min) "
        "VALUES (?,?,?,?,?,?)",
        (trade_id, chat_id, json.dumps(data, default=str),
         entry_price, start_time.isoformat(), interval_min)
    )
    conn.commit()
    conn.close()

def db_remove_trade(trade_id: str):
    """Remove one tracked trade by trade_id."""
    conn = db_connect()
    conn.execute("DELETE FROM tracked_trades WHERE trade_id=?", (trade_id,))
    conn.commit()
    conn.close()

def db_remove_all_user_trades(chat_id: int):
    """Remove all tracked trades for a user."""
    conn = db_connect()
    conn.execute("DELETE FROM tracked_trades WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()

def db_load_all_trades() -> dict:
    """
    Load all tracked trades from DB on startup.
    Returns { chat_id: { trade_id: {signal, entry_price, start_time, interval, job} } }
    """
    conn = db_connect()
    rows = conn.execute("SELECT * FROM tracked_trades ORDER BY chat_id, start_time").fetchall()
    conn.close()
    result: dict = {}
    for r in rows:
        try:
            sig = json.loads(r['signal_json'])
            cid = r['chat_id']
            tid = r['trade_id']
            result.setdefault(cid, {})[tid] = {
                'signal':      sig,
                'entry_price': r['entry_price'],
                'start_time':  datetime.fromisoformat(r['start_time']),
                'interval':    r['interval_min'],
                'job':         None,
                'trade_id':    tid,
            }
        except Exception as e:
            logger.debug("skipping unparseable tracked-trade row (with id): %s", e)
    return result

def db_save_price_alert(chat_id: int, symbol: str, exchange: str,
                        target: float, direction: str) -> str:
    """Persist a price-level alert. Returns the generated alert id."""
    alert_id = uuid.uuid4().hex[:10]
    conn = db_connect()
    conn.execute(
        "INSERT INTO price_level_alerts (id, chat_id, symbol, exchange, target, direction, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (alert_id, chat_id, symbol.upper(), exchange.upper(),
         target, direction, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return alert_id

def db_get_price_alerts(chat_id: int) -> list:
    """All untriggered price alerts for a user."""
    conn = db_connect()
    rows = conn.execute(
        "SELECT * FROM price_level_alerts WHERE chat_id=? AND triggered=0 ORDER BY created_at",
        (chat_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_all_price_alerts() -> list:
    """All untriggered price alerts across all users (for the background checker)."""
    conn = db_connect()
    rows = conn.execute(
        "SELECT * FROM price_level_alerts WHERE triggered=0"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_remove_price_alert(alert_id: str):
    conn = db_connect()
    conn.execute("DELETE FROM price_level_alerts WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()

def db_mark_price_alert_triggered(alert_id: str):
    conn = db_connect()
    conn.execute("UPDATE price_level_alerts SET triggered=1 WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()

def db_remove_price_alerts_for_symbol(chat_id: int, symbol: str):
    conn = db_connect()
    conn.execute(
        "DELETE FROM price_level_alerts WHERE chat_id=? AND symbol=? AND triggered=0",
        (chat_id, symbol.upper())
    )
    conn.commit()
    conn.close()

def db_pro_subscribe(chat_id: int):
    conn = db_connect()
    conn.execute(
        "INSERT OR REPLACE INTO pro_subscribers (chat_id, enabled_at) VALUES (?,?)",
        (chat_id, datetime.now().isoformat())
    )
    conn.commit(); conn.close()

def db_pro_unsubscribe(chat_id: int):
    conn = db_connect()
    conn.execute("DELETE FROM pro_subscribers WHERE chat_id=?", (chat_id,))
    conn.commit(); conn.close()

def db_pro_is_subscribed(chat_id: int) -> bool:
    conn = db_connect()
    row  = conn.execute(
        "SELECT 1 FROM pro_subscribers WHERE chat_id=?", (chat_id,)
    ).fetchone()
    conn.close()
    return row is not None

def db_pro_get_all_subscribers() -> list:
    conn = db_connect()
    rows = conn.execute("SELECT chat_id FROM pro_subscribers").fetchall()
    conn.close()
    return [r["chat_id"] for r in rows]

def db_pro_upsert_uptrend(symbol: str, exchange: str, daily_gains: list):
    now = datetime.now().isoformat()
    conn = db_connect()
    if conn.execute(
        "SELECT id FROM pro_uptrend_log WHERE symbol=? AND exchange=?",
        (symbol, exchange)
    ).fetchone():
        conn.execute(
            "UPDATE pro_uptrend_log SET last_checked=?, daily_gains=? "
            "WHERE symbol=? AND exchange=?",
            (now, json.dumps(daily_gains), symbol, exchange)
        )
    else:
        conn.execute(
            "INSERT INTO pro_uptrend_log "
            "(symbol, exchange, first_seen, last_checked, daily_gains) VALUES (?,?,?,?,?)",
            (symbol, exchange, now, now, json.dumps(daily_gains))
        )
    conn.commit(); conn.close()

def db_pro_mark_uptrend_alerted(symbol: str, exchange: str):
    conn = db_connect()
    conn.execute(
        "UPDATE pro_uptrend_log SET alert_sent=1 WHERE symbol=? AND exchange=?",
        (symbol, exchange)
    )
    conn.commit(); conn.close()

def db_pro_cleanup_uptrends():
    cutoff = (datetime.now() - timedelta(days=4)).isoformat()
    conn   = db_connect()
    conn.execute("DELETE FROM pro_uptrend_log WHERE last_checked < ?", (cutoff,))
    conn.commit(); conn.close()

def db_pro_get_uptrend_rows() -> list:
    conn = db_connect()
    rows = conn.execute(
        "SELECT * FROM pro_uptrend_log ORDER BY last_checked DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_pro_upsert_gainer(symbol: str):
    now = datetime.now().isoformat()
    conn = db_connect()
    if conn.execute(
        "SELECT id FROM pro_gainers_log WHERE symbol=?", (symbol,)
    ).fetchone():
        conn.execute(
            "UPDATE pro_gainers_log SET last_seen=?, times_top10=times_top10+1 WHERE symbol=?",
            (now, symbol)
        )
    else:
        conn.execute(
            "INSERT INTO pro_gainers_log (symbol, first_seen, last_seen) VALUES (?,?,?)",
            (symbol, now, now)
        )
    conn.commit(); conn.close()

def db_pro_mark_gainer_alerted(symbol: str):
    conn = db_connect()
    conn.execute("UPDATE pro_gainers_log SET alert_sent=1 WHERE symbol=?", (symbol,))
    conn.commit(); conn.close()

def db_pro_cleanup_gainers():
    cutoff = (datetime.now() - timedelta(days=4)).isoformat()
    conn   = db_connect()
    conn.execute("DELETE FROM pro_gainers_log WHERE last_seen < ?", (cutoff,))
    conn.commit(); conn.close()

def db_pro_get_gainer_rows() -> list:
    conn = db_connect()
    rows = conn.execute(
        "SELECT * FROM pro_gainers_log ORDER BY last_seen DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_pro_upsert_manip(symbol: str, exchange: str, manip_score: int, reasons: list):
    now = datetime.now().isoformat()
    conn = db_connect()
    if conn.execute(
        "SELECT id FROM pro_manip_log WHERE symbol=? AND exchange=?", (symbol, exchange)
    ).fetchone():
        conn.execute(
            "UPDATE pro_manip_log SET detected_at=?, manip_score=?, "
            "reasons_json=?, alert_sent=0 WHERE symbol=? AND exchange=?",
            (now, manip_score, json.dumps(reasons), symbol, exchange)
        )
    else:
        conn.execute(
            "INSERT INTO pro_manip_log "
            "(symbol, exchange, detected_at, manip_score, reasons_json) VALUES (?,?,?,?,?)",
            (symbol, exchange, now, manip_score, json.dumps(reasons))
        )
    conn.commit(); conn.close()

def db_pro_mark_manip_alerted(symbol: str, exchange: str):
    conn = db_connect()
    conn.execute(
        "UPDATE pro_manip_log SET alert_sent=1 WHERE symbol=? AND exchange=?",
        (symbol, exchange)
    )
    conn.commit(); conn.close()

def db_pro_cleanup_manip():
    """Remove entries older than 24 h so the detector can re-fire daily."""
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    conn   = db_connect()
    conn.execute("DELETE FROM pro_manip_log WHERE detected_at < ?", (cutoff,))
    conn.commit(); conn.close()

def db_pro_get_manip_rows() -> list:
    conn = db_connect()
    rows = conn.execute(
        "SELECT * FROM pro_manip_log ORDER BY detected_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_safemode_load() -> set:
    """Load all safemode-enabled chat_ids from DB on startup."""
    conn = db_connect()
    rows = conn.execute("SELECT chat_id FROM safemode_users").fetchall()
    conn.close()
    return {r['chat_id'] for r in rows}

def db_safemode_enable(chat_id: int):
    conn = db_connect()
    conn.execute(
        "INSERT OR REPLACE INTO safemode_users (chat_id, enabled_at) VALUES (?,?)",
        (chat_id, datetime.now().isoformat())
    )
    conn.commit(); conn.close()

def db_safemode_disable(chat_id: int):
    conn = db_connect()
    conn.execute("DELETE FROM safemode_users WHERE chat_id=?", (chat_id,))
    conn.commit(); conn.close()

def db_autoscan_load() -> dict:
    """Load all /autoscan subscriptions {chat_id: tf_pref} on startup.
    Empty-string tf_pref (meaning 'always / all timeframes') is mapped back to None."""
    conn = db_connect()
    rows = conn.execute("SELECT chat_id, tf_pref FROM autoscan_subs").fetchall()
    conn.close()
    return {r['chat_id']: (r['tf_pref'] or None) for r in rows}

def db_autoscan_set(chat_id: int, tf_pref):
    """Persist (or update) a user's autoscan subscription. tf_pref None == all timeframes."""
    conn = db_connect()
    conn.execute(
        "INSERT OR REPLACE INTO autoscan_subs (chat_id, tf_pref, updated_at) VALUES (?,?,?)",
        (chat_id, tf_pref or '', datetime.now().isoformat())
    )
    conn.commit(); conn.close()

def db_autoscan_remove(chat_id: int):
    conn = db_connect()
    conn.execute("DELETE FROM autoscan_subs WHERE chat_id=?", (chat_id,))
    conn.commit(); conn.close()

def db_snail_unlock(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("INSERT OR IGNORE INTO snail_unlocked (chat_id, unlocked_at) VALUES (?,?)",
              (chat_id, datetime.now().isoformat()))
    conn.commit(); conn.close()

def db_snail_is_unlocked(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT 1 FROM snail_unlocked WHERE chat_id=?", (chat_id,))
    row = c.fetchone(); conn.close()
    return row is not None

def db_snail_start_session(chat_id):
    now     = datetime.now()
    expires = now + timedelta(days=7)
    conn    = db_connect()
    c       = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO snail_sessions
                 (chat_id, activated_at, expires_at, signals_sent, day_wins, active)
                 VALUES (?,?,?,0,'[]',1)""",
              (chat_id, now.isoformat(), expires.isoformat()))
    conn.commit(); conn.close()
    return {'activated_at': now, 'expires_at': expires, 'signals_sent': 0, 'day_wins': []}

def db_snail_get_session(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT * FROM snail_sessions WHERE chat_id=? AND active=1", (chat_id,))
    row  = c.fetchone(); conn.close()
    if not row: return None
    return {
        'activated_at':  datetime.fromisoformat(row['activated_at']),
        'expires_at':    datetime.fromisoformat(row['expires_at']),
        'signals_sent':  row['signals_sent'],
        'day_wins':      json.loads(row['day_wins']),
        'active':        row['active']
    }

def db_snail_end_session(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("UPDATE snail_sessions SET active=0 WHERE chat_id=?", (chat_id,))
    conn.commit(); conn.close()

def db_snail_increment_signals(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("UPDATE snail_sessions SET signals_sent=signals_sent+1 WHERE chat_id=? AND active=1", (chat_id,))
    conn.commit(); conn.close()

def db_snail_save_signal(chat_id, r):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("""INSERT INTO snail_signals
                 (chat_id, symbol, exchange, bias, entry_price, t2, stop_loss, sent_at)
                 VALUES (?,?,?,?,?,?,?,?)""",
              (chat_id, r['symbol'], r['exchange'], r['bias'],
               r['price'], r['t2'], r['stop_loss'], datetime.now().isoformat()))
    conn.commit(); conn.close()

def db_snail_get_pending_signals(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT * FROM snail_signals WHERE chat_id=? AND outcome='pending'", (chat_id,))
    rows = c.fetchall(); conn.close()
    return [dict(r) for r in rows]

def db_snail_update_outcome(sig_id, outcome):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("UPDATE snail_signals SET outcome=?, closed_at=? WHERE id=?",
              (outcome, datetime.now().isoformat(), sig_id))
    conn.commit(); conn.close()

def db_snail_load_all_active():
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT chat_id FROM snail_sessions WHERE active=1")
    rows = c.fetchall(); conn.close()
    return [r['chat_id'] for r in rows]

def db_snail_load_unlocked():
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT chat_id FROM snail_unlocked")
    rows = c.fetchall(); conn.close()
    return {r['chat_id'] for r in rows}

def db_init_user_tracking():
    """Create both user_activity and user_interactions tables if missing."""
    try:
        conn = db_connect()
        c    = conn.cursor()
        # Primary table — used by admin dashboard
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_activity (
                chat_id       INTEGER PRIMARY KEY,
                username      TEXT,
                first_name    TEXT,
                first_seen    TEXT NOT NULL DEFAULT \'\',
                last_seen     TEXT NOT NULL DEFAULT \'\',
                command_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Legacy table — kept for compatibility
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                chat_id            INTEGER PRIMARY KEY,
                first_seen         TEXT NOT NULL,
                last_seen          TEXT NOT NULL,
                interaction_count  INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit(); conn.close()
        logger.info("db_init_user_tracking: tables ready")
    except Exception as e:
        logger.warning("db_init_user_tracking: %s", e)

def db_track_user(chat_id, username, first_name):
    """
    Upsert user activity record.
    Called both from the TypeHandler middleware AND directly at the top
    of every major command handler, so tracking is guaranteed even if
    the middleware fails or is not reached.
    """
    try:
        conn = db_connect()
        c    = conn.cursor()
        now  = datetime.now().isoformat()
        c.execute("""
            INSERT INTO user_activity (chat_id, username, first_name, first_seen, last_seen, command_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(chat_id) DO UPDATE SET
                username      = excluded.username,
                first_name    = excluded.first_name,
                last_seen     = excluded.last_seen,
                command_count = command_count + 1
        """, (chat_id, username or '', first_name or '', now, now))
        conn.commit()
        conn.close()
        logger.debug("db_track_user: tracked chat_id=%s username=%s", chat_id, username)
    except Exception as e:
        logger.error("db_track_user FAILED for chat_id=%s: %s", chat_id, e)

def db_admin_is_authed(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT authenticated_at FROM admin_sessions WHERE chat_id=?", (chat_id,))
    row  = c.fetchone()
    conn.close()
    return row is not None

def db_admin_set_auth(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("INSERT OR REPLACE INTO admin_sessions (chat_id, authenticated_at) VALUES (?,?)",
              (chat_id, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def db_admin_revoke(chat_id):
    conn = db_connect()
    c    = conn.cursor()
    c.execute("DELETE FROM admin_sessions WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()

def db_admin_get_stats():
    """Return all user activity rows and compute totals."""
    conn   = db_connect()
    c      = conn.cursor()
    c.execute("SELECT chat_id, username, first_name, first_seen, last_seen, command_count FROM user_activity ORDER BY last_seen DESC")
    rows   = c.fetchall()
    conn.close()

    total     = len(rows)
    cutoff    = (datetime.now() - timedelta(minutes=ACTIVE_WINDOW_MIN)).isoformat()
    active    = [r for r in rows if r['last_seen'] >= cutoff]
    today_cut = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    new_today = [r for r in rows if r['first_seen'] >= today_cut]
    week_cut  = (datetime.now() - timedelta(days=7)).isoformat()
    active_7d = [r for r in rows if r['last_seen'] >= week_cut]

    return {
        'total':      total,
        'active_now': active,
        'new_today':  new_today,
        'active_7d':  active_7d,
        'all_users':  rows,
    }



# ============================================================================
# PHASE 1 EXECUTION LAYER - per-user encrypted Bybit API key vault.
# Secrets arrive already ENCRYPTED from sakz_execution.encrypt_secret(); this
# layer only persists/retrieves opaque ciphertext - it never sees plaintext.
# rowcount is avoided (the Turso cursor wrapper does not expose it); existence
# is checked with an explicit SELECT to stay backend-agnostic.
# ============================================================================
def db_save_user_keys(chat_id, api_key_enc, api_secret_enc, testnet=True):
    """Upsert a user's encrypted Bybit API key pair (preserves created_at)."""
    conn = db_connect()
    c    = conn.cursor()
    now  = datetime.now().isoformat()
    c.execute("SELECT created_at FROM user_api_keys WHERE chat_id=?", (chat_id,))
    existing = c.fetchone()
    if existing:
        c.execute(
            "UPDATE user_api_keys SET api_key_enc=?, api_secret_enc=?, testnet=?, updated_at=? "
            "WHERE chat_id=?",
            (api_key_enc, api_secret_enc, 1 if testnet else 0, now, chat_id)
        )
    else:
        c.execute(
            "INSERT INTO user_api_keys (chat_id, api_key_enc, api_secret_enc, testnet, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (chat_id, api_key_enc, api_secret_enc, 1 if testnet else 0, now, now)
        )
    conn.commit()
    conn.close()


def db_get_user_keys(chat_id):
    """Return (api_key_enc, api_secret_enc, testnet_bool) or None if not set."""
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT api_key_enc, api_secret_enc, testnet FROM user_api_keys WHERE chat_id=?", (chat_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return (row['api_key_enc'], row['api_secret_enc'], bool(row['testnet']))


def db_delete_user_keys(chat_id):
    """Remove a user's stored API keys. Returns True if a row existed."""
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT 1 FROM user_api_keys WHERE chat_id=?", (chat_id,))
    existed = c.fetchone() is not None
    c.execute("DELETE FROM user_api_keys WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()
    return existed


def db_user_has_keys(chat_id):
    """True if the user has stored API keys."""
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT 1 FROM user_api_keys WHERE chat_id=?", (chat_id,))
    row = c.fetchone()
    conn.close()
    return row is not None
