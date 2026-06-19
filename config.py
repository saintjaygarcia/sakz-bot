"""config.py - Centralised configuration for the sakz bot.

Single source of truth for database/Turso settings and HTTP/exchange tuning
constants. Loads environment variables once so every module sees consistent
values. The database constants below were moved verbatim from sakz_bot.py.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# --- Database / Turso ---
# Accept both our names and the standard Turso names that Railway/Turso set
# by default (TURSO_DATABASE_URL / TURSO_AUTH_TOKEN), so existing setups work.
TURSO_URL   = os.environ.get("TURSO_URL", "")   or os.environ.get("TURSO_DATABASE_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_TOKEN", "") or os.environ.get("TURSO_AUTH_TOKEN", "")
_USE_TURSO  = bool(TURSO_URL and TURSO_TOKEN)
DB_PATH = os.path.abspath(os.environ.get("SAKZ_DB_PATH", "sakz_data.db"))
ACTIVE_WINDOW_MIN = 30

# --- HTTP / exchange tuning ---
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache',
}
HTTP_TIMEOUT = 15  # default request timeout (seconds)

# MEXC contract (futures) base hosts. Tried in order so a single blocked/slow
# host can't take scanning down. First reachable host wins for that call.
MEXC_CONTRACT_HOSTS = [
    "https://contract.mexc.com",
    "https://futures.mexc.com",
]
# HTTP retry tuning for the shared session (handles transient 403/429/5xx blocks).
HTTP_MAX_RETRIES = 3
HTTP_BACKOFF = 0.6

# Mid-tier (rank 51-200) minimum 24h volume per venue.
BYBIT_MID_MIN_VOL   = 500_000
MEXC_MID_MIN_VOL    = 250_000
BINANCE_MID_MIN_VOL = 2_000_000

# Top-tier minimum 24h volume per venue (mirrors values baked into fetchers).
BYBIT_TOP_MIN_VOL   = 1_000_000
MEXC_TOP_MIN_VOL    = 500_000
BINANCE_TOP_MIN_VOL = 5_000_000
