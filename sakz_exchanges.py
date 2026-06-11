"""sakz_exchanges.py - Raw-requests exchange layer extracted from sakz_bot.py.

Bybit / MEXC / Binance OHLCV, price, funding and symbol-universe fetchers, plus
the BYBIT_AVAILABLE / BINANCE_AVAILABLE runtime flags they mutate via `global`.
Other modules MUST read those flags as attributes (sakz_exchanges.BYBIT_AVAILABLE)
to see live updates. Behaviour identical to the original in-line code.
"""
import os
import logging
import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover - very old urllib3
    from requests.packages.urllib3.util.retry import Retry
import pandas as pd

logger = logging.getLogger(__name__)

from config import (
    HEADERS,
    HTTP_TIMEOUT,
    MEXC_CONTRACT_HOSTS,
    HTTP_MAX_RETRIES,
    HTTP_BACKOFF,
)  # centralised configuration

# Single source of truth for the default request timeout. Lightweight ticker/
# price endpoints below intentionally keep their own shorter timeouts (8/10s).
_TIMEOUT = HTTP_TIMEOUT


# ── CRYPTO-ONLY UNIVERSE FILTER ──────────────────────────────────────────────
# The bot trades crypto perpetuals only. Some venues (notably MEXC) also list
# tokenised equities, metals, oil and FX/index synthetics (e.g. *STOCK*, gold/
# silver, oil, indices). These do NOT behave like crypto, so crypto indicators
# and the BTC-regime gate are meaningless for them. They must never enter the
# scanner. /analyse stays unrestricted and can still inspect them on request.
NON_CRYPTO_SYMBOLS = {
    "XAUTUSDT", "XAUUSDT", "XAGUSDT", "XPTUSDT", "XPDUSDT",
    "UKOILUSDT", "USOILUSDT", "WTIUSDT", "BRENTUSDT", "XBRUSDT", "XTIUSDT", "NGASUSDT",
    "SPXUSDT", "US500USDT", "US30USDT", "US100USDT", "NAS100USDT", "NDXUSDT",
    "GER40USDT", "UK100USDT", "JP225USDT", "HK50USDT", "EU50USDT",
    "EURUSDT", "GBPUSDT", "JPYUSDT", "AUDUSDT", "CHFUSDT", "CADUSDT", "NZDUSDT",
}
NON_CRYPTO_SUBSTRINGS = ("STOCK", "EQUITY")

def is_crypto_symbol(symbol) -> bool:
    """True only for genuine crypto perp symbols.

    Filters tokenised stocks (e.g. MRVLSTOCKUSDT), metals (XAUT/XAG), oil
    (UK/USOIL) and FX/index synthetics so they never reach the crypto scorer."""
    if not symbol:
        return False
    s = str(symbol).upper().replace('_USDT', 'USDT').replace('/', '').strip()
    if s in NON_CRYPTO_SYMBOLS:
        return False
    for pat in NON_CRYPTO_SUBSTRINGS:
        if pat in s:
            return False
    return True


# ── Shared HTTP session with automatic retry/backoff ─────────────────────────
# A single pooled Session is reused for every exchange request. Connection reuse
# (keep-alive) makes repeated scans noticeably faster, and the mounted Retry
# transparently re-attempts transient blocks (403/429) and 5xx errors with
# exponential backoff instead of failing the whole scan on the first hiccup.
def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=HTTP_MAX_RETRIES,
        connect=HTTP_MAX_RETRIES,
        read=HTTP_MAX_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=(403, 408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=64)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = _build_session()



# === MEXC-ONLY MODE ==========================================================
# Bybit and Binance public APIs are geo-blocked on the deployment region
# (e.g. Railway), so this bot runs EXCLUSIVELY on MEXC's public market-data API.
#
# IMPORTANT: this bot uses NO exchange API keys/secrets anywhere. Every endpoint
# it calls (MEXC, and the now-disabled Bybit/Binance) is a PUBLIC market-data
# endpoint that needs no authentication. You therefore never have to set any
# exchange API variable on Railway (or anywhere). The only env vars the bot
# reads are TELEGRAM_TOKEN and the optional TURSO_URL / TURSO_TOKEN.
#
# The switch below is controlled by the MEXC_ONLY env var (default OFF). While on:
#   * Bybit/Binance availability is forced False (no startup network probe), and
#   * every Bybit/Binance fetcher short-circuits to an empty result, so NO
#     request is ever sent to those venues regardless of which call site runs.
# The existing "if BYBIT_AVAILABLE ... else MEXC" fallbacks throughout the bot
# then route 100% of market-data traffic to MEXC automatically.
# Trial mode: probe Bybit FIRST, fall back to MEXC. Controlled by the MEXC_ONLY
# env var (default OFF, now that the host can sit in a Bybit-reachable region).
# Set MEXC_ONLY=true on the host to force the old MEXC-only behaviour.
MEXC_ONLY = os.environ.get("MEXC_ONLY", "false").strip().lower() in ("1", "true", "yes", "on")

BYBIT_AVAILABLE = False if MEXC_ONLY else None    # None=unchecked, True=ok, False=blocked
BINANCE_AVAILABLE = False if MEXC_ONLY else None  # None=unchecked, True=ok, False=blocked

def bybit_get_top_symbols(limit=50):
    if MEXC_ONLY:
        return []
    try:
        r    = SESSION.get("https://api.bybit.com/v5/market/tickers?category=linear",
                            headers=HEADERS, timeout=_TIMEOUT)
        data = r.json()
        if data.get('retCode') != 0:
            return []
        vol_list = []
        for t in data['result']['list']:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT'):
                continue
            try:
                vol   = float(t.get('turnover24h', 0) or 0)
                price = float(t.get('lastPrice', 0) or 0)
                if vol > 1_000_000 and price > 0:
                    vol_list.append({'symbol': sym, 'volume': vol, 'price': price})
            except Exception:
                continue
        vol_list.sort(key=lambda x: x['volume'], reverse=True)
        return [i['symbol'] for i in vol_list[:limit]]
    except Exception:
        return []

def bybit_get_mid_symbols(rank_from=51, rank_to=200, min_vol=500_000):
    """
    CEILING #6 — Mid-tier universe: ranks 51–200 by 24h volume.
    These coins are liquid enough to trade on perps but get less algorithmic
    attention than the top 50, so pricing inefficiency is higher.
    min_vol filters out dead pairs with insufficient liquidity.
    Returns symbol list sorted by volume descending (rank_from first).
    """
    if MEXC_ONLY:
        return []
    try:
        r    = SESSION.get("https://api.bybit.com/v5/market/tickers?category=linear",
                            headers=HEADERS, timeout=_TIMEOUT)
        data = r.json()
        if data.get('retCode') != 0:
            return []
        vol_list = []
        for t in data['result']['list']:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT'):
                continue
            try:
                vol   = float(t.get('turnover24h', 0) or 0)
                price = float(t.get('lastPrice', 0) or 0)
                if vol >= min_vol and price > 0:
                    vol_list.append({'symbol': sym, 'volume': vol})
            except Exception:
                continue
        vol_list.sort(key=lambda x: x['volume'], reverse=True)
        # Slice to the requested rank window (1-indexed)
        mid_slice = vol_list[rank_from - 1 : rank_to]
        return [i['symbol'] for i in mid_slice]
    except Exception:
        return []

def bybit_check_available():
    global BYBIT_AVAILABLE
    if MEXC_ONLY:
        BYBIT_AVAILABLE = False
        return False
    try:
        r = SESSION.get("https://api.bybit.com/v5/market/time",
                         headers=HEADERS, timeout=8)
        if r.status_code == 200 and r.json().get('retCode') == 0:
            BYBIT_AVAILABLE = True
            logger.info("Bybit API: reachable ✅")
        else:
            BYBIT_AVAILABLE = False
            logger.warning("Bybit API: blocked/unavailable (status %s) — skipping", r.status_code)
    except Exception as e:
        BYBIT_AVAILABLE = False
        logger.warning("Bybit API: unreachable (%s) — skipping", e)
    return BYBIT_AVAILABLE

def bybit_fetch_ohlcv(symbol, interval='240', limit=100):
    if BYBIT_AVAILABLE is False:
        return None
    try:
        r    = SESSION.get("https://api.bybit.com/v5/market/kline",
                            params={'category': 'linear', 'symbol': symbol,
                                    'interval': interval, 'limit': limit},
                            headers=HEADERS, timeout=_TIMEOUT)
        data = r.json()
        if data.get('retCode') != 0:
            logger.debug("BYBIT %s kline retCode %s: %s", symbol, data.get('retCode'), data.get('retMsg',''))
            return None
        raw = data['result']['list']
        if not raw or len(raw) < 20:
            return None
        raw = raw[::-1]
        df  = pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        return df.dropna()
    except Exception as e:
        logger.warning("BYBIT %s ohlcv exception: %s", symbol, e)
        return None

def bybit_fetch_funding(symbol):
    if MEXC_ONLY:
        return 0
    try:
        r    = SESSION.get("https://api.bybit.com/v5/market/funding/history",
                            params={'category': 'linear', 'symbol': symbol, 'limit': 1},
                            headers=HEADERS, timeout=10)
        data = r.json()
        if data.get('retCode') == 0:
            lst = data['result']['list']
            if lst:
                return float(lst[0].get('fundingRate', 0) or 0)
        return 0
    except Exception:
        return 0

def bybit_get_current_price(symbol):
    if MEXC_ONLY:
        return 0
    try:
        r    = SESSION.get("https://api.bybit.com/v5/market/tickers",
                            params={'category': 'linear', 'symbol': symbol},
                            headers=HEADERS, timeout=10)
        data = r.json()
        if data.get('retCode') == 0:
            lst = data['result']['list']
            if lst:
                return float(lst[0].get('lastPrice', 0) or 0)
        return 0
    except Exception:
        return 0

def mexc_get_top_symbols(limit=50):
    try:
        r    = SESSION.get("https://contract.mexc.com/api/v1/contract/ticker",
                            headers=HEADERS, timeout=_TIMEOUT)
        data = r.json()
        vol_list = []
        if data.get('success') and data.get('data'):
            for t in data['data']:
                sym = t.get('symbol', '')
                if not sym.endswith('_USDT'):
                    continue
                try:
                    vol   = float(t.get('amount24', 0) or 0)
                    price = float(t.get('lastPrice', 0) or 0)
                    if vol > 500_000 and price > 0:
                        # Normalise BTC_USDT → BTCUSDT immediately so fetch functions are consistent
                        clean_sym = sym.replace('_USDT', 'USDT')
                        vol_list.append({'symbol': clean_sym, 'volume': vol})
                except Exception:
                    continue
        else:
            # Spot fallback — already in BTCUSDT format
            r2 = SESSION.get("https://api.mexc.com/api/v3/ticker/24hr",
                              headers=HEADERS, timeout=_TIMEOUT)
            for t in r2.json():
                sym = t.get('symbol', '')
                if not sym.endswith('USDT'):
                    continue
                try:
                    vol   = float(t.get('quoteVolume', 0) or 0)
                    price = float(t.get('lastPrice', 0) or 0)
                    if vol > 500_000 and price > 0:
                        vol_list.append({'symbol': sym, 'volume': vol})
                except Exception:
                    continue
        vol_list.sort(key=lambda x: x['volume'], reverse=True)
        return [i['symbol'] for i in vol_list[:limit]]
    except Exception as e:
        logger.error("MEXC top symbols error: %s", e)
        return []

def mexc_get_mid_symbols(rank_from=51, rank_to=200, min_vol=250_000):
    """
    CEILING #6 — Mid-tier universe for MEXC: ranks 51–200 by 24h volume.
    Uses the same futures ticker endpoint as mexc_get_top_symbols.
    """
    try:
        r    = SESSION.get("https://contract.mexc.com/api/v1/contract/ticker",
                            headers=HEADERS, timeout=_TIMEOUT)
        data = r.json()
        vol_list = []
        if data.get('success') and data.get('data'):
            for t in data['data']:
                sym = t.get('symbol', '')
                if not sym.endswith('_USDT'):
                    continue
                try:
                    vol   = float(t.get('amount24', 0) or 0)
                    price = float(t.get('lastPrice', 0) or 0)
                    if vol >= min_vol and price > 0:
                        clean_sym = sym.replace('_USDT', 'USDT')
                        vol_list.append({'symbol': clean_sym, 'volume': vol})
                except Exception:
                    continue
        else:
            r2 = SESSION.get("https://api.mexc.com/api/v3/ticker/24hr",
                              headers=HEADERS, timeout=_TIMEOUT)
            for t in r2.json():
                sym = t.get('symbol', '')
                if not sym.endswith('USDT'):
                    continue
                try:
                    vol   = float(t.get('quoteVolume', 0) or 0)
                    price = float(t.get('lastPrice', 0) or 0)
                    if vol >= min_vol and price > 0:
                        vol_list.append({'symbol': sym, 'volume': vol})
                except Exception:
                    continue
        vol_list.sort(key=lambda x: x['volume'], reverse=True)
        mid_slice = vol_list[rank_from - 1 : rank_to]
        return [i['symbol'] for i in mid_slice]
    except Exception as e:
        logger.error("MEXC mid symbols error: %s", e)
        return []

def mexc_fetch_ohlcv(symbol, interval='4h', limit=100):
    """Fetch from MEXC FUTURES kline endpoint (correct for perps scanning)."""
    try:
        # Convert BTCUSDT → BTC_USDT for futures endpoint (suffix-only replace)
        sym_clean = symbol.upper().replace('_USDT', 'USDT').replace('/', '')
        if sym_clean.endswith('USDT'):
            futures_sym = sym_clean[:-4] + '_USDT'
        else:
            futures_sym = sym_clean + '_USDT'

        # MEXC futures interval mapping
        interval_map = {'4h': 'Hour4', '1d': 'Day1', '1h': 'Min60', '15m': 'Min15'}
        mexc_interval = interval_map.get(interval, 'Hour4')

        # FIX #MEXC-URL -- the kline URL was previously wrapped in literal
        # braces (an f-string brace-escape bug), producing the malformed URL
        # "{https://contract.mexc.com/...}". Every MEXC candle fetch raised
        # MissingSchema and was swallowed, which looked exactly like MEXC
        # being "blocked from scanning". The path is now built correctly and
        # we try each configured contract host so one blocked endpoint can't
        # stop scanning.
        data = None
        last_status = None
        for base in MEXC_CONTRACT_HOSTS:
            url = f"{base}/api/v1/contract/kline/{futures_sym}"
            try:
                r = SESSION.get(
                    url,
                    params={'interval': mexc_interval, 'limit': limit},
                    timeout=_TIMEOUT,
                )
            except Exception as e:
                logger.debug("MEXC %s kline host %s error: %s", futures_sym, base, e)
                continue
            last_status = r.status_code
            if r.status_code != 200:
                logger.debug("MEXC %s kline host %s HTTP %d", futures_sym, base, r.status_code)
                continue
            try:
                payload = r.json()
            except Exception:
                continue
            if payload.get('success') and payload.get('data'):
                data = payload
                break
        if data is None:
            logger.warning("MEXC %s futures kline unavailable (last HTTP %s)",
                           futures_sym, last_status)
            return None

        d = data['data']
        # MEXC futures returns parallel arrays
        df = pd.DataFrame({
            'timestamp': d.get('time', []),
            'open':      d.get('open', []),
            'high':      d.get('high', []),
            'low':       d.get('low', []),
            'close':     d.get('close', []),
            'volume':    d.get('vol', []),
        })
        if len(df) < 20:
            logger.debug("MEXC %s klines too short: %d rows", futures_sym, len(df))
            return None
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
        return df[['timestamp','open','high','low','close','volume']].dropna()
    except Exception as e:
        logger.warning("MEXC %s ohlcv exception: %s", symbol, e)
        return None

def mexc_get_current_price(symbol):
    try:
        clean = symbol.replace('_USDT', 'USDT').replace('/', '')
        r     = SESSION.get("https://api.mexc.com/api/v3/ticker/price",
                             params={'symbol': clean}, headers=HEADERS, timeout=10)
        data  = r.json()
        return float(data.get('price', 0) or 0)
    except Exception:
        return 0

def binance_check_available():
    """Quick probe to see if Binance futures API is reachable."""
    global BINANCE_AVAILABLE
    if MEXC_ONLY:
        BINANCE_AVAILABLE = False
        return False
    try:
        r = SESSION.get("https://fapi.binance.com/fapi/v1/ping",
                         headers=HEADERS, timeout=8)
        if r.status_code == 200:
            BINANCE_AVAILABLE = True
            logger.info("Binance futures API: reachable ✅")
        else:
            BINANCE_AVAILABLE = False
            logger.warning("Binance futures API: blocked/unavailable (status %s) — skipping", r.status_code)
    except Exception as e:
        BINANCE_AVAILABLE = False
        logger.warning("Binance futures API: unreachable (%s) — skipping", e)
    return BINANCE_AVAILABLE

def binance_get_top_symbols(limit=50):
    if not BINANCE_AVAILABLE:
        return []
    try:
        r    = SESSION.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                            headers=HEADERS, timeout=_TIMEOUT)
        data = r.json()
        if not isinstance(data, list):
            return []
        vol_list = []
        for t in data:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT'):
                continue
            try:
                vol   = float(t.get('quoteVolume', 0) or 0)
                price = float(t.get('lastPrice', 0) or 0)
                if vol > 5_000_000 and price > 0:
                    vol_list.append({'symbol': sym, 'volume': vol})
            except Exception:
                continue
        vol_list.sort(key=lambda x: x['volume'], reverse=True)
        return [i['symbol'] for i in vol_list[:limit]]
    except Exception:
        return []

def binance_get_mid_symbols(rank_from=51, rank_to=200, min_vol=2_000_000):
    """
    CEILING #6 — Mid-tier universe for Binance Futures: ranks 51–200 by quoteVolume.
    Higher min_vol than Bybit/MEXC because Binance volume figures are larger.
    """
    if not BINANCE_AVAILABLE:
        return []
    try:
        r    = SESSION.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                            headers=HEADERS, timeout=_TIMEOUT)
        data = r.json()
        if not isinstance(data, list):
            return []
        vol_list = []
        for t in data:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT'):
                continue
            try:
                vol   = float(t.get('quoteVolume', 0) or 0)
                price = float(t.get('lastPrice', 0) or 0)
                if vol >= min_vol and price > 0:
                    vol_list.append({'symbol': sym, 'volume': vol})
            except Exception:
                continue
        vol_list.sort(key=lambda x: x['volume'], reverse=True)
        mid_slice = vol_list[rank_from - 1 : rank_to]
        return [i['symbol'] for i in mid_slice]
    except Exception:
        return []

def binance_fetch_ohlcv(symbol, interval='4h', limit=100):
    if not BINANCE_AVAILABLE:
        return None
    try:
        r    = SESSION.get("https://fapi.binance.com/fapi/v1/klines",
                            params={'symbol': symbol, 'interval': interval, 'limit': limit},
                            headers=HEADERS, timeout=_TIMEOUT)
        data = r.json()
        if not isinstance(data, list) or len(data) < 20:
            return None
        df = pd.DataFrame(data, columns=[
            'timestamp','open','high','low','close','volume',
            'close_time','quote_volume','trades',
            'taker_buy_base','taker_buy_quote','ignore'])
        for col in ['open','high','low','close','volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['timestamp','open','high','low','close','volume']].dropna()
    except Exception:
        return None

def binance_fetch_funding(symbol):
    if not BINANCE_AVAILABLE:
        return 0
    try:
        r    = SESSION.get("https://fapi.binance.com/fapi/v1/fundingRate",
                            params={'symbol': symbol, 'limit': 1},
                            headers=HEADERS, timeout=10)
        data = r.json()
        if isinstance(data, list) and data:
            return float(data[0].get('fundingRate', 0) or 0)
        return 0
    except Exception:
        return 0

def binance_get_current_price(symbol):
    if not BINANCE_AVAILABLE:
        return 0
    try:
        r    = SESSION.get("https://fapi.binance.com/fapi/v1/ticker/price",
                            params={'symbol': symbol}, headers=HEADERS, timeout=10)
        data = r.json()
        return float(data.get('price', 0) or 0)
    except Exception:
        return 0

