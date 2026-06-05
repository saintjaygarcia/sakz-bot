"""sakz_exchanges.py - Raw-requests exchange layer extracted from sakz_bot.py.

Bybit / MEXC / Binance OHLCV, price, funding and symbol-universe fetchers, plus
the BYBIT_AVAILABLE / BINANCE_AVAILABLE runtime flags they mutate via `global`.
Other modules MUST read those flags as attributes (sakz_exchanges.BYBIT_AVAILABLE)
to see live updates. Behaviour identical to the original in-line code.
"""
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

from config import HEADERS, HTTP_TIMEOUT  # centralised configuration

# Single source of truth for the default request timeout. Lightweight ticker/
# price endpoints below intentionally keep their own shorter timeouts (8/10s).
_TIMEOUT = HTTP_TIMEOUT



BYBIT_AVAILABLE = None  # None=unchecked, True=ok, False=blocked
BINANCE_AVAILABLE = None  # None=unchecked, True=ok, False=blocked

def bybit_get_top_symbols(limit=50):
    try:
        r    = requests.get("https://api.bybit.com/v5/market/tickers?category=linear",
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
    try:
        r    = requests.get("https://api.bybit.com/v5/market/tickers?category=linear",
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
    try:
        r = requests.get("https://api.bybit.com/v5/market/time",
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
        r    = requests.get("https://api.bybit.com/v5/market/kline",
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
    try:
        r    = requests.get("https://api.bybit.com/v5/market/funding/history",
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
    try:
        r    = requests.get("https://api.bybit.com/v5/market/tickers",
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
        r    = requests.get("https://contract.mexc.com/api/v1/contract/ticker",
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
            r2 = requests.get("https://api.mexc.com/api/v3/ticker/24hr",
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
        r    = requests.get("https://contract.mexc.com/api/v1/contract/ticker",
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
            r2 = requests.get("https://api.mexc.com/api/v3/ticker/24hr",
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

        r = requests.get(
            f"https://contract.mexc.com/api/v1/contract/kline/{futures_sym}",
            params={'interval': mexc_interval, 'limit': limit},
            headers=HEADERS, timeout=_TIMEOUT
        )
        if r.status_code != 200:
            logger.warning("MEXC %s futures kline HTTP %d", futures_sym, r.status_code)
            return None
        data = r.json()
        if not data.get('success') or not data.get('data'):
            logger.debug("MEXC %s futures kline empty response", futures_sym)
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
        r     = requests.get("https://api.mexc.com/api/v3/ticker/price",
                             params={'symbol': clean}, headers=HEADERS, timeout=10)
        data  = r.json()
        return float(data.get('price', 0) or 0)
    except Exception:
        return 0

def binance_check_available():
    """Quick probe to see if Binance futures API is reachable."""
    global BINANCE_AVAILABLE
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/ping",
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
        r    = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
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
        r    = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
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
        r    = requests.get("https://fapi.binance.com/fapi/v1/klines",
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
        r    = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
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
        r    = requests.get("https://fapi.binance.com/fapi/v1/ticker/price",
                            params={'symbol': symbol}, headers=HEADERS, timeout=10)
        data = r.json()
        return float(data.get('price', 0) or 0)
    except Exception:
        return 0

