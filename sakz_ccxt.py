"""
sakz_ccxt.py — Unified Exchange Layer for sakz_bot
=====================================================
Drop-in replacements for every raw-requests exchange function.
Uses ccxt for rate-limit handling, error normalisation, and
exchange-agnostic OHLCV / ticker / funding access.

INSTALL
───────
    pip install ccxt

INTEGRATE into sakz_bot.py
───────────────────────────
At the very top of sakz_bot.py, AFTER the existing imports, add:

    try:
        from sakz_ccxt import (
            bybit_fetch_ohlcv, bybit_get_current_price, bybit_fetch_funding,
            bybit_get_top_symbols, bybit_get_mid_symbols,
            binance_fetch_ohlcv, binance_get_current_price, binance_fetch_funding,
            binance_get_top_symbols, binance_get_mid_symbols,
            mexc_fetch_ohlcv, mexc_get_current_price,
            mexc_get_top_symbols, mexc_get_mid_symbols,
        )
        print("[sakz_bot] ccxt exchange layer loaded")
        _CCXT_AVAILABLE = True
    except ImportError:
        print("[sakz_bot] sakz_ccxt.py not found — using legacy raw-requests layer")
        _CCXT_AVAILABLE = False

All functions keep the EXACT same signatures as the originals so no
other code in sakz_bot.py needs to change.

WHY THIS MATTERS OVER RAW REQUESTS
────────────────────────────────────
• Automatic rate-limit handling (no 429 storms during heavy scans)
• Unified error types (ccxt.BadSymbol vs generic Exception)
• Exchange-agnostic symbol resolution
• Switching exchanges is a config change, not a code change
• 100+ additional exchanges available for free if needed
"""

import logging
import time
from typing import Optional, List

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# ── Interval mapping ──────────────────────────────────────────────────────────
# Bybit uses minute-integers as strings ('240' for 4h, 'D' for daily)
# ccxt unified always uses standard labels ('4h', '1d', '15m', etc.)
_BYBIT_TO_CCXT: dict = {
    '1':   '1m',  '3':   '3m',  '5':   '5m',  '15':  '15m',
    '30':  '30m', '60':  '1h',  '120': '2h',   '240': '4h',
    '360': '6h',  '720': '12h', 'D':   '1d',   'W':   '1w',   'M': '1M',
}
# Reverse map for backtest module
_CCXT_TO_BYBIT: dict = {v: k for k, v in _BYBIT_TO_CCXT.items()}

def _bybit_tf(interval: str) -> str:
    """'240' → '4h',  'D' → '1d',  etc."""
    return _BYBIT_TO_CCXT.get(str(interval), interval)


# ── Exchange singletons ───────────────────────────────────────────────────────
# One instance per exchange. ccxt manages rate limiting internally.
# FIX: track creation time and recreate after TTL to prevent stale sessions.
_exchanges: dict = {}
_exchange_created: dict = {}          # { name: float (epoch) }
_EXCHANGE_SESSION_TTL = 1800          # recreate session every 30 min

def _get_exchange(name: str) -> ccxt.Exchange:
    now = time.time()
    # Recreate if missing OR session has gone stale (prevents silent dead sessions)
    if name not in _exchanges or (now - _exchange_created.get(name, 0)) > _EXCHANGE_SESSION_TTL:
        if name == 'bybit':
            ex = ccxt.bybit({'options': {'defaultType': 'linear'}})
        elif name == 'binance':
            ex = ccxt.binanceusdm({'options': {'defaultType': 'future'}})
        elif name == 'mexc':
            ex = ccxt.mexc({'options': {'defaultType': 'swap'}})
        else:
            raise ValueError(f"Unknown exchange: {name}")
        ex.enableRateLimit = True
        _exchanges[name] = ex
        _exchange_created[name] = now
        logger.debug("ccxt: (re)created %s exchange session", name)
    return _exchanges[name]


# ── Symbol normalisation ──────────────────────────────────────────────────────
# sakz_bot uses raw symbols like 'POWERUSDT', 'BTCUSDT'.
# ccxt linear-perp markets use 'POWER/USDT:USDT' format.

def _to_ccxt_symbol(symbol: str) -> str:
    """'POWERUSDT' → 'POWER/USDT:USDT'"""
    s = symbol.upper().replace('-PERP', '').replace('_PERP', '')
    if s.endswith('USDT'):
        base = s[:-4]
        return f"{base}/USDT:USDT"
    return s  # fallback — pass through unchanged


def _from_ccxt_symbol(unified: str) -> str:
    """'POWER/USDT:USDT' → 'POWERUSDT'"""
    base = unified.split('/')[0]
    return f"{base}USDT"


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  OHLCV
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_ohlcv(exchange_name: str, symbol: str, timeframe: str,
                 limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Generic ccxt OHLCV fetcher.
    Returns a DataFrame with columns [open, high, low, close, volume]
    indexed by UTC datetime, or None on failure.
    """
    try:
        ex      = _get_exchange(exchange_name)
        unified = _to_ccxt_symbol(symbol)
        data    = ex.fetch_ohlcv(unified, timeframe=timeframe, limit=limit)
        if not data:
            return None
        df = pd.DataFrame(data,
                          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df if len(df) >= 10 else None
    except ccxt.BadSymbol:
        logger.debug("ccxt %s: bad symbol %s", exchange_name, symbol)
        return None
    except ccxt.NetworkError as e:
        logger.warning("ccxt %s network error fetching %s: %s", exchange_name, symbol, e)
        return None
    except ccxt.ExchangeError as e:
        logger.debug("ccxt %s exchange error %s: %s", exchange_name, symbol, e)
        return None
    except Exception as e:
        logger.debug("ccxt %s ohlcv %s %s: %s", exchange_name, symbol, timeframe, e)
        return None


def bybit_fetch_ohlcv(symbol: str, interval: str = '240',
                       limit: int = 100) -> Optional[pd.DataFrame]:
    """Same signature as original.  interval is Bybit-format ('240', 'D', etc.)"""
    return _fetch_ohlcv('bybit', symbol, _bybit_tf(interval), limit)


def binance_fetch_ohlcv(symbol: str, interval: str = '4h',
                         limit: int = 100) -> Optional[pd.DataFrame]:
    return _fetch_ohlcv('binance', symbol, interval, limit)


def mexc_fetch_ohlcv(symbol: str, interval: str = '4h',
                      limit: int = 100) -> Optional[pd.DataFrame]:
    return _fetch_ohlcv('mexc', symbol, interval, limit)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CURRENT PRICE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_current_price(exchange_name: str, symbol: str) -> Optional[float]:
    try:
        ex      = _get_exchange(exchange_name)
        unified = _to_ccxt_symbol(symbol)
        ticker  = ex.fetch_ticker(unified)
        price   = ticker.get('last') or ticker.get('close') or ticker.get('bid')
        return float(price) if price else None
    except ccxt.BadSymbol:
        return None
    except Exception as e:
        logger.debug("ccxt %s price %s: %s", exchange_name, symbol, e)
        return None


def bybit_get_current_price(symbol: str) -> Optional[float]:
    return _get_current_price('bybit', symbol)


def binance_get_current_price(symbol: str) -> Optional[float]:
    return _get_current_price('binance', symbol)


def mexc_get_current_price(symbol: str) -> Optional[float]:
    return _get_current_price('mexc', symbol)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FUNDING RATE
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_funding(exchange_name: str, symbol: str) -> float:
    """Returns current funding rate as a float (e.g. 0.0001 = 0.01%)."""
    try:
        ex      = _get_exchange(exchange_name)
        unified = _to_ccxt_symbol(symbol)
        info    = ex.fetch_funding_rate(unified)
        rate    = (info.get('fundingRate')
                   or info.get('funding_rate')
                   or info.get('lastFundingRate')
                   or 0.0)
        return float(rate)
    except ccxt.BadSymbol:
        return 0.0
    except Exception as e:
        logger.debug("ccxt %s funding %s: %s", exchange_name, symbol, e)
        return 0.0


def bybit_fetch_funding(symbol: str) -> float:
    return _fetch_funding('bybit', symbol)


def binance_fetch_funding(symbol: str) -> float:
    return _fetch_funding('binance', symbol)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SYMBOL LISTS  (top N by 24h USD volume)
# ═══════════════════════════════════════════════════════════════════════════════

# Cache tickers per exchange to avoid re-fetching for top/mid calls
_ticker_cache: dict = {}   # { exchange_name: {'data': [...], 'time': float} }
_TICKER_TTL = 300          # 5 min — same as scan cycle

# FIX: MEXC full symbol set cache — used to validate symbols before fetching
_mexc_symbol_set: set = set()
_mexc_symbol_set_time: float = 0.0
_MEXC_SYMBOL_SET_TTL = 600   # refresh every 10 min

def _get_mexc_symbol_set() -> set:
    """
    Returns the full set of MEXC perpetual symbols (BTCUSDT format).
    Cached for _MEXC_SYMBOL_SET_TTL seconds.
    Used to validate /cscan inputs before wasting an API call.
    """
    global _mexc_symbol_set, _mexc_symbol_set_time
    if _mexc_symbol_set and (time.time() - _mexc_symbol_set_time) < _MEXC_SYMBOL_SET_TTL:
        return _mexc_symbol_set
    try:
        ex = _get_exchange('mexc')
        markets = ex.load_markets(reload=True)
        result = set()
        for unified in markets:
            if unified.endswith('/USDT:USDT'):
                result.add(_from_ccxt_symbol(unified))
        _mexc_symbol_set = result
        _mexc_symbol_set_time = time.time()
        logger.debug("Loaded %d MEXC perp symbols", len(result))
    except Exception as e:
        logger.warning("_get_mexc_symbol_set error: %s", e)
    return _mexc_symbol_set


def mexc_symbol_exists(symbol: str) -> bool:
    """
    Fast check: is this symbol listed as a MEXC perpetual?
    BTCUSDT or BTC_USDT format both accepted.
    """
    clean = symbol.upper().replace('_USDT', 'USDT').replace('/', '')
    syms = _get_mexc_symbol_set()
    return clean in syms

def _get_sorted_symbols(exchange_name: str, min_vol: float = 0) -> List[str]:
    """
    Returns raw symbols ('BTCUSDT') sorted by 24h quote volume, descending.
    Linear USDT-settled perps only.  Results cached for _TICKER_TTL seconds.
    """
    cache = _ticker_cache.get(exchange_name)
    if cache and (time.time() - cache['time']) < _TICKER_TTL:
        return cache['data']

    try:
        ex      = _get_exchange(exchange_name)
        tickers = ex.fetch_tickers()
        rows    = []
        for unified, t in tickers.items():
            # Linear perp filter: must end in /USDT:USDT
            if not unified.endswith('/USDT:USDT'):
                continue
            vol_q = float(t.get('quoteVolume') or 0)
            if vol_q < min_vol:
                continue
            rows.append((_from_ccxt_symbol(unified), vol_q))
        rows.sort(key=lambda x: x[1], reverse=True)
        result = [r[0] for r in rows]
        _ticker_cache[exchange_name] = {'data': result, 'time': time.time()}
        return result
    except Exception as e:
        logger.warning("ccxt %s ticker list: %s — clearing cache to force retry", exchange_name, e)
        # FIX: clear stale cache entry so next call retries rather than returning empty forever
        _ticker_cache.pop(exchange_name, None)
        # Also force exchange session recreation on next call
        _exchange_created.pop(exchange_name, None)
        return []


def bybit_get_top_symbols(limit: int = 50) -> List[str]:
    return _get_sorted_symbols('bybit', min_vol=500_000)[:limit]


def bybit_get_mid_symbols(rank_from: int = 51, rank_to: int = 200,
                           min_vol: float = 500_000) -> List[str]:
    all_syms = _get_sorted_symbols('bybit', min_vol=min_vol)
    return all_syms[rank_from - 1 : rank_to]


def binance_get_top_symbols(limit: int = 50) -> List[str]:
    return _get_sorted_symbols('binance', min_vol=2_000_000)[:limit]


def binance_get_mid_symbols(rank_from: int = 51, rank_to: int = 200,
                             min_vol: float = 2_000_000) -> List[str]:
    all_syms = _get_sorted_symbols('binance', min_vol=min_vol)
    return all_syms[rank_from - 1 : rank_to]


def mexc_get_top_symbols(limit: int = 50) -> List[str]:
    return _get_sorted_symbols('mexc', min_vol=250_000)[:limit]


def mexc_get_mid_symbols(rank_from: int = 51, rank_to: int = 200,
                          min_vol: float = 250_000) -> List[str]:
    all_syms = _get_sorted_symbols('mexc', min_vol=min_vol)
    return all_syms[rank_from - 1 : rank_to]


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  EXCHANGE AVAILABILITY CHECK  (mirrors bybit_check_available / binance_check_available)
# ═══════════════════════════════════════════════════════════════════════════════

def bybit_check_available() -> bool:
    """Lightweight ping to verify Bybit reachability."""
    try:
        ex = _get_exchange('bybit')
        ex.fetch_time()
        return True
    except Exception:
        return False


def binance_check_available() -> bool:
    try:
        ex = _get_exchange('binance')
        ex.fetch_time()
        return True
    except Exception:
        return False


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    print("sakz_ccxt smoke test\n" + "─" * 40)

    sym = 'BTCUSDT'
    print(f"Symbol normalisation: {sym} → {_to_ccxt_symbol(sym)}")
    print(f"Bybit tf map: '240' → {_bybit_tf('240')}, 'D' → {_bybit_tf('D')}")

    print("\nTesting bybit_fetch_ohlcv(BTCUSDT, 240, limit=5)...")
    df = bybit_fetch_ohlcv(sym, '240', limit=5)
    if df is not None:
        print(f"  OK — {len(df)} rows, columns: {list(df.columns)}")
        print(f"  Latest close: {df['close'].iloc[-1]:.2f}")
    else:
        print("  FAILED — returned None")

    print("\nTesting bybit_get_current_price(BTCUSDT)...")
    price = bybit_get_current_price(sym)
    print(f"  Price: {price}")

    print("\nTesting bybit_fetch_funding(BTCUSDT)...")
    funding = bybit_fetch_funding(sym)
    print(f"  Funding: {funding:.6f} ({funding*100:.4f}%)")

    print("\nTesting bybit_get_top_symbols(limit=5)...")
    tops = bybit_get_top_symbols(5)
    print(f"  Top 5: {tops}")

    print("\nAll checks done.")
