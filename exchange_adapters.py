"""exchange_adapters.py - Uniform adapter layer over sakz_exchanges.

Provides one interface (get_top_symbols, get_mid_symbols, fetch_ohlcv,
fetch_funding, get_current_price, check_available, available) across Bybit,
MEXC and Binance so call sites stop branching on exchange name.

Adapters DELEGATE to the existing, proven functions in sakz_exchanges (zero
behaviour change). The availability flags remain OWNED by sakz_exchanges and are
read live via attribute access (ex.BYBIT_AVAILABLE / ex.BINANCE_AVAILABLE).
Mid-tier volume defaults come from config. MEXC has no availability probe in the
legacy layer, so its adapter reports available=True and funding=0 (parity with
the original code, which has no mexc funding fetcher).
"""
from __future__ import annotations
from typing import List, Optional

import config
import sakz_exchanges as ex


class ExchangeAdapter:
    name = "BASE"
    default_interval = "4h"
    mid_min_vol = None

    @property
    def available(self) -> Optional[bool]:
        return True

    def check_available(self) -> Optional[bool]:
        return self.available

    def get_top_symbols(self, limit: int = 50) -> List[str]:
        raise NotImplementedError

    def get_mid_symbols(self, rank_from: int = 51, rank_to: int = 200, min_vol=None) -> List[str]:
        raise NotImplementedError

    def fetch_ohlcv(self, symbol, interval=None, limit: int = 100):
        raise NotImplementedError

    def fetch_funding(self, symbol) -> float:
        return 0

    def get_current_price(self, symbol) -> float:
        raise NotImplementedError

    def _vol(self, min_vol):
        return self.mid_min_vol if min_vol is None else min_vol


class BybitAdapter(ExchangeAdapter):
    name = "BYBIT"
    default_interval = "240"
    mid_min_vol = config.BYBIT_MID_MIN_VOL

    @property
    def available(self):
        return ex.BYBIT_AVAILABLE

    def check_available(self):
        return ex.bybit_check_available()

    def get_top_symbols(self, limit=50):
        return ex.bybit_get_top_symbols(limit)

    def get_mid_symbols(self, rank_from=51, rank_to=200, min_vol=None):
        return ex.bybit_get_mid_symbols(rank_from, rank_to, self._vol(min_vol))

    def fetch_ohlcv(self, symbol, interval=None, limit=100):
        return ex.bybit_fetch_ohlcv(symbol, interval or self.default_interval, limit)

    def fetch_funding(self, symbol):
        return ex.bybit_fetch_funding(symbol)

    def get_current_price(self, symbol):
        return ex.bybit_get_current_price(symbol)


class MexcAdapter(ExchangeAdapter):
    name = "MEXC"
    default_interval = "4h"
    mid_min_vol = config.MEXC_MID_MIN_VOL

    @property
    def available(self):
        return True  # no probe in legacy layer

    def get_top_symbols(self, limit=50):
        return ex.mexc_get_top_symbols(limit)

    def get_mid_symbols(self, rank_from=51, rank_to=200, min_vol=None):
        return ex.mexc_get_mid_symbols(rank_from, rank_to, self._vol(min_vol))

    def fetch_ohlcv(self, symbol, interval=None, limit=100):
        return ex.mexc_fetch_ohlcv(symbol, interval or self.default_interval, limit)

    def fetch_funding(self, symbol):
        return 0  # parity: legacy code has no mexc funding fetcher

    def get_current_price(self, symbol):
        return ex.mexc_get_current_price(symbol)


class BinanceAdapter(ExchangeAdapter):
    name = "BINANCE"
    default_interval = "4h"
    mid_min_vol = config.BINANCE_MID_MIN_VOL

    @property
    def available(self):
        return ex.BINANCE_AVAILABLE

    def check_available(self):
        return ex.binance_check_available()

    def get_top_symbols(self, limit=50):
        return ex.binance_get_top_symbols(limit)

    def get_mid_symbols(self, rank_from=51, rank_to=200, min_vol=None):
        return ex.binance_get_mid_symbols(rank_from, rank_to, self._vol(min_vol))

    def fetch_ohlcv(self, symbol, interval=None, limit=100):
        return ex.binance_fetch_ohlcv(symbol, interval or self.default_interval, limit)

    def fetch_funding(self, symbol):
        return ex.binance_fetch_funding(symbol)

    def get_current_price(self, symbol):
        return ex.binance_get_current_price(symbol)


# Module-level singletons. SAFE because every adapter is stateless: each method
# just delegates to the corresponding sakz_exchanges function and holds no
# instance data. If an adapter ever gains mutable per-instance state, switch to
# constructing a fresh adapter per call (or guard with a lock) instead.
ADAPTERS = dict((a.name, a) for a in (BybitAdapter(), MexcAdapter(), BinanceAdapter()))


def get_adapter(name: str) -> ExchangeAdapter:
    key = name.upper()
    try:
        return ADAPTERS[key]
    except KeyError:
        raise ValueError(
            "Unknown exchange %r; valid: %s" % (name, list(ADAPTERS))
        ) from None
