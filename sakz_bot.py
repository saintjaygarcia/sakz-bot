import logging
import os
import sakz_state as state
import re
import time
import asyncio
import sqlite3
import json
import uuid
from datetime import datetime, timedelta

# ── Turso / libsql support ────────────────────────────────────────────────────

from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler, CallbackQueryHandler
)
import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover - very old urllib3 layout
    from requests.packages.urllib3.util.retry import Retry
import pandas as pd
import ta
import io
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────
# MEMORY — persistent user data across reuploads
# ─────────────────────────────────────────────
import sakz_memory

# ─────────────────────────────────────────────
# CONVICTION LAYER — CVD / OI / VWAP
# sakz_conviction.py must live in the same directory as sakz_bot.py
# ─────────────────────────────────────────────
try:
    from sakz_conviction import conviction_scores, CONVICTION_DISPLAY
    print(f"[sakz_bot] Conviction layer loaded: {CONVICTION_DISPLAY}")
    _CONVICTION_AVAILABLE = True
except ImportError:
    print("[sakz_bot] WARNING: sakz_conviction.py not found — conviction layer disabled")
    _CONVICTION_AVAILABLE = False

# ─────────────────────────────────────────────
# CCXT EXCHANGE LAYER — sakz_ccxt.py
# Provides rate-limit-safe, unified OHLCV / price / funding / symbol fetchers.
# When available, its functions shadow the raw-requests versions defined below.
# ─────────────────────────────────────────────
try:
    from sakz_ccxt import mexc_symbol_exists
    print("[sakz_bot] ccxt symbol-validation layer loaded ✅")
    _CCXT_AVAILABLE = True
except ImportError:
    print("[sakz_bot] sakz_ccxt.py not found — using legacy raw-requests exchange layer")
    _CCXT_AVAILABLE = False
    def mexc_symbol_exists(symbol): return True   # assume valid when ccxt absent

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()

# FIX #8 — one shared HTTP session with automatic retry + backoff so transient
# network blips / 429 / 5xx no longer bubble up as bogus empty or "neutral"
# scan results. Drop-in: every http_get(...) below is routed via http_get().
HTTP_RETRIES = int(os.environ.get("SAKZ_HTTP_RETRIES", "3"))
HTTP_BACKOFF = float(os.environ.get("SAKZ_HTTP_BACKOFF", "0.5"))
HTTP_TIMEOUT = int(os.environ.get("SAKZ_HTTP_TIMEOUT", "15"))


def _make_retry():
    common = dict(
        total=HTTP_RETRIES, connect=HTTP_RETRIES, read=HTTP_RETRIES,
        status=HTTP_RETRIES, backoff_factor=HTTP_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504), raise_on_status=False,
    )
    try:
        return Retry(allowed_methods=frozenset(["GET", "POST"]), **common)
    except TypeError:                     # older urllib3 used method_whitelist
        return Retry(method_whitelist=frozenset(["GET", "POST"]), **common)


def _build_http_session():
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=_make_retry())
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_HTTP_SESSION = _build_http_session()


def http_get(url, **kwargs):
    """Drop-in replacement for requests.get with retry/backoff + default timeout."""
    kwargs.setdefault("timeout", HTTP_TIMEOUT)
    return _HTTP_SESSION.get(url, **kwargs)


# === SAKZ_TOKEN_LOGO_V1 — real token logo fetch + circular render ===
# Pulls a token's actual logo PNG from symbol-keyed icon CDNs, caches it on
# disk + in memory, and circular-masks it so it drops into a card chip. Every
# failure path returns None so callers fall back to the lettered colour badge.
# Rendered with matplotlib (imshow) — the card axes use 0..1 coords, so the
# masked RGBA composites straight onto the glass without extra figures.
_TOKEN_LOGO_DIR = os.environ.get("SAKZ_LOGO_CACHE", "/tmp/sakz_logos")
_TOKEN_LOGO_MEM = {}

def _token_logo_urls(sym):
    s = sym.lower()
    return [
        "https://assets.coincap.io/assets/icons/" + s + "@2x.png",
        "https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/" + s + ".png",
    ]

def _logo_bytes_to_rgba(data, px=128):
    """Decode PNG bytes -> square, circular-masked RGBA numpy array."""
    from PIL import Image, ImageDraw, ImageChops
    import numpy as _np
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    w, h = im.size
    m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    im = im.resize((px, px), Image.LANCZOS)
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, px - 1, px - 1), fill=255)
    alpha = ImageChops.multiply(im.split()[3], mask)
    im.putalpha(alpha)
    return _np.asarray(im)

def fetch_token_logo(symbol, *, timeout=3):
    """Return an RGBA numpy array of the token's real logo, or None."""
    try:
        base = str(symbol or "").upper().replace("/", "").replace("_", "")
        if base.endswith("USDT"):
            base = base[:-4]
        if not base:
            return None
        if base in _TOKEN_LOGO_MEM:
            return _TOKEN_LOGO_MEM[base]
        try:
            os.makedirs(_TOKEN_LOGO_DIR, exist_ok=True)
        except Exception:
            pass
        cache_png = os.path.join(_TOKEN_LOGO_DIR, base.lower() + ".png")
        data = None
        if os.path.exists(cache_png) and os.path.getsize(cache_png) > 0:
            with open(cache_png, "rb") as f:
                data = f.read()
        if data is None:
            for url in _token_logo_urls(base):
                try:
                    r = http_get(url, timeout=timeout)
                    if getattr(r, "status_code", 0) == 200 and r.content and len(r.content) > 200:
                        data = r.content
                        try:
                            with open(cache_png, "wb") as f:
                                f.write(data)
                        except Exception:
                            pass
                        break
                except Exception as _e:
                    logger.debug("token logo fetch failed %s: %s", url, _e)
        if not data:
            _TOKEN_LOGO_MEM[base] = None
            return None
        arr = _logo_bytes_to_rgba(data)
        _TOKEN_LOGO_MEM[base] = arr
        return arr
    except Exception as e:
        logger.debug("fetch_token_logo error for %s: %s", symbol, e)
        return None

def _draw_token_logo(ax, cx, cy, r, arr, aspect, zorder=7):
    """Composite a circular-masked logo RGBA array onto the card at (cx, cy)."""
    half_w = r / aspect
    ax.imshow(arr, extent=[cx - half_w, cx + half_w, cy - r, cy + r],
              aspect="auto", zorder=zorder, interpolation="bilinear", origin="upper")


# === Extracted data-access layer (sakz_db.py) ===
from sakz_db import (  # noqa: F401  re-exported; existing call sites unchanged
    db_connect,
    db_init,
    db_save_scan,
    db_load_last_scan,
    db_find_signals_by_symbol,
    db_append_price_history,
    db_load_price_history,
    db_save_signal_bias,
    db_load_signal_bias,
    db_save_card_cache,
    db_load_card_cache,
    db_register_outcome,
    db_save_user_alert,
    db_remove_user_alert,
    db_get_user_alerts,
    db_get_all_alerts,
    db_add_watch,
    db_remove_watch,
    db_get_watchlist,
    db_get_all_watchlist,
    db_add_broadcast,
    db_remove_broadcast,
    db_get_broadcast_channels,
    db_save_btc_price,
    db_get_btc_price_1h_ago,
    db_btc_alert_get_state,
    db_btc_alert_save_state,
    db_save_tracking,
    db_remove_tracking,
    db_load_all_tracking,
    db_save_trade,
    db_remove_trade,
    db_remove_all_user_trades,
    db_load_all_trades,
    db_save_price_alert,
    db_get_price_alerts,
    db_get_all_price_alerts,
    db_remove_price_alert,
    db_mark_price_alert_triggered,
    db_remove_price_alerts_for_symbol,
    db_pro_subscribe,
    db_pro_unsubscribe,
    db_pro_is_subscribed,
    db_pro_get_all_subscribers,
    db_pro_upsert_uptrend,
    db_pro_mark_uptrend_alerted,
    db_pro_cleanup_uptrends,
    db_pro_get_uptrend_rows,
    db_pro_upsert_gainer,
    db_pro_mark_gainer_alerted,
    db_pro_cleanup_gainers,
    db_pro_get_gainer_rows,
    db_pro_upsert_manip,
    db_pro_mark_manip_alerted,
    db_pro_cleanup_manip,
    db_pro_get_manip_rows,
    db_safemode_load,
    db_safemode_enable,
    db_safemode_disable,
    db_autoscan_load,
    db_autoscan_set,
    db_autoscan_remove,
    db_snail_unlock,
    db_snail_is_unlocked,
    db_snail_start_session,
    db_snail_get_session,
    db_snail_end_session,
    db_snail_increment_signals,
    db_snail_save_signal,
    db_snail_get_pending_signals,
    db_snail_update_outcome,
    db_snail_load_all_active,
    db_snail_load_unlocked,
    db_init_user_tracking,
    db_track_user,
    db_admin_is_authed,
    db_admin_set_auth,
    db_admin_revoke,
    db_admin_get_stats,
    DB_PATH, ACTIVE_WINDOW_MIN, TURSO_URL, TURSO_TOKEN, _USE_TURSO,
)
# === Extracted exchange layer (sakz_exchanges.py) ===
import sakz_exchanges  # live sakz_exchanges.BYBIT_AVAILABLE / sakz_exchanges.BINANCE_AVAILABLE reads
from exchange_adapters import get_adapter  # uniform exchange dispatch (migration target)
from sakz_exchanges import (  # noqa: F401  re-exported; existing call sites unchanged
    bybit_get_top_symbols,
    bybit_get_mid_symbols,
    bybit_check_available,
    bybit_fetch_ohlcv,
    bybit_fetch_funding,
    bybit_get_current_price,
    mexc_get_top_symbols,
    mexc_get_mid_symbols,
    mexc_fetch_ohlcv,
    mexc_get_current_price,
    binance_check_available,
    binance_get_top_symbols,
    binance_get_mid_symbols,
    binance_fetch_ohlcv,
    binance_fetch_funding,
    binance_get_current_price,
    HEADERS,
)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN environment variable is not set! "
                     "Add it in Render/Railway → Environment tab.")

# ── Diagnostic scan result ────────────────────────────────────────────────────
# ScanFailure is returned (instead of None) by score_pair, analyze_symbol_mtf,
# and _cscan_pair_mtf when we can identify *why* a signal was suppressed.
# cscan_command aggregates these and produces a meaningful error message.

from dataclasses import dataclass, field
from typing import Optional

# Signal engine extracted to sakz_scanner.py (Stage 2 of monolith refactor)
from sakz_scanner import (
    ScanFailure,
    score_pair,
    add_indicators,
    get_btc_regime,
    confirm_rr,
    _rr_ratio,
    calculate_leverage,
    candle_quality_score,
    fib_confluence_score,
    find_pivot_support,
    find_pivot_resistance,
    _get_btc_dominance,
    _get_btc_price_cached,
    session_context,
    # Phase 0 — signal surfacing
    SCAN_DISPLAY_CONF_MIN,
    AUTOSCAN_DISPLAY_CONF_MIN,
    _SCAN_SHOW_ALL_FLAGS,
    _conf_of,
    passes_display_floor,
    risk_reasons,
)

# Reason constants — used for aggregation in cscan_command
REASON_NO_CONTRACT    = "NO_CONTRACT"     # API returned empty / unknown symbol
REASON_SHORT_HISTORY  = "SHORT_HISTORY"   # < min candles for the requested TF
REASON_LOW_CONF       = "LOW_CONF"        # scored < 4 or abs-score gate
REASON_REGIME_BLOCK   = "REGIME_BLOCK"    # BTC regime gate suppressed it
REASON_GAP_BLOCK      = "GAP_BLOCK"       # winning - losing < 2
REASON_FLIP_BLOCK     = "FLIP_BLOCK"      # direction flip cooldown
REASON_COUNTER_TREND  = "COUNTER_TREND"   # counter-trend min_score_req not met
REASON_NEUTRAL        = "NEUTRAL"         # market too neutral (no directional bias)
REASON_LOW_VOLUME     = "LOW_VOLUME"       # 24h volume below minimum threshold



# Conversation states
PICK_TRADE   = 1
ASK_REMINDER = 2
ASK_INTERVAL = 3

# FIX H3 — bound the per-chat caches so they can't grow without limit (OOM guard).
class _BoundedDict(dict):
    """dict with FIFO eviction once it exceeds max_size entries."""
    def __init__(self, max_size=2000):
        super().__init__()
        self._max_size = max_size
    def __setitem__(self, key, value):
        if key not in self and len(self) >= self._max_size:
            try:
                del self[next(iter(self))]
            except StopIteration:
                pass
        super().__setitem__(key, value)

# Custom scan results per chat { chat_id: [signals] }
cscan_results    = _BoundedDict(2000)
# Stores per-chat scan context so any Refresh knows which result set to use.
# Key: chat_id  Value: {'source': 'scan'|'scalp'|'swing'|'custom', 'results': [...], 'title': str}
_chat_scan_ctx   = _BoundedDict(2000)
# Snapshot taken at scan time for /compare PnL { chat_id: {symbol: {entry_price, leverage, bias, scan_time}} }
compare_snapshot = {}
# Signal card cache for Details/Back button { cache_key: {signal, rank, primary_kb} }
# Flip cooldown cache { symbol: {bias, time} } — prevents rapid direction reversals

# Auto-scan interval (seconds). 4 hours = 14400
AUTO_SCAN_INTERVAL = 14400

# ── FIX #HEARTBEAT — Self-monitoring ─────────────────────────────────────────
# Set HEALTH_CHECK_URL in env (BetterStack, UptimeRobot, etc.) to get
# notified when the bot goes silent. A ping is sent every 5 minutes.
HEALTH_CHECK_URL   = os.environ.get("HEALTH_CHECK_URL", "")   # e.g. https://uptime.betterstack.com/api/v1/heartbeat/XXXX
ADMIN_ALERT_CHAT   = os.environ.get("ADMIN_ALERT_CHAT", "")   # chat_id to receive self-alerts (optional)

# ── FIX #SCANTIME — Scan duration percentile tracking ────────────────────────

# ─────────────────────────────────────────────
# 🐌 SNAIL MODE — Hidden easter egg feature
# Activated ONLY via secret command /scan1234JP$$
# /snail alone does nothing unless user is unlocked
# ──�������������������������������������������������������������������──────────────────────────────────────────
SNAIL_SECRET_CMD   = "scan1234JP$$"      # secret unlock passphrase
snail_active       = {}                  # chat_id → { activated_at, expires_at, signals_sent, week_log }


# ─────────────────────────────────────────────
# IMPROVEMENT #2 — SQLite PERSISTENCE
# All scan results, price history, signal outcomes,
# and user alert registrations survive restarts.
# ─────────────────────────────────────────────




# ── DB helpers ────────────────────────────────




# ── FIX #PERSIST-BIAS — signal bias persistence helpers ──────────────────────



# ── FIX #PERSIST-CACHE — signal card cache persistence helpers ────────────────








# ── Watchlist helpers ─────────────────────────




# ── Broadcast channel helpers ──────────────────



# ── BTC snapshot helpers ───────────────────────





# ── Multi-trade tracking DB ───────────────────────────────────────────────────





# ── Price-level alert DB ──────────────────────────────────────────────────────







# ═══════════════════════════════════════════════════════════════════════════════
# /pro FEATURE — PRO ALERT SUITE
# • Sustained Uptrend Tracker  (≥8 %/day, 20 h – 72 h window)
# • Top-10 Gainer Persistence  (1 – 3 days in top 10)
# • Scam Pump / Manipulation Detector
# ═══════════════════════════════════════════════════════════════════════════════

# ── /pro DB helpers ─────────────────────────────────────────────────────────────





# ── Uptrend log ─────────────────────────────────────────────────────────────────





# ── Gainers log ──────────���───────────────────────────────────────────────────────





# ── Manipulation log ──────────────────────────────────────────────────────────────






# ── /pro Detection engine ────────────────���─������������������─────────────────────────────────��────

def _pro_fetch_top_gainers(limit: int = 20) -> list:
    """
    Returns top gainers by 24-h price change (list of dicts).
    Primary: CoinGecko /coins/markets.  Fallback: MEXC /ticker/24hr.
    """
    results = []

    # CoinGecko primary ─────────────────────────────────────────────────────────
    try:
        url  = (
            "https://api.coingecko.com/api/v3/coins/markets"
            "?vs_currency=usd&order=price_change_percentage_24h_desc"
            f"&per_page={limit}&page=1&sparkline=false"
        )
        resp = http_get(url, timeout=10, headers=HEADERS)
        if resp.status_code == 200:
            for c in resp.json():
                chg = c.get("price_change_percentage_24h", 0) or 0
                results.append({
                    "symbol":           c.get("symbol", "").upper() + "USDT",
                    "name":             c.get("name", ""),
                    "price_change_24h": chg,
                    "volume_24h":       c.get("total_volume", 0) or 0,
                    "market_cap":       c.get("market_cap", 0) or 0,
                    "current_price":    c.get("current_price", 0) or 0,
                    "market_cap_rank":  c.get("market_cap_rank", 9999) or 9999,
                    "source":           "coingecko",
                })
            if results:
                return results[:10]
    except Exception as _e:
        logger.debug("_pro_fetch_top_gainers CoinGecko: %s", _e)

    # MEXC fallback ───────────────────────────��─────────────────────────────────
    try:
        resp = http_get(
            "https://api.mexc.com/api/v3/ticker/24hr", timeout=10, headers=HEADERS
        )
        if resp.status_code == 200:
            usdt = [
                t for t in resp.json()
                if str(t.get("symbol", "")).endswith("USDT")
                and float(t.get("quoteVolume", 0) or 0) > 1_000_000
            ]
            usdt.sort(
                key=lambda t: float(t.get("priceChangePercent", 0) or 0),
                reverse=True
            )
            for t in usdt[:limit]:
                results.append({
                    "symbol":           t.get("symbol", ""),
                    "name":             t.get("symbol", "").replace("USDT", ""),
                    "price_change_24h": float(t.get("priceChangePercent", 0) or 0),
                    "volume_24h":       float(t.get("quoteVolume", 0) or 0),
                    "market_cap":       0,
                    "current_price":    float(t.get("lastPrice", 0) or 0),
                    "market_cap_rank":  9999,
                    "source":           "mexc",
                })
            return results[:10]
    except Exception as _e:
        logger.debug("_pro_fetch_top_gainers MEXC: %s", _e)

    return results


def _pro_check_uptrend(symbol: str, exchange: str) -> dict:
    """
    Checks sustained organic uptrend.

    Pass conditions (ALL must be true):
      1. Last closed daily candle ≥ +8 %  (close/open − 1)
      2. At least 1 consecutive qualifying daily candle (up to 3)
      3. 4H bullish streak of 5–18 candles  (20 h – 72 h)

    Returns dict with qualifies, streak_hours, daily_gains, avg_daily_gain,
    current_price, reasons, warnings.
    """
    out = dict(
        qualifies=False, streak_hours=0.0, streak_days=0.0,
        daily_gains=[], avg_daily_gain=0.0,
        current_price=0.0, exchange=exchange, symbol=symbol,
        reasons=[], warnings=[]
    )
    try:
        if exchange == "BYBIT" and sakz_exchanges.BYBIT_AVAILABLE:
            df1d = bybit_fetch_ohlcv(symbol, "D",   10)
            df4h = bybit_fetch_ohlcv(symbol, "240", 30)
        elif exchange == "BINANCE" and sakz_exchanges.BINANCE_AVAILABLE:
            df1d = binance_fetch_ohlcv(symbol, "1d", 10)
            df4h = binance_fetch_ohlcv(symbol, "4h", 30)
        else:
            df1d = mexc_fetch_ohlcv(symbol, "1d", 10)
            df4h = mexc_fetch_ohlcv(symbol, "4h", 30)

        if df1d is None or df4h is None or len(df1d) < 2 or len(df4h) < 5:
            out["warnings"].append("Insufficient candle data")
            return out

        out["current_price"] = float(df4h.iloc[-1]["close"])

        # Daily gains — skip the current (still-open) daily candle
        closed = df1d.iloc[-4:-1]
        gains  = []
        for _, row in closed.iterrows():
            o, c = float(row["open"]), float(row["close"])
            if o > 0:
                gains.append(round((c - o) / o * 100, 2))
        out["daily_gains"] = gains

        if not gains or gains[-1] < 8.0:
            out["warnings"].append(
                f"Last daily +{gains[-1]:.1f}%" if gains else "No closed daily candle data"
            )
            return out

        # Count consecutive qualifying days
        consec = 0
        for g in reversed(gains):
            if g >= 8.0:
                consec += 1
            else:
                break

        out["avg_daily_gain"] = sum(gains[-consec:]) / consec

        # 4H bullish streak (consecutive bullish candles + no-backslide)
        recent4h = df4h.iloc[-20:]
        cls = recent4h["close"].tolist()
        opn = recent4h["open"].tolist()

        streak4h = 0
        for i in range(len(cls) - 1, -1, -1):
            is_bull     = cls[i] > opn[i]
            no_backslide = (i == 0) or (cls[i] >= cls[i - 1] * 0.995)
            if is_bull and no_backslide:
                streak4h += 1
            else:
                break

        streak_hrs = streak4h * 4
        out["streak_hours"] = streak_hrs
        out["streak_days"]  = streak_hrs / 24

        if streak_hrs < 20:
            out["warnings"].append(f"4H streak {streak_hrs:.0f}h — need ≥20h")
            return out

        lbl = f"{streak_hrs/24:.1f}d" if streak_hrs >= 24 else f"{streak_hrs:.0f}h"
        out["reasons"] = [
            f"✅ {consec} consecutive daily candle(s) each ≥+8%",
            f"✅ Average daily gain: +{out['avg_daily_gain']:.1f}%",
            f"✅ 4H bullish streak: {streak4h} candles ({lbl})",
            f"✅ Daily gains: {', '.join(f'+{g:.1f}%' for g in gains[-consec:])}",
        ]
        if streak_hrs > 72:
            out["reasons"].append(
                f"⚠️ Streak {streak_hrs:.0f}h exceeds 3 days — watch for exhaustion"
            )
        out["qualifies"] = True

    except Exception as _e:
        logger.warning("_pro_check_uptrend %s %s: %s", exchange, symbol, _e)
        out["warnings"].append(str(_e))
    return out


def _pro_detect_manipulation(symbol: str, exchange: str) -> dict:
    """
    Multi-factor manipulation / scam-pump detector.

    Score table (max 180):
      1. RSI >80 + MACD divergence          → 30 pts
      2. Dominant wick on latest 4H (>60 %) → 20 pts
      3. Volume ≥5x avg, price move <1 %    → 30 pts  (wash trading)
      4. Erratic volume (CV >1.5)           → 20 pts
      5. Micro-cap (<$5M) + 24h gain >25 %  → 35 pts
      6. Vol/mktcap ratio >0.5              → 20 pts
      7. +100 % in 7 days on small cap      → 25 pts

    is_scam = True when total score ≥ 60.
    """
    out = dict(
        manip_score=0, is_scam=False,
        verdict_label="🟢 CLEAN",
        reasons=[], current_price=0.0,
        symbol=symbol, exchange=exchange
    )
    try:
        if exchange == "BYBIT" and sakz_exchanges.BYBIT_AVAILABLE:
            df4h = bybit_fetch_ohlcv(symbol, "240", 40)
        elif exchange == "BINANCE" and sakz_exchanges.BINANCE_AVAILABLE:
            df4h = binance_fetch_ohlcv(symbol, "4h", 40)
        else:
            df4h = mexc_fetch_ohlcv(symbol, "4h", 40)

        if df4h is None or len(df4h) < 6:
            return out
        df4h = add_indicators(df4h, timeframe="4h")
        if df4h is None or len(df4h) < 6:
            return out

        L = df4h.iloc[-1]
        P = df4h.iloc[-2]
        score   = 0
        reasons = []

        price = float(L.get("close", 0))
        out["current_price"] = price

        # 1 — RSI extreme + MACD divergence ────────────────────────────────────
        rsi       = float(L.get("rsi", 50))
        macd_now  = float(L.get("macd_diff", 0))
        macd_prev = float(P.get("macd_diff", 0))
        if rsi > 80:
            if macd_now < macd_prev:
                score += 30
                reasons.append(
                    f"🚩 RSI extreme ({rsi:.1f}) + MACD fading — exhaustion pump"
                )
            else:
                score += 15
                reasons.append(f"⚠️ RSI very overbought ({rsi:.1f}) — parabolic territory")

        # 2 — Dominant wick ─────────────────────────────────────────────────────
        rng = float(L["high"]) - float(L["low"])
        if rng > 0:
            up_w = (float(L["high"]) - max(float(L["open"]), float(L["close"]))) / rng
            dn_w = (min(float(L["open"]), float(L["close"])) - float(L["low"]))  / rng
            if up_w > 0.60:
                score += 20
                reasons.append(
                    f"🚩 Upper wick {up_w*100:.0f}% of candle — stop hunt / rejection"
                )
            if dn_w > 0.60:
                score += 15
                reasons.append(
                    f"🚩 Lower wick {dn_w*100:.0f}% of candle — liquidity grab"
                )

        # 3 — Volume spike + flat price (wash trading) ──────────────────────────
        vol    = float(L.get("volume", 0))
        vol_ma = float(L.get("volume_ma", vol) or vol)
        p_prev = float(P.get("close", price))
        p_chg  = abs(price - p_prev) / p_prev * 100 if p_prev > 0 else 0
        if vol_ma > 0:
            vr = vol / vol_ma
            if vr >= 5.0 and p_chg < 1.0:
                score += 30
                reasons.append(
                    f"🚩 Volume {vr:.1f}x avg, price moved {p_chg:.2f}% — wash trading"
                )
            elif vr >= 3.0 and p_chg < 0.5:
                score += 20
                reasons.append(
                    f"🚩 Volume {vr:.1f}x with near-zero price move — suspect"
                )

        # 4 — Erratic volume (CV) ───────────────────────────────────────────────
        rvols = df4h["volume"].iloc[-8:-1].tolist()
        if len(rvols) >= 4:
            avg_v = sum(rvols) / len(rvols)
            if avg_v > 0:
                std_v = (sum((v - avg_v) ** 2 for v in rvols) / len(rvols)) ** 0.5
                cv    = std_v / avg_v
                if cv > 2.0:
                    score += 20
                    reasons.append(
                        f"🚩 Extremely erratic volume (CV={cv:.2f}) — spoofing pattern"
                    )
                elif cv > 1.5:
                    score += 10
                    reasons.append(
                        f"⚠️ Unstable volume (CV={cv:.2f}) — abnormal participation"
                    )

        # 5-7 — CoinGecko fundamentals (best-effort) ��─────────��────���───���─���─���─���──
        try:
            slug    = symbol.replace("USDT", "").lower()
            cg_resp = http_get(
                f"https://api.coingecko.com/api/v3/coins/{slug}",
                timeout=6, headers=HEADERS
            )
            if cg_resp.status_code == 200:
                cg      = cg_resp.json()
                mkt     = cg.get("market_data", {})
                mkt_cap = (mkt.get("market_cap",   {}) or {}).get("usd", 0) or 0
                vol_24h = (mkt.get("total_volume", {}) or {}).get("usd", 0) or 0
                pc_24h  = mkt.get("price_change_percentage_24h", 0) or 0
                pc_7d   = mkt.get("price_change_percentage_7d",  0) or 0

                # 5 — micro/small-cap + big pump
                if 0 < mkt_cap < 5_000_000 and pc_24h > 25:
                    score += 35
                    reasons.append(
                        f"🚩 Micro-cap (${mkt_cap/1e6:.2f}M) +{pc_24h:.1f}% in 24h"
                    )
                elif 0 < mkt_cap < 20_000_000 and pc_24h > 30:
                    score += 25
                    reasons.append(
                        f"🚩 Small-cap (${mkt_cap/1e6:.1f}M) +{pc_24h:.1f}% — suspicious"
                    )

                # 6 — vol/mktcap ratio
                if mkt_cap > 0:
                    vmr = vol_24h / mkt_cap
                    if vmr > 0.5:
                        score += 20
                        reasons.append(
                            f"🚩 Vol/Mktcap {vmr:.2f} — extreme wash-trading risk"
                        )
                    elif vmr > 0.25:
                        score += 10
                        reasons.append(
                            f"⚠️ Vol/Mktcap {vmr:.2f} — elevated, monitor closely"
                        )

                # 7 — 7-day parabolic
                if pc_7d > 100 and (mkt_cap == 0 or mkt_cap < 50_000_000):
                    score += 25
                    reasons.append(
                        f"🚩 +{pc_7d:.0f}% in 7 days on small/unknown cap — parabolic pump"
                    )
        except Exception:
            pass

        score = min(score, 180)
        out["manip_score"] = score
        out["reasons"]     = reasons

        if score >= 80:
            out["is_scam"]       = True
            out["verdict_label"] = "🔴 HIGH RISK — Likely manipulated / scam pump"
        elif score >= 60:
            out["is_scam"]       = True
            out["verdict_label"] = "🟠 ELEVATED RISK — Inorganic price action detected"
        elif score >= 35:
            out["verdict_label"] = "🟡 MODERATE — Some manipulation signals, use caution"
        else:
            out["verdict_label"] = "🟢 CLEAN — No significant manipulation signals"

    except Exception as _e:
        logger.warning("_pro_detect_manipulation %s %s: %s", exchange, symbol, _e)
    return out


# ── /pro Alert card formatters ─────────────────────────────────────────────────────

def _pro_format_uptrend_card(uptrend: dict, rank: int = 1) -> str:
    sym   = uptrend["symbol"]
    exch  = uptrend["exchange"]
    gains = uptrend["daily_gains"]
    avg   = uptrend["avg_daily_gain"]
    hrs   = uptrend["streak_hours"]
    price = uptrend["current_price"]
    lbl   = f"{hrs/24:.1f}d" if hrs >= 24 else f"{hrs:.0f}h"
    bar   = "█" * min(int(avg / 5), 10) + "░" * max(10 - int(avg / 5), 0)
    g_str = "  ".join(f"+{g:.1f}%" for g in gains[-3:]) if gains else "—"
    rsns  = "\n".join(f"   • {r}" for r in uptrend.get("reasons", []))

    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🚀 PRO ALERT — SUSTAINED UPTREND\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"#{rank}  {exch} | {sym}\n"
        f"\n"
        f"��� BIAS: LONG (trend confirmed)\n"
        f"⭐ AVG DAILY GAIN: {bar} +{avg:.1f}%/day\n"
        f"\n"
        f"⏱ STREAK DURATION: {lbl}\n"
        f"📅 DAILY GAINS:     {g_str}\n"
        f"💰 CURRENT PRICE:   ${price:.6f}\n"
        f"\n"
        f"✅ CONVICTION SIGNALS:\n"
        f"{rsns}\n"
        f"\n"
        "⚠️ Risk note: Enter on pullbacks — not at candle highs.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 /pro alert  ·  /cscan {sym.replace('USDT', '')} for entry zone"
    )


def _pro_format_gainer_card(row: dict, rank: int = 1) -> str:
    sym   = row["symbol"]
    times = row["times_top10"]
    first = row.get("first_seen", "")
    last  = row.get("last_seen",  "")
    try:
        dur_hrs = (
            datetime.fromisoformat(last) - datetime.fromisoformat(first)
        ).total_seconds() / 3600
        dur_lbl = f"{dur_hrs/24:.1f}d" if dur_hrs >= 24 else f"{dur_hrs:.0f}h"
    except Exception:
        dur_lbl = "unknown"

    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🏆 PRO ALERT — PERSISTENT TOP-10 GAINER\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━��━━\n"
        f"#{rank}  {sym}\n"
        f"\n"
        f"📈 IN TOP-10 GAINERS:  {times}x check(s)\n"
        f"⏱ DURATION IN TOP 10: {dur_lbl}\n"
        f"\n"
        "✅ Why this matters:\n"
        "   • Sustained top-10 = persistent buying pressure\n"
        "   • Not a flash pump — volume supporting the move\n"
        "   • Momentum likely to continue short-term\n"
        f"\n"
        f"⚠️ Run /cscan {sym.replace('USDT', '')} for entry zone & stop-loss.\n"
        "━━━━━━━━━━━━━━���━━━━━━━━━━━━━━━\n"
        f"📡 /pro alert  ·  /cscan {sym.replace('USDT', '')} for full signal"
    )


def _pro_format_manip_card(manip: dict, rank: int = 1) -> str:
    sym     = manip["symbol"]
    exch    = manip["exchange"]
    score   = manip["manip_score"]
    verdict = manip["verdict_label"]
    price   = manip["current_price"]
    reasons = manip.get("reasons", [])
    bar     = "█" * min(int(score / 18), 10) + "░" * max(10 - int(score / 18), 0)
    rsns    = "\n".join(f"   • {r}" for r in reasons) if reasons else "   • No specific flags"

    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🚨 PRO ALERT — MANIPULATION DETECTED\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"#{rank}  {exch} | {sym}\n"
        f"\n"
        f"🔍 VERDICT: {verdict}\n"
        f"🎯 RISK SCORE: {bar} {score}/180\n"
        f"💰 CURRENT PRICE: ${price:.6f}\n"
        f"\n"
        f"🚩 MANIPULATION SIGNALS:\n"
        f"{rsns}\n"
        f"\n"
        "📌 What this means:\n"
        "   • Chart action is NOT organic\n"
        "   • Likely driven by whales / coordinated bots\n"
        "   • AVOID opening a long — dump risk is HIGH\n"
        "   • If already holding: reduce exposure now\n"
        f"\n"
        "⚠️ Do NOT trade this on TA alone.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📡 /pro alert  ·  This is a RISK WARNING, not a signal"
    )


# ── /pro In-memory dedup ───────────────────────────────���─────���──���──────────────────

_pro_alerted_uptrend: set = set()
_pro_alerted_gainers: set = set()
_pro_alerted_manip:   set = set()

# ── /pro Top-gainers cache (5-min TTL) ────────────────────────────────────────────
_PRO_GAINERS_TTL: int         = 300   # seconds


def _pro_fetch_top_gainers_cached() -> list:
    """Return cached top-gainers list; refresh if older than 5 minutes."""

    import time as _time
    if state._pro_gainers_cache and (_time.time() - state._pro_gainers_cache_ts) < _PRO_GAINERS_TTL:
        return state._pro_gainers_cache
    fresh = _pro_fetch_top_gainers()
    if fresh:
        state._pro_gainers_cache    = fresh
        state._pro_gainers_cache_ts = _time.time()
    return fresh


# ── /pro Fast job — uptrend + manipulation (every 30 min) ─────────────────────────

# ── FIX #HEARTBEAT — Heartbeat job ──────────────────────────────────────────────
async def heartbeat_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Sends a ping to HEALTH_CHECK_URL every 5 minutes.
    If the bot is alive and the job runs, the ping fires.
    If the bot crashes, the ping stops — and BetterStack/UptimeRobot alerts you.

    Set HEALTH_CHECK_URL in your environment (Railway/Render env vars):
        BetterStack:  https://uptime.betterstack.com/api/v1/heartbeat/XXXX
        UptimeRobot:  https://heartbeat.uptimerobot.com/XXXX
    """
    if not HEALTH_CHECK_URL:
        return
    try:
        http_get(HEALTH_CHECK_URL, timeout=5)
        logger.debug("Heartbeat ping sent ✅")
    except Exception as e:
        logger.warning("Heartbeat ping failed: %s", e)

    # Also emit scan duration stats to admin if scans are slow

    if len(state._scan_durations) >= 5:
        recent = state._scan_durations[-10:]
        avg    = sum(recent) / len(recent)
        p90    = sorted(recent)[int(len(recent) * 0.9)]
        if p90 > 120 and ADMIN_ALERT_CHAT:
            try:
                await context.bot.send_message(
                    chat_id=int(ADMIN_ALERT_CHAT),
                    text=f"⚠️ *Scan Slowdown* — p90={p90:.0f}s  avg={avg:.0f}s\n"
                         f"Last {len(recent)} scans. Check exchange API health.",
                    parse_mode='Markdown'
                )
                state._scan_durations = []   # reset after alert
            except Exception:
                pass


async def pro_fast_job(context):
    """
    PRO fast scan — runs every 30 minutes.

    • Pre-screens candidates by 24h gain (≥8 %) before any OHLCV fetch.
    • Fetches uptrend + manipulation data in parallel via asyncio.gather().
    • Uses cached top-gainers list (5-min TTL) to avoid redundant API calls.
    """
    subscribers = db_pro_get_all_subscribers()
    if not subscribers:
        return

    loop = asyncio.get_event_loop()
    db_pro_cleanup_uptrends()
    db_pro_cleanup_manip()

    # Fetch top gainers (cached) ───────────────────────────────────────────────
    gainers = []
    try:
        gainers = await loop.run_in_executor(SCAN_EXECUTOR, _pro_fetch_top_gainers_cached)
    except Exception as _e:
        logger.warning("pro_fast_job fetch_top_gainers: %s", _e)

    # Pre-screen: only tokens with ≥8% 24h gain pass to OHLCV stage ──────────
    pre_screened = {g["symbol"] for g in gainers if g.get("price_change_24h", 0) >= 8.0}

    # Also re-check tokens already tracked in the uptrend log ─────────────────
    existing_ut = {(r["symbol"], r["exchange"]): r for r in db_pro_get_uptrend_rows()}
    candidates  = list(pre_screened)
    for sym, _ in existing_ut:
        if sym not in candidates:
            candidates.append(sym)

    exchanges = ["BYBIT", "MEXC"] if sakz_exchanges.BYBIT_AVAILABLE else ["MEXC"]

    # Build all (sym, exchange) pairs ─────────────────────────────────────────
    pairs = [(sym, exch) for sym in candidates[:25] for exch in exchanges]

    async def _check_pair(sym: str, exchange: str):
        """Run uptrend + manipulation checks for one pair and return results."""
        ut_result    = None
        manip_result = None
        try:
            ut_result, manip_result = await asyncio.gather(
                loop.run_in_executor(SCAN_EXECUTOR, lambda s=sym, e=exchange: _pro_check_uptrend(s, e)),
                loop.run_in_executor(SCAN_EXECUTOR, lambda s=sym, e=exchange: _pro_detect_manipulation(s, e)),
            )
        except Exception as _e:
            logger.warning("pro_fast_job pair %s %s: %s", sym, exchange, _e)
        return sym, exchange, ut_result, manip_result

    # Run all pairs in parallel ────────────────────────────────────────────────
    results = await asyncio.gather(*[_check_pair(s, e) for s, e in pairs])

    uptrend_alerts = []
    manip_alerts   = []

    for sym, exchange, uptrend, manip in results:
        ut_key = f"{sym}_{exchange}"

        # Uptrend ──────────────────────────────────────────────────────────────
        if uptrend and uptrend.get("qualifies"):
            try:
                db_pro_upsert_uptrend(sym, exchange, uptrend["daily_gains"])
            except Exception as e:
                logger.warning("pro_scan: db_pro_upsert_uptrend failed for %s/%s: %s", sym, exchange, e)
            existing_row    = existing_ut.get((sym, exchange))
            already_alerted = existing_row and existing_row.get("alert_sent", 0)
            if not already_alerted and ut_key not in _pro_alerted_uptrend:
                uptrend_alerts.append(uptrend)
                _pro_alerted_uptrend.add(ut_key)
                try:
                    db_pro_mark_uptrend_alerted(sym, exchange)
                except Exception as e:
                    logger.warning("pro_scan: db_pro_mark_uptrend_alerted failed for %s/%s: %s", sym, exchange, e)

        # Manipulation ─────────────────────────────────────────────────────────
        if manip and manip.get("is_scam") and manip["manip_score"] >= 60:
            mk = f"{sym}_{exchange}"
            try:
                db_pro_upsert_manip(sym, exchange, manip["manip_score"], manip["reasons"])
            except Exception as e:
                logger.warning("pro_scan: db_pro_upsert_manip failed for %s/%s: %s", sym, exchange, e)
            if mk not in _pro_alerted_manip:
                already = any(
                    r["symbol"] == sym and r["exchange"] == exchange and r["alert_sent"]
                    for r in db_pro_get_manip_rows()
                )
                if not already:
                    manip_alerts.append(manip)
                    _pro_alerted_manip.add(mk)
                    try:
                        db_pro_mark_manip_alerted(sym, exchange)
                    except Exception as e:
                        logger.warning("pro_scan: db_pro_mark_manip_alerted failed for %s/%s: %s", sym, exchange, e)

    # Broadcast ────────────────────────────────────────────────────────────────
    if not (uptrend_alerts or manip_alerts):
        return

    for chat_id in subscribers:
        try:
            for i, ut in enumerate(uptrend_alerts[:5], 1):
                sym  = ut["symbol"]
                exch = ut["exchange"]
                kb   = InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔗 Trade", url=get_exchange_link(exch, sym)),
                ]])
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=_pro_format_uptrend_card(ut, rank=i),
                    reply_markup=kb
                )
                await asyncio.sleep(0.4)

            for i, ma in enumerate(manip_alerts[:5], 1):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=_pro_format_manip_card(ma, rank=i)
                )
                await asyncio.sleep(0.4)

        except Exception as _e:
            logger.warning("pro_fast_job send %s: %s", chat_id, _e)


# ── /pro Gainers job — persistence tracker (every 4 hours) ───────────────────────

async def pro_gainers_job(context):
    """
    PRO gainers scan — runs every 4 hours.

    Top-10 gainer persistence is a slow signal by nature (needs multiple
    appearances over hours). Polling it every 30 min wastes API calls on
    a signal that can only mature every few hours. Fires alert after:
      • ≥2 appearances in the top 10
      • first_seen between 6h and 72h ago (confirmed persistence, not exhausted)
    """
    subscribers = db_pro_get_all_subscribers()
    if not subscribers:
        return

    loop = asyncio.get_event_loop()
    db_pro_cleanup_gainers()

    gainers = []
    try:
        gainers = await loop.run_in_executor(SCAN_EXECUTOR, _pro_fetch_top_gainers_cached)
    except Exception as _e:
        logger.warning("pro_gainers_job fetch_top_gainers: %s", _e)

    for g in gainers:
        if g.get("price_change_24h", 0) >= 8.0:
            try:
                db_pro_upsert_gainer(g["symbol"])
            except Exception as e:
                logger.warning("pro_scan: db_pro_upsert_gainer failed for %s: %s", g.get("symbol", "?"), e)

    persistent_gainer_alerts = []
    cutoff_6h = (datetime.now() - timedelta(hours=6)).isoformat()
    cutoff_3d = (datetime.now() - timedelta(hours=72)).isoformat()

    for row in db_pro_get_gainer_rows():
        sym     = row["symbol"]
        first   = row["first_seen"]
        times   = row["times_top10"]
        alerted = row["alert_sent"]
        if alerted or sym in _pro_alerted_gainers:
            continue
        # ≥2 appearances, first seen >6h ago (confirmed) but <72h (not stale)
        if times < 2 or first > cutoff_6h or first < cutoff_3d:
            continue
        persistent_gainer_alerts.append(row)
        _pro_alerted_gainers.add(sym)
        try:
            db_pro_mark_gainer_alerted(sym)
        except Exception as e:
            logger.warning("pro_scan: db_pro_mark_gainer_alerted failed for %s: %s", sym, e)

    if not persistent_gainer_alerts:
        return

    for chat_id in subscribers:
        try:
            for i, gr in enumerate(persistent_gainer_alerts[:5], 1):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=_pro_format_gainer_card(gr, rank=i)
                )
                await asyncio.sleep(0.4)
        except Exception as _e:
            logger.warning("pro_gainers_job send %s: %s", chat_id, _e)


# ── /pro Command handler ───────────────────────────────────────������───���������─────────────

def _pro_full_command_guide() -> str:
    """Single source of truth for the bot's full PUBLIC command list.
    Admin/hidden commands (/admin, /optimize, /xgtrain, /rftrain, /dbcheck)
    and secret commands are intentionally excluded."""
    return (
        "📖  SAKZ BOT — FULL COMMAND GUIDE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━��━━\n"
        "Everything the bot can do, in one place.\n\n"
        "���  SCANNING\n"
        "/scan                Full market scan (4H, top pairs)\n"
        "/scan BTC            Scan a specific pair\n"
        "/scan BTC 1h         Specific pair on a custom timeframe\n"
        "/scan new 24h        New listings scan (m/h/d/w units)\n"
        "/cscan ZEC           Custom pair scan (auto-detect TF)\n"
        "/cscan ZEC 15m       Custom pair on 15m / 1h / 4h / 1d\n"
        "/analyse ZEC         Raw analysis — no gates (⚠️ hot ground)\n"
        "/analyse ZEC 10m     Raw analysis on any timeframe\n"
        "/scalp               Scalp mode — 15M + 1H signals\n"
        "/scalp ETH           Scalp a specific pair\n"
        "/swing               Swing mode — 4H + 1D signals\n"
        "/swing BTC           Swing a specific pair\n"
        "/scanmid             Mid-market scan (ranks 51-200)\n"
        "/chart SOL 4h        Chart with full technical analysis\n\n"
        "🎯  SIGNALS\n"
        "/best                Highest confidence signal right now\n"
        "/top5                Top 5 current signals\n"
        "/top10               Top 10 current signals\n"
        "/filter LONG 8       Filter by bias + min confidence\n"
        "/confirm BTC         Re-validate a stored signal live\n"
        "/tg                  Top 10 gainers (24h)\n"
        "/tl                  Top 10 losers (24h)\n\n"
        "💼  TRADE MANAGER\n"
        "/pick                Track a trade with auto-reminders\n"
        "/stoptrade           Stop tracking your current trade\n"
        "/check BTC LONG 98000 95000\n"
        "                     Validate an open trade vs live data\n"
        "/pnl                 PnL calculator\n\n"
        "📊  ANALYTICS\n"
        "/compare             All signals vs current prices\n"
        "/compare BTC         One pair — signal vs current price\n"
        "/stats 168           Win rate stats (24 / 168 / 720 hrs)\n"
        "/lb 24               Leaderboard (24 / 168 hrs)\n"
        "/leaderboard 7       Best pairs (7-day leaderboard)\n"
        "/backtest 720        Backtest over a time window\n"
        "/fgi                 Fear & Greed Index + guidance\n"
        "/calibrate BTC       ATR param calibration (adv.)\n\n"
        "🔔  ALERTS & AUTOMATION\n"
        "/alert BTCUSDT 8     Alert when confidence >= 8\n"
        "/unalert BTCUSDT     Remove an alert\n"
        "/watch BTCUSDT 7     Watchlist — auto-notify on signal\n"
        "/unwatch BTCUSDT     Remove from watchlist\n"
        "/autoscan [tf]       Auto-scan on/off — e.g. /autoscan 4h, /autoscan off\n"
        "/broadcast on|off    Toggle signal auto-posting here\n"
        "/safemode            Toggle automatic dying-trend alerts\n\n"
        "🔬  PRO ALERT SUITE\n"
        "/pro                 Show this full command guide\n"
        "/pro on | /pro off   Subscribe / unsubscribe to PRO alerts\n"
        "/pro status          Live PRO tracking stats\n"
        "/pro alerts          PRO alert suite overview\n\n"
        "⚙���  GENERAL\n"
        "/status              Bot health & uptime\n"
        "/menu                Interactive command menu\n"
        "/start               Welcome message\n"
    )


# ═══════════════════════════════════════════════════════════════════════════
# PRIME — secret high-conviction subscription, per-user GMT alerts, 15-min cache
# ═════════��════���════���═══���═══�����══════════════════════════════════════════════
import sakz_prime as prime
import json as _json
import sakz_signal_logic as _SIGLOGIC   # FIX #SIGLOGIC-UNDEFINED — this module
# was referenced as _SIGLOGIC in the autoscan push loop (continuous_scan_job),
# the lifecycle maintenance job and the /pnl card path, but was never actually
# imported. Every one of those calls raised NameError at runtime, which (caught
# by the job/error handlers) silently killed autoscan delivery and lifecycle
# upkeep. Importing it here restores all three.
from sakz_db import (
    db_prime_subscribe, db_prime_unsubscribe, db_prime_is_subscribed,
    db_prime_set_offset, db_prime_get_user, db_prime_get_all_users,
    db_prime_set_slots, db_prime_get_slots, db_prime_toggle_slot,
    db_prime_cache_put, db_prime_cache_get,
    db_prime_alert_already_sent, db_prime_mark_alert_sent, db_prime_cleanup_alert_sent,
    db_lifecycle_track, db_lifecycle_active, db_lifecycle_mark,
    db_lifecycle_close, db_lifecycle_prune,
)

PRIME_UNLOCK_CODE = os.environ.get("PRIME_UNLOCK_CODE", "").strip()

# Alert slots are LOCAL hours (the user's GMT offset is applied by the scheduler)
_PRIME_SLOTS = [8, 14, 20]
_PRIME_SLOT_EMOJI = {8: "\U0001F305", 14: "\u2600\ufe0f", 20: "\U0001F319"}  # morning/afternoon/evening
_PRIME_SLOT_LABEL = {8: "08:00", 14: "14:00", 20: "20:00"}

# Timezone picker (label, GMT offset in hours; fractional allowed)
_PRIME_TZ_CHOICES = [
    ("GMT-8", -8), ("GMT-5", -5), ("GMT-3", -3),
    ("GMT+0", 0), ("GMT+1", 1), ("GMT+2", 2),
    ("GMT+3", 3), ("GMT+5:30", 5.5), ("GMT+8", 8),
    ("GMT+9", 9),
]


def _prime_signal_to_payload(r, score):
    """JSON-safe dict for caching/rendering one Prime pick (no datetime objects)."""
    return {
        "symbol": r.get("symbol"),
        "bias": r.get("bias"),
        "exchange": r.get("exchange", ""),
        "price": r.get("price"),
        "entry_low": r.get("entry_low"),
        "entry_high": r.get("entry_high"),
        "t1": r.get("t1"), "t2": r.get("t2"), "t3": r.get("t3"),
        "confidence": r.get("confidence"),
        "confidence_precise": r.get("confidence_precise"),
        "winning_score": r.get("winning_score"),
        "score": score,
    }


async def prime_refresh_job(context):
    """Every 15 min: rank the latest scan cache into the Prime top-picks cache.

    Two-pass so entry-distance decay uses LIVE prices without hammering the
    network: rank-without-decay -> shortlist -> fetch live prices for the
    shortlist only -> final rank with decay.
    """
    try:
        cache = getattr(state, "_scan_cache", None)
        results = (cache or {}).get("results") if cache else None
        if not results:
            results = getattr(state, "last_scan_results", None) or []
        if not results:
            db_prime_cache_put(_json_dumps_safe([]))
            return
        # Pass 1 — composite without decay to find the shortlist.
        shortlist = prime.rank_prime(results, current_prices=None,
                                     bar=prime.PRIME_BAR, top_n=8)
        live_prices = {}
        for entry in shortlist:
            sig = entry["signal"]; sym = sig.get("symbol")
            if not sym:
                continue
            try:
                px = await asyncio.to_thread(_get_live_price, sym, sig.get("exchange", ""))
                if px and px > 0:
                    live_prices[sym] = px
            except Exception:
                pass
        # Pass 2 — final ranking with entry-distance decay applied.
        ranked = prime.rank_prime(results, current_prices=live_prices,
                                  bar=prime.PRIME_BAR, top_n=prime.PRIME_TOP_N)
        picks = [_prime_signal_to_payload(x["signal"], x["score"]) for x in ranked]
        db_prime_cache_put(_json.dumps(picks))
        logger.info("PRIME refresh: %d picks cached (scanned %d signals)",
                    len(picks), len(results))
    except Exception as e:
        logger.warning("prime_refresh_job error: %s", e)


def _json_dumps_safe(obj):
    try:
        return _json.dumps(obj)
    except Exception:
        return "[]"


def _prime_get_cached_picks():
    row = db_prime_cache_get()
    if not row:
        return [], None
    try:
        picks = _json.loads(row["payload"])
    except Exception:
        picks = []
    return picks, row.get("refreshed_at")


def _prime_render_dashboard(picks, refreshed_at=None):
    if not picks:
        return ("\U0001F531 PRIME \u2014 Top Calls\n\n"
                f"{prime.PRIME_EMPTY_MESSAGE}\n\n{prime.PRIME_DISCLAIMER}")
    lines = ["\U0001F531 PRIME \u2014 Highest-Conviction Calls", ""]
    medals = ["\U0001F947", "\U0001F948", "\U0001F949"]
    for i, p in enumerate(picks):
        medal = medals[i] if i < len(medals) else f"#{i+1}"
        conf = p.get("confidence_precise")
        if conf is None:
            conf = p.get("confidence")
        try:
            conf_s = f"{float(conf):.1f}/10"
        except Exception:
            conf_s = "\u2014"
        bias = str(p.get("bias", "")).upper()
        arrow = "\U0001F7E2" if bias == "LONG" else "\U0001F534"
        try:
            entry_s = f"${float(p['entry_low']):.4f} \u2013 ${float(p['entry_high']):.4f}"
        except Exception:
            entry_s = "\u2014"
        lines.append(f"{medal} {arrow} {p.get('symbol','?')} {bias}  \u00b7  conf {conf_s}  \u00b7  Prime {p.get('score','?')}")
        lines.append(f"     Entry {entry_s}")
        try:
            lines.append(f"     \U0001F3AF {float(p['t1']):.4f} / {float(p['t2']):.4f} / {float(p['t3']):.4f}")
        except Exception:
            pass
        lines.append("")
    if refreshed_at:
        ts = str(refreshed_at).replace("T", " ")[:16]
        lines.append(f"\U0001F551 Updated {ts} \u00b7 refreshes every {prime.PRIME_CACHE_TTL_MIN}m")
    lines.append("")
    lines.append(prime.PRIME_DISCLAIMER)
    return "\n".join(lines)


def _prime_tz_keyboard():
    rows, cur = [], []
    for label, off in _PRIME_TZ_CHOICES:
        cur.append(InlineKeyboardButton(label, callback_data=f"prime_tz|{off}"))
        if len(cur) == 3:
            rows.append(cur); cur = []
    if cur:
        rows.append(cur)
    return InlineKeyboardMarkup(rows)


def _prime_main_keyboard(chat_id):
    slots = set(db_prime_get_slots(chat_id))
    slot_row = []
    for h in _PRIME_SLOTS:
        mark = "\u2705" if h in slots else "\u2B1C"
        slot_row.append(InlineKeyboardButton(
            f"{mark}{_PRIME_SLOT_EMOJI[h]}", callback_data=f"prime_slot|{h}"))
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("\U0001F504 Refresh", callback_data="prime_show|1"),
         InlineKeyboardButton("\U0001F30D Timezone", callback_data="prime_tzmenu|1")],
        slot_row,
        [InlineKeyboardButton("\U0001F515 Unsubscribe", callback_data="prime_off|1")],
    ])


async def prime_command(update, context):
    """/prime — secret subscription. Subscribe + timezone buttons + live dashboard."""
    chat_id = update.effective_chat.id
    args = list(getattr(context, "args", []) or [])
    # Secret unlock gate (only enforced if PRIME_UNLOCK_CODE is configured)
    if PRIME_UNLOCK_CODE and not db_prime_is_subscribed(chat_id):
        supplied = args[0].strip() if args else ""
        if supplied != PRIME_UNLOCK_CODE:
            await update.message.reply_text(
                "\U0001F531 PRIME is invite-only.\nEnter the access code:  /prime <code>")
            return
    newly = not db_prime_is_subscribed(chat_id)
    db_prime_subscribe(chat_id)
    if newly:
        await update.message.reply_text(
            "\U0001F531 Welcome to PRIME \u2014 you'll receive only the highest-conviction calls.\n\n"
            "First, choose your timezone so alerts arrive at the right local time:",
            reply_markup=_prime_tz_keyboard())
        return
    picks, refreshed = _prime_get_cached_picks()
    await update.message.reply_text(
        _prime_render_dashboard(picks, refreshed),
        reply_markup=_prime_main_keyboard(chat_id))


async def prime_tz_callback(update, context):
    query = update.callback_query
    chat_id = query.from_user.id
    off = float(query.data.split("|", 1)[1])
    db_prime_set_offset(chat_id, off)
    if not db_prime_get_slots(chat_id):
        db_prime_set_slots(chat_id, _PRIME_SLOTS)  # enable all 3 by default
    await query.answer(f"Timezone set to {prime.format_offset(off)}")
    await query.edit_message_text(
        f"\u2705 Timezone: {prime.format_offset(off)}\n\n"
        "Default alert times enabled (\U0001F305 08:00 \u00b7 \u2600\ufe0f 14:00 \u00b7 \U0001F319 20:00 local).\n"
        "Toggle the times below, or tap Refresh to see live calls.",
        reply_markup=_prime_main_keyboard(chat_id))


async def prime_tzmenu_callback(update, context):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("\U0001F30D Choose your timezone:",
                                  reply_markup=_prime_tz_keyboard())


async def prime_slot_callback(update, context):
    query = update.callback_query
    chat_id = query.from_user.id
    h = int(query.data.split("|", 1)[1])
    new_slots = db_prime_toggle_slot(chat_id, h)
    label = ", ".join(_PRIME_SLOT_LABEL.get(s, f"{s:02d}:00") for s in new_slots) or "none"
    await query.answer(f"Alerts: {label}")
    try:
        await query.edit_message_reply_markup(reply_markup=_prime_main_keyboard(chat_id))
    except Exception:
        pass


async def prime_show_callback(update, context):
    query = update.callback_query
    await query.answer()
    chat_id = query.from_user.id
    picks, refreshed = _prime_get_cached_picks()
    try:
        await query.edit_message_text(_prime_render_dashboard(picks, refreshed),
                                      reply_markup=_prime_main_keyboard(chat_id))
    except Exception:
        pass


async def prime_off_callback(update, context):
    query = update.callback_query
    await query.answer("Unsubscribed")
    chat_id = query.from_user.id
    db_prime_unsubscribe(chat_id)
    await query.edit_message_text("\U0001F515 You've unsubscribed from PRIME. Use /prime to rejoin anytime.")


async def prime_alert_scheduler_job(context):
    """Runs every 60s. Sends each user their Top-3 at their chosen LOCAL slots."""
    now_utc = datetime.utcnow()
    try:
        users = db_prime_get_all_users()
    except Exception as e:
        logger.warning("prime scheduler: user fetch failed: %s", e)
        return
    if not users:
        return
    picks, refreshed = _prime_get_cached_picks()
    for u in users:
        chat_id = u["chat_id"]
        off = u.get("gmt_offset", 0) or 0
        try:
            slots = db_prime_get_slots(chat_id)
        except Exception:
            continue
        if not slots:
            continue
        for slot in prime.due_slots(now_utc, off, slots):
            key = prime.slot_dedup_key(chat_id, now_utc, off, slot)
            if db_prime_alert_already_sent(key):
                continue
            db_prime_mark_alert_sent(key)
            text = ("\U0001F531 PRIME ALERT \u2014 your scheduled call check\n\n"
                    + _prime_render_dashboard(picks, refreshed))
            try:
                await context.bot.send_message(chat_id=chat_id, text=text)
            except Exception as e:
                logger.warning("prime alert send failed chat=%s: %s", chat_id, e)


async def prime_cleanup_job(context):
    try:
        db_prime_cleanup_alert_sent(days=3)
    except Exception as e:
        logger.warning("prime cleanup error: %s", e)


async def pro_command(update, context):
    """
    /pro           — show the FULL command guide (all public commands + usage)
    /pro on|off    — subscribe / unsubscribe to PRO alerts
    /pro status    — show live tracking stats
    /pro alerts    — PRO alert suite overview
    """
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args or []
    cmd     = args[0].lower() if args else "guide"

    # Bare /pro (and /pro help|commands|manual|guide) shows the FULL command guide.
    if cmd in ("guide", "help", "commands", "command", "manual", "menu", "list", ""):
        await update.message.reply_text(_pro_full_command_guide())
        return

    # /pro alerts → PRO alert suite overview
    if cmd in ("alerts", "alert", "suite", "info"):
        await update.message.reply_text(
            "🔬 PRO ALERT SUITE — OVERVIEW\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🚀 SUSTAINED UPTREND TRACKER\n"
            "   Perps gaining ≥8%/day for 20h–3 days.\n"
            "   Alerts when the streak is confirmed organic.\n\n"
            "🏆 TOP-10 GAINER PERSISTENCE\n"
            "   Tokens holding top-10 for 1–3 days —\n"
            "   sustained institutional buying interest.\n\n"
            "🚨 SCAM PUMP DETECTOR\n"
            "   Wash trading, stop hunts, parabolic pumps,\n"
            "   micro-cap manipulation. Warns you BEFORE\n"
            "   you get dumped on.\n\n"
            "Commands:\n"
            "  /pro on/off   — toggle alerts on/off\n"
            "  /pro status   — live tracking stats\n"
            "  /pro alerts   — this overview\n"
            "  /pro          — full command guide\n\n"
            "🚀 Uptrend + 🚨 Manipulation: scans every 30 min\n"
            "🏆 Gainer persistence: checked every 4 hours."
        )
        return

    if cmd == "status":
        sub    = db_pro_is_subscribed(chat_id)
        ut_ct  = len([r for r in db_pro_get_uptrend_rows() if not r["alert_sent"]])
        g_ct   = len([r for r in db_pro_get_gainer_rows()
                      if not r["alert_sent"] and r["times_top10"] >= 2])
        m_ct   = len([r for r in db_pro_get_manip_rows()
                      if not r["alert_sent"] and r["manip_score"] >= 60])
        badge  = "✅ ACTIVE" if sub else "❌ OFF"
        note   = ("Alerts will fire to you when conditions are met."
                  if sub else "Use /pro on to subscribe.")
        await update.message.reply_text(
            "🔬 PRO SUITE — STATUS\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📡 Your subscription: {badge}\n"
            f"{note}\n\n"
            "📊 LIVE TRACKING STATS\n"
            f"  🚀 Tokens in uptrend window:  {ut_ct}\n"
            f"  🏆 Persistent top-10 gainers: {g_ct}\n"
            f"  🚨 Active manipulation flags: {m_ct}\n\n"
            "⚡ Uptrend/Manipulation: next scan ≤30 min\n"
            "🏆 Gainers: next check ≤4 hours\n"
            "Use /pro on / /pro off to manage your subscription."
        )
        return

    # Toggle ─────────────────────��──────────────────────────────────────────────
    subscribed = db_pro_is_subscribed(chat_id)
    if cmd == "on":
        want_off = False
    elif cmd == "off":
        want_off = True
    else:                       # /pro toggle / subscribe / unsubscribe → flip state
        want_off = subscribed

    if want_off:
        if subscribed:
            db_pro_unsubscribe(chat_id)
        await update.message.reply_text(
            "❌ PRO ALERTS DISABLED\n\n"
            "You've been unsubscribed from the PRO alert suite.\n"
            "Use /pro on to re-enable at any time.\n"
            "Your scan features (/scan, /cscan etc.) are unaffected."
        )
    else:
        if not subscribed:
            db_pro_subscribe(chat_id)
        await update.message.reply_text(
            "✅ PRO ALERTS ACTIVATED!\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "You're subscribed to three live monitoring streams:\n\n"
            "🚀 SUSTAINED UPTREND TRACKER\n"
            "   Perps trending ≥8%/day for 20h–3 days\n\n"
            "🏆 TOP-10 GAINER PERSISTENCE\n"
            "   Tokens holding the top-10 list for 1–3 days\n\n"
            "🚨 SCAM PUMP DETECTOR\n"
            "   Inorganic chart patterns & manipulation flags\n\n"
            "⏱ Uptrend + Manipulation: every 30 min\n"
            "   Gainers persistence: every 4 hours\n"
            "   Alerts fire the moment conditions are met.\n\n"
            "Use /pro status to see live tracking stats.\n"
            "Use /pro alerts for feature details.\n"
            "Use /pro off to disable."
        )


# ─────────────────────────────────────────────
# GLOBAL IN-MEMORY STATE
# (loaded from DB on startup, kept in sync)
# ─────────────────────────────────────────────

# ── Safe Mode state ────────────────────────────────────────────────────────────
# chat_ids with safe mode ON (loaded from DB on startup)
# Last signals each safemode user received from any scan source
# { chat_id: [ signal_dict, ... ] }  — capped at 20 signals per user
safemode_last_signals: dict = {}
_SAFEMODE_MAX_SIGNALS = 20    # how many recent signals to monitor per user




def _safemode_store_signals(chat_id: int, signals: list):
    """
    Called whenever a scan result is presented to a safemode user.
    Stores up to _SAFEMODE_MAX_SIGNALS recent signals for trend monitoring.
    """
    if chat_id not in state.safemode_users:
        return
    existing = safemode_last_signals.get(chat_id, [])
    # Merge — avoid duplicate symbols, prefer newest signal per symbol
    sym_map = {s['symbol']: s for s in existing}
    for sig in signals:
        sym_map[sig.get('symbol', '')] = sig
    merged = list(sym_map.values())
    # Keep only newest _SAFEMODE_MAX_SIGNALS
    safemode_last_signals[chat_id] = merged[-_SAFEMODE_MAX_SIGNALS:]


# ── Concurrency & caching ────────────────────��
from concurrent.futures import ThreadPoolExecutor
import threading

# SAKZ_PERF_V2 — concurrency widened so many users can scan / pull PnL cards at
# the same time without queuing behind one another.
#   * SCAN_EXECUTOR  — network-bound (requests release the GIL), so a larger
#     pool lets several users' scans + price lookups run truly in parallel.
#   * RENDER_EXECUTOR — image work. matplotlib's pyplot state is NOT thread-safe,
#     so the actual drawing is guarded by _PLT_LOCK below. The pool is still
#     widened so the network prep (live price + candles) for each PnL card runs
#     in parallel; only the short draw section serializes.
SCAN_EXECUTOR   = ThreadPoolExecutor(max_workers=12, thread_name_prefix="scan")
RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=4,  thread_name_prefix="render")

# Global lock that serializes every matplotlib/pyplot drawing section. Hold it
# only around the actual figure build + savefig, never around network I/O.
_PLT_LOCK = threading.Lock()


def _plt_locked(fn, *args, **kwargs):
    """Run a matplotlib-drawing function while holding _PLT_LOCK.

    pyplot keeps global state, so only one figure may be built at a time even
    though several render workers prepare data in parallel. Do NOT nest calls to
    this helper (the lock is non-reentrant)."""
    with _PLT_LOCK:
        return fn(*args, **kwargs)

CACHE_TTL_SECS = 600                                  # 10 minutes — matches continuous_scan_job interval

# Global scan cache — served to users who scan within TTL of last scan
_scan_cache_lock   = asyncio.Lock()   # prevents cache stampede

# ── BTC Market Regime Cache ───────────────────
# Cached once per scan so all 150 pairs share the same regime read.
# { 'regime': 'BULL'|'BEAR'|'NEUTRAL', 'time': datetime }
_btc_regime_cache_ttl = 900  # 15 minutes — same as scan cache

# ── BTC regime-shift market alert (hybrid: regime flip + price confirmation) ──
# The watcher fires ONLY on a genuine BTC direction change (bullish<->bearish),
# not on every wiggle. Anti-spam is the state-change itself — no fixed cooldown.
_BTC_ALERT_HOLD_SECONDS = 600   # new direction must persist 10 min before broadcasting (whipsaw guard)
_BTC_ALERT_CONFIRM_PCT  = 0.2   # BTC must move >=0.2% in the regime direction over the hold window
_BTC_BULL_REGIMES = ("STRONG_BULL", "BULL")
_BTC_BEAR_REGIMES = ("STRONG_BEAR", "BEAR")

# ── BTC Dominance Cache ─────────────────────────��───��─��───��──���───����������─��─��─��─��
# BTC.D rising = capital flowing out of alts → penalise altcoin LONGs
# Fetched from Bybit BTCDOMUSDT or Binance BTCDOMUSDT (may not always be available)
# { 'btcd': float, 'trend': 'rising'|'falling'|'flat', 'time': datetime }
_BTCD_TTL = 1800  # 30 min — BTC.D doesn't move that fast


# Portfolio-level correlation summary — updated after each scan
# { btc_regime, total_signals, corr_long, corr_short, independent,
#   corr_slot_cap, over_cap, risk_level }

# Per-chat locks — lets each user run independently without blocking others
_chat_scan_locks   = {}     # { chat_id: asyncio.Lock() }

def get_chat_lock(chat_id):
    if chat_id not in _chat_scan_locks:
        _chat_scan_locks[chat_id] = asyncio.Lock()
    return _chat_scan_locks[chat_id]

# ─────────────────────────────────────────────
# LEVERAGE CALCULATOR
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# BYBIT API
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
# ML MODEL AVAILABILITY FLAGS
# Graceful fallback — bot runs normally if these
# optional modules are not installed.
# ───────────────────────────────────��─────────
try:
    from xgboost_train import predict_signal as xgb_predict_signal, train as xgb_train, model_meta as xgb_model_meta
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost_train not found — XGBoost ML scoring disabled")

try:
    from rf_train import (
        predict_signal as rf_predict_signal,
        train          as rf_train_model,
        model_meta     as rf_model_meta,
        consensus_verdict,
    )
    _RF_AVAILABLE = True
except ImportError:
    _RF_AVAILABLE = False
    logger.warning("rf_train not found — RF ML scoring disabled")

try:
    from optimize import run_hyperopt
    _OPTIMIZE_AVAILABLE = True
except ImportError:
    _OPTIMIZE_AVAILABLE = False
    logger.warning("optimize.py not found — /optimize disabled")

# ───────────��─────────────────────────────────
# HISTORICAL WALK-FORWARD BACKTEST — sakz_backtest_hist.py
# ─────────────────────────────────────────────
try:
    from sakz_backtest_hist import run_hist_backtest, format_hist_result
    _HIST_BT_AVAILABLE = True
    logger.info("Historical backtest module loaded")
except ImportError:
    _HIST_BT_AVAILABLE = False
    logger.warning("sakz_backtest_hist.py not found — /btfull disabled")

# ─────────────────────────────────────────────
# AUTO PAPER TRADING — sakz_paper.py
# ─────��───────────────────────���───────────────
try:
    from sakz_paper import (
        paper_init_db, paper_maybe_open, paper_mark_all,
        paper_close_expired, paper_get_open, paper_summary,
        format_paper_open, format_paper_summary,
        PAPER_CONF_MIN, PAPER_MAX_OPEN,
    )
    _PAPER_AVAILABLE = True
    logger.info("Paper trading module loaded")
except ImportError:
    _PAPER_AVAILABLE = False
    logger.warning("sakz_paper.py not found — /paper disabled")

# ─────────────���───────────────────────────────
# REAL-TIME WEBSOCKET LAYER — sakz_ws.py
# Streams live price, funding, liquidation, volume spikes from MEXC.
# ws_price() / ws_funding() are used as a fast cache before REST fallback.
# ────────────────────────────────────���─────���──
try:
    from sakz_ws import (
        start_ws,
        ws_price,
        ws_funding,
        ws_status,
    )
    _WS_AVAILABLE = True
    logger.info("WebSocket layer loaded")
except ImportError:
    _WS_AVAILABLE = False

    def ws_price(symbol):  return None
    def ws_funding(symbol): return None
    def ws_status():        return {"ws_available": False}

    logger.warning("sakz_ws.py not found — WebSocket layer disabled, using REST only")

BEST_PARAMS_PATH = "best_params.json"


def _load_best_params():
    """Load optimized params from best_params.json if available."""

    params = state.DEFAULT_OPTIM_PARAMS.copy()
    try:
        if os.path.exists(BEST_PARAMS_PATH):
            with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                params.update({k: loaded[k] for k in loaded.keys() if k in params})
                logger.info("Loaded optimization params from %s", BEST_PARAMS_PATH)
    except Exception as e:
        logger.warning("Failed to load %s: %s", BEST_PARAMS_PATH, e)
    state.OPTIM_BEST_PARAMS = params








# ─────────────────────────────────────────────
# MEXC API
# ─────────────────────────────────────────────






# ───────���─────��������──────────────────────────────
# ───────────────────────────────────���─────────
# IMPROVEMENT #7 — BINANCE PERPETUALS (free public API)
# Graceful skip if Binance blocks Railway's IP.
# ─────────────────────────────────────────────









# ── WS-FIRST PRICE AND FUNDING HELPERS ───────────────────────────────────────
# All price lookups go through here. Tries the WebSocket cache first (instant,
# zero network cost). Falls back to REST only if WS hasn't seen the symbol yet.
# This is the single change that eliminates REST calls for price during tracking,
# PnL updates, price alerts, and any other live-price read in the bot.

_PRICE_CACHE = {}          # symbol -> (price, ts); short-TTL REST de-dupe
_PRICE_CACHE_TTL = 3.0     # seconds; WS cache still short-circuits first

def _get_live_price(symbol: str, exchange: str = '') -> float:
    """
    WS-first price lookup with a short REST cache. Returns float (0 on failure).
    exchange hint: 'BYBIT' | 'BINANCE' | 'MEXC' — used only for REST fallback.
    """
    # 1. WebSocket cache (sub-millisecond, no network)
    cached = ws_price(symbol)
    if cached and cached > 0:
        return cached
    # 2. Short-TTL REST cache — avoids hammering REST on repeated calls
    _now = time.time()
    _entry = _PRICE_CACHE.get(symbol)
    if _entry is not None and (_now - _entry[1]) < _PRICE_CACHE_TTL:
        return _entry[0]
    # 3. REST fallback — ordered by preference / availability
    exch = (exchange or '').upper()
    price = 0
    if exch == 'BYBIT' or (not exch and sakz_exchanges.BYBIT_AVAILABLE is not False):
        p = bybit_get_current_price(symbol)
        if p and p > 0:
            price = p
    if not price and (exch == 'BINANCE' or (not exch and sakz_exchanges.BINANCE_AVAILABLE)):
        p = binance_get_current_price(symbol)
        if p and p > 0:
            price = p
    if not price:
        p = mexc_get_current_price(symbol)
        if p and p > 0:
            price = p
    if price > 0:
        _PRICE_CACHE[symbol] = (price, _now)
    return price


def _get_live_funding(symbol: str, exchange: str = '') -> float:
    """
    WS-first funding rate lookup. Returns float (0 on failure).
    """
    cached = ws_funding(symbol)
    if cached is not None:
        return cached

    exch = (exchange or '').upper()
    if exch == 'BYBIT' or not exch:
        return bybit_fetch_funding(symbol)
    if exch == 'BINANCE':
        return binance_fetch_funding(symbol)
    return 0

def analyze_binance(symbol):
    if not sakz_exchanges.BINANCE_AVAILABLE:
        return None
    try:
        df4h = binance_fetch_ohlcv(symbol, '4h', 100)
        if df4h is None or len(df4h) < 52: return None
        df1d = binance_fetch_ohlcv(symbol, '1d', 60)
        if df1d is None or len(df1d) < 22: return None
        df4h = add_indicators(df4h, timeframe="4h")
        df1d = add_indicators(df1d, timeframe="1d")
        if df4h is None or df1d is None: return None
        if len(df4h) < 5 or len(df1d) < 5: return None
        funding = binance_fetch_funding(symbol)
        result  = score_pair(df4h, df1d, funding, symbol)
        if isinstance(result, dict) and result.get('regime_blocked'):
            return None   # HARD REGIME GATE — never surface in autoscan/scan
        if result:
            result['exchange'] = 'BINANCE'
        return result
    except Exception:
        return None


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────






# ─────────────────────────────────────────────
# BTC MARKET REGIME GATE
# FIX #RG  — Original: EMA + RSI only (2-variable classifier).
# FIX #RG2 — Upgraded: 5-factor classifier with ADX, EMA slope,
#             crossover recency, and 5 regime levels.
#
# Problems with the 2-variable version:
#   • RSI 45–55 satisfied BOTH bull (>45) and bear (<55) conditions
#     simultaneously — the gate was ambiguous for ~25% of market time.
#   • NEUTRAL got only -1 conf penalty despite being the most dangerous
#     regime (choppy markets produce the most false signals).
#   • No trend strength measurement: EMA20 > EMA50 after 1 candle and
#     after 200 candles looked identical to the classifier.
#   • No awareness of whether the trend was accelerating or dying.
#
# New 5-factor logic:
#   Factor 1 — EMA alignment    : direction of trend
#   Factor 2 — ADX (14-period)  : is there actually a trend (>20)?
#   Factor 3 — EMA gap slope    : is trend strengthening or weakening?
#   Factor 4 — Crossover recency: how many candles since EMA cross?
#   Factor 5 — RSI              : momentum confirmation (tighter bands)
#
# Regime levels (5):
#   STRONG_BULL  EMA aligned + ADX>25 + gap widening + RSI>50
#   BULL         EMA aligned + (ADX>20 OR recent cross) + RSI>45
#   STRONG_BEAR  Mirror of STRONG_BULL
#   BEAR         Mirror of BULL
#   NEUTRAL      Anything else — choppy, transitioning, or exhausting
#
# Gate behaviour per regime:
#   STRONG_BULL  → LONGs free, SHORTs blocked below conf 9
#   BULL         → LONGs free, SHORTs blocked below conf 8
#   STRONG_BEAR  → SHORTs free, LONGs blocked below conf 9
#   BEAR         → SHORTs free, LONGs blocked below conf 8
#   NEUTRAL      → All signals blocked below conf 8 (not just -1 penalty)
#
# Cached 15 minutes — same as before.
# ──────────────���────────────────��─────────────
# ── BTC spot price cache (for regime invalidation) ────────────────────────────
_BTC_PRICE_TTL = 60           # refresh every 60 s





# ─────────────────────────────────────────────
# FEATURE: SESSION AWARENESS
# ─────────────────────────────────────────────
# Crypto sessions (all times UTC):
#   ASIAN    00:00–08:00  low volatility, ranging, fake-out prone
#   LONDON   07:00–12:00  manipulation wicks, trend initiation
#   OVERLAP  12:00–17:00  NY + London overlap — highest real volume, trend continuation
#   NY       13:00–21:00  trend follow-through, volume peaks
#   DEAD     21:00–00:00  closing, thin — signals here often reverse at London open
#
# Session windows overlap deliberately: 07:00–12:00 is tagged LONDON even though
# it's technically still mid-Asian-close, because the London open is the dominant
# driver in that window.  The OVERLAP tag is given to 12:00–17:00 only, when both
# London and NY are simultaneously active — this is the cleanest trend window.
# ────────────────────────���────────────────────


# ────────────────────���────────────────────────
# FEATURE: FIBONACCI RETRACEMENT CONFLUENCE
# ─────────────────────────────────────────────
# Auto-compute 0.382, 0.5, and 0.618 Fibonacci retracement levels between
# the last major swing high and swing low in the primary TF dataframe.
# When a pivot S/R level from find_pivot_support/resistance coincides with
# a Fib level within a tight tolerance (0.5% of price), that zone is confluent —
# a significantly stronger structural level than either signal alone.
#
# A confluent LONG near the 0.618 retrace of a recent swing is textbook support.
# A confluent SHORT near the 0.382 retrace of a downswing is textbook resistance.
# ─────────────────────────────────────────────
_FIB_RATIOS = (0.236, 0.382, 0.500, 0.618, 0.786)



# ─────────────────────────────────────────────
# SCORING ENGINE
# ───────────────────────────────────��─────────


# ── FIX #RR-GATE — Minimum entry Risk:Reward enforcement ──────────────────
# sakz_risk.py sizes positions well (Kelly, drawdown gates, correlation caps)
# and sakz_paper._write_close measures *realised* R:R (net_pct / risk_pct)
# AFTER a trade closes — but nothing stopped a structurally poor setup (e.g.
# T1 only 0.8R away while SL sits a full 1R out) from firing in the first
# place. confirm_rr() is the pre-trade gate: a signal may only fire if its
# entry->T1 reward is at least `min_rr`x the entry->SL risk.
_MIN_SIGNAL_RR = max(0.1, float(os.getenv("MIN_SIGNAL_RR", "1.5")))








# ──────────────────────────────────────��──────
# MULTI-TIMEFRAME ENGINE
# ───────��─────────────────────────────────────
# TF_CONFIGS defines every supported timeframe.
# Each entry specifies:
#   primary      — candle interval fed to score_pair as df4h (the "fast" TF)
#   confirm      — confirmation candle interval fed as df1d (the "slow" TF)
#   bybit_pri    — Bybit API interval string for primary
#   bybit_con    — Bybit API interval string for confirmation
#   mexc_pri     �� MEXC interval string for primary
#   mexc_con     — MEXC interval string for confirmation
#   binance_pri  — Binance interval string for primary
#   binance_con  — Binance interval string for confirmation
#   min_candles  — minimum closed candles required on the primary TF
#   label        ��� human-readable label shown in signals
#   hold_map     — dur_score → (hold_hours, hold_label) overrides for this TF
#   weight       — confidence multiplier when auto-selecting best TF
#                  (lower TFs get slight penalty to avoid noise dominance)
#   stoch_window — Stochastic window appropriate for this TF
# ─────────────────────────────────────────────
TF_CONFIGS = {
    '15m': {
        'label':       '15M',
        'bybit_pri':   '15',    'bybit_con':   '60',
        'mexc_pri':    '15m',   'mexc_con':    '1h',
        'binance_pri': '15m',   'binance_con': '1h',
        'min_pri':     40,      'min_con':     20,
        'stoch_window': 5,
        'weight':      0.80,    # short TF noisier — slight penalty
        'hold_map': {
            # dur_score → (hours, label)
            9:  (2,   '~2 hours'),
            7:  (1,   '~1 hour'),
            5:  (0.5, '~30 min'),
            3:  (0.25,'~15 min'),
            1:  (0.1, '~5-10 min'),
            0:  (0.07,'~5 min scalp'),
        },
    },
    '1h': {
        'label':       '1H',
        'bybit_pri':   '60',    'bybit_con':   '240',
        'mexc_pri':    '1h',    'mexc_con':    '4h',
        'binance_pri': '1h',    'binance_con': '4h',
        'min_pri':     40,      'min_con':     20,
        'stoch_window': 5,
        'weight':      0.90,
        'hold_map': {
            9:  (8,  '~8 hours'),
            7:  (5,  '~5 hours'),
            5:  (3,  '~3 hours'),
            3:  (1.5,'~90 min'),
            1:  (0.5,'~30 min'),
            0:  (0.25,'~15 min scalp'),
        },
    },
    '4h': {
        'label':       '4H',
        'bybit_pri':   '240',   'bybit_con':   'D',
        'mexc_pri':    '4h',    'mexc_con':    '1d',
        'binance_pri': '4h',    'binance_con': '1d',
        'min_pri':     30,      'min_con':     15,
        'stoch_window': 5,
        'weight':      1.00,    # baseline — no penalty
        'hold_map': {           # same as existing score_pair buckets
            9:  (72, '~3 days'),
            7:  (48, '~2 days'),
            5:  (24, '~1 day'),
            3:  (10, '~10 hours'),
            1:  (4,  '~4 hours'),
            0:  (1,  '~1 hour'),
        },
    },
    '1d': {
        'label':       '1D',
        'bybit_pri':   'D',     'bybit_con':   'W',
        'mexc_pri':    '1d',    'mexc_con':    '1d',   # MEXC has no weekly; use 1d×confirm
        'binance_pri': '1d',    'binance_con': '1w',
        'min_pri':     30,      'min_con':     10,
        'stoch_window': 14,
        'weight':      0.95,
        'hold_map': {
            9:  (336, '~2 weeks'),
            7:  (168, '~1 week'),
            5:  (96,  '~4 days'),
            3:  (48,  '~2 days'),
            1:  (24,  '~1 day'),
            0:  (12,  '~12 hours'),
        },
    },
}

# Aliases so users can type /cscan ZEC 15 or /cscan ZEC h1 etc.
TF_ALIASES = {
    '15':  '15m', '15m': '15m', '15min': '15m',
    '1h':  '1h',  '60m': '1h', '60': '1h', 'h1': '1h', '1': '1h',
    '4h':  '4h',  '240m': '4h', '240': '4h', 'h4': '4h', '4': '4h',
    '1d':  '1d',  'daily': '1d', 'd': '1d', 'day': '1d',
}


def _mtf_hold(dur_score, tf_key):
    """Return (hold_hours, hold_label) for a given dur_score and timeframe."""
    hold_map = TF_CONFIGS[tf_key]['hold_map']
    for threshold in sorted(hold_map.keys(), reverse=True):
        if dur_score >= threshold:
            return hold_map[threshold]
    return hold_map[min(hold_map.keys())]


def _fetch_tf_candles(exchange, symbol, tf_key, role='pri'):
    """
    Fetch OHLCV candles for a given exchange, symbol, timeframe, and role (pri/con).
    Returns a DataFrame or None.
    role: 'pri' = primary (fast) TF, 'con' = confirmation (slow) TF
    """
    cfg    = TF_CONFIGS[tf_key]
    prefix = 'bybit' if exchange == 'BYBIT' else ('mexc' if exchange == 'MEXC' else 'binance')
    iv_key = f'{prefix}_{role}'
    interval   = cfg[iv_key]
    min_needed = cfg[f'min_{role}']

    try:
        if exchange == 'BYBIT':
            df = bybit_fetch_ohlcv(symbol, interval, max(min_needed + 20, 100))
        elif exchange == 'MEXC':
            df = mexc_fetch_ohlcv(symbol, interval, max(min_needed + 20, 100))
        else:
            df = binance_fetch_ohlcv(symbol, interval, max(min_needed + 20, 100))

        if df is None:
            return ScanFailure(REASON_NO_CONTRACT, exchange=exchange, tf=tf_key,
                               detail=f"{exchange} returned no data for {symbol} [{interval}]")
        if len(df) < min_needed:
            return ScanFailure(REASON_SHORT_HISTORY, exchange=exchange, tf=tf_key,
                               detail=f"{exchange} {symbol}: only {len(df)} candles (need {min_needed}) on {interval}")
        return df
    except Exception as e:
        logger.debug("_fetch_tf_candles %s %s %s %s: %s", exchange, symbol, tf_key, role, e)
        return ScanFailure(REASON_NO_CONTRACT, exchange=exchange, tf=tf_key,
                           detail=f"{exchange} {symbol} fetch exception: {e}")


# ── NEW-LISTING DYNAMIC SCAN ─────────────────────�����────────────────────────────
# Handles tokens listed minutes or hours ago that don't yet have enough
# candle history for the standard 4H + 1D analysis.
#
# Strategy (on-demand only — never fires in auto-scans):
#   1. Try MEXC first (Bybit/Binance geo-blocked on this hosting region)
#   2. Walk down from 5m → 15m → 1h → 4h, accepting the
#      shortest TF that has ≥ MIN_CANDLES candles.
#   3. Use the same df as BOTH primary and confirmation when no higher
#      TF data exists (the confirmation adds structural weight from
#      the same data, which is honest — the bot tells the user this).\
#   4. Relax the score_pair minimum from 5 rows to 3 rows post-indicators.
#   5. Tag the result with 'new_listing': True so the signal card can
#      surface a prominent risk warning.
#
# Min candles per TF (absolute floor — new/ultra-early listings):
#   5m  → 3 candles  (~15 min of data — earliest possible signal)
#   15m → 10 candles (~2.5 hours)
#   1h  → 6 candles  (~6 hours)
#   4h  → 4 candles  (~16 hours)
# ───────────────────���─────────────────────────────────────────────────────────

# Min candles per TF (much lower than normal — new listings have thin history)
#   5m  → 3 candles  (~15 minutes of data — ultra-early floor)
#   15m → 10 candles (~2.5 hours)
#   1h  → 6 candles  (~6 hours)
#   4h  → 4 candles  (~16 hours)
_NL_MIN_CANDLES = {'5m': 3, '15m': 10, '1h': 6, '4h': 4}
_NL_INTERVALS = {
    'MEXC':    {'5m': 'Min5', '15m': 'Min15', '1h': 'Min60', '4h': 'Hour4'},
    'BYBIT':   {'5m': '5',    '15m': '15',    '1h': '60',     '4h': '240'},
    'BINANCE': {'5m': '5m',   '15m': '15m',   '1h': '1h',     '4h': '4h'},
}
# Migrated onto the adapter layer. Behaviour-identical: each adapter.fetch_ohlcv
# delegates to the same *_fetch_ohlcv function (an explicit interval is always
# passed here, so the adapter's default-interval fallback never triggers).
_NL_FETCH = {
    'MEXC':    lambda sym, iv, lim: get_adapter('MEXC').fetch_ohlcv(sym, iv, lim),
    'BYBIT':   lambda sym, iv, lim: get_adapter('BYBIT').fetch_ohlcv(sym, iv, lim),
    'BINANCE': lambda sym, iv, lim: get_adapter('BINANCE').fetch_ohlcv(sym, iv, lim),
}
_NL_TF_LABEL = {'5m': '5M', '15m': '15M', '1h': '1H', '4h': '4H'}

def analyze_symbol_new_listing(symbol, funding=0.0):
    """
    On-demand scan for brand-new or low-history pairs.

    Returns a signal dict tagged with new_listing=True, or a ScanFailure.
    Never called by auto-scans — only by cscan_tf_command when the standard
    MTF scan returns SHORT_HISTORY or NO_CONTRACT.
    """
    exchange_order = []
    if sakz_exchanges.BYBIT_AVAILABLE is not False:
        exchange_order.append('BYBIT')
    exchange_order.append('MEXC')
    if sakz_exchanges.BINANCE_AVAILABLE:
        exchange_order.append('BINANCE')

    best_result = None
    all_failures = []

    for exch in exchange_order:
        fetch_fn  = _NL_FETCH[exch]
        intervals = _NL_INTERVALS[exch]

        for tf_short in ['5m', '15m', '1h', '4h']:
            min_c    = _NL_MIN_CANDLES[tf_short]
            iv       = intervals[tf_short]
            # Fetch with the exchange-native interval string
            # mexc_fetch_ohlcv already accepts 'Min15' style via interval_map
            # but we need the standard keys for bybit/binance — pass native strings
            try:
                if exch == 'MEXC':
                    df = mexc_fetch_ohlcv(symbol, tf_short, min_c + 10)
                elif exch == 'BYBIT':
                    df = bybit_fetch_ohlcv(symbol, iv, min_c + 10)
                else:
                    df = binance_fetch_ohlcv(symbol, iv, min_c + 10)
            except Exception as e:
                all_failures.append(ScanFailure(REASON_NO_CONTRACT, exchange=exch, tf=tf_short,
                                                detail=str(e)))
                continue

            if df is None or len(df) < min_c:
                got = len(df) if df is not None else 0
                all_failures.append(ScanFailure(REASON_SHORT_HISTORY, exchange=exch, tf=tf_short,
                                                detail=f"only {got} candles (need {min_c}) on {tf_short}"))
                continue

            # Add indicators — stochastic window 5 for speed
            df_ind = add_indicators(df, timeframe='4h')
            if df_ind is None or len(df_ind) < 3:
                all_failures.append(ScanFailure(REASON_SHORT_HISTORY, exchange=exch, tf=tf_short,
                                                detail="add_indicators produced < 3 rows"))
                continue

            # Use the same df for both primary and confirmation.
            # This is honest — there is no higher-TF context yet.
            # score_pair will derive daily indicators from the same candles
            # (treated as the "daily" confirmation with the same data).
            try:
                result = score_pair(df_ind, df_ind, funding, symbol)
            except Exception as e:
                all_failures.append(ScanFailure(REASON_NO_CONTRACT, exchange=exch, tf=tf_short,
                                                detail=f"score_pair error: {e}"))
                continue

            if isinstance(result, ScanFailure):
                result.exchange = result.exchange or exch
                result.tf       = result.tf or tf_short
                all_failures.append(result)
                continue
            if result is None:
                all_failures.append(ScanFailure(REASON_NEUTRAL, exchange=exch, tf=tf_short))
                continue

            # HARD REGIME GATE — drop counter-regime / below-floor signals
            if isinstance(result, dict) and result.get('regime_blocked'):
                all_failures.append(ScanFailure(REASON_REGIME_BLOCK, exchange=exch, tf=tf_short,
                                                detail=result.get('regime_block_detail') or 'below BTC regime floor'))
                continue

            # Tag the signal so display layer can warn the user
            result['exchange']        = exch
            result['new_listing']     = True
            result['signal_tf']       = tf_short
            result['signal_tf_label'] = _NL_TF_LABEL[tf_short]
            result['vol_24h_usdt']    = 0
            # Hold time — cap at 4h for new listings (not enough history to predict longer)
            dur_score = result.get('dur_score', 0)
            if dur_score >= 5:
                result['hold'] = '~4 hours'; result['hold_hours'] = 4
            elif dur_score >= 2:
                result['hold'] = '~2 hours'; result['hold_hours'] = 2
            else:
                result['hold'] = '~1 hour';  result['hold_hours'] = 1
            result['tf_note'] = (
                f"⚠️ NEW LISTING — analysis based on {_NL_TF_LABEL[tf_short]} data only. "
                f"Confidence values are less reliable. Use reduced position size."
            )

            logger.info("NEW LISTING SCAN: %s %s %s — conf=%d bias=%s tf=%s",
                        exch, symbol, tf_short,
                        result['confidence'], result['bias'], tf_short)

            # Keep the best result (highest confidence across exchanges/TFs)
            if best_result is None or result['confidence'] > best_result['confidence']:
                best_result = result
            break  # found usable data on this exchange — move to next exchange

    if best_result:
        return best_result

    # All failed — return most informative failure
    priority = [REASON_REGIME_BLOCK, REASON_LOW_CONF, REASON_GAP_BLOCK,
                REASON_COUNTER_TREND, REASON_NEUTRAL, REASON_SHORT_HISTORY,
                REASON_NO_CONTRACT]
    for reason in priority:
        for f in all_failures:
            if f.reason == reason:
                return f
    return ScanFailure(REASON_NO_CONTRACT, detail=f"no data found for {symbol} on any exchange/TF")


def analyze_symbol_mtf(symbol, exchange, tf_key=None, funding=0.0, user_requested=False):
    """
    Analyse a symbol on one or all timeframes for a given exchange.

    If tf_key is given  → analyse only that timeframe.
    If tf_key is None   → analyse all TF_CONFIGS, return the best signal.

    user_requested=True → bypass volume filter (user explicitly asked for this pair).

    Returns a single signal dict (best signal), or a ScanFailure if every
    timeframe failed — the failure carries the most informative reason seen
    across all TFs so cscan_command can tell the user why.

    Best = highest (confidence × weight).  Ties broken by dur_score.
    """
    tfs_to_try = [tf_key] if tf_key else list(TF_CONFIGS.keys())
    candidates = []
    failures   = []   # list[ScanFailure] — collected across all TFs

    for tf in tfs_to_try:
        cfg = TF_CONFIGS[tf]
        try:
            # ── LIQUIDITY FILTER — skip for user-requested scans ─────────��────
            passes_vol, vol_usdt, vol_threshold = _passes_volume_filter(exchange, symbol, tf)
            if not passes_vol and not user_requested:
                vol_m = vol_usdt / 1_000_000
                thr_m = vol_threshold / 1_000_000
                failures.append(ScanFailure(
                    REASON_LOW_VOLUME, exchange=exchange, tf=tf,
                    detail=f"24h vol ${vol_m:.1f}M < ${thr_m:.1f}M floor for {tf}"
                ))
                logger.debug("VOL FILTER: %s %s %s — $%.1fM < $%.1fM",
                             exchange, symbol, tf, vol_m, thr_m)
                continue
            # ─────────────────────────────────────────────────────────────────

            df_pri = _fetch_tf_candles(exchange, symbol, tf, 'pri')
            df_con = _fetch_tf_candles(exchange, symbol, tf, 'con')

            # Propagate ScanFailure from candle fetch
            if isinstance(df_pri, ScanFailure):
                failures.append(df_pri)
                continue
            if isinstance(df_con, ScanFailure):
                failures.append(df_con)
                continue
            if df_pri is None or df_con is None:
                failures.append(ScanFailure(REASON_NO_CONTRACT, exchange=exchange, tf=tf))
                logger.debug("MTF %s %s %s: insufficient candles", exchange, symbol, tf)
                continue

            # ── FIX #STALE — Stale data detection ────────────────────────────
            # If the last candle timestamp is older than 1.5x the primary TF,
            # the exchange feed is stale (outage / API issue / delisted).
            # Reject rather than analyse phantom price.
            _tf_minutes = {
                '15m': 15, '1h': 60, '4h': 240, '1d': 1440
            }.get(tf, 240)
            try:
                _last_ts = df_pri.index[-1] if hasattr(df_pri.index[-1], 'timestamp') else None
                if _last_ts is None and 'timestamp' in df_pri.columns:
                    _last_ts = pd.Timestamp(df_pri['timestamp'].iloc[-1])
                elif _last_ts is None:
                    # DataFrame indexed by timestamp (ccxt path)
                    _last_ts = pd.Timestamp(df_pri.index[-1])
                _age_minutes = (pd.Timestamp.utcnow().tz_localize(None) - _last_ts.tz_localize(None)).total_seconds() / 60
                if _age_minutes > _tf_minutes * 1.5:
                    failures.append(ScanFailure(REASON_NO_CONTRACT, exchange=exchange, tf=tf,
                                                detail=f"stale data: last candle {_age_minutes:.0f}m ago (>{_tf_minutes*1.5:.0f}m threshold)"))
                    logger.warning("STALE DATA: %s %s %s — last candle %.0fm ago",
                                   exchange, symbol, tf, _age_minutes)
                    continue
            except Exception:
                pass   # timestamp check failed — continue with analysis

            stoch_tf = '4h' if cfg['stoch_window'] == 5 else '1d'
            df_pri   = add_indicators(df_pri, timeframe=stoch_tf)
            df_con   = add_indicators(df_con, timeframe='1d')
            if df_pri is None or df_con is None:
                failures.append(ScanFailure(REASON_SHORT_HISTORY, exchange=exchange, tf=tf,
                                            detail="add_indicators returned None"))
                continue
            if len(df_pri) < 5 or len(df_con) < 5:
                failures.append(ScanFailure(REASON_SHORT_HISTORY, exchange=exchange, tf=tf,
                                            detail=f"after indicators: pri={len(df_pri)} con={len(df_con)} rows"))
                continue

            result = score_pair(df_pri, df_con, funding, symbol,
                                      user_requested=user_requested)

            # Propagate ScanFailure from score_pair
            if isinstance(result, ScanFailure):
                result.exchange = result.exchange or exchange
                result.tf       = result.tf or tf
                failures.append(result)
                continue
            if result is None:
                failures.append(ScanFailure(REASON_NEUTRAL, exchange=exchange, tf=tf))
                continue

            # HARD REGIME GATE — drop counter-regime / below-floor signals from
            # autoscan and /scan. Surfaced as a failure so single-pair /scan can
            # redirect the user to the unrestricted /analyse command.
            if isinstance(result, dict) and result.get('regime_blocked'):
                failures.append(ScanFailure(REASON_REGIME_BLOCK, exchange=exchange, tf=tf,
                                            detail=result.get('regime_block_detail') or 'below BTC regime floor'))
                continue

            # Override hold duration with TF-appropriate buckets
            dur_score = result.get('dur_score', 0)
            hold_hours, hold_label = _mtf_hold(dur_score, tf)
            result['hold_hours'] = hold_hours
            result['hold']       = hold_label

            dur_reasons  = result.get('dur_reasons', [])
            first_reason = dur_reasons[0] if dur_reasons else ''
            if hold_hours >= 24:
                result['tf_note'] = f"{cfg['label']} signal — hold {hold_label}. {first_reason}"
            else:
                result['tf_note'] = f"{cfg['label']} signal — target within {hold_label}. {first_reason}"

            result['signal_tf']       = tf
            result['signal_tf_label'] = cfg['label']
            result['exchange']        = exchange
            result['_weighted_conf']  = result['confidence'] * cfg['weight']
            result['vol_24h_usdt']    = vol_usdt   # from volume filter check above

            candidates.append(result)

        except Exception as e:
            logger.warning("analyze_symbol_mtf %s %s %s: %s", exchange, symbol, tf, e)
            failures.append(ScanFailure(REASON_NO_CONTRACT, exchange=exchange, tf=tf, detail=str(e)))
            continue

    if candidates:
        candidates.sort(
            key=lambda r: (r['_weighted_conf'], r.get('dur_score', 0)),
            reverse=True
        )
        best = candidates[0]
        best.pop('_weighted_conf', None)
        return best

    # No signal — return the most informative failure.
    # Priority: REGIME_BLOCK > LOW_CONF > GAP_BLOCK > COUNTER_TREND >
    #           FLIP_BLOCK > SHORT_HISTORY > NEUTRAL > NO_CONTRACT
    priority = [REASON_REGIME_BLOCK, REASON_LOW_CONF, REASON_GAP_BLOCK,
                REASON_COUNTER_TREND, REASON_FLIP_BLOCK, REASON_SHORT_HISTORY,
                REASON_NEUTRAL, REASON_LOW_VOLUME, REASON_NO_CONTRACT]
    for reason in priority:
        for f in failures:
            if f.reason == reason:
                return f
    return ScanFailure(REASON_NO_CONTRACT, exchange=exchange,
                       detail="no candle data from any exchange")


def _cscan_pair(symbol):
    """
    Legacy default scan — runs _cscan_pair_mtf pinned to 4H timeframe.
    Used as a fallback when the MTF auto-detect returns no results.
    Returns a list of signal dicts sorted best-first.
    """
    return _cscan_pair_mtf(symbol, tf_key='4h')


def _cscan_pair_mtf(symbol, tf_key=None):
    """
    MTF-aware version of _cscan_pair.
    Runs analyze_symbol_mtf across all available exchanges.
    tf_key: None = auto-detect best TF; '1h' / '4h' etc = pin to that TF.

    Returns either:
      - A non-empty list of signal dicts (sorted best-first) — success path
      - A list of ScanFailure objects only                   — every exchange failed
        Caller must check: isinstance(results[0], ScanFailure)
    """
    signals  = []
    failures = []
    exchanges = []

    # MEXC-only — Bybit and Binance are geo-blocked on this hosting region
    exchanges.append(('MEXC', 0.0))

    for exch, funding in exchanges:
        try:
            result = analyze_symbol_mtf(symbol, exch, tf_key=tf_key, funding=funding,
                                        user_requested=True)
            if isinstance(result, ScanFailure):
                failures.append(result)
            elif result:
                signals.append(result)
        except Exception as e:
            logger.warning("_cscan_pair_mtf %s %s: %s", exch, symbol, e)
            failures.append(ScanFailure(REASON_NO_CONTRACT, exchange=exch, detail=str(e)))

    if signals:
        signals.sort(
            key=lambda r: (r['confidence'], r.get('dur_score', 0)),
            reverse=True
        )
        return signals

    # ── NEW-LISTING FALLBACK ───────────────────���──────────────────────────────
    # If every standard TF failed due to insufficient history (new listing) or
    # no contract found, try the dynamic new-listing scanner which accepts
    # as few as 20 candles on the shortest available TF.
    # This only fires when the standard scan produced NO usable signals.
    has_short_history = any(f.reason == REASON_SHORT_HISTORY for f in failures)
    has_no_contract   = any(f.reason == REASON_NO_CONTRACT   for f in failures)
    if has_short_history or (has_no_contract and not failures):
        nl_result = analyze_symbol_new_listing(symbol, funding=0.0)
        if not isinstance(nl_result, ScanFailure) and nl_result:
            return [nl_result]
        # If new-listing scan also failed, fold its failure in for better messaging
        if isinstance(nl_result, ScanFailure):
            failures.append(nl_result)

    # All exchanges failed — return failure list so cscan_command can explain why
    return failures or [ScanFailure(REASON_NO_CONTRACT, detail="no exchanges tried")]


# ─────────────────────────────────────────────
# CHART GENERATOR — matplotlib server-side rendering
# Produces a .png with:
#   • Candlestick OHLC  (panel 1)
#   • EMA20 / EMA50 overlaid on candles
#   • Bollinger Bands (shaded)
#   • Entry zone (green), Stop Loss (red), T1/T2/T3 dashed lines
#   ��� Volume bars    (panel 2, coloured by candle direction)
#   • RSI with 30/70 levels (panel 3)
# ──���─���────────────────────────────────────────
def generate_chart(signal, df4h):
    """
    Generate a chart PNG (bytes) for the given signal using the 4H OHLCV dataframe
    (indicators must already be added via add_indicators).

    Returns: bytes (PNG image) or None on failure.
    """
    try:
        df = df4h.tail(60).copy().reset_index(drop=True)
        n  = len(df)
        x  = list(range(n))

        price      = signal['price']
        bias       = signal['bias']
        entry_low  = signal['entry_low']
        entry_high = signal['entry_high']
        stop_loss  = signal['stop_loss']
        t1         = signal['t1']
        t2         = signal['t2']
        t3         = signal['t3']
        symbol     = signal['symbol']
        exchange   = signal.get('exchange', '')
        conf       = signal['confidence']

        BG        = '#0d1117'
        PANEL_BG  = '#161b22'
        BULL_C    = '#26a641'
        BEAR_C    = '#e05c5c'
        EMA20_C   = '#f0c060'
        EMA50_C   = '#60a0f0'
        BB_C      = '#8855cc'
        ENTRY_C   = '#26a641'
        SL_C      = '#e05c5c'
        T1_C      = '#aaffaa'
        T2_C      = '#55ffaa'
        T3_C      = '#00ffcc'
        RSI_C     = '#f0c060'
        VOL_BULL  = '#1a6630'
        VOL_BEAR  = '#7a2020'
        GRID_C    = '#21262d'
        TEXT_C    = '#c9d1d9'

        fig = plt.figure(figsize=(14, 9), facecolor=BG)
        gs  = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[4, 1.2, 1.2], hspace=0.04)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)

        for ax in (ax1, ax2, ax3):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_C, labelsize=7)
            ax.spines['top'].set_color(GRID_C)
            ax.spines['bottom'].set_color(GRID_C)
            ax.spines['left'].set_color(GRID_C)
            ax.spines['right'].set_color(GRID_C)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        W_BODY = 0.55
        W_WICK = 0.15

        for i in x:
            row   = df.iloc[i]
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            colour = BULL_C if c >= o else BEAR_C
            ax1.bar(i, h - l, width=W_WICK, bottom=l, color=colour, linewidth=0)
            ax1.bar(i, abs(c - o) or (h - l) * 0.01,
                    width=W_BODY, bottom=min(o, c), color=colour, linewidth=0)

        ax1.fill_between(x, df['bb_lower'], df['bb_upper'], alpha=0.07, color=BB_C, label='BB')
        ax1.plot(x, df['bb_upper'], color=BB_C, linewidth=0.6, alpha=0.5)
        ax1.plot(x, df['bb_lower'], color=BB_C, linewidth=0.6, alpha=0.5)
        ax1.plot(x, df['ema20'], color=EMA20_C, linewidth=1.0, label='EMA20')
        ax1.plot(x, df['ema50'], color=EMA50_C, linewidth=1.0, label='EMA50')

        ax1.axhspan(entry_low, entry_high, alpha=0.15, color=ENTRY_C, zorder=2)
        ax1.axhline(entry_low,  color=ENTRY_C, linewidth=0.8, linestyle='--', alpha=0.7)
        ax1.axhline(entry_high, color=ENTRY_C, linewidth=0.8, linestyle='--', alpha=0.7)
        ax1.axhline(stop_loss, color=SL_C, linewidth=1.2, linestyle='-.')
        ax1.axhline(t1, color=T1_C, linewidth=0.9, linestyle='--')
        ax1.axhline(t2, color=T2_C, linewidth=0.9, linestyle='--')
        ax1.axhline(t3, color=T3_C, linewidth=0.9, linestyle='--')

        def _label(ax, y, text, colour, xpos=n - 0.3):
            ax.text(xpos, y, text, color=colour, fontsize=6.5,
                    va='center', ha='left', fontweight='bold',
                    bbox=dict(facecolor=BG, edgecolor='none', alpha=0.7, pad=1))

        _label(ax1, stop_loss,  f'SL {stop_loss:.4f}',    SL_C)
        _label(ax1, entry_low,  f'ENT {entry_low:.4f}',   ENTRY_C)
        _label(ax1, entry_high, f'ENT {entry_high:.4f}',  ENTRY_C)
        _label(ax1, t1,         f'T1  {t1:.4f}',          T1_C)
        _label(ax1, t2,         f'T2  {t2:.4f}',          T2_C)
        _label(ax1, t3,         f'T3  {t3:.4f}',          T3_C)
        ax1.axhline(price, color=TEXT_C, linewidth=0.8, linestyle=':', alpha=0.6)
        _label(ax1, price, f'NOW {price:.4f}', TEXT_C)
        ax1.grid(True, color=GRID_C, linewidth=0.4, alpha=0.5)

        bias_arrow = '▲ LONG' if bias == 'LONG' else '▼ SHORT'
        bias_col   = BULL_C if bias == 'LONG' else BEAR_C
        fig.text(0.01, 0.975, f'{exchange} · {symbol}  4H Chart',
                 color=TEXT_C, fontsize=11, fontweight='bold', va='top')
        fig.text(0.35, 0.975, f'{bias_arrow}  {conf}/10 confidence',
                 color=bias_col, fontsize=10, fontweight='bold', va='top')

        legend_elements = [
            Line2D([0],[0], color=EMA20_C, linewidth=1.2, label='EMA20'),
            Line2D([0],[0], color=EMA50_C, linewidth=1.2, label='EMA50'),
            mpatches.Patch(facecolor=BB_C, alpha=0.25, label='BB'),
            mpatches.Patch(facecolor=ENTRY_C, alpha=0.35, label='Entry Zone'),
            Line2D([0],[0], color=SL_C, linewidth=1.2, linestyle='-.', label='Stop Loss'),
            Line2D([0],[0], color=T1_C, linewidth=1.0, linestyle='--', label='T1'),
            Line2D([0],[0], color=T2_C, linewidth=1.0, linestyle='--', label='T2'),
            Line2D([0],[0], color=T3_C, linewidth=1.0, linestyle='--', label='T3'),
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=6,
                   facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT_C, ncol=4, framealpha=0.85)

        vol_colors = [VOL_BULL if df.iloc[i]['close'] >= df.iloc[i]['open']
                      else VOL_BEAR for i in x]
        ax2.bar(x, df['volume'], color=vol_colors, width=0.7, linewidth=0)
        if 'volume_ma' in df.columns:
            ax2.plot(x, df['volume_ma'], color='#aaaaaa', linewidth=0.7, linestyle='--', alpha=0.6)
        ax2.set_ylabel('Vol', color=TEXT_C, fontsize=7, rotation=0, labelpad=20, va='center')
        ax2.grid(True, color=GRID_C, linewidth=0.3, alpha=0.4)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v/1e6:.1f}M' if v >= 1e6 else f'{v/1e3:.0f}K')
        )

        ax3.plot(x, df['rsi'], color=RSI_C, linewidth=1.0)
        ax3.axhline(70, color=BEAR_C, linewidth=0.6, linestyle='--', alpha=0.6)
        ax3.axhline(30, color=BULL_C, linewidth=0.6, linestyle='--', alpha=0.6)
        ax3.axhline(50, color=TEXT_C, linewidth=0.4, linestyle=':', alpha=0.3)
        ax3.fill_between(x, df['rsi'], 70, where=[v > 70 for v in df['rsi']],
                         alpha=0.2, color=BEAR_C)
        ax3.fill_between(x, df['rsi'], 30, where=[v < 30 for v in df['rsi']],
                         alpha=0.2, color=BULL_C)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel('RSI', color=TEXT_C, fontsize=7, rotation=0, labelpad=20, va='center')
        ax3.grid(True, color=GRID_C, linewidth=0.3, alpha=0.4)

        step    = max(1, n // 10)
        xticks  = list(range(0, n, step))
        xlabels = [df.iloc[i]['timestamp'].strftime('%m/%d %H:%M') for i in xticks]
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(xlabels, rotation=30, ha='right', fontsize=6, color=TEXT_C)
        ax1.tick_params(labelbottom=False)
        ax2.tick_params(labelbottom=False)

        plt.subplots_adjust(left=0.01, right=0.88, top=0.96, bottom=0.07)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130, facecolor=BG, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    except Exception as e:
        logger.warning("generate_chart error for %s: %s", signal.get('symbol','?'), e)
        try:
            plt.close('all')
        except Exception:
            pass
        return None


# ───────────���───────────���─������─���─���──────────────
# ANALYZE FUNCTIONS
# ────────���───���────────────────────────────────
def analyze_bybit(symbol):
    try:
        df4h = bybit_fetch_ohlcv(symbol, '240', 100)
        if df4h is None or len(df4h) < 30:
            logger.debug("BYBIT %s: insufficient 4H candles", symbol)
            return None
        df1d = bybit_fetch_ohlcv(symbol, 'D', 60)
        if df1d is None or len(df1d) < 15:
            logger.debug("BYBIT %s: insufficient Daily candles", symbol)
            return None
        df4h = add_indicators(df4h, timeframe="4h")
        df1d = add_indicators(df1d, timeframe="1d")
        if df4h is None or df1d is None: return None
        if len(df4h) < 5 or len(df1d) < 5: return None
        funding = bybit_fetch_funding(symbol)
        result  = score_pair(df4h, df1d, funding, symbol)
        if isinstance(result, ScanFailure) or result is None:
            return None
        if isinstance(result, dict) and result.get('regime_blocked'):
            return None   # HARD REGIME GATE — never surface in autoscan/scan
        result['exchange'] = 'BYBIT'
        return result
    except Exception as e:
        logger.warning("BYBIT %s analyze error: %s", symbol, e)
        return None

def analyze_mexc(symbol):
    try:
        df4h = mexc_fetch_ohlcv(symbol, '4h', 100)
        if df4h is None or len(df4h) < 30:
            logger.debug("MEXC %s: insufficient 4H candles", symbol)
            return None
        df1d = mexc_fetch_ohlcv(symbol, '1d', 60)
        if df1d is None or len(df1d) < 15:
            logger.debug("MEXC %s: insufficient Daily candles", symbol)
            return None
        df4h = add_indicators(df4h, timeframe="4h")
        df1d = add_indicators(df1d, timeframe="1d")
        if df4h is None or df1d is None: return None
        if len(df4h) < 5 or len(df1d) < 5: return None
        result = score_pair(df4h, df1d, 0, symbol)
        if isinstance(result, ScanFailure) or result is None:
            return None
        if isinstance(result, dict) and result.get('regime_blocked'):
            return None   # HARD REGIME GATE — never surface in autoscan/scan
        result['exchange'] = 'MEXC'
        return result
    except Exception as e:
        logger.warning("MEXC %s analyze error: %s", symbol, e)
        return None


def _display_exchanges() -> list:
    """Single source-of-truth exchange policy for ALL scan output.

    Bybit is the only venue we display. MEXC is used *only* as a fallback
    when Bybit's API is unavailable. Binance is never a display source.
    Restricting output to one venue means each pair appears exactly once
    (no MEXC/BYBIT/BINANCE triplicates in /scan, /tg, /tl, /pnl, etc.).
    """
    if sakz_exchanges.BYBIT_AVAILABLE is not False:
        return ["BYBIT"]
    return ["MEXC"]


def run_mid_scan(rank_from=51, rank_to=200):
    """
    CEILING #6 — Mid-tier universe scan.
    Scans coins ranked 51–200 by 24h volume on each exchange.

    Rationale: the top-50 universe is the most efficiently priced — tracked
    by thousands of algorithms simultaneously.  Coins in the 51–200 band are
    liquid enough for safe perp trading but receive far less algorithmic
    attention, so their price discovery lags and signals have higher alpha.

    Key differences from run_full_scan():
    • Uses *_get_mid_symbols() instead of *_get_top_symbols()
    • Tags every result with tier='MID' for UI display
    • Does NOT update last_scan_results (mid results are separate from the
      main scan cache so /scan and /scanmid coexist without collision)
    • Correlation gate and portfolio summary are computed and returned
      but not stored in _last_portfolio_summary (would pollute main scan state)

    Returns: list of signal dicts (sorted by confidence desc)
    """


    results = []

    bybit_check_available()
    binance_check_available()

    # Warm regime cache — same gate logic as full scan
    state._btc_regime_cache = None
    regime = get_btc_regime()
    logger.info("=== MID SCAN START | BTC Regime: %s | ranks %d–%d ===",
                regime, rank_from, rank_to)

    def _process(r):
        if not r:
            return
        r['tier'] = 'MID'   # tag so UI can badge these signals
        results.append(r)

    # ── SINGLE DISPLAY VENUE ── scan only the active venue (Bybit, or MEXC
    # when Bybit is down) so a pair never appears more than once.
    _venue = _display_exchanges()[0]
    if _venue == 'BYBIT':
        for sym in bybit_get_mid_symbols(rank_from, rank_to):
            _process(analyze_bybit(sym))
            time.sleep(0.15)
    else:
        for sym in mexc_get_mid_symbols(rank_from, rank_to):
            _process(analyze_mexc(sym))
            time.sleep(0.15)

    _vol_rank = {"MEDIUM": 4, "HIGH": 3, "LOW": 2, "EXTREME": 1, "RANGING": 0}
    results.sort(
        key=lambda x: (x["confidence"], (x.get("consensus_score") or 0.5), x["score"],
                       _vol_rank.get(x.get("vol_regime", "MEDIUM"), 2)),
        reverse=True
    )

    # De-duplicate: if the same symbol appears on multiple exchanges, keep
    # the highest-confidence version only
    seen_syms = set()
    deduped   = []
    for r in results:
        base = r['symbol'].replace('_USDT', 'USDT')
        if base not in seen_syms:
            seen_syms.add(base)
            deduped.append(r)

    logger.info("Mid scan complete — %d signals found (%d before dedup)",
                len(deduped), len(results))
    return deduped


# ─────────────────────────────────────────────
# RUN FULL SCAN
# • 3-thread executor allows concurrent scans
# • 15-minute cache — second user within TTL
#   gets instant results, no duplicate API calls
# ─────────────────────────────────────────────
# ��════════════════��════����══��══����══����══����══════════════����═══════════════════════
# LIQUIDITY FILTER
# ───────────────��──────────────────────────────────────────────────────────────
# Every signal must clear a minimum 24h USDT volume before scoring begins.
# Thresholds by timeframe:
#   SCALP  (15m / 1h)  →  $5M   — need tight spreads and fast fills
#   SWING  (4h)        →  $20M  — need enough volume to sustain a multi-hour move
#   DAILY  (1d)        →  $50M  — large moves need institutional participation
#
# Per-exchange minimums are lower than "true" thresholds because each exchange
# shows only its own volume — the real on-chain volume is the sum of all venues.
# We use 60% of the target as the per-exchange floor.
#
# Why this matters:
#   A signal on a $500k/day pair is noise — spread alone can eat the TP.
#   Filtering these out has an outsized effect on reported win rate because
#   low-volume pairs are disproportionately represented in false breakouts.
# ═��════════════════════════════════════════════════════���═══════════════════════

# Volume thresholds in USDT — keyed by TF role
VOL_THRESHOLDS = {
    '15m': 5_000_000,
    '1h':  5_000_000,
    '4h':  20_000_000,
    '1d':  50_000_000,
    # Fallback for any unlisted TF
    'default': 10_000_000,
}

# Per-exchange multiplier — each exchange shows ~40-70% of total volume
# Using 0.5 as a conservative floor (passes if single exchange shows ≥50% of target)
_VOL_EXCHANGE_FRACTION = 0.5

_vol_cache: dict = {}   # { 'BYBIT_BTCUSDT': (vol_usdt, fetched_at) }
_VOL_CACHE_TTL_SECS = 900   # 15 min — volume doesn't change fast enough to need more


def _get_symbol_volume(exchange: str, symbol: str) -> float:
    """
    Return the 24h USDT volume for a symbol on a given exchange.
    Results are cached for 15 minutes to avoid hammering the ticker endpoint.
    Returns 0.0 if the fetch fails.
    """
    import time as _time
    cache_key = f"{exchange}_{symbol}"
    now       = _time.time()

    # Check cache
    cached = _vol_cache.get(cache_key)
    if cached and (now - cached[1]) < _VOL_CACHE_TTL_SECS:
        return cached[0]

    vol = 0.0
    try:
        if exchange == 'BYBIT':
            r    = http_get(
                f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}",
                headers=HEADERS, timeout=8
            )
            data = r.json()
            if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                vol = float(data['result']['list'][0].get('turnover24h', 0) or 0)

        elif exchange == 'MEXC':
            _sym_c = symbol.upper().replace('_USDT', 'USDT'); futures_sym = (_sym_c[:-4] + '_USDT') if _sym_c.endswith('USDT') else (_sym_c + '_USDT')
            r    = http_get(
                f"https://contract.mexc.com/api/v1/contract/ticker?symbol={futures_sym}",
                headers=HEADERS, timeout=8
            )
            data = r.json()
            if data.get('success') and data.get('data'):
                d   = data['data']
                vol = float(d.get('amount24', 0) or d.get('volume24', 0) or 0)

        elif exchange == 'BINANCE':
            r    = http_get(
                f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}",
                headers=HEADERS, timeout=8
            )
            data = r.json()
            vol  = float(data.get('quoteVolume', 0) or 0)

    except Exception as e:
        logger.debug("_get_symbol_volume %s %s: %s", exchange, symbol, e)

    _vol_cache[cache_key] = (vol, now)
    return vol


def _passes_volume_filter(exchange: str, symbol: str, tf_key: str) -> tuple:
    """
    Check if a symbol passes the liquidity filter for a given timeframe.

    Returns (passes: bool, vol_usdt: float, threshold: float)
    so callers can include the actual volume in ScanFailure details.
    """
    threshold     = VOL_THRESHOLDS.get(tf_key, VOL_THRESHOLDS['default'])
    per_exch_min  = threshold * _VOL_EXCHANGE_FRACTION
    vol           = _get_symbol_volume(exchange, symbol)

    # vol == 0 means the fetch failed — don't block the signal, just warn
    if vol == 0:
        return True, 0.0, per_exch_min

    return vol >= per_exch_min, vol, per_exch_min


def run_full_scan():
    """Blocking scan — always runs in SCAN_EXECUTOR thread."""


    _scan_t0 = time.time()

    state.previous_scan_results = state.last_scan_results.copy()
    results = []

    # Check exchange availability once per scan
    bybit_check_available()
    binance_check_available()

    # FIX #RG — Invalidate regime cache so the first score_pair call
    # this cycle fetches a fresh BTC read.  All subsequent calls within
    # the scan will hit the newly populated cache (TTL = 15 min).
    state._btc_regime_cache = None
    regime = get_btc_regime()   # warm the cache once, log it prominently
    logger.info("=== SCAN START | BTC Regime: %s ===", regime)

    # Pre-fetch volume for all symbols once per scan — avoids per-symbol HTTP calls
    # during the tight scan loop.  Stored in _vol_cache so the per-TF filter
    # in analyze_symbol_mtf hits the cache on every subsequent call.
    def _warm_vol_cache(symbols, exchange):
        """Batch-warm the volume cache by fetching the exchange ticker once."""
        try:
            import time as _t
            now = _t.time()
            if exchange == 'BYBIT':
                r    = http_get("https://api.bybit.com/v5/market/tickers?category=linear",
                                    headers=HEADERS, timeout=15)
                data = r.json()
                if data.get('retCode') == 0:
                    for t in data['result']['list']:
                        sym = t.get('symbol', '')
                        if sym in symbols:
                            vol = float(t.get('turnover24h', 0) or 0)
                            _vol_cache[f"BYBIT_{sym}"] = (vol, now)
            elif exchange == 'MEXC':
                r    = http_get("https://contract.mexc.com/api/v1/contract/ticker",
                                    headers=HEADERS, timeout=15)
                data = r.json()
                if data.get('success') and data.get('data'):
                    for t in data['data']:
                        raw_sym   = t.get('symbol', '')
                        clean_sym = raw_sym.replace('_USDT', 'USDT')
                        if clean_sym in symbols:
                            vol = float(t.get('amount24', 0) or 0)
                            _vol_cache[f"MEXC_{clean_sym}"] = (vol, now)
            elif exchange == 'BINANCE':
                r    = http_get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                                    headers=HEADERS, timeout=15)
                for t in r.json():
                    sym = t.get('symbol', '')
                    if sym in symbols:
                        vol = float(t.get('quoteVolume', 0) or 0)
                        _vol_cache[f"BINANCE_{sym}"] = (vol, now)
        except Exception as e:
            logger.warning("_warm_vol_cache %s: %s", exchange, e)

    def _process(r, exchange_key_prefix):
        if not r:
            return
        # Tag 24h volume from cache if available (populated by _warm_vol_cache)
        if 'vol_24h_usdt' not in r:
            cached = _vol_cache.get(f"{r.get('exchange','BYBIT')}_{r.get('symbol','')}")
            r['vol_24h_usdt'] = cached[0] if cached else 0
        results.append(r)
        key = f"{exchange_key_prefix}_{r['symbol']}"
        db_append_price_history(key, r['exchange'], r['symbol'], r['price'])
        if key not in state.price_history:
            state.price_history[key] = []
        state.price_history[key].append({'time': datetime.now(), 'price': r['price'],
                                   'exchange': r['exchange'], 'symbol': r['symbol']})
        cutoff = datetime.now() - timedelta(hours=24)
        state.price_history[key] = [p for p in state.price_history[key] if p['time'] > cutoff]

    # ── SINGLE DISPLAY VENUE ──
    # Only ONE exchange feeds the scan output. Bybit is primary; MEXC is the
    # fallback used only when Bybit's API is down. Binance is never a display
    # source. This guarantees each pair appears exactly once — no
    # MEXC/BYBIT/BINANCE triplicates in /scan, /tg, /tl or /pnl.
    _venue = _display_exchanges()[0]
    if _venue == 'BYBIT':
        bybit_syms = bybit_get_top_symbols(50)
        _warm_vol_cache(set(bybit_syms), 'BYBIT')
        for sym in bybit_syms:
            _process(analyze_bybit(sym), 'BYBIT')
            time.sleep(0.2)
    else:
        mexc_syms = mexc_get_top_symbols(50)
        _warm_vol_cache(set(mexc_syms), 'MEXC')
        for sym in mexc_syms:
            _process(analyze_mexc(sym), 'MEXC')
            time.sleep(0.2)

    _vol_rank = {"MEDIUM": 4, "HIGH": 3, "LOW": 2, "EXTREME": 1, "RANGING": 0}
    results.sort(
        key=lambda x: (x["confidence"], (x.get("consensus_score") or 0.5), x["score"], _vol_rank.get(x.get("vol_regime", "MEDIUM"), 2)),
        reverse=True
    )

    # ── FIX #DD — Portfolio-level correlation gate ──────────────────────────
    # Problem: every signal in the list is treated as independent.  In reality,
    # 10 altcoin LONGs in a BULL regime are all the same trade — long BTC with
    # extra steps.  A single BTC dump stops all of them simultaneously.
    #
    # This gate does NOT drop signals (the user decides what to trade).
    # It TAGS them with a portfolio-level risk annotation and enforces a
    # CORRELATED_SLOT_CAP: signals beyond the cap are marked
    # corr_flagged=True so the UI can warn the user prominently.
    #
    # Classification:
    #   correlated  = same direction AND regime-aligned (LONG in BULL/STRONG_BULL,
    #                 SHORT in BEAR/STRONG_BEAR).  These all move with BTC.
    #   independent = counter-regime or NEUTRAL regime (less BTC-coupled).
    #
    # The cap is 4 correlated slots.  Evidence: most retail accounts cannot
    # manage more than 4 open positions, and beyond 4 same-direction positions
    # the marginal signal quality drops sharply (you're running out of
    # genuinely independent setups in the top-50 universe).
    #
    # Portfolio summary is attached to every result list as a synthetic first
    # entry keyed 'portfolio_summary' — auto_scan_job and send_signal_cards
    # read this to produce a system-level warning header.
    # FIX #CF — Correlation filter tightened: cap 4→3 for regime-aligned signals.
    # Hard cutoff added: >5 same-bias signals in a single scan batch are
    # marked corr_hard_blocked so the UI can hide them from the default view.
    CORRELATED_SLOT_CAP = 3   # was 4
    SAME_DIR_HARD_CAP   = 5   # absolute max same-bias signals shown

    btc_r = get_btc_regime()
    bull_regime = btc_r in ('BULL', 'STRONG_BULL')
    bear_regime = btc_r in ('BEAR', 'STRONG_BEAR')

    corr_long_count  = 0   # regime-aligned LONGs seen so far
    corr_short_count = 0   # regime-aligned SHORTs seen so far
    indep_count      = 0   # counter-regime or NEUTRAL signals

    for r in results:
        bias = r['bias']
        is_corr_long  = (bias == 'LONG'  and bull_regime)
        is_corr_short = (bias == 'SHORT' and bear_regime)
        is_correlated = is_corr_long or is_corr_short

        if is_corr_long:
            corr_long_count += 1
            slot = corr_long_count
            r['corr_type']    = 'correlated_long'
            r['corr_slot']    = slot
            r['corr_flagged']      = slot > CORRELATED_SLOT_CAP
            r['corr_hard_blocked'] = slot > SAME_DIR_HARD_CAP
        elif is_corr_short:
            corr_short_count += 1
            slot = corr_short_count
            r['corr_type']         = 'correlated_short'
            r['corr_slot']         = slot
            r['corr_flagged']      = slot > CORRELATED_SLOT_CAP
            r['corr_hard_blocked'] = slot > SAME_DIR_HARD_CAP
        else:
            indep_count += 1
            r['corr_type']         = 'independent'
            r['corr_slot']         = indep_count
            r['corr_flagged']      = False
            r['corr_hard_blocked'] = False

    total_corr = corr_long_count + corr_short_count
    over_cap   = max(0, corr_long_count - CORRELATED_SLOT_CAP) + \
                 max(0, corr_short_count - CORRELATED_SLOT_CAP)

    # Attach portfolio summary so the UI layer can read it without recomputing
    portfolio_summary = {
        'btc_regime':        btc_r,
        'total_signals':     len(results),
        'corr_long':         corr_long_count,
        'corr_short':        corr_short_count,
        'independent':       indep_count,
        'corr_slot_cap':     CORRELATED_SLOT_CAP,
        'over_cap':          over_cap,
        'risk_level':        (
            'CRITICAL' if total_corr >= 10 else
            'HIGH'   if over_cap >= 4 or total_corr >= 7 else
            'MEDIUM' if over_cap >= 2 or total_corr >= 4  else
            'LOW'
        ),
        'hard_blocked':      sum(1 for r in results if r.get('corr_hard_blocked')),
    }
    # Store on the module-level so any function can read it

    state._last_portfolio_summary = portfolio_summary

    logger.info(
        "Correlation gate: %d corr_long / %d corr_short / %d independent | "
        "%d over cap | risk=%s",
        corr_long_count, corr_short_count, indep_count,
        over_cap, portfolio_summary['risk_level']
    )

    # ── USER DIRECTIVE — ABANDON conflicting over-correlated signals ─────────
    # Previously the gate only TAGGED over-cap correlated signals with ⚠️ and
    # still displayed them.  Per user request we now DROP them entirely and keep
    # only the optimum set:
    #   • the top CORRELATED_SLOT_CAP regime-aligned signals per direction
    #     (highest-confidence, since `results` is already sorted best-first), and
    #   • all independent / counter-regime / NEUTRAL signals.
    # Conflicting duplicates (corr_flagged == True) are no longer signalled.
    _pre_filter_count = len(results)
    results = [r for r in results if not r.get('corr_flagged')]
    _dropped_corr = _pre_filter_count - len(results)
    if _dropped_corr:
        logger.info(
            "Correlation gate: dropped %d conflicting over-correlated signal(s) "
            "(kept %d optimum)", _dropped_corr, len(results)
        )

    state.last_scan_results = results
    state.last_scan_time    = datetime.now()

    db_save_scan(results)
    for r in results[:20]:
        db_register_outcome(r)

    # FIRST-SIGNAL MEMORY — record/refresh every scanned pair so /pnl anchors to
    # the ORIGINAL call (kept while the coin keeps running), and so a re-scan
    # clears any prior eviction. Pure logic + persistence are unit-tested.
    for r in results:
        _register_lifecycle_signal(r)

    # AUTO PAPER TRADING — auto-open positions for high-confidence signals
    if _PAPER_AVAILABLE:
        for r in results:
            try:
                opened = paper_maybe_open(r, db_connect)
                if opened:
                    logger.info("Paper position auto-opened: %s %s (conf %s)",
                                r.get('bias'), r.get('symbol'), r.get('confidence'))
            except Exception as _pe:
                logger.debug("paper_maybe_open error: %s", _pe)

    logger.info("Scan complete — %d signals found", len(results))
    state._scan_durations.append(time.time() - _scan_t0)
    if len(state._scan_durations) > 50:
        state._scan_durations = state._scan_durations[-50:]   # keep last 50
    return results


async def get_scan_results(force=False):
    """
    Async wrapper that:
    1. Returns cached results if within TTL and not forced
    2. Otherwise runs a fresh scan in SCAN_EXECUTOR (up to 3 concurrent)
    3. Uses a lock to prevent cache stampede (multiple users triggering
       simultaneous full scans at the exact same moment)
    """


    # Serve cache if fresh enough
    if not force and state._scan_cache:
        age = (datetime.now() - state._scan_cache['time']).total_seconds()
        if age < CACHE_TTL_SECS:
            logger.info("Serving cached scan results (%.0fs old)", age)
            return state._scan_cache['results'], True  # (results, from_cache)

    # Use cache lock only to avoid stampede — other users can still run
    # their own scans via SCAN_EXECUTOR independently
    async with _scan_cache_lock:
        # Re-check inside lock in case another coroutine just populated cache
        if not force and state._scan_cache:
            age = (datetime.now() - state._scan_cache['time']).total_seconds()
            if age < CACHE_TTL_SECS:
                return state._scan_cache['results'], True

        loop    = asyncio.get_event_loop()
        results = await loop.run_in_executor(SCAN_EXECUTOR, run_full_scan)

        state._scan_cache = {'results': results, 'time': datetime.now()}
        return results, False  # (results, from_cache)


# ──����─────────────────────────────────────────
# IMPROVEMENT #3 — SIGNAL OUTCOME TRACKER
# Background job that checks signal outcomes at
# 4h, 8h, 24h, 48h intervals and updates the DB.
# ─────────────────────────────────────────────
def _fetch_ohlc_since(exchange, symbol, scan_time, limit=60):
    """
    FIX A2 — Fetch 1H OHLC candles covering the period from scan_time to now.
    Returns a list of dicts {ts, open, high, low, close} sorted oldest-first.
    Uses each exchange's existing futures kline endpoint (perps only).
    Limit 60 = up to 60 hours of 1h candles — more than enough for the 48h window.
    """
    try:
        if exchange == 'BYBIT':
            df = bybit_fetch_ohlcv(symbol, '60', limit)       # '60' = 1h on Bybit
        elif exchange == 'BINANCE':
            df = binance_fetch_ohlcv(symbol, '1h', limit)
        else:
            df = mexc_fetch_ohlcv(symbol, '1h', limit)

        if df is None or len(df) == 0:
            return []

        scan_dt = datetime.fromisoformat(scan_time)
        # Keep only candles whose timestamp is at or after the scan time
        rows = []
        for _, row in df.iterrows():
            ts = row['timestamp']
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()
            if ts.replace(tzinfo=None) >= scan_dt.replace(tzinfo=None):
                rows.append({
                    'ts':    ts,
                    'open':  float(row['open']),
                    'high':  float(row['high']),
                    'low':   float(row['low']),
                    'close': float(row['close']),
                })
        return rows  # already sorted oldest-first from fetch functions
    except Exception as e:
        logger.warning("_fetch_ohlc_since error %s %s: %s", exchange, symbol, e)
        return []


def _compute_peak_pct(signal, current_price=None):
    """
    Highest favorable % a signal reached since it was scanned, plus WHEN.

    Returns (peak_pct, peak_at):
      • peak_pct : largest UNLEVERAGED favorable move (%) from entry, using
                   candle wicks (highs for LONG, lows for SHORT) over 1H candles
                   since scan_time, folded with the current live price.
      • peak_at  : datetime the peak occurred (candle timestamp, or now if the
                   live price is the most favorable point).
    Returns (None, None) when no data is available.
    """
    try:
        entry = float(signal.get('price') or 0)
        bias  = str(signal.get('bias', 'LONG')).upper()
        if entry <= 0:
            return None, None

        scan_time = signal.get('scan_time')
        if hasattr(scan_time, 'isoformat'):
            scan_iso = scan_time.isoformat()
        else:
            scan_iso = str(scan_time) if scan_time else None
        if not scan_iso:
            return None, None

        candles = _fetch_ohlc_since(
            signal.get('exchange', ''), signal.get('symbol', ''), scan_iso, limit=500
        )

        best    = None
        best_at = None
        for cdl in candles:
            if bias == 'LONG':
                fav = (cdl['high'] - entry) / entry * 100
            else:
                fav = (entry - cdl['low']) / entry * 100
            if best is None or fav > best:
                best, best_at = fav, cdl['ts']

        if current_price and current_price > 0:
            if bias == 'LONG':
                cur_fav = (current_price - entry) / entry * 100
            else:
                cur_fav = (entry - current_price) / entry * 100
            if best is None or cur_fav > best:
                best, best_at = cur_fav, datetime.now()

        if best is None:
            return None, None
        if hasattr(best_at, 'replace'):
            try:
                best_at = best_at.replace(tzinfo=None)
            except Exception:
                pass
        return best, best_at
    except Exception as e:
        logger.warning("_compute_peak_pct error: %s", e)
        return None, None


def _compute_peak_excursions(signal, current_price=None):
    """Best favorable AND worst adverse % moves since scan (UNLEVERAGED, raw).

    Returns (fav_raw, fav_at, adv_raw, adv_at):
      • fav_raw / fav_at : largest favorable move (%) + when (peak-profit path).
      • adv_raw / adv_at : worst adverse move (%, <= 0) + when (peak drawdown).
    Direction-aware: LONG favors highs / fears lows, SHORT the inverse. Uses
    candle wicks since scan_time, folded with the current live price. The
    renderer multiplies adv_raw by leverage and caps the loss at -100%.
    Returns (None, None, None, None) when no data is available.
    """
    try:
        entry = float(signal.get('price') or 0)
        bias  = str(signal.get('bias', 'LONG')).upper()
        if entry <= 0:
            return None, None, None, None

        scan_time = signal.get('scan_time')
        if hasattr(scan_time, 'isoformat'):
            scan_iso = scan_time.isoformat()
        else:
            scan_iso = str(scan_time) if scan_time else None
        if not scan_iso:
            return None, None, None, None

        candles = _fetch_ohlc_since(
            signal.get('exchange', ''), signal.get('symbol', ''), scan_iso, limit=500
        )

        fav = fav_at = None
        adv = adv_at = None
        for cdl in candles:
            if bias == 'LONG':
                f = (cdl['high'] - entry) / entry * 100
                a = (cdl['low']  - entry) / entry * 100
            else:
                f = (entry - cdl['low'])  / entry * 100
                a = (entry - cdl['high']) / entry * 100
            if fav is None or f > fav:
                fav, fav_at = f, cdl['ts']
            if adv is None or a < adv:
                adv, adv_at = a, cdl['ts']

        if current_price and current_price > 0:
            if bias == 'LONG':
                cf = (current_price - entry) / entry * 100
            else:
                cf = (entry - current_price) / entry * 100
            if fav is None or cf > fav:
                fav, fav_at = cf, datetime.now()
            if adv is None or cf < adv:
                adv, adv_at = cf, datetime.now()

        def _strip(dt):
            if hasattr(dt, 'replace'):
                try:
                    return dt.replace(tzinfo=None)
                except Exception:
                    return dt
            return dt
        return fav, _strip(fav_at), adv, _strip(adv_at)
    except Exception as e:
        logger.warning("_compute_peak_excursions error: %s", e)
        return None, None, None, None


def _resolve_outcome_from_candles(candles, bias, sl, t1, t2, t3):
    """
    FIX A1 — Walk candles chronologically and determine the TRUE outcome.

    Rules:
    - Use candle HIGH (wick) to check LONG targets.
    - Use candle LOW  (wick) to check SHORT targets.
    - FIX SL1 — SL is confirmed only when the candle CLOSE breaches the SL
      level (not a wick touch).  Rationale: in real futures trading, wick-
      hunting is extremely common on 1H charts — a candle whose low wicks 0.3%
      below an SL level but CLOSES back above it is NOT a genuine breakdown.
      Using close prevents ephemeral wicks from being counted as SL hits.
      Targets keep wick-based detection because partial fills at target price
      during a candle are real and should be credited.
    - FIX SL2 — When BOTH a target wick AND a close-based SL occur in the
      same candle: if open > sl (price started above SL), target hit is
      awarded first (a fill at the target wick was possible before the close
      broke SL).  If open <= sl (price started at or below SL for LONG), SL
      wins — the candle opened in trouble.
    - best_target_hit tracks the highest target reached before any final SL.
    - sl_after_target=1 means partial-win: target hit then SL on remainder.

    Returns: (outcome, best_target_hit, sl_after_target)
      outcome          — 'sl_hit' | 't1_hit' | 't2_hit' | 't3_hit' | 'pending'
      best_target_hit  — None | 't1' | 't2' | 't3'
      sl_after_target  — 0 | 1
    """
    best_target = None   # highest target confirmed so far
    sl_fired    = False

    target_map = [('t3', t3), ('t2', t2), ('t1', t1)]  # check highest first

    for candle in candles:
        hi    = candle['high']
        lo    = candle['low']
        close = candle['close']
        open_ = candle['open']

        if bias == 'LONG':
            # FIX SL1: SL requires close below the level, not just a wick
            candle_sl_hit = close < sl
            target_hit_this_candle = None
            for tname, tlevel in target_map:
                if hi >= tlevel:
                    target_hit_this_candle = tname
                    break

            if target_hit_this_candle and candle_sl_hit:
                # FIX SL2: both in same candle — use open to determine order
                if best_target is not None:
                    # A prior candle already hit a target; this is sl_after_target
                    sl_fired = True
                elif open_ >= sl:
                    # Opened above SL → target wick was reachable first
                    best_target = target_hit_this_candle
                    sl_fired    = True   # SL still fires after (sl_after_target=1)
                else:
                    # Opened below SL → SL condition was present from the start
                    sl_fired = True
                break
            elif target_hit_this_candle:
                best_target = target_hit_this_candle   # clean target hit
            elif candle_sl_hit:
                sl_fired = True
                break

        else:  # SHORT
            # FIX SL1: SL requires close above the SL level for shorts
            candle_sl_hit = close > sl
            target_hit_this_candle = None
            for tname, tlevel in target_map:
                if lo <= tlevel:
                    target_hit_this_candle = tname
                    break

            if target_hit_this_candle and candle_sl_hit:
                # FIX SL2: use open to determine order
                if best_target is not None:
                    sl_fired = True
                elif open_ <= sl:
                    # Opened below SL → target wick was reachable first
                    best_target = target_hit_this_candle
                    sl_fired    = True
                else:
                    # Opened above SL → SL condition present from start
                    sl_fired = True
                break
            elif target_hit_this_candle:
                best_target = target_hit_this_candle
            elif candle_sl_hit:
                sl_fired = True
                break

    # Build result
    sl_after_target = 1 if (sl_fired and best_target is not None) else 0

    if sl_fired and best_target is None:
        outcome = 'sl_hit'
    elif best_target == 't3':
        outcome = 't3_hit'
    elif best_target == 't2':
        outcome = 't2_hit'
    elif best_target == 't1':
        outcome = 't1_hit'
    else:
        outcome = 'pending'

    return outcome, best_target, sl_after_target


async def check_signal_outcomes(context: ContextTypes.DEFAULT_TYPE):
    """Async wrapper — offloads the blocking outcome sweep to a worker thread.

    PERFORMANCE: the outcome checker does purely blocking work (sqlite reads +
    OHLC/price network fetches per pending signal) and has no awaits, so running
    it directly on the event loop froze ALL command handling for the length of
    the sweep every 5 minutes. Offloading to SCAN_EXECUTOR keeps the loop free.
    """
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(SCAN_EXECUTOR, _check_signal_outcomes_blocking, context)
    except Exception as e:
        logger.warning("check_signal_outcomes failed: %s", e)


def _check_signal_outcomes_blocking(context: ContextTypes.DEFAULT_TYPE):
    """
    FIX A1 — Sequential win/loss: outcome determined by candle sequence,
              not a single price snapshot. SL after target = sl_after_target flag.
    FIX A2 — OHLC candle data used (high/low), not tick price. A wick that
              touches SL is correctly caught even between poll cycles.
    FIX A6 — Deduplication already handled at insert time (db_register_outcome).
    FIX P3 — Resolve outcome IMMEDIATELY when SL or any target is definitively hit
              (candle closed beyond it). 48h is the maximum window, not the trigger.
              Signals still pending at 48h are expired as 'expired' — not left as
              'pending', which would pollute the pending count and delay stats.
    FIX P5 — entry_confirmed: set to 1 the first time a candle's [low, high] range
              overlaps the signal's [entry_low, entry_high] zone. Signals that never
              traded through the entry zone are marked entry_confirmed=0 when resolved
              and are excluded from win/loss stats so they don't skew results.
    """
    conn = db_connect()
    c    = conn.cursor()
    c.execute("""SELECT id, exchange, symbol, bias, entry_price, entry_low, entry_high,
                        stop_loss, t1, t2, t3, scan_time, check_4h, check_8h, check_24h, check_48h,
                        best_target_hit, sl_after_target, entry_confirmed
                 FROM signal_outcomes WHERE outcome='pending'""")
    rows = c.fetchall()
    conn.close()

    now = datetime.now()
    for row in rows:
        try:
            scan_dt  = datetime.fromisoformat(row['scan_time'])
            hours    = (now - scan_dt).total_seconds() / 3600
            exchange = row['exchange']
            symbol   = row['symbol']
            bias     = row['bias']
            sl       = row['stop_loss']
            t1, t2, t3 = row['t1'], row['t2'], row['t3']
            # entry zone for confirmation check (fall back to entry_price if zone wasn't stored)
            e_low  = row['entry_low']  if row['entry_low']  else row['entry_price']
            e_high = row['entry_high'] if row['entry_high'] else row['entry_price']

            # FIX A2 — fetch OHLC candles since signal was generated
            candles = _fetch_ohlc_since(exchange, symbol, row['scan_time'])

            # FIX A1 — resolve outcome from candle sequence
            if candles:
                outcome, best_tgt, sl_after = _resolve_outcome_from_candles(
                    candles, bias, sl, t1, t2, t3
                )
            else:
                # Fallback to tick price if candle fetch fails
                current = _get_live_price(symbol, exchange)
                if current == 0:
                    continue
                if bias == 'LONG':
                    sl_hit = current <= sl
                    outcome = 'sl_hit' if sl_hit else (
                        't3_hit' if current >= t3 else
                        't2_hit' if current >= t2 else
                        't1_hit' if current >= t1 else 'pending')
                else:
                    sl_hit = current >= sl
                    outcome = 'sl_hit' if sl_hit else (
                        't3_hit' if current <= t3 else
                        't2_hit' if current <= t2 else
                        't1_hit' if current <= t1 else 'pending')
                best_tgt  = None
                sl_after  = 0

            updates = {}

            # Snapshot checkpoints (keep for historical reference)
            if candles:
                latest_price = f"{candles[-1]['close']:.6f}"
            else:
                latest_price = "0"

            if hours >= 4  and not row['check_4h']:  updates['check_4h']  = latest_price
            if hours >= 8  and not row['check_8h']:  updates['check_8h']  = latest_price
            if hours >= 24 and not row['check_24h']: updates['check_24h'] = latest_price
            if hours >= 48 and not row['check_48h']: updates['check_48h'] = latest_price

            # ── FIX P5 + FIX EC: Entry confirmation ───────────────────────────────
            # FIX EC: Old logic confirmed entry if ANY candle's [low,high] range
            # overlapped the entry zone — a pure wick through the zone qualified.
            # Problem: since the scan price IS within the entry zone at scan time,
            # almost every subsequent candle's range overlaps the zone trivially
            # (price doesn't teleport). This caused nearly all signals to be
            # marked entry_confirmed=1, inflating the "confirmed" trade count.
            #
            # New logic: require the candle OPEN or CLOSE (the candle BODY) to
            # lie within [entry_low, entry_high]. This means price sustained inside
            # the zone long enough to open or close there — a realistic fill
            # condition — rather than just wicking through it momentarily.
            entry_confirmed = row['entry_confirmed']
            if entry_confirmed != 1 and candles:
                for candle in candles:
                    body_in_zone = (e_low <= candle['open'] <= e_high or
                                    e_low <= candle['close'] <= e_high)
                    if body_in_zone:
                        entry_confirmed = 1
                        break
                else:
                    if entry_confirmed == -1:   # legacy row: treat as unconfirmed
                        entry_confirmed = 0

            if entry_confirmed != row['entry_confirmed']:
                updates['entry_confirmed'] = entry_confirmed

            # ── FIX P3: Resolve immediately; expire hard at 48h ───────────────
            if outcome != 'pending':
                # Definitively resolved — write now, don't wait for any checkpoint
                updates['outcome']         = outcome
                updates['best_target_hit'] = best_tgt
                updates['sl_after_target'] = sl_after
                # Also stamp the 48h checkpoint if we're past 48h
                if hours >= 48 and not row['check_48h']:
                    updates['check_48h'] = latest_price
            elif hours >= 48:
                # Maximum window reached and still undecided → expire cleanly.
                # 'expired' is excluded from win/loss counts just like legacy
                # 'pending' rows were, but it no longer pollutes the pending count.
                updates['outcome']         = 'expired'
                updates['best_target_hit'] = best_tgt
                updates['sl_after_target'] = sl_after
                if not row['check_48h']:
                    updates['check_48h'] = latest_price

            if updates:
                conn2 = db_connect()
                c2    = conn2.cursor()
                set_clause = ', '.join(f"{k}=?" for k in updates)
                c2.execute(f"UPDATE signal_outcomes SET {set_clause} WHERE id=?",
                           list(updates.values()) + [row['id']])
                conn2.commit()
                conn2.close()

            # ── DONE → REMOVE FROM SCAN HISTORY ──
            # Once a call is resolved (target hit, SL hit, or expired at 48h),
            # drop its scan_results row so /pnl only ever reflects a live call
            # and never serves a stale, already-finished signal for this pair.
            final_outcome = updates.get('outcome')
            if final_outcome and final_outcome != 'pending':
                try:
                    conn3 = db_connect()
                    conn3.execute(
                        "DELETE FROM scan_results WHERE exchange=? AND symbol=? AND bias=?",
                        (exchange, symbol, bias),
                    )
                    conn3.commit()
                    conn3.close()
                except Exception as e:
                    logger.warning("scan_results cleanup failed for %s %s: %s", exchange, symbol, e)

            if outcome != 'pending':
                conf_tag = {1: 'confirmed', 0: 'missed-entry', -1: 'legacy'}
                logger.info("Outcome resolved: %s %s %s → %s (best=%s, sl_after=%s, entry=%s)",
                            exchange, symbol, bias, outcome, best_tgt, sl_after,
                            conf_tag.get(entry_confirmed, '?'))

        except Exception as e:
            logger.warning("Outcome check error for %s: %s", row['symbol'], e)


# ─────────────────────────────────────────────
# /stats — Dynamic win rate stats
# Supports minute-level windows for short-term live evaluation.
#
# Usage:
#   /stats          → all time (resolved signal_outcomes DB)
#   /stats 5m       → last 5 minutes  (live price check vs scan_results)
#   /stats 30m      → last 30 minutes (live price check)
#   /stats 1h       → last 1 hour     (live price check)
#   /stats 24       → last 24 hours   (resolved DB outcomes)
#   /stats 168      → last 7 days     (resolved DB outcomes)
#   /stats 720      → last 30 days    (resolved DB outcomes)
#
# SHORT-WINDOW MODE (< 2h):
#   For windows too short for signal_outcomes to have resolved check-ins,
#   we pull raw signals from scan_results and do a live price fetch for
#   every signal in the window — the same approach /compare uses.
#   Reported as Raw PnL (unlevered) + a win count based on green/red move.
#   This gives instant feedback without waiting for the 4h outcome checker.
# ───────────────────────────────────────────��─

def _parse_stats_arg(arg: str):
    """
    Parse a /stats argument into (minutes, label, mode).
    mode: 'live' for short windows (<= 120 min), 'db' for longer ones.

    Accepted formats:
      5m   30m   1h   2h          → live mode
      24   168   720  (integers)  → db mode (hours)
      24h  168h                   → db mode (hours)
    """
    arg = arg.strip().lower()

    # Minute suffix: 5m, 30m, 90m …
    if arg.endswith('m') and arg[:-1].isdigit():
        mins = int(arg[:-1])
        if mins <= 0:
            raise ValueError
        label = f"LAST {mins}MIN"
        return mins, label, 'live'

    # Hour suffix: 1h, 2h, 24h …
    if arg.endswith('h') and arg[:-1].isdigit():
        hrs = int(arg[:-1])
        if hrs <= 0:
            raise ValueError
        mins = hrs * 60
        if mins <= 120:
            label = f"LAST {hrs}H"
            return mins, label, 'live'
        else:
            label = f"LAST {hrs}H" if hrs % 24 != 0 else f"LAST {hrs // 24}D"
            return mins, label, 'db'

    # Plain integer = hours (legacy)
    if arg.isdigit():
        hrs = int(arg)
        if hrs <= 0:
            raise ValueError
        mins = hrs * 60
        if mins <= 120:
            label = f"LAST {hrs}H"
            return mins, label, 'live'
        else:
            label = f"LAST {hrs}H" if hrs % 24 != 0 else f"LAST {hrs // 24}D"
            return mins, label, 'db'

    raise ValueError(f"Unrecognised format: {arg}")


async def _stats_live(update_or_query, label: str, minutes: int, *, is_callback=False):
    """
    SHORT-WINDOW STATS — live price comparison against scan_results rows.
    Mirrors the /compare logic: fetch current price for every signal that
    was scanned within the last `minutes` minutes, then count green/red.

    Reported metrics (all unlevered / raw):
      • Win rate  = fraction of signals where current price moved in signal direction
      • Avg raw % move
      • T1 proximity: how many signals have current price within 30% of T1 range
      • SL proximity: how many are within 20% of stop loss
    """
    cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()

    conn = db_connect()
    rows = conn.execute(
        "SELECT exchange, symbol, bias, confidence, data_json, scan_time "
        "FROM scan_results WHERE scan_time >= ? ORDER BY confidence DESC",
        (cutoff,)
    ).fetchall()
    conn.close()

    if not rows:
        msg = (
            f"📊 LIVE STATS ({label})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"No signals found in the last {minutes} minute(s).\n\n"
            f"Run /scan to generate signals, then check back."
        )
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("30M",  callback_data="stats_time|m30"),
            InlineKeyboardButton("1H",   callback_data="stats_time|m60"),
            InlineKeyboardButton("24H",  callback_data="stats_time|24"),
            InlineKeyboardButton("All",  callback_data="stats_time|0"),
        ]])
        if is_callback:
            await update_or_query.edit_message_text(msg, reply_markup=kb)
        else:
            await update_or_query.message.reply_text(msg, reply_markup=kb)
        return

    wins = 0; losses = 0; skipped = 0
    t1_near = 0; sl_near = 0
    raw_pcts = []
    seen = set()   # deduplicate same symbol appearing in multiple scans

    for r in rows:
        key = f"{r['exchange']}_{r['symbol']}"
        if key in seen:
            continue
        seen.add(key)

        try:
            d = json.loads(r['data_json'])
        except Exception:
            d = {}

        signal_price = d.get('price', 0)
        stop_loss    = d.get('stop_loss', 0)
        t1           = d.get('t1', 0)
        bias         = r['bias']
        exchange     = r['exchange']
        symbol       = r['symbol']

        # Fetch live price
        try:
            live = _get_live_price(symbol, exchange)
        except Exception:
            live = 0

        if not live or not signal_price:
            skipped += 1
            continue

        raw_pct = ((live - signal_price) / signal_price) * 100
        if bias == 'SHORT':
            raw_pct = -raw_pct   # positive = moving in signal direction

        raw_pcts.append(raw_pct)

        if raw_pct >= 0:
            wins += 1
        else:
            losses += 1

        # T1 proximity: within 30% of the distance from signal_price to t1
        if t1 and signal_price:
            t1_dist  = abs(t1 - signal_price)
            cur_dist = abs(live - signal_price)
            if t1_dist > 0 and (cur_dist / t1_dist) >= 0.7:
                t1_near += 1

        # SL proximity: within 20% of distance from signal_price to stop_loss
        if stop_loss and signal_price:
            sl_dist  = abs(stop_loss - signal_price)
            cur_dist = abs(live - signal_price)
            if sl_dist > 0 and bias == 'LONG'  and live < signal_price and (cur_dist / sl_dist) >= 0.8:
                sl_near += 1
            if sl_dist > 0 and bias == 'SHORT' and live > signal_price and (cur_dist / sl_dist) >= 0.8:
                sl_near += 1

    total    = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    avg_pct  = (sum(raw_pcts) / len(raw_pcts)) if raw_pcts else 0
    bar      = "█" * int(win_rate / 10) + "░" * (10 - int(win_rate / 10))
    avg_emoji = "🟢" if avg_pct >= 0 else "🔴"

    # Quick-filter keyboard for the data path (the empty-rows branch above
    # defines its own kb; this path previously referenced an undefined `kb`).
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("30M",  callback_data="stats_time|m30"),
        InlineKeyboardButton("1H",   callback_data="stats_time|m60"),
        InlineKeyboardButton("24H",  callback_data="stats_time|24"),
        InlineKeyboardButton("All",  callback_data="stats_time|0"),
    ]])

    now_str = datetime.now().strftime("%H:%M:%S")

    warning = ""
    if minutes <= 15:
        warning = (
            "⚠️ Very short window — most moves are noise.\n"
            "Green/red reflects price tick direction only,\n"
            "not whether T1/SL was reached.\n\n"
        )

    msg = (
        f"📊 LIVE STATS ({label})  |  🔄 {now_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{warning}"
        f"🎯 WIN RATE (price moving in signal direction)\n"
        f"{bar} {win_rate:.1f}%\n"
        f"({wins} green / {total} signals checked)\n\n"
        f"{avg_emoji} Avg raw move: {avg_pct:+.3f}%\n\n"
        f"📍 PROXIMITY\n"
        f"   Near T1 (≥70% of way): {t1_near}\n"
        f"   Near SL (≥80% of way): {sl_near}\n\n"
        f"ℹ️ Signals checked: {total}  |  Skipped (no price): {skipped}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Live price vs scan price  |  unlevered raw %\n"
        f"Quick filter 👇"
    )

    if is_callback:
        await update_or_query.edit_message_text(msg, reply_markup=kb)
    else:
        await update_or_query.message.reply_text(msg, reply_markup=kb)


# ────────────────────────────────────
# /stats best-return leaderboard (SAKZ_STATS_V2)
#
# Dynamic time windows. The suffix decides the unit:
#   m = minutes, h = hours, d = days, w = weeks
# Examples: /stats 5m  /stats 4h  /stats 12h  /stats 1d  /stats 1w  /stats 2w
# A bare number is treated as hours (legacy). No argument = last 15 minutes.
# The card ranks every signal scanned in the window by leverage-adjusted
# return (live price vs the scan entry, in the signal's direction).
# ──────────────────────────────���─────
_STATS_UNIT_MIN = {"m": 1, "h": 60, "d": 1440, "w": 10080}
_STATS_MAX_SYMBOLS = 60   # cap live-price lookups so the card stays snappy


def _fmt_window_label(minutes: int) -> str:
    """Human label for a window length in minutes (e.g. 15M, 4H, 1D, 2W)."""
    if minutes % 10080 == 0:
        return f"{minutes // 10080}W"
    if minutes % 1440 == 0:
        return f"{minutes // 1440}D"
    if minutes % 60 == 0:
        return f"{minutes // 60}H"
    return f"{minutes}M"


def _parse_stats_window(arg: str):
    """Parse a dynamic /stats argument into (minutes, label).

    Supports a unit suffix m/h/d/w (e.g. 5m, 4h, 1d, 2w) and a bare integer
    that is interpreted as hours for backwards compatibility. Raises ValueError
    on anything else."""
    arg = (arg or "").strip().lower()
    if not arg:
        return 15, _fmt_window_label(15)

    unit = arg[-1]
    if unit in _STATS_UNIT_MIN:
        num = arg[:-1]
        if not num.isdigit() or int(num) <= 0:
            raise ValueError(f"bad stats window: {arg}")
        minutes = int(num) * _STATS_UNIT_MIN[unit]
        return minutes, _fmt_window_label(minutes)

    # Bare integer = hours (legacy behaviour)
    if arg.isdigit() and int(arg) > 0:
        minutes = int(arg) * 60
        return minutes, _fmt_window_label(minutes)

    raise ValueError(f"Unrecognised window: {arg}")


def _stats_best_keyboard(minutes: int) -> InlineKeyboardMarkup:
    """Default quick-filter buttons: 4h / 12h / 1w / 2w, plus refresh + win-rate."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("4H",  callback_data="statsbest|240"),
            InlineKeyboardButton("12H", callback_data="statsbest|720"),
            InlineKeyboardButton("1W",  callback_data="statsbest|10080"),
            InlineKeyboardButton("2W",  callback_data="statsbest|20160"),
        ],
        [
            InlineKeyboardButton("🔄 Refresh",  callback_data=f"statsbest|{minutes}"),
            InlineKeyboardButton("🎯 Win rate", callback_data="stats_time|0"),
        ],
    ])


def _collect_best_signals(minutes: int):
    """Pull unique signals scanned within the window from scan_results.

    Returns a list of dicts (symbol, exchange, bias, confidence, signal_price,
    leverage) ranked by confidence, capped at _STATS_MAX_SYMBOLS."""
    cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
    conn = db_connect()
    rows = conn.execute(
        "SELECT exchange, symbol, bias, confidence, data_json, scan_time "
        "FROM scan_results WHERE scan_time >= ? ORDER BY confidence DESC",
        (cutoff,)
    ).fetchall()
    conn.close()

    out = []
    seen = set()
    for r in rows:
        key = f"{r['exchange']}_{r['symbol']}"
        if key in seen:
            continue
        seen.add(key)
        try:
            d = json.loads(r['data_json'])
        except Exception:
            d = {}
        sig_price = d.get('price', 0) or 0
        if not sig_price:
            continue
        # leverage is stored as a dict (e.g. {'suggested': 4}) by
        # calculate_leverage; normalise to a numeric multiplier so the
        # leverage-adjusted return math (raw_pct * lev) never blows up.
        lev_raw = d.get('leverage', 1) or 1
        if isinstance(lev_raw, dict):
            lev_raw = lev_raw.get('suggested', 1) or 1
        try:
            lev_num = float(lev_raw)
        except (TypeError, ValueError):
            lev_num = 1.0
        out.append({
            "symbol":       r['symbol'],
            "exchange":     r['exchange'],
            "bias":         r['bias'],
            "confidence":   r['confidence'],
            "signal_price": sig_price,
            "leverage":     lev_num,
        })
        if len(out) >= _STATS_MAX_SYMBOLS:
            break
    return out


async def _stats_best_card(update_or_query, minutes: int, label: str, *, is_callback=False):
    """Build + send the best-return leaderboard card for a dynamic window."""
    sigs = _collect_best_signals(minutes)
    kb   = _stats_best_keyboard(minutes)

    async def _send(text):
        if is_callback:
            await update_or_query.edit_message_text(text, reply_markup=kb)
        else:
            await update_or_query.message.reply_text(text, reply_markup=kb)

    if not sigs:
        await _send(
            f"🏆 BEST SIGNALS — LAST {label}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"No signals were scanned in this window.\n"
            f"Run /scan first, then check back.\n\n"
            f"Pick another window 👇"
        )
        return

    # Fetch all live prices concurrently (network-bound) so the card is fast
    # even with dozens of symbols.
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(SCAN_EXECUTOR, _get_live_price, s["symbol"], s["exchange"])
        for s in sigs
    ]
    prices = await asyncio.gather(*tasks, return_exceptions=True)

    ranked = []
    skipped = 0
    for s, live in zip(sigs, prices):
        if isinstance(live, Exception) or not live:
            skipped += 1
            continue
        sig_price = s["signal_price"]
        raw_pct = ((live - sig_price) / sig_price) * 100
        if s["bias"] == "SHORT":
            raw_pct = -raw_pct
        lev = s["leverage"] or 1
        ranked.append({
            **s,
            "live":     live,
            "raw_pct":  raw_pct,
            "lev_pct":  raw_pct * lev,
        })

    if not ranked:
        await _send(
            f"🏆 BEST SIGNALS — LAST {label}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Found {len(sigs)} signal(s) but no live prices were available "
            f"right now. Try again shortly."
        )
        return

    ranked.sort(key=lambda x: x["lev_pct"], reverse=True)
    top = ranked[:10]

    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    lines_out = []
    for i, s in enumerate(top):
        rank  = medals.get(i, f"{i+1}.")
        arrow = "🟢" if s["lev_pct"] >= 0 else "🔴"
        dir_e = "👈 LONG" if s["bias"] == "LONG" else "👉 SHORT"
        lines_out.append(
            f"{rank} {arrow} {s['symbol']}  {dir_e}\n"
            f"     {s['lev_pct']:+.2f}% @ {s['leverage']:.0f}x "
            f"(raw {s['raw_pct']:+.2f}%)  · conf {s['confidence']}/10"
        )

    avg_lev = sum(s["lev_pct"] for s in ranked) / len(ranked)
    winners = sum(1 for s in ranked if s["lev_pct"] > 0)
    best    = ranked[0]
    now_str = datetime.now().strftime("%H:%M:%S")

    note = ""
    if minutes <= 15:
        note = ("⚠️ Short window — moves are mostly noise and reflect live price\n"
                "direction only, not whether T1/SL was hit.\n\n")

    msg = (
        f"🏆 BEST SIGNALS — LAST {label}  |  🔄 {now_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{note}"
        + "\n".join(lines_out) +
        f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👈 Top: {best['symbol']} {best['lev_pct']:+.2f}% (lev-adj)\n"
        f"📊 Avg: {avg_lev:+.2f}%  ·  {winners}/{len(ranked)} in profit\n"
        f"🔎 {len(ranked)} priced" + (f"  ·  {skipped} skipped (no price)" if skipped else "") + "\n"
        f"Live price vs scan entry  ·  leverage-adjusted return\n"
        f"Pick a window 👇"
    )
    await _send(msg)


async def statsbest_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inline button handler for the best-return leaderboard windows."""
    query = update.callback_query
    await query.answer()
    try:
        minutes = int(query.data.split('|')[1])
    except (IndexError, ValueError):
        minutes = 15
    await _stats_best_card(query, minutes, _fmt_window_label(minutes), is_callback=True)



async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    args = context.args

    # SAKZ_STATS_V2 — /stats now shows the best-return leaderboard. Default
    # window is the last 15 minutes; any dynamic window (m/h/d/w or bare hours)
    # is accepted. Detailed win-rate stats remain reachable via the
    # "🎯 Win rate" button on the card (stats_time|0).
    if not args:
        # Default to a 24h window. A 15-minute default made /stats look broken:
        # most users have not scanned in the last 15 min, so the leaderboard was
        # almost always empty. 24h gives a useful default; buttons still allow
        # 4H/12H/1W/2W and minute windows.
        await _stats_best_card(update, 1440, _fmt_window_label(1440), is_callback=False)
        return

    try:
        minutes, label = _parse_stats_window(args[0])
    except ValueError:
        await update.message.reply_text(
            "⚠️ Invalid time format. The unit decides the window:\n"
            "   m = minutes · h = hours · d = days · w = weeks\n\n"
            "/stats          → best signals, last 15 minutes\n"
            "/stats 5m       → last 5 minutes\n"
            "/stats 4h       → last 4 hours\n"
            "/stats 12h      → last 12 hours\n"
            "/stats 1d       → last 1 day\n"
            "/stats 1w       → last 1 week\n"
            "/stats 2w       → last 2 weeks"
        )
        return

    await _stats_best_card(update, minutes, label, is_callback=False)
    return


async def _stats_winrate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Legacy detailed win-rate report (kept for reference / future reuse)."""
    args = context.args
    hours_filter = None
    label = "ALL TIME"

    if args:
        try:
            minutes, label, mode = _parse_stats_arg(args[0])
        except ValueError:
            await update.message.reply_text("⚠️ Invalid time format.")
            return

        if mode == 'live':
            await _stats_live(update, label, minutes, is_callback=False)
            return

        # DB mode — convert minutes back to hours for cutoff
        hours_filter = minutes // 60

    conn = db_connect()
    c    = conn.cursor()

    if hours_filter:
        cutoff = (datetime.now() - timedelta(hours=hours_filter)).isoformat()
        c.execute(
            "SELECT outcome, bias, confidence, sl_after_target, entry_confirmed "
            "FROM signal_outcomes "
            "WHERE outcome NOT IN ('pending','expired') AND scan_time >= ?", (cutoff,))
    else:
        c.execute(
            "SELECT outcome, bias, confidence, sl_after_target, entry_confirmed "
            "FROM signal_outcomes "
            "WHERE outcome NOT IN ('pending','expired')")

    rows = c.fetchall()

    if hours_filter:
        c.execute("SELECT COUNT(*) as cnt FROM signal_outcomes WHERE outcome='pending' AND scan_time >= ?", (cutoff,))
    else:
        c.execute("SELECT COUNT(*) as cnt FROM signal_outcomes WHERE outcome='pending'")
    pending = c.fetchone()['cnt']

    # FIX P5 — count expired and unconfirmed-entry signals separately
    if hours_filter:
        c.execute("SELECT COUNT(*) as cnt FROM signal_outcomes WHERE outcome='expired' AND scan_time >= ?", (cutoff,))
    else:
        c.execute("SELECT COUNT(*) as cnt FROM signal_outcomes WHERE outcome='expired'")
    expired_cnt = c.fetchone()['cnt']
    conn.close()

    # FIX P5 — split into confirmed vs missed-entry rows for stats
    confirmed_rows = [r for r in rows if r['entry_confirmed'] != 0]  # 1 or -1 (legacy)
    missed_rows    = [r for r in rows if r['entry_confirmed'] == 0]

    if not rows:
        await update.message.reply_text(
            f"📊 No completed signal outcomes for {label}.\n\n"
            f"Outcomes are tracked at 4h, 8h, 24h, and 48h after each scan.\n"
            f"For short windows use minute format: /stats 5m  /stats 30m  /stats 1h"
        )
        return

    # FIX P5 — base all win-rate stats on confirmed entries only
    total        = len(confirmed_rows)
    wins         = sum(1 for r in confirmed_rows if r['outcome'] in ('t1_hit', 't2_hit', 't3_hit'))
    sl_hits      = sum(1 for r in confirmed_rows if r['outcome'] == 'sl_hit')
    t1_hits      = sum(1 for r in confirmed_rows if r['outcome'] == 't1_hit')
    t2_hits      = sum(1 for r in confirmed_rows if r['outcome'] == 't2_hit')
    t3_hits      = sum(1 for r in confirmed_rows if r['outcome'] == 't3_hit')
    # FIX A1 — partial wins: hit a target but SL triggered on remainder
    partial_wins = sum(1 for r in confirmed_rows if r['outcome'] == 'sl_hit'
                       and r['sl_after_target'] == 1)

    win_rate = (wins / total * 100) if total > 0 else 0

    high_conf_rows = [r for r in confirmed_rows if r['confidence'] >= 8]
    hc_wins  = sum(1 for r in high_conf_rows if r['outcome'] in ('t1_hit','t2_hit','t3_hit'))
    hc_rate  = (hc_wins / len(high_conf_rows) * 100) if high_conf_rows else 0

    t1_rate = (t1_hits / total * 100) if total > 0 else 0
    t2_rate = (t2_hits / total * 100) if total > 0 else 0
    t3_rate = (t3_hits / total * 100) if total > 0 else 0
    sl_rate = (sl_hits / total * 100) if total > 0 else 0

    bar_filled = int(win_rate / 10)
    bar_empty  = 10 - bar_filled
    bar        = "█" * bar_filled + "���" * bar_empty

    longs      = [r for r in confirmed_rows if r['bias'] == 'LONG']
    shorts     = [r for r in confirmed_rows if r['bias'] == 'SHORT']
    long_wins  = sum(1 for r in longs  if r['outcome'] in ('t1_hit','t2_hit','t3_hit'))
    short_wins = sum(1 for r in shorts if r['outcome'] in ('t1_hit','t2_hit','t3_hit'))

    long_rate  = (long_wins  / len(longs)  * 100) if longs  else 0
    short_rate = (short_wins / len(shorts) * 100) if shorts else 0

    # Inline time filter buttons — includes minute shortcuts
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("5M",  callback_data="stats_time|m5"),
        InlineKeyboardButton("30M", callback_data="stats_time|m30"),
        InlineKeyboardButton("1H",  callback_data="stats_time|m60"),
        InlineKeyboardButton("24H", callback_data="stats_time|24"),
        InlineKeyboardButton("All", callback_data="stats_time|0"),
    ]])

    partial_line  = (f"   ↪️ Partial wins (T hit→SL): {partial_wins}\n" if partial_wins else "")
    missed_line   = (f"⚠️ Missed entries (price never hit zone): {len(missed_rows)}\n" if missed_rows else "")
    expired_line  = (f"⏸ Expired (48h, no resolution): {expired_cnt}\n" if expired_cnt else "")

    msg = (
        f"📊 SAKZ BOT — SIGNAL STATS ({label})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🎯 OVERALL WIN RATE\n"
        f"{bar} {win_rate:.1f}%\n"
        f"({wins} wins / {total} confirmed trades)\n\n"
        f"⭐ HIGH CONFIDENCE (8–10/10)\n"
        f"   Win rate: {hc_rate:.1f}% ({hc_wins}/{len(high_conf_rows)})\n\n"
        f"📈 TARGET BREAKDOWN\n"
        f"   T1 reached:   {t1_hits}  ({t1_rate:.0f}% of trades)\n"
        f"   T2 reached:   {t2_hits}  ({t2_rate:.0f}% of trades)\n"
        f"   T3 reached:   {t3_hits}  ({t3_rate:.0f}% of trades)\n"
        f"   SL triggered: {sl_hits}  ({sl_rate:.0f}% of trades)\n"
        f"{partial_line}"
        f"\n"
        f"🟢 LONG  win rate: {long_rate:.1f}% ({long_wins}/{len(longs)})\n"
        f"🔴 SHORT win rate: {short_rate:.1f}% ({short_wins}/{len(shorts)})\n\n"
        f"⏳ Pending (still tracking): {pending}\n"
        f"{expired_line}"
        f"{missed_line}"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Outcomes: candle OHLC verified, deduplicated\n"
        f"Stats: confirmed entries only (entry zone traded)\n"
        f"Quick filter 👇"
    )

    await update.message.reply_text(msg, reply_markup=keyboard)



# ─────────────────────────────────────────────
# FIX #BT — /backtest COMMAND
# Queries signal_outcomes to compute actual win rates
# per confidence band and vol regime, then recommends
# threshold recalibrations.
#
# Usage:
#   /backtest          → full breakdown, all time
#   /backtest 168      ��� last 7 days only
# ─��───────────��───────────────────────────────
async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    args         = context.args
    hours_filter = None
    label        = "ALL TIME"

    if args:
        try:
            hours_filter = int(args[0])
            if hours_filter <= 0:
                raise ValueError
            label = f"LAST {hours_filter // 24}D" if hours_filter % 24 == 0 else f"LAST {hours_filter}H"
        except ValueError:
            await update.message.reply_text(
                "⚠️ Usage:\n"
                "/backtest          — all time\n"
                "/backtest 168      — last 7 days\n"
                "/backtest 720      — last 30 days"
            )
            return

    await update.message.reply_text("🔬 Running backtest analysis���")

    conn = db_connect()
    c    = conn.cursor()

    if hours_filter:
        cutoff = (datetime.now() - timedelta(hours=hours_filter)).isoformat()
        c.execute(
            "SELECT outcome, bias, confidence, sl_after_target, entry_confirmed, data_json "
            "FROM signal_outcomes so "
            "LEFT JOIN scan_results sr ON sr.id = so.signal_id "
            "WHERE so.outcome NOT IN ('pending','expired') AND so.scan_time >= ?",
            (cutoff,)
        )
    else:
        c.execute(
            "SELECT outcome, bias, confidence, sl_after_target, entry_confirmed, data_json "
            "FROM signal_outcomes so "
            "LEFT JOIN scan_results sr ON sr.id = so.signal_id "
            "WHERE so.outcome NOT IN ('pending','expired')"
        )

    rows = c.fetchall()
    conn.close()

    # Filter to confirmed entries only (exclude missed-entry noise)
    confirmed = [r for r in rows if r['entry_confirmed'] != 0]

    if len(confirmed) < 10:
        await update.message.reply_text(
            f"⚠️ Only {len(confirmed)} confirmed trades in DB for {label}.\n"
            f"Need at least 10 to produce meaningful analysis.\n\n"
            f"Keep the bot running and try again after more signals complete."
        )
        return

    # ��─ Win rate by confidence band ���───────────────────────────────────
    bands = [
        ('4–5', lambda r: r['confidence'] in (4, 5)),
        ('6–7', lambda r: r['confidence'] in (6, 7)),
        ('8',   lambda r: r['confidence'] == 8),
        ('9',   lambda r: r['confidence'] == 9),
        ('10',  lambda r: r['confidence'] == 10),
    ]
    band_lines = []
    worst_band_rate = 100.0
    best_band       = None
    min_viable_conf = None  # lowest conf band with ≥50% win rate and ≥5 trades

    for label_b, fn in bands:
        subset = [r for r in confirmed if fn(r)]
        if not subset:
            band_lines.append(f"   {label_b}/10 : — (no data)")
            continue
        wins = sum(1 for r in subset if r['outcome'] in ('t1_hit','t2_hit','t3_hit'))
        rate = wins / len(subset) * 100
        bar  = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
        band_lines.append(f"   {label_b}/10 : {bar} {rate:.0f}%  ({wins}/{len(subset)})")
        if rate < worst_band_rate:
            worst_band_rate = rate
        if rate >= 50 and len(subset) >= 5:
            if min_viable_conf is None:
                min_viable_conf = label_b

    # ── Win rate by vol regime ─────────────────────────────────────────
    # vol_regime is stored in data_json if the signal was scored after FIX #VA
    vol_band_lines = []
    vol_regimes    = ['RANGING', 'LOW', 'MEDIUM', 'HIGH', 'EXTREME']
    for vr in vol_regimes:
        subset = []
        for r in confirmed:
            try:
                d  = json.loads(r['data_json']) if r['data_json'] else {}
                vr_val = d.get('vol_regime', 'UNKNOWN')
            except Exception:
                vr_val = 'UNKNOWN'
            if vr_val == vr:
                subset.append(r)
        if not subset:
            continue
        wins = sum(1 for r in subset if r['outcome'] in ('t1_hit','t2_hit','t3_hit'))
        rate = wins / len(subset) * 100
        vol_band_lines.append(f"   {vr:<8}: {rate:.0f}%  ({wins}/{len(subset)})")

    # ── Bias split ─────────────────────────────────────────────────────
    longs  = [r for r in confirmed if r['bias'] == 'LONG']
    shorts = [r for r in confirmed if r['bias'] == 'SHORT']
    lr = sum(1 for r in longs  if r['outcome'] in ('t1_hit','t2_hit','t3_hit')) / len(longs)  * 100 if longs  else 0
    sr = sum(1 for r in shorts if r['outcome'] in ('t1_hit','t2_hit','t3_hit')) / len(shorts) * 100 if shorts else 0

    # ── Target depth analysis ──────────────────────────────────────────
    t1_c = sum(1 for r in confirmed if r['outcome'] in ('t1_hit','t2_hit','t3_hit'))
    t2_c = sum(1 for r in confirmed if r['outcome'] in ('t2_hit','t3_hit'))
    t3_c = sum(1 for r in confirmed if r['outcome'] == 't3_hit')
    sl_c = sum(1 for r in confirmed if r['outcome'] == 'sl_hit')
    tot  = len(confirmed)

    # ── Calibration recommendations ────────────────────────────────────
    recommendations = []
    overall_wr = sum(1 for r in confirmed if r['outcome'] in ('t1_hit','t2_hit','t3_hit')) / tot * 100

    if overall_wr < 45:
        recommendations.append("⚠️ Overall win rate < 45% — consider raising the minimum confidence floor from 4 to 5.")
    if overall_wr > 70:
        recommendations.append("✅ Win rate > 70% — bot is over-filtering. You can lower the confidence floor to capture more trades.")

    if worst_band_rate < 35:
        recommendations.append(f"⚠️ A confidence band has win rate < 35% — consider raising minimum score floor for that band.")

    t2_rate = t2_c / tot * 100 if tot else 0
    if t2_rate < 20:
        recommendations.append("📉 T2 reached < 20% of the time — targets may be too wide for current vol. Consider reducing T2/T3 multipliers.")
    elif t2_rate > 50:
        recommendations.append("📈 T2 reached > 50% — targets may be too conservative. Consider expanding T2/T3 multipliers.")

    if lr < 40 and len(longs) >= 10:
        recommendations.append("🔴 LONG win rate < 40% — regime gate may need tightening (lower threshold from conf��9 to conf≥8 in BEAR).")
    if sr < 40 and len(shorts) >= 10:
        recommendations.append("🔴 SHORT win rate < 40% — consider tightening short qualification bar.")

    if not recommendations:
        recommendations.append("✅ No critical issues detected. Bot calibration looks reasonable for current market conditions.")

    # ── Format output ──────────────────────────────────────────────────
    band_block   = "\n".join(band_lines)
    vol_block    = "\n".join(vol_band_lines) if vol_band_lines else "   (No vol regime data — run more scans after FIX #VA)"
    rec_block    = "\n".join(f"• {r}" for r in recommendations)
    viable_line  = (f"   Lowest viable conf band: {min_viable_conf}/10\n" if min_viable_conf else
                    "   No band with ≥50% win rate yet — not enough data.\n")

    msg = (
        f"��� BACKTEST ANALYSIS — {label}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Confirmed trades: {tot}  |  Overall WR: {overall_wr:.1f}%\n\n"
        f"📊 WIN RATE BY CONFIDENCE BAND\n"
        f"{band_block}\n"
        f"{viable_line}\n"
        f"📊 WIN RATE BY VOL REGIME\n"
        f"{vol_block}\n\n"
        f"📊 BIAS SPLIT\n"
        f"   🟢 LONG  : {lr:.1f}% ({len(longs)} trades)\n"
        f"   🔴 SHORT : {sr:.1f}% ({len(shorts)} trades)\n\n"
        f"📊 TARGET DEPTH\n"
        f"   T1+ reached : {t1_c}/{tot} ({t1_c/tot*100:.0f}%)\n"
        f"   T2+ reached : {t2_c}/{tot} ({t2_c/tot*100:.0f}%)\n"
        f"   T3  reached : {t3_c}/{tot} ({t3_c/tot*100:.0f}%)\n"
        f"   SL hit      : {sl_c}/{tot} ({sl_c/tot*100:.0f}%)\n\n"
        f"🛠 CALIBRATION RECOMMENDATIONS\n"
        f"{rec_block}\n"
        f"━━━━━━━━━━━━━━━━━━━━━��━━━━━━━━\n"
        f"💡 Tip: run /backtest 168 weekly to track drift"
    )

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📊 Full Stats", callback_data="stats_time|0"),
        InlineKeyboardButton("7D View",       callback_data="bt_time|168"),
        InlineKeyboardButton("30D View",      callback_data="bt_time|720"),
    ]])
    await update.message.reply_text(msg, reply_markup=keyboard)


async def bt_time_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inline button re-runs backtest for selected time window."""
    query = update.callback_query
    await query.answer()
    hours = int(query.data.split('|')[1])
    context.args = [str(hours)]
    # Patch update to use the original message for reply
    update._effective_message = query.message
    await backtest_command(update, context)


async def stats_time_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Inline button handler for stats time filters.
    Handles both minute buttons (m5, m30, m60) and hour/all-time buttons (24, 168, 0).
    """
    query = update.callback_query
    await query.answer()
    token = query.data.split('|')[1]

    # ── Minute-resolution buttons → live mode ──────────────────────────
    if token.startswith('m') and token[1:].isdigit():
        minutes = int(token[1:])
        label   = f"LAST {minutes}MIN" if minutes < 60 else f"LAST {minutes // 60}H"
        await _stats_live(query, label, minutes, is_callback=True)
        return

    # ── Hour / all-time buttons → DB resolved outcomes ─────────────────
    hours_filter = int(token)

    if hours_filter == 0:
        label  = "ALL TIME"
        cutoff = None
    elif hours_filter % 24 == 0:
        label  = f"LAST {hours_filter // 24}D"
        cutoff = (datetime.now() - timedelta(hours=hours_filter)).isoformat()
    else:
        label  = f"LAST {hours_filter}H"
        cutoff = (datetime.now() - timedelta(hours=hours_filter)).isoformat()

    conn = db_connect()
    c    = conn.cursor()
    if cutoff:
        c.execute(
            "SELECT outcome, bias, confidence, entry_confirmed FROM signal_outcomes "
            "WHERE outcome NOT IN ('pending','expired') AND scan_time >= ?", (cutoff,))
    else:
        c.execute(
            "SELECT outcome, bias, confidence, entry_confirmed FROM signal_outcomes "
            "WHERE outcome NOT IN ('pending','expired')")
    rows = c.fetchall()
    if cutoff:
        c.execute("SELECT COUNT(*) as cnt FROM signal_outcomes WHERE outcome='pending' AND scan_time >= ?", (cutoff,))
    else:
        c.execute("SELECT COUNT(*) as cnt FROM signal_outcomes WHERE outcome='pending'")
    pending = c.fetchone()['cnt']
    conn.close()

    if not rows:
        await query.edit_message_text(
            f"📊 No completed outcomes for {label}.\n"
            f"Try a shorter live window: tap 5M or 30M."
        )
        return

    confirmed_rows = [r for r in rows if r['entry_confirmed'] != 0]
    missed_cnt     = sum(1 for r in rows if r['entry_confirmed'] == 0)

    total      = len(confirmed_rows)
    wins       = sum(1 for r in confirmed_rows if r['outcome'] in ('t1_hit','t2_hit','t3_hit'))
    sl_hits    = sum(1 for r in confirmed_rows if r['outcome'] == 'sl_hit')
    t1_hits    = sum(1 for r in confirmed_rows if r['outcome'] == 't1_hit')
    t2_hits    = sum(1 for r in confirmed_rows if r['outcome'] == 't2_hit')
    t3_hits    = sum(1 for r in confirmed_rows if r['outcome'] == 't3_hit')
    win_rate   = (wins / total * 100) if total > 0 else 0
    hc_rows    = [r for r in confirmed_rows if r['confidence'] >= 8]
    hc_wins    = sum(1 for r in hc_rows if r['outcome'] in ('t1_hit','t2_hit','t3_hit'))
    hc_rate    = (hc_wins / len(hc_rows) * 100) if hc_rows else 0
    bar        = "█" * int(win_rate / 10) + "░" * (10 - int(win_rate / 10))
    longs      = [r for r in confirmed_rows if r['bias'] == 'LONG']
    shorts     = [r for r in confirmed_rows if r['bias'] == 'SHORT']
    long_wins  = sum(1 for r in longs  if r['outcome'] in ('t1_hit','t2_hit','t3_hit'))
    short_wins = sum(1 for r in shorts if r['outcome'] in ('t1_hit','t2_hit','t3_hit'))
    long_rate  = (long_wins  / len(longs)  * 100) if longs  else 0
    short_rate = (short_wins / len(shorts) * 100) if shorts else 0

    missed_line = (f"⚠️ Missed entries: {missed_cnt}\n" if missed_cnt else "")

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("5M",  callback_data="stats_time|m5"),
        InlineKeyboardButton("30M", callback_data="stats_time|m30"),
        InlineKeyboardButton("1H",  callback_data="stats_time|m60"),
        InlineKeyboardButton("24H", callback_data="stats_time|24"),
        InlineKeyboardButton("All", callback_data="stats_time|0"),
    ]])

    msg = (
        f"📊 SAKZ BOT — SIGNAL STATS ({label})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🎯 OVERALL WIN RATE\n"
        f"{bar} {win_rate:.1f}%\n"
        f"({wins} wins / {total} confirmed trades)\n\n"
        f"⭐ HIGH CONF (8–10/10): {hc_rate:.1f}% ({hc_wins}/{len(hc_rows)})\n\n"
        f"📈 TARGET BREAKDOWN\n"
        f"   T1: {t1_hits}  T2: {t2_hits}  T3: {t3_hits}  SL: {sl_hits}\n\n"
        f"🟢 LONG  {long_rate:.1f}% ({long_wins}/{len(longs)})\n"
        f"🔴 SHORT {short_rate:.1f}% ({short_wins}/{len(shorts)})\n\n"
        f"⏳ Pending: {pending}\n"
        f"{missed_line}"
        f"\nStats: confirmed entries only\n"
        f"Quick filter 👇"
    )
    await query.edit_message_text(msg, reply_markup=keyboard)



# ─────────────────────────────────────────────
# IMPROVEMENT #5 — /alert COMMAND
# Users register interest in a specific symbol.
# When that symbol appears in a scan above their
# min confidence threshold, they're notified.
# ─────────────────────────────────────────────
async def alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args  # e.g. ['BTCUSDT'] or ['BTCUSDT', '8']

    if not args:
        alerts = db_get_user_alerts(chat_id)
        if not alerts:
            await update.message.reply_text(
                "🔔 COIN ALERTS\n\n"
                "No active alerts. Set one with:\n"
                "/alert BTCUSDT        — notify when BTC appears (conf ≥ 7)\n"
                "/alert ETHUSDT 8      — notify only if conf ≥ 8\n"
                "/unalert BTCUSDT      — remove an alert"
            )
        else:
            lines = ["🔔 YOUR ACTIVE ALERTS\n"]
            for sym, conf in alerts:
                lines.append(f"  • {sym}  (min confidence: {conf}/10)")
            lines.append("\nUse /unalert SYMBOL to remove one.")
            await update.message.reply_text("\n".join(lines))
        return

    symbol   = args[0].upper().replace('_USDT', 'USDT')
    if not symbol.endswith('USDT'):
        symbol += 'USDT'
    min_conf = int(args[1]) if len(args) > 1 and args[1].isdigit() else 7
    min_conf = max(1, min(10, min_conf))

    db_save_user_alert(chat_id, symbol, min_conf)
    await update.message.reply_text(
        f"✅ Alert set for {symbol}\n"
        f"I'll notify you when it appears in a scan\n"
        f"with confidence ≥ {min_conf}/10.\n\n"
        f"Use /alert to see all your alerts.\n"
        f"Use /unalert {symbol} to remove it."
    )

async def unalert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args
    if not args:
        await update.message.reply_text("Usage: /unalert BTCUSDT")
        return
    symbol = args[0].upper().replace('_USDT', 'USDT')
    if not symbol.endswith('USDT'):
        symbol += 'USDT'
    db_remove_user_alert(chat_id, symbol)
    await update.message.reply_text(f"✅ Alert removed for {symbol}.")

async def notify_alerts(results, bot):
    """Called after every scan to push matching coin alerts to users."""
    all_alerts = db_get_all_alerts()
    if not all_alerts:
        return
    result_map = {f"{r['exchange']}_{r['symbol']}": r for r in results}
    # also index by symbol only (exchange-agnostic)
    sym_map = {}
    for r in results:
        sym = r['symbol'].replace('_USDT','USDT').replace('/','')
        if sym not in sym_map or r['confidence'] > sym_map[sym]['confidence']:
            sym_map[sym] = r

    notified = set()
    for chat_id, symbol, min_conf in all_alerts:
        if symbol in sym_map:
            r = sym_map[symbol]
            if r['confidence'] >= min_conf:
                key = f"{chat_id}_{symbol}"
                if key not in notified:
                    notified.add(key)
                    emoji = "🟢" if r['bias'] == "LONG" else "🔴"
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=(
                                f"🔔 ALERT: {symbol} just appeared in scan!\n\n"
                                f"{emoji} {r['bias']} on {r['exchange']}\n"
                                f"⭐ Confidence: {r['confidence']}/10\n"
                                f"💰 Price: ${r['price']:.6f}\n"
                                f"📥 Entry: ${r['entry_low']:.6f} → ${r['entry_high']:.6f}\n"
                                f"🛑 SL: ${r['stop_loss']:.6f}\n"
                                f"🎯 T1: ${r['t1']:.6f}\n\n"
                                f"Use /pick to set full tracking."
                            )
                        )
                    except Exception as e:
                        logger.warning("Alert notification failed for chat %s: %s", chat_id, e)


# ─────────────────────────────────────────────
# IMPROVEMENT #6 — /filter COMMAND
# Filter current results by bias and min confidence.
# e.g. /filter LONG 8
# ───────────────��─────────────────────────────
async def filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)

    if not state.last_scan_results:
        await update.message.reply_text("⚠️ No scan data yet. Run /scan first.")
        return

    args     = context.args
    bias     = None
    min_conf = 5

    for arg in args:
        if arg.upper() in ('LONG', 'SHORT'):
            bias = arg.upper()
        elif arg.isdigit():
            min_conf = max(1, min(10, int(arg)))

    filtered = [r for r in state.last_scan_results
                if r['confidence'] >= min_conf and (bias is None or r['bias'] == bias)]

    if not filtered:
        await update.message.reply_text(
            f"⚠️ No signals match: bias={bias or 'ANY'}, min conf={min_conf}/10\n"
            f"Try lowering the confidence threshold."
        )
        return

    age  = datetime.now() - state.last_scan_time
    mins = int(age.total_seconds() // 60)

    header_parts = []
    if bias:       header_parts.append(bias)
    header_parts.append(f"conf≥{min_conf}")
    header = " | ".join(header_parts)

    await send_signal_cards(
        update.message, filtered,
        title=f"🔍 FILTERED: {header} ({len(filtered)} signals, {mins} min ago)",
        max_show=20
    )


# ─────────────────────────────────────────────
# IMPROVEMENT #4 — AUTO-SCAN JOB
# Continuous auto-scan engine.
# Checks for high-confidence signals every 10 minutes.
# Sends full signal cards (>=8/10) instantly — no waiting for 4h intervals.
# Deduplication: same symbol+bias is not re-sent within 4 hours.
# ─────────────────────────────────────────────
auto_scan_subscribers: dict = {}  # chat_id → tf_pref (None = always, '15m'/'4h'/'1d'/etc.)
_autoscan_awaiting_tf: set = set()  # chat_ids that have been shown the TF menu and may type a custom TF

# Dedup cache (confidence-aware): { dedup_key: {"confidence": float, "sent_at": datetime} }
# The same call is NOT re-pushed to "always" subscribers unless the coin's
# confidence rises above the last value that was pushed.
_autoscan_sent: dict = {}
# Per-user, per-call last-notify time for timeframe subscribers, so a user who
# picked a timeframe is only alerted on that timeframe's cadence.
# { (chat_id, dedup_key): datetime_last_notified }
_autoscan_user_last: dict = {}
_AUTOSCAN_COOLDOWN_H = 4    # hours before the same signal can fire again
_AUTOSCAN_MIN_CONF   = 8    # minimum confidence to push a signal
_AUTOSCAN_SL_SUPPRESS_H = 24   # don't re-push a setup that hit SL within this window
# REMINDER cadence — re-surface a still-valid call (already pushed, confidence
# not risen) as a REMINDER once this many hours have elapsed since the last push.
_AUTOSCAN_REMINDER_H = float(os.environ.get("AUTOSCAN_REMINDER_H", "6") or 6)

# FIX #RESTART-FLOOD — _autoscan_sent lives in memory and is wiped on every
# restart.  Without a guard, the first continuous_scan_job tick after a restart
# treats every live high-confidence signal as "never sent" and floods every
# subscriber with re-sends.  During this startup grace window we still RECORD
# signals as sent (seeding the dedup table) but suppress the actual push, so
# only genuinely new signals fire once the bot has settled.
_BOT_START_TS                = datetime.now()
_AUTOSCAN_STARTUP_GRACE_SECS = 900   # 15 min — covers the first scan cycle(s)


def _autoscan_trade_type(signal: dict) -> tuple:
    """
    Classify a signal as Scalp / Intraday / Swing / Position based on hold_hours.
    Returns (label, emoji).
    """
    h = signal.get('hold_hours', 0)
    if h <= 4:
        return "Scalp",          "⚡"
    elif h <= 12:
        return "Intraday",       "🕐"
    elif h <= 48:
        return "Swing Trade",    "📈"
    else:
        return "Position Trade", "🏦"


def _autoscan_is_fresh(key: str) -> bool:
    """Legacy helper kept for compatibility. Confidence-aware dedup now lives in
    sakz_signal_logic.autoscan_decide_send and is applied in continuous_scan_job."""
    rec = _autoscan_sent.get(key)
    if not rec:
        return True
    sent_at = rec.get('sent_at') if isinstance(rec, dict) else rec
    if not sent_at:
        return True
    return (datetime.now() - sent_at).total_seconds() > _AUTOSCAN_COOLDOWN_H * 3600


def _autoscan_prune():
    """Drop dedup + per-user notify entries older than 2x the cooldown window."""
    cutoff = datetime.now() - timedelta(hours=_AUTOSCAN_COOLDOWN_H * 2)
    for k in list(_autoscan_sent):
        rec = _autoscan_sent.get(k)
        sent_at = rec.get('sent_at') if isinstance(rec, dict) else rec
        if sent_at and sent_at < cutoff:
            del _autoscan_sent[k]
    for k in list(_autoscan_user_last):
        if _autoscan_user_last.get(k) and _autoscan_user_last[k] < cutoff:
            del _autoscan_user_last[k]


def _register_lifecycle_signal(r, current_price=None):
    """Record/refresh the first-signal memory for a scanned signal (keeps the
    original call while the coin keeps running). Safe no-op on any error."""
    try:
        sig = dict(r)
        lev = sig.get('leverage')
        if isinstance(lev, dict):
            sig['leverage'] = lev.get('suggested')
        db_register_first_signal(sig, current_price=current_price)
    except Exception as e:
        logger.debug("_register_lifecycle_signal failed for %s: %s", r.get('symbol', '?'), e)


def _autoscan_recently_stopped(exch: str, sym: str, bias: str) -> bool:
    """
    FIX #SL-SUPPRESS — Return True if this exact setup (exchange + symbol +
    bias) has hit its stop-loss recently.

    Autoscan re-detects the same pairs every cycle. Once a call gets stopped
    out, re-surfacing it within a short window just spams subscribers and drags
    the win streak / win-rate down with the same loser being logged again and
    again. Suppressing recently stopped-out setups keeps the streak clean and
    lets a pair re-qualify only after conditions have had time to genuinely
    change (outside the suppression window).
    """
    try:
        cutoff = (datetime.now() - timedelta(hours=_AUTOSCAN_SL_SUPPRESS_H)).isoformat()
        conn = db_connect()
        c    = conn.cursor()
        c.execute(
            "SELECT 1 FROM signal_outcomes "
            "WHERE exchange=? AND symbol=? AND bias=? "
            "AND outcome='sl_hit' AND scan_time >= ? LIMIT 1",
            (exch, sym, bias, cutoff),
        )
        stopped = c.fetchone() is not None
        conn.close()
        return stopped
    except Exception as e:
        logger.warning("autoscan SL-suppress check failed for %s %s %s: %s", exch, sym, bias, e)
        return False


async def autoscandiag_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Unlisted diagnostic — explains exactly why autoscan is (or isn't) pushing."""
    _track(update)
    chat_id = update.effective_chat.id

    lines = ["\U0001F50E *Autoscan diagnostics*", ""]

    # 1. Subscribers
    n_subs = len(auto_scan_subscribers)
    mine = chat_id in auto_scan_subscribers
    my_tf = auto_scan_subscribers.get(chat_id)
    my_tf_label = (_tf_display(my_tf) if my_tf else "All timeframes") if mine else "\u2014"
    lines.append(f"\u2022 Subscribers: *{n_subs}*")
    lines.append(f"\u2022 You subscribed: *{'YES' if mine else 'NO'}* ({my_tf_label})")
    if not mine:
        lines.append("  \u21B3 Run /autoscan to subscribe.")

    # 2. Boot / grace status
    since_boot = (datetime.now() - _BOT_START_TS).total_seconds()
    in_grace = since_boot < _AUTOSCAN_STARTUP_GRACE_SECS
    lines.append(f"\u2022 Uptime: *{int(since_boot // 60)}m {int(since_boot % 60)}s*")
    if in_grace:
        remaining = int(_AUTOSCAN_STARTUP_GRACE_SECS - since_boot)
        lines.append(f"\u2022 Startup grace: *ACTIVE* \u2014 no pushes for ~{remaining // 60}m {remaining % 60}s more")
    else:
        lines.append("\u2022 Startup grace: *cleared* (pushes allowed)")

    # 3. Scan cache
    cache = getattr(state, "_scan_cache", None)
    passing = []
    have_cache = bool(cache and cache.get('results') is not None)
    if have_cache:
        age = (datetime.now() - cache['time']).total_seconds()
        results = cache['results'] or []
        passing = [r for r in results if passes_display_floor(r, AUTOSCAN_DISPLAY_CONF_MIN)]
        lines.append(f"\u2022 Last scan: *{len(results)}* signals, *{age / 60:.1f}m* old")
        lines.append(f"\u2022 Pass conf floor (\u2265{AUTOSCAN_DISPLAY_CONF_MIN:.0f}): *{len(passing)}*")
        top = sorted(results, key=lambda r: float(r.get('confidence') or 0), reverse=True)[:5]
        if top:
            lines.append("\u2022 Top by confidence:")
            for r in top:
                lines.append(
                    f"   \u2013 {r.get('symbol', '?')} {r.get('bias', '?')} "
                    f"{float(r.get('confidence') or 0):.0f}/10"
                )
    else:
        lines.append("\u2022 Last scan: *no cache yet* (scanner hasn't produced results)")

    # 4. Verdict
    lines.append("")
    if not mine:
        lines.append("\u27A1\uFE0F Not subscribed \u2014 that's why you get nothing. Run /autoscan.")
    elif in_grace:
        lines.append("\u27A1\uFE0F In startup grace \u2014 pushes resume after the grace window.")
    elif not have_cache:
        lines.append("\u27A1\uFE0F Scanner has no results yet \u2014 wait for the next scan, or check exchange access on the host.")
    elif len(passing) == 0:
        lines.append("\u27A1\uFE0F No signal currently clears the confidence floor \u2014 nothing to push right now.")
    else:
        lines.append("\u27A1\uFE0F Conditions look OK \u2014 qualifying signals should push on the next tick.")

    await update.message.reply_text("\n".join(lines), parse_mode='Markdown')


async def continuous_scan_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Runs every 10 minutes.
    Re-uses the scan cache when it is fresh (<15 min old) — no extra exchange calls.
    For any >=8/10 signal not recently sent, pushes a full signal card
    to each autoscan subscriber — filtered by their chosen TF preference.
    """
    if not auto_scan_subscribers:
        return

    results, from_cache = await get_scan_results(force=False)
    if not results:
        return

    # Phase 0: use AUTOSCAN_DISPLAY_CONF_MIN (env-tunable, >= SCAN_DISPLAY_CONF_MIN)
    high_conf = [r for r in results if passes_display_floor(r, AUTOSCAN_DISPLAY_CONF_MIN)]
    if not high_conf:
        return

    for r in high_conf:
        exch      = r.get('exchange', 'MEXC')
        sym       = r['symbol']
        bias      = r['bias']
        sig_tf    = r.get('timeframe', '4h')
        conf      = r['confidence']
        dedup_key = f"{exch}_{sym}_{bias}_{sig_tf}"

        # FIX #SL-SUPPRESS ��� don't keep re-surfacing a setup that already hit
        # its stop-loss recently. Re-pushing a freshly stopped-out pair spams
        # subscribers and pollutes the win streak with the same repeat loser.
        if _autoscan_recently_stopped(exch, sym, bias):
            logger.debug("AUTOSCAN SL-suppress: skipping %s %s %s (recent stop-out)", exch, sym, bias)
            continue

        # Keep the first-signal memory current so /pnl always anchors to the
        # original call even when the coin keeps running and gets re-detected.
        _register_lifecycle_signal(r)

        # FIX #RESTART-FLOOD — within the post-restart grace window we skip the
        # push so a restart doesn't re-blast every currently-live signal.
        #
        # FIX #DEAD-AUTOSCAN — this skip MUST happen BEFORE seeding the dedup
        # high-water mark below. Previously the mark was seeded first and then
        # the push was skipped, so every signal seen during the 15-min grace was
        # permanently marked "already sent" without ever being delivered. In a
        # stable market (or a bot that restarts periodically and re-enters the
        # grace each time) this made autoscan go completely silent. By skipping
        # before seeding, these signals are delivered once the grace elapses.
        if (datetime.now() - _BOT_START_TS).total_seconds() < _AUTOSCAN_STARTUP_GRACE_SECS:
            continue

        # CONFIDENCE-AWARE DEDUP ("always" subscribers):
        # The same call is not repeated consecutively unless its confidence has
        # risen above the last value we pushed. Genuinely newer/stronger calls
        # still flow through.
        prev = _autoscan_sent.get(dedup_key)
        prev_conf = float(prev.get('confidence', 0) or 0) if isinstance(prev, dict) else 0.0
        should_send_always, new_rec = _SIGLOGIC.autoscan_decide_send(
            prev, conf, reminder_secs=_AUTOSCAN_REMINDER_H * 3600)
        # A send where confidence did NOT rise above the prior push is a REMINDER
        # (the reminder window elapsed for a still-valid call). New/stronger
        # calls keep the normal AUTOSCAN heading.
        is_reminder = bool(prev) and should_send_always and float(conf or 0) <= prev_conf
        if should_send_always:
            _autoscan_sent[dedup_key] = new_rec
            _autoscan_prune()

        trade_type, tt_emoji = _autoscan_trade_type(r)
        conf       = r['confidence']
        conf_bar   = _conf_bar_emoji(conf)
        bias_emoji = "🟢" if bias == 'LONG' else "🔴"
        now_str    = datetime.now().strftime('%H:%M')

        # Entry edge used for lifecycle (T1/T2/T3/SL) milestone tracking.
        if bias == 'LONG':
            _track_entry = r.get('entry_high', r.get('price', 0))
        else:
            _track_entry = r.get('entry_low', r.get('price', 0))

        _title = "🔁 AUTOSCAN REMINDER" if is_reminder else f"{tt_emoji} AUTOSCAN"
        header = (
            f"{_title} — {trade_type.upper()}\n"
            f"{'━'*30}\n"
            f"{bias_emoji} {sym}  ·  {exch}  ·  {conf}/10 {conf_bar}\n"
            f"🕐 {now_str}  ·  Hold: {r.get('hold', '?')}\n"
        )

        link     = get_exchange_link(exch, sym)
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔗 Trade Now", url=link),
            InlineKeyboardButton("📊 Chart", callback_data=f"chart_inline|{sym}|4h"),
        ]])
        _, keyboard = cache_signal_card(r, 1, keyboard)

        for chat_id, tf_pref in list(auto_scan_subscribers.items()):
            if tf_pref:
                # TIMEFRAME SUBSCRIBER — only on the timeframe they picked, and
                # only on that timeframe's cadence so they're alerted when they
                # want (e.g. a 4h subscriber at most once per 4h per call).
                if sig_tf != tf_pref:
                    continue
                ukey = (chat_id, dedup_key)
                if not _SIGLOGIC.timeframe_due(_autoscan_user_last.get(ukey), tf_pref):
                    continue
            else:
                # "ALWAYS" SUBSCRIBER — gated only by the confidence dedup above.
                if not should_send_always:
                    continue
            try:
                await context.bot.send_message(chat_id=chat_id, text=header)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=format_signal_primary(r, 1),
                    reply_markup=keyboard,
                )
                _safemode_store_signals(chat_id, [r])
                _autoscan_user_last[(chat_id, dedup_key)] = datetime.now()
                # Track this delivered signal so the lifecycle monitor can alert
                # this subscriber when T1/T2/T3 or SL is reached.
                db_lifecycle_track(
                    dedup_key, exch, sym, bias, sig_tf,
                    _track_entry, r.get('stop_loss'), r.get('t1'), r.get('t2'), r.get('t3'),
                    chat_id)
                await asyncio.sleep(0.15)
            except Exception as e:
                logger.warning("Autoscan push failed for %s: %s", chat_id, e)

        await asyncio.sleep(0.5)


async def global_error_handler(update, context):
    """Catch-all error handler so one failed update/job never crashes the bot.

    Without this, an unhandled exception can propagate out of polling and trip
    the crash-recovery restart loop (which drops pending updates and makes
    commands appear to "pause"). Transient Telegram/network errors are common
    and harmless — we log and swallow them; anything unexpected is logged with a
    full traceback but contained so the bot keeps serving other users.
    """
    err = getattr(context, 'error', None)
    try:
        from telegram.error import Conflict, NetworkError, TimedOut, RetryAfter
    except Exception:
        Conflict = NetworkError = TimedOut = RetryAfter = tuple()
    try:
        if isinstance(err, Conflict):
            logger.error("Telegram Conflict (another getUpdates instance?) — %s", err)
            return
        if isinstance(err, RetryAfter):
            logger.warning("Rate limited by Telegram; retry after %ss",
                           getattr(err, 'retry_after', '?'))
            return
        if isinstance(err, (NetworkError, TimedOut)):
            logger.warning("Transient network error (ignored): %s", err)
            return
        logger.exception("Unhandled error while processing an update: %s", err)
    except Exception as _e:
        # The error handler itself must never raise.
        logger.error("error handler failed: %s", _e)


def _maintenance_apply(actives, prices, now):
    """Blocking tail of signal_maintenance_job — pure logic + sqlite writes.

    Runs inside SCAN_EXECUTOR (a worker thread) so the asyncio event loop is
    NEVER blocked while the lifecycle sweep runs. Live prices are prefetched by
    the async wrapper and passed in via `prices` (parallel to `actives`).

    For each tracked signal:
      0. STOP-LOSS DETECTION — counter-direction hit of the bot's SL pins a
         terminal 'sl_hit' status (exempt from eviction so /pnl can show the
         realized loss card).
      1. Advance the favourable peak + last-motion timestamp.
      2. PEAK-REVERSAL EVICTION — a profitable trade reversing >=20% from its
         peak is deleted (eviction stamp kept for the 'not scanned recently'
         guard).
      3. DORMANCY CLEAR — no motion for 15 minutes clears the signal.
    """
    for rec, price in zip(actives, prices):
        try:
            exch = rec.get('exchange', '')
            sym  = rec.get('symbol', '')
            bias = rec.get('bias', '')
            entry = float(rec.get('entry') or 0)
            peak  = float(rec.get('peak_price') or entry or 0)
            price = float(price) if price else 0.0

            # 0) STOP-LOSS DETECTION
            sl = float(rec.get('stop_loss') or 0)
            if price > 0 and sl > 0 and str(rec.get('status', '')).lower() != 'sl_hit':
                hit = (price <= sl) if str(bias).upper() == 'LONG' else (price >= sl)
                if hit:
                    db_mark_signal_status(exch, sym, 'sl_hit')
                    rec['status'] = 'sl_hit'
                    logger.info("Lifecycle: %s %s hit stop-loss (loss card pinned)", exch, sym)

            # 1) Advance favourable peak + motion timestamp.
            if str(rec.get('status', '')).lower() == 'sl_hit':
                continue
            if price > 0 and entry > 0:
                if str(bias).upper() == 'LONG':
                    new_peak = max(peak, price) if peak else price
                else:
                    new_peak = min(peak, price) if peak else price
                if new_peak != peak:
                    db_update_signal_peak(exch, sym, new_peak, now)
                    rec['peak_price'] = new_peak
                    rec['last_motion_time'] = now.isoformat()
                    peak = new_peak

            # 2) 20% peak-reversal eviction (only meaningful once in profit).
            if price > 0 and entry > 0 and _SIGLOGIC.should_evict_peak_reversal(
                bias, entry, peak, price
            ):
                db_evict_signal(exch, sym, now.isoformat(), reason='peak_reversal_20pct')
                logger.info("Lifecycle: evicted %s %s (20%% reversal from peak)", exch, sym)
                continue

            # 3) Dormancy clear after 15 minutes of no motion.
            if _SIGLOGIC.should_clear_dormant(rec, now=now):
                db_evict_signal(exch, sym, now.isoformat(), reason='dormant')
                logger.info("Lifecycle: cleared dormant %s %s (no motion 15m)", exch, sym)
                continue
        except Exception as e:
            logger.debug("signal_maintenance_job error for %s: %s", rec.get('symbol', '?'), e)


async def signal_maintenance_job(context: ContextTypes.DEFAULT_TYPE):
    """Lifecycle maintenance for the first-signal memory (runs every ~5 min).

    PERFORMANCE: this used to call the blocking _get_live_price() sequentially
    for every active signal *on the event loop*, which froze all command
    handling for the duration of the sweep (REST fallbacks can each take several
    seconds). It now (a) loads signals off-loop, (b) prefetches every live price
    concurrently through SCAN_EXECUTOR, and (c) runs the DB/logic tail in a
    worker thread — so the event loop stays responsive throughout.
    """
    loop = asyncio.get_running_loop()
    try:
        actives = await loop.run_in_executor(SCAN_EXECUTOR, db_all_active_signals)
    except Exception as e:
        logger.warning("signal_maintenance_job: load failed: %s", e)
        return
    if not actives:
        return

    # Prefetch all live prices concurrently, off the event loop. WS-cache hits
    # return instantly; only genuine misses touch REST, and never on the loop.
    sem = asyncio.Semaphore(12)

    async def _price(rec):
        async with sem:
            try:
                p = await loop.run_in_executor(
                    SCAN_EXECUTOR, _get_live_price,
                    rec.get('symbol', ''), rec.get('exchange', ''))
                return float(p) if p else 0.0
            except Exception:
                return 0.0

    try:
        prices = await asyncio.gather(*[_price(r) for r in actives])
    except Exception as e:
        logger.warning("signal_maintenance_job: price prefetch failed: %s", e)
        return

    now = datetime.now()
    try:
        await loop.run_in_executor(SCAN_EXECUTOR, _maintenance_apply, actives, prices, now)
    except Exception as e:
        logger.warning("signal_maintenance_job: apply failed: %s", e)


def _lifecycle_alert_text(tag, rec, price):
    """Build the milestone / stop-loss alert card for a tracked autoscan signal."""
    sym  = rec.get('symbol', '?')
    bias = str(rec.get('bias', '')).upper()
    bias_emoji = "🟢" if bias == 'LONG' else "🔴"
    try:
        price = float(price or 0)
    except Exception:
        price = 0.0
    if tag == 'sl':
        sl = float(rec.get('stop_loss') or 0)
        return (
            f"🛑 STOP-LOSS HIT — {sym}\n"
            f"{'━'*28}\n"
            f"{bias_emoji} {bias}  ·  SL {sl:g}\n"
            f"📍 Price: {price:g}\n"
            f"This setup is now closed."
        )
    labels = {'t1': "Target 1", 't2': "Target 2", 't3': "Target 3"}
    lvl = float(rec.get(tag) or 0)
    tail = "  —  final target reached, setup closed." if tag == 't3' else ""
    return (
        f"🎯 {labels.get(tag, tag.upper())} REACHED — {sym}\n"
        f"{'━'*28}\n"
        f"{bias_emoji} {bias}  ·  {labels.get(tag, tag)} {lvl:g}\n"
        f"📍 Price: {price:g}{tail}"
    )


def _lifecycle_alerts_apply(rows, prices, now):
    """Blocking tail of autoscan_lifecycle_job — detects newly-reached
    milestones, marks them in the DB, closes finished setups, and returns a
    list of (chat_id, text) alerts for the async wrapper to send.

    Runs in SCAN_EXECUTOR so the event loop is never blocked by DB writes.

    Dedup guarantee: DB is written BEFORE output is built so a crash between
    mark and send causes an omission (acceptable) rather than a duplicate.
    An in-memory _fired set guards against the same key firing twice within a
    single job tick even if a DB write fails silently.
    """
    out = []
    _fired: set = set()   # (key, tag) pairs already processed this tick

    for rec, price in zip(rows, prices):
        try:
            key = rec.get('key')
            already = {
                't1': bool(rec.get('t1_alerted')),
                't2': bool(rec.get('t2_alerted')),
                't3': bool(rec.get('t3_alerted')),
                'sl': bool(rec.get('sl_alerted')),
            }
            hits = _SIGLOGIC.lifecycle_milestones(
                rec.get('bias'), price, rec.get('stop_loss'),
                rec.get('t1'), rec.get('t2'), rec.get('t3'), already)
            if not hits:
                continue
            try:
                recips = _json.loads(rec.get('recipients') or '[]')
            except Exception:
                recips = []
            for tag in hits:
                fire_key = (key, tag)
                if fire_key in _fired:
                    continue
                # ── Write DB FIRST; only send if the mark succeeds ──────────
                db_lifecycle_mark(key, f"{tag}_alerted")
                _fired.add(fire_key)
                text = _lifecycle_alert_text(tag, rec, price)
                for chat_id in recips:
                    out.append((chat_id, text))
            if 'sl' in hits or 't3' in hits:
                db_lifecycle_close(key)
        except Exception as e:
            logger.debug("lifecycle alert apply error for %s: %s", rec.get('symbol', '?'), e)
    return out


async def autoscan_lifecycle_job(context: ContextTypes.DEFAULT_TYPE):
    """Every ~5 min: for each tracked autoscan signal, fetch the live price and
    alert its recipients when a profit target (T1/T2/T3) is reached or the
    stop-loss is hit. Mirrors signal_maintenance_job's off-loop price-prefetch
    pattern so the event loop stays responsive throughout.
    """
    loop = asyncio.get_running_loop()
    try:
        rows = await loop.run_in_executor(SCAN_EXECUTOR, db_lifecycle_active)
    except Exception as e:
        logger.warning("autoscan_lifecycle_job: load failed: %s", e)
        return
    if not rows:
        return

    sem = asyncio.Semaphore(12)

    async def _price(rec):
        async with sem:
            try:
                p = await loop.run_in_executor(
                    SCAN_EXECUTOR, _get_live_price,
                    rec.get('symbol', ''), rec.get('exchange', ''))
                return float(p) if p else 0.0
            except Exception:
                return 0.0

    try:
        prices = await asyncio.gather(*[_price(r) for r in rows])
    except Exception as e:
        logger.warning("autoscan_lifecycle_job: price prefetch failed: %s", e)
        return

    now = datetime.now()
    try:
        alerts = await loop.run_in_executor(SCAN_EXECUTOR, _lifecycle_alerts_apply, rows, prices, now)
    except Exception as e:
        logger.warning("autoscan_lifecycle_job: apply failed: %s", e)
        return

    for chat_id, text in alerts:
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning("lifecycle alert send failed for %s: %s", chat_id, e)

    try:
        await loop.run_in_executor(SCAN_EXECUTOR, db_lifecycle_prune, 72)
    except Exception:
        pass


async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    """
    4-hour periodic digest — housekeeping only.
    Forces a fresh full scan, fires coin alerts, watchlist checks, and broadcast posts.
    Signal delivery to autoscan subscribers is handled by continuous_scan_job.
    """
    logger.info("Auto-scan (4h digest) triggered")
    results, from_cache = await get_scan_results(force=True)

    if not results:
        return

    await notify_alerts(results, context.bot)
    await watchlist_check_job(context)
    await post_broadcast(results, context.bot)


async def mid_scan_job(context: ContextTypes.DEFAULT_TYPE):
    """
    FIX #MID-JOB — Mid-tier universe rotation (ranks 51-200).

    run_mid_scan() already existed and works, but was NEVER scheduled, so the
    auto-push pipeline only ever saw the top-50 (BTC/ETH/SOL/...) and the same
    names repeated endlessly.  This job runs every 4h (offset 2h from the full
    scan) and MERGES its mid-tier signals into the live scan cache.  The
    existing continuous_scan_job then delivers them through the normal
    confidence gate + dedup + per-subscriber TF filter — the push/dedup logic
    itself is left completely untouched.
    """

    if not auto_scan_subscribers:
        return

    loop = asyncio.get_event_loop()
    try:
        mid = await loop.run_in_executor(SCAN_EXECUTOR, run_mid_scan)
    except Exception as e:
        logger.warning("mid_scan_job failed: %s", e)
        return
    if not mid:
        return

    # Start from the current full-scan result set (cached or fresh), then append
    # mid-tier signals not already present, keyed by exchange+symbol+bias.
    base_results, _ = await get_scan_results(force=False)
    merged = list(base_results) if base_results else []
    seen   = {(r.get('exchange'), r.get('symbol'), r.get('bias')) for r in merged}

    added = 0
    for r in mid:
        key = (r.get('exchange'), r.get('symbol'), r.get('bias'))
        if key not in seen:
            merged.append(r)
            seen.add(key)
            added += 1

    # Refresh the cache so the next continuous_scan_job tick sees the mid-tier
    # signals and pushes any that clear the confidence gate.
    # FIX C2 — write under the cache lock so this can't race with
    # get_scan_results()/continuous_scan_job readers (torn/partial cache).
    async with _scan_cache_lock:
        state._scan_cache = {'results': merged, 'time': datetime.now()}
    logger.info("mid_scan_job merged %d new mid-tier signals into scan cache (%d total)",
                added, len(merged))


async def autoscan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    chat_id = update.effective_chat.id

    # ── Direct timeframe argument — no buttons needed ──
    # e.g. /autoscan 4h | /autoscan 15m | /autoscan always | /autoscan off
    if context.args:
        arg = str(context.args[0]).strip().lower()

        if arg in ('off', 'stop', 'disable', 'none'):
            auto_scan_subscribers.pop(chat_id, None)
            db_autoscan_remove(chat_id)
            _autoscan_awaiting_tf.discard(chat_id)
            await update.message.reply_text(
                "🔕 Auto-scan *OFF*. Use /autoscan to turn back on.",
                parse_mode='Markdown',
            )
            return

        if arg in ('always', 'all', 'any', 'on'):
            tf_pref = None
        else:
            tf_pref = _parse_tf_arg(arg)
            if not tf_pref:
                await update.message.reply_text(
                    f"❓ `{arg}` isn't a recognised timeframe.\n"
                    f"Try: `/autoscan 15m`, `/autoscan 1h`, `/autoscan 4h`, "
                    f"`/autoscan 1d`, `/autoscan always`, or `/autoscan off`.",
                    parse_mode='Markdown',
                )
                return

        tf_label = _tf_display(tf_pref) if tf_pref else "All timeframes"
        auto_scan_subscribers[chat_id] = tf_pref
        db_autoscan_set(chat_id, tf_pref)
        _autoscan_awaiting_tf.discard(chat_id)
        await update.message.reply_text(
            f"✅ *Auto-scan ON!*  Receiving *{tf_label}* signals.\n\n"
            f"Change anytime with `/autoscan <tf>` (e.g. `/autoscan 1h`) "
            f"or stop with `/autoscan off`.",
            parse_mode='Markdown',
        )
        return

    if chat_id in auto_scan_subscribers:
        # Already ON — show status + options
        tf_pref    = auto_scan_subscribers[chat_id]
        tf_label   = _tf_display(tf_pref) if tf_pref else "All timeframes"
        keyboard   = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔄 Change TF",  callback_data="autoscan_tf|change"),
            InlineKeyboardButton("🔕 Turn Off",   callback_data="autoscan_tf|off"),
        ]])
        await update.message.reply_text(
            f"📡 Auto-scan is *ON* — receiving signals from: *{tf_label}*\n\n"
            f"Change your timeframe filter or turn off below.",
            parse_mode='Markdown',
            reply_markup=keyboard,
        )
    else:
        # OFF — show TF selection menu
        _autoscan_awaiting_tf.add(chat_id)
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔄 Always",  callback_data="autoscan_tf|always"),
            InlineKeyboardButton("⏱ 15M",      callback_data="autoscan_tf|15m"),
            InlineKeyboardButton("📊 4H",       callback_data="autoscan_tf|4h"),
            InlineKeyboardButton("📅 1D",       callback_data="autoscan_tf|1d"),
        ]])
        await update.message.reply_text(
            "📡 *Auto-scan* — Choose your signal timeframe:\n\n"
            "• *Always* — receive signals from any timeframe\n"
            "• *15M* — 15-minute signals only (scalp/intraday)\n"
            "• *4H* — 4-hour signals only (swing trades)\n"
            "• *1D* — daily signals only (position trades)\n\n"
            "💡 *No buttons needed* — just type it directly, e.g. "
            "`/autoscan 4h`, `/autoscan 15m`, `/autoscan always`, or `/autoscan off`.",
            parse_mode='Markdown',
            reply_markup=keyboard,
        )


async def autoscan_tf_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles autoscan_tf|<choice> button presses."""
    query   = update.callback_query
    chat_id = query.message.chat_id
    data    = query.data  # e.g. "autoscan_tf|4h"
    choice  = data.split('|')[1]

    await query.answer()

    if choice == 'off':
        auto_scan_subscribers.pop(chat_id, None)
        db_autoscan_remove(chat_id)
        _autoscan_awaiting_tf.discard(chat_id)
        await query.edit_message_text("🔕 Auto-scan *OFF*. Use /autoscan to turn back on.", parse_mode='Markdown')
        return

    if choice == 'change':
        # Re-show the TF menu
        _autoscan_awaiting_tf.add(chat_id)
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔄 Always",  callback_data="autoscan_tf|always"),
            InlineKeyboardButton("⏱ 15M",      callback_data="autoscan_tf|15m"),
            InlineKeyboardButton("📊 4H",       callback_data="autoscan_tf|4h"),
            InlineKeyboardButton("📅 1D",       callback_data="autoscan_tf|1d"),
        ]])
        await query.edit_message_text(
            "📡 *Auto-scan* — Choose your new signal timeframe:\n\n"
            "• *Always* — any timeframe\n"
            "• *15M* — scalp/intraday signals\n"
            "• *4H* — swing trade signals\n"
            "• *1D* — position trade signals\n\n"
            "Or *reply with a custom timeframe* — e.g. `1h`, `30m`, `1w`",
            parse_mode='Markdown',
            reply_markup=keyboard,
        )
        return

    # Activate subscription with chosen TF
    tf_pref  = None if choice == 'always' else choice
    tf_label = _tf_display(tf_pref) if tf_pref else "All timeframes"
    auto_scan_subscribers[chat_id] = tf_pref
    db_autoscan_set(chat_id, tf_pref)
    _autoscan_awaiting_tf.discard(chat_id)

    await query.edit_message_text(
        f"✅ *Auto-scan ON!*\n\n"
        f"📡 Receiving: *{tf_label}* signals\n\n"
        f"Each alert shows the trade type:\n"
        f"  ⚡ Scalp  (hold ≤4h)\n"
        f"  🕐 Intraday  (4–12h)\n"
        f"  📈 Swing Trade  (12–48h)\n"
        f"  🏦 Position Trade  (48h+)\n\n"
        f"Same signal won't repeat for 4 hours.\n"
        f"Use /autoscan to change TF or turn off.",
        parse_mode='Markdown',
    )


async def autoscan_custom_tf_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Intercepts plain-text messages from users who are in the TF-selection flow.
    Returns True if the message was consumed (custom TF applied), False otherwise.
    """
    chat_id = update.effective_chat.id
    if chat_id not in _autoscan_awaiting_tf:
        return False

    raw = (update.message.text or '').strip()
    tf  = _parse_tf_arg(raw)

    if not tf:
        await update.message.reply_text(
            f"❓ `{raw}` isn't a recognised timeframe.\n"
            f"Try something like `5m`, `1h`, `4h`, `1d`, `1w`.\n"
            f"Or pick from the buttons above.",
            parse_mode='Markdown',
        )
        return True  # consumed — don't pass to other handlers

    auto_scan_subscribers[chat_id] = tf
    db_autoscan_set(chat_id, tf)
    _autoscan_awaiting_tf.discard(chat_id)

    await update.message.reply_text(
        f"✅ *Auto-scan ON!*  Receiving *{_tf_display(tf)}* signals.\n\n"
        f"Use /autoscan to change or turn off.",
        parse_mode='Markdown',
    )
    return True


# ─────────────────────────────────────────────
# FORMAT SIGNAL
# IMPROVEMENT #8 — deeplinks to exchange added
# ──────��──────────────────────────────────────
def get_exchange_link(exchange, symbol):
    clean = symbol.replace('_USDT', 'USDT').replace('/', '')
    if exchange == 'BYBIT':
        return f"https://www.bybit.com/trade/usdt/{clean}"
    elif exchange == 'BINANCE':
        return f"https://www.binance.com/en/futures/{clean}"
    else:  # MEXC
        return f"https://futures.mexc.com/exchange/{clean.replace('USDT','_USDT')}"

def _conf_bar_emoji(conf):
    """
    Build a 5-square emoji confidence bar.
    Each square = 2 points of the 10-point scale.
      full  (≥2 pts remaining) → 🟩
      half  (1 pt remaining)   → 🟨
      empty                    → ⬜
    """
    squares = []
    remaining = conf
    for _ in range(5):
        if remaining >= 2:
            squares.append('🟩')
            remaining -= 2
        elif remaining == 1:
            squares.append('🟨')
            remaining = 0
        else:
            squares.append('⬜')
    return ''.join(squares)


def _panel_type(tf_label):
    """Map TF label to panel type string shown in the card header."""
    return {'15M': 'SCALP', '1H': 'SCALP', '4H': 'SIGNAL', '1D': 'SWING'}.get(tf_label, 'SIGNAL')




def _build_signal_common(r):
    """Pre-compute shared fields used by both primary and details cards."""
    bias_emoji = "🟢" if r['bias'] == "LONG" else "🔴"
    conf       = r['confidence']
    conf_bar   = f"{_conf_bar_emoji(conf)} {conf}/10"
    lev        = r.get('leverage')
    exchange   = r.get('exchange', '')
    tf_label   = r.get('signal_tf_label', '4H')
    symbol     = r['symbol']
    scan_price = r['price']

    live_price = 0
    try:
        live_price = _get_live_price(symbol, exchange)
    except Exception as e:
        logger.warning("signal refresh: live price lookup failed for %s/%s: %s", symbol, exchange, e)
    display_price = live_price if live_price > 0 else scan_price

    scan_time = r.get('scan_time')
    if scan_time and isinstance(scan_time, datetime):
        age_min = int((datetime.now() - scan_time).total_seconds() / 60)
        age_str = f"{age_min}m ago" if age_min < 60 else f"{age_min // 60}h {age_min % 60}m ago"
    else:
        age_str = "unknown"

    entry_low  = r['entry_low']
    entry_high = r['entry_high']
    bias       = r['bias']

    if entry_low <= display_price <= entry_high:
        zone_status = "✅ IN ZONE — price is within entry range"
    elif bias == 'LONG' and display_price < entry_low:
        gap_pct = ((entry_low - display_price) / display_price) * 100
        zone_status = f"📉 BELOW ZONE by {gap_pct:.2f}% — overshooting pullback, wait for bounce"
    elif bias == 'LONG' and display_price > entry_high:
        gap_pct = ((display_price - entry_high) / entry_high) * 100
        zone_status = f"📈 ABOVE ZONE by {gap_pct:.2f}% — price ran, wait for pullback to zone"
    elif bias == 'SHORT' and display_price > entry_high:
        gap_pct = ((display_price - entry_high) / entry_high) * 100
        zone_status = f"📈 ABOVE ZONE by {gap_pct:.2f}% — overshooting bounce, wait for rejection"
    elif bias == 'SHORT' and display_price < entry_low:
        gap_pct = ((entry_low - display_price) / entry_low) * 100
        zone_status = f"📉 BELOW ZONE by {gap_pct:.2f}% — price dumped past zone, wait for bounce into zone"
    else:
        zone_status = "⚪ ZONE STATUS UNKNOWN"

    return dict(
        bias_emoji=bias_emoji, conf=conf, conf_bar=conf_bar, lev=lev,
        exchange=exchange, tf_label=tf_label, symbol=symbol,
        scan_price=scan_price, live_price=live_price,
        display_price=display_price, age_str=age_str,
        zone_status=zone_status,
        panel_type=_panel_type(tf_label),
        rr=_rr_ratio(r),
    )


def _db_symbol_track_record(symbol: str, min_signals: int = 5) -> dict:
    """
    FIX #TRACK — Per-symbol win rate from resolved signal_outcomes.
    Returns {'wins': int, 'total': int, 'win_rate': float, 'badge': str}
    or empty dict if not enough history.
    Cached per symbol for 10 minutes — called inside format_signal so must be fast.
    """
    cache_key = f"track_{symbol}"
    cached = state._signal_card_cache.get(cache_key)
    if cached and (datetime.now() - cached.get('_ts', datetime.min)).total_seconds() < 600:
        return cached

    try:
        conn = db_connect()
        row  = conn.execute(
            "SELECT "
            "  COUNT(*) as total, "
            "  SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins "
            "FROM signal_outcomes WHERE symbol=? AND outcome IN ('win','loss')",
            (symbol,)
        ).fetchone()
        conn.close()
        if not row or (row['total'] or 0) < min_signals:
            return {}
        total    = int(row['total'])
        wins     = int(row['wins'] or 0)
        win_rate = wins / total
        if win_rate >= 0.60:
            badge = f"🟢 {wins}/{total} ({win_rate:.0%} hist WR)"
        elif win_rate >= 0.45:
            badge = f"🟡 {wins}/{total} ({win_rate:.0%} hist WR)"
        else:
            badge = f"🔴 {wins}/{total} ({win_rate:.0%} hist WR)"
        result = {'wins': wins, 'total': total, 'win_rate': win_rate, 'badge': badge, '_ts': datetime.now()}
        state._signal_card_cache[cache_key] = result
        return result
    except Exception:
        return {}


def format_signal_primary(r, rank):
    """
    Req #15 — Unified signal panel used everywhere:
      /scan number tap, /cscan, /scalp, auto-signal detection.

    Layout:
      ━━━━━��━━━━━━━━━━━━━━━━━━━━━━
        ⚡️ SWING | SIGNAL
        CONFIDENCE: 🟩🟩🟩🟩🟩 10/10
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      📊 Token: XMRUSDT
      🟢 Direction: LONG
      ��� Entry: $x – $x
      ��️ Leverage: 8x
      📐 R:R: 1:2.3
      🎯 TP1: $x  (+X% profit) 💰
      🎯 TP2: $x
      🎯 TP3: $x
      🛑 Stop Loss: $x

    Buttons: [🔄 Refresh]  [��� Details]
    Details button edits this same panel in-place (no new message).
    """
    c          = _build_signal_common(r)
    lev        = c['lev']
    conf       = c['conf']
    exchange   = c['exchange']
    panel_type = c['panel_type']
    rr         = c['rr']

    # TP profit % lines
    price = r['price'] if r['price'] > 0 else 1
    def tp_pct(tp_val, bias):
        raw = (tp_val - price) / price * 100
        return raw if bias == 'LONG' else -raw

    t1_pct = tp_pct(r['t1'], r['bias'])
    t2_pct = tp_pct(r['t2'], r['bias'])
    t3_pct = tp_pct(r['t3'], r['bias'])
    lev_mult = lev['suggested'] if lev else 1

    lev_str = f"{lev_mult}x" if lev else "1x"
    rr_str  = f"1:{rr:.1f}" if rr is not None else "N/A"

    # counter-trend warning prefix
    ct_line = "⚠️ COUNTER-TREND — Higher risk\n" if r.get('counter_trend') else ""

    # ── TP/SL hit ticks ─────────────────────────────────────────────────────
    # Once price reaches a target (or the stop), tick that line. We track the
    # high-water extremes of the live price since this card first rendered so a
    # tick STAYS even if price later retraces. `r` is the cached card object,
    # so these high-water marks persist across the 30s auto-refresh.
    _bias = r['bias']
    _lp = c.get('live_price') or 0
    if _lp > 0:
        _hw_hi = r.get('_hw_high'); _hw_lo = r.get('_hw_low')
        r['_hw_high'] = _lp if _hw_hi is None else max(_hw_hi, _lp)
        r['_hw_low']  = _lp if _hw_lo is None else min(_hw_lo, _lp)
    _hw_high = r.get('_hw_high'); _hw_low = r.get('_hw_low')

    def _tp_hit(level):
        if level is None or _hw_high is None or _hw_low is None:
            return False
        try:
            level = float(level)
        except Exception:
            return False
        return (_hw_high >= level) if _bias == 'LONG' else (_hw_low <= level)

    def _sl_hit(level):
        if level is None or _hw_high is None or _hw_low is None:
            return False
        try:
            level = float(level)
        except Exception:
            return False
        return (_hw_low <= level) if _bias == 'LONG' else (_hw_high >= level)

    _t1_tick = "  ✅ HIT" if _tp_hit(r.get('t1')) else ""
    _t2_tick = "  ✅ HIT" if _tp_hit(r.get('t2')) else ""
    _t3_tick = "  ✅ HIT" if _tp_hit(r.get('t3')) else ""
    _sl_tick = "  ❌ SL HIT" if _sl_hit(r.get('stop_loss')) else ""

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  ⚡️ {panel_type} | SIGNAL",
        f"  CONFIDENCE: {c['conf_bar']}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    if ct_line:
        lines.append(ct_line)

    # NEW LISTING warning — surfaces when analyze_symbol_new_listing fired
    if r.get('new_listing'):
        lines.append(
            "🆕 NEW LISTING — Limited history. Analysis uses short-TF data only.\n"
            "   Signals are less reliable. Use reduced position size.\n"
        )

    lines += [
        f"📊 Token: {r['symbol']}",
        f"{c['bias_emoji']} Direction: {r['bias']}",
        f"💰 Entry: ${r['entry_low']:.4f} - ${r['entry_high']:.4f}",
        f"⚡️ Leverage: {lev_str}",
        f"📐 R:R: {rr_str}",
        "",
        f"🎯 TP1: ${r['t1']:.4f}  (+{t1_pct:.1f}% profit) 💰{_t1_tick}",
        f"🎯 TP2: ${r['t2']:.4f}  (+{t2_pct:.1f}%){_t2_tick}",
        f"🎯 TP3: ${r['t3']:.4f}  (+{t3_pct:.1f}%){_t3_tick}",
        f"🛑 Stop Loss: ${r['stop_loss']:.4f}{_sl_tick}",
    ]

    # ML line if available
    consensus_score   = r.get('consensus_score')
    consensus_verdict = r.get('consensus_verdict', '')
    ml_score          = r.get('ml_score')
    rf_score          = r.get('rf_score')
    if consensus_score is not None:
        pct = int(round(consensus_score * 100))
        _adj = r.get('ml_conf_adjust', 0)
        _adj_txt = (f"  ({'+' if _adj > 0 else ''}{_adj} conf)" if _adj else "")
        _cal = ""
        try:
            if (rf_model_meta() or {}).get('calibrated'):
                _cal = " ✓cal"
        except Exception:
            pass
        lines.append(f"🤖 ML: {pct}% win prob{_cal} — {consensus_verdict}{_adj_txt}")
    elif ml_score is not None and rf_score is not None:
        lines.append(f"🤖 ML: XGB {int(round(ml_score*100))}%  |  RF {int(round(rf_score*100))}%")
    elif ml_score is not None:
        lines.append(f"🤖 ML: {int(round(ml_score*100))}% (XGBoost)")
    elif rf_score is not None:
        lines.append(f"🤖 ML: {int(round(rf_score*100))}% (RF)")

    # Per-symbol track record badge (FIX #TRACK)
    _tr = _db_symbol_track_record(r.get('symbol', ''))
    if _tr:
        lines.append(f"📈 Track: {_tr['badge']}")

    # FIX #EXPLAIN — Primary driver explanation
    # Surface the single most important reason the signal fired,
    # so the user immediately understands what they're trading.
    _reasons = r.get('reasons', [])
    _driver_priority = [
        'divergence', 'squeeze breakout', 'MACD confirmed', 'extreme',
        'funding', 'CLV', 'Stochastic', 'MACD', 'EMA stack', 'RSI'
    ]
    _primary_driver = None
    for _kw in _driver_priority:
        for _r in _reasons:
            if _kw.lower() in _r.lower():
                _primary_driver = _r
                break
        if _primary_driver:
            break
    if not _primary_driver and _reasons:
        _primary_driver = _reasons[0]
    if _primary_driver:
        # Trim long strings
        _pd_display = _primary_driver[:90] + '…' if len(_primary_driver) > 90 else _primary_driver
        lines.append(f"🔑 Key driver: {_pd_display}")

    # FIX #AUTOREFRESH-VISIBILITY — live "updated" stamp. Makes each 30s tick
    # visible to the user AND guarantees the rendered text changes every
    # refresh, so Telegram never suppresses the edit as "message not modified".
    lines.append("")
    lines.append(f"🔄 Live • updated {datetime.now().strftime('%H:%M:%S')} (auto every {AUTO_REFRESH_SECS}s)")

    return "\n".join(lines)


def format_signal_details(r, rank):
    """Details card — regime, indicators, conviction reasons, duration analysis, portfolio."""
    c        = _build_signal_common(r)
    exchange = c['exchange']
    dur_reasons = r.get('dur_reasons', [])

    # BTC regime block
    regime = r.get('btc_regime', 'NEUTRAL')
    regime_emoji = {
        'STRONG_BULL': '🟢🟢', 'BULL': '🟢',
        'NEUTRAL': '⚪',
        'BEAR': '🔴', 'STRONG_BEAR': '🔴🔴'
    }.get(regime, '⚪')
    is_aligned   = (regime in ('STRONG_BULL', 'BULL') and r['bias'] == 'LONG') or \
                   (regime in ('STRONG_BEAR', 'BEAR') and r['bias'] == 'SHORT')
    is_divergent = (regime in ('STRONG_BULL', 'BULL') and r['bias'] == 'SHORT') or \
                   (regime in ('STRONG_BEAR', 'BEAR') and r['bias'] == 'LONG')
    regime_line  = (f"{regime_emoji} BTC REGIME: {regime}"
                    + (" — regime-aligned ✅" if is_aligned
                       else " — regime-divergent ⚠️" if is_divergent
                       else " — choppy/transitioning ⚠️"))

    lines = [
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"#{rank}  {exchange} | {r['symbol']}  [{c['tf_label']}] — DETAILS",
        f"━━━━━━━━━━━━━━━━━━━━━━���━━━━━━━",
        f"",
        regime_line,
        f"",
        f"📊 INDICATORS",
        f"   RSI 4H:    {r['rsi4']:.1f}",
        f"   RSI Daily: {r['rsi_d']:.1f}  (closed candle)",
        f"   Stoch K:   {r['stoch_k']:.1f}",
        f"   Funding:   {r['funding']:.4f}%",
        f"   ATR:       ${r['atr']:.6f}",
    ]

    # 24h volume line
    vol_24h = r.get('vol_24h_usdt', 0)
    vol_str = (f"${vol_24h/1_000_000:.1f}M" if vol_24h >= 1_000_000
               else f"${vol_24h/1_000:.0f}K" if vol_24h > 0 else "N/A")
    lines.append(f"   24h Vol:   {vol_str}")

    lines += [f"", f"✅ CONVICTION REASONS:"]
    for reason in r['reasons']:
        lines.append(f"   • {reason}")

    # CONVICTION LAYER — CVD / OI / VWAP (if available)
    conv = r.get('conviction', {})
    conv_lines = conv.get('display_lines', []) if isinstance(conv, dict) else []
    if conv_lines:
        lines += [f"", f"📊 CONVICTION LAYER:"]
        for dl in conv_lines:
            lines.append(f"   {dl}")
        vwap_data = conv.get('vwap', {})
        if isinstance(vwap_data, dict) and vwap_data.get('vwap', 0) > 0:
            lines.append(f"   VWAP ref: ${vwap_data['vwap']:.4f}")

    # FIX #SESSION — session context at signal time
    sess_note = r.get('session_note', '')
    if sess_note:
        lines += [f"", f"🕐 SESSION CONTEXT:"]
        lines.append(f"   {sess_note}")
        if r.get('session_warning'):
            lines.append(f"   ⚠️ Entry caution — session prone to fakeouts / manipulation wicks")

    # FIX #FIB — Fibonacci levels display
    fib_levels  = r.get('fib_levels', [])
    fib_conf    = r.get('fib_confluence', 0)
    if fib_levels:
        sorted_fibs = sorted(fib_levels)
        ratio_names = ['0.236', '0.382', '0.500', '0.618', '0.786']
        lines += [f"", f"📐 FIBONACCI RETRACEMENT LEVELS:"]
        if fib_conf >= 2:
            lines.append(f"   🎯 Tight confluence detected — pivot S/R aligns with Fib zone")
        elif fib_conf == 1:
            lines.append(f"   ✅ Fib confluence — pivot S/R near a Fibonacci level")
        for i, fib_price in enumerate(sorted_fibs):
            name = ratio_names[i] if i < len(ratio_names) else f"Fib{i}"
            lines.append(f"   {name}:  ${fib_price:.4f}")

    if dur_reasons:
        lines += [f"", f"🕐 DURATION ANALYSIS:"]
        for reason in dur_reasons:
            lines.append(f"   • {reason}")

    # Portfolio correlation context
    corr_type    = r.get('corr_type', 'independent')
    corr_slot    = r.get('corr_slot', 1)
    corr_flagged = r.get('corr_flagged', False)
    cap          = 3
    if corr_type in ('correlated_long', 'correlated_short'):
        direction = 'LONG' if corr_type == 'correlated_long' else 'SHORT'
        if corr_flagged:
            lines += [
                f"",
                f"⚠️ PORTFOLIO RISK — Correlated {direction} #{corr_slot} of {corr_slot}",
                f"   This is signal #{corr_slot} in the same BTC-correlated direction.",
                f"   Recommended cap: {cap}. Opening this adds concentrated exposure.",
                f"   All {direction}s in this regime stop out together on a BTC reversal.",
            ]
        else:
            lines += [f"", f"📊 PORTFOLIO SLOT — Correlated {direction} #{corr_slot}/{cap}"]
    else:
        lines += [f"", f"📊 PORTFOLIO SLOT — Independent signal (counter-regime or NEUTRAL)"]

    return "\n".join(lines)


def format_signal(r, rank):
    """Backwards-compatible wrapper — returns the primary card text."""
    return format_signal_primary(r, rank)


def cache_signal_card(r, rank, primary_keyboard):
    """
    Req #15 — Signal card buttons:
      [🔄 Refresh]  [📋 Details]   ← side by side on one row

    Details edits the message in-place (no new message).
    Back button restores the primary card.
    """
    key = uuid.uuid4().hex[:12]
    exchange = r.get('exchange', '')
    symbol   = r.get('symbol', '')
    _card_entry = {
        'signal':     r,
        'rank':       rank,
        'primary_kb': primary_keyboard,
    }
    state._signal_card_cache[key] = _card_entry
    db_save_card_cache(key, _card_entry)

    link = get_exchange_link(exchange, symbol)
    action_row = [
        InlineKeyboardButton("🔄 Refresh",  callback_data=f"sig_refresh|{key}"),
        InlineKeyboardButton("📋 Details",  callback_data=f"sig_details|{key}"),
    ]
    trade_row = [
        InlineKeyboardButton("🔗 Trade Now", url=link),
    ]
    return key, InlineKeyboardMarkup([action_row, trade_row])


async def signal_details_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Req #15 — Tap 📋 Details: edits the SAME panel in-place with technical analysis.
    Back button restores the primary signal card.
    """
    query = update.callback_query
    await query.answer()
    key   = query.data.split('|')[1]
    entry = state._signal_card_cache.get(key)
    if not entry:
        await query.answer("Signal data expired. Re-run the scan.", show_alert=True)
        return
    r    = entry['signal']
    rank = entry['rank']
    link = get_exchange_link(r.get('exchange', ''), r['symbol'])
    # FIX #AUTOREFRESH — pause the 30s timer while the Details sub-view is open
    # so the card isn't yanked back to the primary view mid-read.
    _autorefresh_pause(query)
    details_kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("⬅️ Back",     callback_data=f"sig_back|{key}"),
        InlineKeyboardButton("🔗 Trade Now", url=link),
    ]])
    try:
        await query.edit_message_text(
            format_signal_details(r, rank),
            reply_markup=details_kb,
        )
    except Exception:
        await query.message.reply_text(
            format_signal_details(r, rank), reply_markup=details_kb
        )


async def signal_refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Req #15 — Refresh button on signal card: re-fetches live price and
    rebuilds the primary card in-place.
    """
    query = update.callback_query
    await query.answer("Refreshing...")
    key   = query.data.split('|')[1]
    entry = state._signal_card_cache.get(key)
    if not entry:
        await query.answer("Signal data expired. Re-run the scan.", show_alert=True)
        return
    r    = entry['signal']
    rank = entry['rank']
    # FIX #AUTOREFRESH — back on the primary card, resume the 30s timer
    _autorefresh_resume(query)
    # Rebuild card — _build_signal_common fetches a fresh live price
    _, keyboard = cache_signal_card(r, rank, InlineKeyboardMarkup([]))
    try:
        await query.edit_message_text(
            format_signal_primary(r, rank),
            reply_markup=keyboard,
        )
    except Exception:
        pass


async def signal_back_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Restore primary card when user taps ⬅ Back."""
    query = update.callback_query
    await query.answer()
    key = query.data.split('|')[1]
    entry = state._signal_card_cache.get(key)
    if not entry:
        await query.answer("Signal data expired. Please re-run the scan.", show_alert=True)
        return
    r    = entry['signal']
    rank = entry['rank']
    # FIX #AUTOREFRESH — back on the primary card, resume the 30s timer
    _autorefresh_resume(query)
    # Restore original keyboard (already has the Details button baked in)
    existing_rows = entry['primary_kb'].inline_keyboard if entry['primary_kb'] else []
    restored_kb   = InlineKeyboardMarkup(list(existing_rows) + [
        [InlineKeyboardButton("📋 Details", callback_data=f"sig_details|{key}")]
    ])
    try:
        await query.edit_message_text(
            format_signal_primary(r, rank),
            reply_markup=restored_kb,
        )
    except Exception:
        await query.message.reply_text(format_signal_primary(r, rank), reply_markup=restored_kb)




# ───────────────────────────────��─────────────
# TRADE REMINDER JOB
# ───────────────────��─────────────────────────
async def send_trade_update(context: ContextTypes.DEFAULT_TYPE):
    chat_id  = context.job.chat_id
    trade_id = (context.job.data or {}).get('trade_id')
    trades   = state.user_tracking.get(chat_id, {})
    data     = trades.get(trade_id) if trade_id else (next(iter(trades.values()), None) if trades else None)
    if not data:
        return

    signal   = data['signal']
    exchange = signal['exchange']
    symbol   = signal['symbol']
    bias     = signal['bias']
    entry    = data['entry_price']
    start    = data['start_time']
    elapsed  = datetime.now() - start
    hours    = elapsed.total_seconds() / 3600

    current = _get_live_price(symbol, exchange)

    if current == 0:
        current = signal['price']

    pnl_pct      = ((current - entry) / entry * 100) if bias == "LONG" else ((entry - current) / entry * 100)
    pnl_emoji    = "📈" if pnl_pct >= 0 else "📉"
    status_emoji = "🟢" if pnl_pct >= 0 else "🔴"

    alerts = []
    if bias == "LONG":
        if current >= signal['t1']: alerts.append("🎯 TARGET 1 HIT!")
        if current >= signal['t2']: alerts.append("🎯🎯 TARGET 2 HIT!")
        if current >= signal['t3']: alerts.append("🎯🎯🎯 TARGET 3 HIT!")
        if current <= signal['stop_loss']: alerts.append("🛑 STOP LOSS HIT — Consider closing!")
    else:
        if current <= signal['t1']: alerts.append("🎯 TARGET 1 HIT!")
        if current <= signal['t2']: alerts.append("🎯🎯 TARGET 2 HIT!")
        if current <= signal['t3']: alerts.append("���🎯🎯 TARGET 3 HIT!")
        if current >= signal['stop_loss']: alerts.append("🛑 STOP LOSS HIT — Consider closing!")

    link = get_exchange_link(exchange, symbol)

    msg = (
        f"⏰ TRADE UPDATE\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"{exchange} | {symbol}\n"
        f"{status_emoji} {bias} | {signal['confidence']}/10\n\n"
        f"📥 Entry:   ${entry:.6f}\n"
        f"💰 Current: ${current:.6f}\n"
        f"🛑 SL:      ${signal['stop_loss']:.6f}\n"
        f"���� T1:      ${signal['t1']:.6f}\n\n"
        f"{pnl_emoji} PnL: {pnl_pct:+.2f}%\n"
        f"⏱ Open: {hours:.1f} hours\n"
    )
    if alerts:
        msg += f"\n{''.join(alerts)}\n"

    max_hold   = signal.get('hold_hours', 24)
    hold_label = signal.get('hold', f'{max_hold}h')
    warn_at    = max_hold * 0.85
    if hours >= max_hold:
        msg += f"\n��� RECOMMENDED HOLD ({hold_label}) REACHED. Close this trade now!\n"
    elif hours >= warn_at:
        msg += f"\n⚠️ Approaching recommended hold duration ({hold_label}). Consider closing soon.\n"

    stop_cb = f"stop_trade|{chat_id}|{trade_id}" if trade_id else "stop_trade_all"
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🛑 Stop Tracking", callback_data=stop_cb),
        InlineKeyboardButton("🔗 Trade Now", url=link),
    ]])
    msg += f"\n🔗 {link}"
    await context.bot.send_message(chat_id=chat_id, text=msg, reply_markup=keyboard)


# ───────────────────────────────���─────────────
# CONVERSATION: PICK → REMINDER → INTERVAL
# ─────────────────────────────────────────────
async def pick_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    if not state.last_scan_results:
        await update.message.reply_text("⚠️ No scan data yet. Run /scan first.")
        return ConversationHandler.END

    total = len(state.last_scan_results)
    lines = [f"📊 LAST SCAN — {total} signals\n"]
    for i, r in enumerate(state.last_scan_results[:20], 1):
        emoji = "🟢" if r['bias'] == "LONG" else "🔴"
        lines.append(f"{i}. {emoji} {r['exchange']} {r['symbol']} — {r['bias']} {r['confidence']}/10")
    if total > 20:
        lines.append(f"... and {total-20} more")
    lines.append(f"\n💬 Reply with a number (1 to {total})")
    await update.message.reply_text("\n".join(lines))
    return PICK_TRADE

async def receive_trade_pick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    try:
        n = int(text)
        if n < 1 or n > len(state.last_scan_results):
            await update.message.reply_text(f"⚠️ Enter a number between 1 and {len(state.last_scan_results)}.")
            return PICK_TRADE
        signal = state.last_scan_results[n - 1]
        context.user_data['picked_signal'] = signal
        context.user_data['picked_rank']   = n
        await update.message.reply_text(format_signal(signal, n))
        keyboard = [["Yes", "No"]]
        await update.message.reply_text(
            f"✅ You picked #{n} — {signal['exchange']} {signal['symbol']}\n\n"
            f"🔔 Set a live reminder for this trade?",
            reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        )
        return ASK_REMINDER
    except ValueError:
        await update.message.reply_text("⚠️ Reply with a number only.")
        return PICK_TRADE

async def receive_reminder_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    if text in ['yes', 'y']:
        await update.message.reply_text(
            "⏱ How often should I send updates? (minutes)\nExamples: 5, 10, 15, 30, 60",
            reply_markup=ReplyKeyboardRemove()
        )
        return ASK_INTERVAL
    else:
        signal = context.user_data.get('picked_signal', {})
        await update.message.reply_text(
            f"✅ No reminders set. Good luck on your {signal.get('exchange','')} {signal.get('symbol','')} trade! 🚀",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

async def receive_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    try:
        interval_mins = int(text)
        if interval_mins < 1:
            await update.message.reply_text("��️ Minimum interval is 1 minute.")
            return ASK_INTERVAL
        if interval_mins > 1440:
            await update.message.reply_text("⚠️ Maximum interval is 1440 minutes (24 hours).")
            return ASK_INTERVAL

        signal   = context.user_data.get('picked_signal')
        chat_id  = update.effective_chat.id
        trade_id = uuid.uuid4().hex[:12]
        start_time = datetime.now()

        # Add to multi-trade dict (never overwrites other trades)
        if chat_id not in state.user_tracking:
            state.user_tracking[chat_id] = {}
        state.user_tracking[chat_id][trade_id] = {
            'signal':      signal,
            'entry_price': signal['price'],
            'start_time':  start_time,
            'interval':    interval_mins,
            'job':         None,
            'trade_id':    trade_id,
        }
        db_save_trade(chat_id, trade_id, signal, signal['price'], start_time, interval_mins)

        job = context.job_queue.run_repeating(
            send_trade_update,
            interval=interval_mins * 60,
            first=interval_mins * 60,
            chat_id=chat_id,
            name=f"trade_{chat_id}_{trade_id}",
            data={'trade_id': trade_id}
        )
        state.user_tracking[chat_id][trade_id]['job'] = job
        active_count = len(state.user_tracking[chat_id])

        await update.message.reply_text(
            f"✅ REMINDER SET!\n\n"
            f"📊 {signal['exchange']} | {signal['symbol']}\n"
            f"📥 Entry: ${signal['price']:.6f}\n"
            f"⏱ Updates every {interval_mins} min\n\n"
            f"I'll alert you on T1/T2/T3 hits, SL, and max hold.\n"
            f"You now have {active_count} active trade(s). Use /trades to view all.\n"
            f"Use /stoptrade to manage tracked trades.",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text("⚠️ Please reply with a number (minutes).")
        return ASK_INTERVAL

async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Cancelled.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


# ─────────────────────────────────────────────
# STOP TRADE
# ─────────────────────────────────────────────
async def stoptrade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all active tracked trades with inline Stop buttons."""
    _track(update)
    chat_id = update.effective_chat.id
    trades  = state.user_tracking.get(chat_id, {})

    if not trades:
        await update.message.reply_text("⚠️ No active trades being tracked. Use /pick to start one.")
        return

    now   = datetime.now()
    lines = ["🛑  A C T I V E  T R A D E S\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
    buttons = []

    for i, (tid, data) in enumerate(trades.items(), 1):
        sig   = data['signal']
        hours = (now - data['start_time']).total_seconds() / 3600
        emoji = "🟢" if sig['bias'] == "LONG" else "🔴"
        lines.append(
            f"{emoji} #{i}  {sig['exchange']} | {sig['symbol']}\n"
            f"     {sig['bias']} — {sig['confidence']}/10 — open {hours:.1f}h\n"
            f"     Updates every {data['interval']} min"
        )
        buttons.append([InlineKeyboardButton(
            f"🛑 Stop #{i} — {sig['symbol']}",
            callback_data=f"stop_trade|{chat_id}|{tid}"
        )])

    if len(trades) > 1:
        buttons.append([InlineKeyboardButton(
            "🛑 Stop ALL trades", callback_data=f"stop_trade_all|{chat_id}"
        )])

    lines.append(f"\n👆 Tap to stop a specific trade. Use /pick to add another.")
    await update.message.reply_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(buttons)
    )


async def stop_trade_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle Stop buttons from /stoptrade and trade update reminders."""
    query = update.callback_query
    await query.answer()
    parts = query.data.split("|")

    if parts[0] == "stop_trade_all":
        chat_id = int(parts[1])
        trades  = state.user_tracking.pop(chat_id, {})
        for tid, data in trades.items():
            try:
                if data.get('job'): data['job'].schedule_removal()
            except Exception: pass
            db_remove_trade(tid)
            # FIX: clear flip-cooldown so stopped pairs can be re-scanned immediately
            stopped_sym = data.get('signal', {}).get('symbol', '')
            if stopped_sym:
                state._last_signal_bias.pop(stopped_sym, None)
        await query.edit_message_text(f"✅ All {len(trades)} trade(s) stopped.")

    elif parts[0] == "stop_trade":
        chat_id  = int(parts[1])
        trade_id = parts[2]
        trades   = state.user_tracking.get(chat_id, {})
        data     = trades.pop(trade_id, None)
        if data:
            try:
                if data.get('job'): data['job'].schedule_removal()
            except Exception: pass
            db_remove_trade(trade_id)
            sig       = data['signal']
            remaining = len(trades)
            # ── FIX: clear flip-cooldown so a fresh /scan on this pair
            #    works immediately after the user explicitly stops the trade.
            stopped_sym = sig.get('symbol', '')
            if stopped_sym and stopped_sym in state._last_signal_bias:
                state._last_signal_bias.pop(stopped_sym, None)
                logger.debug("stop_trade: cleared _last_signal_bias for %s", stopped_sym)
            msg = (
                f"✅ Stopped tracking {sig['exchange']} | {sig['symbol']}.\n"
                + (f"{remaining} trade(s) still active. Use /trades to view." if remaining
                   else "No active trades remaining.")
            )
            await query.edit_message_text(msg)
        else:
            await query.edit_message_text("⚠️ Trade not found — may have already been stopped.")


async def trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Live status panel for all active tracked trades."""
    _track(update)
    chat_id = update.effective_chat.id
    trades  = state.user_tracking.get(chat_id, {})

    if not trades:
        await update.message.reply_text("⚠️ No active trades. Run /scan then /pick to start tracking.")
        return

    now   = datetime.now()
    lines = ["📊  T R A D E  P O R T F O L I O\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]

    for i, (tid, data) in enumerate(trades.items(), 1):
        sig      = data['signal']
        exchange = sig['exchange']
        symbol   = sig['symbol']
        bias     = sig['bias']
        entry    = data['entry_price']
        hours    = (now - data['start_time']).total_seconds() / 3600

        try:
            current = _get_live_price(symbol, exchange)
        except Exception:
            current = 0

        if current > 0:
            pnl_pct   = (current - entry) / entry * 100 if bias == "LONG" else (entry - current) / entry * 100
            pnl_emoji = "📈" if pnl_pct >= 0 else "📉"
            t1_hit    = (current >= sig['t1'] if bias == "LONG" else current <= sig['t1'])
            t2_hit    = (current >= sig['t2'] if bias == "LONG" else current <= sig['t2'])
            sl_hit    = (current <= sig['stop_loss'] if bias == "LONG" else current >= sig['stop_loss'])
            status    = "🎯 T2 HIT" if t2_hit else ("🎯 T1 HIT" if t1_hit else ("🛑 SL HIT" if sl_hit else "🔄 Open"))
            price_str = f"${entry:.4f} → ${current:.4f}  {pnl_emoji}{pnl_pct:+.1f}%  {status}"
        else:
            price_str = f"${entry:.4f}  (live price unavailable)"

        emoji = "🟢" if bias == "LONG" else "🔴"
        lines.append(
            f"\n{emoji} #{i}  {exchange} | {symbol}\n"
            f"     {bias} — {sig['confidence']}/10 — {hours:.1f}h open\n"
            f"     {price_str}\n"
            f"     SL ${sig['stop_loss']:.4f}  T1 ${sig['t1']:.4f}  T2 ${sig['t2']:.4f}"
        )

    lines.append(f"\n\nUse /stoptrade to stop any trade  |  /pick to add another.")
    await update.message.reply_text("\n".join(lines))


# ─────────────────────────────────────────────
# PRICE-LEVEL ALERTS  (/palert, /unpalert)
# ────────────────���────────────────────────────

async def palert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /palert BTCUSDT 95000         — alert when BTC crosses $95,000
    /palert BTCUSDT 90000 below   — explicit direction override
    Direction is auto-detected vs live price if not specified.
    """
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args

    if not args or len(args) < 2:
        await update.message.reply_text(
            "⚠️ Usage: /palert SYMBOL TARGET [above|below]\n\n"
            "Examples:\n"
            "  /palert BTCUSDT 95000\n"
            "  /palert ETHUSDT 3200 below"
        )
        return

    symbol = args[0].upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    try:
        target = float(args[1].replace(",", ""))
    except ValueError:
        await update.message.reply_text("⚠️ Invalid price. Use a number like 95000 or 3200.50")
        return

    exchange = "BYBIT"
    current = _get_live_price(symbol, exchange)

    if len(args) >= 3 and args[2].lower() in ('above', 'below'):
        direction = args[2].lower()
    elif current > 0:
        direction = "above" if target > current else "below"
    else:
        direction = "above"

    arrow       = "📈" if direction == "above" else "📉"
    alert_id    = db_save_price_alert(chat_id, symbol, exchange, target, direction)
    current_str = f"  (now ${current:,.4f})" if current > 0 else ""

    await update.message.reply_text(
        f"🔔 PRICE ALERT SET\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{arrow} {symbol} {direction} ${target:,.4f}{current_str}\n\n"
        f"I'll notify you the moment price crosses that level.\n"
        f"Use /unpalert {symbol} to remove."
    )


async def unpalert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/unpalert BTCUSDT — remove alerts, or /unpalert alone to show and pick."""
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args

    if args:
        symbol = args[0].upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        db_remove_price_alerts_for_symbol(chat_id, symbol)
        await update.message.reply_text(f"✅ All price alerts for {symbol} removed.")
        return

    alerts = db_get_price_alerts(chat_id)
    if not alerts:
        await update.message.reply_text("⚠️ No active price alerts. Use /palert to set one.")
        return

    lines   = ["🔔  A C T I V E  P R I C E  A L E R T S\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
    buttons = []
    for i, a in enumerate(alerts, 1):
        arrow = "📈" if a['direction'] == 'above' else "📉"
        lines.append(f"{arrow} #{i}  {a['symbol']} {a['direction']} ${a['target']:,.4f}")
        buttons.append([InlineKeyboardButton(
            f"🗑 Remove #{i} — {a['symbol']} {a['direction']} ${a['target']:,.4f}",
            callback_data=f"rm_palert|{a['id']}"
        )])

    await update.message.reply_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(buttons)
    )


async def remove_palert_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inline button handler to remove a single price alert."""
    query    = update.callback_query
    await query.answer()
    alert_id = query.data.split("|")[1]
    db_remove_price_alert(alert_id)
    await query.edit_message_text("✅ Price alert removed.")


async def price_alert_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Runs every 3 minutes. Checks all untriggered price-level alerts
    against live prices and fires a message when the level is crossed.
    """
    alerts = db_get_all_price_alerts()
    if not alerts:
        return

    prices: dict = {}
    for a in alerts:
        key = f"{a['exchange']}_{a['symbol']}"
        if key not in prices:
            try:
                p = _get_live_price(a['symbol'], a['exchange'])
                prices[key] = p
            except Exception:
                prices[key] = 0
            await asyncio.sleep(0.1)

    for a in alerts:
        key     = f"{a['exchange']}_{a['symbol']}"
        current = prices.get(key, 0)
        if current == 0:
            continue
        fired = (
            (a['direction'] == 'above' and current >= a['target']) or
            (a['direction'] == 'below' and current <= a['target'])
        )
        if not fired:
            continue

        db_mark_price_alert_triggered(a['id'])
        arrow = "📈" if a['direction'] == 'above' else "📉"
        link  = get_exchange_link(a['exchange'], a['symbol'])
        try:
            await context.bot.send_message(
                chat_id=a['chat_id'],
                text=(
                    f"🔔 PRICE ALERT TRIGGERED!\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{arrow} {a['symbol']} has gone {a['direction']} ${a['target']:,.4f}\n"
                    f"💰 Current price: ${current:,.4f}\n\n"
                    f"🔗 {link}"
                )
            )
        except Exception as e:
            logger.warning("price_alert_job: failed to notify chat %s: %s", a['chat_id'], e)


# ─────────────────────────────────────────────
# PNL CARD — /pnl command + inline keyboard
# Users can query PnL with bot leverage or custom
# ───────────────────────────────��─────��─��──����─
def build_pnl_card(signal, leverage, capital, custom=False):
    """Generate a full PnL card for a signal at given leverage and capital."""
    bias       = signal['bias']
    entry      = signal['price']
    sl         = signal['stop_loss']
    t1         = signal['t1']
    t2         = signal['t2']
    t3         = signal['t3']
    exchange   = signal.get('exchange', '')
    symbol     = signal['symbol']
    lev_data   = signal.get('leverage')
    bot_lev    = lev_data['suggested'] if lev_data else None
    lev_note   = "Custom" if custom else "Bot suggested"

    position_size = capital * leverage

    def pnl(target):
        if bias == 'LONG':
            pct = (target - entry) / entry * 100
        else:
            pct = (entry - target) / entry * 100
        raw_pnl = capital * (pct / 100) * leverage
        return pct, raw_pnl

    sl_pct, sl_pnl   = pnl(sl)
    t1_pct, t1_pnl   = pnl(t1)
    t2_pct, t2_pnl   = pnl(t2)
    t3_pct, t3_pnl   = pnl(t3)

    bias_emoji = "🟢" if bias == "LONG" else "🔴"
    liq_price  = entry * (1 - 0.9/leverage) if bias == "LONG" else entry * (1 + 0.9/leverage)

    lines = [
        f"💰 PNL CARD — {exchange} | {symbol}",
        f"━━━━━━━���━━━━━━━━━━━━━━━━━━━━━━",
        f"{bias_emoji} {bias}  |  {signal['confidence']}/10  |  {lev_note} leverage",
        f"",
        f"💵 Capital:       ${capital:,.2f}",
        f"⚡ Leverage:      {leverage}x",
        f"📦 Position size: ${position_size:,.2f}",
        f"",
        f"📥 Entry:         ${entry:.6f}",
        f"💀 Liq. Price:    ${liq_price:.6f}",
        f"",
        f"━━━━━━━━��━━━━━━━━━━━━━━━━━━━━━",
        f"📊 SCENARIO ANALYSIS",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"🛑 Stop Loss   ${sl:.6f}",
        f"   Move: {sl_pct:+.2f}%  →  {'🔴 -$' if sl_pnl < 0 else '🟢 +$'}{abs(sl_pnl):,.2f}",
        f"",
        f"🎯 Target 1    ${t1:.6f}",
        f"   Move: {t1_pct:+.2f}%  ��  🟢 +${t1_pnl:,.2f}",
        f"",
        f"🎯 Target 2    ${t2:.6f}",
        f"   Move: {t2_pct:+.2f}%  →  🟢 +${t2_pnl:,.2f}",
        f"",
        f"🎯 Target 3    ${t3:.6f}",
        f"   Move: {t3_pct:+.2f}%  →  🟢 +${t3_pnl:,.2f}",
        f"",
        f"━━━━���━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if bot_lev and not custom:
        lines.append(f"ℹ️ Bot suggested {bot_lev}x based on ATR & confidence.")
    elif custom:
        if bot_lev:
            lines.append(f"ℹ️ Bot suggested leverage: {bot_lev}x (you used {leverage}x).")
        lines.append(f"⚠️ Always use isolated margin with custom leverage.")

    lines.append(f"\n🔗 {get_exchange_link(exchange, symbol)}")
    return "\n".join(lines)


def _fmt_held_for(scan_time):
    """Human 'Xd Yh Zm' duration since scan_time (datetime or ISO string)."""
    if not scan_time:
        return "\u2014"
    try:
        st = datetime.fromisoformat(scan_time) if isinstance(scan_time, str) else scan_time
        if not isinstance(st, datetime):
            return "\u2014"
        secs = int((datetime.now() - st.replace(tzinfo=None)).total_seconds())
        secs = max(secs, 0)
        days, rem = divmod(secs, 86400)
        hours, rem = divmod(rem, 3600)
        mins = rem // 60
        if days > 0:
            return f"{days}d {hours}h {mins}m"
        if hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m"
    except Exception:
        return "\u2014"


def _fetch_sparkline_closes(exchange, symbol, limit=60):
    """Recent 1h close prices for the PnL-card sparkline (oldest-first floats)."""
    try:
        exch = (exchange or '').upper()
        if exch == 'BYBIT':
            df = bybit_fetch_ohlcv(symbol, '60', limit)
        elif exch == 'BINANCE':
            df = binance_fetch_ohlcv(symbol, '1h', limit)
        else:
            df = mexc_fetch_ohlcv(symbol, '1h', limit)
        if df is None or len(df) == 0:
            return []
        return [float(c) for c in df['close'].tolist()]
    except Exception as e:
        logger.warning("sparkline fetch failed %s %s: %s", exchange, symbol, e)
        return []


def _fetch_pnl_chart_closes(signal, current=None):
    """Closes for the PnL-card sparkline, ending at the live price.

    Wraps _fetch_sparkline_closes using the signal's symbol/exchange. The
    live `current` price (when provided and positive) is appended so the
    chart ends at the latest tick. Always returns a list of floats.
    """
    try:
        symbol   = signal.get('symbol', '')   if isinstance(signal, dict) else ''
        exchange = signal.get('exchange', '') if isinstance(signal, dict) else ''
        closes = _fetch_sparkline_closes(exchange, symbol) or []
        try:
            if current is not None and float(current) > 0:
                closes = list(closes) + [float(current)]
        except (TypeError, ValueError):
            pass
        return closes
    except Exception as e:
        logger.warning("_fetch_pnl_chart_closes failed: %s", e)
        return []


def render_pnl_card_image(signal, current_price, leverage, capital=None, closes=None,
                          peak_raw_pct=None, peak_at=None, peak_loss_pct=None, peak_loss_at=None,
                          username=None, out_scale=1.0):
    """Render the SAKZBOTT vault-style PnL card (PNG bytes).

    Front-face layout (flat, dark, green/red accent):
      • header: bias triangle + SAKZBOTT wordmark
      • avatar + Telegram username + gold 'Trader' badge
      • chips row: bias / pair / leverage / exchange (MEXC)
      • 'My Vault PnL': peak ROI %% (and profit $ beside it when capital is set)
      • footer: Entry Price (from the signal) + Exit Price (peak/current)

    The headline %% is the PEAK leveraged ROI — the best the pair reached from
    signal to its max profit before retracement. With no capital the card shows
    the percentage only; with capital it shows '$amount  +%%' side by side.
    No QR code, no manager fee.
    """
    import io
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse

    # ---- palette --------------------------------------------------------
    BG       = "#06080B"
    CARD     = "#0B0F14"
    CARD_ED  = "#1C2630"
    CHIP_BG  = "#10161D"
    CHIP_ED  = "#222C37"
    GREEN    = "#2FD675"
    GREEN_DK = "#1B9B57"
    RED      = "#F0556B"
    RED_DK   = "#A33442"
    WHITE    = "#FFFFFF"
    SOFT     = "#AEB6BF"
    GRAY     = "#7E8893"
    GOLD     = "#F2C744"
    INK      = "#0A0D11"
    BTC_ORANGE = "#F7931A"
    MEXC_BLUE  = "#1D6CFF"

    ASPECT = 10.24 / 6.83

    fig = plt.figure(figsize=(10.24, 6.83), dpi=100, facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    def disc(x, y, r, **kw):
        ax.add_patch(Ellipse((x, y), width=2 * r / ASPECT, height=2 * r, **kw))

    # ---- derive values --------------------------------------------------
    brand = os.environ.get("BOT_NAME", "Sakz")

    entry = float(signal.get('price') or 0) or float(current_price or 0)
    bias  = str(signal.get('bias', 'LONG')).upper()
    is_long = bias == 'LONG'
    cur   = float(current_price or 0)
    lev   = int(leverage or 1)

    # current raw (unleveraged) move, direction-aware
    if entry > 0 and cur > 0:
        cur_raw = ((cur - entry) / entry * 100) if is_long else ((entry - cur) / entry * 100)
    else:
        cur_raw = 0.0

    # PEAK favourable move (raw) → leveraged headline ROI (best before retrace)
    fav_raw = peak_raw_pct if peak_raw_pct is not None else cur_raw
    if fav_raw < cur_raw:
        fav_raw = cur_raw
    peak_lev = fav_raw * lev
    if peak_lev < -100.0:
        peak_lev = -100.0

    up = peak_lev >= 0
    if not up:                       # loss → flip every accent-green to red
        GREEN, GREEN_DK = RED, RED_DK
    accent = GREEN
    # bias colour is PERMANENT identity: LONG=green, SHORT=red (independent of PnL flip)
    bias_color = "#2FD675" if is_long else "#F0556B"
    tri = "\u25B2" if is_long else "\u25BC"

    raw_sym  = str(signal.get('symbol', '')).upper().replace('/', '').replace('_', '')
    base_sym = raw_sym[:-4] if raw_sym.endswith('USDT') else raw_sym
    exch     = (str(signal.get('exchange', '')).upper() or 'MEXC')

    # exit price ALWAYS reflects the same peak (favourable) move the headline %
    # is built from, so Entry -> Exit is internally consistent with 'My Vault PnL'.
    # (current price is shown separately in its own column.)
    if entry > 0 and fav_raw is not None:
        exit_price = entry * (1 + fav_raw / 100.0) if is_long else entry * (1 - fav_raw / 100.0)
    else:
        exit_price = cur or entry

    def _fmt_price(p):
        if not p:
            return "\u2014"
        p = float(p)
        if p >= 1000:
            return f"{p:,.2f}"
        if p >= 1:
            return f"{p:,.4f}"
        return f"{p:.6f}"

    pct_str = f"{peak_lev:+.2f}%"
    show_amount = (capital is not None) and (capital > 0)
    dollar_str = None
    if show_amount:
        dval = capital * peak_lev / 100.0
        if abs(dval) >= 100:
            dollar_str = f"{'+' if dval >= 0 else '-'}${abs(dval):,.0f}"
        else:
            dollar_str = f"{'+' if dval >= 0 else '-'}${abs(dval):,.2f}"

    uname = str(username or signal.get('username') or "Trader").lstrip('@')
    if len(uname) > 16:
        uname = uname[:15] + "\u2026"

    # ---- card body ------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.035, 0.06), 0.93, 0.88,
        boxstyle="round,pad=0,rounding_size=0.045",
        linewidth=1.4, edgecolor=CARD_ED, facecolor=CARD, zorder=1))

    # right-side green/red glow (echoes the reference art)
    gx = np.linspace(0, 1, 220); gy = np.linspace(0, 1, 220)
    GXX, GYY = np.meshgrid(gx, gy)
    dist = np.sqrt((GXX - 0.80) ** 2 + (GYY - 0.60) ** 2)
    glow = np.clip(0.13 - dist * 0.40, 0, 0.13)
    rgba = np.zeros((glow.shape[0], glow.shape[1], 4))
    if up:
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.18, 0.84, 0.47
    else:
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.94, 0.33, 0.42
    rgba[..., 3] = glow * 2.4
    ax.imshow(rgba, extent=[0.50, 0.965, 0.16, 0.92], aspect='auto', origin='lower',
              zorder=1, interpolation='bilinear')

    # stacked chevrons pointing up (LONG) / down (SHORT)
    chev_dir = 1 if is_long else -1
    for k in range(5):
        bx = 0.60 + k * 0.068
        amp = 0.12
        ya = 0.46
        alpha = max(0.30 - k * 0.055, 0.05)
        ax.plot([bx - 0.046, bx, bx + 0.046],
                [ya, ya + amp * chev_dir, ya],
                color=accent, lw=9, solid_capstyle='round',
                solid_joinstyle='round', alpha=alpha, zorder=1)

    # ---- header: bias triangle + wordmark -------------------------------
    hx, hy = 0.082, 0.876
    ax.add_patch(Polygon([(hx - 0.013, hy - 0.015), (hx + 0.013, hy - 0.015), (hx, hy + 0.017)],
        closed=(is_long), facecolor=bias_color, edgecolor='none', zorder=5))
    if not is_long:
        ax.add_patch(Polygon([(hx - 0.013, hy + 0.015), (hx + 0.013, hy + 0.015), (hx, hy - 0.017)],
            closed=True, facecolor=bias_color, edgecolor='none', zorder=5))
    ax.text(hx + 0.028, hy, brand, color=WHITE, fontsize=23, fontweight='bold',
            va='center', ha='left', zorder=5, fontstyle='italic')

    # ---- avatar + username + Trader badge -------------------------------
    av_x, av_y = 0.098, 0.758
    disc(av_x, av_y, 0.030, facecolor="#263241", edgecolor=CARD_ED, lw=1.0, zorder=4)
    ax.text(av_x, av_y, (uname[:1].upper() or 'T'), color=WHITE, fontsize=15,
            fontweight='bold', va='center', ha='center', zorder=5)
    name_x = av_x + 0.052
    ax.text(name_x, av_y, uname, color=WHITE, fontsize=15, fontweight='bold',
            va='center', ha='left', zorder=5)
    badge_x = name_x + 0.0150 * len(uname) + 0.020
    badge_w = 0.018 * 2 + 0.0118 * len("Trader")
    ax.add_patch(FancyBboxPatch((badge_x, av_y - 0.026), badge_w, 0.052,
        boxstyle="round,pad=0,rounding_size=0.020",
        linewidth=0, facecolor=GOLD, zorder=4))
    ax.text(badge_x + badge_w / 2, av_y, "Trader", color=INK, fontsize=11,
            fontweight='bold', va='center', ha='center', zorder=5)

    # ---- chips row ------------------------------------------------------
    def chip(x, y, label, *, fg=WHITE, border=CHIP_ED, fill=CHIP_BG, icon=None, icon_col=None):
        pad = 0.018
        icon_w = 0.032 if icon else 0.0
        tw = 0.0140 * len(label)
        w = pad * 2 + icon_w + tw
        h = 0.052
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.020",
            linewidth=1.3, edgecolor=border, facecolor=fill, zorder=4))
        tx = x + pad
        if icon in ('coin', 'mexc'):
            disc(x + pad + 0.013, y + h / 2, 0.014, facecolor=icon_col, edgecolor='none', zorder=5)
            glyph = base_sym[:1] if icon == 'coin' else 'M'
            ax.text(x + pad + 0.013, y + h / 2, glyph, color=WHITE if icon == 'mexc' else INK,
                    fontsize=8.5, fontweight='bold', va='center', ha='center', zorder=6)
            tx = x + pad + 0.032
        ax.text(tx, y + h / 2, label, color=fg, fontsize=12, fontweight='bold',
                va='center', ha='left', zorder=6)
        return x + w + 0.016

    cy = 0.628
    nx = 0.078
    nx = chip(nx, cy, f"{tri} {bias}", fg=bias_color, border=bias_color)
    nx = chip(nx, cy, base_sym, icon='coin', icon_col=BTC_ORANGE)
    nx = chip(nx, cy, f"{lev}x")
    nx = chip(nx, cy, exch, icon='mexc', icon_col=MEXC_BLUE)

    # ---- Pair symbol (always white) sits directly above the PnL number --
    ax.text(0.080, 0.508, raw_sym, color="#FFFFFF", fontsize=18, fontweight='bold',
            va='center', ha='left', zorder=5)
    if show_amount:
        _td = ax.text(0.076, 0.410, dollar_str, color=accent, fontsize=44, fontweight='bold',
                      va='center', ha='left', zorder=5)
        fig.canvas.draw()  # realise text geometry so we can place % right after it
        _bb = _td.get_window_extent(renderer=fig.canvas.get_renderer())
        _x_right = ax.transData.inverted().transform((_bb.x1, _bb.y0))[0]
        ax.text(min(_x_right + 0.024, 0.60), 0.398, pct_str, color=accent, fontsize=19,
                fontweight='bold', va='center', ha='left', zorder=5)
    else:
        ax.text(0.076, 0.410, pct_str, color=accent, fontsize=46, fontweight='bold',
                va='center', ha='left', zorder=5)

    # ---- footer: Duration + balanced Entry / Exit / Current price row ----
    # Time it took to run from the signal (entry) to the peak (exit) price.
    def _fmt_dur(a, b):
        try:
            if a is None or b is None:
                return "\u2014"
            if isinstance(a, str):
                a = datetime.fromisoformat(a)
            if isinstance(b, str):
                b = datetime.fromisoformat(b)
            if not isinstance(a, datetime) or not isinstance(b, datetime):
                return "\u2014"
            a = a.replace(tzinfo=None)
            b = b.replace(tzinfo=None)
            secs = max(int((b - a).total_seconds()), 0)
            days, rem = divmod(secs, 86400)
            hours, rem = divmod(rem, 3600)
            mins = rem // 60
            if days > 0:
                return f"{days}d {hours}h {mins}m"
            if hours > 0:
                return f"{hours}h {mins}m"
            return f"{mins}m"
        except Exception:
            return "\u2014"

    peak_dur = _fmt_dur(signal.get('scan_time'), peak_at)

    # Duration (first call -> peak) sits under the headline, left-aligned & lit.
    ax.text(0.080, 0.312, "Duration", color=GRAY, fontsize=10.5, fontweight='bold',
            va='center', ha='left', zorder=5)
    ax.text(0.080, 0.268, peak_dur, color=WHITE, fontsize=14, fontweight='bold',
            va='center', ha='left', zorder=5)

    # Three balanced, evenly-spaced price columns kept inside the lit face so
    # they read as one clean row on the tilted card (no cascade off the edge).
    fy = 0.168
    _foot = [
        ("Entry Price",   _fmt_price(entry),      WHITE),
        ("Exit Price",    _fmt_price(exit_price),  accent),
        ("Current Price", _fmt_price(cur),         WHITE),
    ]
    _col_x = [0.080, 0.290, 0.500]
    for (_lbl, _val, _col), _fx in zip(_foot, _col_x):
        ax.text(_fx, fy + 0.030, _lbl, color=GRAY, fontsize=10.5, fontweight='bold',
                va='center', ha='left', zorder=5)
        ax.text(_fx, fy - 0.014, _val, color=_col, fontsize=14, fontweight='bold',
                va='center', ha='left', zorder=5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=BG, edgecolor='none', dpi=100 * out_scale)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_pnl_card_flat_v2(signal, current_price, leverage, capital=None, closes=None,
                            peak_raw_pct=None, peak_at=None, peak_loss_pct=None, peak_loss_at=None,
                            username=None, out_scale=1.0):
    """Render the flat 'terminal' SAKZ PnL card (PNG bytes) -- the second display.

    Clean grid face, nested right-pointing chevrons, a 'MY VAULT PNL' headline
    with a percentage badge, and a single Entry / Exit / Current / Duration
    footer row. Inputs and value semantics mirror render_pnl_card_image so the
    two card styles always show identical numbers; only the look differs.
    """
    import io
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse

    # ---- palette (sampled from the reference card) ----------------------
    BG        = "#07080A"
    CARD      = "#080D0B"
    CARD_ED   = "#1E3A2B"
    CHIP_BG   = "#0C140F"
    CHIP_ED   = "#24352B"
    GRID      = "#103A26"
    GREEN     = "#21F07A"
    GREEN_DK  = "#0E3D1E"
    RED       = "#F0556B"
    RED_DK    = "#3D1119"
    WHITE     = "#F4F8F5"
    SOFT      = "#9AA6A0"
    GRAY      = "#6E7A74"
    GOLD      = "#E7B53C"
    INK       = "#0A0D0B"
    BTC_ORANGE = "#F7931A"
    BYBIT_GOLD = "#F7A600"
    EX_BLUE    = "#2E6BFF"

    ASPECT = 648.0 / 371.0
    W_IN = 10.24
    H_IN = W_IN / ASPECT
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=100, facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    def disc(x, y, r, **kw):
        ax.add_patch(Ellipse((x, y), width=2 * r / ASPECT, height=2 * r, **kw))

    # ---- derive values (same logic as the vault card) -------------------
    brand = os.environ.get("BOT_NAME", "Sakz")
    entry = float(signal.get('price') or 0) or float(current_price or 0)
    bias  = str(signal.get('bias', 'LONG')).upper()
    is_long = bias == 'LONG'
    cur   = float(current_price or 0)
    lev   = int(leverage or 1)

    if entry > 0 and cur > 0:
        cur_raw = ((cur - entry) / entry * 100) if is_long else ((entry - cur) / entry * 100)
    else:
        cur_raw = 0.0
    fav_raw = peak_raw_pct if peak_raw_pct is not None else cur_raw
    if fav_raw < cur_raw:
        fav_raw = cur_raw
    peak_lev = fav_raw * lev
    if peak_lev < -100.0:
        peak_lev = -100.0

    up = peak_lev >= 0
    if not up:
        GREEN, GREEN_DK = RED, RED_DK
    accent = GREEN
    bias_color = "#21F07A" if is_long else "#F0556B"
    tri = "\u25B2" if is_long else "\u25BC"

    raw_sym  = str(signal.get('symbol', '')).upper().replace('/', '').replace('_', '')
    base_sym = raw_sym[:-4] if raw_sym.endswith('USDT') else raw_sym
    exch     = (str(signal.get('exchange', '')).upper() or 'MEXC')

    if entry > 0 and fav_raw is not None:
        exit_price = entry * (1 + fav_raw / 100.0) if is_long else entry * (1 - fav_raw / 100.0)
    else:
        exit_price = cur or entry

    def _fmt_price(p):
        if not p:
            return "\u2014"
        p = float(p)
        if p >= 1000:
            return f"{p:,.2f}"
        if p >= 1:
            return f"{p:,.4f}"
        return f"{p:.6f}"

    pct_str = f"{peak_lev:+.2f}%"
    show_amount = (capital is not None) and (capital > 0)
    if show_amount:
        dval = capital * peak_lev / 100.0
        if abs(dval) >= 100:
            headline = f"{'+' if dval >= 0 else '-'}${abs(dval):,.0f}"
        else:
            headline = f"{'+' if dval >= 0 else '-'}${abs(dval):,.2f}"
    else:
        headline = pct_str

    uname = str(username or signal.get('username') or "Trader").lstrip('@')
    if len(uname) > 16:
        uname = uname[:15] + "\u2026"

    # ---- card face ------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.012, 0.022), 0.976, 0.956,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=1.4, edgecolor=CARD_ED, facecolor=CARD, zorder=1))

    clip = FancyBboxPatch((0.012, 0.022), 0.976, 0.956,
        boxstyle="round,pad=0,rounding_size=0.05", transform=ax.transData)
    for gx in np.arange(0.05, 0.99, 0.0455):
        ln, = ax.plot([gx, gx], [0.03, 0.97], color=GRID, lw=0.6, alpha=0.5, zorder=1)
        ln.set_clip_path(clip)
    for gy in np.arange(0.06, 0.97, 0.08):
        ln, = ax.plot([0.02, 0.98], [gy, gy], color=GRID, lw=0.6, alpha=0.5, zorder=1)
        ln.set_clip_path(clip)

    gx = np.linspace(0, 1, 240); gy = np.linspace(0, 1, 240)
    GXX, GYY = np.meshgrid(gx, gy)
    glow = np.zeros_like(GXX)
    for cxp, cyp, s in [(0.92, 0.92, 0.16), (0.86, 0.10, 0.10), (0.07, 0.55, 0.06)]:
        d = np.sqrt((GXX - cxp) ** 2 + (GYY - cyp) ** 2)
        glow = np.maximum(glow, np.clip(s - d * 0.32, 0, s))
    rgba = np.zeros((glow.shape[0], glow.shape[1], 4))
    if up:
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.13, 0.94, 0.48
    else:
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.94, 0.33, 0.42
    rgba[..., 3] = glow * 2.2
    im = ax.imshow(rgba, extent=[0.0, 1.0, 0.0, 1.0], aspect='auto', origin='lower',
                   zorder=1, interpolation='bilinear')
    im.set_clip_path(clip)

    # ---- nested right chevrons (the >> arrow) ---------------------------
    chev_cx = 0.79
    chev_cy = 0.56
    for k in range(3):
        bx = chev_cx + k * 0.052
        h = 0.135
        w = 0.052
        alpha = [0.9, 0.55, 0.28][k]
        lw = [10, 9, 8][k]
        ln, = ax.plot([bx - w, bx, bx - w],
                      [chev_cy + h, chev_cy, chev_cy - h],
                      color=accent, lw=lw, solid_capstyle='round',
                      solid_joinstyle='round', alpha=alpha, zorder=2)
        ln.set_clip_path(clip)

    # ---- header: bias triangle + wordmark -------------------------------
    hx, hy = 0.066, 0.875
    if is_long:
        ax.add_patch(Polygon([(hx - 0.012, hy - 0.018), (hx + 0.012, hy - 0.018), (hx, hy + 0.020)],
            closed=True, facecolor=bias_color, edgecolor='none', zorder=5))
    else:
        ax.add_patch(Polygon([(hx - 0.012, hy + 0.018), (hx + 0.012, hy + 0.018), (hx, hy - 0.020)],
            closed=True, facecolor=bias_color, edgecolor='none', zorder=5))
    ax.text(hx + 0.026, hy, brand, color=WHITE, fontsize=21, fontweight='bold',
            va='center', ha='left', zorder=5, fontstyle='italic')

    # ---- avatar + username + TRADER badge -------------------------------
    av_x, av_y = 0.085, 0.715
    disc(av_x, av_y, 0.034, facecolor="#13251A", edgecolor=GREEN_DK, lw=1.2, zorder=4)
    ax.text(av_x, av_y, (uname[:1].upper() or 'T'), color=GREEN, fontsize=15,
            fontweight='bold', va='center', ha='center', zorder=5)
    name_x = av_x + 0.058
    ax.text(name_x, av_y, uname, color=WHITE, fontsize=15, fontweight='bold',
            va='center', ha='left', zorder=5)
    badge_x = name_x + 0.0150 * len(uname) + 0.024
    badge_w = 0.018 * 2 + 0.0112 * len("TRADER")
    ax.add_patch(FancyBboxPatch((badge_x, av_y - 0.027), badge_w, 0.054,
        boxstyle="round,pad=0,rounding_size=0.022",
        linewidth=0, facecolor=GOLD, zorder=4))
    ax.text(badge_x + badge_w / 2, av_y, "TRADER", color=INK, fontsize=10.5,
            fontweight='bold', va='center', ha='center', zorder=5)

    # ---- chips row ------------------------------------------------------
    def chip(x, y, label, *, fg=WHITE, border=CHIP_ED, fill=CHIP_BG, dot=None):
        pad = 0.019
        dot_w = 0.030 if dot else 0.0
        tw = 0.0132 * len(label)
        w = pad * 2 + dot_w + tw
        h = 0.058
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.024",
            linewidth=1.3, edgecolor=border, facecolor=fill, zorder=4))
        tx = x + pad
        if dot:
            disc(x + pad + 0.011, y + h / 2, 0.012, facecolor=dot, edgecolor='none', zorder=5)
            tx = x + pad + 0.030
        ax.text(tx, y + h / 2, label, color=fg, fontsize=12, fontweight='bold',
                va='center', ha='left', zorder=6)
        return x + w + 0.017

    cy = 0.560
    nx = 0.066
    nx = chip(nx, cy, f"{tri} {bias}", fg=bias_color, border=GREEN_DK, fill=GREEN_DK)
    nx = chip(nx, cy, base_sym, dot=BTC_ORANGE)
    nx = chip(nx, cy, f"{lev}x")
    nx = chip(nx, cy, exch, dot=(BYBIT_GOLD if exch == 'BYBIT' else EX_BLUE))

    # ---- Pair symbol (always white) sits directly above the PnL number --
    ax.text(0.068, 0.452, raw_sym, color="#FFFFFF", fontsize=17, fontweight='bold',
            va='center', ha='left', zorder=5)
    _td = ax.text(0.064, 0.340, headline, color=accent, fontsize=46, fontweight='bold',
                  va='center', ha='left', zorder=5)
    fig.canvas.draw()
    _bb = _td.get_window_extent(renderer=fig.canvas.get_renderer())
    _x_right = ax.transData.inverted().transform((_bb.x1, _bb.y0))[0]
    bx0 = min(_x_right + 0.022, 0.62)
    bw = 0.020 * 2 + 0.0118 * len(pct_str)
    ax.add_patch(FancyBboxPatch((bx0, 0.312), bw, 0.060,
        boxstyle="round,pad=0,rounding_size=0.022",
        linewidth=1.2, edgecolor=accent, facecolor=GREEN_DK, zorder=5))
    ax.text(bx0 + bw / 2, 0.342, pct_str, color=accent, fontsize=14, fontweight='bold',
            va='center', ha='center', zorder=6)

    # ---- footer: Entry / Exit / Current / Duration ----------------------
    def _fmt_dur(a, b):
        try:
            if a is None or b is None:
                return "\u2014"
            if isinstance(a, str):
                a = datetime.fromisoformat(a)
            if isinstance(b, str):
                b = datetime.fromisoformat(b)
            if not isinstance(a, datetime) or not isinstance(b, datetime):
                return "\u2014"
            a = a.replace(tzinfo=None); b = b.replace(tzinfo=None)
            secs = max(int((b - a).total_seconds()), 0)
            days, rem = divmod(secs, 86400)
            hours, rem = divmod(rem, 3600)
            mins = rem // 60
            if days > 0:
                return f"{days}d {hours}h {mins}m"
            if hours > 0:
                return f"{hours}h {mins}m"
            return f"{mins}m"
        except Exception:
            return "\u2014"

    peak_dur = _fmt_dur(signal.get('scan_time'), peak_at)
    fy = 0.150
    _foot = [
        ("ENTRY PRICE",   _fmt_price(entry),      WHITE),
        ("EXIT PRICE",    _fmt_price(exit_price),  accent),
        ("CURRENT PRICE", _fmt_price(cur),         WHITE),
        ("DURATION",      peak_dur,                WHITE),
    ]
    _col_x_v2 = [0.068, 0.300, 0.520, 0.760]
    for (_lbl, _val, _col), _fx in zip(_foot, _col_x_v2):
        ax.text(_fx, fy + 0.040, _lbl, color=GRAY, fontsize=9.5, fontweight='bold',
                va='center', ha='left', zorder=5)
        ax.text(_fx, fy - 0.018, _val, color=_col, fontsize=13.5, fontweight='bold',
                va='center', ha='left', zorder=5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=BG, edgecolor='none', dpi=100 * out_scale)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ──────────────────────────────────────────────
# 3D PnL card compositor (light tilt + stacked deck + glow/shadow)
# ─────────────────────────���──���────���───���────────
def _persp_coeffs(dst, src):
    """Solve the 8 perspective coefficients mapping output->input for PIL."""
    import numpy as np
    A = []
    for (dx, dy), (sx, sy) in zip(dst, src):
        A.append([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
        A.append([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
    A = np.array(A, dtype=float)
    B = np.array(dst, dtype=float).reshape(8)
    res = np.linalg.solve(A, B)
    return res.tolist()


def _dim_rgba(img, factor):
    """Multiply RGB channels by factor, keep alpha (for the dimmed back card)."""
    import numpy as np
    from PIL import Image
    arr = np.array(img).astype(float)
    arr[..., :3] *= factor
    return Image.fromarray(arr.clip(0, 255).astype('uint8'), 'RGBA')


def _radial_glow(canvas, center, radius, color, strength=0.5):
    """Alpha-composite a soft radial glow blob onto the canvas."""
    import numpy as np
    from PIL import Image
    CW, CH = canvas.size
    yy, xx = np.ogrid[:CH, :CW]
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    a = np.clip(1.0 - d / float(radius), 0, 1) ** 2
    a = (a * strength * 255).astype('uint8')
    glow = Image.new('RGBA', (CW, CH), color + (0,))
    glow.putalpha(Image.fromarray(a))
    return Image.alpha_composite(canvas, glow)


def _paste_card(canvas, card, x, y, accent, shadow=0.5, glow=0.35):
    """Paste a (warped) card sprite with an accent glow + drop shadow beneath it."""
    from PIL import Image, ImageFilter
    CW, CH = canvas.size
    cw, ch = card.size
    alpha = card.split()[3]
    if glow > 0:
        g = Image.new('RGBA', (CW, CH), (0, 0, 0, 0))
        tint = Image.new('RGBA', (cw, ch), accent + (255,))
        g.paste(tint, (x, y), alpha)
        g = g.filter(ImageFilter.GaussianBlur(max(1, int(cw * 0.06))))
        ga = g.split()[3].point(lambda a: int(a * glow))
        g.putalpha(ga)
        canvas.alpha_composite(g)
    if shadow > 0:
        s = Image.new('RGBA', (CW, CH), (0, 0, 0, 0))
        blk = Image.new('RGBA', (cw, ch), (0, 0, 0, 255))
        s.paste(blk, (x + int(cw * 0.02), y + int(ch * 0.05)), alpha)
        s = s.filter(ImageFilter.GaussianBlur(max(1, int(cw * 0.045))))
        sa = s.split()[3].point(lambda a: int(a * shadow))
        s.putalpha(sa)
        canvas.alpha_composite(s)
    layer = Image.new('RGBA', (CW, CH), (0, 0, 0, 0))
    layer.paste(card, (x, y), card)
    canvas.alpha_composite(layer)


def _warp_card(img, tilt=0.05, rot=-7.0, scale=1.0):
    """Light perspective tilt (right edge recedes) + slight rotation."""
    from PIL import Image
    if scale != 1.0:
        img = img.resize((max(1, int(img.size[0] * scale)),
                          max(1, int(img.size[1] * scale))), Image.LANCZOS)
    w, h = img.size
    dy = tilt * h
    dx = tilt * 0.6 * w
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(0, 0), (w - dx, dy), (w - dx, h - dy), (0, h)]
    coeffs = _persp_coeffs(dst, src)
    out = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    if rot:
        out = out.rotate(rot, expand=True, resample=Image.BICUBIC)
    return out


def _lerp_rgb(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _add_card_patterns(sprite, accent):
    """Bake subtle surface patterns onto the card face (clipped to its shape)."""
    from PIL import Image, ImageDraw, ImageChops
    fw, fh = sprite.size
    base_a = sprite.split()[3]
    ov = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    # faint diagonal hatch (two directions) -> circuit/topographic feel
    step = max(8, int(fw * 0.038))
    for x in range(-fh, fw, step):
        d.line([(x, 0), (x + fh, fh)], fill=accent + (9,), width=1)
    for x in range(0, fw + fh, step * 2):
        d.line([(x, 0), (x - fh, fh)], fill=(255, 255, 255, 5), width=1)
    # dotted node mesh
    dstep = max(10, int(fw * 0.042))
    for yy in range(dstep // 2, fh, dstep):
        for xx in range(dstep // 2, fw, dstep):
            d.ellipse([xx - 1, yy - 1, xx + 1, yy + 1], fill=accent + (16,))
    # soft top sheen band
    sheen = Image.new('L', (fw, fh), 0)
    sd = ImageDraw.Draw(sheen)
    band = max(1, int(fh * 0.42))
    for yy in range(band):
        a = int(26 * (1 - yy / band))
        sd.line([(0, yy), (fw, yy)], fill=a)
    ov2 = Image.new('RGBA', (fw, fh), (255, 255, 255, 0))
    ov2.putalpha(sheen)
    ov = Image.alpha_composite(ov, ov2)
    ov.putalpha(ImageChops.multiply(ov.split()[3], base_a))
    return Image.alpha_composite(sprite, ov)


def _build_card_slab(face, depth_frac=0.05, ux=0.5, uy=1.0):
    """Extrude a warped face into a 3D slab with a shaded side wall (gives real depth)."""
    from PIL import Image
    fw, fh = face.size
    D = max(6, int(depth_frac * fh))
    ex = int(D * abs(ux)) + 4
    ey = int(D * uy) + 4
    slab = Image.new('RGBA', (fw + ex, fh + ey), (0, 0, 0, 0))
    alpha = face.split()[3]
    top_col = (34, 44, 55)
    bot_col = (5, 8, 12)
    for i in range(D, 0, -1):
        t = i / D
        col = _lerp_rgb(top_col, bot_col, t)
        layer = Image.new('RGBA', (fw, fh), col + (0,))
        layer.putalpha(alpha)
        slab.alpha_composite(layer, (int(i * ux), int(i * uy)))
    slab.alpha_composite(face, (0, 0))
    return slab


def render_pnl_card_tablet_v3(signal, current_price, leverage, capital=None, closes=None,
                               peak_raw_pct=None, peak_at=None, peak_loss_pct=None, peak_loss_at=None,
                               username=None, out_scale=1.0):
    """Render the 'tablet' SAKZ PnL card (PNG bytes) — the third display style.

    Dark glass-panel aesthetic: portrait-leaning layout with a bold header bar,
    a large avatar circle with initials, pill chips row, the vault PnL headline
    with inline % badge, a subtle arc/curve decoration on the right, and a
    clean four-column footer row. Inputs and value semantics mirror the other
    two card renderers so all three always show identical numbers.
    """
    import io
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, Arc, Circle

    # ── palette (dark glass, silver edge, green/red accent) ──────────────
    BG        = "#05070A"
    CARD      = "#090C10"
    CARD_ED   = "#1A2330"
    HEADER_BG = "#0D1118"
    CHIP_BG   = "#0F1520"
    CHIP_ED   = "#1E2C3A"
    GREEN     = "#2FD675"
    GREEN_DK  = "#0D3B22"
    RED       = "#F0556B"
    RED_DK    = "#3D1119"
    WHITE     = "#FFFFFF"
    SOFT      = "#A8B4C0"
    GRAY      = "#6A7888"
    GOLD      = "#F2C744"
    INK       = "#08090B"
    SILVER    = "#8A96A4"
    BTC_ORANGE = "#F7931A"
    BYBIT_GOLD = "#F7A600"
    EX_BLUE    = "#2E6BFF"

    # Landscape card — same proportions as the tablet template (≈10.24 × 6.83)
    ASPECT = 10.24 / 6.83
    W_IN = 10.24
    H_IN = 6.83
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=100, facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    def disc(x, y, r, **kw):
        ax.add_patch(Ellipse((x, y), width=2 * r / ASPECT, height=2 * r, **kw))

    # ── derive values (identical logic to the other two renderers) ────────
    brand    = os.environ.get("BOT_NAME", "Sakz")
    entry    = float(signal.get('price') or 0) or float(current_price or 0)
    bias     = str(signal.get('bias', 'LONG')).upper()
    is_long  = bias == 'LONG'
    cur      = float(current_price or 0)
    lev      = int(leverage or 1)

    if entry > 0 and cur > 0:
        cur_raw = ((cur - entry) / entry * 100) if is_long else ((entry - cur) / entry * 100)
    else:
        cur_raw = 0.0
    fav_raw  = peak_raw_pct if peak_raw_pct is not None else cur_raw
    if fav_raw < cur_raw:
        fav_raw = cur_raw
    peak_lev = fav_raw * lev
    if peak_lev < -100.0:
        peak_lev = -100.0

    up = peak_lev >= 0
    if not up:
        GREEN, GREEN_DK = RED, RED_DK
    accent     = GREEN
    bias_color = "#2FD675" if is_long else "#F0556B"
    tri        = "\u25B2" if is_long else "\u25BC"

    raw_sym  = str(signal.get('symbol', '')).upper().replace('/', '').replace('_', '')
    base_sym = raw_sym[:-4] if raw_sym.endswith('USDT') else raw_sym
    exch     = (str(signal.get('exchange', '')).upper() or 'MEXC')

    if entry > 0 and fav_raw is not None:
        exit_price = entry * (1 + fav_raw / 100.0) if is_long else entry * (1 - fav_raw / 100.0)
    else:
        exit_price = cur or entry

    def _fmt_price(p):
        if not p:
            return "\u2014"
        p = float(p)
        if p >= 1000:
            return f"{p:,.2f}"
        if p >= 1:
            return f"{p:,.4f}"
        return f"{p:.6f}"

    pct_str    = f"{peak_lev:+.2f}%"
    show_amount = (capital is not None) and (capital > 0)
    dollar_str  = None
    if show_amount:
        dval = capital * peak_lev / 100.0
        if abs(dval) >= 100:
            dollar_str = f"{'+' if dval >= 0 else '-'}${abs(dval):,.0f}"
        else:
            dollar_str = f"{'+' if dval >= 0 else '-'}${abs(dval):,.2f}"

    uname = str(username or signal.get('username') or "Trader").lstrip('@')
    if len(uname) > 16:
        uname = uname[:15] + "\u2026"

    def _fmt_dur(a, b):
        try:
            if a is None or b is None:
                return "\u2014"
            if isinstance(a, str):
                a = datetime.fromisoformat(a)
            if isinstance(b, str):
                b = datetime.fromisoformat(b)
            if not isinstance(a, datetime) or not isinstance(b, datetime):
                return "\u2014"
            a = a.replace(tzinfo=None); b = b.replace(tzinfo=None)
            secs = max(int((b - a).total_seconds()), 0)
            days, rem = divmod(secs, 86400)
            hours, rem = divmod(rem, 3600)
            mins = rem // 60
            if days > 0:
                return f"{days}d {hours}h {mins}m"
            if hours > 0:
                return f"{hours}h {mins}m"
            return f"{mins}m"
        except Exception:
            return "\u2014"

    peak_dur = _fmt_dur(signal.get('scan_time'), peak_at)

    # ── SAKZ_CARD3_V2_MOCKUP — clean continuous glass body (no header bar) ──
    ax.add_patch(FancyBboxPatch((0.028, 0.040), 0.944, 0.920,
        boxstyle="round,pad=0,rounding_size=0.052",
        linewidth=1.6, edgecolor=CARD_ED, facecolor=CARD, zorder=1))
    ax.add_patch(FancyBboxPatch((0.028, 0.040), 0.944, 0.920,
        boxstyle="round,pad=0,rounding_size=0.052",
        linewidth=3.0, edgecolor=SILVER, facecolor="none", alpha=0.16, zorder=2))

    # faint sweeping chart curve + nested chevrons on the right
    _cx = np.linspace(0.50, 0.95, 60)
    _cy = 0.30 + 0.34 * (((_cx - 0.50) / 0.45) ** 1.6) + 0.015 * np.sin((_cx - 0.50) * 22)
    for _w, _a in [(4.5, 0.06), (2.4, 0.13), (1.3, 0.28)]:
        ax.plot(_cx, _cy, color=accent, lw=_w, alpha=_a, solid_capstyle="round", zorder=2)
    for _k in range(4):
        _o = 0.022 * _k
        ax.add_patch(FancyBboxPatch((0.60 + _o, 0.28 + _o), 0.32 - 2 * _o, 0.40 - 2 * _o,
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=1.0, edgecolor=SILVER, facecolor="none", alpha=0.06, zorder=1))

    # soft radial glow behind the headline
    gx = np.linspace(0, 1, 200); gy = np.linspace(0, 1, 200)
    GXX, GYY = np.meshgrid(gx, gy)
    dist = np.sqrt((GXX - 0.34) ** 2 + (GYY - 0.44) ** 2)
    glow = np.clip(0.22 - dist * 0.55, 0, 0.22)
    rgba = np.zeros((200, 200, 4))
    if up:
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.18, 0.84, 0.46
    else:
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.94, 0.33, 0.42
    rgba[..., 3] = glow * 1.8
    ax.imshow(rgba, extent=[0.03, 0.97, 0.03, 0.97], aspect="auto", origin="lower",
              zorder=2, interpolation="bilinear")

    # brand wordmark (triangle + Sakz) top-left
    bx0, by0 = 0.082, 0.886
    ax.add_patch(Polygon([(bx0 - 0.012, by0 - 0.014), (bx0 + 0.012, by0 - 0.014),
                          (bx0, by0 + 0.018)], closed=True,
                         facecolor=accent, edgecolor="none", zorder=6))
    ax.text(bx0 + 0.026, by0, brand, color=WHITE, fontsize=21, fontweight="bold",
            va="center", ha="left", zorder=6, fontstyle="italic")

    # avatar circle + username
    av_x, av_y = 0.099, 0.762
    disc(av_x, av_y, 0.034, facecolor="#182535", edgecolor=CARD_ED, lw=1.4, zorder=4)
    ax.text(av_x, av_y, (uname[:1].upper() or "T"), color=WHITE, fontsize=15,
            fontweight="bold", va="center", ha="center", zorder=5)
    ax.text(av_x + 0.060, av_y, uname, color=WHITE, fontsize=15, fontweight="bold",
            va="center", ha="left", zorder=5)

    # chip helper
    def chip(x, y, label, *, fg=WHITE, border=CHIP_ED, fill=CHIP_BG,
             dot=None, dot_glyph=None, logo=None, h=0.054, fs=12):
        pad   = 0.018
        has_badge = (logo is not None) or bool(dot)
        dot_w = 0.030 if has_badge else 0.0
        tw    = 0.0140 * len(label)
        w     = pad * 2 + dot_w + tw
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.020",
            linewidth=1.3, edgecolor=border, facecolor=fill, zorder=4))
        tx = x + pad
        if has_badge:
            bxc = x + pad + 0.012
            if logo is not None:
                _draw_token_logo(ax, bxc, y + h / 2, 0.0155, logo, ASPECT, zorder=7)
            else:
                disc(bxc, y + h / 2, 0.013, facecolor=dot, edgecolor="none", zorder=5)
                if dot_glyph:
                    ax.text(bxc, y + h / 2, dot_glyph, color=WHITE,
                            fontsize=8, fontweight="bold", va="center", ha="center", zorder=6)
            tx = x + pad + 0.030
        ax.text(tx, y + h / 2, label, color=fg, fontsize=fs, fontweight="bold",
                va="center", ha="left", zorder=6)
        return x + w + 0.015

    sym_glyph = base_sym[:1] or "?"   # font lacks the bitcoin glyph; orange dot conveys it
    _tok_logo = fetch_token_logo(base_sym)   # real logo (None -> lettered fallback)

    # chips row 1: bias . asset . Trader badge
    cy1 = 0.650
    nx = 0.082
    nx = chip(nx, cy1, f"{tri} {bias}", fg=bias_color, border=bias_color, fill=CHIP_BG)
    nx = chip(nx, cy1, base_sym, dot=BTC_ORANGE, dot_glyph=sym_glyph, logo=_tok_logo)
    nx = chip(nx, cy1, "Trader", fg=INK, border=GOLD, fill=GOLD)

    # My Vault PnL headline
    ax.text(0.082, 0.552, "My Vault PnL", color=SOFT, fontsize=13, fontweight="bold",
            va="center", ha="left", zorder=5)
    if show_amount:
        _td = ax.text(0.078, 0.456, dollar_str, color=accent, fontsize=44, fontweight="bold",
                      va="center", ha="left", zorder=5)
        fig.canvas.draw()
        _bb  = _td.get_window_extent(renderer=fig.canvas.get_renderer())
        _x_r = ax.transData.inverted().transform((_bb.x1, _bb.y0))[0]
        ax.text(min(_x_r + 0.020, 0.64), 0.444, pct_str, color=accent, fontsize=19,
                fontweight="bold", va="center", ha="left", zorder=5)
    else:
        ax.text(0.078, 0.456, pct_str, color=accent, fontsize=46, fontweight="bold",
                va="center", ha="left", zorder=5)

    # chips row 2: asset . leverage . exchange
    cy2 = 0.322
    nx = 0.082
    nx = chip(nx, cy2, base_sym, dot=BTC_ORANGE, dot_glyph=sym_glyph, logo=_tok_logo, h=0.050, fs=11)
    nx = chip(nx, cy2, f"{lev}x", h=0.050, fs=11)
    nx = chip(nx, cy2, exch, dot=(BYBIT_GOLD if exch == "BYBIT" else EX_BLUE),
              dot_glyph=exch[:1], h=0.050, fs=11)

    # staggered footer prices (diagonal, following the curve, like the mockup)
    _foot = [
        ("Entry Price",   _fmt_price(entry),     WHITE,  0.086, 0.232),
        ("Exit Price",    _fmt_price(exit_price), accent, 0.312, 0.196),
        ("Current Price", _fmt_price(cur),        WHITE,  0.548, 0.158),
        ("Duration",      peak_dur,               WHITE,  0.782, 0.120),
    ]
    for _lbl, _val, _col, _fx, _fy in _foot:
        ax.text(_fx, _fy + 0.040, _lbl, color=GRAY, fontsize=10, fontweight="bold",
                va="center", ha="left", zorder=5)
        ax.text(_fx, _fy - 0.008, _val, color=_col, fontsize=14, fontweight="bold",
                va="center", ha="left", zorder=5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=BG, edgecolor='none', dpi=100 * out_scale)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def compose_3d_card(flat_png, up=True):
    """Turn a flat PnL card PNG into a realistic 3D render: extruded card
    thickness (depth), baked surface patterns, a stacked deck card behind,
    floor reflection, accent glow and drop shadow on a dark backdrop.
    Returns PNG bytes.
    """
    import io as _io
    from PIL import Image, ImageDraw, ImageFilter, ImageChops

    ACCENT = (47, 214, 117) if up else (240, 85, 107)
    BG = (5, 7, 10)

    card = Image.open(_io.BytesIO(flat_png)).convert('RGBA')
    W, H = card.size

    # crop to the card face (FancyBboxPatch at 0.035,0.06 w0.93 h0.88)
    left   = int(round(0.035 * W)); right  = int(round(0.965 * W))
    top    = int(round(0.060 * H)); bottom = int(round(0.940 * H))
    sprite = card.crop((left, top, right, bottom))
    sw, sh = sprite.size

    # rounded-corner alpha so the warp has clean transparent corners
    rad = max(1, int(0.055 * sw))
    mask = Image.new('L', (sw, sh), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, sw - 1, sh - 1], radius=rad, fill=255)
    sprite.putalpha(mask)
    sprite = _add_card_patterns(sprite, ACCENT)

    front = _build_card_slab(_warp_card(sprite, tilt=0.07, rot=-7.0, scale=1.0))
    back  = _build_card_slab(_dim_rgba(_warp_card(sprite, tilt=0.07, rot=-11.0, scale=0.93), 0.62))

    fw, fh = front.size
    bw, bh = back.size
    # Tighter canvas → the deck fills more of the frame (cards appear bigger).
    CW = int(fw * 1.42); CH = int(fh * 1.52)
    canvas = Image.new('RGBA', (CW, CH), BG + (255,))

    # Center the front+back deck within the canvas, biased slightly upward so
    # the floor reflection still has room below.
    bdx = int(fw * 0.17)            # back card x-offset (to the right) from front
    bdy = int(fh * 0.14)            # back card y-offset (upward) from front
    span_w = max(fw, bdx + bw)      # union width of the stacked deck
    span_h = bdy + fh               # union height (back top → front bottom)
    fx = int((CW - span_w) / 2)
    fy = int((CH - span_h) / 2) + bdy - int(fh * 0.04)
    bx = fx + bdx; by = fy - bdy

    canvas = _radial_glow(canvas, center=(int(CW * 0.60), int(CH * 0.42)),
                          radius=int(CW * 0.50), color=ACCENT, strength=0.5)

    # floor reflection of the front card
    refl = front.transpose(Image.FLIP_TOP_BOTTOM)
    rfade = Image.new('L', refl.size, 0)
    rd = ImageDraw.Draw(rfade)
    rh = refl.size[1]
    for yy in range(rh):
        rd.line([(0, yy), (refl.size[0], yy)], fill=int(60 * (1 - yy / rh)))
    refl.putalpha(ImageChops.multiply(refl.split()[3], rfade))
    refl = refl.filter(ImageFilter.GaussianBlur(6))
    rl = Image.new('RGBA', (CW, CH), (0, 0, 0, 0))
    rl.paste(refl, (fx, fy + fh - int(fh * 0.06)), refl)
    canvas.alpha_composite(rl)

    _paste_card(canvas, back,  bx, by, ACCENT, shadow=0.30, glow=0.22)
    _paste_card(canvas, front, fx, fy, ACCENT, shadow=0.55, glow=0.42)

    out = _io.BytesIO()
    canvas.convert('RGB').save(out, format='PNG')
    return out.getvalue()


def _pnl_symbol_base(sym):
    """Normalise a symbol to its base (strip separators + USDT quote)."""
    q = str(sym or '').upper().replace('/', '').replace('_', '')
    return q[:-4] if q.endswith('USDT') else q


def _pnl_lifecycle_mode(signal):
    """Resolve how /pnl should render for a signal, honouring the memory model.

    Returns (mode, record) where mode is one of 'not_recent' | 'loss_at_sl' |
    'normal'. A pair that was evicted from memory and not re-scanned since is
    'not_recent'; a pair that hit its stop-loss is 'loss_at_sl'.
    """
    try:
        exch = signal.get('exchange', '')
        sym  = signal.get('symbol', '')
        base = _pnl_symbol_base(sym)
        record = db_get_active_signal(exch, sym)
        evic   = db_get_eviction(exch, sym) or db_find_eviction_by_symbol(base)
        removed_at = evic.get('removed_at') if evic else None
        latest_scan = (record or {}).get('first_scan_time') or signal.get('scan_time')
        mode = _SIGLOGIC.pnl_card_mode(record or signal, latest_scan, removed_at)
        return mode, record
    except Exception as e:
        logger.debug("_pnl_lifecycle_mode failed: %s", e)
        return 'normal', None


def _build_sl_loss_card(signal, record=None):
    """Text PnL card for a stopped-out pair: realized loss at the bot's SL and
    suggested leverage, using the original signal the bot gave."""
    src = record or signal
    bias  = str(src.get('bias', '')).upper()
    entry = float(src.get('entry') or src.get('price') or signal.get('price') or 0)
    sl    = float(src.get('stop_loss') or signal.get('stop_loss') or 0)
    exch  = src.get('exchange', '') or signal.get('exchange', '')
    sym   = src.get('symbol', '') or signal.get('symbol', '')
    lev   = src.get('leverage', signal.get('leverage'))
    if isinstance(lev, dict):
        lev = lev.get('suggested')
    bot_lev = float(lev or 10)
    loss_pct = _SIGLOGIC.loss_at_sl_pct(bias, entry, sl, bot_lev)
    bias_emoji = "��" if bias == 'LONG' else "🔴"
    return (
        f"💥 PNL CARD — {exch} | {sym}\n"
        f"{'━'*30}\n"
        f"{bias_emoji} {bias}  ·  STOPPED OUT 🛑\n\n"
        f"This pair reversed against the call and hit the bot's stop-loss.\n\n"
        f"📥 Entry:        ${entry:.6f}\n"
        f"🛑 Stop-loss:    ${sl:.6f}  (bot's SL)\n"
        f"⚡ Leverage:    {bot_lev:.0f}x  (bot suggested)\n"
        f"{'━'*30}\n"
        f"📉 Realized PnL: {loss_pct:+.2f}%\n\n"
        f"🔗 {get_exchange_link(exch, sym)}"
    )


async def prompt_pnl_display_mode(update, context):
    """Pop up leverage + display-mode buttons after a signal is picked.

    Lets the user adjust leverage and choose whether to show a profit $ amount
    (which then asks for capital) or just the percentage. Picking either path
    renders the vault card via send_pnl_image_card.
    """
    signal = context.user_data.get('pnl_signal')
    tgt = update.effective_message
    if not signal:
        await tgt.reply_text("\u26a0\ufe0f Signal data lost. Run /pnl again.")
        return

    # ALTERNATE the card style on every /pnl query so the two designs rotate
    # automatically (no Refresh tap needed): query 1 -> 🃏 3D deck (0),
    # query 2 -> 🟩 flat terminal (1), query 3 -> 3D deck (0), and so on. The
    # seed persists per-user in user_data and is applied to the normal,
    # stopped-out, and not-recent render paths alike.
    _seed = (int(context.user_data.get('pnl_style_seed', -1)) + 1) % 2
    context.user_data['pnl_style_seed'] = _seed
    context.user_data['pnl_card_style'] = _seed

    # MEMORY-MODEL GUARD — honour evictions + stop-loss before rendering.
    mode, record = _pnl_lifecycle_mode(signal)
    if mode == 'not_recent':
        raw_sym = str(signal.get('symbol', '')).replace('_USDT', 'USDT')
        await tgt.reply_text(
            f"⚠\ufe0f {raw_sym} wasn't scanned recently — its last call rolled off "
            f"the bot's memory. Run /scan (or scan the pair) to get a fresh "
            f"signal before pulling a PnL."
        )
        return
    if mode == 'loss_at_sl':
        # Render the stopped-out card as a branded image (matching the other
        # PnL cards), pinned to the bot's SL + suggested leverage. Fall back to
        # the text card if rendering isn't available.
        try:
            _u = update.effective_user
            username = (_u.username or _u.first_name or "Trader") if _u else "Trader"
        except Exception:
            username = "Trader"
        lev = (record or {}).get('leverage', signal.get('leverage'))
        if isinstance(lev, dict):
            lev = lev.get('suggested')
        bot_lev = int(float(lev or 10))
        _style = int(context.user_data.get('pnl_card_style', 0)) % 2
        try:
            loop = asyncio.get_running_loop()
            png = await loop.run_in_executor(
                RENDER_EXECUTOR,
                _build_sl_loss_card_png, signal, bot_lev, username, _style)
            raw_sym = str(signal.get('symbol', '')).replace('_USDT', 'USDT')
            caption = (
                f"💥 {signal.get('exchange', '')} {raw_sym} — STOPPED OUT 🛑\n"
                f"Loss shown at the bot's stop-loss and {bot_lev}x suggested leverage."
            )
            await tgt.reply_photo(photo=io.BytesIO(png), caption=caption)
        except Exception as e:
            logger.warning("SL-loss image card render failed, using text card: %s", e)
            await tgt.reply_text(_build_sl_loss_card(signal, record))
        return
    lev_data = signal.get('leverage')
    sugg = lev_data['suggested'] if lev_data else 10
    context.user_data.setdefault('pnl_leverage', sugg)
    context.user_data['pnl_capital'] = None
    # pnl_card_style is set above and ALTERNATES on each /pnl query — do not reset it here.
    cur_lev = int(context.user_data.get('pnl_leverage', sugg))

    def _lvb(n):
        mark = "✅ " if int(n) == cur_lev else ""
        return InlineKeyboardButton(f"{mark}{n}x", callback_data=f"pnlimg_lev|{n}")

    raw_sym = str(signal.get('symbol', '')).replace('_USDT', 'USDT')
    bias = signal.get('bias', '')
    keyboard = InlineKeyboardMarkup([
        [_lvb(5), _lvb(10), _lvb(20), _lvb(50),
         InlineKeyboardButton("✏️", callback_data="pnlimg_lev|custom")],
        [InlineKeyboardButton("💵 Show profit $", callback_data="pnlimg_mode|usdt"),
         InlineKeyboardButton("📊 % only", callback_data="pnlimg_mode|pct")],
    ])
    await tgt.reply_text(
        f"📈 PnL for {signal.get('exchange', '')} {raw_sym} — {bias}\n"
        f"Suggested leverage: {sugg}x  (tap to change)\n\n"
        f"Choose how to show your PnL:",
        reply_markup=keyboard,
    )


class _PnLPriceUnavailable(Exception):
    """Raised inside the render worker when no live price is available."""
    pass


def _build_pnl_card_png(signal, leverage, capital, username, style):
    """Blocking PnL pipeline (live price + chart closes + excursions + matplotlib
    render). Runs in RENDER_EXECUTOR so the asyncio event loop is never blocked
    while a card image is generated."""
    current = _get_live_price(signal['symbol'], signal.get('exchange', ''))
    if not current or current <= 0:
        raise _PnLPriceUnavailable()
    closes = _fetch_pnl_chart_closes(signal, current)
    fav_raw, fav_at, adv_raw, adv_at = _compute_peak_excursions(signal, current)
    _fav = fav_raw if fav_raw is not None else 0.0
    up_flag = (_fav * (leverage or 1)) >= 0
    # Network prep above runs in parallel across render workers; only the actual
    # matplotlib drawing below is serialized via _PLT_LOCK (pyplot isn't
    # thread-safe). This keeps multi-user PnL requests fast without corruption.
    if style == 1:
        with _PLT_LOCK:
            return render_pnl_card_flat_v2(signal, current, leverage, capital, closes,
                                           peak_raw_pct=fav_raw, peak_at=fav_at,
                                           peak_loss_pct=adv_raw, peak_loss_at=adv_at,
                                           username=username, out_scale=2.0)
    with _PLT_LOCK:
        flat = render_pnl_card_image(signal, current, leverage, capital, closes,
                                     peak_raw_pct=fav_raw, peak_at=fav_at,
                                     peak_loss_pct=adv_raw, peak_loss_at=adv_at,
                                     username=username, out_scale=2.0)
        try:
            return compose_3d_card(flat, up=up_flag)
        except Exception as _e3d:
            logger.warning("3D compose failed, using flat card: %s", _e3d)
            return flat


def _build_sl_loss_card_png(signal, leverage, username, style):
    """Blocking render of a STOPPED-OUT PnL card. Identical pipeline to
    _build_pnl_card_png, but the exit/current price is pinned to the bot's
    stop-loss so the card shows the realized loss at the bot's SL and suggested
    leverage (instead of a live price). Runs in RENDER_EXECUTOR."""
    sl = float(signal.get('stop_loss') or 0)
    if sl <= 0:
        raise _PnLPriceUnavailable()
    closes = _fetch_pnl_chart_closes(signal, sl)
    fav_raw, fav_at, adv_raw, adv_at = _compute_peak_excursions(signal, sl)
    if style == 1:
        with _PLT_LOCK:
            return render_pnl_card_flat_v2(signal, sl, leverage, None, closes,
                                           peak_raw_pct=fav_raw, peak_at=fav_at,
                                           peak_loss_pct=adv_raw, peak_loss_at=adv_at,
                                           username=username, out_scale=2.0)
    with _PLT_LOCK:
        flat = render_pnl_card_image(signal, sl, leverage, None, closes,
                                     peak_raw_pct=fav_raw, peak_at=fav_at,
                                     peak_loss_pct=adv_raw, peak_loss_at=adv_at,
                                     username=username, out_scale=2.0)
        try:
            return compose_3d_card(flat, up=False)
        except Exception as _e3d:
            logger.warning("3D compose failed for SL-loss card, using flat: %s", _e3d)
            return flat


async def send_pnl_image_card(update, context):
    """Fetch live price, render the branded PnL card image, and send it with buttons."""
    msg = update.effective_message
    signal = context.user_data.get('pnl_signal')
    if not signal:
        await msg.reply_text("⚠�� Signal data lost. Run /pnl again.")
        return

    capital  = context.user_data.get('pnl_capital')
    lev_data = signal.get('leverage')
    sugg_lev = lev_data['suggested'] if lev_data else 10
    leverage = context.user_data.get('pnl_leverage') or sugg_lev

    try:
        _u = update.effective_user
        username = (_u.username or _u.first_name or "Trader") if _u else "Trader"
    except Exception:
        username = "Trader"

    _style = int(context.user_data.get('pnl_card_style', 0)) % 2
    loop = asyncio.get_running_loop()
    try:
        png = await loop.run_in_executor(
            RENDER_EXECUTOR,
            _build_pnl_card_png, signal, leverage, capital, username, _style)
    except _PnLPriceUnavailable:
        await msg.reply_text("⚠️ Couldn't fetch the live price right now. Try again in a moment.")
        return
    except Exception as e:
        logger.exception("PnL card render failed")
        await msg.reply_text(f"⚠️ Couldn't render the PnL card: {e}")
        return

    raw_sym  = str(signal.get('symbol', '')).replace('_USDT', 'USDT')
    cap_note = f" • capital ${capital:,.0f}" if capital else ""
    _style_now = int(context.user_data.get('pnl_card_style', 0)) % 2
    style_note = ("🃏 3D deck" if _style_now == 0
                  else "🟩 Flat card")
    caption  = (f"📈 Live PnL • {signal.get('exchange', '')} {raw_sym} • {leverage}x{cap_note}\n"
                f"{style_note} • 🔄 Refresh updates the price & flips the card")

    _cur_lev = int(leverage)
    def _lvb(n):
        mark = "✅ " if int(n) == _cur_lev else ""
        return InlineKeyboardButton(f"{mark}{n}x", callback_data=f"pnlimg_lev|{n}")
    if capital:
        amt_btn = InlineKeyboardButton("📊 % only", callback_data="pnlimg_mode|pct")
    else:
        amt_btn = InlineKeyboardButton("💵 Show profit $", callback_data="pnlimg_mode|usdt")
    keyboard = InlineKeyboardMarkup([
        [_lvb(5), _lvb(10), _lvb(20), _lvb(50),
         InlineKeyboardButton("✏️", callback_data="pnlimg_lev|custom")],
        [amt_btn, InlineKeyboardButton("🔄 Refresh", callback_data="pnlimg_refresh")],
        [InlineKeyboardButton("✅ Done", callback_data="pnlimg_done")],
    ])
    await msg.reply_photo(photo=io.BytesIO(png), caption=caption, reply_markup=keyboard)


def _resolve_pnl_signal(arg: str, results: list):
    """Resolve a /pnl argument to a signal from the last scan.

    Accepts either a 1-based rank number (e.g. "1") or a symbol as it was
    shown on the bot (e.g. "btc", "btcusdt", "ton"). Returns the matching
    signal dict, or None if nothing matched.
    """
    arg = (arg or '').strip()
    if not arg:
        return None

    # Numeric rank: /pnl 1
    if arg.isdigit():
        n = int(arg)
        if 1 <= n <= len(results):
            return results[n - 1]
        return None

    # Symbol match: /pnl btc  (normalise away separators + optional USDT quote)
    def _norm(s):
        return str(s or '').upper().replace('/', '').replace('_', '')
    q      = _norm(arg)
    q_full = q if q.endswith('USDT') else q + 'USDT'

    # 1) exact symbol match (with or without the USDT quote)
    for r in results:
        sym = _norm(r.get('symbol'))
        if sym == q_full or sym == q:
            return r
    # 2) prefix match on the base symbol (e.g. "ton" -> TONCOINUSDT)
    for r in results:
        if _norm(r.get('symbol')).startswith(q):
            return r
    return None


def _gather_pnl_matches(arg, results):
    """
    Find every scanned signal for a symbol — from the current scan AND from the
    persisted scan history — so users can pull a PnL for a past call even if it
    rolled off the bot or has since reversed direction.

    Returns a de-duplicated list, current-scan matches first, then historical
    matches newest-first.
    """
    def _norm(s):
        return str(s or '').upper().replace('/', '').replace('_', '')

    q = _norm(arg)
    if not q:
        return []
    q_base = q[:-4] if q.endswith('USDT') else q

    matches = []
    seen    = set()

    def _key(sig):
        st = sig.get('scan_time')
        st = st.isoformat() if hasattr(st, 'isoformat') else str(st)
        return (str(sig.get('exchange')), _norm(sig.get('symbol')), st)

    # 1) Current scan results (freshest)
    for r in (results or []):
        sym = _norm(r.get('symbol'))
        if sym == q or sym == q_base + 'USDT' or sym.startswith(q_base):
            k = _key(r)
            if k not in seen:
                seen.add(k)
                matches.append(r)

    # 2) Persisted scan history
    try:
        for r in db_find_signals_by_symbol(q_base):
            k = _key(r)
            if k not in seen:
                seen.add(k)
                matches.append(r)
    except Exception as e:
        logger.warning("_gather_pnl_matches DB lookup failed: %s", e)

    # Option A — collapse to the FIRST (earliest) call per exchange+symbol+bias.
    # The autoscanner re-saves a fresh signal every time it re-detects a pair, so
    # without this users see the same call repeated at many timestamps. We keep
    # one clean card per direction, anchored to the original (first) call.
    def _ts(sig):
        st = sig.get('scan_time')
        try:
            if hasattr(st, 'isoformat'):
                st = st.isoformat()
            if isinstance(st, str) and st:
                return datetime.fromisoformat(st).replace(tzinfo=None)
        except Exception:
            pass
        return datetime.max

    grouped = {}
    for r in matches:
        gk = (str(r.get('exchange')), _norm(r.get('symbol')), str(r.get('bias')).upper())
        keep = grouped.get(gk)
        if keep is None or _ts(r) < _ts(keep):
            grouped[gk] = r

    collapsed = list(grouped.values())
    collapsed.sort(key=_ts, reverse=True)   # most-recent first call first
    return collapsed


async def pnl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entry point — pick a signal, or look one up directly via /pnl <rank|symbol>."""
    _track(update)

    # Starting /pnl cancels any pending autoscan TF entry so the dispatch chain
    # doesn't mistake a PnL reply (e.g. "6") for a timeframe.
    try:
        _autoscan_awaiting_tf.discard(update.effective_chat.id)
    except Exception as e:
        logger.debug("pnl_command: could not clear autoscan TF state: %s", e)

    results = state.last_scan_results or []

    # Direct lookup: /pnl 1   or   /pnl btc   (symbol works even with no current scan)
    args = context.args or []
    if args:
        arg0 = args[0].strip()

        # Numeric rank → pick from the current scan only.
        if arg0.isdigit():
            if not results:
                await update.message.reply_text("⚠️ No scan data yet. Run /scan first, or use /pnl <symbol>.")
                return
            sig = _resolve_pnl_signal(arg0, results)
            if sig is None:
                await update.message.reply_text(f"⚠️ Enter a number between 1 and {len(results)}.")
                return
            context.user_data['pnl_signal']  = sig
            context.user_data['pnl_capital'] = None
            context.user_data['pnl_step']    = None
            await prompt_pnl_display_mode(update, context)
            return

        # Symbol → search current scan AND persisted history (incl. reversed calls).
        gathered = _gather_pnl_matches(arg0, results)
        if not gathered:
            await update.message.reply_text(
                f"⚠️ Couldn't find any scanned signal for \"{arg0}\".\n"
                f"It has to have been scanned on the chart at least once."
            )
            return
        # The /pnl for a pair anchors to the OLDEST (first) recorded signal,
        # regardless of how many times it was subsequently re-called.
        sig = _SIGLOGIC.pick_oldest_signal(gathered) or gathered[0]
        context.user_data['pnl_signal']  = sig
        context.user_data['pnl_capital'] = None
        context.user_data['pnl_step']    = None
        await prompt_pnl_display_mode(update, context)
        return

    if not results:
        await update.message.reply_text("⚠️ No scan data yet. Run /scan first.")
        return

    total = len(state.last_scan_results)
    lines = [f"��� PNL CALCULATOR\n\nPick a signal:\n"]
    for i, r in enumerate(state.last_scan_results[:20], 1):
        emoji  = "🟢" if r['bias'] == "LONG" else "🔴"
        lev    = r.get('leverage')
        lev_str = f" | {lev['suggested']}x" if lev else ""
        lines.append(f"{i}. {emoji} {r['exchange']} {r['symbol']} — {r['bias']} {r['confidence']}/10{lev_str}")
    if total > 20:
        lines.append(f"... and {total-20} more")
    lines.append(f"\nReply with a number (1–{min(total,20)}), then I'll ask your capital.")

    context.user_data['pnl_step'] = 'pick'
    await update.message.reply_text("\n".join(lines))

async def pnl_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the multi-step PnL conversation via plain messages."""
    step = context.user_data.get('pnl_step')
    if not step:
        return  # not in a PnL flow — ignore

    text = update.message.text.strip()

    if step == 'pick':
        try:
            n = int(text)
            if n < 1 or n > len(state.last_scan_results):
                await update.message.reply_text(f"⚠️ Enter 1–{len(state.last_scan_results)}.")
                return
            context.user_data['pnl_signal']  = state.last_scan_results[n - 1]
            context.user_data['pnl_capital'] = None
            context.user_data['pnl_step']    = None   # display-mode prompt takes over
            await prompt_pnl_display_mode(update, context)
        except ValueError:
            await update.message.reply_text("⚠️ Reply with a number only.")

    elif step == 'pnl_pick_match':
        matches = context.user_data.get('pnl_matches') or []
        try:
            n = int(text)
            if n < 1 or n > len(matches):
                await update.message.reply_text(f"⚠️ Enter 1–{len(matches)}.")
                return
            context.user_data['pnl_signal']  = matches[n - 1]
            context.user_data['pnl_capital'] = None
            context.user_data['pnl_step']    = None
            await prompt_pnl_display_mode(update, context)
        except ValueError:
            await update.message.reply_text("⚠️ Reply with a number only.")

    elif step == 'pnlimg_capital':
        try:
            capital = float(text.replace(',', ''))
            if capital <= 0:
                await update.message.reply_text("⚠️ Capital must be greater than 0.")
                return
            context.user_data['pnl_capital'] = capital
            context.user_data['pnl_step']    = None
            await send_pnl_image_card(update, context)
        except ValueError:
            await update.message.reply_text("⚠️ Enter a valid number (e.g. 100 or 500).")

    elif step == 'pnlimg_lev_custom':
        try:
            lv = int(text.replace('x', '').strip())
            if lv < 1 or lv > 125:
                await update.message.reply_text("⚠️ Leverage must be 1–125.")
                return
            context.user_data['pnl_leverage'] = lv
            context.user_data['pnl_step']     = None
            await send_pnl_image_card(update, context)
        except ValueError:
            await update.message.reply_text("⚠️ Enter a number like 10 or 25.")

    elif step == 'capital':
        try:
            capital = float(text.replace(',', ''))
            if capital <= 0:
                await update.message.reply_text("⚠️ Capital must be greater than 0.")
                return
            context.user_data['pnl_capital'] = capital
            context.user_data['pnl_step']    = None  # clear step — buttons take over

            signal  = context.user_data['pnl_signal']
            lev     = signal.get('leverage')
            bot_lev = lev['suggested'] if lev else None

            # Build inline keyboard
            buttons = []
            if bot_lev:
                buttons.append(InlineKeyboardButton(
                    f"⚡ Use bot leverage ({bot_lev}x)",
                    callback_data=f"pnl_bot|{bot_lev}|{capital}"
                ))
            buttons.append(InlineKeyboardButton(
                "✏️ Enter custom leverage",
                callback_data=f"pnl_custom|{capital}"
            ))
            keyboard = InlineKeyboardMarkup([buttons] if len(buttons) == 1 else [[buttons[0]], [buttons[1]]])

            sym_line = f"{signal['exchange']} {signal['symbol']} — {signal['bias']} {signal['confidence']}/10"
            await update.message.reply_text(
                f"✅ Capital: ${capital:,.2f}\n"
                f"📊 Signal: {sym_line}\n\n"
                f"Choose leverage option:",
                reply_markup=keyboard
            )
        except ValueError:
            await update.message.reply_text("⚠️ Enter a valid number (e.g. 100 or 500).")

    elif step == 'custom_lev':
        try:
            leverage = int(text.replace('x', '').strip())
            if leverage < 1 or leverage > 125:
                await update.message.reply_text("⚠️ Leverage must be between 1 and 125.")
                return
            signal  = context.user_data.get('pnl_signal')
            capital = context.user_data.get('pnl_capital', 100)
            context.user_data['pnl_step'] = None

            card = build_pnl_card(signal, leverage, capital, custom=True)
            await update.message.reply_text(card)
        except ValueError:
            await update.message.reply_text("⚠️ Enter a number like 10 or 20x.")

async def pnl_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button presses for PnL card."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith('pnl_bot|'):
        _, lev_str, cap_str = data.split('|')
        leverage = int(lev_str)
        capital  = float(cap_str)
        signal   = context.user_data.get('pnl_signal')
        if not signal:
            await query.edit_message_text("⚠️ Signal data lost. Run /pnl again.")
            return
        card = build_pnl_card(signal, leverage, capital, custom=False)
        await query.edit_message_text(card)

    elif data.startswith('pnl_custom|'):
        _, cap_str = data.split('|')
        context.user_data['pnl_step']    = 'custom_lev'
        context.user_data['pnl_capital'] = float(cap_str)
        await query.edit_message_text(
            "✏️ Enter your custom leverage (1–125):\nExamples: 5  10  20  50"
        )


# ─────────────────────────────────────────────
# /best
# ──────────────────────────────��──��───────────
async def pnl_img_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Buttons under the image PnL card: add capital / refresh / done."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith('pnlimg_lev|'):
        if not context.user_data.get('pnl_signal'):
            await query.message.reply_text("\u26a0\ufe0f Signal data lost. Run /pnl again.")
            return
        val = data.split('|', 1)[1]
        if val == 'custom':
            context.user_data['pnl_step'] = 'pnlimg_lev_custom'
            await query.message.reply_text("\u270f\ufe0f Enter leverage (1\u2013125): e.g. 10, 25, 75")
            return
        try:
            context.user_data['pnl_leverage'] = max(1, min(125, int(val)))
        except ValueError:
            context.user_data['pnl_leverage'] = 10
        await send_pnl_image_card(update, context)
        return

    if data.startswith('pnlimg_mode|'):
        mode = data.split('|', 1)[1]
        if not context.user_data.get('pnl_signal'):
            await query.message.reply_text("\u26a0\ufe0f Signal data lost. Run /pnl again.")
            return
        if mode == 'usdt':
            context.user_data['pnl_mode'] = 'usdt'
            context.user_data['pnl_step'] = 'pnlimg_capital'
            await query.message.reply_text(
                "\U0001F4B5 Enter your capital in USDT (e.g. 100, 500, 1000):"
            )
        else:
            context.user_data['pnl_mode']    = 'pct'
            context.user_data['pnl_capital'] = None
            context.user_data['pnl_step']    = None
            await send_pnl_image_card(update, context)
        return

    if data == 'pnlimg_cap':
        if not context.user_data.get('pnl_signal'):
            await query.message.reply_text("⚠️ Signal data lost. Run /pnl again.")
            return
        context.user_data['pnl_step'] = 'pnlimg_capital'
        await query.message.reply_text(
            "💵 Enter your capital in USDT (e.g. 100, 500, 1000):"
        )
    elif data == 'pnlimg_refresh':
        # Refresh re-fetches the live price AND flips to the other card style,
        # so each tap both updates the numbers and alternates the two displays.
        context.user_data['pnl_card_style'] = (int(context.user_data.get('pnl_card_style', 0)) + 1) % 2
        await send_pnl_image_card(update, context)
    elif data == 'pnlimg_done':
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass


async def best_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    if not state.last_scan_results:
        await update.message.reply_text("⚠️ No scan data yet. Run /scan first.")
        return
    best = state.last_scan_results[0]
    age  = datetime.now() - state.last_scan_time
    mins = int(age.total_seconds() // 60)

    action_buttons = [[
        InlineKeyboardButton("💰 PnL Calculator", callback_data="pnl_from_signal|0"),
        InlineKeyboardButton("🔗 Trade Now", url=get_exchange_link(best['exchange'], best['symbol']))
    ]]
    keyboard = InlineKeyboardMarkup(action_buttons)

    await update.message.reply_text(
        f"🏆 BEST TRADE — {mins} mins ago\n"
        f"Confidence: {best['confidence']}/10 | {best['score']} confluence points\n"
        f"Hold: {best['hold']}"
    )
    _ck, keyboard = cache_signal_card(best, 1, keyboard)
    await update.message.reply_text(format_signal_primary(best, 1), reply_markup=keyboard)


# ─────────────────────────────────────────────
# /tg — TOP GAINS
# ─────────────────���───────────────────────────
async def tg_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    ph = db_load_price_history() if not state.price_history else state.price_history
    if not ph:
        await update.message.reply_text("⚠️ No price history yet. Run /scan a few times.")
        return
    gainers = []
    for key, history in ph.items():
        if len(history) < 2: continue
        oldest = history[0]['price']
        newest = history[-1]['price']
        if oldest > 0:
            pct = ((newest - oldest) / oldest) * 100
            ex, sym = key.split('_', 1)
            gainers.append({'exchange': ex, 'symbol': sym, 'change_pct': pct,
                            'price_now': newest, 'price_then': oldest})
    gainers.sort(key=lambda x: x['change_pct'], reverse=True)
    # ── SINGLE DISPLAY VENUE ── show only the active venue (Bybit, or MEXC
    # when Bybit is down) and one row per pair, so residual cross-exchange
    # price history can't surface the same pair multiple times.
    _venue = _display_exchanges()[0]
    _seen = set()
    top = []
    for g in gainers:
        if g['change_pct'] <= 0:
            continue
        if g['exchange'] != _venue or g['symbol'] in _seen:
            continue
        _seen.add(g['symbol'])
        top.append(g)
        if len(top) >= 10:
            break
    if not top:
        await update.message.reply_text("📊 No positive gainers yet. Run /scan more times.")
        return
    lines = [f"📈 TOP GAINS — Last 24 Hours\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]
    for i, g in enumerate(top, 1):
        lines.append(f"🟢 #{i} {g['exchange']} | {g['symbol']}\n"
                     f"   Change: +{g['change_pct']:.2f}%\n"
                     f"   Then: ${g['price_then']:.6f}  Now: ${g['price_now']:.6f}\n")
    await update.message.reply_text("\n".join(lines))


# ─────────────────────────────────────────────
# /tl — TOP LOSSES
# ─────────────────────────────────────────────
async def tl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    ph = db_load_price_history() if not state.price_history else state.price_history
    if not ph:
        await update.message.reply_text("⚠️ No price history yet. Run /scan a few times.")
        return
    losers = []
    for key, history in ph.items():
        if len(history) < 2: continue
        oldest = history[0]['price']
        newest = history[-1]['price']
        if oldest > 0:
            pct = ((newest - oldest) / oldest) * 100
            ex, sym = key.split('_', 1)
            losers.append({'exchange': ex, 'symbol': sym, 'change_pct': pct,
                           'price_now': newest, 'price_then': oldest})
    losers.sort(key=lambda x: x['change_pct'])
    # ── SINGLE DISPLAY VENUE ── (see /tg) show only the active venue, one row
    # per pair.
    _venue = _display_exchanges()[0]
    _seen = set()
    top = []
    for g in losers:
        if g['change_pct'] >= 0:
            continue
        if g['exchange'] != _venue or g['symbol'] in _seen:
            continue
        _seen.add(g['symbol'])
        top.append(g)
        if len(top) >= 10:
            break
    if not top:
        await update.message.reply_text("📊 No losses recorded yet. Run /scan more times.")
        return
    lines = [f"📉 TOP LOSSES — Last 24 Hours\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]
    for i, g in enumerate(top, 1):
        lines.append(f"🔴 #{i} {g['exchange']} | {g['symbol']}\n"
                     f"   Change: {g['change_pct']:.2f}%\n"
                     f"   Then: ${g['price_then']:.6f}  Now: ${g['price_now']:.6f}\n")
    await update.message.reply_text("\n".join(lines))


# ────────────────���──────����─────���───────────────
# /menu
# ─────────────────────────────────────────────
async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔍 Scanning",      callback_data="menu|scanning"),
            InlineKeyboardButton("🔔 Alerts",         callback_data="menu|alerts"),
        ],
        [
            InlineKeyboardButton("📊 Performance",   callback_data="menu|performance"),
            InlineKeyboardButton("ℹ️ Info",           callback_data="menu|info"),
        ],
        [
            InlineKeyboardButton("💡 Tips",           callback_data="menu|tips"),
            InlineKeyboardButton("❓ Help",           callback_data="menu|help"),
        ],
        [
            InlineKeyboardButton("📞 Contact Owner",  callback_data="menu|contact"),
        ],
    ])
    await update.message.reply_text(
        "👋 Welcome to *Sakz Scan Bot!*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "What would you like to do? Tap a category below 👇",
        parse_mode="Markdown",
        reply_markup=keyboard
    )


# ─────────────────────────────────────────────
# MENU CALLBACK HANDLER — sub-menus per section
# ─────────────────────���───────────────────────
async def menu_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    section = query.data.split("|")[1]

    BACK = [[InlineKeyboardButton("◀️ Back to Menu", callback_data="menu|main")]]

    if section == "main":
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔍 Scanning",      callback_data="menu|scanning"),
                InlineKeyboardButton("�� Alerts",         callback_data="menu|alerts"),
            ],
            [
                InlineKeyboardButton("📊 Performance",   callback_data="menu|performance"),
                InlineKeyboardButton("ℹ️ Info",           callback_data="menu|info"),
            ],
            [
                InlineKeyboardButton("💡 Tips",           callback_data="menu|tips"),
                InlineKeyboardButton("❓ Help",           callback_data="menu|help"),
            ],
            [
                InlineKeyboardButton("📞 Contact Owner",  callback_data="menu|contact"),
            ],
        ])
        await query.edit_message_text(
            "👋 Welcome to *Sakz Scan Bot!*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "What would you like to do? Tap a category below 👇",
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    elif section == "scanning":
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔍 Full Scan",          callback_data="menu_run|scan")],
            [InlineKeyboardButton("🆕 New Listings Scan",  callback_data="menu_run|scannew")],
            [InlineKeyboardButton("📡 Custom Pair Scan",   callback_data="menu_run|cscan")],
            [InlineKeyboardButton("📊 Chart (TA Image)",   callback_data="menu_run|chart")],
            [InlineKeyboardButton("🏆 Best Trade Now",     callback_data="menu_run|best")],
            [InlineKeyboardButton("📋 Top 5 Signals",      callback_data="menu_run|top5")],
            [InlineKeyboardButton("📋 Top 10 Signals",     callback_data="menu_run|top10")],
            [InlineKeyboardButton("🔎 Filter Signals",     callback_data="menu_run|filter")],
            *BACK
        ])
        await query.edit_message_text(
            "🔍 *SCANNING*\n"
            "━━━━━━━━━━━━━━━━━━━━━���━━━━━━━━\n\n"
            "• *Full Scan* — analyses top 50 pairs on Bybit, MEXC & Binance\n"
            "  ⏱ Runs on the *4H timeframe* — swing trade signals only\n\n"
            "• *Custom Pair Scan* — analyse any coin on *your preferred timeframe*\n"
            "  `/cscan ZEC`       auto-detects the strongest timeframe\n"
            "  `/cscan ZEC 15m`   15-min chart → scalp signals (mins–2h)\n"
            "  `/cscan ZEC 1h`    1H chart → intraday signals (30min–8h)\n"
            "  `/cscan ZEC 4h`    4H chart → swing signals (4h–3 days)\n"
            "  `/cscan ZEC 1d`    Daily chart → position signals (1d��2wks)\n\n"
            "• *New Listings* — `/scan new 24h` scans recently listed pairs\n"
            "• *Best Trade Now* — highest conviction signal from last scan\n"
            "• *Top 5 / Top 10* — ranked list of the best current signals\n"
            "• *Filter* — narrow down by LONG/SHORT and confidence level",
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    elif section == "alerts":
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔔 Set Coin Alert",     callback_data="menu_run|alert")],
            [InlineKeyboardButton("🔕 Remove Alert",       callback_data="menu_run|unalert")],
            [InlineKeyboardButton("👁 My Watchlist",       callback_data="menu_run|watch")],
            [InlineKeyboardButton("🤖 Toggle Auto-Scan",   callback_data="menu_run|autoscan")],
            [InlineKeyboardButton("📡 Broadcast Mode",     callback_data="menu_run|broadcast")],
            [InlineKeyboardButton("📈 Pick & Track Trade", callback_data="menu_run|pick")],
            [InlineKeyboardButton("⛔ Stop Trade Tracker", callback_data="menu_run|stoptrade")],
            [InlineKeyboardButton("💰 PnL Calculator",     callback_data="menu_run|pnl")],
            *BACK
        ])
        await query.edit_message_text(
            "🔔 *ALERTS & TRACKING*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Manage your alerts and live trade tracking.\n\n"
            "• *Coin Alert* — get notified when a specific coin hits a signal\n"
            "• *Watchlist* — monitor your favourite pairs automatically\n"
            "• *Auto-Scan* — bot runs a fresh scan every 4 hours for you\n"
            "• *Broadcast* ��� auto-post signals to your group or channel\n"
            "• *Pick & Track* — set up live PnL reminders for a trade\n"
            "• *PnL Calculator* — calculate profit/loss with leverage",
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    elif section == "performance":
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Win Rate Stats",     callback_data="menu_run|stats")],
            [InlineKeyboardButton("🏆 Leaderboard",        callback_data="menu_run|leaderboard")],
            [InlineKeyboardButton("😨 Fear & Greed Index", callback_data="menu_run|fgi")],
            [InlineKeyboardButton("📈 Top Gainers (24h)",  callback_data="menu_run|tg")],
            [InlineKeyboardButton("📉 Top Losers (24h)",   callback_data="menu_run|tl")],
            [InlineKeyboardButton("🔄 Compare vs Entry",   callback_data="menu_run|compare")],
            *BACK
        ])
        await query.edit_message_text(
            "📊 *PERFORMANCE*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Track signals and market sentiment.\n\n"
            "• *Win Rate Stats* — overall + high-confidence win rates\n"
            "• *Leaderboard* — best performing pairs by win rate\n"
            "• *Fear & Greed* — market sentiment + trading guidance\n"
            "• *Top Gainers/Losers* — 24h movers\n"
            "• *Compare* — live PnL vs scan entry\n"
            "   /compare BTC to check a specific pair\n"
            "   /compare BTC 20x for custom leverage",
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    elif section == "info":
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📡 Bot Status",         callback_data="menu_run|status")],
            [InlineKeyboardButton("📋 Full Command List",  callback_data="menu|help")],
            *BACK
        ])
        await query.edit_message_text(
            "ℹ️ *INFO*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "*Sakz Scan Bot* analyses crypto perpetuals across Bybit, MEXC and Binance.\n\n"
            "• Scans 150+ pairs simultaneously\n"
            "• Uses RSI, MACD, EMA, Bollinger Bands, Stochastic & ATR\n"
            "• Funding rate analysis included\n"
            "�� Auto-calculates suggested leverage per signal\n"
            "• Tracks signal outcomes at 4h, 8h, 24h & 48h\n\n"
            "Free to use — no API key required.",
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    elif section == "tips":
        keyboard = InlineKeyboardMarkup(BACK)
        await query.edit_message_text(
            "💡 *PRO TIPS*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔹 *Best workflow:*\n"
            "   /autoscan → /watch BTCUSDT 8\n"
            "   Bot notifies you automatically when your pair signals!\n\n"
            "🔹 *High conviction only:*\n"
            "   Use `/filter LONG 8` or `/filter SHORT 9`\n"
            "   to see only the strongest setups.\n\n"
            "🔹 *Share with your group:*\n"
            "   `/broadcast on` in your channel — bot auto-posts\n"
            "   top signals after every scan.\n\n"
            "🔹 *Track a trade properly:*\n"
            "   `/pick` → choose a signal → set reminder interval\n"
            "   Bot alerts you on T1/T2/T3 hits and stop loss.\n\n"
            "🔹 *Always use isolated margin.*\n"
            "   Never cross-margin with leveraged signals.\n\n"
            "🔹 *First scan takes 5–10 min.*\n"
            "   Subsequent scans are cached for 15 minutes.",
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    elif section == "help":
        keyboard = InlineKeyboardMarkup(BACK)
        await query.edit_message_text(
            "❓ *FULL COMMAND REFERENCE*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "The complete command list now lives under a single command.\n\n"
            "👉  Type /pro to see *all* commands and how to use them.",
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    elif section == "contact":
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("💬 Message Owner", url="https://t.me/Sakazuki_01")],
            *BACK
        ])
        await query.edit_message_text(
            "📞 *CONTACT OWNER*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "For support, feedback, or feature requests — reach out directly:\n\n"
            "👤 *@Sakazuki\\_01*\n\n"
            "Tap the button below to open a chat. 👇",
            parse_mode="Markdown",
            reply_markup=keyboard
        )


# ────────────────────────────────────────���────
# MENU RUN CALLBACK — execute commands from menu buttons
# ─────────────────────────────────────────────
async def menu_run_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """When user taps a command button inside a menu sub-section, execute that command."""
    query = update.callback_query
    await query.answer()
    action = query.data.split("|")[1]

    BACK = [[InlineKeyboardButton("◀️ Back to Menu", callback_data="menu|main")]]

    command_map = {
        "scan":        "/scan — Full market scan on the *4H timeframe*.\nAnalyses top 50 pairs across Bybit, MEXC & Binance.\nTap /scan or type it to start.",
        "scannew":     "/scan new — type `/scan new 24h`, `/scan new 7d`, `/scan new 30m` etc.\nm=minutes  h=hours  d=days  w=weeks",
        "cscan":       (
            "/cscan — Custom scan on your *preferred timeframe*.\n\n"
            "Auto-detect best TF:  `/cscan ZEC`\n"
            "15-min scalp:         `/cscan ZEC 15m`\n"
            "1-hour intraday:      `/cscan ZEC 1h`\n"
            "4-hour swing:         `/cscan ZEC 4h`\n"
            "Daily position:       `/cscan ZEC 1d`\n\n"
            "The bot analyses the coin on that chart and decides the bias."
        ),
        "chart":       "/chart — type `/chart SOL` or `/chart BTC t1h` to generate a TA chart image.",
        "best":        "/best ��� fetching best signal now...",
        "top5":        "/top5",
        "top10":       "/top10",
        "filter":      "/filter — type `/filter LONG 8` to filter by bias and confidence.",
        "alert":       "/alert — type `/alert BTCUSDT 8` to set a coin alert.",
        "unalert":     "/unalert — type `/unalert BTCUSDT` to remove an alert.",
        "watch":       "/watch — type `/watch BTCUSDT 7` to add a coin to your watchlist.",
        "autoscan":    "/autoscan",
        "broadcast":   "/broadcast — type `/broadcast on` to enable or `/broadcast off` to disable.",
        "pick":        "/pick",
        "stoptrade":   "/stoptrade",
        "pnl":         "/pnl",
        "stats":       "/stats — type `/stats 24` for 24h or `/stats 168` for 7 days.",
        "leaderboard": "/leaderboard — type `/leaderboard 7` for 7-day leaderboard.",
        "tg":          "/tg",
        "tl":          "/tl",
        "compare":     "/compare — all signals vs entry.\n/compare BTC — just BTC.\n/compare BTC 20x — BTC at 20x leverage.",
        "status":      "/status",
        "fgi":         "/fgi — Fear & Greed Index + trading guidance.",
    }

    direct_commands = {
        "best":        best_command,
        "top5":        None,
        "top10":       None,
        "autoscan":    autoscan_command,
        "pick":        pick_command,
        "stoptrade":   stoptrade_command,
        "tg":          tg_command,
        "tl":          tl_command,
        "compare":     compare_command,
        "status":      status_command,
        "pnl":         pnl_command,
        "stats":       stats_command,
        "leaderboard": leaderboard_command,
        "fgi":         fgi_command,
    }

    if action in direct_commands and direct_commands[action] is not None:
        # Inject a fake Update-like message so the command handler works
        await query.edit_message_text(
            f"⏳ Running `/{action}`...\n\nResults will appear below.",
            parse_mode="Markdown"
        )
        # Create a minimal shim — send result as a new message
        class _FakeMessage:
            chat = query.message.chat
            message_id = query.message.message_id
            async def reply_text(self, text, **kwargs):
                await context.bot.send_message(chat_id=query.message.chat_id, text=text, **kwargs)

        class _FakeUpdate:
            message        = _FakeMessage()
            effective_chat = query.message.chat
            callback_query = None

        await direct_commands[action](_FakeUpdate(), context)

    elif action in ("top5", "top10"):
        n = 5 if action == "top5" else 10
        await query.edit_message_text(f"⏳ Fetching top {n} signals...", parse_mode="Markdown")

        class _FakeMessage:
            chat = query.message.chat
            text = f"/top{n}"
            async def reply_text(self, text, **kwargs):
                await context.bot.send_message(chat_id=query.message.chat_id, text=text, **kwargs)

        class _FakeUpdate:
            message        = _FakeMessage()
            effective_chat = query.message.chat
            callback_query = None

        await top_command(_FakeUpdate(), context)

    else:
        # For commands that need arguments, show usage guide with Back button
        hint = command_map.get(action, f"Type `/{action}` to use this feature.")
        keyboard = InlineKeyboardMarkup(BACK)
        await query.edit_message_text(
            f"💬 *How to use this command:*\n\n{hint}",
            parse_mode="Markdown",
            reply_markup=keyboard
        )


# ─────────────────────────────────────────────
# /start
# ─────────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    track_user_interaction(update.effective_chat.id)
    _track(update)
    await update.message.reply_text(
        "👋 Welcome to Sakz Scan Bot!\n\n"
        "Crypto perpetuals scanner for\n"
        "Bybit, MEXC & Binance — free, no API key needed.\n\n"
        "📋 /pro — see all commands\n"
        "🔍 /scan — run your first scan\n"
        "🔔 /autoscan — auto-updates every 4h\n\n"
        "⏳ First scan takes 5–10 minutes."
    )


# ─────────────────────────────────────────────
# /status
# ─────────────────────────────────────────────
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    chat_id  = update.effective_chat.id
    tracking = state.user_tracking.get(chat_id)

    if state.last_scan_time:
        age  = datetime.now() - state.last_scan_time
        mins = int(age.total_seconds() // 60)
        longs  = sum(1 for r in state.last_scan_results if r['bias'] == 'LONG')
        shorts = sum(1 for r in state.last_scan_results if r['bias'] == 'SHORT')
        scan_info = (
            f"🕐 Last scan: {state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"⏱ {mins} minutes ago\n"
            f"📊 Signals: {len(state.last_scan_results)}  🟢{longs}  🔴{shorts}"
        )
    else:
        scan_info = "⚠️ No scan run yet."

    # FIX #SCANTIME — Scan duration stats
    if state._scan_durations:
        recent   = state._scan_durations[-10:]
        avg_secs = sum(recent) / len(recent)
        p90_secs = sorted(recent)[int(len(recent) * 0.9)]
        scan_perf = f"⏳ Scan perf: avg {avg_secs:.0f}s  p90 {p90_secs:.0f}s"
        if p90_secs > 120:
            scan_perf += " ⚠️ slow"
    else:
        scan_perf = ""

    # FIX #BTCD — BTC Dominance in status
    _btcd = _get_btc_dominance()
    if _btcd['btcd'] > 0:
        _btcd_arrow = "📈" if _btcd['trend'] == 'rising' else "📉" if _btcd['trend'] == 'falling' else "➡️"
        btcd_line = f"{_btcd_arrow} BTC.D: {_btcd['btcd']:.1f}% ({_btcd['trend']})"
    else:
        btcd_line = ""

    # FIX #HEARTBEAT — Heartbeat status
    hb_line = f"💓 Heartbeat: {'active → ' + HEALTH_CHECK_URL[:40] if HEALTH_CHECK_URL else 'not configured (set HEALTH_CHECK_URL)'}"

    # WebSocket layer status
    _ws = ws_status()
    if _ws.get('ws_available') and _ws.get('symbols_tracked', 0) > 0:
        ws_line = f"⚡ WebSocket: live · {_ws['symbols_tracked']} symbols · oldest price {_ws.get('oldest_price_s', '?')}s ago"
    elif _ws.get('ws_available'):
        ws_line = "⚡ WebSocket: connecting…"
    else:
        ws_line = "⚡ WebSocket: disabled (sakz_ws.py not found)"

    # BTC regime
    regime = state._btc_regime_cache['regime'] if state._btc_regime_cache else "unknown"
    regime_emoji = {'STRONG_BULL':'🟢🟢','BULL':'🟢','NEUTRAL':'⚪','BEAR':'🔴','STRONG_BEAR':'🔴🔴'}.get(regime,'⚪')

    if chat_id in auto_scan_subscribers:
        _tf = auto_scan_subscribers[chat_id]
        auto = f"🟢 ON ({_tf_display(_tf) if _tf else 'All TFs'})"
    else:
        auto = "🔴 OFF"
    alerts = db_get_user_alerts(chat_id)
    alert_info = f"🔔 Coin alerts: {len(alerts)} active" if alerts else "🔔 No coin alerts set"

    if tracking:
        signal    = tracking['signal']
        elapsed   = datetime.now() - tracking['start_time']
        hours     = elapsed.total_seconds() / 3600
        trade_info = (
            f"\n\n📈 ACTIVE TRACKING:\n"
            f"{signal['exchange']} {signal['symbol']} — {signal['bias']}\n"
            f"⏱ Open {hours:.1f}h | Updates every {tracking['interval']} min"
        )
    else:
        trade_info = "\n\n📭 No active trade tracking."

    lines = [
        "✅ Bot is running\n",
        scan_info,
        scan_perf,
        f"{regime_emoji} BTC Regime: {regime}",
        btcd_line,
        "",
        f"🤖 Auto-scan: {auto}",
        alert_info,
        ws_line,
        hb_line,
        trade_info,
    ]
    await update.message.reply_text("\n".join(l for l in lines if l is not None))


# ─────────────────────────────────────────────
# /scan
# ──────────────────────────────────���──────────
# ─────────────────────────────────────────────
# COMPACT SIGNAL CARDS — inline button display
# ────────────────��────────────────────────────
def format_compact_card(rank, r, current_price=None):
    """
    One-line summary shown in the /scan list panel.
    Req #13: symbol + bias + leverage only — no price, no PnL.
    Full details revealed when user taps the numbered button.
    """
    bias_emoji = "🟢" if r['bias'] == "LONG" else "🔴"
    lev        = r.get('leverage')
    lev_str    = f"{lev['suggested']}x" if lev else "1x"
    corr_flag  = " ⚠️" if r.get('corr_flagged') else ""
    return f"{bias_emoji} #{rank}  {r['symbol']}  {r['bias']}  {lev_str}{corr_flag}"

async def send_signal_cards(message, results, title="📊 SIGNALS", max_show=20, chat_id=None, source="scan"):
    """
    Req #13: compact list showing symbol + bias + leverage only.
    Req #14: numbered square buttons 4-per-row, refresh sits between panel and number buttons.
    Full signal card shown when user taps a number button.

    source: tag stored in callback_data so Refresh knows which result set to use.
      'scan'   — full market scan (last_scan_results)
      'scalp'  — scalp scan stored in _chat_scan_ctx
      'swing'  — swing scan stored in _chat_scan_ctx
      'custom' — single-pair scan stored in _chat_scan_ctx
    """
    if not results:
        await message.reply_text("⚠️ No signals to display.")
        return

    cid = chat_id or 0

    # Store results in context so refresh can retrieve them without touching
    # last_scan_results — each source overwrites only its own slot.
    if cid and source != "scan":
        _chat_scan_ctx[cid] = {
            'source':  source,
            'results': results,
            'title':   title,
            'max_show': max_show,
        }

    signals = results[:max_show]
    now_str = datetime.now().strftime('%H:%M:%S')

    # Build compact signal list
    lines = [f"{title}", "━" * 30]
    for rank, r in enumerate(signals, 1):
        lines.append(format_compact_card(rank, r))

    if len(results) > max_show:
        lines.append(f"\n... and {len(results) - max_show} more.")

    lines.append(f"\n🔄 {now_str}")

    n_show = min(max_show, len(signals))

    # Number buttons: 4 per row — encode source so view_signal knows where to look
    num_rows = []
    row = []
    for rank in range(1, n_show + 1):
        row.append(InlineKeyboardButton(
            str(rank), callback_data=f"view_signal|{rank - 1}|{source}|{cid}"
        ))
        if len(row) == 4:
            num_rows.append(row)
            row = []
    if row:
        num_rows.append(row)

    refresh_row = [InlineKeyboardButton(
        "🔄 Refresh", callback_data=f"feed_refresh|{cid}|{n_show}|{source}"
    )]

    all_rows = [refresh_row] + num_rows
    keyboard = InlineKeyboardMarkup(all_rows)
    await message.reply_text("\n".join(lines), reply_markup=keyboard)


async def pnl_from_cscan_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Jump into PnL calculator from a cscan result."""
    query   = update.callback_query
    await query.answer()
    chat_id = int(query.data.split('|')[1])

    results = cscan_results.get(chat_id)
    if not results:
        await query.message.reply_text("⚠️ Custom scan data expired. Run /cscan again.")
        return

    signal = results[0]
    context.user_data['pnl_signal'] = signal
    context.user_data['pnl_step']   = 'capital'
    await query.message.reply_text(
        f"💰 PnL Calculator — {signal['exchange']} {signal['symbol']}\n\n"
        f"How much capital are you trading with? (USDT)\n"
        f"Examples: 50  100  500  1000"
    )

async def view_signal_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Show full signal card when user taps a numbered button.
    callback_data: view_signal|{idx}|{source}|{chat_id}
    Routes to the correct result set (scan / scalp / swing / custom)
    so tapping a number after /scalp shows the scalp signal, not a scan signal.
    """
    query = update.callback_query
    await query.answer()
    parts = query.data.split('|')
    idx     = int(parts[1])
    source  = parts[2] if len(parts) > 2 else 'scan'
    chat_id = int(parts[3]) if len(parts) > 3 else 0

    # Resolve result pool by source
    if source == 'scan':
        pool = state.last_scan_results
    else:
        ctx  = _chat_scan_ctx.get(chat_id, {})
        pool = ctx.get('results', []) if ctx.get('source') == source else []

    if not pool or idx >= len(pool):
        label = {'scalp': '/scalp', 'swing': '/swing', 'custom': '/scan <PAIR>'}.get(source, '/scan')
        await query.message.reply_text(
            f"⚠️ Signal data expired. Re-run {label} for fresh results."
        )
        return

    r    = pool[idx]
    rank = idx + 1

    base_kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("💰 PnL", callback_data=f"pnl_from_signal|{idx}"),
        InlineKeyboardButton("🔗 Trade Now", url=get_exchange_link(r['exchange'], r['symbol']))
    ]])
    _ck, keyboard = cache_signal_card(r, rank, base_kb)
    await query.message.reply_text(format_signal_primary(r, rank), reply_markup=keyboard)

async def pnl_from_signal_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Jump straight into PnL calculator from a signal card button."""
    query = update.callback_query
    await query.answer()
    idx   = int(query.data.split('|')[1])

    if not state.last_scan_results or idx >= len(state.last_scan_results):
        await query.message.reply_text("⚠️ Signal expired. Run /scan again.")
        return

    signal = state.last_scan_results[idx]
    context.user_data['pnl_signal'] = signal
    context.user_data['pnl_step']   = 'capital'

    await query.message.reply_text(
        f"💰 PnL Calculator — {signal['exchange']} {signal['symbol']}\n\n"
        f"How much capital are you trading with? (USDT)\n"
        f"Examples: 50  100  500  1000"
    )


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    chat_id   = update.effective_chat.id
    chat_lock = get_chat_lock(chat_id)

    # Don't stack scans for the same user
    if chat_lock.locked():
        await update.message.reply_text(
            "⏳ Your scan is still running. Please wait for it to finish."
        )
        return

    async with chat_lock:
        # Check cache first
        if state._scan_cache:
            age_secs = (datetime.now() - state._scan_cache['time']).total_seconds()
            if age_secs < CACHE_TTL_SECS:
                mins_old = int(age_secs // 60)
                await update.message.reply_text(
                    f"⚡ Serving cached scan ({mins_old} min old — refreshes every 15 min)\n"
                    f"Use /scan again after 15 min for a fresh scan."
                )
                results = state._scan_cache['results']
                from_cache = True
            else:
                from_cache = False
                results    = None
        else:
            from_cache = False
            results    = None

        if not from_cache:
            await update.message.reply_text(
                "🔍 Starting full scan across exchanges\n"
                "📡 ......\n"
                "⏳ Please wait 5–10 minutes..."
            )
            try:
                results, from_cache = await get_scan_results(force=False)
            except Exception as e:
                await update.message.reply_text(f"❌ Scan failed: {str(e)}")
                return

        if not results:
            await update.message.reply_text(
                "⚠️ No signals found. Market may be consolidating.\nTry again in 30 minutes."
            )
            return

        # ── Phase 0: signal surfacing threshold ─────────────────────────────
        _, _, show_all = parse_scan_args(context.args)

        if show_all:
            shown = sorted(results, key=_conf_of, reverse=True)
        else:
            passing = [r for r in results if passes_display_floor(r)]
            shown   = sorted(passing, key=_conf_of, reverse=True)

        hidden = len(results) - len(shown) if not show_all else 0

        if not shown:
            top = max(results, key=_conf_of, default=None)
            msg = "⚠️ No high-confidence setups right now."
            if top is not None:
                msg += (
                    f" Top scorer: {top.get('symbol')} at {_conf_of(top):.0f}/10.\n"
                    f"Use /scan all to see everything."
                )
            await update.message.reply_text(msg)
            return
        # ────────────────────────────────────────────────────────────────────

        bybit_count  = sum(1 for r in shown if r.get('exchange') == 'BYBIT')
        bin_count    = sum(1 for r in shown if r.get('exchange') == 'BINANCE')
        bybit_note   = f"BYBIT: {bybit_count}" if sakz_exchanges.BYBIT_AVAILABLE else "BYBIT: skipped (blocked)"
        binance_note = f"BINANCE: {bin_count}"  if sakz_exchanges.BINANCE_AVAILABLE else "BINANCE: skipped (blocked)"
        cache_note   = "⚡ cached" if from_cache else "🔄 fresh"

        if not from_cache:
            await notify_alerts(results, context.bot)
            await post_broadcast(results, context.bot)

        title = f"📊 TOP SIGNALS — {state.last_scan_time.strftime('%H:%M')}"
        if hidden:
            title += f"\n({hidden} lower-confidence setup(s) hidden — use /scan all to view)"

        await send_signal_cards(
            update.message, shown,
            title=title,
            max_show=20,
            chat_id=chat_id,
            source="scan"
        )


# ─────────────────────────────────────────────
# /top[n]
# ────────────────────��────────────────────────
async def top_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    text  = update.message.text.strip()
    match = re.match(r'^/top(\d+)', text, re.IGNORECASE)
    if not match:
        await update.message.reply_text("⚠️ Add a number: /top3  /top5  /top10")
        return
    n = int(match.group(1))
    if n < 1:
        await update.message.reply_text("⚠️ Number must be at least 1."); return
    if n > 100:
        await update.message.reply_text("⚠️ Maximum is 100."); return

    if not state.last_scan_results:
        await update.message.reply_text("⏳ No scan yet. Fetching now...")
        results, _ = await get_scan_results()
    else:
        results = state.last_scan_results
        age     = datetime.now() - state.last_scan_time
        mins    = int(age.total_seconds() // 60)
        cache_note = "⚡ cached" if mins < 15 else "🕐 old"
        await update.message.reply_text(
            f"📊 Scan from {mins} min ago ({cache_note})\n"
            f"Showing top {min(n, len(results))} of {len(results)} signals…"
        )

    if not results:
        await update.message.reply_text("⚠��� No signals. Try /scan first.")
        return

    chat_id  = update.effective_chat.id
    actual_n = min(n, len(results))
    signals  = results[:actual_n]
    now_str  = datetime.now().strftime('%H:%M:%S')
    age_min  = int((datetime.now() - state.last_scan_time).total_seconds() // 60) if state.last_scan_time else 0

    lines    = [f"📊 TOP {actual_n} SIGNALS  |  scan {age_min}m ago  |  🔄 {now_str}\n{'━'*30}\n"]
    winners  = losers = unavail = 0

    for i, r in enumerate(signals, 1):
        exchange = r['exchange']
        symbol   = r['symbol']
        entry    = r['price']
        bias     = r['bias']
        conf     = r['confidence']
        lev_data = r.get('leverage')
        leverage = lev_data['suggested'] if lev_data else 1
        b_emoji  = "🟢" if bias == "LONG" else "🔴"

        current = _get_live_price(symbol, exchange)

        if current == 0:
            lines.append(f"{b_emoji} #{i} {symbol} {bias} {conf}/10\n"
                         f"   Signal: ${entry:.4f}  |  ⚠️ live price unavailable\n")
            unavail += 1
            continue

        raw_pct = (current - entry)/entry*100 if bias == 'LONG' else (entry - current)/entry*100
        lev_pnl = raw_pct * leverage
        p_emoji = "🟢" if lev_pnl >= 0 else "🔴"

        if lev_pnl >= 0: winners += 1
        else:            losers  += 1

        # Milestone badge
        if bias == 'LONG':
            if current >= r['t3']:          badge = " 🎯🎯🎯"
            elif current >= r['t2']:        badge = " 🎯🎯"
            elif current >= r['t1']:        badge = " 🎯"
            elif current <= r['stop_loss']: badge = " 🛑"
            else:                           badge = ""
        else:
            if current <= r['t3']:          badge = " 🎯🎯🎯"
            elif current <= r['t2']:        badge = " 🎯🎯"
            elif current <= r['t1']:        badge = " 🎯"
            elif current >= r['stop_loss']: badge = " 🛑"
            else:                           badge = ""

        lines.append(
            f"{b_emoji} #{i} {symbol} {bias} {conf}/10  {leverage}x{badge}\n"
            f"   Signal: ${entry:.4f}  →  Now: ${current:.4f}\n"
            f"   {p_emoji} Raw: {raw_pct:+.2f}%  |  Lev PnL: {lev_pnl:+.2f}%\n"
        )
        await asyncio.sleep(0.05)

    total = winners + losers
    lines.append(f"{'━'*30}")
    lines.append(f"🟢 Winning: {winners}  🔴 Losing: {losers}  📊 Tracked: {total}")
    if unavail:
        lines.append(f"⚠️ Price unavailable: {unavail}")
    lines.append(f"\n💡 Tap Refresh to update prices in real time")

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(
            f"🔄 Refresh top {actual_n}",
            callback_data=f"feed_refresh|{chat_id}|{actual_n}"
        )
    ]])
    await update.message.reply_text("\n".join(lines), reply_markup=keyboard)


# ����────────────────────────────────────────────
# /compare
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# /cscan — Custom pair scanner
# Usage: /cscan BTC   or   /cscan BTCUSDT
# Analyses that specific perp on all available
# exchanges and shows full signal + refresh button.
# ─────────────────────────────────────────────
async def cscan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /cscan SYMBOL [TIMEFRAME]

    Auto-detect best timeframe (default):
        /cscan ZEC
        /cscan ETHUSDT

    Pin to a specific timeframe:
        /cscan ZEC 15m    — 15-minute scalp signals
        /cscan ZEC 1h     — 1-hour swing
        /cscan ZEC 4h     — 4-hour swing (classic mode)
        /cscan ZEC 1d     — daily swing/position

    Aliases: 15 / 1h / 60 / 4h / 240 / 1d / d / daily
    """
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args

    if not args:
        await update.message.reply_text(
            "📡 CUSTOM PAIR SCAN — Pick Your Timeframe\n\n"
            "Usage:\n"
            "  /cscan ZEC          — auto-detects the strongest timeframe\n\n"
            "  /cscan ZEC 15m      — analyses ZEC on the 15-min chart\n"
            "                        bias decision for scalp trades\n"
            "                        hold: minutes to ~2 hours\n\n"
            "  /cscan ZEC 1h       — analyses ZEC on the 1H chart\n"
            "                        bias decision for intraday trades\n"
            "                        hold: 30 min to ~8 hours\n\n"
            "  /cscan ZEC 4h       — analyses ZEC on the 4H chart\n"
            "                        bias decision for swing trades\n"
            "                        hold: 4 hours to ~3 days\n\n"
            "  /cscan ZEC 1d       — analyses ZEC on the Daily chart\n"
            "                        bias decision for position trades\n"
            "                        hold: 1 day to ~2 weeks\n\n"
            "The bot scans all available exchanges at that timeframe\n"
            "and decides LONG or SHORT with a confidence score."
        )
        return

    raw    = args[0].upper().replace('/', '').strip()
    symbol = raw if raw.endswith('USDT') else raw + 'USDT'

    # Parse optional timeframe arg
    tf_key = None
    if len(args) > 1:
        tf_raw = args[1].lower().strip()
        tf_key = TF_ALIASES.get(tf_raw)
        if tf_key is None:
            await update.message.reply_text(
                f"⚠️ Unknown timeframe: {args[1]}\n\n"
                f"Supported: 15m, 1h, 4h, 1d\n"
                f"Example: /cscan ZEC 1h"
            )
            return

    tf_display = TF_CONFIGS[tf_key]['label'] if tf_key else 'AUTO'
    await update.message.reply_text(
        "🔍 Starting full scan across exchanges\n"
        "📡 ......\n"
        "⏳ Please wait 5–10 minutes..."
    )

    loop    = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        SCAN_EXECUTOR, lambda: _cscan_pair_mtf(symbol, tf_key)
    )

    # Check if all results are ScanFailures (no signal produced on any exchange/TF)
    all_failures = results and all(isinstance(r, ScanFailure) for r in results)

    # Fallback to legacy _cscan_pair if MTF returns nothing useful
    if not results or all_failures:
        fallback = await loop.run_in_executor(SCAN_EXECUTOR, lambda: _cscan_pair(symbol))
        fallback_failures = fallback and all(isinstance(r, ScanFailure) for r in fallback)
        if fallback and not fallback_failures:
            results     = fallback
            all_failures = False
        elif fallback and fallback_failures:
            # Merge failure lists, prefer richer failures from fallback
            results = (results or []) + fallback
            all_failures = True

    if not results or all_failures:
        failures = [r for r in (results or []) if isinstance(r, ScanFailure)]

        # ── Aggregate failures by reason ──────────────────────────────
        reason_counts = {}
        for f in failures:
            reason_counts[f.reason] = reason_counts.get(f.reason, 0) + 1

        # Pick the dominant reason (most frequent, priority-ordered)
        priority_order = [REASON_REGIME_BLOCK, REASON_LOW_CONF, REASON_GAP_BLOCK,
                          REASON_COUNTER_TREND, REASON_FLIP_BLOCK,
                          REASON_SHORT_HISTORY, REASON_NEUTRAL, REASON_NO_CONTRACT]
        dominant_reason = REASON_NO_CONTRACT
        for r in priority_order:
            if r in reason_counts:
                dominant_reason = r
                break

        # Best detail string for the dominant reason
        best_detail = next((f.detail for f in failures if f.reason == dominant_reason and f.detail), "")
        exchanges_tried = list({f.exchange for f in failures if f.exchange})
        exchanges_str   = ", ".join(exchanges_tried) if exchanges_tried else "all exchanges"
        sym_base        = symbol.replace("USDT", "")

        # ── Build a specific, actionable message per reason ───────────
        if dominant_reason == REASON_LOW_VOLUME:
            best_detail = next((f.detail for f in failures
                                if f.reason == REASON_LOW_VOLUME and f.detail), "")
            diag_msg = (
                f"💧 {symbol} — Insufficient liquidity\n\n"
                f"{best_detail}\n\n"
                f"This pair doesn't have enough 24h trading volume to produce\n"
                f"reliable signals. Low-volume pairs have wide spreads and\n"
                f"are prone to false breakouts.\n\n"
                f"💡 The signal might exist but isn't tradeable.\n"
                f"Try higher-volume pairs from /scan instead."
            )
        elif dominant_reason == REASON_NO_CONTRACT:
            diag_msg = (
                f"⚠️ {symbol} — No perpetual contract found\n\n"
                f"Checked: {exchanges_str}\n\n"
                f"This symbol likely exists only on spot markets,\n"
                f"or uses a different ticker on futures.\n\n"
                f"💡 Tips:\n"
                f"• Verify the perp exists: search '{sym_base}USDT' on Bybit/MEXC futures\n"
                f"• Some tokens use '1000{sym_base}USDT' format (e.g. 1000BONKUSDT)\n"
                f"• Try /cscan 1000{sym_base} if it's a low-price token"
            )
        elif dominant_reason == REASON_SHORT_HISTORY:
            diag_msg = (
                f"⚠️ {symbol} — Not enough candle history\n\n"
                f"This pair appears to be a very recent listing.\n"
                f"Standard 4H + 1D analysis needs more history than is available.\n\n"
                f"🔄 The dynamic new-listing scanner was attempted but also\n"
                f"couldn't produce a signal (not enough candles on any timeframe yet).\n\n"
                f"💡 Options:\n"
                f"��� Wait ~1–2 hours and try again — new listings fill up fast\n"
                f"• Try /cscan {sym_base} 15m once more candles accumulate\n"
                f"• Check the pair exists as a perpetual on MEXC/Bybit futures"
            )
        elif dominant_reason == REASON_REGIME_BLOCK:
            btc_regime = get_btc_regime()
            regime_details = next((f.detail for f in failures
                                   if f.reason == REASON_REGIME_BLOCK), "")
            diag_msg = (
                f"🚫 {symbol} — Blocked by the BTC Regime gate\n\n"
                f"BTC Regime: {btc_regime}\n"
                f"Detail: {regime_details}\n\n"
                f"This signal fights the current market regime and scored below the\n"
                f"required confidence floor, so the scanner will not publish it.\n\n"
                f"💡 Use /analyse {sym_base} to inspect this pair with no restrictions\n"
                f"   (raw indicators, levels and context — information only)."
            )
            await update.message.reply_text(diag_msg)
            return
            # legacy fall-through (now unreachable — regime-blocked pairs redirect to /analyse)
            loop2 = asyncio.get_event_loop()
            regime_results = await loop2.run_in_executor(
                SCAN_EXECUTOR, lambda: _cscan_pair_mtf(symbol, tf_key)
            )
            valid_regime = [r for r in (regime_results or []) if not isinstance(r, ScanFailure)]
            if valid_regime:
                valid_regime.sort(key=lambda x: (x['confidence'], x.get('dur_score', 0)), reverse=True)
                best_r = valid_regime[0]
                if 'signal_tf_label' not in best_r:
                    best_r['signal_tf_label'] = '4H'
                cscan_results[chat_id] = valid_regime
                _chat_scan_ctx[chat_id] = {
                    'source':   'custom',
                    'results':  valid_regime,
                    'title':    f"���� {best_r['symbol']}  [{tf_display}]",
                    'max_show': 15,
                }
                _safemode_store_signals(chat_id, valid_regime)
                lev2 = best_r.get('leverage')
                compare_snapshot[chat_id] = {
                    f"{best_r['exchange']}_{best_r['symbol']}": {
                        'entry_price': best_r['price'],
                        'leverage':    lev2['suggested'] if lev2 else 1,
                        'bias':        best_r['bias'],
                        'scan_time':   datetime.now(),
                        'signal':      best_r,
                    }
                }
                keyboard2 = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 Trade Now",
                        url=get_exchange_link(best_r['exchange'], best_r['symbol']))]
                ])
                _ck2, keyboard2 = cache_signal_card(best_r, 1, keyboard2)
                await update.message.reply_text(format_signal_primary(best_r, 1), reply_markup=keyboard2)
            return
        elif dominant_reason in (REASON_LOW_CONF, REASON_GAP_BLOCK):
            diag_msg = (
                f"🟠 {symbol} — Doesn't pass the signal requirements\n\n"
                f"A setup was found but it's below the confidence/edge threshold\n"
                f"({best_detail or 'indicators disagreed'}), so it won't be published.\n\n"
                f"💡 Use /analyse {sym_base} to scan this pair with no restrictions\n"
                f"   and see the full breakdown (information only)."
            )
            await update.message.reply_text(diag_msg)
            return
            loop3 = asyncio.get_event_loop()
            lc_results = await loop3.run_in_executor(
                SCAN_EXECUTOR, lambda: _cscan_pair_mtf(symbol, tf_key)
            )
            valid_lc = [r for r in (lc_results or []) if not isinstance(r, ScanFailure)]
            if valid_lc:
                valid_lc.sort(key=lambda x: (x['confidence'], x.get('dur_score', 0)), reverse=True)
                best_lc = valid_lc[0]
                if 'signal_tf_label' not in best_lc:
                    best_lc['signal_tf_label'] = '4H'
                cscan_results[chat_id] = valid_lc
                _chat_scan_ctx[chat_id] = {
                    'source':   'custom',
                    'results':  valid_lc,
                    'title':    f"📡 {best_lc['symbol']}  [{tf_display}]",
                    'max_show': 15,
                }
                _safemode_store_signals(chat_id, valid_lc)
                lev_lc = best_lc.get('leverage')
                compare_snapshot[chat_id] = {
                    f"{best_lc['exchange']}_{best_lc['symbol']}": {
                        'entry_price': best_lc['price'],
                        'leverage':    lev_lc['suggested'] if lev_lc else 1,
                        'bias':        best_lc['bias'],
                        'scan_time':   datetime.now(),
                        'signal':      best_lc,
                    }
                }
                kb_lc = InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔗 Trade Now",
                        url=get_exchange_link(best_lc['exchange'], best_lc['symbol']))
                ]])
                _ck_lc, kb_lc = cache_signal_card(best_lc, 1, kb_lc)
                await update.message.reply_text(format_signal_primary(best_lc, 1), reply_markup=kb_lc)
            return

        elif dominant_reason == REASON_COUNTER_TREND:
            diag_msg = (
                f"↔️ {symbol} — Counter-trend signal blocked\n\n"
                f"The 4H bias opposes the daily EMA direction.\n"
                f"Counter-trend trades require a higher score to qualify.\n\n"
                f"Detail: {best_detail}\n\n"
                f"💡 The daily trend is stronger than the 4H signal.\n"
                f"Consider trading in the daily direction instead."
            )
        elif dominant_reason == REASON_FLIP_BLOCK:
            diag_msg = (
                f"🔄 {symbol} — Direction flip blocked (cooldown active)\n\n"
                f"A signal in the opposite direction fired recently.\n"
                f"The flip-cooldown requires confidence ≥ 9 to reverse.\n\n"
                f"Detail: {best_detail}\n\n"
                f"💡 Wait 8h from the last signal, or the market needs\n"
                f"to show a much stronger reversal signal (conf 9+)."
            )
        else:  # NEUTRAL or unknown
            diag_msg = (
                f"😐 {symbol} — Market too neutral\n\n"
                f"Indicators are balanced — no clear LONG or SHORT edge.\n"
                f"This is normal in sideways/consolidating markets.\n\n"
                f"💡 Try:\n"
                f"• /cscan {sym_base} 15m — shorter TF may be trending\n"
                f"• Check back after the next candle close\n"
                f"• Run /scan for pairs with active momentum"
            )

        await update.message.reply_text(diag_msg)
        return

    # Filter out any stray ScanFailure objects (should not happen after the guard above)
    results = [r for r in results if not isinstance(r, ScanFailure)]
    if not results:
        await update.message.reply_text(f"⚠️ No signal found for {symbol} after filtering.")
        return
    results.sort(key=lambda x: (x['confidence'], x.get('dur_score', 0)), reverse=True)
    best = results[0]

    # Tag with signal_tf_label if missing (legacy fallback)
    if 'signal_tf_label' not in best:
        best['signal_tf_label'] = '4H'
    if 'signal_tf' not in best:
        best['signal_tf'] = '4h'

    # Store for this user so refresh works
    cscan_results[chat_id] = results
    # Persist so /stats leaderboard includes this scan.
    try:
        db_save_scan(results)
    except Exception as _e_save:
        logger.debug("db_save_scan (cscan) failed: %s", _e_save)
    _chat_scan_ctx[chat_id] = {
        'source':   'custom',
        'results':  results,
        'title':    f"📡 {best['symbol']}  [{tf_display}]",
        'max_show': 15,
    }
    _safemode_store_signals(chat_id, results)

    lev = best.get('leverage')
    compare_snapshot[chat_id] = {
        f"{best['exchange']}_{best['symbol']}": {
            'entry_price': best['price'],
            'leverage':    lev['suggested'] if lev else 1,
            'bias':        best['bias'],
            'scan_time':   datetime.now(),
            'signal':      best
        }
    }

    # Build TF switcher buttons — let user re-scan at any TF instantly
    tf_buttons = []
    for tf_opt, cfg in TF_CONFIGS.items():
        label = f"{'✅ ' if tf_opt == best.get('signal_tf') else ''}{cfg['label']}"
        tf_buttons.append(
            InlineKeyboardButton(label, callback_data=f"cscan_tf|{chat_id}|{symbol}|{tf_opt}")
        )

    keyboard = InlineKeyboardMarkup([
        tf_buttons,
        [InlineKeyboardButton("🔄 Refresh (live price + PnL)",
                              callback_data=f"cscan_refresh|{chat_id}|{best['exchange']}|{symbol}")],
        [
            InlineKeyboardButton("💰 PnL Calculator", callback_data=f"pnl_from_cscan|{chat_id}"),
            InlineKeyboardButton("🔗 Trade Now",       url=get_exchange_link(best['exchange'], best['symbol']))
        ]
    ])

    # Count how many TFs produced signals
    tf_found = list({r.get('signal_tf', '4h') for r in results})
    tf_found_str = ' / '.join(sorted(set(
        TF_CONFIGS[t]['label'] for t in tf_found if t in TF_CONFIGS
    )))

    _ck, keyboard = cache_signal_card(best, 1, keyboard)
    await update.message.reply_text(format_signal_primary(best, 1), reply_markup=keyboard)



async def cscan_tf_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Fires when user taps a TF switcher button on the cscan result.
    Re-runs the scan pinned to the chosen timeframe and edits the message.
    callback_data: cscan_tf|{chat_id}|{symbol}|{tf_key}
    """
    query = update.callback_query
    await query.answer(f"Scanning…")
    parts   = query.data.split('|')
    chat_id = int(parts[1])
    symbol  = parts[2]
    tf_key  = parts[3]

    cfg = TF_CONFIGS.get(tf_key)
    if not cfg:
        await query.answer("⚠️ Unknown timeframe", show_alert=True)
        return

    loop    = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        SCAN_EXECUTOR, lambda: _cscan_pair_mtf(symbol, tf_key)
    )

    # _cscan_pair_mtf returns ScanFailure objects when all exchanges fail
    if not results or isinstance(results[0], ScanFailure):
        failures     = results if results else []
        reasons_seen = [f.reason for f in failures]
        sym_base     = symbol.replace('USDT', '')

        if not failures or REASON_NO_CONTRACT in reasons_seen:
            alert_msg = (
                f"❌ {symbol} not found as a perp on any exchange.\n"
                f"Check spelling or try 1000{sym_base}USDT format."
            )
        elif REASON_SHORT_HISTORY in reasons_seen:
            alert_msg = (
                f"📋 {symbol} — not enough candle history on {cfg['label']}.\n"
                f"Try a shorter TF or check back later."
            )
        elif REASON_LOW_VOLUME in reasons_seen:
            alert_msg = (
                f"💧 {symbol} — insufficient volume for reliable signals.\n"
                f"Try a higher-volume pair from /scan."
            )
        else:
            alert_msg = (
                f"⚠️ No signal for {symbol} on {cfg['label']}. "
                f"Market may be too neutral — try /cscan {sym_base} 1h"
            )
        await query.answer(alert_msg, show_alert=True)
        return

    results.sort(key=lambda x: (x['confidence'], x.get('dur_score', 0)), reverse=True)
    best = results[0]
    if 'signal_tf_label' not in best:
        best['signal_tf_label'] = cfg['label']
    if 'signal_tf' not in best:
        best['signal_tf'] = tf_key

    cscan_results[chat_id]    = results
    # Persist so /stats leaderboard includes this scan.
    try:
        db_save_scan(results)
    except Exception as _e_save:
        logger.debug("db_save_scan (cscan tf) failed: %s", _e_save)
    _chat_scan_ctx[chat_id] = {
        'source':   'custom',
        'results':  results,
        'title':    f"���� {best['symbol']}",
        'max_show': 15,
    }
    _safemode_store_signals(chat_id, results)
    lev = best.get('leverage')
    compare_snapshot[chat_id] = {
        f"{best['exchange']}_{best['symbol']}": {
            'entry_price': best['price'],
            'leverage':    lev['suggested'] if lev else 1,
            'bias':        best['bias'],
            'scan_time':   datetime.now(),
            'signal':      best
        }
    }

    tf_buttons = []
    for tf_opt, tcfg in TF_CONFIGS.items():
        label = f"{'✅ ' if tf_opt == tf_key else ''}{tcfg['label']}"
        tf_buttons.append(
            InlineKeyboardButton(label, callback_data=f"cscan_tf|{chat_id}|{symbol}|{tf_opt}")
        )

    keyboard = InlineKeyboardMarkup([
        tf_buttons,
        [InlineKeyboardButton("���� Refresh (live price + PnL)",
                              callback_data=f"cscan_refresh|{chat_id}|{best['exchange']}|{symbol}")],
        [
            InlineKeyboardButton("💰 PnL Calculator", callback_data=f"pnl_from_cscan|{chat_id}"),
            InlineKeyboardButton("🔗 Trade Now",       url=get_exchange_link(best['exchange'], best['symbol']))
        ]
    ])

    try:
        _ck, keyboard = cache_signal_card(best, 1, keyboard)
        await query.edit_message_text(format_signal_primary(best, 1), reply_markup=keyboard)
    except Exception:
        await query.message.reply_text(format_signal_primary(best, 1), reply_markup=keyboard)



    """Synchronous: analyse a single symbol across all available exchanges."""
    results = []
    mexc_sym_c = symbol.upper().replace('_USDT', 'USDT'); mexc_sym = (mexc_sym_c[:-4] + '_USDT') if mexc_sym_c.endswith('USDT') else (mexc_sym_c + '_USDT')

    # MEXC futures
    try:
        df4h = mexc_fetch_ohlcv(symbol, '4h', 100)
        df1d = mexc_fetch_ohlcv(symbol, '1d', 60)
        if df4h is not None and len(df4h) >= 20 and df1d is not None and len(df1d) >= 10:
            df4h = add_indicators(df4h, timeframe="4h")
            df1d = add_indicators(df1d, timeframe="1d")
            if df4h is not None and df1d is not None and len(df4h) >= 5 and len(df1d) >= 5:
                r = score_pair(df4h, df1d, 0, symbol)
                if r:
                    r['exchange'] = 'MEXC'
                    results.append(r)
    except Exception as e:
        logger.warning("cscan MEXC %s: %s", symbol, e)

    # Bybit
    if sakz_exchanges.BYBIT_AVAILABLE is not False:
        try:
            df4h = bybit_fetch_ohlcv(symbol, '240', 100)
            df1d = bybit_fetch_ohlcv(symbol, 'D', 60)
            if df4h is not None and len(df4h) >= 20 and df1d is not None and len(df1d) >= 10:
                df4h = add_indicators(df4h, timeframe="4h")
                df1d = add_indicators(df1d, timeframe="1d")
                if df4h is not None and df1d is not None and len(df4h) >= 5 and len(df1d) >= 5:
                    funding = bybit_fetch_funding(symbol)
                    r = score_pair(df4h, df1d, funding, symbol)
                    if r:
                        r['exchange'] = 'BYBIT'
                        results.append(r)
        except Exception as e:
            logger.warning("cscan Bybit %s: %s", symbol, e)

    # Binance
    if sakz_exchanges.BINANCE_AVAILABLE:
        try:
            r = analyze_binance(symbol)
            if r:
                results.append(r)
        except Exception as e:
            logger.warning("cscan Binance %s: %s", symbol, e)

    # If no signal passed scoring, force a neutral analysis on MEXC at lower threshold
    if not results:
        try:
            df4h = mexc_fetch_ohlcv(symbol, '4h', 100)
            df1d = mexc_fetch_ohlcv(symbol, '1d', 60)
            if df4h is not None and len(df4h) >= 15 and df1d is not None and len(df1d) >= 8:
                df4h_ind = add_indicators(df4h, timeframe="4h")
                df1d_ind = add_indicators(df1d, timeframe="1d")
                if df4h_ind is not None and df1d_ind is not None:
                    # Force score at lower threshold for cscan
                    L     = df4h_ind.iloc[-1]
                    price = L['close']
                    atr   = L['atr']
                    rsi4  = L['rsi']
                    bias  = "LONG" if rsi4 < 50 else "SHORT"
                    lev   = calculate_leverage(price, price - atr*0.4, price + atr*0.2,
                                               price - atr*2.5 if bias=="LONG" else price + atr*2.5,
                                               atr, 5, bias)
                    results.append({
                        'symbol': symbol, 'exchange': 'MEXC', 'bias': bias,
                        'confidence': 5, 'score': 3,
                        'price': price,
                        'entry_low':  price - atr*0.4, 'entry_high': price + atr*0.2,
                        'stop_loss':  price - atr*2.5 if bias=="LONG" else price + atr*2.5,
                        't1': price + atr*1.5 if bias=="LONG" else price - atr*1.5,
                        't2': price + atr*3.0 if bias=="LONG" else price - atr*3.0,
                        't3': price + atr*5.0 if bias=="LONG" else price - atr*5.0,
                        'hold': '~24 hours', 'tf_note': 'Forced analysis — low confidence',
                        'hold_hours': 24, 'dur_score': 3, 'dur_reasons': ['Forced analysis at lower threshold'],
                        'funding': 0.0, 'rsi4': rsi4, 'rsi_d': 50.0, 'stoch_k': 50.0,
                        'atr': atr, 'reasons': [f'RSI {rsi4:.1f} suggests {bias}'],
                        'leverage': lev, 'scan_time': datetime.now()
                    })
        except Exception as e:
            logger.warning("cscan forced analysis %s: %s", symbol, e)

    return results


async def cscan_refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Refresh button on cscan — edits the original message in place with live PnL.
    Preserves the original signal panel format and appends a live-price PnL block."""
    query    = update.callback_query
    await query.answer("Fetching live price…")
    parts    = query.data.split('|')
    chat_id  = int(parts[1])
    exchange = parts[2]
    symbol   = parts[3]

    current = _get_live_price(symbol, exchange)

    if current == 0:
        await query.answer("⚠️ Could not fetch live price.", show_alert=True)
        return

    snap       = compare_snapshot.get(chat_id, {})
    key        = f"{exchange}_{symbol}"
    entry_data = snap.get(key)
    now_str    = datetime.now().strftime('%H:%M:%S')

    # Retrieve stored signal so we can rebuild the original signal card
    stored_signal = entry_data.get('signal') if entry_data else None

    if stored_signal:
        entry    = entry_data['entry_price']
        leverage = entry_data['leverage']
        bias     = entry_data['bias']
        scan_dt  = entry_data['scan_time']
        elapsed  = (datetime.now() - scan_dt).total_seconds() / 3600
        raw_pct  = (current - entry)/entry*100 if bias == 'LONG' else (entry - current)/entry*100
        lev_pnl  = raw_pct * leverage
        p_emoji  = "🟢" if lev_pnl >= 0 else "🔴"
        b_emoji  = "🟢" if bias == "LONG" else "🔴"

        # Milestone badge
        t1, t2, t3, sl = stored_signal['t1'], stored_signal['t2'], stored_signal['t3'], stored_signal['stop_loss']
        if bias == 'LONG':
            if current >= t3:   milestone = "🎯🎯🎯 T3 HIT"
            elif current >= t2: milestone = "🎯🎯 T2 HIT"
            elif current >= t1: milestone = "🎯 T1 HIT"
            elif current <= sl: milestone = "🛑 SL HIT"
            else:               milestone = "⏳ In progress"
        else:
            if current <= t3:   milestone = "🎯🎯🎯 T3 HIT"
            elif current <= t2: milestone = "🎯🎯 T2 HIT"
            elif current <= t1: milestone = "🎯 T1 HIT"
            elif current >= sl: milestone = "🛑 SL HIT"
            else:               milestone = "⏳ In progress"

        # Rebuild the original signal card then append a live PnL block
        signal_card = format_signal(stored_signal, 1)
        pnl_block = (
            f"\n{'━'*30}\n"
            f"🔄 LIVE UPDATE — {now_str}  ({elapsed:.1f}h since scan)\n"
            f"{'━'*30}\n"
            f"{b_emoji} {bias} | {leverage}x  |  {milestone}\n"
            f"💵 Signal price:  ${entry:.6f}\n"
            f"💰 Current price: ${current:.6f}\n"
            f"📊 Raw move:    {raw_pct:+.2f}%\n"
            f"{p_emoji} Lev PnL:     {lev_pnl:+.2f}%\n"
            f"{'🎯 In profit — watch targets!' if lev_pnl > 0 else '⚠️ In loss — watch stop loss!'}"
        )
        msg = signal_card + pnl_block
    else:
        msg = (
            f"�� CSCAN — {exchange} | {symbol}\n"
            f"{'━'*30}\n"
            f"💰 Current: ${current:.6f}  |  🔄 {now_str}\n\n"
            f"ℹ️ Run /cscan {symbol.replace('USDT','')} again to track PnL from entry."
        )

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔄 Refresh again", callback_data=f"cscan_refresh|{chat_id}|{exchange}|{symbol}")
    ]])
    try:
        await query.edit_message_text(msg, reply_markup=keyboard)
    except Exception:
        pass


# ─────────────────────────────────────────────
# SCAN FEED REFRESH — for /scan and /top results
# Refreshes live prices of all signals in the
# last scan feed, shown as a PnL table.
# ───────────────────────────────��─────────────
async def feed_refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Refresh handler for /scan, /scalp, /swing, and single-pair scan feeds.
    Routes by source tag encoded in callback_data so each command retains
    its own result set after refresh — no cross-contamination from last_scan_results.

    callback_data format: feed_refresh|{chat_id}|{n_show}|{source}
      source: 'scan' | 'scalp' | 'swing' | 'custom'
    """
    query   = update.callback_query
    await query.answer("Refreshing prices…")
    parts   = query.data.split('|')
    chat_id = int(parts[1])
    n_show  = int(parts[2]) if len(parts) > 2 else 10
    source  = parts[3] if len(parts) > 3 else 'scan'

    # ── Resolve result set by source ────────��─────────────────────────
    if source == 'scan':
        if not state.last_scan_results:
            await query.edit_message_text("⚠️ No scan data. Run /scan first.")
            return
        signals_pool = state.last_scan_results
        title_prefix = "📊 TOP SIGNALS"
    else:
        ctx = _chat_scan_ctx.get(chat_id)
        if not ctx or ctx.get('source') != source:
            # Fallback gracefully — tell user to re-run instead of showing wrong data
            label = {'scalp': '/scalp', 'swing': '/swing', 'custom': '/scan <PAIR>'}.get(source, '/scan')
            await query.edit_message_text(
                f"⚠️ {source.title()} scan data expired or not found.\n"
                f"Re-run {label} to get a fresh set."
            )
            return
        signals_pool = ctx['results']
        title_prefix = ctx.get('title', f"📊 {source.upper()} SIGNALS")

    signals = signals_pool[:max(n_show, 1)]
    now_str = datetime.now().strftime('%H:%M:%S')
    age_min = int((datetime.now() - state.last_scan_time).total_seconds() // 60) if state.last_scan_time and source == 'scan' else 0

    age_str = f"  |  scan {age_min}m ago" if age_min > 0 else ""
    lines   = [f"{title_prefix}{age_str}  |  🔄 {now_str}\n{'━'*30}\n"]
    winners = losers = unavail = 0

    for i, r in enumerate(signals, 1):
        exchange = r['exchange']
        symbol   = r['symbol']
        entry    = r['price']
        bias     = r['bias']
        lev_data = r.get('leverage')
        leverage = lev_data['suggested'] if lev_data else 1

        current = _get_live_price(symbol, exchange)

        if current == 0:
            lines.append(f"#{i} {symbol} — ⚠️ price unavailable\n")
            unavail += 1
            continue

        raw_pct = (current - entry)/entry*100 if bias == 'LONG' else (entry - current)/entry*100
        lev_pnl = raw_pct * leverage
        p_emoji = "🟢" if lev_pnl >= 0 else "🔴"
        b_emoji = "🟢" if bias == "LONG" else "🔴"

        if lev_pnl >= 0: winners += 1
        else:            losers  += 1

        if bias == 'LONG':
            if current >= r['t3']:          badge = " 🎯🎯🎯"
            elif current >= r['t2']:        badge = " 🎯🎯"
            elif current >= r['t1']:        badge = " 🎯"
            elif current <= r['stop_loss']: badge = " 🛑"
            else:                           badge = ""
        else:
            if current <= r['t3']:          badge = " 🎯🎯🎯"
            elif current <= r['t2']:        badge = " 🎯🎯"
            elif current <= r['t1']:        badge = " 🎯"
            elif current >= r['stop_loss']: badge = " 🛑"
            else:                           badge = ""

        lines.append(
            f"{b_emoji} #{i} {symbol} {bias} {leverage}x{badge}\n"
            f"   Signal: ${entry:.4f}  →  Now: ${current:.4f}\n"
            f"   {p_emoji} Raw: {raw_pct:+.2f}%  |  Lev PnL: {lev_pnl:+.2f}%\n"
        )
        await asyncio.sleep(0.05)

    total = winners + losers
    lines.append(f"{'━'*30}")
    lines.append(f"🟢 Winning: {winners}  🔴 Losing: {losers}  📊 Tracked: {total}")
    if unavail:
        lines.append(f"⚠️ Unavailable: {unavail}")
    lines.append(f"⏱ Signal prices from scan  |  🔄 Prices live at {now_str}")

    # Rebuild buttons — preserve source in callback_data
    refresh_row = [InlineKeyboardButton("🔄 Refresh", callback_data=f"feed_refresh|{chat_id}|{n_show}|{source}")]
    num_rows    = []
    row         = []
    for rank in range(1, len(signals) + 1):
        row.append(InlineKeyboardButton(str(rank), callback_data=f"view_signal|{rank-1}|{source}|{chat_id}"))
        if len(row) == 4:
            num_rows.append(row)
            row = []
    if row:
        num_rows.append(row)
    keyboard = InlineKeyboardMarkup([refresh_row] + num_rows)
    try:
        await query.edit_message_text("\n".join(lines), reply_markup=keyboard)
    except Exception:
        pass

# ─────────────────────────────────────────────
# /compare — Enhanced per-symbol price tracker
#
# Usage:
#   /compare ZEC         — live price vs scan entry, full fluctuation report
#   /compare ZEC 20x     — same with custom leverage for PnL calc
#   /compare             — full scan summary (all signals vs current prices)
#
# Shows:
#   • Entry price from last scan (cscan or full scan)
#   • Current live price
#   • Raw % move and leveraged PnL since scan
#   • Price HIGH and LOW since scan (actual swing range)
#   • % distance to each target and SL (so user knows how close each level is)
#   • Milestone (T1/T2/T3/SL hit indicator)
#   • Age of signal
#   • Refresh button (live, one-tap)
# ─────────────��─────────────────────���─────────
def _compare_fetch_swing(exchange, symbol, since: datetime):
    """
    Fetch 1H OHLC candles since `since` and return (high, low, candle_count).
    Used to show actual price swing within the signal window.
    Falls back to (None, None, 0) on any error.
    """
    try:
        if exchange == 'BYBIT':
            df = bybit_fetch_ohlcv(symbol, '60', 60)
        elif exchange == 'BINANCE':
            df = binance_fetch_ohlcv(symbol, '1h', 60)
        else:
            df = mexc_fetch_ohlcv(symbol, '1h', 60)

        if df is None or len(df) == 0:
            return None, None, 0

        mask = df['timestamp'].apply(
            lambda t: t.replace(tzinfo=None) >= since.replace(tzinfo=None)
        )
        window = df[mask]
        if len(window) == 0:
            return None, None, 0

        return float(window['high'].max()), float(window['low'].min()), len(window)
    except Exception as e:
        logger.warning("_compare_fetch_swing %s %s: %s", exchange, symbol, e)
        return None, None, 0


def _build_compare_card(signal, current, leverage, custom_lev=False, swing_high=None, swing_low=None, candle_count=0):
    """
    Build the full compare message for a single symbol.
    Separated from the command handler so the refresh callback can reuse it.
    """
    exchange   = signal['exchange']
    sym        = signal['symbol']
    bias       = signal['bias']
    entry      = signal['price']
    t1, t2, t3 = signal['t1'], signal['t2'], signal['t3']
    sl         = signal['stop_loss']
    conf       = signal.get('confidence', 0)
    scan_dt    = signal.get('scan_time')

    bias_e    = "🟢" if bias == "LONG" else "🔴"
    pnl_emoji = lambda p: "🟢" if p >= 0 else "🔴"
    now_s     = datetime.now().strftime('%H:%M:%S UTC')

    # Age
    age_str = "unknown"
    if isinstance(scan_dt, datetime):
        elapsed = (datetime.now() - scan_dt).total_seconds()
        h, m    = int(elapsed // 3600), int((elapsed % 3600) // 60)
        age_str = f"{h}h {m}m ago"

    # Price move
    if bias == 'LONG':
        raw_pct = (current - entry) / entry * 100
    else:
        raw_pct = (entry - current) / entry * 100
    lev_pnl = raw_pct * leverage

    # Distance to each level (always positive %)
    def dist(level):
        return abs(current - level) / entry * 100

    def reached(level, direction):
        return (current >= level) if direction == 'LONG' else (current <= level)

    def past_sl(direction):
        return (current <= sl) if direction == 'LONG' else (current >= sl)

    # Milestone
    if bias == 'LONG':
        if current >= t3:   milestone = "🎯🎯🎯 T3 HIT"
        elif current >= t2: milestone = "🎯🎯 T2 HIT"
        elif current >= t1: milestone = "🎯 T1 HIT"
        elif current <= sl: milestone = "🛑 SL HIT"
        else:               milestone = "⏳ In progress"
    else:
        if current <= t3:   milestone = "🎯🎯🎯 T3 HIT"
        elif current <= t2: milestone = "🎯🎯 T2 HIT"
        elif current <= t1: milestone = "🎯 T1 HIT"
        elif current >= sl: milestone = "🛑 SL HIT"
        else:               milestone = "⏳ In progress"

    # Build distance lines — show ✅ if already past that level
    def level_line(label, level, direction):
        if reached(level, direction):
            return f"   {label}: ${level:.6f} ✅ REACHED ({dist(level):.2f}% away)"
        return f"   {label}: ${level:.6f} ({dist(level):.2f}% away)"

    def sl_line():
        if past_sl(bias):
            return f"   🛑 SL:  ${sl:.6f} ✅ HIT ({dist(sl):.2f}% away)"
        return f"   🛑 SL:  ${sl:.6f} ({dist(sl):.2f}% away)"

    lev_note = f"custom {leverage}x" if custom_lev else f"bot suggested {leverage}x"

    # Swing section
    swing_lines = ""
    if swing_high is not None and swing_low is not None:
        swing_range_pct = (swing_high - swing_low) / entry * 100
        high_vs_entry   = (swing_high - entry) / entry * 100
        low_vs_entry    = (swing_low  - entry) / entry * 100
        swing_lines = (
            f"\n📐 PRICE SWING SINCE SCAN ({candle_count}h window)\n"
            f"   High: ${swing_high:.6f}  ({high_vs_entry:+.2f}% vs entry)\n"
            f"   Low:  ${swing_low:.6f}  ({low_vs_entry:+.2f}% vs entry)\n"
            f"   Range: {swing_range_pct:.2f}% total swing\n"
        )

    msg = (
        f"📊 COMPARE — {exchange} | {sym}\n"
        f"{'━'*30}\n"
        f"{bias_e} {bias} | {conf}/10  |  {milestone}\n"
        f"🕐 Signal: {age_str}  |  🔄 {now_s}\n"
        f"{'━'*30}\n\n"
        f"💵 PRICES\n"
        f"   Entry (scan): ${entry:.6f}\n"
        f"   Current:      ${current:.6f}\n"
        f"   Raw move:     {raw_pct:+.2f}%\n\n"
        f"⚡ PnL ({lev_note})\n"
        f"   {pnl_emoji(lev_pnl)} Leveraged: {lev_pnl:+.2f}%\n"
        f"{'━'*30}\n\n"
        f"🎯 LEVELS\n"
        f"{level_line('T3', t3, bias)}\n"
        f"{level_line('T2', t2, bias)}\n"
        f"{level_line('T1', t1, bias)}\n"
        f"{sl_line()}"
        f"{swing_lines}"
    )
    return msg


async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /compare          — compare ALL signals from last scan vs current prices
    /compare ZEC      — full live compare for ZECUSDT: entry, price, swing, levels
    /compare ZEC 18x  — same with custom leverage override
    """
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args

    # ── CUSTOM COMPARE: /compare SYMBOL [leverage] ────────────────────────
    if args:
        raw    = args[0].upper().replace('_USDT', 'USDT')
        symbol = raw if raw.endswith('USDT') else raw + 'USDT'

        # Optional leverage override
        custom_lev = None
        if len(args) > 1:
            try:
                custom_lev = int(args[1].lower().replace('x', ''))
            except ValueError:
                pass

        # 1. Check cscan_results for this user (most recent cscan)
        signal = None
        user_cscan = cscan_results.get(chat_id, [])
        for r in user_cscan:
            if r['symbol'].replace('_USDT', 'USDT') == symbol:
                signal = r
                break

        # 2. Check last full scan results
        if not signal:
            for r in state.last_scan_results:
                if r['symbol'].replace('_USDT', 'USDT') == symbol:
                    signal = r
                    break

        # 3. Fallback: signal_outcomes DB (most recent entry for this symbol)
        if not signal:
            conn = db_connect()
            c    = conn.cursor()
            c.execute(
                "SELECT * FROM signal_outcomes WHERE symbol=? OR symbol=? "
                "ORDER BY scan_time DESC LIMIT 1",
                (symbol, symbol.replace('USDT', '_USDT'))
            )
            row = c.fetchone()
            conn.close()
            if row:
                signal = {
                    'symbol':     row['symbol'],
                    'exchange':   row['exchange'],
                    'bias':       row['bias'],
                    'price':      row['entry_price'],
                    'confidence': row['confidence'],
                    'stop_loss':  row['stop_loss'],
                    't1': row['t1'], 't2': row['t2'], 't3': row['t3'],
                    'scan_time':  datetime.fromisoformat(row['scan_time']),
                    'leverage':   None
                }

        if not signal:
            await update.message.reply_text(
                f"⚠️ No signal found for {symbol}.\n\n"
                f"Run /cscan {symbol.replace('USDT', '')} first to generate one,\n"
                f"or use /scan to find it in the next full scan."
            )
            return

        exchange = signal['exchange']
        lev_data = signal.get('leverage')
        leverage = custom_lev or (lev_data['suggested'] if lev_data else 1)

        # Live price
        current = _get_live_price(signal['symbol'], exchange)

        if current == 0:
            await update.message.reply_text(f"⚠️ Could not fetch live price for {symbol}.")
            return

        # Swing data since scan
        scan_dt = signal.get('scan_time')
        swing_high, swing_low, candle_count = None, None, 0
        if isinstance(scan_dt, datetime):
            loop = asyncio.get_event_loop()
            swing_high, swing_low, candle_count = await loop.run_in_executor(
                None, _compare_fetch_swing, exchange, signal['symbol'], scan_dt
            )

        msg = _build_compare_card(
            signal, current, leverage,
            custom_lev=bool(custom_lev),
            swing_high=swing_high, swing_low=swing_low,
            candle_count=candle_count
        )

        # Encode scan_time as unix timestamp in callback data (avoids pipe issues)
        scan_ts = int(scan_dt.timestamp()) if isinstance(scan_dt, datetime) else 0

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "🔄 Refresh live",
                    callback_data=f"cmp_refresh|{exchange}|{signal['symbol']}|{signal['price']}|{signal['bias']}|{leverage}|{int(bool(custom_lev))}|{scan_ts}|{signal['t1']}|{signal['t2']}|{signal['t3']}|{signal['stop_loss']}|{signal.get('confidence',0)}"
                ),
            ],
            [
                InlineKeyboardButton("🔗 Trade", url=get_exchange_link(exchange, signal['symbol'])),
                InlineKeyboardButton("💰 PnL Calc", callback_data=f"pnl_from_cscan|{chat_id}"),
            ]
        ])
        await update.message.reply_text(msg, reply_markup=keyboard)
        return

    # ── FULL COMPARE: /compare (all signals) ─────────────────────────────
    if not state.last_scan_results:
        await update.message.reply_text("⚠️ No scan yet. Run /scan first.")
        return

    age   = datetime.now() - state.last_scan_time
    mins  = int(age.total_seconds() // 60)
    now_s = datetime.now().strftime('%H:%M:%S')

    await update.message.reply_text(
        f"⏳ Fetching live prices for top 15 signals...\n"
        f"Scan was {mins} min ago."
    )

    lines     = [f"📊 FULL COMPARE — {now_s}\n🕐 Scan: {mins} min ago\n{'━'*30}\n"]
    winners   = 0
    losers    = 0
    best_pnl  = ('', -9999)
    worst_pnl = ('', 9999)

    # FIX #RI — get current BTC price once for regime invalidation
    btc_now = _get_btc_price_cached()

    for i, r in enumerate(state.last_scan_results[:15], 1):
        exchange = r['exchange']
        symbol   = r['symbol']
        entry    = r['price']
        bias     = r['bias']
        lev_data = r.get('leverage')
        leverage = lev_data['suggested'] if lev_data else 1

        current = _get_live_price(symbol, exchange)

        if current == 0:
            lines.append(f"#{i} {symbol} — ⚠️ price unavailable\n")
            continue

        raw_pct = (current - entry)/entry*100 if bias=='LONG' else (entry-current)/entry*100
        lev_pnl = raw_pct * leverage
        p_emoji = "🟢" if lev_pnl >= 0 else "🔴"
        b_emoji = "🟢" if bias == "LONG" else "🔴"

        if lev_pnl >= 0: winners += 1
        else:            losers  += 1
        if lev_pnl > best_pnl[1]:  best_pnl  = (symbol, lev_pnl)
        if lev_pnl < worst_pnl[1]: worst_pnl = (symbol, lev_pnl)

        # FIX #RI — Regime invalidation: stale signal warning
        btc_scan_price = r.get('btc_price_at_scan', 0)
        stale_line = ''
        if btc_scan_price and btc_now and btc_now > 0:
            btc_move_pct = (btc_now - btc_scan_price) / btc_scan_price * 100
            if bias == 'SHORT' and btc_move_pct > 1.5:
                stale_line = f"   ⚠️ STALE — BTC +{btc_move_pct:.1f}% since signal fired\n"
            elif bias == 'LONG' and btc_move_pct < -1.5:
                stale_line = f"   ⚠️ STALE — BTC {btc_move_pct:.1f}% since signal fired\n"

        # FIX #CF — warn if signal is hard-blocked by correlation filter
        hard_blocked = r.get('corr_hard_blocked', False)
        corr_line = ''
        if hard_blocked:
            slot = r.get('corr_slot', '?')
            corr_line = f"   📛 OVER-CORR — #{slot} same-direction signal, skip this\n"

        # Milestone
        milestone = ""
        if bias == 'LONG':
            if current >= r['t3']:          milestone = " 🎯🎯🎯"
            elif current >= r['t2']:        milestone = " 🎯🎯"
            elif current >= r['t1']:        milestone = " 🎯"
            elif current <= r['stop_loss']: milestone = " 🛑"
        else:
            if current <= r['t3']:          milestone = " 🎯🎯🎯"
            elif current <= r['t2']:        milestone = " 🎯🎯"
            elif current <= r['t1']:        milestone = " 🎯"
            elif current >= r['stop_loss']: milestone = " 🛑"

        lines.append(
            f"{b_emoji} #{i} {symbol} {bias} {leverage}x{milestone}\n"
            f"   ${entry:.4f} → ${current:.4f}  ({raw_pct:+.2f}%)\n"
            f"   {p_emoji} Lev PnL: {lev_pnl:+.2f}%\n"
            + stale_line + corr_line
        )
        await asyncio.sleep(0.05)

    total_done = winners + losers
    lines.append(f"{'━'*30}")
    lines.append(f"🟢 Winning: {winners}  🔴 Losing: {losers}  📊 Tracked: {total_done}")
    if best_pnl[0]:  lines.append(f"🏆 Best:  {best_pnl[0]}  {best_pnl[1]:+.2f}%")
    if worst_pnl[0]: lines.append(f"💀 Worst: {worst_pnl[0]}  {worst_pnl[1]:+.2f}%")
    lines.append(f"\n💡 /compare ZEC — track a specific pair")
    lines.append(f"💡 /compare ZEC 20x — with custom leverage")

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔄 Refresh All PnL", callback_data=f"cmp_full_refresh|{chat_id}")
    ]])
    await update.message.reply_text("\n".join(lines), reply_markup=keyboard)


async def compare_refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Refresh button handler for the enhanced /compare SYMBOL card.
    Encodes all signal data in callback_data to avoid needing server-side state.
    Format: cmp_refresh|exchange|symbol|entry|bias|leverage|custom_lev|scan_ts|t1|t2|t3|sl|conf
    """
    query = update.callback_query
    await query.answer("Fetching live price…")

    try:
        parts      = query.data.split('|')
        exchange   = parts[1]
        symbol     = parts[2]
        entry      = float(parts[3])
        bias       = parts[4]
        leverage   = int(parts[5])
        custom_lev = bool(int(parts[6]))
        scan_ts    = int(parts[7])
        t1         = float(parts[8])
        t2         = float(parts[9])
        t3         = float(parts[10])
        sl         = float(parts[11])
        conf       = int(parts[12])
        scan_dt    = datetime.fromtimestamp(scan_ts) if scan_ts else None
    except Exception as e:
        logger.warning("compare_refresh_callback parse error: %s", e)
        await query.answer("⚠️ Refresh data corrupt — run /compare again.", show_alert=True)
        return

    # Live price
    current = _get_live_price(symbol, exchange)

    if current == 0:
        await query.answer("⚠️ Price unavailable — try again.", show_alert=True)
        return

    # Swing since scan
    swing_high, swing_low, candle_count = None, None, 0
    if scan_dt:
        loop = asyncio.get_event_loop()
        swing_high, swing_low, candle_count = await loop.run_in_executor(
            None, _compare_fetch_swing, exchange, symbol, scan_dt
        )

    # Reconstruct minimal signal dict for card builder
    signal = {
        'symbol': symbol, 'exchange': exchange, 'bias': bias,
        'price': entry, 'confidence': conf,
        'stop_loss': sl, 't1': t1, 't2': t2, 't3': t3,
        'scan_time': scan_dt, 'leverage': None
    }

    msg = _build_compare_card(
        signal, current, leverage,
        custom_lev=custom_lev,
        swing_high=swing_high, swing_low=swing_low,
        candle_count=candle_count
    )

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🔄 Refresh live",
                callback_data=query.data   # same data, re-runs identical refresh
            ),
        ],
        [
            InlineKeyboardButton("🔗 Trade", url=get_exchange_link(exchange, symbol)),
        ]
    ])
    try:
        await query.edit_message_text(msg, reply_markup=keyboard)
    except Exception:
        pass


async def custom_compare_refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Legacy refresh handler — kept for backwards compatibility with old callback data."""
    query = update.callback_query
    await query.answer("Refreshing...")
    parts    = query.data.split('|')
    exchange = parts[1]
    symbol   = parts[2]
    entry    = float(parts[3])
    bias     = parts[4]
    leverage = int(parts[5])

    current = _get_live_price(symbol, exchange)

    if current == 0:
        await query.answer("⚠️ Price unavailable", show_alert=True)
        return

    raw_pct = (current-entry)/entry*100 if bias=='LONG' else (entry-current)/entry*100
    lev_pnl = raw_pct * leverage
    p_emoji = "🟢" if lev_pnl >= 0 else "🔴"
    b_emoji = "🟢" if bias == "LONG" else "🔴"
    now_s   = datetime.now().strftime('%H:%M:%S')

    msg = (
        f"🔄 REFRESHED — {exchange} | {symbol}  [{now_s}]\n"
        f"{'━'*30}\n"
        f"{b_emoji} {bias} | {leverage}x\n\n"
        f"📥 Entry:   ${entry:.6f}\n"
        f"💰 Current: ${current:.6f}\n\n"
        f"📊 Raw:       {raw_pct:+.2f}%\n"
        f"{p_emoji} Lev PnL: {lev_pnl:+.2f}%\n\n"
        f"💡 Run /compare {symbol.replace('USDT','')} for the full breakdown."
    )

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(
            "🔄 Refresh again",
            callback_data=f"custom_compare_refresh|{exchange}|{symbol}|{entry}|{bias}|{leverage}"
        ),
        InlineKeyboardButton("🔗 Trade", url=get_exchange_link(exchange, symbol))
    ]])
    try:
        await query.edit_message_text(msg, reply_markup=keyboard)
    except Exception:
        pass

async def compare_full_refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Refresh handler for /compare (no-args) full panel.
    Edits the original message in-place and maintains the exact same
    'FULL COMPARE' panel format including win/loss counts.
    callback_data: cmp_full_refresh|{chat_id}
    """
    query   = update.callback_query
    await query.answer("Refreshing all prices…")

    if not state.last_scan_results:
        try:
            await query.edit_message_text("⚠️ No scan data. Run /scan first.")
        except Exception:
            pass
        return

    age   = datetime.now() - state.last_scan_time
    mins  = int(age.total_seconds() // 60)
    now_s = datetime.now().strftime('%H:%M:%S')

    lines     = [f"📊 FULL COMPARE — {now_s}\n🕐 Scan: {mins} min ago\n{'━'*30}\n"]
    winners   = 0
    losers    = 0
    best_pnl  = ('', -9999)
    worst_pnl = ('', 9999)

    parts   = query.data.split('|')
    chat_id = int(parts[1])

    for i, r in enumerate(state.last_scan_results[:15], 1):
        exchange = r['exchange']
        symbol   = r['symbol']
        entry    = r['price']
        bias     = r['bias']
        lev_data = r.get('leverage')
        leverage = lev_data['suggested'] if lev_data else 1

        current = _get_live_price(symbol, exchange)

        if current == 0:
            lines.append(f"#{i} {symbol} — ⚠️ price unavailable\n")
            continue

        raw_pct = (current - entry)/entry*100 if bias=='LONG' else (entry-current)/entry*100
        lev_pnl = raw_pct * leverage
        p_emoji = "🟢" if lev_pnl >= 0 else "🔴"
        b_emoji = "🟢" if bias == "LONG" else "🔴"

        if lev_pnl >= 0: winners += 1
        else:            losers  += 1
        if lev_pnl > best_pnl[1]:  best_pnl  = (symbol, lev_pnl)
        if lev_pnl < worst_pnl[1]: worst_pnl = (symbol, lev_pnl)

        # Milestone
        milestone = ""
        if bias == 'LONG':
            if current >= r['t3']:          milestone = " 🎯🎯🎯"
            elif current >= r['t2']:        milestone = " 🎯🎯"
            elif current >= r['t1']:        milestone = " 🎯"
            elif current <= r['stop_loss']: milestone = " 🛑"
        else:
            if current <= r['t3']:          milestone = " 🎯🎯🎯"
            elif current <= r['t2']:        milestone = " 🎯🎯"
            elif current <= r['t1']:        milestone = " 🎯"
            elif current >= r['stop_loss']: milestone = " 🛑"

        lines.append(
            f"{b_emoji} #{i} {symbol} {bias} {leverage}x{milestone}\n"
            f"   ${entry:.4f} → ${current:.4f}  ({raw_pct:+.2f}%)\n"
            f"   {p_emoji} Lev PnL: {lev_pnl:+.2f}%\n"
        )
        await asyncio.sleep(0.05)

    total_done = winners + losers
    lines.append(f"{'━'*30}")
    lines.append(f"�� Winning: {winners}  🔴 Losing: {losers}  📊 Tracked: {total_done}")
    if best_pnl[0]:  lines.append(f"🏆 Best:  {best_pnl[0]}  {best_pnl[1]:+.2f}%")
    if worst_pnl[0]: lines.append(f"💀 Worst: {worst_pnl[0]}  {worst_pnl[1]:+.2f}%")
    lines.append(f"\n💡 /compare ZEC — track a specific pair")
    lines.append(f"💡 /compare ZEC 20x — with custom leverage")

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔄 Refresh All PnL", callback_data=f"cmp_full_refresh|{chat_id}")
    ]])
    try:
        await query.edit_message_text("\n".join(lines), reply_markup=keyboard)
    except Exception:
        pass



# Filters out signals older than their hold_hours
# ───────────────����────��───────────────────��────
def filter_live_signals(results):
    """Return signals that have not yet exceeded their recommended hold duration."""
    now = datetime.now()
    live = []
    for r in results:
        scan_dt = r.get('scan_time')
        if not isinstance(scan_dt, datetime):
            live.append(r)
            continue
        hold_hours = r.get('hold_hours', 24)
        age_hours  = (now - scan_dt).total_seconds() / 3600
        if age_hours <= hold_hours:
            live.append(r)
    return live


# ─────────────────────────────────────────────
# /watch — Personal watchlist
# /watch BTCUSDT [min_conf]  — add to watchlist
# /watch                     — show watchlist
# /unwatch BTCUSDT           — remove
# Bot pings every 30 min when signal appears
# ─────��───��───��───────────────────────────────
async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args

    if not args:
        wl = db_get_watchlist(chat_id)
        if not wl:
            await update.message.reply_text(
                "👁 WATCHLIST\n\n"
                "No pairs on your watchlist yet.\n\n"
                "Add one:\n"
                "/watch BTCUSDT       — notify when signal conf ≥ 6\n"
                "/watch ETHUSDT 8     — notify only if conf ≥ 8\n"
                "/unwatch BTCUSDT     — remove from watchlist\n\n"
                "Bot checks your watchlist every 30 minutes."
            )
        else:
            lines = ["👁 YOUR WATCHLIST\n━━━━━━━━━━━━━━━���━━━━━━━━━━━━━━\n"]
            for sym, mc in wl:
                lines.append(f"  • {sym}  (min confidence: {mc}/10)")
            lines.append("\n/unwatch SYMBOL to remove")
            await update.message.reply_text("\n".join(lines))
        return

    raw    = args[0].upper().replace('_USDT','USDT').replace('/','')
    symbol = raw if raw.endswith('USDT') else raw + 'USDT'
    min_conf = int(args[1]) if len(args) > 1 and args[1].isdigit() else 6
    min_conf = max(1, min(10, min_conf))
    db_add_watch(chat_id, symbol, min_conf)
    await update.message.reply_text(
        f"✅ Added {symbol} to your watchlist!\n"
        f"Min confidence: {min_conf}/10\n\n"
        f"I'll alert you whenever {symbol} appears in a scan.\n"
        f"Checked every 30 minutes automatically."
    )

async def unwatch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args
    if not args:
        await update.message.reply_text("Usage: /unwatch BTCUSDT"); return
    raw    = args[0].upper().replace('_USDT','USDT').replace('/','')
    symbol = raw if raw.endswith('USDT') else raw + 'USDT'
    db_remove_watch(chat_id, symbol)
    await update.message.reply_text(f"✅ Removed {symbol} from your watchlist.")


async def safemode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /safemode — Toggle automatic dying-trend alerts.

    When ON, the bot monitors every signal you receive from ANY source
    (/scan, /scalp, /swing, /cscan, /autoscan, etc.) and automatically
    alerts you if the trend starts showing reversal signs — even if you
    never used /pick to track the trade.

    The check runs every 30 minutes in the background.
    """
    _track(update)
    chat_id = update.effective_chat.id

    if chat_id in state.safemode_users:
        # Turn OFF
        state.safemode_users.discard(chat_id)
        safemode_last_signals.pop(chat_id, None)
        db_safemode_disable(chat_id)
        await update.message.reply_text(
            "🛡️ Safe Mode *disabled*.\n\n"
            "You will no longer receive automatic dying-trend alerts.\n"
            "Use /safemode again to re-enable.",
            parse_mode="Markdown"
        )
    else:
        # Turn ON
        state.safemode_users.add(chat_id)
        db_safemode_enable(chat_id)
        await update.message.reply_text(
            "🛡️ Safe Mode *enabled* ✅\n\n"
            "I'll now automatically alert you whenever a trend from your scans "
            "starts dying — no need to use /pick.\n\n"
            "Works with: /scan, /scalp, /swing, /cscan, /autoscan and all other scan sources.\n\n"
            "Checks run every 30 minutes. Use /safemode again to turn off.",
            parse_mode="Markdown"
        )


async def watchlist_check_job(context: ContextTypes.DEFAULT_TYPE):
    """Runs every 30 min. Alerts watchlist users if their symbol has a signal."""
    if not state.last_scan_results:
        return
    all_watches = db_get_all_watchlist()
    if not all_watches:
        return

    # Build symbol map from last scan (exchange-agnostic)
    sym_map = {}
    for r in state.last_scan_results:
        sym = r['symbol']
        if sym not in sym_map or r['confidence'] > sym_map[sym]['confidence']:
            sym_map[sym] = r

    # Only process live (non-expired) signals
    live_sym_map = {}
    for sym, r in sym_map.items():
        if filter_live_signals([r]):
            live_sym_map[sym] = r

    for chat_id, symbol, min_conf in all_watches:
        if symbol in live_sym_map:
            r = live_sym_map[symbol]
            if r['confidence'] >= min_conf:
                emoji = "🟢" if r['bias'] == 'LONG' else "🔴"
                age_m = int((datetime.now() - r.get('scan_time', datetime.now())).total_seconds() // 60)
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"👁 WATCHLIST ALERT: {symbol}\n\n"
                            f"{emoji} {r['bias']} on {r['exchange']}\n"
                            f"⭐ Confidence: {r['confidence']}/10\n"
                            f"💰 Price: ${r['price']:.6f}\n"
                            f"📥 Entry: ${r['entry_low']:.6f} → ${r['entry_high']:.6f}\n"
                            f"🛑 SL: ${r['stop_loss']:.6f}\n"
                            f"🎯 T1: ${r['t1']:.6f}\n"
                            f"⏱ Hold: {r['hold']} | Scan: {age_m} min ago\n\n"
                            f"Use /pick to track this trade."
                        )
                    )
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning("Watchlist notify failed for %s/%s: %s", chat_id, symbol, e)


# ─────────────────────────────────────────────
# FUNDING RATE EXTREMES ALERT
# Checks Bybit/Binance funding every hour.
# Sends alert if funding < -0.05% or > 0.05%
# ─────────────────────────────────────────────
async def funding_alert_job(context: ContextTypes.DEFAULT_TYPE):
    """Hourly job: scan top 20 symbols for extreme funding rates."""
    if not auto_scan_subscribers:
        return

    extremes = []

    # Check Bybit top symbols
    if sakz_exchanges.BYBIT_AVAILABLE:
        top_syms = bybit_get_top_symbols(20)
        for sym in top_syms:
            try:
                fr = bybit_fetch_funding(sym)
                fp = fr * 100
                if abs(fp) >= 0.05:
                    bias_hint = "SHORT squeeze risk 🚀" if fp < 0 else "LONG squeeze risk 📉"
                    extremes.append({'symbol': sym, 'exchange': 'BYBIT', 'rate': fp, 'hint': bias_hint})
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.debug("funding scan (BYBIT) skipped %s: %s", sym, e)

    # Check Binance top symbols
    if sakz_exchanges.BINANCE_AVAILABLE:
        top_syms = binance_get_top_symbols(20)
        for sym in top_syms:
            try:
                fr = binance_fetch_funding(sym)
                fp = fr * 100
                if abs(fp) >= 0.05:
                    bias_hint = "SHORT squeeze risk 🚀" if fp < 0 else "LONG squeeze risk 📉"
                    extremes.append({'symbol': sym, 'exchange': 'BINANCE', 'rate': fp, 'hint': bias_hint})
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.debug("funding scan (BINANCE) skipped %s: %s", sym, e)

    if not extremes:
        return

    extremes.sort(key=lambda x: abs(x['rate']), reverse=True)
    lines = [f"⚡ EXTREME FUNDING ALERT — {datetime.now().strftime('%H:%M')}\n"
             f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]
    for e in extremes[:10]:
        emoji = "🔴" if e['rate'] > 0 else "🟢"
        lines.append(f"{emoji} {e['exchange']} | {e['symbol']}\n"
                     f"   Funding: {e['rate']:+.4f}% — {e['hint']}\n")
    lines.append("\nExtreme funding = high probability reversal signal.")

    msg = "\n".join(lines)
    for chat_id in list(auto_scan_subscribers):
        try:
            await context.bot.send_message(chat_id=chat_id, text=msg)
        except Exception as e:
            logger.warning("Funding alert failed for %s: %s", chat_id, e)


# ───────────────────────────────────────────���─
# BTC VOLATILITY MONITOR — Smart re-scan trigger
# Checks BTC price every 15 min.
# If BTC moved >3% in 1 hour → emergency re-scan
# ─────────────────────────────────────────────
def _btc_regime_bucket(regime: str) -> str:
    """Collapse the 5 regime labels into a tradable direction bucket."""
    if regime in _BTC_BULL_REGIMES:
        return "BULLISH"
    if regime in _BTC_BEAR_REGIMES:
        return "BEARISH"
    return "NEUTRAL"


def _btc_alert_recipients() -> set:
    """Union of every subscriber surface, deduplicated to one chat per id.
    Sources: autoscan subs + /pro subs + /prime subs + broadcast channels.
    A user subscribed to several still receives exactly one alert."""
    ids: set = set()
    try:
        ids.update(int(c) for c in auto_scan_subscribers.keys())
    except Exception as e:
        logger.warning("BTC alert: autoscan recipients failed: %s", e)
    for label, fetch in (
        ("pro",       lambda: db_pro_get_all_subscribers()),
        ("prime",     lambda: [u["chat_id"] for u in db_prime_get_all_users()]),
        ("broadcast", lambda: db_get_broadcast_channels()),
    ):
        try:
            ids.update(int(c) for c in fetch())
        except Exception as e:
            logger.warning("BTC alert: %s recipients failed: %s", label, e)
    return ids


def _btc_alert_message(regime: str, bucket: str, price: float, move_pct: float) -> str:
    """Build the counter-trend risk advisory for a BTC direction shift."""
    label = regime.replace("_", " ").title()
    if bucket == "BULLISH":
        head   = "🟢 BTC TURNING BULLISH"
        advice = (
            "⚠️ Counter-trend risk for SHORTS\n"
            "BTC is showing bullish strength, and it drives the whole market.\n"
            "A counter-trend bounce can squeeze short positions.\n\n"
            "• Minimize risk / tighten stops on SHORT positions\n"
            "• Consider trimming or holding off on new shorts\n"
            "• Longs are favored while BTC stays strong"
        )
    else:  # BEARISH
        head   = "🔴 BTC TURNING BEARISH"
        advice = (
            "⚠️ Counter-trend risk for LONGS\n"
            "BTC is showing bearish weakness, and it drives the whole market.\n"
            "A counter-trend drop can flush long positions.\n\n"
            "• Minimize risk / tighten stops on LONG positions\n"
            "• Consider trimming or holding off on new longs\n"
            "• Shorts are favored while BTC stays weak"
        )
    return (
        "🚨 BTC MARKET SHIFT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{head}   ({label})\n"
        f"BTC: ${price:,.0f}   ·   {move_pct:+.2f}% over last ~10m\n\n"
        f"{advice}\n\n"
        "ℹ️ Advisory only — not financial advice. Manage your own risk.\n"
        "Use /scan or /best for fresh setups."
    )


async def _btc_alert_broadcast(bot, regime: str, bucket: str, price: float, move_pct: float) -> int:
    """Send the advisory to every subscriber, throttled for Telegram limits."""
    recipients = _btc_alert_recipients()
    if not recipients:
        logger.info("BTC regime alert (%s): no subscribers to notify", bucket)
        return 0
    msg  = _btc_alert_message(regime, bucket, price, move_pct)
    sent = 0
    for i, chat_id in enumerate(recipients, 1):
        try:
            await bot.send_message(chat_id=chat_id, text=msg)
            sent += 1
        except Exception as e:
            logger.warning("BTC regime alert failed for %s: %s", chat_id, e)
        if i % 25 == 0:
            await asyncio.sleep(1)   # ~25 msgs/sec — under Telegram's ~30/s global cap
    logger.info("BTC regime alert (%s) delivered to %d/%d chats", bucket, sent, len(recipients))
    return sent


async def btc_regime_alert_job(context: ContextTypes.DEFAULT_TYPE):
    """Every 60s: detect a genuine BTC regime *direction* shift and broadcast a
    counter-trend risk advisory to all subscribers (deduplicated to one msg/chat).

    Hybrid fire condition — ALL must hold:
      1. direction differs from the last alerted direction (real state change)
      2. direction is not NEUTRAL (skip ambiguous middle states)
      3. the new direction has held >= _BTC_ALERT_HOLD_SECONDS (whipsaw guard)
         AND BTC price has moved >= _BTC_ALERT_CONFIRM_PCT in that direction over
         the hold window (the price-action confirmation half of the hybrid).
    Anti-spam is the state-change itself: it only fires on a transition, so it
    won't repeat unless BTC actually reverses direction. State persists in the DB.
    """
    btc_price = _get_live_price("BTCUSDT", "BYBIT")
    if not btc_price:
        return
    try:
        db_save_btc_price(btc_price)
    except Exception:
        pass

    bucket = _btc_regime_bucket(get_btc_regime())
    now    = datetime.now()

    try:
        st = db_btc_alert_get_state()
    except Exception as e:
        logger.warning("BTC alert state read failed: %s", e)
        return

    last_alerted  = st.get("last_regime")
    pending       = st.get("pending_regime")
    pending_since = st.get("pending_since")
    pending_price = st.get("pending_price")

    def _save(**kw):
        base = {
            "last_regime":    last_alerted,
            "pending_regime": pending,
            "pending_since":  pending_since,
            "pending_price":  pending_price,
            "last_alert_ts":  st.get("last_alert_ts"),
            "last_price":     btc_price,
        }
        base.update(kw)
        try:
            db_btc_alert_save_state(**base)
        except Exception as e:
            logger.warning("BTC alert state save failed: %s", e)

    # Ambiguous middle state — never alert; drop any pending candidate.
    if bucket == "NEUTRAL":
        if pending is not None:
            _save(pending_regime=None, pending_since=None, pending_price=None)
        return

    # Same direction we already alerted on — nothing new to say.
    if bucket == last_alerted:
        if pending is not None:
            _save(pending_regime=None, pending_since=None, pending_price=None)
        return

    # New direction — (re)start the 10-minute hold timer and anchor the price.
    if pending != bucket:
        _save(pending_regime=bucket, pending_since=now.isoformat(), pending_price=btc_price)
        return

    # Candidate direction is holding — has it held long enough?
    try:
        since = datetime.fromisoformat(pending_since) if pending_since else now
    except Exception:
        since = now
    if (now - since).total_seconds() < _BTC_ALERT_HOLD_SECONDS:
        return

    # Hybrid price-action confirmation over the hold window.
    move_pct = ((btc_price - pending_price) / pending_price * 100.0) if pending_price else 0.0
    if bucket == "BULLISH" and move_pct < _BTC_ALERT_CONFIRM_PCT:
        return
    if bucket == "BEARISH" and move_pct > -_BTC_ALERT_CONFIRM_PCT:
        return

    # All conditions met — broadcast, then lock in the new alerted direction.
    regime_label = get_btc_regime()
    await _btc_alert_broadcast(context.bot, regime_label, bucket, btc_price, move_pct)
    last_alerted = bucket
    _save(last_regime=bucket, pending_regime=None, pending_since=None,
          pending_price=None, last_alert_ts=now.isoformat())


async def btc_volatility_job(context: ContextTypes.DEFAULT_TYPE):
    """Every 15 min: snapshot BTC price & check for >3% 1h move."""
    # Get current BTC price
    btc_price = _get_live_price('BTCUSDT', 'BYBIT')
    if btc_price == 0:
        return

    db_save_btc_price(btc_price)
    price_1h_ago = db_get_btc_price_1h_ago()

    if not price_1h_ago:
        return

    pct_change = abs((btc_price - price_1h_ago) / price_1h_ago * 100)
    if pct_change < 3.0:
        return

    direction = "🚀 PUMPING" if btc_price > price_1h_ago else "📉 DUMPING"
    logger.info("BTC volatility spike detected: %.2f%% — triggering emergency scan", pct_change)

    # Trigger emergency re-scan
    results, _ = await get_scan_results(force=True)

    if not results or not auto_scan_subscribers:
        return

    longs  = sum(1 for r in results if r['bias'] == 'LONG')
    shorts = sum(1 for r in results if r['bias'] == 'SHORT')
    best   = results[0]
    emoji  = "🟢" if best['bias'] == 'LONG' else "🔴"

    msg = (
        f"🚨 EMERGENCY SCAN — BTC VOLATILITY SPIKE\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"BTC {direction}: {pct_change:+.2f}% in 1 hour\n"
        f"BTC: ${price_1h_ago:.0f} → ${btc_price:.0f}\n\n"
        f"��� Triggered emergency market scan\n"
        f"📊 {len(results)} signals | 🟢 {longs}L  🔴 {shorts}S\n\n"
        f"🏆 BEST NOW: {emoji} {best['exchange']} {best['symbol']}\n"
        f"   {best['bias']} | {best['confidence']}/10 | Hold {best['hold']}\n\n"
        f"Use /best  /top5  /filter for full results"
    )
    for chat_id in list(auto_scan_subscribers):
        try:
            await context.bot.send_message(chat_id=chat_id, text=msg)
        except Exception as e:
            logger.warning("BTC volatility notify failed for %s: %s", chat_id, e)


# ─────────────────────────────────────────────
# /broadcast — Register a channel/group for
# auto-posting top signals after every scan.
# Usage: run in the target channel as admin.
# /broadcast on   — enable for this chat
# /broadcast off  — disable for this chat
# ─────────────────────���─────────���────────��────
async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args

    if not args or args[0].lower() not in ('on', 'off'):
        channels = db_get_broadcast_channels()
        status   = "🟢 ACTIVE" if chat_id in channels else "🔴 INACTIVE"
        await update.message.reply_text(
            f"📡 BROADCAST MODE\n\n"
            f"Status in this chat: {status}\n\n"
            f"When broadcast is ON, this chat will automatically receive\n"
            f"top signals after every scan (every 4 hours).\n\n"
            f"/broadcast on   — enable auto-posting here\n"
            f"/broadcast off  — disable\n\n"
            f"💡 Works in groups, channels, and DMs."
        )
        return

    if args[0].lower() == 'on':
        db_add_broadcast(chat_id)
        await update.message.reply_text(
            f"✅ BROADCAST ON!\n\n"
            f"This chat will now receive top signals automatically\n"
            f"after every scan (every 4 hours + emergency scans).\n\n"
            f"/broadcast off to disable."
        )
    else:
        db_remove_broadcast(chat_id)
        await update.message.reply_text(
            f"🔕 Broadcast disabled for this chat.\n"
            f"/broadcast on to re-enable."
        )


async def post_broadcast(results, bot):
    """Post top signals to all broadcast channels after a scan."""
    channels = db_get_broadcast_channels()
    if not channels or not results:
        return

    live    = filter_live_signals(results)
    top5    = live[:5]
    longs   = sum(1 for r in results if r['bias'] == 'LONG')
    shorts  = sum(1 for r in results if r['bias'] == 'SHORT')
    hc      = sum(1 for r in results if r['confidence'] >= 8)
    best    = results[0]
    emoji   = "🟢" if best['bias'] == 'LONG' else "🔴"
    ts      = datetime.now().strftime('%Y-%m-%d %H:%M')

    header = (
        f"🤖 SAKZ SCAN — {ts}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 {len(results)} signals | 🟢 {longs}L 🔴 {shorts}S | ⭐ {hc} high-conf\n\n"
        f"🏆 BEST: {emoji} {best['exchange']} {best['symbol']} "
        f"{best['bias']} {best['confidence']}/10 | Hold {best['hold']}\n\n"
        f"TOP 5 SIGNALS\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━��━\n"
    )
    lines = [header]
    for i, r in enumerate(top5, 1):
        e = "🟢" if r['bias'] == 'LONG' else "🔴"
        lev = r.get('leverage')
        lev_str = f" | {lev['suggested']}x" if lev else ""
        lines.append(
            f"{e} #{i} {r['exchange']} | {r['symbol']}\n"
            f"   {r['bias']} {r['confidence']}/10{lev_str} | Hold {r['hold']}\n"
            f"   Entry: ${r['entry_low']:.4f}–${r['entry_high']:.4f} | SL: ${r['stop_loss']:.4f}\n"
        )
    lines.append("\nType /scan in bot DM for full analysis & PnL calculator.")
    msg = "\n".join(lines)

    for chat_id in list(channels):
        try:
            await bot.send_message(chat_id=chat_id, text=msg)
        except Exception as e:
            logger.warning("Broadcast failed for chat %s: %s", chat_id, e)


# ──────────��──────────────────────���─────���─────
# /leaderboard [hours] — Best performing pairs
# Usage: /leaderboard          → all time
#        /leaderboard 24       → last 24 hours
#        /leaderboard 168      → last 7 days
# ──────────────────────────────────��──────────
async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    args = context.args
    hours_filter = None
    label = "ALL TIME"

    if args:
        try:
            hours_filter = int(args[0])
            if hours_filter <= 0:
                raise ValueError
            label = f"LAST {hours_filter}H" if hours_filter < 24 else f"LAST {hours_filter // 24}D" if hours_filter % 24 == 0 else f"LAST {hours_filter}H"
        except ValueError:
            await update.message.reply_text(
                "⚠️ Invalid time.\n"
                "/leaderboard        → all time\n"
                "/leaderboard 24     → last 24h\n"
                "/leaderboard 168    → last 7 days"
            )
            return

    conn = db_connect()
    c    = conn.cursor()
    if hours_filter:
        cutoff = (datetime.now() - timedelta(hours=hours_filter)).isoformat()
        c.execute("""SELECT symbol, exchange, bias, outcome, confidence
                     FROM signal_outcomes
                     WHERE outcome != 'pending' AND scan_time >= ?""", (cutoff,))
    else:
        c.execute("""SELECT symbol, exchange, bias, outcome, confidence
                     FROM signal_outcomes WHERE outcome != 'pending'""")
    rows = c.fetchall()
    conn.close()

    if not rows:
        await update.message.reply_text(
            f"📊 No completed outcomes for {label}.\n"
            f"Try /leaderboard 168 for a wider window."
        )
        return

    # Aggregate per symbol
    symbol_stats = {}
    for r in rows:
        key = r['symbol']
        if key not in symbol_stats:
            symbol_stats[key] = {'exchange': r['exchange'], 'wins': 0, 'total': 0, 'best_conf': 0}
        symbol_stats[key]['total'] += 1
        if r['outcome'] in ('t1_hit', 't2_hit', 't3_hit'):
            symbol_stats[key]['wins'] += 1
        symbol_stats[key]['best_conf'] = max(symbol_stats[key]['best_conf'], r['confidence'])

    # Require at least 2 signals to appear on leaderboard
    ranked = [
        {'symbol': sym, **stats,
         'win_rate': stats['wins'] / stats['total'] * 100}
        for sym, stats in symbol_stats.items()
        if stats['total'] >= 2
    ]
    ranked.sort(key=lambda x: (x['win_rate'], x['wins']), reverse=True)

    top = ranked[:15]
    if not top:
        await update.message.reply_text(
            f"📊 Not enough data for {label}.\n"
            f"Each pair needs at least 2 tracked signals.\n"
            f"Try /leaderboard 720 for a 30-day view."
        )
        return

    # Inline time filter buttons
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("24H",  callback_data="lb_time|24"),
        InlineKeyboardButton("7D",   callback_data="lb_time|168"),
        InlineKeyboardButton("30D",  callback_data="lb_time|720"),
        InlineKeyboardButton("All",  callback_data="lb_time|0"),
    ]])

    lines = [
        f"🏆 SIGNAL LEADERBOARD — {label}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Min 2 tracked signals required\n"
    ]
    medals = ["🥇","🥈","🥉"] + ["🔹"] * 20
    for i, r in enumerate(top):
        bar_w = int(r['win_rate'] / 10)
        bar   = "█" * bar_w + "░" * (10 - bar_w)
        lines.append(
            f"{medals[i]} #{i+1} {r['exchange']} | {r['symbol']}\n"
            f"   {bar} {r['win_rate']:.0f}%  ({r['wins']}/{r['total']}) | Best conf: {r['best_conf']}/10\n"
        )

    lines.append("\nQuick filter 👇")
    await update.message.reply_text("\n".join(lines), reply_markup=keyboard)


async def leaderboard_time_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inline button handler for leaderboard time filters."""
    query = update.callback_query
    await query.answer()
    hours_str    = query.data.split('|')[1]
    hours_filter = int(hours_str)

    if hours_filter == 0:
        label  = "ALL TIME"
        cutoff = None
    elif hours_filter < 24:
        label  = f"LAST {hours_filter}H"
        cutoff = (datetime.now() - timedelta(hours=hours_filter)).isoformat()
    elif hours_filter % 24 == 0:
        label  = f"LAST {hours_filter // 24}D"
        cutoff = (datetime.now() - timedelta(hours=hours_filter)).isoformat()
    else:
        label  = f"LAST {hours_filter}H"
        cutoff = (datetime.now() - timedelta(hours=hours_filter)).isoformat()

    conn = db_connect()
    c    = conn.cursor()
    if cutoff:
        c.execute("""SELECT symbol, exchange, bias, outcome, confidence
                     FROM signal_outcomes WHERE outcome != 'pending' AND scan_time >= ?""", (cutoff,))
    else:
        c.execute("""SELECT symbol, exchange, bias, outcome, confidence
                     FROM signal_outcomes WHERE outcome != 'pending'""")
    rows = c.fetchall()
    conn.close()

    if not rows:
        await query.edit_message_text(f"📊 No data for {label}. Try a wider window.")
        return

    symbol_stats = {}
    for r in rows:
        key = r['symbol']
        if key not in symbol_stats:
            symbol_stats[key] = {'exchange': r['exchange'], 'wins': 0, 'total': 0, 'best_conf': 0}
        symbol_stats[key]['total'] += 1
        if r['outcome'] in ('t1_hit','t2_hit','t3_hit'):
            symbol_stats[key]['wins'] += 1
        symbol_stats[key]['best_conf'] = max(symbol_stats[key]['best_conf'], r['confidence'])

    ranked = [
        {'symbol': sym, **stats, 'win_rate': stats['wins'] / stats['total'] * 100}
        for sym, stats in symbol_stats.items() if stats['total'] >= 2
    ]
    ranked.sort(key=lambda x: (x['win_rate'], x['wins']), reverse=True)
    top = ranked[:15]

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("24H",  callback_data="lb_time|24"),
        InlineKeyboardButton("7D",   callback_data="lb_time|168"),
        InlineKeyboardButton("30D",  callback_data="lb_time|720"),
        InlineKeyboardButton("All",  callback_data="lb_time|0"),
    ]])

    if not top:
        await query.edit_message_text(
            f"📊 Not enough data for {label}. Try a wider window.",
            reply_markup=keyboard
        )
        return

    lines = [f"🏆 SIGNAL LEADERBOARD — {label}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]
    medals = ["🥇","🥈","🥉"] + ["🔹"] * 20
    for i, r in enumerate(top):
        bar_w = int(r['win_rate'] / 10)
        bar   = "█" * bar_w + "░" * (10 - bar_w)
        lines.append(
            f"{medals[i]} #{i+1} {r['exchange']} | {r['symbol']}\n"
            f"   {bar} {r['win_rate']:.0f}% ({r['wins']}/{r['total']}) | Best conf: {r['best_conf']}/10\n"
        )
    lines.append("\nQuick filter 👇")
    await query.edit_message_text("\n".join(lines), reply_markup=keyboard)


# ═════════════════════════════════════════════��═════════════════
# 🐌  S N A I L   M O D E  — HIDDEN PREMIUM FEATURE
# ═══════════════���═════════════════════════════���═══���════════════��
# Access gate: user must first type the secret passphrase
#   /scan1234JP$$
# That unlocks the /snail command for that chat permanently.
# Until unlocked, /snail and /scan1234JP$$ produce ZERO output.
# ═══════════════════════════════════════════════════════════════

# ── Snail DB helpers ──────────────────────────












# ── Snail signal analysis engine ──────────────
def snail_score_signal(r, df4h, df1d, funding):
    """
    Deep quality scoring for SNAIL mode.
    Returns a dict with snail_score (0-100), verdict, manipulation risk,
    fundamental flags, and full reasoning.
    Requires confidence=10 AND snail_score >= 80 to qualify.
    """
    try:
        L   = df4h.iloc[-1]
        P   = df4h.iloc[-2]
        P2  = df4h.iloc[-3]
        # FIX #3 — last CLOSED daily candle (iloc[-1] is still forming)
        LD  = df1d.iloc[-2]
        PD  = df1d.iloc[-3]
        PD2 = df1d.iloc[-3]

        price  = L['close']
        bias   = r['bias']
        atr    = L['atr']
        rsi4   = L['rsi']
        rsi_d  = LD['rsi']
        vol    = L['volume']
        vol_ma = L['volume_ma']

        score       = 0
        reasons     = []
        warnings    = []
        manip_risk  = 0
        manip_notes = []

        # ── 1. TREND ALIGNMENT (multi-timeframe) ──────────────
        ema20_4h = L['ema20']; ema50_4h = L['ema50']
        ema20_1d = LD['ema20']; ema50_1d = LD['ema50']

        if bias == 'LONG':
            if price > ema20_4h > ema50_4h:
                score += 15; reasons.append("✅ 4H EMA stack fully bullish (price > EMA20 > EMA50)")
            if ema20_1d > ema50_1d:
                score += 15; reasons.append("✅ Daily EMA stack bullish — macro trend aligned")
            elif ema20_1d < ema50_1d:
                score -= 10; warnings.append("⚠️ Daily EMAs bearish — counter-trend trade, higher risk")
        else:
            if price < ema20_4h < ema50_4h:
                score += 15; reasons.append("✅ 4H EMA stack fully bearish (price < EMA20 < EMA50)")
            if ema20_1d < ema50_1d:
                score += 15; reasons.append("✅ Daily EMA stack bearish �� macro trend aligned")
            elif ema20_1d > ema50_1d:
                score -= 10; warnings.append("⚠️ Daily EMAs bullish — counter-trend short, higher risk")

        # ── 2. MOMENTUM QUALITY ──────────────────────────────
        macd_4h   = L['macd_diff']
        macd_4h_p = P['macd_diff']
        macd_1d   = LD['macd_diff']
        macd_1d_p = PD['macd_diff']

        if bias == 'LONG':
            if macd_4h > macd_4h_p > 0:
                score += 10; reasons.append("✅ MACD 4H accelerating bullish — momentum building")
            if macd_1d > 0 and macd_1d > macd_1d_p:
                score += 10; reasons.append("✅ Daily MACD trending bullish — higher timeframe momentum")
        else:
            if macd_4h < macd_4h_p < 0:
                score += 10; reasons.append("✅ MACD 4H accelerating bearish — selling pressure building")
            if macd_1d < 0 and macd_1d < macd_1d_p:
                score += 10; reasons.append("✅ Daily MACD trending bearish — higher timeframe momentum")

        # ── 3. RSI QUALITY ───────────────────────────────────
        if bias == 'LONG':
            if 40 <= rsi4 <= 60:
                score += 8; reasons.append(f"✅ RSI 4H in healthy bullish zone ({rsi4:.1f}) — room to run")
            elif rsi4 < 35:
                score += 5; reasons.append(f"✅ RSI 4H oversold ({rsi4:.1f}) — bounce fuel")
            elif rsi4 > 65:
                score -= 5; warnings.append(f"⚠️ RSI 4H elevated ({rsi4:.1f}) — limited upside room")
            if 40 <= rsi_d <= 60:
                score += 7; reasons.append(f"✅ Daily RSI in trend zone ({rsi_d:.1f}) — sustainable")
        else:
            if 40 <= rsi4 <= 60:
                score += 8; reasons.append(f"✅ RSI 4H in healthy bearish zone ({rsi4:.1f}) — room to fall")
            elif rsi4 > 65:
                score += 5; reasons.append(f"✅ RSI 4H overbought ({rsi4:.1f}) — dump fuel")
            elif rsi4 < 35:
                score -= 5; warnings.append(f"⚠️ RSI 4H oversold ({rsi4:.1f}) — limited downside room")
            if 40 <= rsi_d <= 60:
                score += 7; reasons.append(f"✅ Daily RSI in bearish zone ({rsi_d:.1f}) — sustainable")

        # ── 4. VOLUME CONVICTION ─��───────────────────────────
        vol_ratio       = vol / vol_ma if vol_ma > 0 else 1.0
        price_change_pct = abs(price - P['close']) / P['close'] * 100 if P['close'] > 0 else 0

        if vol_ratio > 3.0 and price_change_pct < 0.5:
            manip_risk += 25
            manip_notes.append("🚩 Massive volume with tiny price move — potential wash trading")
        elif vol_ratio > 2.0:
            if (bias == 'LONG' and price > P['close']) or (bias == 'SHORT' and price < P['close']):
                score += 10; reasons.append(f"✅ Volume surge {vol_ratio:.1f}x avg confirming direction")
            else:
                manip_risk += 15
                manip_notes.append(f"🚩 High volume ({vol_ratio:.1f}x) against price direction — suspect")
        elif vol_ratio > 1.3:
            score += 5; reasons.append(f"✅ Above-average volume ({vol_ratio:.1f}x) supports move")
        elif vol_ratio < 0.7:
            score -= 5; warnings.append(f"⚠️ Low volume ({vol_ratio:.1f}x avg) — weak conviction")

        # ── 5. CANDLE PATTERN QUALITY ─────────────────��──────
        candles         = df4h.iloc[-4:-1]
        bullish_candles = sum(1 for _, c in candles.iterrows() if c['close'] > c['open'])
        bearish_candles = 3 - bullish_candles

        if bias == 'LONG' and bullish_candles >= 2:
            score += 8; reasons.append(f"✅ {bullish_candles}/3 recent 4H candles bullish — consistent pressure")
        elif bias == 'SHORT' and bearish_candles >= 2:
            score += 8; reasons.append(f"✅ {bearish_candles}/3 recent 4H candles bearish — consistent selling")
        else:
            warnings.append("⚠️ Mixed candle pattern — momentum not yet fully established")

        # ── 6. STRUCTURE / KEY LEVELS ────────────────────────
        support    = L['support']
        resistance = L['resistance']

        if bias == 'LONG':
            dist_from_support = (price - support) / atr if atr > 0 else 5
            if dist_from_support < 1.0:
                score += 10; reasons.append(f"✅ Price near key support (${support:.4f}) — strong bounce base")
            r_space = (resistance - price) / atr if atr > 0 else 0
            if r_space > 3.0:
                score += 8; reasons.append(f"✅ Wide runway to resistance �� {r_space:.1f}x ATR of clear space")
            elif r_space < 1.5:
                score -= 8; warnings.append(f"⚠️ Resistance close ({r_space:.1f}x ATR) — limited upside before wall")
        else:
            dist_from_res = (resistance - price) / atr if atr > 0 else 5
            if dist_from_res < 1.0:
                score += 10; reasons.append(f"✅ Price near key resistance (${resistance:.4f}) — rejection base")
            s_space = (price - support) / atr if atr > 0 else 0
            if s_space > 3.0:
                score += 8; reasons.append(f"✅ Wide runway to support — {s_space:.1f}x ATR of downside space")
            elif s_space < 1.5:
                score -= 8; warnings.append(f"⚠️ Support close ({s_space:.1f}x ATR) — limited downside before floor")

        # ── 7. FUNDING RATE ANALYSIS ──��─────────────────��────
        fp = funding * 100
        if bias == 'LONG':
            if funding < -0.015:
                score += 12; reasons.append(f"✅ Extreme negative funding ({fp:.4f}%) — short squeeze risk favors LONG")
            elif funding < -0.005:
                score += 8;  reasons.append(f"✅ Negative funding ({fp:.4f}%) — shorts paying, favorable for LONG")
            elif funding > 0.02:
                score -= 10; warnings.append(f"⚠️ Positive funding ({fp:.4f}%) — longs overpaying, crowded trade")
        else:
            if funding > 0.015:
                score += 12; reasons.append(f"✅ Extreme positive funding ({fp:.4f}%) — long squeeze risk favors SHORT")
            elif funding > 0.005:
                score += 8;  reasons.append(f"✅ Positive funding ({fp:.4f}%) — longs paying, favorable for SHORT")
            elif funding < -0.02:
                score -= 10; warnings.append(f"⚠️ Negative funding ({fp:.4f}%) — shorts overpaying, crowded")

        # ── 8. MANIPULATION DETECTION ───────────────────────
        candle_range = L['high'] - L['low']
        if candle_range > 0:
            upper_wick     = L['high'] - max(L['open'], L['close'])
            lower_wick     = min(L['open'], L['close']) - L['low']
            upper_wick_pct = upper_wick / candle_range
            lower_wick_pct = lower_wick / candle_range
            if upper_wick_pct > 0.6:
                manip_risk += 20
                manip_notes.append("🚩 Long upper wick (>60% of candle) — possible stop hunt / rejection")
            if lower_wick_pct > 0.6:
                manip_risk += 20
                manip_notes.append("🚩 Long lower wick (>60% of candle) — possible stop hunt / liquidity grab")

        atr_pct = (atr / price) * 100
        if atr_pct > 8.0:
            manip_risk += 15
            manip_notes.append(f"🚩 Extreme ATR ({atr_pct:.1f}%) — very high volatility, manipulation-prone")
        elif atr_pct > 5.0:
            manip_risk += 8
            manip_notes.append(f"⚠️ High ATR ({atr_pct:.1f}%) — elevated volatility, exercise caution")

        recent_vols = df4h['volume'].iloc[-6:-1].tolist()
        if len(recent_vols) >= 4:
            avg_recent = sum(recent_vols) / len(recent_vols)
            vol_std    = (sum((v - avg_recent)**2 for v in recent_vols) / len(recent_vols)) ** 0.5
            cv         = vol_std / avg_recent if avg_recent > 0 else 0
            if cv > 1.5:
                manip_risk += 15
                manip_notes.append("🚩 Erratic volume (CV>1.5) — inconsistent participation, possible spoofing")

        # ── 9. FUNDAMENTAL ANALYSIS (CoinGecko free API) ────
        fa_notes = []
        fa_score = 0
        coin_slug = r['symbol'].replace('USDT','').lower()
        try:
            cg_url  = f"https://api.coingecko.com/api/v3/coins/{coin_slug}"
            cg_resp = http_get(cg_url, timeout=8, headers=HEADERS)
            if cg_resp.status_code == 200:
                cg  = cg_resp.json()
                mkt = cg.get('market_data', {})

                rank = cg.get('market_cap_rank', 9999)
                if rank and rank <= 20:
                    fa_score += 15; fa_notes.append(f"✅ Top-{rank} by market cap — high liquidity, low manip risk")
                elif rank and rank <= 100:
                    fa_score += 8;  fa_notes.append(f"✅ Top-{rank} by market cap — established asset")
                elif rank and rank > 500:
                    fa_score -= 5;  fa_notes.append(f"⚠️ Rank #{rank} — small cap, higher manipulation risk")
                    manip_risk += 10
                    manip_notes.append(f"🚩 Small-cap asset (rank #{rank}) ��� easier to manipulate")

                dev = cg.get('developer_data', {})
                commits = dev.get('commit_count_4_weeks', 0) or 0
                if commits > 50:
                    fa_score += 8; fa_notes.append(f"✅ Active dev ({commits} commits/month) — healthy project")
                elif commits > 10:
                    fa_score += 4; fa_notes.append(f"✅ Moderate dev activity ({commits} commits/month)")
                elif commits == 0:
                    fa_notes.append("⚠️ No recent dev activity — verify project status")

                community = cg.get('community_data', {})
                twitter_f = community.get('twitter_followers', 0) or 0
                if twitter_f > 500_000:
                    fa_score += 6; fa_notes.append(f"✅ Large community ({twitter_f:,} Twitter followers)")
                elif twitter_f > 50_000:
                    fa_score += 3; fa_notes.append(f"✅ Active community ({twitter_f:,} followers)")

                pc_7d  = mkt.get('price_change_percentage_7d',  0) or 0
                if bias == 'LONG':
                    if pc_7d > 5:
                        fa_score += 5; fa_notes.append(f"✅ 7-day momentum: +{pc_7d:.1f}% — bullish macro")
                    elif pc_7d < -15:
                        fa_notes.append(f"⚠️ 7-day: {pc_7d:.1f}% — heavy selloff, capitulation possible")
                else:
                    if pc_7d < -5:
                        fa_score += 5; fa_notes.append(f"✅ 7-day momentum: {pc_7d:.1f}% — bearish macro confirmed")
                    elif pc_7d > 15:
                        fa_notes.append(f"⚠️ 7-day: +{pc_7d:.1f}% — recent pump, short squeeze risk")

                vol_24h      = mkt.get('total_volume', {}).get('usd', 0) or 0
                mkt_cap      = mkt.get('market_cap',   {}).get('usd', 1) or 1
                vol_ratio_fa = vol_24h / mkt_cap
                if vol_ratio_fa > 0.2:
                    fa_notes.append(f"⚠️ Very high vol/mktcap ratio ({vol_ratio_fa:.2f}) — possible news or manipulation")
                    manip_risk += 10
                elif vol_ratio_fa > 0.05:
                    fa_score += 4; fa_notes.append(f"✅ Healthy volume/mktcap ratio ({vol_ratio_fa:.3f})")
        except Exception:
            fa_notes.append(f"ℹ️ FA data unavailable for {r['symbol']} — on-chain analysis skipped")

        score += fa_score
        score  = max(0, min(100, score))

        # ── MANIPULATION RISK LABEL ──────────────────────────
        if manip_risk >= 50:
            manip_label = "🔴 HIGH — Extreme caution recommended"
        elif manip_risk >= 25:
            manip_label = "🟡 MEDIUM — Monitor closely"
        else:
            manip_label = "🟢 LOW — Clean price action"

        # ── FINAL VERDICT ────────────────────────────────────
        if score >= 80 and manip_risk < 25:
            verdict       = "🐌💎 ELITE SNAIL — Maximum conviction. 2x target highly probable."
            verdict_emoji = "🟢"
        elif score >= 65 and manip_risk < 40:
            verdict       = "🐌✅ STRONG SNAIL — High quality setup. 2x target likely within window."
            verdict_emoji = "🟡"
        elif score >= 50:
            verdict       = "🐌⚠️ MODERATE SNAIL — Setup has merit but proceed cautiously."
            verdict_emoji = "🟠"
        else:
            verdict       = "🐌❌ WEAK SNAIL — Does not meet standards. Skip this one."
            verdict_emoji = "🔴"

        return {
            'snail_score':   score,
            'manip_risk':    manip_risk,
            'manip_label':   manip_label,
            'manip_notes':   manip_notes,
            'ta_reasons':    reasons,
            'warnings':      warnings,
            'fa_notes':      fa_notes,
            'verdict':       verdict,
            'verdict_emoji': verdict_emoji,
            'qualifies':     score >= 80 and manip_risk < 25
        }
    except Exception as e:
        logger.warning("Snail score error: %s", e)
        return None


def snail_full_analyze(symbol):
    """Run full TA + snail score for a symbol. Returns (signal, snail_analysis) or (None, None)."""
    result       = None
    snail_result = None

    for exchange, analyze_fn in [
        ('BYBIT',   analyze_bybit),
        ('BINANCE', analyze_binance),
        ('MEXC',    analyze_mexc),
    ]:
        try:
            r = analyze_fn(symbol)
            if r and r.get('confidence', 0) == 10:
                if exchange == 'BYBIT' and sakz_exchanges.BYBIT_AVAILABLE:
                    df4h = bybit_fetch_ohlcv(symbol, '240', 100)
                    df1d = bybit_fetch_ohlcv(symbol, 'D',   60)
                    fund = bybit_fetch_funding(symbol)
                elif exchange == 'BINANCE' and sakz_exchanges.BINANCE_AVAILABLE:
                    df4h = binance_fetch_ohlcv(symbol, '4h', 100)
                    df1d = binance_fetch_ohlcv(symbol, '1d', 60)
                    fund = binance_fetch_funding(symbol)
                else:
                    df4h = mexc_fetch_ohlcv(symbol, '4h', 100)
                    df1d = mexc_fetch_ohlcv(symbol, '1d', 60)
                    fund = 0

                if df4h is None or df1d is None:
                    continue
                df4h = add_indicators(df4h, timeframe="4h")
                df1d = add_indicators(df1d, timeframe="1d")
                if df4h is None or df1d is None or len(df4h) < 6 or len(df1d) < 3:
                    continue

                sa = snail_score_signal(r, df4h, df1d, fund)
                if sa and sa['qualifies']:
                    result       = r
                    snail_result = sa
                    break
        except Exception as e:
            logger.warning("Snail analyze %s on %s: %s", symbol, exchange, e)

    return result, snail_result


def format_snail_signal(r, sa, day_num, days_left):
    """Format a full SNAIL TRADE alert message."""
    bias_e = "���" if r['bias'] == 'LONG' else "🔴"
    lev    = r.get('leverage')
    link   = get_exchange_link(r['exchange'], r['symbol'])
    bar_w  = int(sa['snail_score'] / 10)
    bar    = "█" * bar_w + "░" * (10 - bar_w)

    # ── Leverage-aware 2x target ──────────────────────────────
    t2_lev, t2_desc = _snail_2x_target(r)
    lev_val = lev['suggested'] if lev else 5

    lines = [
        f"🐌  S N A I L   T R A D E  🐌",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📅 Day {day_num}/7 of your SNAIL week  |  {days_left} days left",
        f"",
        f"🏆 {r['exchange']} | {r['symbol']}",
        f"{bias_e} BIAS: {r['bias']}  |  Confidence: 10/10",
        f"",
        f"🎯 SNAIL SCORE",
        f"   {bar} {sa['snail_score']}/100",
        f"",
        f"💰 CURRENT PRICE:    ${r['price']:.6f}",
        f"📥 ENTRY ZONE:       ${r['entry_low']:.6f} → ${r['entry_high']:.6f}",
        f"🛑 STOP LOSS:        ${r['stop_loss']:.6f}",
        f"",
        f"🎯 TARGET 1:         ${r['t1']:.6f}",
        f"🎯 TARGET 2 (2x 🐌): ${t2_lev:.6f}  ← SNAIL TARGET",
        f"   ({t2_desc})",
        f"🎯 TARGET 3 (max):   ${r['t3']:.6f}",
        f"⏱  HOLD DURATION:    {r['hold']}",
        f"",
    ]

    if lev and r['confidence'] >= 8:
        lines += [
            f"⚡ LEVERAGE: {lev['suggested']}x suggested  (max safe: {lev['max_safe']}x)",
            f"   SL dist: {lev['sl_dist']:.2f}%  |  Liq dist: ~{lev['liq_dist']:.2f}%",
            f"",
        ]

    lines += [
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━���━━",
        f"📈 TECHNICAL ANALYSIS",
    ]
    for note in sa['ta_reasons'][:6]:
        lines.append(f"  {note}")

    if sa['warnings']:
        lines += [f"", f"⚠️  CAUTIONS:"]
        for w in sa['warnings']:
            lines.append(f"  {w}")

    lines += [
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🔬 FUNDAMENTAL ANALYSIS",
    ]
    for note in sa['fa_notes'][:5]:
        lines.append(f"  {note}")
    if not sa['fa_notes']:
        lines.append(f"  ℹ️ No FA data available for this token.")

    lines += [
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🕵️  MANIPULATION RISK: {sa['manip_label']}",
    ]
    for note in sa['manip_notes']:
        lines.append(f"  {note}")
    if not sa['manip_notes']:
        lines.append(f"  ✅ No manipulation signals detected.")

    lines += [
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"⚖️  FINAL VERDICT",
        f"  {sa['verdict']}",
        f"",
        f"🔗 Trade on {r['exchange']}: {link}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📌 /snailvault — view your week log",
    ]
    return "\n".join(lines)


# ── Snail daily background job ────────────────
async def snail_daily_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Runs daily at 08:00 UTC. For each active snail session:
    1. Check outcomes of open snail signals
    2. Scan top symbols for 10/10 + snail score >= 80
    3. Send SNAIL TRADE alert if found, otherwise send daily update
    4. Close expired sessions with a final report
    """
    active_chats = db_snail_load_all_active()
    if not active_chats:
        return

    now              = datetime.now()
    elite_candidates = [r for r in state.last_scan_results if r.get('confidence', 0) == 10]
    logger.info("Snail daily job — %d sessions, %d elite candidates", len(active_chats), len(elite_candidates))

    for chat_id in active_chats:
        try:
            session = db_snail_get_session(chat_id)
            if not session:
                continue

            if now >= session['expires_at']:
                db_snail_end_session(chat_id)
                snail_active.pop(chat_id, None)
                await _send_snail_final_report(context.bot, chat_id)
                continue

            days_elapsed = (now - session['activated_at']).days + 1
            days_left    = max(0, (session['expires_at'] - now).days)

            # ── Check outcomes of pending snail signals ──────
            for ps in db_snail_get_pending_signals(chat_id):
                try:
                    sent_dt   = datetime.fromisoformat(ps['sent_at'])
                    hours_old = (now - sent_dt).total_seconds() / 3600
                    if hours_old < 4:
                        continue

                    exc = ps['exchange']
                    sym = ps['symbol']
                    cp = _get_live_price(sym, exc)
                    if cp == 0:
                        continue

                    bias = ps['bias']
                    sl   = ps['stop_loss']

                    # Leverage-aware 2x target: reconstruct from stored entry price
                    # We need to recompute since the signal dict isn't stored directly
                    # Use a simple proxy: leverage from snail_signals table isn't stored,
                    # so we re-derive: 2x on 5x leverage = 20% price move (conservative default)
                    # If the full signal is available via last_scan_results, use it
                    t2_lev = ps['t2']  # fallback to stored T2
                    for cached_r in state.last_scan_results:
                        if cached_r.get('symbol') == sym and cached_r.get('exchange') == exc:
                            t2_lev, _ = _snail_2x_target(cached_r)
                            break

                    hit_t2 = (bias == 'LONG' and cp >= t2_lev) or (bias == 'SHORT' and cp <= t2_lev)
                    hit_sl = (bias == 'LONG' and cp <= sl) or (bias == 'SHORT' and cp >= sl)

                    if hit_t2:
                        db_snail_update_outcome(ps['id'], 'win_2x')
                        raw_pnl = abs(t2_lev - ps['entry_price']) / ps['entry_price'] * 100
                        await context.bot.send_message(chat_id=chat_id, text=(
                            f"🐌🎯 SNAIL WIN!\n\n"
                            f"✅ {sym} hit the 2x SNAIL TARGET!\n"
                            f"Entry: ${ps['entry_price']:.6f} → Target: ${t2_lev:.6f}\n"
                            f"Raw gain: +{raw_pnl:.2f}%\n\n"
                            f"Day {days_elapsed}/7 — {days_left} days left.\n"
                            f"Watching for tomorrow's signal... 🐌"
                        ))
                    elif hit_sl:
                        db_snail_update_outcome(ps['id'], 'stopped')
                        raw_loss = abs(cp - ps['entry_price']) / ps['entry_price'] * 100
                        await context.bot.send_message(chat_id=chat_id, text=(
                            f"🐌🛑 SNAIL STOPPED OUT\n\n"
                            f"❌ {sym} hit stop loss.\n"
                            f"Loss: -{raw_loss:.2f}%\n\n"
                            f"SNAIL mode continues. Slow and steady. 🐌\n"
                            f"Day {days_elapsed}/7 — {days_left} days left."
                        ))
                except Exception as e:
                    logger.warning("Snail outcome check: %s", e)

            # ── Find today's SNAIL signal ────────────────────
            found_signal = False
            for r in elite_candidates:
                try:
                    sym          = r['symbol']
                    signal, sa   = await asyncio.get_event_loop().run_in_executor(
                        SCAN_EXECUTOR, lambda s=sym: snail_full_analyze(s)
                    )
                    if signal and sa and sa['qualifies']:
                        session   = db_snail_get_session(chat_id)  # re-fetch for accuracy
                        days_left = max(0, (session['expires_at'] - now).days)
                        msg       = format_snail_signal(signal, sa, days_elapsed, days_left)
                        await context.bot.send_message(chat_id=chat_id, text=msg)
                        db_snail_save_signal(chat_id, signal)
                        db_snail_increment_signals(chat_id)
                        found_signal = True
                        break
                except Exception as e:
                    logger.warning("Snail signal send error: %s", e)

            if not found_signal:
                await context.bot.send_message(chat_id=chat_id, text=(
                    f"🐌 SNAIL DAILY SCAN — Day {days_elapsed}/7\n"
                    f"━━━━━━━━━━━━━━━━━━���━━━━━━━━━━━\n\n"
                    f"No signal reached SNAIL standards today.\n"
                    f"Requirements: 10/10 confidence + Snail Score ≥ 80 + Low manipulation risk\n\n"
                    f"📊 Scanned: {len(state.last_scan_results)} pairs\n"
                    f"⭐ 10/10 signals found: {len(elite_candidates)}\n"
                    f"🐌 SNAIL qualified: 0\n\n"
                    f"{days_left} days left in your session.\n"
                    f"Patience is the snail's greatest weapon. 🐌"
                ))

        except Exception as e:
            logger.warning("Snail daily job error for %s: %s", chat_id, e)


async def _send_snail_final_report(bot, chat_id):
    """Send a 7-day SNAIL session summary."""
    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT * FROM snail_signals WHERE chat_id=? ORDER BY sent_at ASC", (chat_id,))
    rows = c.fetchall(); conn.close()

    total   = len(rows)
    wins    = sum(1 for r in rows if r['outcome'] == 'win_2x')
    stopped = sum(1 for r in rows if r['outcome'] == 'stopped')
    pending = sum(1 for r in rows if r['outcome'] == 'pending')
    win_rt  = (wins / total * 100) if total > 0 else 0
    bar     = "█" * int(win_rt / 10) + "░" * (10 - int(win_rt / 10))

    lines = [
        f"🐌  SNAIL WEEK COMPLETE!\n",
        f"━━━━━��━━━━━━━━━━━━���━━━━━━━━━━━",
        f"Your 7-day SNAIL session has ended.\n",
        f"📊 SESSION RESULTS",
        f"   Signals fired: {total}",
        f"   ✅ 2x Wins:    {wins}",
        f"   ❌ Stopped:    {stopped}",
        f"   ⏳ Unresolved: {pending}",
        f"",
        f"🎯 WIN RATE",
        f"   {bar} {win_rt:.0f}%",
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━���━━━━━",
    ]
    if wins >= 5:
        lines.append("🏆 LEGENDARY SNAIL WEEK. Maximum execution.")
    elif wins >= 3:
        lines.append("🟢 Solid week. The patience paid off.")
    elif wins >= 1:
        lines.append("🟡 Some wins in the mix. Keep building.")
    else:
        lines.append("🔴 Tough week. Markets were difficult. SNAIL is patient.")
    lines += [f"", f"Use /snail to start a new 7-day session anytime."]
    await bot.send_message(chat_id=chat_id, text="\n".join(lines))


# ── Snail Telegram command handlers ──────────

async def secret_unlock_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Hidden handler for the secret passphrase message.
    Shows snail activation menu. Completely silent unless the exact text matches.
    """
    chat_id = update.effective_chat.id
    state.snail_unlocked.add(chat_id)
    db_snail_unlock(chat_id)

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🐌 Activate SNAIL MODE (7 days)", callback_data="snail_activate")],
        [InlineKeyboardButton("📋 Snail Status",                  callback_data="snail_status")],
        [InlineKeyboardButton("📜 Week Report",                   callback_data="snail_report")],
        [InlineKeyboardButton("🛑 Stop Snail Mode",               callback_data="snail_stop")],
    ])

    await update.message.reply_text(
        f"🐌  S N A I L   M O D E\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🔓 Hidden feature unlocked!\n\n"
        f"SNAIL MODE is a 7-day precision system.\n"
        f"One signal per day. Maximum quality only.\n\n"
        f"  🔹 Confidence: 10/10 required\n"
        f"  🔹 Snail Score: ≥ 80/100\n"
        f"  🔹 Full TA + FA + Manipulation check\n"
        f"  🔹 Target: 2x from entry (T2 level)\n"
        f"  🔹 Daily reminder + outcome tracking\n"
        f"  🔹 7-day session with final report\n\n"
        f"Slow. Patient. Precise. 🐌\n",
        reply_markup=keyboard
    )


async def snail_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /snail — completely silent (no reply) unless chat_id is unlocked.
    """
    _track(update)
    chat_id = update.effective_chat.id
    if chat_id not in state.snail_unlocked and not db_snail_is_unlocked(chat_id):
        return  # 🤫 total silence — not even an error message

    session   = db_snail_get_session(chat_id)
    now       = datetime.now()
    if session:
        days_left  = max(0, (session['expires_at'] - now).days)
        day_num    = (now - session['activated_at']).days + 1
        sigs       = session['signals_sent']
        status_str = f"🟢 ACTIVE — Day {day_num}/7  |  {sigs} signals sent  |  {days_left}d left"
    else:
        status_str = "🔴 INACTIVE — Not started"

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🐌 Activate / Restart (7 days)", callback_data="snail_activate")],
        [InlineKeyboardButton("📋 Status",                       callback_data="snail_status")],
        [InlineKeyboardButton("📜 Week Report",                  callback_data="snail_report")],
        [InlineKeyboardButton("🛑 Stop Snail Mode",              callback_data="snail_stop")],
    ])

    await update.message.reply_text(
        f"🐌  S N A I L   M O D E\n"
        f"━━━━━━━━━━�����━━━━━━━━━━━━━━━━━━━\n\n"
        f"Status: {status_str}\n\n"
        f"Goal: 2x per day | Criteria: 10/10 + Snail Score ≥ 80\n"
        f"Use /snailvault to view your signal log.\n",
        reply_markup=keyboard
    )


async def snailvault_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/snailvault — completely silent if not unlocked."""
    _track(update)
    chat_id = update.effective_chat.id
    if chat_id not in state.snail_unlocked and not db_snail_is_unlocked(chat_id):
        return

    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT * FROM snail_signals WHERE chat_id=? ORDER BY sent_at DESC LIMIT 14", (chat_id,))
    rows = c.fetchall(); conn.close()

    if not rows:
        await update.message.reply_text(
            "🐌 SNAIL VAULT\n\n"
            "No snail signals recorded yet.\n"
            "Activate snail mode and wait for qualifying signals."
        )
        return

    lines = ["🐌 SNAIL VAULT — Signal Log\n━━━━━━━━━━━━━━━━━━━━━━━━━��━━━━\n"]
    outcome_map = {'win_2x': '�� WIN 2x', 'stopped': '🛑 Stopped', 'pending': '⏳ Open'}
    for r in rows:
        e       = "🟢" if r['bias'] == 'LONG' else "🔴"
        outcome = outcome_map.get(r['outcome'], r['outcome'])
        dt_str  = datetime.fromisoformat(r['sent_at']).strftime('%b %d %H:%M')
        lines.append(
            f"{e} {r['exchange']} | {r['symbol']} {r['bias']}\n"
            f"   Entry: ${r['entry_price']:.6f}  |  T2: ${r['t2']:.6f}\n"
            f"   {outcome}  ·  {dt_str}\n"
        )

    conn = db_connect()
    c    = conn.cursor()
    c.execute("SELECT outcome FROM snail_signals WHERE chat_id=?", (chat_id,))
    all_rows = c.fetchall(); conn.close()
    total    = len(all_rows)
    wins     = sum(1 for r in all_rows if r['outcome'] == 'win_2x')
    win_rt   = (wins / total * 100) if total > 0 else 0
    lines.append(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"Overall: {wins}/{total} wins ({win_rt:.0f}%)")
    await update.message.reply_text("\n".join(lines))


async def snail_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all snail inline button callbacks."""
    query   = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    data    = query.data

    if chat_id not in state.snail_unlocked and not db_snail_is_unlocked(chat_id):
        return

    if data == "snail_activate":
        db_snail_start_session(chat_id)
        snail_active[chat_id] = True
        state.snail_unlocked.add(chat_id)
        exp = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d %H:%M')
        await query.edit_message_text(
            f"🐌 SNAIL MODE ACTIVATED!\n\n"
            f"✅ 7-day session started.\n"
            f"Expires: {exp}\n\n"
            f"I'll send ONE signal per day when:\n"
            f"  🔹 Confidence = 10/10\n"
            f"  🔹 Snail Score ≥ 80/100\n"
            f"  🔹 Manipulation risk: LOW\n"
            f"  🔹 TA + FA both verified\n\n"
            f"Target: 2x from entry (T2 level). 🐌\n"
            f"Use /snailvault to monitor your trades."
        )

    elif data == "snail_status":
        session = db_snail_get_session(chat_id)
        if not session:
            await query.edit_message_text(
                "🐌 No active SNAIL session.\n\nUse /snail → Activate to start."
            )
            return
        now       = datetime.now()
        days_left = max(0, (session['expires_at'] - now).days)
        day_num   = (now - session['activated_at']).days + 1
        pending   = db_snail_get_pending_signals(chat_id)
        conn      = db_connect()
        c         = conn.cursor()
        c.execute("SELECT outcome FROM snail_signals WHERE chat_id=?", (chat_id,))
        sig_rows  = c.fetchall(); conn.close()
        wins      = sum(1 for r in sig_rows if r['outcome'] == 'win_2x')
        stopped   = sum(1 for r in sig_rows if r['outcome'] == 'stopped')
        await query.edit_message_text(
            f"🐌 SNAIL STATUS\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📅 Day {day_num}/7  |  {days_left} days left\n"
            f"Expires: {session['expires_at'].strftime('%Y-%m-%d %H:%M')}\n\n"
            f"📊 This session:\n"
            f"   Signals sent: {session['signals_sent']}\n"
            f"   ✅ 2x Wins:   {wins}\n"
            f"   ❌ Stopped:   {stopped}\n"
            f"   ⏳ Open:      {len(pending)}\n\n"
            f"/snailvault for full signal log."
        )

    elif data == "snail_report":
        await _send_snail_final_report(context.bot, chat_id)

    elif data == "snail_stop":
        db_snail_end_session(chat_id)
        snail_active.pop(chat_id, None)
        await query.edit_message_text(
            "🐌 SNAIL MODE STOPPED.\n\n"
            "Session ended. Use /snail to start again anytime."
        )





# ═══════���═══════════════════════════════════════════════════════
# ── FEATURE BLOCK — v5 additions ────────────────────────────────
# 1. Timeframe argument for /scan and /cscan  (e.g. /scan t15m)
# 2. Trend-dying notification job
# 3. Snail 2x target tied to actual leverage
# 4. /chart — generate TA chart image
# 5. /lb alias for /leaderboard
# 6. Admin user-count tracking (/admin)
# ══════════════════════════════════════════════════════════���════

# ─── TIMEFRAME HELPERS ──────────────────────────────────���─────
TF_MAP_MEXC   = {'1m':'Min1','3m':'Min3','5m':'Min5','15m':'Min15',
                 '30m':'Min30','1h':'Min60','2h':'Hour2','4h':'Hour4',
                 '6h':'Hour6','12h':'Hour12','1d':'Day1','1w':'Week1'}
TF_MAP_BYBIT  = {'1m':'1','3m':'3','5m':'5','15m':'15','30m':'30',
                 '1h':'60','2h':'120','4h':'240','6h':'360','12h':'720',
                 '1d':'D','1w':'W'}
TF_MAP_BINANCE= {'1m':'1m','3m':'3m','5m':'5m','15m':'15m','30m':'30m',
                 '1h':'1h','2h':'2h','4h':'4h','6h':'6h','12h':'12h',
                 '1d':'1d','1w':'1w'}

def _parse_tf_arg(raw: str) -> str | None:
    """Parse a user-supplied timeframe arg like t15m, t4h, 15m, 4H → canonical key like '15m'."""
    s = raw.lower().lstrip('t')
    # normalise: 15min→15m, 4hour→4h, 1day→1d
    s = re.sub(r'min(ute)?s?$', 'm', s)
    s = re.sub(r'hour?s?$',     'h', s)
    s = re.sub(r'day?s?$',      'd', s)
    s = re.sub(r'week?s?$',     'w', s)
    return s if s in TF_MAP_MEXC else None

def parse_scan_args(args):
    """Parse /scan [pair] [tf] [all] arguments.
    Returns (pair, timeframe, show_all).
    Distinguishes: symbol token vs timeframe token vs show-all flag.
    """
    pair = timeframe = None
    show_all = False
    for a in (args or []):
        tok = str(a).strip().lower()
        if not tok:
            continue
        if tok in _SCAN_SHOW_ALL_FLAGS:
            show_all = True
            continue
        tf = _parse_tf_arg(tok)
        if tf:
            timeframe = tf
            continue
        if pair is None:
            pair = tok
    return pair, timeframe, show_all


def _tf_display(tf: str) -> str:
    labels = {'1m':'1 Min','3m':'3 Min','5m':'5 Min','15m':'15 Min','30m':'30 Min',
              '1h':'1 Hour','2h':'2 Hour','4h':'4 Hour','6h':'6 Hour','12h':'12 Hour',
              '1d':'Daily','1w':'Weekly'}
    return labels.get(tf, tf.upper())

def _cscan_pair_tf(symbol: str, tf: str):
    """
    Scan one symbol on the requested timeframe via MEXC.
    Returns (results, failures).

    Auto TF fallback: if the requested TF has insufficient data (new/thin pair),
    automatically walks down to shorter TFs:  4h → 1h → 15m → 5m
    Uses the first TF that returns >= 3 candles and produces a valid signal.
    The signal dict is tagged with the actual TF used and a switch note when
    the TF was auto-changed from what the user requested.
    """
    results  = []
    failures = []
    limit    = 220

    MIN_PRI = 3   # absolute floor — 3 candles is workable for a new listing signal
    MIN_CON = 3   # confirmation floor — primary df reused if 1d unavailable

    # Fallback order — MEXC does not offer 10m, so 5m is the floor
    _TF_FALLBACK = ['4h', '1h', '15m', '5m']

    # Build TFs to attempt: start from requested, walk down to shorter only
    if tf in _TF_FALLBACK:
        tfs_to_try = _TF_FALLBACK[_TF_FALLBACK.index(tf):]
    else:
        tfs_to_try = [tf]   # non-standard TF (e.g. 30m, 2h) — try as-is only

    tf_used    = None
    df_p_found = None
    df_s_found = None

    # ── MEXC: walk down TFs until one returns usable data ────────────────────
    for attempt_tf in tfs_to_try:
        try:
            mexc_iv = TF_MAP_MEXC.get(attempt_tf, 'Hour4')
            df_p    = mexc_fetch_ohlcv(symbol, attempt_tf, limit)

            if df_p is None or len(df_p) < MIN_PRI:
                got    = len(df_p) if df_p is not None else 0
                reason = REASON_NO_CONTRACT if df_p is None else REASON_SHORT_HISTORY
                detail = (f"MEXC returned no data for {symbol} on {mexc_iv}"
                          if df_p is None else
                          f"MEXC {symbol} on {mexc_iv}: only {got} candles (need {MIN_PRI})")
                failures.append(ScanFailure(reason, exchange='MEXC', tf=attempt_tf, detail=detail))
                logger.debug("cscan_tf %s %s: %s", symbol, attempt_tf, detail)
                continue   # try next shorter TF

            # Got primary data — attempt 1d confirmation
            df_s = mexc_fetch_ohlcv(symbol, '1d', 60)

            if df_s is None or len(df_s) < MIN_CON:
                # Spot 1d fallback
                logger.debug("MEXC %s: futures 1d thin — trying spot 1d", symbol)
                try:
                    r_spot    = http_get(
                        "https://api.mexc.com/api/v3/klines",
                        params={'symbol': symbol, 'interval': '1d', 'limit': 60},
                        headers=HEADERS, timeout=15
                    )
                    spot_data = r_spot.json()
                    if isinstance(spot_data, list) and len(spot_data) >= MIN_CON:
                        df_s = pd.DataFrame(spot_data, columns=[
                            'timestamp','open','high','low','close','volume',
                            'close_time','quote_vol','trades','taker_base','taker_quote','ignore'
                        ])
                        for col in ['open','high','low','close','volume']:
                            df_s[col] = pd.to_numeric(df_s[col], errors='coerce')
                        df_s['timestamp'] = pd.to_datetime(df_s['timestamp'], unit='ms')
                        df_s = df_s[['timestamp','open','high','low','close','volume']].dropna()
                        logger.debug("MEXC %s: spot 1d OK (%d rows)", symbol, len(df_s))
                    else:
                        df_s = None
                except Exception as _fe:
                    logger.debug("MEXC %s spot fallback: %s", symbol, _fe)
                    df_s = None

            # Still no confirmation — self-confirm using primary df (new listing mode)
            if df_s is None or len(df_s) < MIN_CON:
                logger.debug("MEXC %s: self-confirming with primary df (new listing)", symbol)
                df_s = df_p.copy()

            tf_used    = attempt_tf
            df_p_found = df_p
            df_s_found = df_s
            break   # stop walking — we have data

        except Exception as e:
            logger.warning("cscan_tf MEXC %s %s: %s", symbol, attempt_tf, e)
            failures.append(ScanFailure(REASON_NO_CONTRACT, exchange='MEXC', tf=attempt_tf, detail=str(e)))

    # ── Score the pair on whichever TF had data ────────────────────────────���──
    if tf_used is not None and df_p_found is not None:
        try:
            df_p2 = add_indicators(df_p_found, timeframe=tf_used)
            df_s2 = add_indicators(df_s_found, timeframe='1d')

            if df_p2 is None or len(df_p2) < 3:
                failures.append(ScanFailure(REASON_SHORT_HISTORY, exchange='MEXC', tf=tf_used,
                                            detail=f"MEXC {symbol}: indicators reduced below 3 rows on {tf_used}"))
            else:
                result = score_pair(df_p2, df_s2, 0, symbol, user_requested=True)
                if isinstance(result, ScanFailure):
                    result.exchange = result.exchange or 'MEXC'
                    result.tf       = result.tf or tf_used
                    failures.append(result)
                elif isinstance(result, dict) and result.get('regime_blocked'):
                    # HARD REGIME GATE — surface as failure so /scan redirects to /analyse
                    failures.append(ScanFailure(REASON_REGIME_BLOCK, exchange='MEXC', tf=tf_used,
                                                detail=result.get('regime_block_detail') or 'below BTC regime floor'))
                elif result:
                    result['exchange']  = 'MEXC'
                    result['timeframe'] = tf_used
                    # Tag card when TF was auto-switched from what user requested
                    if tf_used != tf:
                        result['tf_note']     = (
                            f"⚠️ Auto-switched to {_tf_display(tf_used)} "
                            f"(not enough {_tf_display(tf)} history yet).\n"
                            f"Signal based on {_tf_display(tf_used)} data — use smaller size."
                        )
                        result['new_listing'] = True
                    results.append(result)
        except Exception as e:
            logger.warning("cscan_tf score_pair %s %s: %s", symbol, tf_used, e)
            failures.append(ScanFailure(REASON_NO_CONTRACT, exchange='MEXC', tf=tf_used, detail=str(e)))

    # ── Cross-exchange fallback — FIX #SCAN-MEXC-ONLY ───────────────────────
    # The single-pair /scan historically queried MEXC only, so pairs that trade
    # on Bybit/Binance perps but NOT on MEXC (e.g. FETUSDT) returned a
    # misleading "not found on MEXC".  If MEXC produced no signal, fall back to
    # analyze_symbol_new_listing, which probes BYBIT -> MEXC -> BINANCE (each
    # gated by its *_AVAILABLE flag, so geo-blocked venues are simply skipped)
    # and walks down timeframes.  This is a no-op where Bybit/Binance are
    # unreachable, and recovers the pair where they are.
    if not results:
        try:
            alt = analyze_symbol_new_listing(symbol)
            if alt and not isinstance(alt, ScanFailure):
                alt.setdefault('timeframe', alt.get('signal_tf', tf))
                results.append(alt)
            elif isinstance(alt, ScanFailure):
                failures.append(alt)
        except Exception as e:
            logger.warning("cscan_tf cross-exchange fallback %s: %s", symbol, e)

    return results, failures


# ��── PATCHED scan_command (timeframe support) ─────────────────
async def scan_tf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Unified /scan dispatcher.  Handles all variants:

      /scan                    — full market scan (top 50, 4H default, cached)
      /scan ETH                — scan a specific pair (4H default)
      /scan ETH 2h             — scan a specific pair on a custom timeframe
      /scan 2h                 — scan top 30 pairs on a custom timeframe
      /scan new [window]       — new-listings scan  e.g. /scan new 24h
      /scan mid                — mid-market scan (ranks 51–200)
    """
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args or []

    # ── /scan new [window] ─────────────────────────────────────
    if args and args[0].lower() == 'new':
        context.args = args[1:]  # shift window arg into position
        await scan_new_command(update, context)
        return

    # ── /scan mid ──────────────────────────────────────────────
    if args and args[0].lower() == 'mid':
        context.args = args[1:]  # pass any extra rank args through
        await scanmid_command(update, context)
        return

    # ── Parse any TF and pair-symbol from the remaining args ───
    tf      = None
    sym_arg = None
    for a in args:
        parsed = _parse_tf_arg(a)
        if parsed and tf is None:
            tf = parsed
        elif sym_arg is None and not parsed:
            sym_arg = a

    # ��─ /scan [pair] or /scan [pair] [tf] ─────────────────────
    if sym_arg is not None:
        track_user_interaction(chat_id)
        raw    = sym_arg.upper().replace('/', '').strip()
        symbol = raw if raw.endswith('USDT') else raw + 'USDT'
        actual_tf    = tf or '4h'
        tf_label     = _tf_display(actual_tf)
        tf_note_msg  = "" if tf else " (auto-selects shortest available TF for new pairs)"

        await update.message.reply_text(
            f"🔍 Scanning {symbol} on {tf_label}...{tf_note_msg}\n"
            f"📡 Checking MEXC perpetuals\n"
            f"⏳ Please wait ~30 seconds..."
        )

        loop              = asyncio.get_event_loop()
        results, failures = await loop.run_in_executor(
            SCAN_EXECUTOR,
            lambda: _cscan_pair_tf(symbol, actual_tf)
        )

        if not results:
            # ── Real reason diagnosis ────────────────��──────────────────��─
            reasons_seen = [f.reason for f in failures]
            details      = [f.detail for f in failures if f.detail]

            # Classify the real cause
            data_nones   = [d for d in details if d and 'returned no data' in d]
            stale_hits   = [d for d in details if d and 'stale data' in d]

            if stale_hits:
                msg = (
                    f"⏱ *{symbol}* — Stale exchange data on {tf_label}\n\n"
                    f"The last candle is significantly older than expected — "
                    f"the exchange feed may be delayed or the pair is inactive.\n\n"
                    f"Detail: `{stale_hits[0][:120]}`\n\n"
                    f"💡 Try again in a few minutes or use `/chart {sym_arg} {actual_tf}`"
                )
            # ── Signal-quality blocks checked FIRST (before "not found") ────────
            # These mean data WAS fetched — the pair exists — but signal quality
            # filters stopped it. Showing "not found" here misleads the user.
            elif REASON_GAP_BLOCK in reasons_seen:
                gap_d = next((f.detail for f in failures if f.reason == REASON_GAP_BLOCK), 'score gap < 2')
                msg = (
                    f"↔️ *{symbol}* — Signal too close to call\n\n"
                    f"Long and Short scores are nearly equal — not enough edge to recommend a trade.\n"
                    f"This is common in choppy, sideways markets.\n\n"
                    f"Detail: `{gap_d}`\n\n"
                    f"💡 Options:\n"
                    f"• Try `/scan {sym_arg} 1h` — a shorter TF may show clearer momentum\n"
                    f"• Check back after the next candle close\n"
                    f"• Use `/chart {sym_arg} {actual_tf}` to assess manually"
                )
            elif REASON_COUNTER_TREND in reasons_seen:
                ct_d = next((f.detail for f in failures if f.reason == REASON_COUNTER_TREND), 'see logs')
                msg = (
                    f"↔️ *{symbol}* — Counter-trend signal blocked\n\n"
                    f"The 4H bias opposes the daily EMA direction.\n"
                    f"Counter-trend trades require a higher score to qualify here.\n\n"
                    f"Detail: `{ct_d}`\n\n"
                    f"💡 The daily trend is stronger than the 4H setup.\n"
                    f"Consider trading in the daily trend direction instead,\n"
                    f"or try `/scan {sym_arg} 1d` to see the higher-timeframe view."
                )
            elif REASON_FLIP_BLOCK in reasons_seen:
                flip_d = next((f.detail for f in failures if f.reason == REASON_FLIP_BLOCK), 'see logs')
                msg = (
                    f"🔄 *{symbol}* — Direction flip blocked (cooldown active)\n\n"
                    f"A signal in the opposite direction fired very recently.\n"
                    f"The flip-cooldown requires confidence ≥ 9 to reverse within 8 hours.\n\n"
                    f"Detail: `{flip_d}`\n\n"
                    f"💡 Options:\n"
                    f"• Wait ~8h from the last signal, then `/scan {sym_arg}` again\n"
                    f"• The market needs to show a much stronger reversal (conf 9+)\n"
                    f"• Use `/chart {sym_arg} {actual_tf}` to read the chart manually"
                )
            elif REASON_LOW_CONF in reasons_seen:
                lc_d = next((f.detail for f in failures if f.reason == REASON_LOW_CONF), 'conf < 4')
                msg = (
                    f"🟠 *{symbol}* — Doesn't pass the signal requirements\n\n"
                    f"A setup was found but confidence is below the minimum threshold.\n\n"
                    f"Detail: `{lc_d}`\n\n"
                    f"💡 Use `/analyse {sym_arg}` to scan this pair with no restrictions\n"
                    f"and get the full breakdown — that command has no gates (information only)."
                )
            elif REASON_SHORT_HISTORY in reasons_seen:
                hint = f"\n\n   Detail: `{details[0][:120]}`" if details else ""
                msg = (
                    f"🆕 *{symbol}* — Very new listing, limited history on {tf_label}.{hint}\n\n"
                    f"The dynamic new-listing scanner was attempted but couldn't\n"
                    f"produce a signal yet (not enough candles on any timeframe).\n\n"
                    f"• Wait ~1 hour and try `/scan {sym_arg}` again\n"
                    f"• Or try `/scan {sym_arg} 15m` once a few candles form"
                )
            elif data_nones and REASON_NO_CONTRACT in reasons_seen and not any(
                    f.reason == REASON_SHORT_HISTORY for f in failures):
                # All exchanges returned None — likely real missing pair
                hint = f"\n\n   Detail: `{details[0][:120]}`" if details else ""
                msg = (
                    f"❌ *{symbol}* not found on any reachable exchange for {tf_label}.{hint}\n\n"
                    f"• Check the symbol spelling (e.g. ZKC not ZKCU)\n"
                    f"• Pair may not be listed as a perpetual future on MEXC\n"
                    f"• Try `/cscan {sym_arg}` for a deeper search"
                )
            elif not failures or REASON_NO_CONTRACT in reasons_seen:
                hint = f"\n\n   Detail: `{details[0][:120]}`" if details else ""
                msg = (
                    f"❌ *{symbol}* not found on any reachable exchange for {tf_label}.{hint}\n\n"
                    f"• Check the symbol spelling (e.g. ZKC not ZKCU)\n"
                    f"• Pair may not be listed as a perpetual future on MEXC\n"
                    f"• Try `/cscan {sym_arg}` for a deeper search"
                )
            elif REASON_REGIME_BLOCK in reasons_seen:
                rb_d = next((f.detail for f in failures if f.reason == REASON_REGIME_BLOCK), 'below BTC regime floor')
                msg = (
                    f"🚫 *{symbol}* — Blocked by the BTC Regime gate\n\n"
                    f"This signal fights the current market regime and scored below\n"
                    f"the required confidence floor, so it won't be published.\n\n"
                    f"Detail: `{rb_d}`\n\n"
                    f"💡 Use `/analyse {sym_arg}` to scan this pair with no restrictions\n"
                    f"and get the full breakdown — that command has no gates (information only)."
                )
            else:
                # NEUTRAL — no gated directional signal. Point the user to
                # /analyse, which runs ungated and may still surface a
                # directional read with a confidence score (4–7/10).
                msg = (
                    f"😐 *{symbol}* — No gated signal on {tf_label}\n\n"
                    f"The scan's strict scoring found no clear LONG/SHORT edge that "
                    f"passes the signal gates right now.\n\n"
                    f"💡 Use `/analyse {sym_arg}` for the full ungated read — it shows "
                    f"the bullish/bearish indicators and, when there's a directional "
                    f"lean, a confidence score (4–7/10) plus trade levels. A truly "
                    f"balanced market shows no confidence; a 4–7/10 read means there's "
                    f"a moderate setup the gates filtered out.\n\n"
                    f"Other options:\n"
                    f"• `/scan {sym_arg} 1h` — shorter TF may be trending\n"
                    f"• `/chart {sym_arg} {actual_tf}` — read the chart yourself\n"
                    f"• `/scan` — find other pairs with active momentum"
                )
            await update.message.reply_text(msg, parse_mode="Markdown")
            return

        results.sort(key=lambda x: (x['confidence'], x.get('score', 0)), reverse=True)
        best  = results[0]
        conf  = best['confidence']
        exch  = best.get('exchange', 'MEXC')

        # ── Risk rating banner ────────────────────────────────────────────
        regime_warn   = best.get('regime_warning')
        low_conf_warn = best.get('low_conf_warning')
        btc_regime    = best.get('btc_regime', get_btc_regime())

        if   conf >= 9: risk_tag = "✅ Strong signal — recommended"
        elif conf >= 8: risk_tag = "✅ Good signal — recommended"
        elif conf >= 7: risk_tag = "🟡 Moderate signal — trade normal size"
        elif conf >= 5: risk_tag = "🟠 Weak signal — reduce position size"
        elif conf >= 3: risk_tag = "🔴 Very weak signal — high risk"
        else:           risk_tag = "⛔ Extremely weak — not recommended"

        banner_lines = [
            f"📡 *{symbol}* — {tf_label}  |  {exch}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"🎯 Confidence: *{conf}/10*  {risk_tag}",
            f"📊 BTC Regime: *{btc_regime}*",
        ]
        if regime_warn:
            banner_lines.append(f"���️ Regime caution: {regime_warn}")
        if low_conf_warn:
            banner_lines.append(f"⚠️ Low confidence: {low_conf_warn}")
        if regime_warn or low_conf_warn:
            banner_lines.append("")
            banner_lines.append("_Signal shown as requested. Trade with caution and manage risk accordingly._")

        await update.message.reply_text("\n".join(banner_lines), parse_mode="Markdown")

        # Store for refresh / compare
        cscan_results[chat_id] = results
        # Persist to scan_results so /stats (best-signal leaderboard) reflects
        # single-pair scans, not just full market scans.
        try:
            db_save_scan(results)
        except Exception as _e_save:
            logger.debug("db_save_scan (single-pair) failed: %s", _e_save)
        if chat_id not in _chat_scan_ctx:
            _chat_scan_ctx[chat_id] = {}
        _chat_scan_ctx[chat_id]['custom'] = {
            'results':  results,
            'title':    f"📡 {symbol} [{tf_label}]",
            'max_show': 10,
        }
        _safemode_store_signals(chat_id, results)
        lev = best.get('leverage')
        compare_snapshot[chat_id] = {
            f"{exch}_{symbol}": {
                'entry_price': best['price'],
                'leverage':    lev['suggested'] if lev else 1,
                'bias':        best['bias'],
                'scan_time':   datetime.now(),
                'signal':      best
            }
        }

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Refresh", callback_data=f"cscan_refresh|{chat_id}|{exch}|{symbol}")],
            [
                InlineKeyboardButton("💰 PnL Calculator", callback_data=f"pnl_from_cscan|{chat_id}"),
                InlineKeyboardButton("🔗 Trade Now", url=get_exchange_link(exch, symbol))
            ]
        ])
        _ck, keyboard = cache_signal_card(best, 1, keyboard)
        await update.message.reply_text(format_signal_primary(best, 1), reply_markup=keyboard)
        return

    # ── /scan [tf] — TF sweep across top 30 pairs ─────────────
    if tf is not None:
        track_user_interaction(chat_id)
        tf_label = _tf_display(tf)
        await update.message.reply_text(
            f"🔍 TF SCAN — {tf_label}\n\n"
            f"Scanning top 30 pairs on {tf_label}...\n"
            f"⏳ Please wait ~60 seconds..."
        )

        top_syms = mexc_get_top_symbols(30)
        if not top_syms:
            top_syms = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
                        "DOGEUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]

        loop    = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            SCAN_EXECUTOR,
            lambda: [r for sym in top_syms
                       for (res, _) in [_cscan_pair_tf(sym, tf)]
                       for r in res]
        )

        if not results:
            await update.message.reply_text(
                f"⚠️ No signals on {tf_label} right now.\n"
                f"Market may be consolidating. Try a different timeframe."
            )
            return

        results.sort(key=lambda x: (x['confidence'], x['score']), reverse=True)
        seen = set(); deduped = []
        for r in results:
            if r['symbol'] not in seen:
                seen.add(r['symbol']); deduped.append(r)

        longs  = sum(1 for r in deduped if r['bias'] == 'LONG')
        shorts = sum(1 for r in deduped if r['bias'] == 'SHORT')
        await update.message.reply_text(
            f"✅ TF SCAN — {tf_label}\n"
            f"📊 {len(deduped)} signals  🟢 {longs}L  🔴 {shorts}S\n"
            f"Showing top 20:"
        )
        await send_signal_cards(
            update.message, deduped,
            title=f"📊 {tf_label} SIGNALS",
            max_show=20, chat_id=chat_id, source="scan"
        )
        return

    # ── /scan — full market scan (default, cached) ─────────────
    await scan_command(update, context)


# ─────────────────────────────────────────────
# /scan new — Newly listed pairs scanner
# Usage:
#   /scan new 24h   — pairs listed in last 24 hours
#   /scan new 30m   — last 30 minutes
#   /scan new 7d    — last 7 days
#   /scan new 2w    — last 2 weeks
# Supports: m (minutes), h (hours), d (days), w (weeks)
# ─────────────────────────────────────────────
def _parse_new_window(arg: str):
    """Parse a time window like 24h, 30m, 7d, 2w into seconds."""
    arg = arg.strip().lower()
    m = re.match(r'^(\d+)(m|h|d|w)$', arg)
    if not m:
        return None, None
    val  = int(m.group(1))
    unit = m.group(2)
    secs = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}[unit]
    label_map = {'m': f"{val} minute(s)", 'h': f"{val} hour(s)",
                 'd': f"{val} day(s)", 'w': f"{val} week(s)"}
    return val * secs, label_map[unit]

def _get_new_listings_mexc(since_seconds: int):
    """
    Fetch MEXC futures pairs and filter to those with candle history
    shorter than since_seconds — a proxy for newly listed.
    Returns list of symbol strings (BTCUSDT format).
    """
    try:
        r    = http_get("https://contract.mexc.com/api/v1/contract/ticker",
                            headers=HEADERS, timeout=15)
        data = r.json()
        if not data.get('success') or not data.get('data'):
            return []

        candidates = []
        cutoff_ts  = (datetime.now() - timedelta(seconds=since_seconds)).timestamp() * 1000

        for t in data['data']:
            sym = t.get('symbol', '')
            if not sym.endswith('_USDT'):
                continue
            try:
                price = float(t.get('lastPrice', 0) or 0)
                if price <= 0:
                    continue
                clean = sym.replace('_USDT', 'USDT')
                # Probe: fetch 1D candles and check how far back data goes
                r2 = http_get(
                    f"https://contract.mexc.com/api/v1/contract/kline/{sym}",
                    params={'interval': 'Day1', 'limit': 10},
                    headers=HEADERS, timeout=8
                )
                d2 = r2.json()
                if not d2.get('success') or not d2.get('data'):
                    continue
                times = d2['data'].get('time', [])
                if not times:
                    continue
                # Oldest candle timestamp (seconds)
                oldest_ts = float(times[0]) * 1000  # convert to ms
                if oldest_ts >= cutoff_ts:
                    # Listed within the window
                    candidates.append({'symbol': clean, 'price': price,
                                       'listed_ts': oldest_ts})
            except Exception:
                continue
            time.sleep(0.1)

        # Sort newest first
        candidates.sort(key=lambda x: x['listed_ts'], reverse=True)
        return [c['symbol'] for c in candidates]
    except Exception as e:
        logger.error("New listings MEXC error: %s", e)
        return []

def _get_new_listings_bybit(since_seconds: int):
    """Bybit: use instrument info launchTime field."""
    if sakz_exchanges.BYBIT_AVAILABLE is False:
        return []
    try:
        r    = http_get("https://api.bybit.com/v5/market/instruments-info",
                            params={'category': 'linear', 'limit': 1000},
                            headers=HEADERS, timeout=15)
        data = r.json()
        if data.get('retCode') != 0:
            return []
        cutoff_ms = (datetime.now() - timedelta(seconds=since_seconds)).timestamp() * 1000
        symbols   = []
        for inst in data['result']['list']:
            sym        = inst.get('symbol', '')
            launch_str = inst.get('launchTime', '0')
            if not sym.endswith('USDT'):
                continue
            try:
                launch_ms = float(launch_str)
                if launch_ms >= cutoff_ms:
                    symbols.append(sym)
            except Exception:
                continue
        return symbols
    except Exception as e:
        logger.error("New listings Bybit error: %s", e)
        return []

def _run_new_listings_scan(since_seconds: int):
    """Blocking: fetch new listings and analyse them."""
    results = []

    bybit_new = _get_new_listings_bybit(since_seconds)
    mexc_new  = _get_new_listings_mexc(since_seconds)

    logger.info("New listings — Bybit: %d, MEXC: %d", len(bybit_new), len(mexc_new))

    # FIX #NL-ANALYZER — new listings have <30 4H candles, so analyze_bybit/
    # analyze_mexc (which REQUIRE 30x4H + 15x1D candles) silently return None
    # for every freshly-listed pair, and the whole scan came back empty.
    # analyze_symbol_new_listing is purpose-built for low-history pairs: it
    # walks 5m -> 15m -> 1h -> 4h, probes each available exchange, and
    # self-confirms on the primary frame when no daily data exists yet.
    # Merge both exchanges into one ordered, de-duplicated symbol set so a pair
    # listed on both venues is scored once (the analyzer already tries all venues).
    combined = list(dict.fromkeys(bybit_new + mexc_new))
    for sym in combined[:40]:
        r = analyze_symbol_new_listing(sym)
        # Returns a signal dict, a ScanFailure, or None — only keep real signals.
        if r and not isinstance(r, ScanFailure):
            results.append(r)
        time.sleep(0.2)

    results.sort(key=lambda x: (x['confidence'], x['score']), reverse=True)
    return results, len(bybit_new), len(mexc_new)


async def scan_new_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /scan new [window]
    Examples:
      /scan new 24h   ����� pairs listed in last 24 hours
      /scan new 30m   — last 30 minutes
      /scan new 7d    — last 7 days
      /scan new 2w    — last 2 weeks
    """
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args or []

    # Filter out 'new' keyword if passed directly
    filtered = [a for a in args if a.lower() != 'new']

    if not filtered:
        await update.message.reply_text(
            "🆕 NEW LISTINGS SCANNER\n\n"
            "Scans pairs newly listed on MEXC & Bybit\n"
            "within a specified time window.\n\n"
            "Usage:\n"
            "  /scan new 24h  — last 24 hours\n"
            "  /scan new 30m  — last 30 minutes\n"
            "  /scan new 7d   — last 7 days\n"
            "  /scan new 2w   — last 2 weeks\n\n"
            "Units: m=minutes  h=hours  d=days  w=weeks"
        )
        return

    secs, label = _parse_new_window(filtered[0])
    if secs is None:
        await update.message.reply_text(
            f"⚠️ Invalid time window: `{filtered[0]}`\n\n"
            f"Examples: 24h  30m  7d  2w\n"
            f"Units: m=minutes  h=hours  d=days  w=weeks",
            parse_mode="Markdown"
        )
        return

    track_user_interaction(chat_id)

    await update.message.reply_text(
        f"🆕 Scanning new listings — last {label}\n\n"
        f"📡 Checking MEXC & Bybit for recently listed pairs...\n"
        f"⏳ Please wait ~60 seconds..."
    )

    loop = asyncio.get_event_loop()
    try:
        results, bybit_n, mexc_n = await loop.run_in_executor(
            SCAN_EXECUTOR, lambda: _run_new_listings_scan(secs)
        )
    except Exception as e:
        await update.message.reply_text(f"❌ New listings scan failed: {e}")
        return

    total_found = bybit_n + mexc_n
    if total_found == 0:
        await update.message.reply_text(
            f"📭 No new pairs found listed in the last {label}.\n\n"
            f"Try a longer window:\n"
            f"  /scan new 7d  or  /scan new 2w"
        )
        return

    if not results:
        await update.message.reply_text(
            f"🆕 Found {total_found} new listing(s) in last {label}\n"
            f"(Bybit: {bybit_n}  |  MEXC: {mexc_n})\n\n"
            f"⚠️ None produced tradeable signals yet.\n"
            f"New listings often lack candle history — try a wider window."
        )
        return

    longs  = sum(1 for r in results if r['bias'] == 'LONG')
    shorts = sum(1 for r in results if r['bias'] == 'SHORT')

    await update.message.reply_text(
        f"✅ NEW LISTINGS SCAN — last {label}\n"
        f"📡 Bybit: {bybit_n} new  |  MEXC: {mexc_n} new\n"
        f"📊 Signals: {len(results)}  🟢 {longs}L  🔴 {shorts}S\n\n"
        f"⚠️ New listings = higher risk. Less candle history,\n"
        f"   lower liquidity, wider spreads. Use lower leverage."
    )
    await send_signal_cards(
        update.message, results,
        title=f"🆕 NEW LISTINGS — {label}",
        max_show=20, chat_id=chat_id, source="scan"
    )


async def cscan_tf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /cscan BTC [t15m]  — scan a specific pair on a specific timeframe.
    Falls back to default 4h if no TF given.
    """
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args or []

    if not args:
        await update.message.reply_text(
            "📡 CUSTOM PAIR SCAN\n\n"
            "Usage: /cscan BTC\n"
            "       /cscan ETH t15m\n"
            "       /cscan SOLUSDT t1h\n\n"
            "Supported timeframes: 1m 3m 5m 15m 30m 1h 2h 4h 6h 12h 1d\n"
            "Prefix with 't' is optional — /cscan SOL 15m also works."
        )
        return

    track_user_interaction(chat_id)

    # Parse symbol and optional TF from args
    tf = None; sym_arg = None
    for a in args:
        parsed = _parse_tf_arg(a)
        if parsed and tf is None:
            tf = parsed
        elif sym_arg is None:
            sym_arg = a

    if sym_arg is None:
        await update.message.reply_text("⚠️ Please specify a coin. e.g. /cscan BTC t15m")
        return

    raw    = sym_arg.upper().replace('/', '').strip()
    symbol = raw if raw.endswith('USDT') else raw + 'USDT'

    if tf:
        await update.message.reply_text(
            "🔍 Starting full scan across exchanges\n"
            "📡 ......\n"
            "⏳ Please wait 5–10 minutes..."
        )
        loop     = asyncio.get_event_loop()
        results, failures = await loop.run_in_executor(SCAN_EXECUTOR, lambda: _cscan_pair_tf(symbol, tf))
        tf_label = _tf_display(tf)
    else:
        await update.message.reply_text(
            "🔍 Starting full scan across exchanges\n"
            "📡 ......\n"
            "⏳ Please wait 5–10 minutes..."
        )
        loop     = asyncio.get_event_loop()
        results, failures = await loop.run_in_executor(SCAN_EXECUTOR, lambda: _cscan_pair_tf(symbol, '4h'))
        tf_label = "4H (default)"

    if not results:
        # ── Smart failure diagnosis (same logic as scan_tf_command) ──────
        sym_base      = symbol.replace('USDT', '')
        actual_tf     = tf or '4h'
        reasons_seen  = [f.reason for f in failures]
        details       = [f.detail for f in failures if f.detail]
        hint          = f"\n   ({details[0]})" if details else ""

        if REASON_GAP_BLOCK in reasons_seen:
            gap_d = next((f.detail for f in failures if f.reason == REASON_GAP_BLOCK), 'score gap < 2')
            msg = (
                f"↔️ *{symbol}* — Signal too close to call on {tf_label}\n\n"
                f"Long and Short scores are nearly equal — choppy/sideways market.\n\n"
                f"Detail: `{gap_d}`\n\n"
                f"💡 Try `/cscan {sym_base} 1h` for a shorter timeframe view."
            )
        elif REASON_COUNTER_TREND in reasons_seen:
            ct_d = next((f.detail for f in failures if f.reason == REASON_COUNTER_TREND), 'see logs')
            msg = (
                f"↔️ *{symbol}* — Counter-trend signal blocked on {tf_label}\n\n"
                f"The bias opposes the daily EMA — higher score required.\n\n"
                f"Detail: `{ct_d}`\n\n"
                f"💡 Try `/cscan {sym_base} 1d` to see higher-timeframe view."
            )
        elif REASON_FLIP_BLOCK in reasons_seen:
            flip_d = next((f.detail for f in failures if f.reason == REASON_FLIP_BLOCK), 'see logs')
            msg = (
                f"🔄 *{symbol}* — Direction flip blocked on {tf_label}\n\n"
                f"Opposite direction fired recently. Needs conf ≥ 9 to reverse.\n\n"
                f"Detail: `{flip_d}`"
            )
        elif REASON_LOW_CONF in reasons_seen:
            lc_d = next((f.detail for f in failures if f.reason == REASON_LOW_CONF), 'conf < 4')
            msg = (
                f"🟠 *{symbol}* — Signal too weak on {tf_label}\n\n"
                f"Detail: `{lc_d}`\n\n"
                f"💡 Try `/cscan {sym_base} 1h` or check back after next candle."
            )
        elif not failures or REASON_NO_CONTRACT in reasons_seen:
            msg = (
                f"❌ *{symbol}* not found on any exchange for {tf_label}.{hint}\n\n"
                f"• Check the symbol spelling (e.g. ZKC not ZKCU)\n"
                f"• Pair may not be listed as a perpetual future\n"
                f"• Some tokens use `1000{sym_base}USDT` format (e.g. 1000BONKUSDT)\n"
                f"• Try `/scan` to see all available top pairs"
            )
        elif REASON_SHORT_HISTORY in reasons_seen:
            msg = (
                f"📋 *{symbol}* — not enough candle history on {tf_label}.{hint}\n\n"
                f"• Try `/cscan {sym_base} 1h` for a shorter timeframe\n"
                f"• Pair may be newly listed — check back later"
            )
        elif REASON_LOW_VOLUME in reasons_seen:
            best_detail = next((f.detail for f in failures
                                if f.reason == REASON_LOW_VOLUME and f.detail), "")
            msg = (
                f"💧 *{symbol}* ��� Insufficient liquidity\n\n"
                f"{best_detail}\n\n"
                f"This pair doesn't have enough 24h volume for reliable signals.\n"
                f"Try higher-volume pairs from /scan instead."
            )
        else:
            msg = (
                f"⚠️ *{symbol}* — no signal produced on {tf_label}.{hint}\n"
                f"Try `/cscan {sym_base} 1h` or `/chart {sym_base} {actual_tf}`"
            )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    results.sort(key=lambda x: (x['confidence'], x['score']), reverse=True)
    best = results[0]
    cscan_results[chat_id] = results
    # Persist so /stats leaderboard includes this scan.
    try:
        db_save_scan(results)
    except Exception as _e_save:
        logger.debug("db_save_scan (custom scan) failed: %s", _e_save)
    _chat_scan_ctx[chat_id] = {
        'source':   'custom',
        'results':  results,
        'title':    f"📡 {best['symbol']}  [{tf_label}]",
        'max_show': 15,
    }
    _safemode_store_signals(chat_id, results)
    lev = best.get('leverage')
    compare_snapshot[chat_id] = {
        f"{best['exchange']}_{best['symbol']}": {
            'entry_price': best['price'],
            'leverage':    lev['suggested'] if lev else 1,
            'bias':        best['bias'],
            'scan_time':   datetime.now(),
            'signal':      best
        }
    }

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Refresh (live price + PnL)", callback_data=f"cscan_refresh|{chat_id}|{best['exchange']}|{symbol}")],
        [
            InlineKeyboardButton("💰 PnL Calculator", callback_data=f"pnl_from_cscan|{chat_id}"),
            InlineKeyboardButton("🔗 Trade Now", url=get_exchange_link(best['exchange'], best['symbol']))
        ]
    ])

    header = (
        f"📡 CUSTOM SCAN — {symbol}  [{tf_label}]\n"
        f"{'━'*30}\n"
        f"Found {len(results)} result(s) across exchanges\n"
        f"Showing best signal:\n"
    )
    await update.message.reply_text(header)
    _ck, keyboard = cache_signal_card(best, 1, keyboard)
    await update.message.reply_text(format_signal_primary(best, 1), reply_markup=keyboard)


# ─── TREND-DYING MONITOR ─────────────────────────────���────────
# Stored as { chat_id: { symbol+exchange: {signal, notified_dying} } }
_trend_monitor_cache = {}

def _check_trend_dying(r: dict, df4h, df1d) -> tuple[bool, str]:
    """
    Returns (is_dying, reason_message).
    A trend is 'dying' when 2+ of these flip against the original bias:
    - MACD crosses against bias
    - RSI moves into opposite extreme zone
    - Price breaks through EMA20 against bias
    - Volume dries up (<0.5x MA) on attempted continuation
    """
    if df4h is None or len(df4h) < 5:
        return False, ""

    bias = r['bias']
    L    = df4h.iloc[-1]
    P    = df4h.iloc[-2]
    warns = []

    macd_now  = L.get('macd_diff', 0)
    macd_prev = P.get('macd_diff', 0)
    if bias == 'LONG' and macd_now < 0 and macd_prev >= 0:
        warns.append("📉 MACD just crossed bearish — momentum flipping")
    elif bias == 'SHORT' and macd_now > 0 and macd_prev <= 0:
        warns.append("📈 MACD just crossed bullish — downtrend losing steam")

    rsi = L.get('rsi', 50)
    if bias == 'LONG' and rsi > 72:
        warns.append(f"⚠️ RSI overbought ({rsi:.1f}) — rally may be exhausted")
    elif bias == 'SHORT' and rsi < 28:
        warns.append(f"⚠️ RSI oversold ({rsi:.1f}) — sell-off may be exhausted")

    price  = L.get('close', 0)
    ema20  = L.get('ema20', price)
    if bias == 'LONG' and price < ema20 and P.get('close', price) >= P.get('ema20', ema20):
        warns.append(f"❌ Price broke below EMA20 (${ema20:.4f}) — support lost")
    elif bias == 'SHORT' and price > ema20 and P.get('close', price) <= P.get('ema20', ema20):
        warns.append(f"❌ Price broke above EMA20 (${ema20:.4f}) — resistance broken")

    vol    = L.get('volume', 0)
    vol_ma = L.get('volume_ma', vol)
    if vol_ma > 0 and vol < vol_ma * 0.5:
        warns.append(f"📊 Volume dried up ({vol/vol_ma:.1f}x avg) — weak follow-through")

    return len(warns) >= 2, "\n".join(f"  {w}" for w in warns)


# ─────────────────────────────────���───────────────────────────────────────────
# /analyse — Raw market analysis, no regime gate, no confidence floor
#
# Usage:
#   /analyse ZBT          → auto-selects 4H, MEXC
#   /analyse ZBT 1h       → pins to 1H
#   /analyse ZBT 10m      → maps 10m to nearest supported TF (15m)
#   /analyse BTCUSDT 4h   ��� explicit USDT pair
#
# Emits a raw indicator snapshot and directional read regardless of whether
# the market is trending, ranging, or regime-divergent. Always preceded by
# a hot-ground risk warning to the user.
#
# TF mapping for non-standard timeframes:
#   ≤7 min  → 15m   |   8–22 min  → 15m   |   23–90 min → 1h
#   91–360 min → 4h  |   >360 min → 1d
# ───────────────────────────────────────��────────────────────────────────────���

# Extended TF alias map that includes non-standard intervals (maps to nearest
# supported TF).  Values are TF_CONFIGS keys: '15m' | '1h' | '4h' | '1d'.
_ANALYSE_TF_ALIASES = {
    # Exact standard aliases
    '15m': '15m', '15': '15m', '15min': '15m',
    '1h':  '1h',  '60m': '1h', '60': '1h', 'h1': '1h', '1': '1h',
    '4h':  '4h',  '240m': '4h', '240': '4h', 'h4': '4h', '4': '4h',
    '1d':  '1d',  'daily': '1d', 'd': '1d', 'day': '1d',
    # Non-standard — map to nearest supported TF
    '1m': '15m', '2m': '15m', '3m': '15m', '5m': '15m',
    '7m': '15m', '10m': '15m', '12m': '15m',
    '20m': '15m', '25m': '1h', '30m': '1h', '45m': '1h',
    '2h': '1h', '3h': '4h', '6h': '4h', '8h': '4h',
    '12h': '4h', '16h': '1d', '2d': '1d', '3d': '1d', 'w': '1d', '1w': '1d',
}

# What we tell the user when their requested TF was remapped
_ANALYSE_TF_REMAP_NOTE = {
    '1m': '1m → 15m (smallest supported)',
    '2m': '2m → 15m',
    '3m': '3m → 15m',
    '5m': '5m → 15m',
    '7m': '7m → 15m',
    '10m': '10m → 15m (nearest supported)',
    '12m': '12m → 15m',
    '20m': '20m → 15m',
    '25m': '25m → 1h',
    '30m': '30m → 1h',
    '45m': '45m → 1h',
    '2h':  '2h → 1h',
    '3h':  '3h → 4h',
    '6h':  '6h → 4h',
    '8h':  '8h → 4h',
    '12h': '12h → 4h',
    '16h': '16h → 1d',
    '2d':  '2d → 1d',
    '3d':  '3d → 1d',
    'w':   '1W → 1d (weekly not supported)',
    '1w':  '1W → 1d',
}


def _analyse_raw_indicators(symbol: str, tf_key: str):
    """
    Fetch candles + compute indicators for /analyse.
    Returns a dict of raw values, or a string error message on failure.
    No gates, no scoring engine — pure snapshot.
    """
    # MEXC interval map (same keys as TF_CONFIGS)
    mexc_interval_map = {
        '15m': 'Min15',
        '1h':  'Min60',
        '4h':  'Hour4',
        '1d':  'Day1',
    }
    mexc_interval = mexc_interval_map.get(tf_key, 'Hour4')
    limit = 120   # enough for indicators + lookback

    try:
        sym_clean = symbol.upper().replace('_USDT', 'USDT').replace('/', '')
        if sym_clean.endswith('USDT'):
            futures_sym = sym_clean[:-4] + '_USDT'
        else:
            futures_sym = sym_clean + '_USDT'


        r = requests.get(
            f"https://contract.mexc.com/api/v1/contract/kline/{futures_sym}",
            params={'interval': mexc_interval, 'limit': limit},
            headers={'User-Agent': 'SakzBot/1.0'},
            timeout=20,
        )
        if r.status_code != 200:
            return f"MEXC API error {r.status_code} — pair may not exist as a perpetual"

        data = r.json()
        if not data.get('success') or not data.get('data'):
            return f"No data for {symbol} on MEXC perpetuals — check the symbol"

        d = data['data']

        df = pd.DataFrame({
            'timestamp': d.get('time', []),
            'open':      d.get('open', []),
            'high':      d.get('high', []),
            'low':       d.get('low', []),
            'close':     d.get('close', []),
            'volume':    d.get('vol', []),
        })
        if len(df) < 20:
            return f"Not enough candle history for {symbol} on {tf_key} (got {len(df)} candles)"

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()

        # Add indicators (use '4h' stoch window for anything ≤4h, else '1d')
        stoch_tf = '4h' if tf_key in ('15m', '1h', '4h') else '1d'
        df_ind = add_indicators(df, timeframe=stoch_tf)
        if df_ind is None or len(df_ind) < 5:
            return f"Indicator calculation failed — not enough clean candles"

        L   = df_ind.iloc[-1]   # latest closed candle
        P   = df_ind.iloc[-2]   # previous candle
        P2  = df_ind.iloc[-3]   # two candles back

        price      = float(L['close'])
        rsi        = float(L['rsi'])
        macd_diff  = float(L['macd_diff'])
        prev_diff  = float(P['macd_diff'])
        macd_line  = float(L.get('macd_line', L.get('macd', 0)))
        macd_sig   = float(L.get('macd_signal', 0))
        bb_upper   = float(L['bb_upper'])
        bb_lower   = float(L['bb_lower'])
        bb_mid     = float(L['bb_mid'])
        bb_bw      = float(L.get('bb_bw', (bb_upper - bb_lower) / max(bb_mid, 0.0001)))
        bb_bw_min  = float(L.get('bb_bw_min', bb_bw))
        ema20      = float(L['ema20'])
        ema50      = float(L['ema50'])
        stoch_k    = float(L['stoch_k'])
        stoch_d    = float(L['stoch_d'])
        atr        = float(L['atr'])
        volume     = float(L['volume'])
        vol_ma     = float(L.get('volume_ma', volume))
        support    = float(L.get('support', price * 0.97))
        resistance = float(L.get('resistance', price * 1.03))
        clv_ma     = float(L.get('clv_ma', 0))

        # Derive raw bias signals (no gating)
        bull_signals, bear_signals = [], []

        # RSI
        if rsi < 30:    bull_signals.append(f"RSI {rsi:.1f} — extremely oversold 🔴")
        elif rsi < 40:  bull_signals.append(f"RSI {rsi:.1f} — oversold")
        elif rsi < 48:  bull_signals.append(f"RSI {rsi:.1f} — leaning oversold")
        elif rsi > 70:  bear_signals.append(f"RSI {rsi:.1f} — extremely overbought 🔴")
        elif rsi > 60:  bear_signals.append(f"RSI {rsi:.1f} — overbought")
        elif rsi > 52:  bear_signals.append(f"RSI {rsi:.1f} — leaning overbought")
        else:
            if rsi > float(P['rsi']): bull_signals.append(f"RSI {rsi:.1f} — crossing 50 upward")
            elif rsi < float(P['rsi']): bear_signals.append(f"RSI {rsi:.1f} — crossing 50 downward")

        # MACD
        if macd_diff > 0 and prev_diff <= 0:
            bull_signals.append(f"MACD bullish crossover (hist={macd_diff:+.4f})")
        elif macd_diff > 0 and prev_diff > 0:
            bull_signals.append(f"MACD hist positive & growing" if macd_diff > prev_diff else f"MACD hist positive but fading")
        elif macd_diff < 0 and prev_diff >= 0:
            bear_signals.append(f"MACD bearish crossover (hist={macd_diff:+.4f})")
        elif macd_diff < 0 and prev_diff < 0:
            bear_signals.append(f"MACD hist negative & deepening" if macd_diff < prev_diff else f"MACD hist negative but recovering")

        # EMA stack
        if price > ema20 > ema50:   bull_signals.append(f"Price > EMA20 > EMA50 — bullish stack")
        elif price < ema20 < ema50: bear_signals.append(f"Price < EMA20 < EMA50 — bearish stack")
        elif price > ema20:         bull_signals.append(f"Price above EMA20 (${ema20:.4f})")
        elif price < ema20:         bear_signals.append(f"Price below EMA20 (${ema20:.4f})")

        # BB position
        bb_pct = (price - bb_lower) / max(bb_upper - bb_lower, 0.0001) * 100
        if price >= bb_upper * 0.99:
            bear_signals.append(f"Price at/above BB upper (${bb_upper:.4f}) — overextended")
        elif price <= bb_lower * 1.01:
            bull_signals.append(f"Price at/below BB lower (${bb_lower:.4f}) — potential reversal zone")
        else:
            if bb_pct > 65:   bear_signals.append(f"Price in upper BB band ({bb_pct:.0f}%)")
            elif bb_pct < 35: bull_signals.append(f"Price in lower BB band ({bb_pct:.0f}%)")

        # BB squeeze breakout
        squeeze_breaking = bb_bw > bb_bw_min * 1.05 if bb_bw_min > 0 else False
        if squeeze_breaking:
            if macd_diff > 0: bull_signals.append("BB squeeze expanding — breakout attempt (bullish direction)")
            else: bear_signals.append("BB squeeze expanding — breakout attempt (bearish direction)")

        # Stochastic
        if stoch_k < 20:   bull_signals.append(f"Stoch K {stoch_k:.1f} — deeply oversold")
        elif stoch_k < 30: bull_signals.append(f"Stoch K {stoch_k:.1f} — oversold")
        elif stoch_k > 80: bear_signals.append(f"Stoch K {stoch_k:.1f} — deeply overbought")
        elif stoch_k > 70: bear_signals.append(f"Stoch K {stoch_k:.1f} — overbought")
        # Stoch cross
        if stoch_k > stoch_d and float(P['stoch_k']) <= float(P['stoch_d']):
            bull_signals.append(f"Stoch K crossed above D — bullish signal")
        elif stoch_k < stoch_d and float(P['stoch_k']) >= float(P['stoch_d']):
            bear_signals.append(f"Stoch K crossed below D — bearish signal")

        # Volume
        vol_ratio = volume / max(vol_ma, 0.0001)
        if vol_ratio > 1.5:   vol_note = f"Volume {vol_ratio:.1f}x avg — elevated (confirms move)"
        elif vol_ratio < 0.5: vol_note = f"Volume {vol_ratio:.1f}x avg — thin (low conviction)"
        else:                 vol_note = f"Volume {vol_ratio:.1f}x avg — normal"

        # CLV (order absorption)
        if clv_ma > 0.3:    bull_signals.append(f"CLV {clv_ma:+.2f} — buyers absorbing candles")
        elif clv_ma < -0.3: bear_signals.append(f"CLV {clv_ma:+.2f} — sellers absorbing candles")

        # Pivot S/R proximity
        dist_to_res = (resistance - price) / price * 100
        dist_to_sup = (price - support) / price * 100
        if dist_to_res < 1.0: bear_signals.append(f"Price within {dist_to_res:.1f}% of resistance (${resistance:.4f})")
        if dist_to_sup < 1.0: bull_signals.append(f"Price within {dist_to_sup:.1f}% of support (${support:.4f})")

        # Raw lean
        if len(bull_signals) > len(bear_signals) + 1:
            lean = 'BULLISH'
            lean_emoji = '🟢'
        elif len(bear_signals) > len(bull_signals) + 1:
            lean = 'BEARISH'
            lean_emoji = '🔴'
        elif len(bull_signals) == 0 and len(bear_signals) == 0:
            lean = 'FLAT'
            lean_emoji = '⚪'
        else:
            lean = 'MIXED'
            lean_emoji = '🟡'

        btc_regime = get_btc_regime()

        return {
            'symbol':       symbol,
            'tf_key':       tf_key,
            'tf_label':     TF_CONFIGS[tf_key]['label'],
            'price':        price,
            'rsi':          rsi,
            'macd_diff':    macd_diff,
            'macd_line':    macd_line,
            'macd_sig':     macd_sig,
            'bb_upper':     bb_upper,
            'bb_lower':     bb_lower,
            'bb_mid':       bb_mid,
            'bb_pct':       bb_pct,
            'ema20':        ema20,
            'ema50':        ema50,
            'stoch_k':      stoch_k,
            'stoch_d':      stoch_d,
            'atr':          atr,
            'vol_ratio':    vol_ratio,
            'vol_note':     vol_note,
            'support':      support,
            'resistance':   resistance,
            'clv_ma':       clv_ma,
            'bull_signals': bull_signals,
            'bear_signals': bear_signals,
            'lean':         lean,
            'lean_emoji':   lean_emoji,
            'btc_regime':   btc_regime,
            'candles':      len(df_ind),
        }

    except Exception as e:
        logger.warning("_analyse_raw_indicators %s %s: %s", symbol, tf_key, e)
        return f"Analysis failed: {e}"


# In-memory cache so the /analyse section buttons can rebuild each view in-place.
# Key: "<chat_id>:<symbol>:<tf_key>" -> result dict from _analyse_raw_indicators.
_analyse_cache: dict = {}


def _an_pre(lines: list) -> str:
    """Join lines, HTML-escape, and wrap in a <pre> block (monospace in Telegram)."""
    body = "\n".join(lines)
    body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<pre>{body}</pre>"


def _an_regime_emoji(regime: str) -> str:
    return {
        'STRONG_BULL': '🟢🟢', 'BULL': '🟢',
        'NEUTRAL': '⚪',
        'BEAR': '🔴', 'STRONG_BEAR': '🔴🔴',
    }.get(regime, '⚪')


def _analyse_kb(symbol: str, tf_key: str, chat_id, n_bull: int, n_bear: int,
                active: str = 'sum') -> InlineKeyboardMarkup:
    """Inline keyboard for the /analyse card. `active` marks the open section."""
    def b(label, sec):
        mark = "• " if sec == active else ""
        return InlineKeyboardButton(f"{mark}{label}",
                                    callback_data=f"an_sec|{sec}|{symbol}|{tf_key}")
    rows = []
    if active != 'sum':
        rows.append([InlineKeyboardButton(
            "⬅️ Summary", callback_data=f"an_sec|sum|{symbol}|{tf_key}")])
    rows.append([b("📊 Indicators", "ind")])
    rows.append([b(f"🟢 Bullish ({n_bull})", "bull"),
                 b(f"🔴 Bearish ({n_bear})", "bear")])
    rows.append([
        InlineKeyboardButton("📈 Chart",
                             callback_data=f"chart_tf_refresh|{symbol}|{tf_key}|{chat_id}"),
        InlineKeyboardButton("📡 Full Scan", callback_data="menu_run|cscan"),
    ])
    return InlineKeyboardMarkup(rows)


def _analyse_build_signal(data: dict):
    """Convert a raw /analyse indicator snapshot into a scan-style signal dict,
    or return None when the market is genuinely neutral (no directional edge).

    Confidence is derived from how strongly the bullish/bearish indicators
    diverge, deliberately capped in the 4–7 band because /analyse is ungated
    (information only) and must never masquerade as a high-conviction signal."""
    try:
        lean = data.get('lean')
        if lean == 'BULLISH':
            bias = 'LONG'
        elif lean == 'BEARISH':
            bias = 'SHORT'
        else:
            return None   # FLAT / MIXED → neutral, no confidence

        price = float(data.get('price') or 0)
        if price <= 0:
            return None
        atr = float(data.get('atr') or 0) or price * 0.01

        nb = len(data.get('bull_signals', []))
        ns = len(data.get('bear_signals', []))
        diff = abs(nb - ns)
        conf = max(4, min(7, 3 + diff))   # 2-net→5, 3→6, 4+→7

        band = max(atr * 0.25, price * 0.002)
        entry_low  = price - band
        entry_high = price + band
        if bias == 'LONG':
            t1, t2, t3 = price + atr * 1.0, price + atr * 1.8, price + atr * 3.0
            stop_loss  = price - atr * 1.5
        else:
            t1, t2, t3 = price - atr * 1.0, price - atr * 1.8, price - atr * 3.0
            stop_loss  = price + atr * 1.5

        try:
            lev = calculate_leverage(price, entry_low, entry_high, stop_loss, atr, conf, bias)
        except Exception:
            lev = None

        reasons = (data.get('bull_signals') if bias == 'LONG' else data.get('bear_signals')) or []
        return {
            'symbol': data['symbol'], 'exchange': 'MEXC', 'bias': bias,
            'confidence': conf, 'price': price,
            'entry_low': entry_low, 'entry_high': entry_high,
            't1': t1, 't2': t2, 't3': t3, 'stop_loss': stop_loss,
            'leverage': lev, 'signal_tf_label': data.get('tf_label', '4H'),
            'reasons': reasons, 'scan_time': datetime.now(),
        }
    except Exception as e:
        logger.debug("_analyse_build_signal failed: %s", e)
        return None


def _format_analyse_signal_card(r: dict) -> str:
    """Scan-style card for /analyse when a directional lean exists. Mirrors the
    /scan primary card layout so the two feel identical, but is clearly badged
    as an ungated ANALYSE read and shown without the live auto-refresh footer."""
    conf = r['confidence']
    bias = r['bias']
    bias_emoji = "🟢" if bias == 'LONG' else "🔴"
    panel = _panel_type(r.get('signal_tf_label', '4H'))
    rr = _rr_ratio(r)
    price = r['price'] if r['price'] > 0 else 1

    def _pct(tp):
        raw = (tp - price) / price * 100
        return raw if bias == 'LONG' else -raw

    lev = r.get('leverage')
    lev_str = f"{lev['suggested']}x" if lev else "1x"
    rr_str = f"1:{rr:.1f}" if rr is not None else "N/A"

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  🔬 {panel} | ANALYSE",
        f"  CONFIDENCE: {_conf_bar_emoji(conf)} {conf}/10",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"📊 Token: {r['symbol']}",
        f"{bias_emoji} Direction: {bias}",
        f"💰 Entry: ${r['entry_low']:.4f} - ${r['entry_high']:.4f}",
        f"⚡️ Leverage: {lev_str}",
        f"📐 R:R: {rr_str}",
        "",
        f"🎯 TP1: ${r['t1']:.4f}  (+{_pct(r['t1']):.1f}% profit) 💰",
        f"🎯 TP2: ${r['t2']:.4f}  (+{_pct(r['t2']):.1f}%)",
        f"🎯 TP3: ${r['t3']:.4f}  (+{_pct(r['t3']):.1f}%)",
        f"🛑 Stop Loss: ${r['stop_loss']:.4f}",
        "",
        "🔬 Ungated /analyse read — derived from raw indicators, not a gated",
        "   signal. Levels are ATR-based estimates. Tap below for detail.",
    ]
    return "\n".join(lines)


def _format_analyse_neutral_card(data: dict) -> str:
    """Neutral /analyse card — NO confidence shown, because the indicators are
    balanced and there is no directional edge to score."""
    lean = data.get('lean', 'MIXED')
    le   = data.get('lean_emoji', '⚪')
    nb   = len(data.get('bull_signals', []))
    ns   = len(data.get('bear_signals', []))
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  🔬 ANALYSE | {data.get('tf_label', '')}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"📊 Token: {data['symbol']}",
        f"💰 Price: ${data['price']:.6g}",
        f"📈 Support: ${data['support']:.6g}    📉 Resistance: ${data['resistance']:.6g}",
        "",
        f"{le} No directional edge — {lean} ({nb} bull / {ns} bear)",
        "   Confidence is withheld: the bullish/bearish signals are balanced.",
        f"{_an_regime_emoji(data['btc_regime'])} BTC Regime: {data['btc_regime']}",
        f"   {data['vol_note']}",
        "",
        "Tap below for indicators, bullish/bearish detail, or the chart.",
    ]
    return "\n".join(lines)


def _format_analyse_card(data: dict, requested_tf_raw: str | None) -> str:
    """Unified /analyse summary — scan-style card. Shows a confidence-scored
    signal card when there's a directional lean, or a no-confidence neutral
    card when indicators are balanced. Detail lives behind the buttons."""
    _sig = _analyse_build_signal(data)
    if _sig is not None:
        return _format_analyse_signal_card(_sig)
    return _format_analyse_neutral_card(data)


def _format_analyse_card_legacy(data: dict, requested_tf_raw: str | None) -> str:
    """Legacy brief summary (kept for reference / fallback)."""
    sym        = data['symbol']
    tf_label   = data['tf_label']
    price      = data['price']
    lean       = data['lean']
    lean_emoji = data['lean_emoji']
    regime     = data['btc_regime']
    nb         = len(data['bull_signals'])
    nbear      = len(data['bear_signals'])

    remap_note = ''
    if requested_tf_raw and requested_tf_raw in _ANALYSE_TF_REMAP_NOTE:
        remap_note = f"  ⚠️ remapped: {_ANALYSE_TF_REMAP_NOTE[requested_tf_raw]}"

    def _row(label, value):
        return f"  {label:<12}{value}"

    lines = [
        f"🔬 {sym}  ·  {tf_label}{remap_note}",
        "═══════════════════════════",
        _row("Price", f"${price:.6g}"),
        _row("Support", f"${data['support']:.6g}"),
        _row("Resistance", f"${data['resistance']:.6g}"),
        "",
        f"{lean_emoji} Lean: {lean}  ({nb} bull / {nbear} bear)",
        f"{_an_regime_emoji(regime)} BTC Regime: {regime}",
        f"  {data['vol_note']}",
        "",
        "ℹ️ Raw data — no gates. Tap below for detail.",
    ]
    return _an_pre(lines)


def _format_analyse_indicators(data: dict) -> str:
    """Full indicator readout for the /analyse Indicators button."""
    def _row(label, value):
        return f"  {label:<12}{value}"
    lines = [
        f"📊 INDICATORS — {data['symbol']} · {data['tf_label']}",
        "══════════���════════════════",
        _row("Price", f"${data['price']:.6g}"),
        _row("RSI", f"{data['rsi']:.1f}"),
        _row("MACD hist", f"{data['macd_diff']:+.5f}"),
        _row("EMA 20", f"${data['ema20']:.6g}"),
        _row("EMA 50", f"${data['ema50']:.6g}"),
        _row("BB %", f"{data['bb_pct']:.0f}%"),
        _row("BB range", f"${data['bb_lower']:.6g} – ${data['bb_upper']:.6g}"),
        _row("Stoch K/D", f"{data['stoch_k']:.1f} / {data['stoch_d']:.1f}"),
        _row("ATR", f"${data['atr']:.6g}"),
        _row("CLV MA", f"{data['clv_ma']:+.2f}"),
        f"  {data['vol_note']}",
    ]
    return _an_pre(lines)


def _format_analyse_signals(data: dict, side: str) -> str:
    """Bullish or bearish signal list for the /analyse section buttons."""
    if side == 'bull':
        title = f"🟢 BULLISH SIGNALS — {data['symbol']} · {data['tf_label']}"
        sigs  = data['bull_signals']
    else:
        title = f"🔴 BEARISH SIGNALS — {data['symbol']} · {data['tf_label']}"
        sigs  = data['bear_signals']
    lines = [title, "═══��═══════════════════════"]
    if sigs:
        for s in sigs:
            lines.append(f"• {s}")
    else:
        lines.append("(none)")
    return _an_pre(lines)


async def analyse_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /analyse [PAIR] [TIMEFRAME]

    Raw market analysis with no regime gate or confidence floor.
    Always produces an indicator snapshot regardless of market condition.

    Examples:
        /analyse ZBT          → ZBTUSDT on 4H (default)
        /analyse ZBT 1h       → ZBTUSDT on 1H
        /analyse ZBT 10m      → ZBTUSDT, 10m remapped to 15m
        /analyse BTCUSDT 4h   → explicit pair
    """
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args or []

    if not args:
        await update.message.reply_text(
            "🔬 RAW CHART ANALYSIS — No Gates, No Filters\n\n"
            "Usage:\n"
            "  /analyse ZBT          — ZBTUSDT on default 4H\n"
            "  /analyse ZBT 1h       — pin to 1H chart\n"
            "  /analyse ZBT 15m      — pin to 15M chart\n"
            "  /analyse ZBT 10m      — non-standard TF (remaps to 15m)\n"
            "  /analyse BTCUSDT 4h   — full USDT pair name also works\n\n"
            "Supported TFs: 15m, 1h, 4h, 1d\n"
            "Non-standard TFs (5m, 10m, 30m, etc.) are remapped to nearest.\n\n"
            "⚠️ Unlike /cscan, /analyse shows raw readings with zero filtering.\n"
            "Use your own judgment on the output."
        )
        return

    # Parse pair
    raw    = args[0].upper().replace('/', '').strip()
    symbol = raw if raw.endswith('USDT') else raw + 'USDT'

    # Parse optional TF
    requested_tf_raw = None
    tf_key = '4h'   # default
    if len(args) > 1:
        requested_tf_raw = args[1].lower().strip()
        tf_key = _ANALYSE_TF_ALIASES.get(requested_tf_raw)
        if tf_key is None:
            await update.message.reply_text(
                f"⚠️ Unknown timeframe: {args[1]}\n\n"
                f"Supported: 15m, 1h, 4h, 1d\n"
                f"Non-standard: 5m, 10m, 30m, 2h, 6h, 12h (remapped to nearest)\n"
                f"Example: /analyse ZBT 10m"
            )
            return

    tf_label = TF_CONFIGS[tf_key]['label']

    # ── HOT-GROUND WARNING — sent before the analysis ─────��────────────��─────
    warning_msg = (
        "⚠️⚠️ HOT GROUND — READ BEFORE PROCEEDING ⚠️⚠️\n\n"
        "You are using /analyse — the unfiltered analysis mode.\n\n"
        "Unlike /cscan or /scan, this command has:\n"
        "  ✗  No BTC regime gate\n"
        "  ✗  No confidence floor\n"
        "  ✗  No counter-trend veto\n"
        "  ✗  No cooldown checks\n"
        "  ✗  No signal validation\n\n"
        "You will receive a raw indicator snapshot.\n"
        "There is NO recommendation to trade. The output tells you\n"
        "what the indicators say — not whether you should act on it.\n\n"
        "🔥 Trading on /analyse output without your own analysis\n"
        "   carries significantly higher risk. Proceed with caution.\n\n"
        "──────────────���───────────────────────\n"
        f"⏳ Fetching {symbol} on {tf_label}..."
    )
    await update.message.reply_text(warning_msg)

    # Run in executor (blocking I/O)
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        SCAN_EXECUTOR,
        lambda: _analyse_raw_indicators(symbol, tf_key)
    )

    if isinstance(result, str):
        # Error message
        await update.message.reply_text(
            f"❌ Could not analyse {symbol} on {tf_label}\n\n"
            f"{result}\n\n"
            f"💡 Tips:\n"
            f"• Verify the pair exists as a MEXC perpetual\n"
            f"• Try: /analyse {raw.replace('USDT','')} 4h\n"
            f"�� Some tokens use 1000{raw.replace('USDT','')} format"
        )
        return

    # Cache the full result so the section buttons can rebuild each view in-place.
    _analyse_cache[f"{chat_id}:{symbol}:{tf_key}"] = result

    card     = _format_analyse_card(result, requested_tf_raw)
    keyboard = _analyse_kb(symbol, tf_key, chat_id,
                           len(result['bull_signals']), len(result['bear_signals']),
                           active='sum')

    await update.message.reply_text(card, reply_markup=keyboard, parse_mode="HTML")


async def analyse_section_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Swap the /analyse card between summary / indicators / bullish / bearish."""
    query = update.callback_query
    await query.answer()
    try:
        _, section, symbol, tf_key = query.data.split('|', 3)
    except ValueError:
        return
    chat_id = query.message.chat_id if query.message else None
    data = _analyse_cache.get(f"{chat_id}:{symbol}:{tf_key}")
    if not data:
        await query.answer("Analysis expired — run /analyse again.", show_alert=True)
        return

    if section == 'ind':
        text = _format_analyse_indicators(data)
    elif section == 'bull':
        text = _format_analyse_signals(data, 'bull')
    elif section == 'bear':
        text = _format_analyse_signals(data, 'bear')
    else:
        section = 'sum'
        text = _format_analyse_card(data, None)

    kb = _analyse_kb(symbol, tf_key, chat_id,
                     len(data['bull_signals']), len(data['bear_signals']),
                     active=section)
    try:
        await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")
    except Exception as e:
        logger.warning("analyse_section edit failed: %s", e)


async def trend_dying_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Background job — runs every 30 minutes.
    Checks three signal pools for dying trends and notifies users:

    1. Legacy single-trade tracking (user_tracking table)
    2. Multi-trade tracking (tracked_trades table via user_tracking in-memory)
    3. Safe Mode users — monitors every signal they received from any scan
       (safemode_last_signals), regardless of whether they used /pick.
    """
    loop = asyncio.get_event_loop()

    async def _check_and_notify(chat_id: int, sig: dict, source_label: str = ""):
        """Fetch data, check dying, send alert if needed. Deduped via cache."""
        symbol   = sig.get('symbol')
        exchange = sig.get('exchange', 'MEXC')
        bias     = sig.get('bias')
        if not symbol or not bias:
            return

        cache_key = f"{chat_id}_{exchange}_{symbol}"
        if _trend_monitor_cache.get(cache_key, {}).get('notified', False):
            return

        try:
            def _fetch():
                if exchange == 'BYBIT' and sakz_exchanges.BYBIT_AVAILABLE:
                    df4h = bybit_fetch_ohlcv(symbol, '240', 50)
                    df1d = bybit_fetch_ohlcv(symbol, 'D',   30)
                elif exchange == 'BINANCE' and sakz_exchanges.BINANCE_AVAILABLE:
                    df4h = binance_fetch_ohlcv(symbol, '4h', 50)
                    df1d = binance_fetch_ohlcv(symbol, '1d', 30)
                else:
                    df4h = mexc_fetch_ohlcv(symbol, '4h', 50)
                    df1d = mexc_fetch_ohlcv(symbol, '1d', 30)
                if df4h is not None: df4h = add_indicators(df4h, timeframe="4h")
                if df1d is not None: df1d = add_indicators(df1d, timeframe="1d")
                return df4h, df1d

            df4h, df1d = await loop.run_in_executor(SCAN_EXECUTOR, _fetch)
            dying, reason = _check_trend_dying(sig, df4h, df1d)

            if dying:
                _trend_monitor_cache[cache_key] = {'notified': True}
                bias_e = "🟢" if bias == 'LONG' else "🔴"
                price  = df4h.iloc[-1]['close'] if df4h is not None and len(df4h) > 0 else 0
                # Different footer depending on source so users know why they got the alert
                if source_label == 'safemode':
                    footer = "���️ Safe Mode alert — trend detected from your recent scan.\nUse /safemode to disable."
                else:
                    footer = "This is an alert, not a close signal.\nUse /stoptrade to stop trade reminders."
                msg = (
                    f"⚠️ TREND ALERT — MONITOR YOUR TRADE\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{bias_e} {exchange} | {symbol} ({bias})\n"
                    f"💰 Current Price: ${price:.6f}\n\n"
                    f"🚨 The trend is showing signs of weakness:\n"
                    f"{reason}\n\n"
                    f"📌 Recommended actions:\n"
                    f"  • Consider moving SL to breakeven\n"
                    f"  • Partial profit-take at current price\n"
                    f"  • Watch next candle close carefully\n\n"
                    f"{footer}"
                )
                await context.bot.send_message(chat_id=chat_id, text=msg)
        except Exception as e:
            logger.warning("trend_dying_job %s %s: %s", exchange, symbol, e)

    # ── Pool 1: Legacy single-trade user_tracking ──────────────────────────────
    try:
        conn = db_connect()
        c    = conn.cursor()
        c.execute("SELECT chat_id, signal_json FROM user_tracking")
        legacy_rows = c.fetchall()
        conn.close()
        for row in legacy_rows:
            try:
                sig = json.loads(row['signal_json'])
                await _check_and_notify(row['chat_id'], sig, source_label='tracked')
            except Exception:
                continue
    except Exception as e:
        logger.warning("trend_dying_job legacy pool error: %s", e)

    # ── Pool 2: Multi-trade tracked_trades (in-memory user_tracking dict) ─────
    for chat_id, trades in list(state.user_tracking.items()):
        for trade_id, tdata in list(trades.items()):
            try:
                sig = tdata.get('signal', {})
                await _check_and_notify(chat_id, sig, source_label='tracked')
            except Exception:
                continue

    # ── Pool 3: Safe Mode users — any signal from any scan ────────────────────
    for chat_id in list(state.safemode_users):
        signals = safemode_last_signals.get(chat_id, [])
        for sig in signals:
            try:
                await _check_and_notify(chat_id, sig, source_label='safemode')
            except Exception:
                continue


# ─── SNAIL 2x TARGET — leverage-aware ���────────────────────────
def _snail_2x_target(r: dict) -> tuple[float, str]:
    """
    Calculate what '2x' means based on leverage.
    With leverage L, a 2x on position = (100% gain / L) price move from entry.
    Returns (target_price, description_str).
    """
    lev_data = r.get('leverage')
    lev      = lev_data['suggested'] if lev_data else 5  # default 5x
    entry    = r.get('price', 0)
    bias     = r.get('bias', 'LONG')
    if entry == 0:
        return r.get('t2', 0), f"T2 (est.)"

    # 2x on account = 100% gain = (100 / lev)% price move
    pct_needed = 100.0 / lev
    if bias == 'LONG':
        target = entry * (1 + pct_needed / 100)
    else:
        target = entry * (1 - pct_needed / 100)

    desc = f"2x position at {lev}x lev = {pct_needed:.1f}% price move"
    return target, desc


def _patch_snail_signal_with_leverage_2x(r: dict) -> dict:
    """Inject leverage-aware 2x target into a signal dict before formatting."""
    t2_lev, t2_desc = _snail_2x_target(r)
    r = dict(r)
    r['t2_leverage'] = t2_lev
    r['t2_lev_desc'] = t2_desc
    return r


# ─── SNAIL ACTIVATION MESSAGE — leverage-aware ────────────────
async def _snail_activate_patched(query, chat_id):
    """Call after db_snail_start_session to send the improved activation message."""
    exp = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d %H:%M')
    await query.edit_message_text(
        f"🐌 SNAIL MODE ACTIVATED!\n\n"
        f"✅ 7-day session started.\n"
        f"Expires: {exp}\n\n"
        f"I'll send ONE signal per day when:\n"
        f"  🔹 Confidence = 10/10\n"
        f"  🔹 Snail Score ≥ 80/100\n"
        f"  🔹 Manipulation risk: LOW\n"
        f"  🔹 TA + FA both verified\n\n"
        f"🎯 Target: 2x your position from entry.\n"
        f"   (Actual price target is calculated\n"
        f"    from the leverage the bot specifies\n"
        f"    in each signal — not a flat 2x price.)\n\n"
        f"🐌 Use /snailvault to monitor your trades."
    )


# ─── /chart COMMAND ─────────────────────────��─────────────────
async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /chart SOL  — generates a TA chart image for the specified pair.
    Uses matplotlib to draw OHLCV candlestick + EMA20/50 + RSI + MACD + BB.
    """
    _track(update)
    chat_id = update.effective_chat.id
    args    = context.args or []

    if not args:
        await update.message.reply_text(
            "📊 CHART\n\n"
            "Usage: /chart SOL\n"
            "       /chart BTCUSDT\n"
            "       /chart ETH t1h\n\n"
            "Generates a TA chart with EMA, RSI, MACD & Bollinger Bands."
        )
        return

    track_user_interaction(chat_id)

    tf = '4h'; sym_arg = None
    for a in args:
        parsed = _parse_tf_arg(a)
        if parsed and tf == '4h':
            tf = parsed
        elif sym_arg is None:
            sym_arg = a

    if not sym_arg:
        await update.message.reply_text("⚠️ Please specify a coin. e.g. /chart SOL t1h")
        return

    raw    = sym_arg.upper().replace('/', '').strip()
    symbol = raw if raw.endswith('USDT') else raw + 'USDT'

    await update.message.reply_text(
        f"📊 Generating {symbol} chart on {_tf_display(tf)}...\n⏳ Please wait..."
    )

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
        import matplotlib.dates as mdates
        import io

        loop = asyncio.get_event_loop()
        def _fetch_df():
            df = mexc_fetch_ohlcv(symbol, tf, 120)
            if df is None and sakz_exchanges.BYBIT_AVAILABLE:
                df = bybit_fetch_ohlcv(symbol, TF_MAP_BYBIT.get(tf,'240'), 120)
            if df is None and sakz_exchanges.BINANCE_AVAILABLE:
                df = binance_fetch_ohlcv(symbol, TF_MAP_BINANCE.get(tf,'4h'), 120)
            return df

        df = await loop.run_in_executor(SCAN_EXECUTOR, _fetch_df)
        if df is None or len(df) < 20:
            await update.message.reply_text(
                f"⚠️ Not enough data to chart {symbol} on {_tf_display(tf)}.\n"
                f"Try a longer timeframe or a different pair."
            )
            return

        df = add_indicators(df, timeframe=tf if tf in ("4h","1d","1h","15m") else "4h")
        if df is None:
            await update.message.reply_text("⚠️ Indicator calculation failed for this pair.")
            return

        # Last 80 candles for clarity
        df = df.tail(80).reset_index(drop=True)

        # ── Score the pair for bias overlay ──────────────────
        # CRITICAL FIX: use a fresh unsliced 4H df for scoring —
        # never pass the display df (already tail'd/reset_index'd).
        loop2 = asyncio.get_event_loop()
        def _get_sig():
            # Fresh 4H fetch — 100 candles, unsliced, for scoring only
            df4h_score = mexc_fetch_ohlcv(symbol, '4h', 100)
            if df4h_score is None and sakz_exchanges.BYBIT_AVAILABLE:
                df4h_score = bybit_fetch_ohlcv(symbol, TF_MAP_BYBIT.get('4h', '240'), 100)
            if df4h_score is None:
                return None
            # Daily fetch for macro context
            df_d = mexc_fetch_ohlcv(symbol, '1d', 60)
            if df_d is None and sakz_exchanges.BYBIT_AVAILABLE:
                df_d = bybit_fetch_ohlcv(symbol, 'D', 60)
            if df_d is None:
                return None
            df4h_ind = add_indicators(df4h_score.copy(), timeframe='4h')
            df1d_ind = add_indicators(df_d.copy(),       timeframe='1d')
            if df4h_ind is None or df1d_ind is None:
                return None
            if len(df4h_ind) < 5 or len(df1d_ind) < 5:
                return None
            # user_requested=True -> never blocked by regime/gap/neutral gates;
            # always returns a best-effort (possibly weak) signal for the overlay.
            return score_pair(df4h_ind, df1d_ind, 0, symbol, user_requested=True)

        sig = await loop2.run_in_executor(SCAN_EXECUTOR, _get_sig)
        # A ScanFailure (e.g. data unavailable) is NOT a usable overlay signal -
        # normalise to None so the chart still renders with a "no strong signal"
        # note rather than crashing on subscript access. Weak signals come back
        # as a dict (user_requested=True) and DO render their overlay.
        if isinstance(sig, ScanFailure):
            sig = None

        # ── Build figure ──────────────────────────────────────
        BG   = '#0d1117'; GRID = '#1c2433'; UP = '#26a641'; DN = '#e3434d'
        TEXT = '#c9d1d9'; BLUE = '#58a6ff'; ORANGE = '#f78166'; PURPLE = '#bc8cff'

        fig = plt.figure(figsize=(14, 10), facecolor=BG)
        gs  = GridSpec(4, 1, figure=fig, height_ratios=[3, 1, 1, 1],
                       hspace=0.04, left=0.06, right=0.97, top=0.93, bottom=0.06)

        ax1 = fig.add_subplot(gs[0])  # Candlestick + EMA + BB
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Volume
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # RSI
        ax4 = fig.add_subplot(gs[3], sharex=ax1)  # MACD

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(BG)
            ax.tick_params(colors=TEXT, labelsize=7)
            ax.yaxis.label.set_color(TEXT)
            ax.xaxis.label.set_color(TEXT)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)
            ax.grid(color=GRID, linewidth=0.4, alpha=0.5)

        x   = range(len(df))
        xts = [i for i in x]

        # Candlesticks
        for i, row in df.iterrows():
            color = UP if row['close'] >= row['open'] else DN
            ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.8)
            ax1.bar(i, abs(row['close'] - row['open']),
                    bottom=min(row['close'], row['open']),
                    color=color, width=0.7, alpha=0.9)

        # BB
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            ax1.plot(x, df['bb_upper'], color=PURPLE, linewidth=0.7, linestyle='--', alpha=0.6, label='BB')
            ax1.plot(x, df['bb_lower'], color=PURPLE, linewidth=0.7, linestyle='--', alpha=0.6)
            ax1.fill_between(x, df['bb_lower'], df['bb_upper'], alpha=0.04, color=PURPLE)

        # EMA
        if 'ema20' in df.columns:
            ax1.plot(x, df['ema20'], color=BLUE,   linewidth=1.0, alpha=0.9, label='EMA20')
        if 'ema50' in df.columns:
            ax1.plot(x, df['ema50'], color=ORANGE, linewidth=1.0, alpha=0.9, label='EMA50')

        # Support/resistance lines
        if 'support' in df.columns:
            ax1.axhline(df['support'].iloc[-1], color='#3fb950', linewidth=0.6,
                       linestyle=':', alpha=0.7, label=f"Sup ${df['support'].iloc[-1]:.4f}")
        if 'resistance' in df.columns:
            ax1.axhline(df['resistance'].iloc[-1], color=DN, linewidth=0.6,
                       linestyle=':', alpha=0.7, label=f"Res ${df['resistance'].iloc[-1]:.4f}")

        # Signal annotations
        if sig:
            bias_color = UP if sig['bias'] == 'LONG' else DN
            bias_arrow = '▲' if sig['bias'] == 'LONG' else '▼'
            ax1.axhline(sig['entry_low'],  color='#ffa500', linewidth=0.8, linestyle='-.', alpha=0.8)
            ax1.axhline(sig['entry_high'], color='#ffa500', linewidth=0.8, linestyle='-.', alpha=0.8)
            ax1.axhline(sig['t1'],  color='#3fb950', linewidth=0.7, linestyle=':', alpha=0.7)
            ax1.axhline(sig['t2'],  color=BLUE,      linewidth=0.7, linestyle=':', alpha=0.7)
            ax1.axhline(sig['stop_loss'], color=DN,  linewidth=0.7, linestyle=':', alpha=0.8)
            conf_str = f"  {bias_arrow} {sig['bias']}  Conf: {sig['confidence']}/10"
            if sig.get('leverage'):
                conf_str += f"  Lev: {sig['leverage']['suggested']}x"
            ax1.text(len(df)-1, sig['price'], conf_str, color=bias_color,
                    fontsize=7.5, fontweight='bold', va='center', ha='right')

        ax1.set_ylabel('Price', color=TEXT, fontsize=8)
        ax1.legend(loc='upper left', fontsize=6, facecolor=BG,
                   edgecolor=GRID, labelcolor=TEXT, framealpha=0.7)

        # Volume
        vol_colors = [UP if df['close'].iloc[i] >= df['open'].iloc[i] else DN for i in range(len(df))]
        ax2.bar(x, df['volume'], color=vol_colors, width=0.7, alpha=0.7)
        if 'volume_ma' in df.columns:
            ax2.plot(x, df['volume_ma'], color=ORANGE, linewidth=0.8, alpha=0.8)
        ax2.set_ylabel('Vol', color=TEXT, fontsize=7)

        # RSI
        if 'rsi' in df.columns:
            ax3.plot(x, df['rsi'], color=PURPLE, linewidth=1.0)
            ax3.axhline(70, color=DN, linewidth=0.6, linestyle='--', alpha=0.6)
            ax3.axhline(30, color=UP, linewidth=0.6, linestyle='--', alpha=0.6)
            ax3.axhline(50, color=GRID, linewidth=0.5, alpha=0.4)
            ax3.fill_between(x, df['rsi'], 50, where=(df['rsi'] > 50),
                            alpha=0.12, color=UP)
            ax3.fill_between(x, df['rsi'], 50, where=(df['rsi'] < 50),
                            alpha=0.12, color=DN)
            ax3.set_ylim(0, 100)
            ax3.set_ylabel('RSI', color=TEXT, fontsize=7)
            rsi_now = df['rsi'].iloc[-1]
            ax3.text(len(df)-1, rsi_now, f' {rsi_now:.1f}', color=PURPLE, fontsize=6.5, va='center')

        # MACD
        if 'macd_diff' in df.columns:
            macd_colors = [UP if v >= 0 else DN for v in df['macd_diff']]
            ax4.bar(x, df['macd_diff'], color=macd_colors, width=0.7, alpha=0.8)
            if 'macd' in df.columns:
                ax4.plot(x, df['macd'],        color=BLUE,   linewidth=0.8)
            if 'macd_signal' in df.columns:
                ax4.plot(x, df['macd_signal'], color=ORANGE, linewidth=0.8, linestyle='--')
            ax4.axhline(0, color=GRID, linewidth=0.5)
            ax4.set_ylabel('MACD', color=TEXT, fontsize=7)

        # X-axis timestamps
        step = max(1, len(df) // 10)
        xt = list(range(0, len(df), step))
        xl = [df['timestamp'].iloc[i].strftime('%m/%d %H:%M') for i in xt]
        ax4.set_xticks(xt); ax4.set_xticklabels(xl, rotation=30, ha='right', fontsize=6)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)

        # Title
        price_now = df['close'].iloc[-1]
        title_parts = [f"📊 {symbol}  |  {_tf_display(tf)}  |  ${price_now:.6f}"]
        if sig:
            title_parts.append(f"  {'▲' if sig['bias']=='LONG' else '▼'} {sig['bias']}  {sig['confidence']}/10")
        fig.suptitle(''.join(title_parts), color=TEXT, fontsize=10, fontweight='bold',
                    y=0.97, x=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                   facecolor=BG, edgecolor='none')
        buf.seek(0)
        plt.close(fig)

        # Build caption — use MarkdownV2 escaping to avoid telegram parse errors
        # caused by $ signs and other special chars in prices
        def _esc(s: str) -> str:
            """Escape special chars for Telegram MarkdownV2."""
            for c in r'\_*[]()~`>#+-=|{}.!':
                s = s.replace(c, f'\\{c}')
            return s

        caption_lines = [f"📊 *{_esc(symbol)}* — {_esc(_tf_display(tf))}\n"]
        if sig:
            bias_e  = "🟢" if sig['bias'] == 'LONG' else "🔴"
            lev     = sig.get('leverage')
            t2_lev, t2_desc = _snail_2x_target(sig)
            e_lo    = _esc(f"{sig['entry_low']:.4f}")
            e_hi    = _esc(f"{sig['entry_high']:.4f}")
            t1_s    = _esc(f"{sig['t1']:.4f}")
            t2_s    = _esc(f"{sig['t2']:.4f}")
            sl_s    = _esc(f"{sig['stop_loss']:.4f}")
            t2l_s   = _esc(f"{t2_lev:.4f}")
            t2d_s   = _esc(t2_desc)
            caption_lines += [
                f"{bias_e} Bias: *{sig['bias']}*  \\|  Confidence: *{sig['confidence']}/10*",
                f"Entry: \\${e_lo} → \\${e_hi}",
                f"T1: \\${t1_s}  T2: \\${t2_s}  SL: \\${sl_s}",
            ]
            if lev:
                caption_lines.append(f"⚡ Leverage: *{lev['suggested']}x*")
                caption_lines.append(f"🎯 2x target: \\${t2l_s} \\({t2d_s}\\)")
            first_reason = sig['reasons'][0] if sig.get('reasons') else 'Multiple indicators aligned'
            caption_lines.append(f"\n📌 {_esc(first_reason)}")
        else:
            pn_s = _esc(f"{price_now:.6f}")
            caption_lines.append(f"Price: \\${pn_s}\n_No strong signal on this TF right now\\._")

        chart_kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔄 Refresh Chart", callback_data=f"chart_tf_refresh|{symbol}|{tf}"),
            InlineKeyboardButton("🔗 Trade", url=get_exchange_link(
                sig.get('exchange', 'BYBIT') if sig else 'BYBIT', symbol
            ))
        ]])

        await update.message.reply_photo(
            photo=buf,
            caption="\n".join(caption_lines),
            parse_mode="MarkdownV2",
            reply_markup=chart_kb
        )

    except ImportError:
        await update.message.reply_text(
            "⚠️ Chart generation requires matplotlib.\n"
            "Install it with: `pip install matplotlib`\n\n"
            "In the meantime, use /cscan to get signal details."
        )
    except Exception as e:
        logger.error("chart_command %s %s: %s", symbol, tf, e)
        await update.message.reply_text(
            f"❌ Chart generation failed: {str(e)}\n\n"
            f"Try /cscan {sym_arg} instead for signal data."
        )


async def chart_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle 🔄 Refresh Chart inline buttons (from signal-based charts)."""
    query = update.callback_query
    await query.answer("Generating chart…")

    data = query.data
    idx  = int(data.split('|')[1])

    if not state.last_scan_results or idx >= len(state.last_scan_results):
        await query.message.reply_text("⚠️ Signal data expired. Run /scan again.")
        return

    signal = state.last_scan_results[idx]
    await query.message.reply_text(
        f"📊 Generating chart for {signal['exchange']} {signal['symbol']}…"
    )

    def _fetch_df4h():
        exchange = signal.get('exchange', 'BYBIT')
        sym      = signal['symbol']
        if exchange == 'BYBIT':
            return bybit_fetch_ohlcv(sym, '240', 100)
        elif exchange == 'BINANCE':
            return binance_fetch_ohlcv(sym, '4h', 100)
        else:
            return mexc_fetch_ohlcv(sym, '4h', 100)

    loop  = asyncio.get_event_loop()
    df4h  = await loop.run_in_executor(SCAN_EXECUTOR, _fetch_df4h)
    if df4h is None:
        await query.message.reply_text("⚠️ Could not fetch candle data.")
        return

    df4h = add_indicators(df4h, timeframe="4h")
    if df4h is None:
        await query.message.reply_text("⚠️ Indicator calculation failed.")
        return

    png = await loop.run_in_executor(
        SCAN_EXECUTOR, lambda: _plt_locked(generate_chart, signal, df4h))
    if png is None:
        await query.message.reply_text("⚠️ Chart generation failed. Try again shortly.")
        return

    caption = (
        f"📊 {signal['exchange']} · {signal['symbol']} — 4H Chart\n"
        f"{'🟢' if signal['bias']=='LONG' else '🔴'} {signal['bias']}  "
        f"{signal['confidence']}/10  |  Hold {signal['hold']}\n"
        f"Entry: ${signal['entry_low']:.4f}–${signal['entry_high']:.4f}  "
        f"SL: ${signal['stop_loss']:.4f}"
    )
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔄 Refresh Chart", callback_data=f"chart_refresh|{idx}"),
        InlineKeyboardButton("🔗 Trade Now",
                             url=get_exchange_link(signal['exchange'], signal['symbol']))
    ]])
    await query.message.reply_photo(
        photo=io.BytesIO(png),
        caption=caption,
        reply_markup=keyboard
    )


async def chart_tf_refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle 🔄 Refresh Chart button from /chart [pair] [tf] output.
    callback_data format: chart_tf_refresh|SYMBOL|TF
    Re-runs the full chart generation for that pair/TF with fresh candle data.
    """
    query = update.callback_query
    await query.answer("Refreshing chart…")

    parts = query.data.split('|')
    if len(parts) < 3:
        await query.answer("⚠️ Could not parse chart refresh data.", show_alert=True)
        return
    symbol, tf = parts[1], parts[2]
    # No intermediate "Refreshing..." message — the toast above is enough, and a
    # new message would defeat in-place refresh.

    # Delegate to the main chart logic by simulating context for chart_command
    # Instead, run the fetch inline to avoid circular context faking
    loop = asyncio.get_event_loop()

    def _fetch():
        df = mexc_fetch_ohlcv(symbol, tf, 120)
        if df is None and sakz_exchanges.BYBIT_AVAILABLE:
            df = bybit_fetch_ohlcv(symbol, TF_MAP_BYBIT.get(tf, '240'), 120)
        if df is None and sakz_exchanges.BINANCE_AVAILABLE:
            df = binance_fetch_ohlcv(symbol, TF_MAP_BINANCE.get(tf, '4h'), 120)
        if df is None or len(df) < 20:
            return None, None
        df_ind = add_indicators(df, timeframe=tf if tf in ("4h","1d","1h","15m") else "4h")
        # scoring df
        df4h_score = mexc_fetch_ohlcv(symbol, '4h', 100)
        if df4h_score is None and sakz_exchanges.BYBIT_AVAILABLE:
            df4h_score = bybit_fetch_ohlcv(symbol, '240', 100)
        df_d = mexc_fetch_ohlcv(symbol, '1d', 60)
        if df_d is None and sakz_exchanges.BYBIT_AVAILABLE:
            df_d = bybit_fetch_ohlcv(symbol, 'D', 60)
        sig = None
        if df4h_score is not None and df_d is not None:
            df4h_ind = add_indicators(df4h_score.copy(), timeframe='4h')
            df1d_ind = add_indicators(df_d.copy(), timeframe='1d')
            if df4h_ind is not None and df1d_ind is not None and len(df4h_ind) >= 5:
                sig = score_pair(df4h_ind, df1d_ind, 0, symbol)
        return df_ind, sig

    df, sig = await loop.run_in_executor(SCAN_EXECUTOR, _fetch)
    if df is None:
        await query.message.reply_text(
            f"⚠️ Not enough data to refresh {symbol} on {_tf_display(tf)}."
        )
        return

    df = df.tail(80).reset_index(drop=True)

    # Rebuild the figure using the same plotting code path
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.dates as mdates

        BG   = '#0d1117'; GRID = '#1c2433'; UP = '#26a641'; DN = '#e3434d'
        TEXT = '#c9d1d9'; BLUE = '#58a6ff'; ORANGE = '#f78166'; PURPLE = '#bc8cff'

        fig = plt.figure(figsize=(14, 10), facecolor=BG)
        gs  = GridSpec(4, 1, figure=fig, height_ratios=[3,1,1,1],
                       hspace=0.04, left=0.06, right=0.97, top=0.93, bottom=0.06)
        ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1); ax4 = fig.add_subplot(gs[3], sharex=ax1)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(BG); ax.tick_params(colors=TEXT, labelsize=7)
            for spine in ax.spines.values(): spine.set_edgecolor(GRID)
            ax.grid(color=GRID, linewidth=0.4, alpha=0.5)

        x = range(len(df))
        for i, row in df.iterrows():
            color = UP if row['close'] >= row['open'] else DN
            ax1.plot([i,i],[row['low'],row['high']], color=color, linewidth=0.8)
            ax1.bar(i, abs(row['close']-row['open']), bottom=min(row['close'],row['open']),
                    color=color, width=0.7, alpha=0.9)
        if 'bb_upper' in df.columns:
            ax1.plot(x, df['bb_upper'], color=PURPLE, linewidth=0.7, linestyle='--', alpha=0.6)
            ax1.plot(x, df['bb_lower'], color=PURPLE, linewidth=0.7, linestyle='--', alpha=0.6)
            ax1.fill_between(x, df['bb_lower'], df['bb_upper'], alpha=0.04, color=PURPLE)
        if 'ema20' in df.columns: ax1.plot(x, df['ema20'], color=BLUE,   lw=1.0, alpha=0.9, label='EMA20')
        if 'ema50' in df.columns: ax1.plot(x, df['ema50'], color=ORANGE, lw=1.0, alpha=0.9, label='EMA50')
        if sig:
            bias_color = UP if sig['bias']=='LONG' else DN
            ax1.axhline(sig['entry_low'],  color='#ffa500', lw=0.8, ls='-.', alpha=0.8)
            ax1.axhline(sig['entry_high'], color='#ffa500', lw=0.8, ls='-.', alpha=0.8)
            ax1.axhline(sig['t1'],  color='#3fb950', lw=0.7, ls=':', alpha=0.7)
            ax1.axhline(sig['stop_loss'], color=DN, lw=0.7, ls=':', alpha=0.8)
            ax1.text(len(df)-1, sig['price'],
                     f"  {'▲' if sig['bias']=='LONG' else '▼'} {sig['bias']}  {sig['confidence']}/10",
                     color=bias_color, fontsize=7.5, fontweight='bold', va='center', ha='right')
        ax1.legend(loc='upper left', fontsize=6, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
        vol_colors = [UP if df['close'].iloc[i]>=df['open'].iloc[i] else DN for i in range(len(df))]
        ax2.bar(x, df['volume'], color=vol_colors, width=0.7, alpha=0.7)
        if 'rsi' in df.columns:
            ax3.plot(x, df['rsi'], color=PURPLE, lw=1.0)
            ax3.axhline(70, color=DN, lw=0.6, ls='--', alpha=0.6)
            ax3.axhline(30, color=UP, lw=0.6, ls='--', alpha=0.6)
            ax3.set_ylim(0,100); ax3.set_ylabel('RSI', color=TEXT, fontsize=7)
        if 'macd_diff' in df.columns:
            macd_c = [UP if v>=0 else DN for v in df['macd_diff']]
            ax4.bar(x, df['macd_diff'], color=macd_c, width=0.7, alpha=0.8)
            ax4.axhline(0, color=GRID, lw=0.5); ax4.set_ylabel('MACD', color=TEXT, fontsize=7)
        step = max(1, len(df)//10)
        xt   = list(range(0, len(df), step))
        xl   = [df['timestamp'].iloc[i].strftime('%m/%d %H:%M') for i in xt]
        ax4.set_xticks(xt); ax4.set_xticklabels(xl, rotation=30, ha='right', fontsize=6)
        for ax in [ax1, ax2, ax3]: plt.setp(ax.get_xticklabels(), visible=False)

        price_now = df['close'].iloc[-1]
        title = f"📊 {symbol}  |  {_tf_display(tf)}  |  ${price_now:.6f}"
        if sig:
            title += f"  {'▲' if sig['bias']=='LONG' else '▼'} {sig['bias']}  {sig['confidence']}/10"
        fig.suptitle(title, color=TEXT, fontsize=10, fontweight='bold', y=0.97, x=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=BG, edgecolor='none')
        buf.seek(0); plt.close(fig)

        caption = f"📊 {symbol} — {_tf_display(tf)}\nRefreshed at {datetime.now().strftime('%H:%M:%S')}"
        if sig:
            bias_e = "🟢" if sig['bias']=='LONG' else "🔴"
            caption += (f"\n{bias_e} {sig['bias']}  {sig['confidence']}/10"
                        f"\nEntry: ${sig['entry_low']:.4f}–${sig['entry_high']:.4f}"
                        f"  SL: ${sig['stop_loss']:.4f}")

        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔄 Refresh Again", callback_data=f"chart_tf_refresh|{symbol}|{tf}"),
            InlineKeyboardButton("🔗 Trade", url=get_exchange_link(
                sig.get('exchange','BYBIT') if sig else 'BYBIT', symbol))
        ]])
        # Refresh in place: if this card is already a photo message, edit the
        # image instead of posting a new one. Only fall back to a new message
        # when we can't edit (e.g. the button lives on a text card).
        if getattr(query.message, 'photo', None):
            from telegram import InputMediaPhoto
            try:
                await query.edit_message_media(
                    media=InputMediaPhoto(media=buf, caption=caption),
                    reply_markup=kb)
            except Exception as e_edit:
                logger.debug("chart refresh edit_media failed, sending new: %s", e_edit)
                buf.seek(0)
                await query.message.reply_photo(photo=buf, caption=caption, reply_markup=kb)
        else:
            await query.message.reply_photo(photo=buf, caption=caption, reply_markup=kb)

    except Exception as e:
        logger.error("chart_tf_refresh_callback %s %s: %s", symbol, tf, e)
        await query.message.reply_text(f"❌ Chart refresh failed: {e}")


# ─── USER TRACKING ────────────────────────────────���───────��───
def track_user_interaction(chat_id: int):
    """Record a unique user interaction in the DB."""
    try:
        conn = db_connect()
        c    = conn.cursor()
        c.execute("""
            INSERT OR IGNORE INTO user_interactions
            (chat_id, first_seen, last_seen, interaction_count)
            VALUES (?, ?, ?, 0)
        """, (chat_id, datetime.now().isoformat(), datetime.now().isoformat()))
        c.execute("""
            UPDATE user_interactions
            SET last_seen=?, interaction_count=interaction_count+1
            WHERE chat_id=?
        """, (datetime.now().isoformat(), chat_id))
        conn.commit(); conn.close()
    except Exception as e:
        logger.warning("track_user_interaction: %s", e)




# ─── ADMIN COMMAND ────────────────────────────────────────────
# Supports two auth methods:
#   1. ADMIN_IDS env variable (original) — silent allow/deny by chat_id
#   2. Password gate (from sakz_bot.py) — password "Sakazuki01" for full dashboard
ADMIN_IDS_ENV = os.environ.get("ADMIN_IDS", "")
ADMIN_IDS = set(int(x.strip()) for x in ADMIN_IDS_ENV.split(',') if x.strip().isdigit())

ADMIN_PASSWORD    = "Sakazuki01"

# ── Admin DB helpers ──────────────────────────


def _track(update: Update):
    """
    One-liner tracking call for use at the top of every command handler.
    Safe — never raises, never blocks. Logs an error if it fails.
    """
    try:
        u = update.effective_user
        if u:
            db_track_user(u.id, u.username, u.first_name)
    except Exception as e:
        logger.error("_track failed: %s", e)


# ���─ Top-level activity middleware (registered in main with group=-1) ──
async def _activity_middleware(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Fires on EVERY Telegram update before any handler.
    Guarantees user_activity is updated regardless of which handler runs.
    Registered with group=-1 in main() so it always runs first.
    """
    try:
        u = update.effective_user
        if u:
            db_track_user(u.id, u.username, u.first_name)
            logger.debug("middleware tracked: %s (@%s)", u.id, u.username)
    except Exception as e:
        logger.error("_activity_middleware failed: %s", e)





_admin_pending_auth = set()

async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Hidden admin command — not listed in /menu.
    If ADMIN_IDS env set: also shows interaction stats from user_interactions table.
    If password-gated: prompts for ADMIN_PASSWORD; shows full user_activity dashboard.
    /admin logout — revokes session.
    """
    chat_id = update.effective_chat.id
    args    = context.args

    # Logout
    if args and args[0].lower() == 'logout':
        db_admin_revoke(chat_id)
        _admin_pending_auth.discard(chat_id)
        await update.message.reply_text("🔒 Admin session ended.")
        return

    # Already authenticated — show dashboard
    if db_admin_is_authed(chat_id) or (ADMIN_IDS and chat_id in ADMIN_IDS):
        await _send_admin_dashboard(update, context)
        return

    # Not authenticated — start password auth flow
    _admin_pending_auth.add(chat_id)
    await update.message.reply_text(
        "🔐 Admin access required.\n\nEnter the admin password:"
    )


async def admin_password_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Intercepts plain-text messages for users in the admin auth flow.
    Checked before the generic pnl_message_handler.
    """
    chat_id = update.effective_chat.id
    if chat_id not in _admin_pending_auth:
        return False   # not in auth flow — let other handlers proceed

    text = update.message.text.strip()
    _admin_pending_auth.discard(chat_id)

    if text == ADMIN_PASSWORD:
        db_admin_set_auth(chat_id)
        try:
            await update.message.delete()
        except Exception:
            pass
        await update.message.reply_text("✅ Authenticated. Welcome, Admin.")
        await _send_admin_dashboard(update, context)
    else:
        await update.message.reply_text("❌ Wrong password. Use /admin to try again.")

    return True   # consumed


async def _send_admin_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Build and send the admin stats dashboard."""
    stats = db_admin_get_stats()

    total      = stats['total']
    active_now = stats['active_now']
    new_today  = stats['new_today']
    active_7d  = stats['active_7d']

    if active_now:
        active_lines = []
        for r in active_now[:15]:
            uname  = f"@{r['username']}" if r['username'] else r['first_name'] or f"id:{r['chat_id']}"
            mins   = int((datetime.now() - datetime.fromisoformat(r['last_seen'])).total_seconds() // 60)
            active_lines.append(f"  • {uname}  ({mins}m ago, {r['command_count']} cmds)")
        if len(active_now) > 15:
            active_lines.append(f"  … and {len(active_now)-15} more")
        active_block = "\n".join(active_lines)
    else:
        active_block = "  None right now"

    if new_today:
        new_lines = []
        for r in new_today[:10]:
            uname = f"@{r['username']}" if r['username'] else r['first_name'] or f"id:{r['chat_id']}"
            new_lines.append(f"  • {uname}")
        if len(new_today) > 10:
            new_lines.append(f"  … and {len(new_today)-10} more")
        new_block = "\n".join(new_lines)
    else:
        new_block = "  None yet today"

    top_users = sorted(stats['all_users'], key=lambda r: r['command_count'], reverse=True)[:10]
    top_lines = []
    for i, r in enumerate(top_users, 1):
        uname = f"@{r['username']}" if r['username'] else r['first_name'] or f"id:{r['chat_id']}"
        top_lines.append(f"  {i}. {uname} — {r['command_count']} commands")
    top_block = "\n".join(top_lines) if top_lines else "  No data yet"

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = (
        f"🛡 ADMIN DASHBOARD\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {now_str}\n\n"
        f"👥 USER OVERVIEW\n"
        f"   Total users ever:        {total}\n"
        f"   🟢 Active now (≤30 min): {len(active_now)}\n"
        f"   📅 Active last 7 days:   {len(active_7d)}\n"
        f"   ��� New today:            {len(new_today)}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 CURRENTLY ACTIVE ({len(active_now)})\n"
        f"{active_block}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🆕 NEW TODAY ({len(new_today)})\n"
        f"{new_block}\n\n"
        f"━━━━━��━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 TOP USERS BY ACTIVITY\n"
        f"{top_block}\n\n"
        f"��━━━��━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 BOT STATUS\n"
        f"   Last scan signals: {len(state.last_scan_results)}\n"
        f"   Tracking active:   {len(state.user_tracking)}\n"
        f"   Auto-scan subs:    {len(auto_scan_subscribers)}\n\n"
        f"/admin logout — end session"
    )
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔄 Refresh", callback_data="admin_refresh"),
        InlineKeyboardButton("🔒 Logout",  callback_data="admin_logout"),
    ]])
    await update.message.reply_text(msg, reply_markup=keyboard)


async def admin_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle admin inline buttons (Refresh / Logout)."""
    query   = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id

    if not db_admin_is_authed(chat_id) and not (ADMIN_IDS and chat_id in ADMIN_IDS):
        await query.answer("Session expired. Use /admin to re-authenticate.", show_alert=True)
        return

    if query.data == "admin_logout":
        db_admin_revoke(chat_id)
        await query.edit_message_text("🔒 Admin session ended.")
        return

    if query.data == "admin_refresh":
        stats      = db_admin_get_stats()
        total      = stats['total']
        active_now = stats['active_now']
        new_today  = stats['new_today']
        active_7d  = stats['active_7d']

        if active_now:
            active_lines = []
            for r in active_now[:15]:
                uname = f"@{r['username']}" if r['username'] else r['first_name'] or f"id:{r['chat_id']}"
                mins  = int((datetime.now() - datetime.fromisoformat(r['last_seen'])).total_seconds() // 60)
                active_lines.append(f"  • {uname}  ({mins}m ago, {r['command_count']} cmds)")
            if len(active_now) > 15:
                active_lines.append(f"  … and {len(active_now)-15} more")
            active_block = "\n".join(active_lines)
        else:
            active_block = "  None right now"

        if new_today:
            new_lines = []
            for r in new_today[:10]:
                uname = f"@{r['username']}" if r['username'] else r['first_name'] or f"id:{r['chat_id']}"
                new_lines.append(f"  • {uname}")
            if len(new_today) > 10:
                new_lines.append(f"  … and {len(new_today)-10} more")
            new_block = "\n".join(new_lines)
        else:
            new_block = "  None yet today"

        top_users = sorted(stats['all_users'], key=lambda r: r['command_count'], reverse=True)[:10]
        top_lines = []
        for i, r in enumerate(top_users, 1):
            uname = f"@{r['username']}" if r['username'] else r['first_name'] or f"id:{r['chat_id']}"
            top_lines.append(f"  {i}. {uname} — {r['command_count']} commands")
        top_block = "\n".join(top_lines) if top_lines else "  No data yet"

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = (
            f"🛡 ADMIN DASHBOARD\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 {now_str}\n\n"
            f"👥 USER OVERVIEW\n"
            f"   Total users ever:        {total}\n"
            f"   🟢 Active now (≤30 min): {len(active_now)}\n"
            f"   📅 Active last 7 days:   {len(active_7d)}\n"
            f"   🆕 New today:            {len(new_today)}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 CURRENTLY ACTIVE ({len(active_now)})\n"
            f"{active_block}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"�� NEW TODAY ({len(new_today)})\n"
            f"{new_block}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏆 TOP USERS BY ACTIVITY\n"
            f"{top_block}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📡 BOT STATUS\n"
            f"   Last scan signals: {len(state.last_scan_results)}\n"
            f"   Tracking active:   {len(state.user_tracking)}\n"
            f"   Auto-scan subs:    {len(auto_scan_subscribers)}\n\n"
            f"/admin logout — end session"
        )
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔄 Refresh", callback_data="admin_refresh"),
            InlineKeyboardButton("🔒 Logout",  callback_data="admin_logout"),
        ]])
        await query.edit_message_text(msg, reply_markup=keyboard)


# ─── /lb ALIAS ────────────────────────────────────────────────
async def lb_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/lb — short alias for /leaderboard"""
    await leaderboard_command(update, context)





# ─────────────────────────────────────────────
# /fgi — Fear & Greed Index
# Uses alternative.me free API (no key needed)
# Gives market sentiment + trading recommendation
# ─────────────────────────────────────────────
def _fetch_fgi():
    """Fetch Fear & Greed Index from alternative.me. Returns dict or None."""
    try:
        r    = http_get("https://api.alternative.me/fng/?limit=7",
                            headers=HEADERS, timeout=10)
        data = r.json()
        return data.get('data', [])
    except Exception as e:
        logger.warning("FGI fetch error: %s", e)
        return None


def _fgi_score_color(value):
    """Return hex colour for a given FGI value."""
    if value <= 25:   return "#F0556B"   # red   — fear
    elif value <= 45: return "#F5D020"   # yellow — neutral/leaning fear
    elif value <= 55: return "#C5CBD3"   # soft   — neutral
    elif value <= 75: return "#F0A93C"   # orange — greed
    else:             return "#F0556B"   # red    — extreme greed


def _fgi_bias_line(value):
    """One-line trading guidance for the card."""
    if value <= 10:   return "Extreme panic — LONG setups historically strong"
    elif value <= 25: return "Fear zone — cautious LONG on key support only"
    elif value <= 45: return "Choppy — stick to 8+/10 confidence signals"
    elif value <= 55: return "No strong bias — let TA lead, use /best"
    elif value <= 75: return "Greed — LONG momentum valid, tighten stops"
    elif value <= 90: return "Extreme greed — SHORT bias, smart money selling"
    else:             return "Peak euphoria — high-probability SHORT zone"


def render_fgi_card(data):
    """Render the upgraded SAKZ FGI card (PNG bytes).

    Amber/gold colour system. Bar chart with today highlighted bright,
    prior days fading. Score badge with amber border. Amber glow accent.
    """
    import math
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Polygon, Arc

    BG       = "#0e0e0f"; PANEL    = "#111215"; PANEL_ED = "#1e2028"
    AMBER    = "#EF9F27"; AMBER_DK = "#BA7517"; AMBER_XDK = "#854F0B"
    AMBER_LT = "#FAC775"
    GREEN    = "#2FD477"; RED      = "#F0556B"
    WHITE    = "#FFFFFF"; SOFT     = "#C5CBD3"; GRAY     = "#8A93A0"
    CHIP_BG  = "#111318"; CHIP_ED  = "#222832"
    GOLD     = "#E7B23C"; GOLD_DK  = "#B8822A"

    ASPECT = 10.24 / 5.36

    def disc(x, y, r, **kw):
        ax.add_patch(Ellipse((x, y), width=2*r/ASPECT, height=2*r, **kw))

    name    = os.environ.get("BOT_NAME", "SAKZ").upper()
    current = data[0]
    value   = int(current['value'])
    classif = current['value_classification']
    ts      = datetime.fromtimestamp(int(current['timestamp']))
    score_col = _fgi_score_color(value)
    bias_line = _fgi_bias_line(value)

    delta     = value - int(data[1]['value']) if len(data) >= 2 else 0
    trend_str = f"+{delta}" if delta > 0 else (str(delta) if delta < 0 else "±0")
    trend_col = GREEN if delta > 0 else (RED if delta < 0 else SOFT)

    history = []
    for d in data[:7]:
        history.append((int(d['value']),
                        d['value_classification'],
                        datetime.fromtimestamp(int(d['timestamp'])).strftime('%b %d')))

    fig = plt.figure(figsize=(10.24, 5.36), dpi=100, facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # ── card panel ───────────────────────────────────────────��────────────
    ax.add_patch(FancyBboxPatch((0.014, 0.035), 0.972, 0.93,
        boxstyle="round,pad=0,rounding_size=0.035",
        linewidth=1.3, edgecolor=PANEL_ED, facecolor=PANEL, zorder=1))

    # ── amber glow accent (top-right) ──────────────────────────────��──────
    _gx = np.linspace(0, 1, 200); _gy = np.linspace(0, 1, 200)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _dist = np.sqrt((_GX - 0.95)**2 + (_GY - 0.95)**2)
    _alpha = np.clip(0.14 - _dist * 0.60, 0, 0.14)
    ax.imshow(_alpha, extent=[0, 1, 0, 1], aspect='auto', origin='lower',
              cmap='YlOrBr', alpha=0.55, zorder=0, interpolation='bilinear')

    # ── wings logo ──────────────────────��────��────────────────────────────
    lx, ly = 0.072, 0.872
    left_wing  = [(lx-0.030,ly+0.000),(lx-0.004,ly+0.026),(lx-0.010,ly+0.008),(lx-0.003,ly+0.016),(lx-0.003,ly-0.010)]
    right_wing = [(lx+0.030,ly+0.000),(lx+0.004,ly+0.026),(lx+0.010,ly+0.008),(lx+0.003,ly+0.016),(lx+0.003,ly-0.010)]
    body       = [(lx-0.005,ly-0.004),(lx+0.005,ly-0.004),(lx,ly-0.030)]
    for poly in (left_wing, right_wing, body):
        ax.add_patch(Polygon(poly, closed=True, facecolor=GOLD, edgecolor=GOLD_DK, linewidth=0.6, zorder=3))
    disc(lx, ly+0.014, 0.006, facecolor=GOLD, edgecolor=GOLD_DK, lw=0.5, zorder=4)
    ax.text(0.122, 0.892, name,                                color=WHITE, fontsize=21,  fontweight='bold', va='center', ha='left', zorder=3)
    ax.text(0.123, 0.836, "T R A D I N G   M A D E   E A S I E R", color=GRAY, fontsize=8.5, fontweight='bold', va='center', ha='left', zorder=3)

    # ── FGI pill ──────────────────────────────────────────────────────────
    tag = "FGI"
    pw  = 0.0135 * len(tag) + 0.052
    px  = 0.96 - pw
    ax.add_patch(FancyBboxPatch((px, 0.850), pw, 0.058,
        boxstyle="round,pad=0,rounding_size=0.016",
        linewidth=1.1, edgecolor=CHIP_ED, facecolor=CHIP_BG, zorder=3))
    ax.text(px+0.022, 0.879, tag, color=SOFT, fontsize=11.5, fontweight='bold', va='center', ha='left', zorder=4)
    ax.add_patch(Rectangle((px+pw-0.016, 0.863), 0.0035, 0.032, facecolor=AMBER_DK, edgecolor='none', zorder=4))

    # ── title ────────────────────────────────────────────���────────────────
    ax.text(0.05, 0.690, "FEAR & GREED", color=WHITE, fontsize=30, fontweight='bold', va='center', ha='left', zorder=3)

    # ── score badge ───────────────────────────────────────────────────────
    # Badge is wide enough to hold score + classif + trend without overlapping bars
    bx, by, bw, bh = 0.05, 0.385, 0.455, 0.195
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
        boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=0, facecolor=AMBER_DK, alpha=0.18, zorder=2))
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
        boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=1.6, edgecolor=AMBER_DK, facecolor='none', zorder=3))
    ax.text(bx+0.022, by+bh-0.035, "SCORE",
            color=AMBER, fontsize=11, fontweight='bold', va='center', ha='left', zorder=4)
    ax.text(bx+0.022, by+bh-0.105, f"{value}/100",
            color=AMBER, fontsize=29, fontweight='bold', va='center', ha='left', zorder=4)
    # classification inside badge (bottom-left) + trend (top-right)
    ax.text(bx+0.022, by+0.040, classif.upper(),
            color=AMBER, fontsize=11, fontweight='bold', va='center', ha='left', zorder=4)
    ax.text(bx+0.260, by+bh-0.035, f"{trend_str} vs yesterday",
            color=trend_col, fontsize=10, fontweight='bold', va='center', ha='left', zorder=4)

    # ── 7-day bar chart ───────────────────────────────────────────────────
    cx0, cx1, cy0, cy1 = 0.52, 0.95, 0.43, 0.78
    n   = len(history)
    gap = (cx1 - cx0) / n
    bar_w = gap * 0.72
    for i, (v, c, dt) in enumerate(history):
        bx_  = cx0 + i * gap
        bh_  = (cy1 - cy0) * (v / 100.0)
        # colour hierarchy: today bright, yesterday mid, rest dim
        if i == 0:
            bar_col = AMBER;    alpha_ = 1.00
        elif i == 1:
            bar_col = AMBER_DK; alpha_ = 0.85
        elif i <= 3:
            bar_col = GRAY;     alpha_ = 0.35
        else:
            bar_col = AMBER_DK; alpha_ = 0.40
        ax.add_patch(FancyBboxPatch((bx_, cy0), bar_w, bh_,
            boxstyle="round,pad=0,rounding_size=0.008",
            linewidth=0, facecolor=bar_col, alpha=alpha_, zorder=3))
        ax.text(bx_ + bar_w/2, cy0 - 0.030, dt,
                color=GRAY, fontsize=7.5, va='center', ha='center', zorder=4)
        ax.text(bx_ + bar_w/2, cy0 + bh_ + 0.025, str(v),
                color=AMBER if i == 0 else SOFT,
                fontsize=7.5, va='center', ha='center',
                fontweight='bold' if i == 0 else 'normal', zorder=4)

    # ── divider ───────────────────────────────────────────────────────────
    ax.plot([0.05, 0.95], [0.315, 0.315], color=PANEL_ED, lw=1.0, zorder=2)

    # ── guidance line ─────────────────────────────────────────────────────
    ax.text(0.05, 0.355, bias_line,
            color=SOFT, fontsize=9.5, va='center', ha='left', zorder=4, style='italic')

    # ── bottom detail chips ───────────────────────────────────────────────
    def chip(x, glyph):
        cw, ch, cy_chip = 0.05, 0.095, 0.135
        ax.add_patch(FancyBboxPatch((x, cy_chip), cw, ch,
            boxstyle="round,pad=0,rounding_size=0.018",
            linewidth=1.1, edgecolor=CHIP_ED, facecolor=CHIP_BG, zorder=3))
        gx, gy = x + cw/2, cy_chip + ch/2
        if glyph == 'gauge':
            disc(gx, gy, 0.022, facecolor='none', edgecolor=AMBER_DK, lw=1.7, zorder=4)
            angle = math.pi * (1.0 - value / 100.0)
            ax.plot([gx, gx + 0.013 * math.cos(angle) / ASPECT],
                    [gy, gy + 0.013 * math.sin(angle)],
                    color=AMBER_DK, lw=1.7, zorder=5, solid_capstyle='round')
            disc(gx, gy, 0.004, facecolor=AMBER_DK, edgecolor='none', zorder=6)
        elif glyph == 'trend':
            col_a = GREEN if delta >= 0 else RED
            ax.annotate('', xy=(gx, gy+0.018 if delta >= 0 else gy-0.018),
                            xytext=(gx, gy-0.018 if delta >= 0 else gy+0.018),
                            arrowprops=dict(arrowstyle='-|>', color=col_a, lw=1.8), zorder=4)
        elif glyph == 'clock':
            disc(gx, gy, 0.022, facecolor='none', edgecolor=AMBER_DK, lw=1.7, zorder=4)
            ax.plot([gx, gx], [gy, gy+0.013], color=AMBER_DK, lw=1.7, zorder=4, solid_capstyle='round')
            ax.plot([gx, gx+0.009/ASPECT], [gy, gy],    color=AMBER_DK, lw=1.7, zorder=4, solid_capstyle='round')

    def detail(x, glyph, label, val_str, val_col=WHITE):
        chip(x, glyph)
        tx = x + 0.066
        ax.text(tx, 0.205, label,   color=GRAY,    fontsize=9.5,  fontweight='bold', va='center', ha='left', zorder=4)
        ax.text(tx, 0.135, val_str, color=val_col, fontsize=14.5, fontweight='bold', va='center', ha='left', zorder=4)

    updated_str = ts.strftime('%b %d  %H:%M')
    detail(0.05, 'gauge', 'INDEX',   f"{value}/100", val_col=AMBER)
    detail(0.38, 'trend', '24H CHG', trend_str,      val_col=trend_col)
    detail(0.71, 'clock', 'UPDATED', updated_str)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=BG, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


async def fgi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /fgi    — current Fear & Greed Index as a branded image card
    """
    _track(update)
    await update.message.reply_text("📊 Fetching Fear & Greed Index...")

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_fgi)

    if not data:
        await update.message.reply_text(
            "⚠️ Could not fetch Fear & Greed Index.\n"
            "alternative.me API may be down. Try again shortly."
        )
        return

    try:
        png = await loop.run_in_executor(None, lambda: _plt_locked(render_fgi_card, data))
    except Exception as e:
        logger.warning("FGI card render error: %s", e)
        await update.message.reply_text("⚠️ Could not render FGI card. Try again shortly.")
        return

    value  = int(data[0]['value'])
    classif = data[0]['value_classification']
    caption = f"😨 Fear & Greed Index — {value}/100 · {classif}"

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔄 Refresh", callback_data="fgi_refresh"),
        InlineKeyboardButton("🔍 Scan Now", callback_data="menu_run|scan"),
    ]])
    await update.message.reply_photo(
        photo=io.BytesIO(png), caption=caption, reply_markup=keyboard
    )


async def fgi_refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Refresh button on FGI card — sends a fresh card as a new photo."""
    query = update.callback_query
    await query.answer("Fetching latest FGI...")

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_fgi)
    if not data:
        await query.answer("⚠️ FGI API unavailable", show_alert=True)
        return

    try:
        png = await loop.run_in_executor(None, lambda: _plt_locked(render_fgi_card, data))
    except Exception as e:
        logger.warning("FGI card render error (refresh): %s", e)
        await query.answer("⚠️ Render failed", show_alert=True)
        return

    value   = int(data[0]['value'])
    classif = data[0]['value_classification']
    ts      = datetime.fromtimestamp(int(data[0]['timestamp'])).strftime('%H:%M:%S')
    caption = f"😨 Fear & Greed Index — {value}/100 · {classif} · refreshed {ts}"

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔄 Refresh again", callback_data="fgi_refresh"),
        InlineKeyboardButton("🔍 Scan Now",      callback_data="menu_run|scan"),
    ]])
    await query.message.reply_photo(
        photo=io.BytesIO(png), caption=caption, reply_markup=keyboard
    )



# ─────────────────���───────────────────────────────────────────────────────────
# CEILING #6 — MID-TIER UNIVERSE SCANNER
# ─────────────────────────────────────────────────────────────────────────────
# Problem:  The standard /scan covers only the top-50 pairs by volume on each
#           exchange.  These are the most heavily traded, most efficiently priced
#           coins — tracked simultaneously by thousands of algos.  Getting edge
#           on BTC, ETH, SOL, BNB at any confidence level is genuinely hard
#           because price discovery is near-instantaneous.
#
# Solution: /scanmid scans coins ranked 51–200 by 24h volume — liquid enough
#           to enter/exit at low slippage on perps, but with far less
#           algorithmic scrutiny.  Pricing inefficiencies persist longer, MACD
#           crossovers and RSI divergences are less frequently arb'd away, and
#           volume spikes carry more signal because fewer HFT bots are watching.
#
#           The same scoring engine, same regime gate, same confidence formula —
#           only the input universe changes.
#
# Usage:
#   /scanmid             — ranks 51–200, all exchanges
#   /scanmid 51 100      — custom rank window
# ─────────────────────────────────────────────────────────────────────────────

async def scanmid_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /scanmid [rank_from] [rank_to]
    Scans the mid-tier universe (default: ranks 51–200 by 24h volume).
    Uses the identical scoring engine as /scan — only the input pool changes.
    """
    _track(update)
    chat_id   = update.effective_chat.id
    args      = context.args or []
    rank_from = 51
    rank_to   = 200

    if len(args) >= 2:
        try:
            rank_from = max(2,   int(args[0]))
            rank_to   = min(500, int(args[1]))
            if rank_from >= rank_to:
                await update.message.reply_text("⚠️ rank_from must be less than rank_to.")
                return
        except ValueError:
            await update.message.reply_text(
                "⚠️ Usage:\n"
                "/scanmid          — ranks 51–200\n"
                "/scanmid 51 100   — custom window"
            )
            return
    elif len(args) == 1:
        try:
            rank_to = min(500, int(args[0]))
        except ValueError:
            pass

    span = rank_to - rank_from + 1
    await update.message.reply_text(
        f"🔍 MID-TIER SCAN — ranks {rank_from}–{rank_to}\n\n"
        f"Scanning ~{span} coins per exchange (Bybit + MEXC + Binance)…\n"
        f"These are the less-watched, higher-inefficiency pairs.\n"
        f"⏳ Please wait 60����90 seconds…"
    )

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            SCAN_EXECUTOR,
            lambda: run_mid_scan(rank_from, rank_to)
        )
    except Exception as e:
        logger.error("scanmid error: %s", e)
        await update.message.reply_text(f"⚠️ Mid scan failed: {e}")
        return

    if not results:
        await update.message.reply_text(
            f"⚠️ No signals found in ranks {rank_from}–{rank_to}.\n\n"
            f"The mid-tier universe may be consolidating, or the regime gate\n"
            f"is filtering heavily. Try /scan for the main universe."
        )
        return

    longs  = sum(1 for r in results if r['bias'] == 'LONG')
    shorts = sum(1 for r in results if r['bias'] == 'SHORT')
    high_conf = sum(1 for r in results if r['confidence'] >= 8)

    # BTC regime for context
    regime = get_btc_regime()
    regime_emoji = {
        'STRONG_BULL': '🟢🟢', 'BULL': '🟢', 'NEUTRAL': '⚪',
        'BEAR': '🔴', 'STRONG_BEAR': '🔴🔴'
    }.get(regime, '⚪')

    await update.message.reply_text(
        f"✅ MID-TIER SCAN COMPLETE — ranks {rank_from}–{rank_to}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 {len(results)} signals  |  🟢 {longs} LONG  🔴 {shorts} SHORT\n"
        f"⭐ High-conf (≥8): {high_conf}  |  BTC: {regime_emoji} {regime}\n\n"
        f"🏷 MID-TIER = ranks {rank_from}–{rank_to} by 24h volume\n"
        f"   Lower liquidity than /scan — size positions accordingly.\n"
        f"   Signals marked [MID] in each card.\n\n"
        f"Showing top {min(15, len(results))}:"
    )

    await send_signal_cards(
        update.message, results,
        title=f"��� MID-TIER SIGNALS (rank {rank_from}–{rank_to})",
        max_show=15, chat_id=chat_id, source="scan"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CEILING #5 — EMPIRICAL PARAMETER CALIBRATION
# ──────────────────────────────────────────────────────────��──────────────────
# Problem:  Every ATR multiplier (T1=1.6×, T2=3.0×, SL=1.0× in LOW regime),
#           every scoring cap (OSC_CAP=3, CROSS_CAP=3…), and every funding
#           threshold (0.0003 / 0.0007) was set by reasoning alone.
#           The parameters are *defensible* but not *proven*. Backtesting on
#           12–24 months of real OHLCV would reveal which values are actually
#           off by 20–40%.
#
# Solution: /calibrate [symbol] [exchange]
#           Fetches 500 candles of historical 4H OHLCV (≈83 days), replays a
#           lightweight version of the scoring engine at each bar (bar 70+,
#           so all indicators are warm), detects LONG/SHORT signals, and then
#           walks forward on subsequent candles to record whether T1 or SL was
#           hit first (close-based, matching the live outcome checker).
#
#           Three ATR-multiplier parameter sets are tested in parallel:
#             TIGHT   — T1 closer (×0.75 of current), SL same  → higher win%, lower R:R
#             CURRENT — live parameters (baseline)
#             WIDE    — T1 further (×1.25 of current), SL same  → lower win%, higher R:R
#
#           This produces an empirical table:
#             Variant | Signals | WR%  | Avg R:R | EV/trade
#           ...so you can see at a glance whether the current T1 multipliers
#           are already at the sweet spot or systematically over/under-shooting.
#
#           Additionally, two confidence-threshold variants are tested:
#             MIN_CONF ≥ 5 vs ≥ 6 vs ≥ 7
#           to show the precision / recall trade-off for each gate level.
#
# Usage:
#   /calibrate              — BTCUSDT on Bybit, 500 candles
#   /calibrate ETH          — ETHUSDT on Bybit
#   /calibrate SOL BINANCE  — SOLUSDT on Binance
# ────────────────────────────────────────────────────────────────────────────��

def _calib_score_fast(df4h_slice, df1d_slice, funding=0.0):
    """
    A stripped-down version of score_pair() for calibration replay.
    Returns (bias, confidence, vol_regime, t1_mult, sl_mult) or None.

    Key simplifications vs the full scorer:
    • BTC regime gate is SKIPPED (we're backtesting the symbol itself, not
      applying an external filter that changes with time).
    • Duration engine is skipped — not relevant to T1/SL outcome.
    • Entry zone freshness is skipped — irrelevant in backsim.
    • All scoring logic, caps, and confidence formulas are identical to live.
    """
    try:
        if len(df4h_slice) < 6 or len(df1d_slice) < 4:
            return None

        L   = df4h_slice.iloc[-1]
        P   = df4h_slice.iloc[-2]
        P2  = df4h_slice.iloc[-3]
        LD  = df1d_slice.iloc[-2]
        PD  = df1d_slice.iloc[-3]

        price      = L['close']
        rsi4       = L['rsi']
        macd_diff  = L['macd_diff']; prev_diff  = P['macd_diff']; prev2_diff = P2['macd_diff']
        bb_upper   = L['bb_upper'];  bb_lower   = L['bb_lower']
        ema20      = L['ema20'];     ema50      = L['ema50']
        support    = float(L['support']); resistance = float(L['resistance'])
        atr        = L['atr']
        vol        = L['volume'];    vol_ma     = L['volume_ma']
        stoch_k    = L['stoch_k'];   stoch_d    = L['stoch_d']
        rsi_d      = LD['rsi']
        macd_d     = LD['macd_diff']; prev_d    = PD['macd_diff']
        ema20_d    = LD['ema20'];    ema50_d    = LD['ema50']

        # ── Scoring buckets (mirrors live caps exactly) ────────────────
        osc_g_l = osc_g_s = 0
        mtf_g_l = mtf_g_s = 0
        cross_g_l = cross_g_s = 0
        pos_g_l = pos_g_s = 0
        vg_l = vg_s = 0
        ig_l = ig_s = 0

        if rsi4 < 30:    osc_g_l+=3
        elif rsi4 < 40:  osc_g_l+=2
        elif rsi4 < 48:  osc_g_l+=1
        elif rsi4 > 70:  osc_g_s+=3
        elif rsi4 > 60:  osc_g_s+=2
        elif rsi4 > 52:  osc_g_s+=1
        rsi4_prev = P['rsi'] if 'rsi' in P.index else rsi4
        if 48 <= rsi4 <= 52:
            if rsi4 > rsi4_prev:   osc_g_l+=1
            elif rsi4 < rsi4_prev: osc_g_s+=1

        if rsi_d < 35:   mtf_g_l+=2
        elif rsi_d < 45: mtf_g_l+=1
        elif rsi_d > 65: mtf_g_s+=2
        elif rsi_d > 55: mtf_g_s+=1

        macd_cross_bull_4h = (macd_diff > 0 and prev_diff <= 0 and abs(macd_diff) > abs(prev_diff))
        macd_cross_bear_4h = (macd_diff < 0 and prev_diff >= 0 and abs(macd_diff) > abs(prev_diff))
        macd_sust_bull_4h  = (macd_diff > 0 and prev_diff > 0 and prev2_diff <= 0 and macd_diff > prev_diff)
        macd_sust_bear_4h  = (macd_diff < 0 and prev_diff < 0 and prev2_diff >= 0 and macd_diff < prev_diff)
        macd_cross_bull_1d = (macd_d > 0 and prev_d <= 0 and abs(macd_d) > abs(prev_d))
        macd_cross_bear_1d = (macd_d < 0 and prev_d >= 0 and abs(macd_d) > abs(prev_d))

        if macd_cross_bull_4h or macd_sust_bull_4h:   cross_g_l+=3
        elif macd_cross_bear_4h or macd_sust_bear_4h: cross_g_s+=3
        elif macd_diff > 0: cross_g_l+=1
        elif macd_diff < 0: cross_g_s+=1

        if macd_cross_bull_1d:   cross_g_l+=3
        elif macd_cross_bear_1d: cross_g_s+=3
        elif macd_d > 0: cross_g_l+=1
        elif macd_d < 0: cross_g_s+=1

        if price > ema20 > ema50:   pos_g_l+=2
        elif price < ema20 < ema50: pos_g_s+=2
        elif price > ema20:         pos_g_l+=1
        elif price < ema20:         pos_g_s+=1
        if price > ema20_d > ema50_d:   pos_g_l+=2
        elif price < ema20_d < ema50_d: pos_g_s+=2

        if price <= bb_lower:   vg_l+=2
        elif price >= bb_upper: vg_s+=2

        bb_bw     = L.get('bb_bw',     None) if hasattr(L, 'get') else (L['bb_bw']     if 'bb_bw'     in L.index else None)
        bb_bw_min = L.get('bb_bw_min', None) if hasattr(L, 'get') else (L['bb_bw_min'] if 'bb_bw_min' in L.index else None)
        if bb_bw is not None and bb_bw_min is not None and bb_bw_min > 0:
            if bb_bw > bb_bw_min * 1.05:
                if price > P['close']: vg_l+=2
                else:                  vg_s+=2

        if stoch_k < 20 and stoch_d < 20:   osc_g_l+=2
        elif stoch_k > 80 and stoch_d > 80: osc_g_s+=2
        if stoch_k > stoch_d and stoch_k < 45:   osc_g_l+=1
        elif stoch_k < stoch_d and stoch_k > 55: osc_g_s+=1

        if vol_ma and vol_ma > 0:
            ratio = vol / vol_ma
            if ratio > 1.8:
                if price > P['close']: ig_l+=2
                else:                  ig_s+=2
            elif ratio > 1.3:
                if price > P['close']: ig_l+=1
                else:                  ig_s+=1

        if atr > 0:
            if (price - support) < atr * 0.5:    vg_l+=2
            if (resistance - price) < atr * 0.5: vg_s+=2

        if funding != 0:
            if funding < -0.0007:   ig_l+=3
            elif funding < -0.0003: ig_l+=2
            elif funding < 0:       ig_l+=1
            elif funding > 0.0007:  ig_s+=3
            elif funding > 0.0003:  ig_s+=2
            elif funding > 0:       ig_s+=1

        try:
            clv_ma = L['clv_ma'] if 'clv_ma' in L.index else None
            if clv_ma is not None and not pd.isna(clv_ma):
                if clv_ma > 0.3:   ig_l+=1
                elif clv_ma < -0.3: ig_s+=1
        except Exception:
            pass

        OSC_CAP = 3; MTF_CAP = 2; CROSS_CAP = 3; POS_CAP = 3; STRUCTURE_CAP = 4
        ls = (min(osc_g_l,OSC_CAP) + min(mtf_g_l,MTF_CAP) +
              min(cross_g_l,CROSS_CAP) + min(pos_g_l,POS_CAP) +
              min(vg_l,STRUCTURE_CAP) + ig_l)
        ss = (min(osc_g_s,OSC_CAP) + min(mtf_g_s,MTF_CAP) +
              min(cross_g_s,CROSS_CAP) + min(pos_g_s,POS_CAP) +
              min(vg_s,STRUCTURE_CAP) + ig_s)

        if ls == ss or (ls < 3 and ss < 3): return None
        if ls > ss:
            bias, winning, losing = 'LONG',  ls, ss
        else:
            bias, winning, losing = 'SHORT', ss, ls

        daily_bearish = ema20_d < ema50_d
        daily_bullish = ema20_d > ema50_d
        counter_trend = (bias=='LONG' and daily_bearish) or (bias=='SHORT' and daily_bullish)
        min_score_req = 6 if counter_trend else 3
        if winning < min_score_req: return None

        ratio_conf    = (winning / (winning + losing)) * 10
        quality_bonus = 0.0
        if (bias=='LONG' and daily_bullish) or (bias=='SHORT' and daily_bearish):
            quality_bonus += 0.5
        vol_ratio_check = (vol / vol_ma) if (vol_ma and vol_ma > 0) else 1.0
        if vol_ratio_check > 1.3 and ((bias=='LONG' and price > P['close']) or
                                       (bias=='SHORT' and price < P['close'])):
            quality_bonus += 0.5
        confidence = min(10, round(ratio_conf + quality_bonus))
        if confidence < 4: return None

        abs_score_floor = {10: 9, 9: 9, 8: 7, 7: 7, 6: 5, 5: 5, 4: 5}.get(confidence, 5)
        if winning < abs_score_floor:
            confidence -= 1
            if confidence < 4: return None

        # ── Vol regime and T1/SL multipliers (exact copy of live parameters) ──
        atr_pct_va = (atr / price) * 100
        if atr_pct_va < 1.0:
            vol_regime = 'RANGING'; t1_mult, sl_mult = 1.2, 0.7
        elif atr_pct_va < 2.0:
            vol_regime = 'LOW';     t1_mult, sl_mult = 1.6, 1.0
        elif atr_pct_va < 3.5:
            vol_regime = 'MEDIUM';  t1_mult, sl_mult = 2.0, 1.3
        elif atr_pct_va < 5.5:
            vol_regime = 'HIGH';    t1_mult, sl_mult = 2.5, 1.6
        else:
            vol_regime = 'EXTREME'; t1_mult, sl_mult = 3.0, 2.0

        return {
            'bias': bias, 'confidence': confidence, 'price': price,
            'atr': atr, 't1_mult': t1_mult, 'sl_mult': sl_mult,
            'vol_regime': vol_regime, 'winning': winning
        }
    except Exception:
        return None


def _calib_run(df4h_full, df1d_full, min_conf=5, t1_scale=1.0, max_bars_forward=18):
    """
    Walk through df4h_full from bar 70 to end. At each bar:
      1. Slice history up to (and including) that bar → replay scoring
      2. If a signal fires at >= min_conf, record the signal
      3. Walk forward up to max_bars_forward bars on CLOSE prices:
           - LONG:  first close >= t1 → WIN, first close <= sl → LOSS
           - SHORT: first close <= t1 → WIN, first close >= sl → LOSS
           - Neither within window → TIMEOUT (neither win nor loss)
    Returns list of trade dicts.

    t1_scale: multiplier applied to t1_mult before computing T1.
              1.0 = current params, 0.75 = tighter, 1.25 = wider.
    """
    trades = []
    n4 = len(df4h_full)
    n1 = len(df1d_full)

    # Pre-compute daily candle index for each 4H bar (use timestamp alignment)
    # For each 4H row, find the corresponding daily row that was closed at that point
    # We use a simple approach: daily bars that are older than the 4H bar's timestamp

    for i in range(70, n4 - max_bars_forward - 1):
        # Slice history up to this bar (inclusive) — no lookahead
        df4h_slice = df4h_full.iloc[:i+1]
        ts_now     = df4h_full.iloc[i]['timestamp']

        # Find daily bars fully closed before ts_now
        daily_mask  = df1d_full['timestamp'] <= ts_now
        df1d_slice  = df1d_full[daily_mask]
        if len(df1d_slice) < 4:
            continue

        result = _calib_score_fast(df4h_slice, df1d_slice)
        if result is None:
            continue
        if result['confidence'] < min_conf:
            continue

        bias    = result['bias']
        price   = result['price']
        atr     = result['atr']

        t1_mult_adj = result['t1_mult'] * t1_scale
        sl_mult     = result['sl_mult']

        if bias == 'LONG':
            t1 = price + atr * t1_mult_adj
            sl = price - atr * sl_mult
        else:
            t1 = price - atr * t1_mult_adj
            sl = price + atr * sl_mult

        rr = (abs(t1 - price) / abs(sl - price)) if abs(sl - price) > 0 else 0.0

        # Walk forward on close prices
        outcome = 'timeout'
        for j in range(1, max_bars_forward + 1):
            if i + j >= n4:
                break
            fwd_close = float(df4h_full.iloc[i + j]['close'])
            if bias == 'LONG':
                if fwd_close >= t1: outcome = 'win';  break
                if fwd_close <= sl: outcome = 'loss'; break
            else:
                if fwd_close <= t1: outcome = 'win';  break
                if fwd_close >= sl: outcome = 'loss'; break

        trades.append({
            'bar_idx':    i,
            'bias':       bias,
            'confidence': result['confidence'],
            'vol_regime': result['vol_regime'],
            'rr':         rr,
            'outcome':    outcome,
        })

    return trades


async def calibrate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /calibrate [symbol] [exchange]

    Empirically validates the bot's ATR multiplier parameters against real
    historical OHLCV data. Tests three T1 variants in parallel and reports
    the win rate, signal count, avg R:R, and EV for each.

    Also tests three minimum-confidence gates (≥5, ≥6, ≥7) to show the
    precision/recall trade-off at each threshold.

    This is the empirical answer to the question: "are our parameters right?"
    """
    _track(update)
    args     = context.args
    symbol   = 'BTCUSDT'
    exchange = 'BYBIT'

    if args:
        raw = args[0].upper().replace('_USDT', 'USDT')
        symbol = raw if raw.endswith('USDT') else raw + 'USDT'
    if len(args) >= 2:
        exchange = args[1].upper()
        if exchange not in ('BYBIT', 'BINANCE', 'MEXC'):
            await update.message.reply_text("⚠️ Exchange must be BYBIT, BINANCE, or MEXC.")
            return

    short_sym = symbol.replace('USDT', '')
    await update.message.reply_text(
        f"🔬 PARAM CALIBRATION — {exchange} | {symbol}\n"
        f"Fetching 500×4H candles + 120×1D candles…\n"
        f"(This takes 5–15 seconds)"
    )

    loop = asyncio.get_event_loop()

    def _fetch_all():
        if exchange == 'BYBIT':
            df4h = bybit_fetch_ohlcv(symbol, '240', 500)
            df1d = bybit_fetch_ohlcv(symbol, 'D',   120)
        elif exchange == 'BINANCE':
            df4h = binance_fetch_ohlcv(symbol, '4h', 500)
            df1d = binance_fetch_ohlcv(symbol, '1d', 120)
        else:
            df4h = mexc_fetch_ohlcv(symbol, '4h', 500)
            df1d = mexc_fetch_ohlcv(symbol, '1d', 120)
        return df4h, df1d

    try:
        df4h_raw, df1d_raw = await loop.run_in_executor(None, _fetch_all)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Failed to fetch candles: {e}")
        return

    if df4h_raw is None or len(df4h_raw) < 150:
        await update.message.reply_text(f"⚠️ Not enough 4H data for {symbol} on {exchange}.")
        return
    if df1d_raw is None or len(df1d_raw) < 30:
        await update.message.reply_text(f"⚠️ Not enough daily data for {symbol} on {exchange}.")
        return

    # Add indicators (same as live)
    df4h = add_indicators(df4h_raw.copy(), timeframe='4h')
    df1d = add_indicators(df1d_raw.copy(), timeframe='1d')
    if df4h is None or df1d is None:
        await update.message.reply_text("⚠️ Indicator calculation failed.")
        return

    # Reset index so iloc works correctly
    df4h = df4h.reset_index(drop=True)
    df1d = df1d.reset_index(drop=True)

    candles_4h = len(df4h)
    candles_1d = len(df1d)
    days_covered = candles_4h * 4 / 24

    await update.message.reply_text(
        f"📊 Data loaded: {candles_4h}×4H ({days_covered:.0f} days) + {candles_1d}×1D\n"
        f"Running parameter sweep… (~3 ATR variants × 3 conf gates)"
    )

    # ── Section 1 — ATR T1 multiplier sweep (at conf≥5, baseline) ────────���────
    # Three variants: TIGHT (×0.75), CURRENT (×1.0), WIDE (×1.25)
    atr_variants = [
        ('TIGHT',   0.75),
        ('CURRENT', 1.00),
        ('WIDE',    1.25),
    ]

    atr_results = {}
    for name, scale in atr_variants:
        trades = await loop.run_in_executor(
            None, _calib_run, df4h, df1d, 5, scale, 18
        )
        wins     = sum(1 for t in trades if t['outcome'] == 'win')
        losses   = sum(1 for t in trades if t['outcome'] == 'loss')
        timeouts = sum(1 for t in trades if t['outcome'] == 'timeout')
        total    = len(trades)
        wr       = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
        avg_rr   = sum(t['rr'] for t in trades) / total if total > 0 else 0.0
        # EV = WR * avg_rr - (1 - WR) — ATR units per trade
        ev       = (wr/100) * avg_rr - (1 - wr/100) if total > 0 else 0.0
        atr_results[name] = {
            'total': total, 'wins': wins, 'losses': losses,
            'timeouts': timeouts, 'wr': wr, 'avg_rr': avg_rr, 'ev': ev
        }

    # ── Section 2 — Confidence gate sweep (current T1 multiplier) ─────────────
    conf_gates = [5, 6, 7]
    conf_results = {}
    for gate in conf_gates:
        trades = await loop.run_in_executor(
            None, _calib_run, df4h, df1d, gate, 1.0, 18
        )
        wins   = sum(1 for t in trades if t['outcome'] == 'win')
        losses = sum(1 for t in trades if t['outcome'] == 'loss')
        total  = len(trades)
        wr     = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
        avg_rr = sum(t['rr'] for t in trades) / total if total > 0 else 0.0
        ev     = (wr/100) * avg_rr - (1 - wr/100) if total > 0 else 0.0
        conf_results[gate] = {'total': total, 'wins': wins, 'losses': losses, 'wr': wr,
                               'avg_rr': avg_rr, 'ev': ev}

    # ── Section 3 — Vol regime breakdown (current params, conf≥5) ��────────────
    cur_trades = await loop.run_in_executor(None, _calib_run, df4h, df1d, 5, 1.0, 18)
    regime_stats = {}
    for vr in ['RANGING', 'LOW', 'MEDIUM', 'HIGH', 'EXTREME']:
        sub    = [t for t in cur_trades if t['vol_regime'] == vr]
        wins   = sum(1 for t in sub if t['outcome'] == 'win')
        losses = sum(1 for t in sub if t['outcome'] == 'loss')
        total  = len(sub)
        wr     = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
        if total > 0:
            regime_stats[vr] = {'total': total, 'wins': wins, 'losses': losses, 'wr': wr}

    # ── Format the report ──────────────────────────────────────────────────────
    cur = atr_results['CURRENT']

    # ATR table
    atr_lines = []
    for name, scale in atr_variants:
        r   = atr_results[name]
        tag = " ◄ CURRENT" if name == 'CURRENT' else ""
        ev_sign = "+" if r['ev'] >= 0 else ""
        atr_lines.append(
            f"  {name:<8} (×{scale:.2f}) | {r['total']:3d} sigs | "
            f"WR {r['wr']:.0f}% | R:R {r['avg_rr']:.2f} | EV {ev_sign}{r['ev']:.3f}{tag}"
        )

    # Conf gate table
    conf_lines = []
    for gate in conf_gates:
        r       = conf_results[gate]
        ev_sign = "+" if r['ev'] >= 0 else ""
        conf_lines.append(
            f"  ≥{gate}/10 | {r['total']:3d} sigs | WR {r['wr']:.0f}% | "
            f"R:R {r['avg_rr']:.2f} | EV {ev_sign}{r['ev']:.3f}"
        )

    # Vol regime table
    regime_lines = []
    for vr, rs in regime_stats.items():
        bar  = "█" * int(rs['wr'] / 10) + "░" * (10 - int(rs['wr'] / 10))
        regime_lines.append(
            f"  {vr:<8}: {bar} {rs['wr']:.0f}%  ({rs['wins']}/{rs['total']})"
        )
    if not regime_lines:
        regime_lines = ["  (No regime data)"]

    # ── Calibration recommendations ───────────────────────────────────────────
    recs = []

    tight_ev = atr_results['TIGHT']['ev']
    wide_ev  = atr_results['WIDE']['ev']
    cur_ev   = atr_results['CURRENT']['ev']

    if tight_ev > cur_ev + 0.05:
        recs.append(
            f"📉 TIGHTER T1 (×0.75) has higher EV ({tight_ev:+.3f} vs {cur_ev:+.3f}) — "
            f"current T1 multipliers may be too ambitious. Consider reducing by ~15–20%."
        )
    elif wide_ev > cur_ev + 0.05:
        recs.append(
            f"��� WIDER T1 (×1.25) has higher EV ({wide_ev:+.3f} vs {cur_ev:+.3f}) — "
            f"current T1 multipliers are leaving money on the table. Consider increasing by ~15–20%."
        )
    else:
        recs.append(
            f"✅ Current T1 multipliers are near-optimal — EV within ±0.05 of all variants. "
            f"No change recommended."
        )

    # Conf gate recommendation
    best_conf_ev    = max(conf_results[g]['ev'] for g in conf_gates)
    best_conf_gate  = max(conf_gates, key=lambda g: conf_results[g]['ev'])
    live_conf_floor = 5   # current minimum in score_pair
    if best_conf_gate > live_conf_floor and best_conf_ev > conf_results[live_conf_floor]['ev'] + 0.05:
        recs.append(
            f"🚧 Raising minimum confidence to ≥{best_conf_gate}/10 improves EV "
            f"({best_conf_ev:+.3f} vs {conf_results[live_conf_floor]['ev']:+.3f}). "
            f"Trades drop from {conf_results[live_conf_floor]['total']} → {conf_results[best_conf_gate]['total']}."
        )
    else:
        recs.append(
            f"✅ Current confidence floor (≥{live_conf_floor}/10) is appropriate for this symbol."
        )

    # Regime-specific warnings
    for vr, rs in regime_stats.items():
        if rs['total'] >= 5 and rs['wr'] < 35:
            recs.append(
                f"⚠️ {vr} regime: only {rs['wr']:.0f}% win rate ({rs['wins']}/{rs['total']}) — "
                f"consider blocking {vr} signals or requiring conf ≥8 in this regime."
            )

    if cur_ev < 0:
        recs.append(
            f"🔴 Negative EV on current params ({cur_ev:+.3f}) for {symbol}. "
            f"Signal quality is insufficient at current thresholds. "
            f"Review regime gate and confidence floor."
        )

    if not recs:
        recs.append("✅ No critical calibration issues detected.")

    rec_block    = "\n".join(f"• {r}" for r in recs)
    atr_block    = "\n".join(atr_lines)
    conf_block   = "\n".join(conf_lines)
    regime_block = "\n".join(regime_lines)

    msg = (
        f"🔬 PARAM CALIBRATION — {exchange} | {symbol}\n"
        f"━━���━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Historical window: {days_covered:.0f} days | {candles_4h} bars\n"
        f"Sim: signals fired at each bar, outcome resolved within 18×4H candles\n\n"

        f"📐 T1 MULTIPLIER VARIANTS (conf≥5)\n"
        f"{atr_block}\n\n"

        f"🎚 CONFIDENCE GATE SWEEP (current T1)\n"
        f"{conf_block}\n\n"

        f"📊 WIN RATE BY VOL REGIME (current params, conf≥5)\n"
        f"{regime_block}\n\n"

        f"🛠 CALIBRATION RECOMMENDATIONS\n"
        f"{rec_block}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━��━━━━━━━\n"
        f"💡 /calibrate ETH — test on ETHUSDT\n"
        f"💡 /calibrate SOL BINANCE — test specific exchange\n"
        f"💡 /backtest — check live signal outcomes from DB"
    )

    # Telegram has a 4096 char limit — truncate safely if needed
    if len(msg) > 4000:
        msg = msg[:3950] + "\n…(truncated — run /calibrate on specific pairs for full output)"

    await update.message.reply_text(msg)


async def manual_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Public user manual. Excludes admin and hidden commands."""
    _track(update)

    scanning = (
        "🔭  S C A N N I N G\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "/scan                  Full market scan (4H, top 50 pairs)\n"
        "/scan BTC              Scan a specific pair\n"
        "/scan BTC 1h           Specific pair on custom timeframe\n"
        "/scan new 24h          New listings scan  (m/h/d/w units)\n"
        "/cscan ZEC             Custom pair + exchange scan\n"
        "/scalp                 Scalp mode — 15M + 1H signals\n"
        "/scalp ETH             Scalp a specific pair\n"
        "/swing                 Swing mode — 4H + 1D signals\n"
        "/swing BTC             Swing a specific pair\n"
        "/scanmid               Mid-market scan (ranks 51–200)\n"
        "/chart SOL 4h          Chart with full technical analysis\n"
    )

    signals = (
        "\n🎯  S I G N A L S\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "/best                  Highest confidence signal right now\n"
        "/top5                  Top 5 current signals\n"
        "/top10                 Top 10 current signals\n"
        "/filter LONG 8         Filter by bias + min confidence\n"
        "/confirm BTC           Re-validate a stored signal live\n"
        "/tg                    Top 10 gainers (24h price history)\n"
        "/tl                    Top 10 losers  (24h price history)\n"
    )

    trade = (
        "\n💼  T R A D E  M A N A G E R\n"
        "━━━━━━━━━━━━━━━━━━━���━━━━━━━━━━\n"
        "/pick                  Track a trade with auto-reminders\n"
        "/stoptrade             Stop tracking your current trade\n"
        "/check BTC LONG 98000 95000\n"
        "                       Validate an open trade vs live data\n"
        "/pnl                   PnL calculator\n"
    )

    analytics = (
        "\n📊  A N A L Y T I C S\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━��\n"
        "/compare               All signals vs current prices\n"
        "/compare BTC           One pair — signal vs current price\n"
        "/stats 168             Win rate stats  (24 / 168 / 720 hrs)\n"
        "/lb 24                 Leaderboard     (24 / 168 hrs)\n"
        "/backtest 720          Backtest over a time window\n"
        "/fgi                   Fear & Greed Index + guidance\n"
        "/calibrate BTC         ATR param calibration (adv.)\n"
    )

    automation = (
        "\n🔔  A L E R T S  &  A U T O M A T I O N\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "/alert BTCUSDT 8       Alert when BTC hits conf ≥ 8\n"
        "/unalert BTCUSDT       Remove an alert\n"
        "/watch BTCUSDT 7       Watchlist — auto-notify on signal\n"
        "/unwatch BTCUSDT       Remove from watchlist\n"
        "/autoscan              Toggle periodic auto-scan\n"
        "/broadcast on|off      Toggle signal auto-posting here\n"
    )

    general = (
        "\n⚙️  G E N E R A L\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "/status                Bot health & uptime\n"
        "/menu                  Interactive command menu\n"
        "/manual                This manual\n"
    )

    await update.message.reply_text(
        "📘  S A K Z B O T  —  C O M M A N D S\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "The full command guide now lives under a single command.\n\n"
        "👉  Type /pro to see ALL commands and how to use them."
    )


# ────────────────��────────────────────────────
# /scalp — Scalp signal scanner (15m + 1h TFs)
# ─────────────────────────────────────────────
async def scalp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /scalp          — Scan top 30 pairs on 15m and 1h for scalp signals.
    /scalp ETH      — Scalp analysis of a specific pair (15m + 1h).
    """
    _track(update)
    chat_id = update.effective_chat.id
    track_user_interaction(chat_id)
    args = context.args or []

    if args:
        # Single pair scalp
        raw    = args[0].upper().replace("/", "").strip()
        symbol = raw if raw.endswith("USDT") else raw + "USDT"
        top_syms  = [symbol]
        scan_desc = f"⚡ Scalp scan: *{symbol}* on 15M + 1H"
    else:
        top_syms  = mexc_get_top_symbols(30)
        if not top_syms:
            top_syms = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
                        "DOGEUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]
        scan_desc = f"⚡ Scalp scan: top {len(top_syms)} pairs on 15M + 1H"

    await update.message.reply_text(
        f"{scan_desc}\n"
        f"📡 Checking MEXC perpetuals\n"
        f"��� Please wait ~60 seconds..."
    )

    loop    = asyncio.get_event_loop()
    results = []
    for tf in ('15m', '1h'):
        batch = await loop.run_in_executor(
            SCAN_EXECUTOR,
            lambda t=tf: [r for sym in top_syms
                          for (res, _) in [_cscan_pair_tf(sym, t)]
                          for r in res]
        )
        results.extend(batch)

    if not results:
        await update.message.reply_text(
            "⚠️ No scalp signals found right now.\n\n"
            "Short TFs need more precise conditions — markets may be\n"
            "consolidating or volatility is too low.\n\n"
            "Try again in 15–30 minutes, or use /scan for 4H signals."
        )
        return

    results.sort(key=lambda x: (x['confidence'], x['score']), reverse=True)
    seen = set(); deduped = []
    for r in results:
        key = f"{r['symbol']}_{r.get('timeframe','')}"
        if key not in seen:
            seen.add(key); deduped.append(r)

    longs  = sum(1 for r in deduped if r['bias'] == 'LONG')
    shorts = sum(1 for r in deduped if r['bias'] == 'SHORT')
    tf_15  = sum(1 for r in deduped if r.get('timeframe') == '15m')
    tf_1h  = sum(1 for r in deduped if r.get('timeframe') == '1h')

    await update.message.reply_text(
        f"✅ SCALP SCAN COMPLETE\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━���━━━━━\n"
        f"📊 {len(deduped)} signals  🟢 {longs}L  🔴 {shorts}S\n"
        f"�� 15M: {tf_15}   1H: {tf_1h}\n\n"
        f"⚠️ Scalp signals carry higher risk — shorter holds,\n"
        f"   tighter SLs, and smaller position sizes recommended.\n\n"
        f"Showing top {min(15, len(deduped))}:"
    )
    await send_signal_cards(
        update.message, deduped,
        title="⚡ SCALP SIGNALS — 15M + 1H", max_show=15, chat_id=chat_id, source="scalp"
    )


# ─────────────────────────────────────────────
# /swing — Swing signal scanner (4h + 1d TFs)
# ─────────────────��───────────────────────────
async def swing_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /swing          — Scan top 30 pairs on 4h and 1d for swing signals.
    /swing BTC      — Swing analysis of a specific pair (4h + 1d).
    """
    _track(update)
    chat_id = update.effective_chat.id
    track_user_interaction(chat_id)
    args = context.args or []

    if args:
        raw    = args[0].upper().replace("/", "").strip()
        symbol = raw if raw.endswith("USDT") else raw + "USDT"
        top_syms  = [symbol]
        scan_desc = f"📈 Swing scan: *{symbol}* on 4H + 1D"
    else:
        top_syms  = mexc_get_top_symbols(30)
        if not top_syms:
            top_syms = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
                        "DOGEUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]
        scan_desc = f"📈 Swing scan: top {len(top_syms)} pairs on 4H + 1D"

    await update.message.reply_text(
        f"{scan_desc}\n"
        f"📡 Checking MEXC perpetuals\n"
        f"⏳ Please wait ~60 seconds..."
    )

    loop    = asyncio.get_event_loop()
    results = []
    for tf in ('4h', '1d'):
        batch = await loop.run_in_executor(
            SCAN_EXECUTOR,
            lambda t=tf: [r for sym in top_syms
                          for (res, _) in [_cscan_pair_tf(sym, t)]
                          for r in res]
        )
        results.extend(batch)

    if not results:
        await update.message.reply_text(
            "⚠️ No swing signals found right now.\n\n"
            "Markets may be ranging. The regime gate may be filtering\n"
            "heavily in current BTC conditions.\n\n"
            "Try /scan for the standard 4H scan, or check back later."
        )
        return

    results.sort(key=lambda x: (x['confidence'], x['score']), reverse=True)
    seen = set(); deduped = []
    for r in results:
        key = f"{r['symbol']}_{r.get('timeframe','')}"
        if key not in seen:
            seen.add(key); deduped.append(r)

    longs  = sum(1 for r in deduped if r['bias'] == 'LONG')
    shorts = sum(1 for r in deduped if r['bias'] == 'SHORT')
    tf_4h  = sum(1 for r in deduped if r.get('timeframe') == '4h')
    tf_1d  = sum(1 for r in deduped if r.get('timeframe') == '1d')

    regime = get_btc_regime()
    regime_emoji = {
        'STRONG_BULL': '🟢🟢', 'BULL': '🟢', 'NEUTRAL': '⚪',
        'BEAR': '🔴', 'STRONG_BEAR': '🔴🔴'
    }.get(regime, '⚪')

    await update.message.reply_text(
        f"✅ SWING SCAN COMPLETE\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 {len(deduped)} signals  🟢 {longs}L  🔴 {shorts}S\n"
        f"⏱ 4H: {tf_4h}   1D: {tf_1d}  |  BTC: {regime_emoji} {regime}\n\n"
        f"📌 Swing trades typically hold 1–7 days.\n"
        f"   Use wider SLs and lower leverage than scalps.\n\n"
        f"Showing top {min(15, len(deduped))}:"
    )
    await send_signal_cards(
        update.message, deduped,
        title="📈 SWING SIGNALS — 4H + 1D", max_show=15, chat_id=chat_id, source="swing"
    )


# ──���──────��───────────────────────────────────
# /check PAIR DIR ENTRY SL — Validate open trade
# Example: /check BTCUSDT LONG 98000 95000
# ──────────────────────────���──────────────────
async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /check PAIR DIR ENTRY SL
    Validates whether an open trade is still valid based on current indicators.
    """
    _track(update)
    args = context.args or []

    if len(args) < 4:
        await update.message.reply_text(
            "🔎 CHECK OPEN TRADE\n\n"
            "Usage: /check PAIR DIRECTION ENTRY STOPLOSS\n\n"
            "Examples:\n"
            "  /check BTCUSDT LONG 98000 95000\n"
            "  /check ETHUSDT SHORT 3200 3350\n\n"
            "The bot will tell you if your setup is still valid,\n"
            "weakening, or should be cut."
        )
        return

    raw       = args[0].upper().replace('/', '').strip()
    symbol    = raw if raw.endswith('USDT') else raw + 'USDT'
    direction = args[1].upper()
    if direction not in ('LONG', 'SHORT'):
        await update.message.reply_text("⚠️ Direction must be LONG or SHORT.\nExample: /check BTCUSDT LONG 98000 95000")
        return

    try:
        entry_price = float(args[2])
        stop_loss   = float(args[3])
    except ValueError:
        await update.message.reply_text("⚠️ Entry and stop loss must be numbers.\nExample: /check BTCUSDT LONG 98000 95000")
        return

    await update.message.reply_text(
        f"🔎 Checking {symbol} {direction} trade...\n⏳ Please wait..."
    )

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(SCAN_EXECUTOR, lambda: _cscan_pair_mtf(symbol, tf_key='4h'))

    # Get live price (WS-first; exchange hint is only used for REST fallback)
    exchange   = ''
    live_price = 0
    try:
        live_price = _get_live_price(symbol, exchange)
    except Exception as e:
        logger.warning("check_command: live price lookup failed for %s: %s", symbol, e)

    if not live_price:
        live_price = entry_price  # fallback

    pct_from_entry = ((live_price - entry_price) / entry_price) * 100
    pct_from_sl    = ((live_price - stop_loss) / stop_loss) * 100

    if direction == 'SHORT':
        pct_from_entry = -pct_from_entry
        pct_from_sl    = -pct_from_sl

    # Check if SL already hit
    if direction == 'LONG' and live_price <= stop_loss:
        await update.message.reply_text(
            f"🚨 STOP LOSS HIT — {symbol} {direction}\n\n"
            f"Current price: ${live_price:,.4f}\n"
            f"Your stop loss: ${stop_loss:,.4f}\n\n"
            f"❌ Price is at or below your stop loss. Exit now."
        )
        return
    elif direction == 'SHORT' and live_price >= stop_loss:
        await update.message.reply_text(
            f"🚨 STOP LOSS HIT — {symbol} {direction}\n\n"
            f"Current price: ${live_price:,.4f}\n"
            f"Your stop loss: ${stop_loss:,.4f}\n\n"
            f"❌ Price is at or above your stop loss. Exit now."
        )
        return

    # Evaluate signal alignment
    verdict = "✅ STILL VALID"
    notes   = []

    valid_results = [r for r in results if not isinstance(r, ScanFailure)] if results else []
    if valid_results:
        best = valid_results[0]
        current_bias = best.get('bias')
        current_conf = best.get('confidence', 0)

        if current_bias != direction:
            verdict = "❌ INVALIDATED"
            notes.append(f"• Signal has FLIPPED to {current_bias} — your bias is now opposed")
        elif current_conf < 5:
            verdict = "⚠️ WEAKENING"
            notes.append(f"• Confidence dropped to {current_conf}/10 — momentum fading")
        elif current_conf >= 7:
            notes.append(f"• Signal still strong at {current_conf}/10 confidence")
        else:
            verdict = "⚠️ WEAKENING"
            notes.append(f"• Confidence at {current_conf}/10 — hold but watch closely")

        rsi = best.get('rsi4', best.get('rsi', 50))
        if direction == 'LONG' and rsi > 72:
            notes.append(f"• RSI overbought ({rsi:.1f}) — consider taking partial profits")
            if verdict == "✅ STILL VALID":
                verdict = "⚠️ WEAKENING"
        elif direction == 'SHORT' and rsi < 28:
            notes.append(f"��� RSI oversold ({rsi:.1f}) — consider taking partial profits")
            if verdict == "✅ STILL VALID":
                verdict = "⚠️ WEAKENING"
    else:
        notes.append("• No fresh signal data available — use price action to decide")

    pnl_emoji  = "🟢" if pct_from_entry > 0 else "🔴"
    sl_pct_str = f"{abs(pct_from_sl):.2f}% from SL"

    msg = (
        f"��� TRADE CHECK — {symbol} {direction}\n"
        f"{'━'*30}\n"
        f"Entry:        ${entry_price:,.4f}\n"
        f"Stop Loss:    ${stop_loss:,.4f}\n"
        f"Live Price:   ${live_price:,.4f}\n"
        f"PnL:          {pnl_emoji} {pct_from_entry:+.2f}%\n"
        f"Distance SL:  {sl_pct_str}\n"
        f"{'━'*30}\n"
        f"Verdict: {verdict}\n"
    )
    if notes:
        msg += "\nDetails:\n" + "\n".join(notes)

    await update.message.reply_text(msg)


# ─────────────────────────────────────────────
# /confirm PAIR — Re-validate a signal on demand
# Example: /confirm BTC
# ─────────────────���────────────────────────���──
async def confirm_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /confirm PAIR — Re-runs full analysis and compares to last stored signal.
    Answers: is this signal still good?
    """
    _track(update)
    args = context.args or []

    if not args:
        await update.message.reply_text(
            "✅ CONFIRM SIGNAL\n\n"
            "Usage: /confirm PAIR\n\n"
            "Examples:\n"
            "  /confirm BTC\n"
            "  /confirm SOLUSDT\n\n"
            "Re-runs analysis and compares to the last stored signal\n"
            "to check if it is still valid."
        )
        return

    raw    = args[0].upper().replace('/', '').strip()
    symbol = raw if raw.endswith('USDT') else raw + 'USDT'

    await update.message.reply_text(
        f"🔄 Re-validating {symbol}...\n⏳ Please wait ~30 seconds..."
    )

    loop = asyncio.get_event_loop()
    fresh_results = await loop.run_in_executor(SCAN_EXECUTOR, lambda: _cscan_pair_mtf(symbol))

    # Filter out ScanFailure objects
    valid_fresh = [r for r in (fresh_results or []) if not isinstance(r, ScanFailure)]

    if not valid_fresh:
        await update.message.reply_text(
            f"🔍 No signal found for {symbol} right now.\n\n"
            f"Market may be neutral or consolidating.\n"
            f"Try /cscan {symbol.replace('USDT','')} for a full analysis."
        )
        return

    fresh      = valid_fresh[0]
    fresh_bias = fresh.get('bias')
    fresh_conf = fresh.get('confidence', 0)

    # Look up last stored signal for this symbol from DB
    prev_bias = None
    prev_conf = None
    try:
        conn = db_connect()
        c    = conn.cursor()
        c.execute("""
            SELECT bias, confidence FROM signal_outcomes
            WHERE symbol = ? ORDER BY scan_time DESC LIMIT 1
        """, (symbol,))
        row = c.fetchone()
        conn.close()
        if row:
            prev_bias = row['bias']
            prev_conf = row['confidence']
    except Exception as e:
        logger.warning("signal bias lookup failed: %s", e)

    # Compare fresh vs previous
    if prev_bias is None:
        status  = "🔍 NO PREVIOUS SIGNAL"
        summary = "No previous signal found in database — showing fresh analysis only."
    elif fresh_bias != prev_bias:
        status  = "🔄 SIGNAL FLIPPED"
        summary = f"Bias changed from {prev_bias} → {fresh_bias}. Original signal is no longer valid."
    elif fresh_conf < prev_conf - 2:
        status  = "⚠️ SIGNAL WEAKENED"
        summary = f"Confidence dropped from {prev_conf}/10 → {fresh_conf}/10. Monitor closely."
    elif fresh_conf >= prev_conf:
        status  = "✅ SIGNAL STILL VALID"
        summary = f"Confidence held at {fresh_conf}/10. Setup remains intact."
    else:
        status  = "⚠️ SIGNAL WEAKENING"
        summary = f"Confidence slightly lower: {prev_conf}/10 → {fresh_conf}/10. Watch for further weakening."

    bias_emoji = "🟢" if fresh_bias == 'LONG' else "🔴"
    conf_bar   = "█" * fresh_conf + "░" * (10 - fresh_conf)

    msg = (
        f"{status} — {symbol}\n"
        f"{'━'*30}\n"
        f"{bias_emoji} Bias:        {fresh_bias}\n"
        f"⭐ Conviction: {conf_bar} {fresh_conf}/10\n"
        f"📊 Exchange:   {fresh.get('exchange','—')}\n"
    )
    if prev_bias:
        msg += f"📋 Previous:   {prev_bias} @ {prev_conf}/10\n"
    msg += f"{'━'*30}\n{summary}"

    # Status header first
    await update.message.reply_text(msg)
    # Then unified signal panel for the fresh signal
    base_kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔗 Trade Now",
            url=get_exchange_link(fresh.get("exchange",""), fresh["symbol"]))
    ]])
    _, keyboard = cache_signal_card(fresh, 1, base_kb)
    await update.message.reply_text(
        format_signal_primary(fresh, 1), reply_markup=keyboard
    )


# ───────────────────────────────────���─────────
# /optimize — Hyperopt parameter search (admin)
# ─────────────────────────────────────────────
async def optimize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run Hyperopt parameter search and persist best_params.json (admin only)."""
    _track(update)
    chat_id = update.effective_chat.id
    if not db_admin_is_authed(chat_id) and not (ADMIN_IDS and chat_id in ADMIN_IDS):
        await update.message.reply_text("⛔ Admin only. Use /admin first.")
        return

    if not _OPTIMIZE_AVAILABLE:
        await update.message.reply_text(
            "❌ optimize.py not found. Make sure it's in the same folder as sakz_bot_main.py."
        )
        return

    await update.message.reply_text("🧠 Starting Hyperopt (this may take a few minutes)...")
    loop = asyncio.get_running_loop()
    try:
        best = await loop.run_in_executor(
            None,
            lambda: run_hyperopt(
                db_path=DB_PATH,
                out_path=BEST_PARAMS_PATH,
                headers=HEADERS,
                max_evals=40,
                top_n=20,
            ),
        )
        _load_best_params()
        await update.message.reply_text(
            "✅ Optimization complete.\n"
            f"Min confidence: {best.get('min_confidence')}\n"
            f"ATR stop mult:  {best.get('atr_stop_mult'):.3f}\n"
            f"ATR target mult:{best.get('atr_target_mult'):.3f}\n"
            f"Saved: {BEST_PARAMS_PATH}"
        )
    except Exception as e:
        logger.exception("/optimize failed: %s", e)
        await update.message.reply_text(f"❌ Optimization failed: {e}")


# ──────────────────────────────��──────────────
# /xgtrain — Retrain XGBoost signal classifier
# Admin only. Uses outcomes already in sakz_data.db.
# ─────────────────────────────────────────────
async def xgtrain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Retrain the XGBoost win-probability model from historical outcomes."""
    _track(update)
    chat_id = update.effective_chat.id
    if not db_admin_is_authed(chat_id) and not (ADMIN_IDS and chat_id in ADMIN_IDS):
        await update.message.reply_text("⛔ Admin only. Use /admin first.")
        return

    if not _XGB_AVAILABLE:
        await update.message.reply_text(
            "❌ xgboost_train.py not found. Make sure it's in the same folder as sakz_bot_main.py."
        )
        return

    await update.message.reply_text(
        "🤖 Starting XGBoost training...\n"
        "This reads your signal outcomes from the DB and trains a win-probability model.\n"
        "May take 30–60 seconds."
    )
    loop = asyncio.get_running_loop()
    try:
        meta = await loop.run_in_executor(None, lambda: xgb_train(db_path=DB_PATH))
        top_feats = "\n".join(
            f"   {k}: {v:.3f}" for k, v in list(meta.get("top_features", {}).items())[:5]
        )
        await update.message.reply_text(
            f"✅ XGBoost training complete!\n\n"
            f"📊 Samples:   {meta['n_samples']} ({meta['n_wins']} wins / {meta['n_losses']} losses)\n"
            f"📈 Win rate:  {meta['win_rate']:.1%}\n"
            f"🎯 ROC-AUC:  {meta['cv_roc_auc_mean']:.3f} ± {meta['cv_roc_auc_std']:.3f}\n\n"
            f"🔝 Top features:\n{top_feats}\n\n"
            f"Model saved → xgb_signal_model.pkl\n"
            f"Signals will now show ML win probability."
        )
    except RuntimeError as e:
        await update.message.reply_text(
            f"⚠️ Not enough data yet: {e}\n\n"
            "Keep running the bot to collect more signal outcomes, then try again."
        )
    except Exception as e:
        logger.exception("/xgtrain failed: %s", e)
        await update.message.reply_text(f"❌ Training failed: {e}")


# ─────────────────────────────────────────────
# /rftrain — Retrain Random Forest signal classifier
# Admin only. Mirrors /xgtrain.
# ────────────��───────────────────────────���────
async def rftrain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Retrain the Random Forest win-probability model from historical outcomes."""
    _track(update)
    chat_id = update.effective_chat.id
    if not db_admin_is_authed(chat_id) and not (ADMIN_IDS and chat_id in ADMIN_IDS):
        await update.message.reply_text("⛔ Admin only. Use /admin first.")
        return

    if not _RF_AVAILABLE:
        await update.message.reply_text(
            "❌ rf_train.py not found. Make sure it's in the same folder as sakz_bot_main.py."
        )
        return

    await update.message.reply_text(
        "��� Starting Random Forest training...\n"
        "Reads signal outcomes from DB and trains a win-probability model.\n"
        "Usually completes in 10–30 seconds."
    )
    loop = asyncio.get_running_loop()
    try:
        meta = await loop.run_in_executor(None, lambda: rf_train_model(db_path=DB_PATH))
        top_feats = "\n".join(
            f"   {k}: {v:.3f}" for k, v in list(meta.get("top_features", {}).items())[:5]
        )
        await update.message.reply_text(
            f"✅ Random Forest training complete!\n\n"
            f"📊 Samples:  {meta['n_samples']} ({meta['n_wins']} wins / {meta['n_losses']} losses)\n"
            f"📈 Win rate: {meta['win_rate']:.1%}\n"
            f"🎯 ROC-AUC: {meta['cv_roc_auc_mean']:.3f} ± {meta['cv_roc_auc_std']:.3f}\n\n"
            f"🔝 Top features:\n{top_feats}\n\n"
            f"Model saved → rf_signal_model.pkl\n"
            f"Signals now show RF + XGBoost consensus score."
        )
    except RuntimeError as e:
        await update.message.reply_text(
            f"⚠️ Not enough data yet: {e}\n\n"
            "Keep running the bot to collect more signal outcomes, then try again."
        )
    except Exception as e:
        logger.exception("/rftrain failed: %s", e)
        await update.message.reply_text(f"❌ RF training failed: {e}")


# ──────────────────────────���──────────────────
# /btfull — Historical Walk-Forward Backtest
# Requires sakz_backtest_hist.py in the same directory.
# ─────────────────────────────────────────────
async def btfull_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /btfull SYMBOL [tf] [bars]
    Example: /btfull POWER 4h 500
    Runs a full historical walk-forward backtest via sakz_backtest_hist.
    """
    _track(update)
    if not _HIST_BT_AVAILABLE:
        await update.message.reply_text(
            "❌ sakz_backtest_hist.py not found. Place it in the same folder as sakz_bot.py."
        )
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: /btfull SYMBOL [tf] [bars]\n"
            "Example: /btfull POWER 4h 500"
        )
        return

    raw    = args[0].upper()
    symbol = raw if raw.endswith('USDT') else raw + 'USDT'
    tf     = args[1].lower() if len(args) > 1 else '4h'
    bars   = int(args[2]) if len(args) > 2 and args[2].isdigit() else 500

    await update.message.reply_text(
        f"🔬 Running historical backtest\n"
        f"📊 {symbol} · {tf.upper()} · {bars} bars\n"
        f"⏳ This takes ~60s…"
    )
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            SCAN_EXECUTOR,
            lambda: run_hist_backtest(symbol, tf_key=tf, total_bars=bars)
        )
        await update.message.reply_text(
            format_hist_result(result),
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.exception("/btfull failed: %s", e)
        await update.message.reply_text(f"❌ Backtest failed: {e}")


# ─────────────────────────────────────────────
# /paper — Auto Paper Trading Dashboard
# Requires sakz_paper.py in the same directory.
# ─────────────────────────────────────────────
async def paper_job(context: ContextTypes.DEFAULT_TYPE):
    """Background job: mark paper positions to market every 15 min."""
    if not _PAPER_AVAILABLE:
        return

    def _price(exchange, symbol):
        try:
            return _get_live_price(symbol, exchange) or None
        except Exception:
            return None

    try:
        closed  = paper_mark_all(db_connect, _price)
        expired = paper_close_expired(db_connect, _price)
        for pos in closed + expired:
            logger.info(
                "Paper closed: %s %s → %s  PnL=%.3f%%",
                pos.get('bias'), pos.get('symbol'),
                pos.get('outcome'), pos.get('pnl_pct', 0)
            )
    except Exception as e:
        logger.warning("paper_job error: %s", e)


async def paper_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /paper          — show open positions + 7-day summary
    /paper history  — last 20 closed positions
    """
    _track(update)
    if not _PAPER_AVAILABLE:
        await update.message.reply_text(
            "❌ sakz_paper.py not found. Place it in the same folder as sakz_bot.py."
        )
        return

    args = context.args or []

    if args and args[0].lower() == 'history':
        try:
            from sakz_paper import paper_get_closed
            rows = paper_get_closed(db_connect, limit=20)
        except ImportError:
            rows = []
        if not rows:
            await update.message.reply_text("📋 No closed paper positions yet.")
            return
        lines = ["📋 *Paper Trading ��� Last 20 Closed*", ""]
        for r in rows:
            oc_e = '✅' if r.get('outcome') not in ('SL', 'EXPIRED') else '❌'
            lines.append(
                f"{oc_e} {r.get('symbol')} {r.get('bias')} → {r.get('outcome')}  "
                f"{(r.get('pnl_pct') or 0):+.2f}%  (conf {r.get('confidence')})"
            )
        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
        return

    # Default: open positions + summary
    try:
        open_pos = paper_get_open(db_connect)
        summary  = paper_summary(db_connect)
        lines    = []
        if open_pos:
            lines.append(format_paper_open(open_pos))
        lines.append(format_paper_summary(summary))
        await update.message.reply_text("\n\n".join(lines), parse_mode='Markdown')
    except Exception as e:
        logger.exception("/paper failed: %s", e)
        await update.message.reply_text(f"❌ Paper trading error: {e}")


# ─────────────────────────────────────────────
# /dbcheck — DB Diagnostic (admin)
# ─────────────────────────────────────────────
async def dbcheck_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin-only: inspect DB tables and user_activity rows.
    Helps diagnose why the admin dashboard shows 0 users.
    Usage: /dbcheck
    """
    _track(update)
    chat_id = update.effective_chat.id
    if not db_admin_is_authed(chat_id) and not (ADMIN_IDS and chat_id in ADMIN_IDS):
        await update.message.reply_text("⛔ Admin only. Use /admin first.")
        return

    lines = ["🔍 *DB Diagnostic Report*\n"]

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # 1. List all tables
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        lines.append(f"📋 *Tables ({len(tables)}):*")
        lines.append("  " + ", ".join(tables) if tables else "  None found")
        lines.append("")

        # 2. user_activity row count + sample
        if "user_activity" in tables:
            count = conn.execute("SELECT COUNT(*) FROM user_activity").fetchone()[0]
            lines.append(f"👥 *user_activity rows:* {count}")
            if count > 0:
                rows = conn.execute(
                    "SELECT chat_id, username, first_name, command_count, last_seen "
                    "FROM user_activity ORDER BY last_seen DESC LIMIT 5"
                ).fetchall()
                lines.append("  Recent users:")
                for r in rows:
                    uname = r["username"] or r["first_name"] or str(r["chat_id"])
                    lines.append(f"  • {uname} — {r['command_count']} cmds — last: {r['last_seen']}")
            else:
                lines.append("  ⚠️ Table exists but is empty — tracking not writing")
        else:
            lines.append("❌ *user_activity table MISSING* — db_init_user_tracking() may not have run")

        lines.append("")

        # 3. signal_outcomes count
        if "signal_outcomes" in tables:
            so_count = conn.execute("SELECT COUNT(*) FROM signal_outcomes").fetchone()[0]
            lines.append(f"📈 *signal_outcomes rows:* {so_count}")
        else:
            lines.append("❌ *signal_outcomes table MISSING*")

        lines.append("")

        # 4. Check if YOUR chat_id is in user_activity
        mine = conn.execute(
            "SELECT * FROM user_activity WHERE chat_id=?", (chat_id,)
        ).fetchone()
        if mine:
            lines.append(f"✅ *Your chat_id ({chat_id}) IS tracked*")
            lines.append(f"  Commands logged: {mine['command_count']}")
        else:
            lines.append(f"⚠️ *Your chat_id ({chat_id}) NOT in user_activity*")
            lines.append("  Middleware may not be firing — check _activity_middleware registration")

        conn.close()

    except Exception as e:
        lines.append(f"❌ DB error: {e}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Catches unrecognised commands.
    CRITICAL: /scan1234JP$$ is silently intercepted here as the snail unlock easter egg.
    It must never appear in menus, help text, or any public response.
    """
    text = update.message.text.strip() if update.message and update.message.text else ""

    # ── Secret easter egg: unlock snail mode ─────────────────────────────────
    if text == f"/{SNAIL_SECRET_CMD}":
        await secret_unlock_command(update, context)
        return

    # ── All other unknown commands → generic nudge ───────────────────────────
    await update.message.reply_text(
        "❓ Unknown command.\n\nType /pro to see all available commands."
    )


# ─────────────���────────────────────��──────────
# MAIN
# ─────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────���─────────
# AUTO-REFRESH ENGINE  (FIX #AUTOREFRESH)
# ────────────────────────���─────────────���──────��────────────────────────────────────
# Every card that carries a 🔄 Refresh button also refreshes itself on a fixed
# cadence (default 30s) — WITHOUT removing the manual button (users can still
# tap it whenever they want).
#
# How it works (no duplicated render logic):
#   We reuse the EXACT callback the manual 🔄 button fires. A tiny headless
#   CallbackQuery/Update shim lets us invoke that callback on a timer with no
#   real button press. The shim:
#     • routes edit_message_text/caption/reply_markup to bot.edit_* on the
#       stored (chat_id, message_id), so the card refreshes IN PLACE;
#     ���� makes query.answer(...) and message.reply_*(...) no-ops, so a scheduled
#       refresh never pops a toast or posts a brand-new message (no spam);
#     • cancels its own timer if the card was deleted or the bot was blocked.
#
# Scheduling — one central hook:
#   We wrap bot.send_message once at startup. Whenever an outgoing TEXT card
#   carries a known refresh button, we arm a repeating job for that message.
#   Image cards (charts, PnL image) go out via send_photo and post a NEW photo
#   on each refresh, so they are intentionally NOT auto-fired (that would spam
#   the chat and is compute-heavy). Their manual button still works.
#
# Env knobs:
#   AUTO_REFRESH_SECS        interval seconds                (default 30)
#   AUTO_REFRESH_MAX_CYCLES  auto-stop after N ticks, 0=never (default 60 = 30m)
#   AUTO_REFRESH_MAX_JOBS    global cap on concurrent timers  (default 400)
# ──────────────────────────────────────────────────────────────────────────────────
AUTO_REFRESH_SECS       = max(5, int(os.getenv("AUTO_REFRESH_SECS", "30")))
AUTO_REFRESH_MAX_CYCLES = int(os.getenv("AUTO_REFRESH_MAX_CYCLES", "60"))
AUTO_REFRESH_MAX_JOBS   = int(os.getenv("AUTO_REFRESH_MAX_JOBS", "400"))

# Live timers (job names) and cards paused because a sub-view is open.
_autorefresh_jobs   = set()
_autorefresh_paused = set()


def _autorefresh_dispatch():
    """callback_data prefix -> the existing handler the manual button fires.
    Only IN-PLACE text cards are listed; image cards are excluded on purpose."""
    return {
        "sig_refresh":            signal_refresh_callback,
        "cscan_refresh":          cscan_refresh_callback,
        "feed_refresh":           feed_refresh_callback,
        "cmp_refresh":            compare_refresh_callback,
        "cmp_full_refresh":       compare_full_refresh_callback,
        "custom_compare_refresh": custom_compare_refresh_callback,
        "fgi_refresh":            fgi_refresh_callback,
    }


def _autorefresh_handler_for(callback_data):
    """Return the handler for a refresh callback_data, or None."""
    if not callback_data:
        return None
    prefix = callback_data.split("|", 1)[0]
    return _autorefresh_dispatch().get(prefix)


def _autorefresh_pause(query):
    """Pause auto-refresh for this card (e.g. user opened the Details sub-view)."""
    try:
        m = query.message
        _autorefresh_paused.add((m.chat_id, m.message_id))
    except Exception:
        pass


def _autorefresh_resume(query):
    """Resume auto-refresh for this card (user returned to the primary view)."""
    try:
        m = query.message
        _autorefresh_paused.discard((m.chat_id, m.message_id))
    except Exception:
        pass


class _HeadlessMessage:
    """Stand-in for query.message during a scheduled refresh. All sends are
    no-ops so a timer can never post a new message or a toast."""
    def __init__(self, chat_id, message_id):
        self.chat_id    = chat_id
        self.message_id = message_id
        self.chat       = type("_AutoChat", (), {"id": chat_id})()

    async def reply_text(self, *a, **k):          return None
    async def reply_photo(self, *a, **k):         return None
    async def reply_markup(self, *a, **k):        return None
    async def edit_reply_markup(self, *a, **k):   return None


class _HeadlessQuery:
    """Stand-in for update.callback_query during a scheduled refresh."""
    def __init__(self, bot, chat_id, message_id, data):
        self._bot       = bot
        self.data       = data
        self.chat_id    = chat_id
        self.message_id = message_id
        self.message    = _HeadlessMessage(chat_id, message_id)
        self.from_user  = None
        self.stop       = False   # set True when the card is gone / bot blocked

    async def answer(self, *a, **k):
        # Scheduled refresh: never surface a toast/alert to the user.
        return True

    async def _edit(self, **kwargs):
        from telegram.error import BadRequest, Forbidden, TelegramError
        try:
            return await self._bot.edit_message_text(
                chat_id=self.chat_id, message_id=self.message_id, **kwargs
            )
        except BadRequest as e:
            msg = str(e).lower()
            if "not modified" in msg:
                return None                       # nothing changed — fine
            if ("not found" in msg or "can't be edited" in msg
                    or "message to edit" in msg or "chat not found" in msg):
                self.stop = True                  # card gone — stop the timer
                return None
            logger.debug("autorefresh edit BadRequest: %s", e)
            return None
        except Forbidden:
            self.stop = True                      # user blocked the bot
            return None
        except TelegramError as e:
            logger.debug("autorefresh edit error: %s", e)
            return None

    async def edit_message_text(self, text=None, reply_markup=None, **k):
        return await self._edit(text=text, reply_markup=reply_markup, **k)

    async def edit_message_caption(self, caption=None, reply_markup=None, **k):
        from telegram.error import TelegramError
        try:
            return await self._bot.edit_message_caption(
                chat_id=self.chat_id, message_id=self.message_id,
                caption=caption, reply_markup=reply_markup, **k)
        except TelegramError:
            return None

    async def edit_message_reply_markup(self, reply_markup=None, **k):
        from telegram.error import TelegramError
        try:
            return await self._bot.edit_message_reply_markup(
                chat_id=self.chat_id, message_id=self.message_id,
                reply_markup=reply_markup, **k)
        except TelegramError:
            return None


class _HeadlessUpdate:
    def __init__(self, query):
        self.callback_query = query
        self.effective_chat = type("_AutoChat", (), {"id": query.chat_id})()
        self.effective_user = None
        self.message        = None


async def _auto_refresh_job(context):
    """JobQueue callback — re-fires the manual refresh handler in place."""
    job  = context.job
    d    = job.data or {}
    name = job.name
    handler = _autorefresh_handler_for(d.get("callback_data"))
    if handler is None:
        job.schedule_removal(); _autorefresh_jobs.discard(name); return

    # Skip (but keep the timer alive) while a sub-view like Details is open.
    if (d.get("chat_id"), d.get("message_id")) in _autorefresh_paused:
        return

    query  = _HeadlessQuery(context.bot, d["chat_id"], d["message_id"],
                            d["callback_data"])
    update = _HeadlessUpdate(query)
    try:
        await handler(update, context)
    except Exception as e:
        logger.debug("auto-refresh handler %s failed: %s", name, e)

    d["cycles"] = d.get("cycles", 0) + 1
    if query.stop or (AUTO_REFRESH_MAX_CYCLES > 0
                      and d["cycles"] >= AUTO_REFRESH_MAX_CYCLES):
        job.schedule_removal()
        _autorefresh_jobs.discard(name)


def _schedule_auto_refresh(job_queue, chat_id, message_id, callback_data):
    """Arm (or skip) a repeating auto-refresh timer for one card."""
    if job_queue is None or _autorefresh_handler_for(callback_data) is None:
        return
    name = f"autoref|{chat_id}|{message_id}"
    if name in _autorefresh_jobs or job_queue.get_jobs_by_name(name):
        return                                    # already armed
    if len(_autorefresh_jobs) >= AUTO_REFRESH_MAX_JOBS:
        logger.warning("auto-refresh cap (%d) reached — not arming %s",
                       AUTO_REFRESH_MAX_JOBS, name)
        return
    try:
        job_queue.run_repeating(
            _auto_refresh_job,
            interval=AUTO_REFRESH_SECS,
            first=AUTO_REFRESH_SECS,
            name=name,
            data={"chat_id": chat_id, "message_id": message_id,
                  "callback_data": callback_data, "cycles": 0},
        )
        _autorefresh_jobs.add(name)
    except Exception as e:
        logger.debug("could not arm auto-refresh %s: %s", name, e)


def _find_refresh_callback(reply_markup):
    """Return the first known refresh callback_data in an InlineKeyboardMarkup."""
    if not reply_markup or not getattr(reply_markup, "inline_keyboard", None):
        return None
    for row in reply_markup.inline_keyboard:
        for btn in row:
            cb = getattr(btn, "callback_data", None)
            if _autorefresh_handler_for(cb) is not None:
                return cb
    return None


def _install_auto_refresh(app):
    """Wrap bot.send_message once so every TEXT card with a 🔄 button arms a
    30s auto-refresh timer. Idempotent and safe with telegram.Bot __slots__
    (patches at the class level, not the instance)."""

    if state._autorefresh_installed:
        return
    bot_cls   = type(app.bot)
    orig_send = bot_cls.send_message

    async def send_message_wrapped(self, *args, **kwargs):
        msg = await orig_send(self, *args, **kwargs)
        try:
            cb = _find_refresh_callback(getattr(msg, "reply_markup", None))
            if cb and getattr(msg, "chat_id", None) and getattr(msg, "message_id", None):
                _schedule_auto_refresh(app.job_queue, msg.chat_id, msg.message_id, cb)
        except Exception as e:
            logger.debug("auto-refresh arm-on-send skipped: %s", e)
        return msg

    bot_cls.send_message   = send_message_wrapped
    state._autorefresh_installed = True
    logger.info("Auto-refresh engine installed (interval=%ds, max_cycles=%d, cap=%d)",
                AUTO_REFRESH_SECS, AUTO_REFRESH_MAX_CYCLES, AUTO_REFRESH_MAX_JOBS)


def main():


    print("=" * 55)
    print("  SAKZ SCAN BOT v2 — Starting")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # MEMORY — restore user data from JSON backup before DB init
    sakz_memory.restore_on_startup()

    # IMPROVEMENT #2 — init DB and restore state from last run
    db_init()
    db_init_user_tracking()

    # PERSIST DIAGNOSTIC — make ephemeral-DB misconfig impossible to miss in logs.
    # If neither Turso nor a custom SAKZ_DB_PATH (volume) is set, the DB is a local
    # file that Railway WIPES on every redeploy — every per-user setting resets.
    if _USE_TURSO:
        logger.info("✅ PERSISTENCE: Turso cloud DB in use — all per-user data survives redeploys.")
    elif os.environ.get("SAKZ_DB_PATH"):
        logger.info("✅ PERSISTENCE: custom DB path %s — ensure this is on a Railway Volume.", os.path.abspath(DB_PATH))
    else:
        logger.warning(
            "⚠️ PERSISTENCE WARNING: using local file %s with NO Turso and NO volume path. "
            "Railway WIPES this on every redeploy — autoscan, watch, pro, snail, safemode, alerts "
            "AND the admin user list will all reset. FIX: set TURSO_URL + TURSO_TOKEN, or attach a "
            "Railway Volume and set SAKZ_DB_PATH to a file on it (e.g. /data/sakz_data.db).",
            os.path.abspath(DB_PATH),
        )

    _load_best_params()   # load calibrated params from best_params.json if present

    # AUTO PAPER TRADING ��� initialise paper DB tables if module is available
    if _PAPER_AVAILABLE:
        paper_init_db(db_connect)
        logger.info("Paper trading DB tables ready")
    restored, ts = db_load_last_scan()
    if restored:
        state.last_scan_results = restored
        state.last_scan_time    = ts
        logger.info("Restored %d signals from DB (scan at %s)", len(restored), ts)
    state.price_history = db_load_price_history()
    logger.info("Loaded %d price history entries from DB", len(state.price_history))

    # FIX #PERSIST-BIAS — restore flip-cooldown state from DB

    state._last_signal_bias = db_load_signal_bias()
    logger.info("Restored %d signal bias entries from DB", len(state._last_signal_bias))

    # FIX #PERSIST-CACHE — restore signal card cache from DB (last 24h only)

    state._signal_card_cache = db_load_card_cache()
    logger.info("Restored %d signal card cache entries from DB", len(state._signal_card_cache))

    # SAFE MODE — restore users who had it enabled before restart

    state.safemode_users = db_safemode_load()
    logger.info("Restored %d safemode users from DB", len(state.safemode_users))

    # PERSIST — restore /autoscan subscriptions so users keep their subs across
    # GitHub redeploys / Railway restarts (mutate in place to preserve the ref).
    auto_scan_subscribers.clear()
    auto_scan_subscribers.update(db_autoscan_load())
    logger.info("Restored %d autoscan subscribers from DB", len(auto_scan_subscribers))

    # SAKZ_PERF_V2 — process every user's update concurrently instead of one at
    # a time. Without concurrent_updates, PTB finishes handling one update
    # (including its awaited scans / PnL renders) before pulling the next, so a
    # single user's /scan stalled everyone else. We also widen the HTTPX pool so
    # many simultaneous Telegram API calls (sendMessage / sendPhoto) don't queue.
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)
        .connection_pool_size(256)
        .pool_timeout(30.0)
        .connect_timeout(15.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .get_updates_connection_pool_size(16)
        .build()
    )

    # FIX #AUTOREFRESH — every card with a 🔄 button also auto-refreshes (30s)
    _install_auto_refresh(app)

    # ── FIX #WS — WebSocket startup via post_init ─────────────────────────
    # app.run_polling() owns the event loop — the only safe way to launch a
    # long-lived coroutine alongside it is through post_init, which fires
    # inside that same loop after the bot is fully initialised.
    if _WS_AVAILABLE:
        async def _ws_post_init(application):
            asyncio.create_task(start_ws(application.bot))
            logger.info("[sakz_bot] WebSocket task started via post_init ✅")
        app.post_init = _ws_post_init

    # MEMORY — start auto-backup job (every 10 min → sakz_memory.json)
    sakz_memory.start_background_backup(app)

    # ── User activity tracking middleware ─────
    # _activity_middleware is defined at module level (not inside main) so it
    # survives crash-recovery restarts and is always importable for testing.
    from telegram.ext import TypeHandler
    app.add_handler(TypeHandler(Update, _activity_middleware), group=-1)
    logger.info("Activity tracking middleware registered (group=-1)")

    # GLOBAL ERROR HANDLER — a single failed update or job must never crash the
    # bot into the restart loop (which drops pending updates and makes commands
    # appear to "pause"). Transient Telegram/network errors are logged and
    # swallowed; everything else is logged with a traceback but contained.
    app.add_error_handler(global_error_handler)
    logger.info("Global error handler registered")

    # Continuous autoscan — checks every 10 min, pushes >=8/10 signals instantly
    app.job_queue.run_repeating(
        continuous_scan_job,
        interval=600,
        first=120,
        name="continuous_scan"
    )

    # 4-hour digest — housekeeping: alerts, watchlist, broadcast
    app.job_queue.run_repeating(
        auto_scan_job,
        interval=AUTO_SCAN_INTERVAL,
        first=AUTO_SCAN_INTERVAL,
        name="auto_scan"
    )

    # SIGNAL LIFECYCLE maintenance — every 5 min: SL detection, dormancy clear
    # (15 min no motion) and 20%-peak-reversal eviction for the /pnl memory.
    app.job_queue.run_repeating(
        signal_maintenance_job,
        interval=300,
        first=180,
        name="signal_maintenance"
    )

    # AUTOSCAN LIFECYCLE ALERTS — every 5 min: notify subscribers when a pushed
    # signal reaches a profit target (T1/T2/T3) or hits its stop-loss.
    app.job_queue.run_repeating(
        autoscan_lifecycle_job,
        interval=300,
        first=240,
        name="autoscan_lifecycle"
    )

    # FIX #MID-JOB — Mid-tier rotation (ranks 51-200) every 4h, offset 2h from
    # the full scan, so the bot stops repeating only the same top-50 names.
    # Merges its results into the scan cache for the normal push pipeline.
    app.job_queue.run_repeating(
        mid_scan_job,
        interval=14400,   # every 4 hours
        first=7200,       # first run 2h after startup (offset from full scan)
        name="mid_scan"
    )

    # IMPROVEMENT #3 — outcome checker every 30 minutes
    app.job_queue.run_repeating(
        check_signal_outcomes,
        interval=1800,
        first=300,
        name="outcome_checker"
    )

    # NEW — Watchlist check every 30 minutes
    app.job_queue.run_repeating(
        watchlist_check_job,
        interval=1800,
        first=900,
        name="watchlist_check"
    )

    # NEW — Funding rate extremes alert every 1 hour
    app.job_queue.run_repeating(
        funding_alert_job,
        interval=3600,
        first=600,
        name="funding_alert"
    )

    # NEW — BTC volatility monitor every 15 minutes
    app.job_queue.run_repeating(
        btc_volatility_job,
        interval=900,
        first=120,
        name="btc_volatility"
    )

    # NEW — BTC regime-shift market alert every 60 seconds
    # Broadcasts a counter-trend risk advisory to all subscribers when BTC
    # genuinely flips bullish<->bearish (hybrid: regime + 10-min hold + price move).
    app.job_queue.run_repeating(
        btc_regime_alert_job,
        interval=60,
        first=90,
        name="btc_regime_alert"
    )

    # NEW — Trend-dying monitor every 30 minutes
    # Automatically alerts users when their tracked trade's trend weakens
    app.job_queue.run_repeating(
        trend_dying_job,
        interval=1800,
        first=600,
        name="trend_dying"
    )

    # Restore active trade tracking jobs from DB (multi-trade)
    saved_tracking = db_load_all_trades()
    for chat_id, trades in saved_tracking.items():
        state.user_tracking[chat_id] = trades
        for trade_id, data in trades.items():
            job = app.job_queue.run_repeating(
                send_trade_update,
                interval=data['interval'] * 60,
                first=data['interval'] * 60,
                chat_id=chat_id,
                name=f"trade_{chat_id}_{trade_id}",
                data={'trade_id': trade_id}
            )
            state.user_tracking[chat_id][trade_id]['job'] = job
        logger.info("Restored %d trade(s) for chat_id %s", len(trades), chat_id)

    # Price-level alert checker — every 3 minutes
    app.job_queue.run_repeating(
        price_alert_job,
        interval=180,
        first=60,
        name="price_alert_checker"
    )

    # ── 🐌 SNAIL MODE — restore unlocked users from DB ───────────────────────

    state.snail_unlocked = db_snail_load_unlocked()
    logger.info("Restored %d snail-unlocked users", len(state.snail_unlocked))

    # 🐌 SNAIL daily signal job — fires every day at 08:00 UTC
    app.job_queue.run_daily(
        snail_daily_job,
        time=__import__('datetime').time(hour=8, minute=0, tzinfo=__import__('datetime').timezone.utc),
        name="snail_daily"
    )

    # Conversation handler

    # ── /pro fast scan — uptrend + manipulation every 30 min ──────
    app.job_queue.run_repeating(
        pro_fast_job,
        interval=1800,
        first=300,
        name="pro_fast"
    )
    # ── /pro gainers scan — persistence tracker every 4 hours ─────
    app.job_queue.run_repeating(
        pro_gainers_job,
        interval=14400,
        first=600,
        name="pro_gainers"
    )
    app.add_handler(CommandHandler("pro", pro_command))

    # ── PRIME — secret high-conviction subscription ───────────────
    app.add_handler(CommandHandler("prime", prime_command))
    # Unlisted autoscan diagnostic — explains why autoscan is/ isn't pushing
    app.add_handler(CommandHandler("autoscandiag", autoscandiag_command))
    # 15-minute live cache refresh of the ranked Top picks
    app.job_queue.run_repeating(
        prime_refresh_job,
        interval=prime.PRIME_CACHE_TTL_MIN * 60,
        first=150,
        name="prime_refresh"
    )
    # Per-user GMT alert scheduler — minute tick
    app.job_queue.run_repeating(
        prime_alert_scheduler_job,
        interval=60,
        first=60,
        name="prime_alert_scheduler"
    )
    # Dedup ledger cleanup every 6h
    app.job_queue.run_repeating(
        prime_cleanup_job,
        interval=21600,
        first=3600,
        name="prime_cleanup"
    )

    # ── Paper trading — mark-to-market every 15 min ────────────────
    if _PAPER_AVAILABLE:
        app.job_queue.run_repeating(
            paper_job,
            interval=900,
            first=120,
            name="paper_mtm"
        )

    # ── FIX #HEARTBEAT — Self-monitoring ping every 5 min ──────────
    app.job_queue.run_repeating(
        heartbeat_job,
        interval=300,
        first=30,
        name="heartbeat"
    )

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("pick", pick_command)],
        states={
            PICK_TRADE:   [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_trade_pick)],
            ASK_REMINDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_reminder_choice)],
            ASK_INTERVAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_interval)],
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation)],
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("start",      start_command))
    app.add_handler(CommandHandler("menu",       menu_command))
    app.add_handler(CommandHandler("status",     status_command))
    app.add_handler(CommandHandler("scan",       scan_tf_command))   # TF-aware e.g. /scan t15m
    app.add_handler(CommandHandler("scannew",    scan_new_command))  # /scannew 24h alias
    app.add_handler(CommandHandler("best",       best_command))
    app.add_handler(CommandHandler("compare",    compare_command))
    app.add_handler(CommandHandler("stoptrade",  stoptrade_command))
    app.add_handler(CommandHandler("trades",     trades_command))
    app.add_handler(CommandHandler("palert",     palert_command))
    app.add_handler(CommandHandler("unpalert",   unpalert_command))
    app.add_handler(CommandHandler("tg",         tg_command))
    app.add_handler(CommandHandler("tl",         tl_command))
    app.add_handler(CommandHandler("stats",      stats_command))
    app.add_handler(CommandHandler("backtest",   backtest_command))
    app.add_handler(CommandHandler("alert",      alert_command))
    app.add_handler(CommandHandler("unalert",    unalert_command))
    app.add_handler(CommandHandler("autoscan",   autoscan_command))
    app.add_handler(CommandHandler("filter",     filter_command))
    app.add_handler(CommandHandler("pnl",        pnl_command))
    app.add_handler(CommandHandler("cscan",      cscan_tf_command))  # TF-aware e.g. /cscan btc t15m
    app.add_handler(CommandHandler("analyse",    analyse_command))   # raw analysis, no gates
    app.add_handler(CallbackQueryHandler(analyse_section_callback,  pattern=r'^an_sec\|'))
    app.add_handler(CommandHandler("watch",      watch_command))
    app.add_handler(CommandHandler("unwatch",    unwatch_command))
    app.add_handler(CommandHandler("safemode",   safemode_command))
    app.add_handler(CommandHandler("broadcast",  broadcast_command))
    app.add_handler(CommandHandler("leaderboard",leaderboard_command))
    app.add_handler(CommandHandler("lb",         lb_command))
    app.add_handler(CommandHandler("chart",      chart_command))
    app.add_handler(CommandHandler("admin",      admin_command))
    app.add_handler(CommandHandler("fgi",        fgi_command))
    app.add_handler(CommandHandler("calibrate",  calibrate_command))
    app.add_handler(CommandHandler("top5",       top_command))        # /top5 shortcut
    app.add_handler(CommandHandler("top10",      top_command))        # /top10 shortcut
    app.add_handler(CommandHandler("top",        top_command))        # /topN generic
    app.add_handler(CommandHandler("scanmid",    scanmid_command))
    app.add_handler(CommandHandler("scalp",      scalp_command))      # short TF scan 15m+1h
    app.add_handler(CommandHandler("swing",      swing_command))      # long TF scan 4h+1d
    app.add_handler(CommandHandler("check",      check_command))      # validate open trade
    app.add_handler(CommandHandler("confirm",    confirm_command))    # re-validate signal on demand
    app.add_handler(CommandHandler("manual",     manual_command))     # full public user guide
    app.add_handler(CommandHandler("optimize",   optimize_command))   # hyperopt param search (admin)
    app.add_handler(CommandHandler("xgtrain",    xgtrain_command))    # retrain XGBoost (admin)
    app.add_handler(CommandHandler("rftrain",    rftrain_command))    # retrain RF (admin)
    app.add_handler(CommandHandler("dbcheck",    dbcheck_command))    # DB diagnostic (admin)
    if _HIST_BT_AVAILABLE:
        app.add_handler(CommandHandler("btfull", btfull_command))     # historical walk-forward backtest
    if _PAPER_AVAILABLE:
        app.add_handler(CommandHandler("paper",  paper_command))      # paper trading dashboard
    app.add_handler(CallbackQueryHandler(stop_trade_callback,    pattern=r'^stop_trade'))
    app.add_handler(CallbackQueryHandler(remove_palert_callback, pattern=r'^rm_palert\|'))
    app.add_handler(CallbackQueryHandler(signal_refresh_callback,          pattern=r'^sig_refresh\|'))
    app.add_handler(CallbackQueryHandler(signal_details_callback,           pattern=r'^sig_details\|'))
    app.add_handler(CallbackQueryHandler(signal_back_callback,              pattern=r'^sig_back\|'))
    app.add_handler(CallbackQueryHandler(admin_callback_handler,           pattern=r'^admin_'))
    app.add_handler(CallbackQueryHandler(chart_callback_handler,           pattern=r'^chart_refresh\|'))
    app.add_handler(CallbackQueryHandler(chart_tf_refresh_callback,        pattern=r'^chart_tf_refresh\|'))
    app.add_handler(CallbackQueryHandler(pnl_callback_handler,             pattern=r'^pnl_bot\|'))
    app.add_handler(CallbackQueryHandler(pnl_callback_handler,             pattern=r'^pnl_custom\|'))
    app.add_handler(CallbackQueryHandler(pnl_img_callback,                 pattern=r'^pnlimg_'))
    app.add_handler(CallbackQueryHandler(view_signal_callback,             pattern=r'^view_signal\|'))
    app.add_handler(CallbackQueryHandler(pnl_from_signal_callback,         pattern=r'^pnl_from_signal\|'))
    app.add_handler(CallbackQueryHandler(pnl_from_cscan_callback,          pattern=r'^pnl_from_cscan\|'))
    app.add_handler(CallbackQueryHandler(cscan_refresh_callback,           pattern=r'^cscan_refresh\|'))
    app.add_handler(CallbackQueryHandler(cscan_tf_callback,                pattern=r'^cscan_tf\|'))
    app.add_handler(CallbackQueryHandler(feed_refresh_callback,            pattern=r'^feed_refresh\|'))
    app.add_handler(CallbackQueryHandler(autoscan_tf_callback,             pattern=r'^autoscan_tf\|'))
    app.add_handler(CallbackQueryHandler(statsbest_callback,               pattern=r'^statsbest\|'))
    app.add_handler(CallbackQueryHandler(stats_time_callback,              pattern=r'^stats_time\|'))
    app.add_handler(CallbackQueryHandler(bt_time_callback,                 pattern=r'^bt_time\|'))
    app.add_handler(CallbackQueryHandler(leaderboard_time_callback,        pattern=r'^lb_time\|'))
    app.add_handler(CallbackQueryHandler(custom_compare_refresh_callback,  pattern=r'^custom_compare_refresh\|'))
    app.add_handler(CallbackQueryHandler(compare_refresh_callback,         pattern=r'^cmp_refresh\|'))
    app.add_handler(CallbackQueryHandler(compare_full_refresh_callback,    pattern=r'^cmp_full_refresh\|'))
    app.add_handler(CallbackQueryHandler(fgi_refresh_callback,             pattern=r'^fgi_refresh'))
    # �� SNAIL callbacks — handles activate/status/report/stop buttons
    app.add_handler(CallbackQueryHandler(snail_callback_handler,    pattern=r'^snail_'))
    # 📋 MENU interactive button callbacks
    app.add_handler(CallbackQueryHandler(menu_callback_handler,     pattern=r'^menu\|'))
    app.add_handler(CallbackQueryHandler(menu_run_callback,         pattern=r'^menu_run\|'))
    # ── PRIME callbacks (timezone picker, alert-slot toggles, dashboard) ──
    app.add_handler(CallbackQueryHandler(prime_tzmenu_callback,     pattern=r'^prime_tzmenu\|'))
    app.add_handler(CallbackQueryHandler(prime_tz_callback,         pattern=r'^prime_tz\|'))
    app.add_handler(CallbackQueryHandler(prime_slot_callback,       pattern=r'^prime_slot\|'))
    app.add_handler(CallbackQueryHandler(prime_show_callback,       pattern=r'^prime_show\|'))
    app.add_handler(CallbackQueryHandler(prime_off_callback,        pattern=r'^prime_off\|'))
    app.add_handler(MessageHandler(filters.Regex(r'^/top\d+') & filters.TEXT, top_command))
    # 🐌 SNAIL commands — visible only to unlocked users
    app.add_handler(CommandHandler("snail",      snail_command))
    app.add_handler(CommandHandler("snailvault", snailvault_command))

    # Admin password entry — must come BEFORE pnl_message_handler
    async def _admin_pw_gate(update: Update, context: ContextTypes.DEFAULT_TYPE):
        consumed = await admin_password_handler(update, context)
        if not consumed:
            # An active PnL conversation takes priority over the autoscan TF
            # catch, so a reply like "6" reaches the PnL flow, not the TF parser.
            if context.user_data.get('pnl_step'):
                await pnl_message_handler(update, context)
                return
            consumed = await autoscan_custom_tf_handler(update, context)
        if not consumed:
            await pnl_message_handler(update, context)

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _admin_pw_gate))
    # unknown_command MUST be last — it catches everything else including /scan1234JP$$
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    print("\n✅ Bot running. Open Telegram and type /menu\n")

    # Crash recovery loop — restarts polling automatically if an exception occurs
    while True:
        try:
            app.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error("Bot crashed: %s — restarting in 15 seconds...", e)
            time.sleep(15)
            continue


if __name__ == "__main__":
    main()
