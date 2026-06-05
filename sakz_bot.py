import logging
import os
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

# === Extracted data-access layer (sakz_db.py) ===
from sakz_db import (  # noqa: F401  re-exported; existing call sites unchanged
    db_connect,
    db_init,
    db_save_scan,
    db_load_last_scan,
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

@dataclass
class ScanFailure:
    """Carries the reason a symbol produced no signal."""
    reason: str          # machine tag: see REASON_* constants below
    exchange: str = ""
    tf: str       = ""
    detail: str   = ""   # human-readable extra context

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

# Custom scan results per chat { chat_id: [signals] }
cscan_results    = {}
# Stores per-chat scan context so any Refresh knows which result set to use.
# Key: chat_id  Value: {'source': 'scan'|'scalp'|'swing'|'custom', 'results': [...], 'title': str}
_chat_scan_ctx   = {}
# Snapshot taken at scan time for /compare PnL { chat_id: {symbol: {entry_price, leverage, bias, scan_time}} }
compare_snapshot = {}
# Signal card cache for Details/Back button { cache_key: {signal, rank, primary_kb} }
_signal_card_cache = {}
# Flip cooldown cache { symbol: {bias, time} } — prevents rapid direction reversals
_last_signal_bias: dict = {}

# Auto-scan interval (seconds). 4 hours = 14400
AUTO_SCAN_INTERVAL = 14400

# ── FIX #HEARTBEAT — Self-monitoring ─────────────────────────────────────────
# Set HEALTH_CHECK_URL in env (BetterStack, UptimeRobot, etc.) to get
# notified when the bot goes silent. A ping is sent every 5 minutes.
HEALTH_CHECK_URL   = os.environ.get("HEALTH_CHECK_URL", "")   # e.g. https://uptime.betterstack.com/api/v1/heartbeat/XXXX
ADMIN_ALERT_CHAT   = os.environ.get("ADMIN_ALERT_CHAT", "")   # chat_id to receive self-alerts (optional)

# ── FIX #SCANTIME — Scan duration percentile tracking ────────────────────────
_scan_durations: list = []   # rolling list of recent scan durations (seconds)

# ─────────────────────────────────────────────
# 🐌 SNAIL MODE — Hidden easter egg feature
# Activated ONLY via secret command /scan1234JP$$
# /snail alone does nothing unless user is unlocked
# ─────────────────────────────────────────────
SNAIL_SECRET_CMD   = "scan1234JP$$"      # secret unlock passphrase
snail_unlocked     = set()               # chat_ids that have unlocked snail mode
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





# ── Gainers log ──────────────────────────────────────────────────────────────────





# ── Manipulation log ──────────────────────────────────────────────────────────────






# ── /pro Detection engine ─────────────────────────────────────────────────────────

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
        resp = requests.get(url, timeout=10, headers=HEADERS)
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

    # MEXC fallback ─────────────────────────────────────────────────────────────
    try:
        resp = requests.get(
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

        # 5-7 — CoinGecko fundamentals (best-effort) ────────────────────────────
        try:
            slug    = symbol.replace("USDT", "").lower()
            cg_resp = requests.get(
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
        f"🟢 BIAS: LONG (trend confirmed)\n"
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
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
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


# ── /pro In-memory dedup ───────────────────────────────────────────────────────────

_pro_alerted_uptrend: set = set()
_pro_alerted_gainers: set = set()
_pro_alerted_manip:   set = set()

# ── /pro Top-gainers cache (5-min TTL) ────────────────────────────────────────────
_pro_gainers_cache: list      = []
_pro_gainers_cache_ts: float  = 0.0
_PRO_GAINERS_TTL: int         = 300   # seconds


def _pro_fetch_top_gainers_cached() -> list:
    """Return cached top-gainers list; refresh if older than 5 minutes."""
    global _pro_gainers_cache, _pro_gainers_cache_ts
    import time as _time
    if _pro_gainers_cache and (_time.time() - _pro_gainers_cache_ts) < _PRO_GAINERS_TTL:
        return _pro_gainers_cache
    fresh = _pro_fetch_top_gainers()
    if fresh:
        _pro_gainers_cache    = fresh
        _pro_gainers_cache_ts = _time.time()
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
        requests.get(HEALTH_CHECK_URL, timeout=5)
        logger.debug("Heartbeat ping sent ✅")
    except Exception as e:
        logger.warning("Heartbeat ping failed: %s", e)

    # Also emit scan duration stats to admin if scans are slow
    global _scan_durations
    if len(_scan_durations) >= 5:
        recent = _scan_durations[-10:]
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
                _scan_durations = []   # reset after alert
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
            except Exception:
                pass
            existing_row    = existing_ut.get((sym, exchange))
            already_alerted = existing_row and existing_row.get("alert_sent", 0)
            if not already_alerted and ut_key not in _pro_alerted_uptrend:
                uptrend_alerts.append(uptrend)
                _pro_alerted_uptrend.add(ut_key)
                try:
                    db_pro_mark_uptrend_alerted(sym, exchange)
                except Exception:
                    pass

        # Manipulation ─────────────────────────────────────────────────────────
        if manip and manip.get("is_scam") and manip["manip_score"] >= 60:
            mk = f"{sym}_{exchange}"
            try:
                db_pro_upsert_manip(sym, exchange, manip["manip_score"], manip["reasons"])
            except Exception:
                pass
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
                    except Exception:
                        pass

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
            except Exception:
                pass

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
        except Exception:
            pass

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


# ── /pro Command handler ────────────────────────────────────────────────────────────

def _pro_full_command_guide() -> str:
    """Single source of truth for the bot's full PUBLIC command list.
    Admin/hidden commands (/admin, /optimize, /xgtrain, /rftrain, /dbcheck)
    and secret commands are intentionally excluded."""
    return (
        "📖  SAKZ BOT — FULL COMMAND GUIDE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Everything the bot can do, in one place.\n\n"
        "🔭  SCANNING\n"
        "/scan                Full market scan (4H, top pairs)\n"
        "/scan BTC            Scan a specific pair\n"
        "/scan BTC 1h         Specific pair on a custom timeframe\n"
        "/scan new 24h        New listings scan (m/h/d/w units)\n"
        "/cscan ZEC           Custom pair scan (auto-detect TF)\n"
        "/cscan ZEC 15m       Custom pair on 15m / 1h / 4h / 1d\n"
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
        "/autoscan            Toggle periodic auto-scan\n"
        "/broadcast on|off    Toggle signal auto-posting here\n"
        "/safemode            Toggle automatic dying-trend alerts\n\n"
        "🔬  PRO ALERT SUITE\n"
        "/pro                 Show this full command guide\n"
        "/pro on | /pro off   Subscribe / unsubscribe to PRO alerts\n"
        "/pro status          Live PRO tracking stats\n"
        "/pro alerts          PRO alert suite overview\n\n"
        "⚙️  GENERAL\n"
        "/status              Bot health & uptime\n"
        "/menu                Interactive command menu\n"
        "/start               Welcome message\n"
    )


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
last_scan_results     = []
previous_scan_results = []
last_scan_time        = None
price_history         = {}   # { 'BYBIT_BTCUSDT': [{time, price},...] }
user_tracking         = {}   # { chat_id: { trade_id: {signal, entry_price, start_time, interval, job} } }

# ── Safe Mode state ────────────────────────────────────────────────────────────
# chat_ids with safe mode ON (loaded from DB on startup)
safemode_users: set = set()
# Last signals each safemode user received from any scan source
# { chat_id: [ signal_dict, ... ] }  — capped at 20 signals per user
safemode_last_signals: dict = {}
_SAFEMODE_MAX_SIGNALS = 20    # how many recent signals to monitor per user




def _safemode_store_signals(chat_id: int, signals: list):
    """
    Called whenever a scan result is presented to a safemode user.
    Stores up to _SAFEMODE_MAX_SIGNALS recent signals for trend monitoring.
    """
    if chat_id not in safemode_users:
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

SCAN_EXECUTOR  = ThreadPoolExecutor(max_workers=3)   # max 3 simultaneous scans
CACHE_TTL_SECS = 600                                  # 10 minutes — matches continuous_scan_job interval

# Global scan cache — served to users who scan within TTL of last scan
_scan_cache        = None   # { 'results': [...], 'time': datetime }
_scan_cache_lock   = asyncio.Lock()   # prevents cache stampede

# ── BTC Market Regime Cache ───────────────────
# Cached once per scan so all 150 pairs share the same regime read.
# { 'regime': 'BULL'|'BEAR'|'NEUTRAL', 'time': datetime }
_btc_regime_cache     = None
_btc_regime_cache_ttl = 900  # 15 minutes — same as scan cache

# ── BTC Dominance Cache ───────────────────────────────────────────�����─────────
# BTC.D rising = capital flowing out of alts → penalise altcoin LONGs
# Fetched from Bybit BTCDOMUSDT or Binance BTCDOMUSDT (may not always be available)
# { 'btcd': float, 'trend': 'rising'|'falling'|'flat', 'time': datetime }
_btcd_cache: dict = {}
_BTCD_TTL = 1800  # 30 min — BTC.D doesn't move that fast

def _get_btc_dominance() -> dict:
    """
    Fetch BTC dominance trend from available sources.
    Returns {'btcd': float_pct, 'trend': 'rising'|'falling'|'flat'}.
    Falls back to {'btcd': 0, 'trend': 'flat'} if unavailable.
    Uses the last 10 daily closes of BTC.D to determine trend direction.
    """
    global _btcd_cache
    now = datetime.now()
    if _btcd_cache and (now - _btcd_cache['time']).total_seconds() < _BTCD_TTL:
        return _btcd_cache

    result = {'btcd': 0.0, 'trend': 'flat', 'time': now}
    try:
        # Binance BTCDOMUSDT is a dominance index (not a futures pair — spot only)
        # We approximate BTC dominance from CoinGecko global API if available,
        # otherwise derive a simple proxy: BTC market cap vs ETH+BTC
        # Practical approach: try binance spot 1d BTCDOMUSDT if present
        for fetch_fn, sym, interval in [
            (bybit_fetch_ohlcv,   'BTCDOMUSDT', 'D'),
            (binance_fetch_ohlcv, 'BTCDOMUSDT', '1d'),
        ]:
            try:
                df_d = fetch_fn(sym, interval, 15)
                if df_d is not None and len(df_d) >= 5:
                    closes = df_d['close'].dropna().values
                    current = float(closes[-1])
                    # Trend: compare 3-period SMA now vs 3-period SMA 5 bars ago
                    sma_now  = float(closes[-3:].mean())
                    sma_prev = float(closes[-8:-5].mean()) if len(closes) >= 8 else sma_now
                    delta    = sma_now - sma_prev
                    trend    = 'rising' if delta > 0.2 else 'falling' if delta < -0.2 else 'flat'
                    result   = {'btcd': current, 'trend': trend, 'time': now}
                    logger.debug("BTC.D: %.2f%% trend=%s (delta=%.3f)", current, trend, delta)
                    break
            except Exception:
                continue
    except Exception as e:
        logger.debug("BTC.D fetch error: %s", e)

    _btcd_cache = result
    return result

# Portfolio-level correlation summary — updated after each scan
# { btc_regime, total_signals, corr_long, corr_short, independent,
#   corr_slot_cap, over_cap, risk_level }
_last_portfolio_summary = None

# Per-chat locks — lets each user run independently without blocking others
_chat_scan_locks   = {}     # { chat_id: asyncio.Lock() }

def get_chat_lock(chat_id):
    if chat_id not in _chat_scan_locks:
        _chat_scan_locks[chat_id] = asyncio.Lock()
    return _chat_scan_locks[chat_id]

# ─────────────────────────────────────────────
# LEVERAGE CALCULATOR
# ─────────────────────────────────────────────
def calculate_leverage(price, entry_low, entry_high, stop_loss, atr, confidence, bias):
    try:
        entry_mid = (entry_low + entry_high) / 2
        sl_dist   = abs(entry_mid - stop_loss) / entry_mid * 100
        if sl_dist == 0:
            return None
        raw_max = 100 / (sl_dist * 1.4)
        atr_pct = (atr / price) * 100
        # Volatility factor + hard ceiling per regime — keeps suggestions realistic
        # High-ATR coins liquidate fast; capping leverage prevents reckless numbers.
        if atr_pct < 1.0:
            vf = 1.0;  vl = "Very Low";  regime_cap = 50
        elif atr_pct < 2.0:
            vf = 0.85; vl = "Low";       regime_cap = 30
        elif atr_pct < 3.5:
            vf = 0.70; vl = "Medium";    regime_cap = 20
        elif atr_pct < 5.5:
            vf = 0.55; vl = "High";      regime_cap = 10
        elif atr_pct < 7.0:
            vf = 0.40; vl = "Very High"; regime_cap = 5
        else:
            vf = 0.25; vl = "Extreme";   regime_cap = 3
        cf        = {10: 1.0, 9: 0.85, 8: 0.70}.get(confidence, 0.50)
        suggested = min(raw_max * vf * cf, regime_cap)
        max_safe  = min(raw_max * vf,      regime_cap)
        def clean(x):
            if x >= 50:   return int(x // 10) * 10
            elif x >= 20: return int(x // 5) * 5
            elif x >= 10: return int(x // 2) * 2
            else:         return max(1, int(x))
        suggested = clean(suggested)
        max_safe  = clean(max_safe)
        liq_dist  = (1 / suggested) * 100 * 0.9 if suggested > 0 else 100
        fluct     = max(0, liq_dist - sl_dist)
        return {'suggested': suggested, 'max_safe': max_safe,
                'sl_dist': sl_dist, 'liq_dist': liq_dist,
                'fluct': fluct, 'atr_pct': atr_pct, 'vol_label': vl}
    except Exception:
        return None


# ─────────────────────────────────────────────
# BYBIT API
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
# ML MODEL AVAILABILITY FLAGS
# Graceful fallback — bot runs normally if these
# optional modules are not installed.
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
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
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# REAL-TIME WEBSOCKET LAYER — sakz_ws.py
# Streams live price, funding, liquidation, volume spikes from MEXC.
# ws_price() / ws_funding() are used as a fast cache before REST fallback.
# ─────────────────────────────────────────────
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
DEFAULT_OPTIM_PARAMS = {
    "atr_stop_mult":    1.0,
    "atr_target_mult":  1.6,
    "min_confidence":   4,
    "cap_osc":          3,
    "cap_mtf":          2,
    "cap_cross":        3,
    "cap_pos":          3,
    "cap_structure":    4,
    "rsi_oversold":     30,
    "rsi_overbought":   70,
}
OPTIM_BEST_PARAMS = DEFAULT_OPTIM_PARAMS.copy()


def _load_best_params():
    """Load optimized params from best_params.json if available."""
    global OPTIM_BEST_PARAMS
    params = DEFAULT_OPTIM_PARAMS.copy()
    try:
        if os.path.exists(BEST_PARAMS_PATH):
            with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                params.update({k: loaded[k] for k in loaded.keys() if k in params})
                logger.info("Loaded optimization params from %s", BEST_PARAMS_PATH)
    except Exception as e:
        logger.warning("Failed to load %s: %s", BEST_PARAMS_PATH, e)
    OPTIM_BEST_PARAMS = params








# ─────────────────────────────────────────────
# MEXC API
# ─────────────────────────────────────────────






# ─────────────������──────────────────────────────
# ─────────────────────────────────────────────
# IMPROVEMENT #7 — BINANCE PERPETUALS (free public API)
# Graceful skip if Binance blocks Railway's IP.
# ─────────────────────────────────────────────









# ── WS-FIRST PRICE AND FUNDING HELPERS ───────────────────────────────────────
# All price lookups go through here. Tries the WebSocket cache first (instant,
# zero network cost). Falls back to REST only if WS hasn't seen the symbol yet.
# This is the single change that eliminates REST calls for price during tracking,
# PnL updates, price alerts, and any other live-price read in the bot.

def _get_live_price(symbol: str, exchange: str = '') -> float:
    """
    WS-first price lookup. Returns float (0 on total failure).
    exchange hint: 'BYBIT' | 'BINANCE' | 'MEXC' — used only for REST fallback.
    """
    # 1. Try WebSocket cache (sub-millisecond, no network)
    cached = ws_price(symbol)
    if cached and cached > 0:
        return cached

    # 2. REST fallback — ordered by preference / availability
    exch = (exchange or '').upper()
    if exch == 'BYBIT' or (not exch and sakz_exchanges.BYBIT_AVAILABLE is not False):
        p = bybit_get_current_price(symbol)
        if p and p > 0:
            return p
    if exch == 'BINANCE' or (not exch and sakz_exchanges.BINANCE_AVAILABLE):
        p = binance_get_current_price(symbol)
        if p and p > 0:
            return p
    # MEXC always last (no futures ticker endpoint as fast)
    p = mexc_get_current_price(symbol)
    return p if p else 0


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
        if result:
            result['exchange'] = 'BINANCE'
        return result
    except Exception:
        return None


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def find_pivot_support(df, lookback=60, swing=3):
    """
    FIX #2 — Real structural support via swing-low detection.
    A swing low is a candle whose low is lower than the `swing`
    candles on each side.

    FIX #SL — Previously returned max(pivots), which is the HIGHEST swing low
    in the window.  That is correct for the "nearest support below price" use
    case ONLY when all pivots are below the current price.  But when the market
    has sold off since those pivots formed, max(pivots) can be ABOVE the current
    price, producing a stop-loss above entry — caught by the direction-safety
    guard but wasteful.

    The fix: filter to only swing-lows that are strictly below the current close,
    then return the highest of those (nearest meaningful floor).  If none qualify
    (e.g. price is already at all-time lows in the window), fall back to the
    rolling-min of the lookback window which is guaranteed to be ≤ current price.
    """
    lows        = df['low'].values
    closes      = df['close'].values
    current_price = closes[-1]
    n           = len(lows)
    pivots      = []
    start       = max(swing, n - lookback)
    for i in range(start, n - swing):
        if all(lows[i] <= lows[i - j] for j in range(1, swing + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, swing + 1)):
            pivots.append(lows[i])

    # Keep only pivots that are genuinely below the current price.
    valid = [p for p in pivots if p < current_price]
    if valid:
        return max(valid)   # nearest support floor (highest pivot below price)
    # Fallback: rolling-min is always ≤ current price by definition
    return df['low'].iloc[-lookback:].min()


def find_pivot_resistance(df, lookback=60, swing=3):
    """
    FIX #2 — Real structural resistance via swing-high detection.
    Returns the lowest (closest relevant ceiling) swing high.
    Falls back to 50-period rolling max.
    """
    highs  = df['high'].values
    n      = len(highs)
    pivots = []
    start  = max(swing, n - lookback)
    for i in range(start, n - swing):
        if all(highs[i] >= highs[i - j] for j in range(1, swing + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, swing + 1)):
            pivots.append(highs[i])
    if pivots:
        return min(pivots)
    return df['high'].iloc[-lookback:].max()


def add_indicators(df, timeframe='4h'):
    """
    FIX #2  — Pivot-based S/R instead of rolling min/max
    FIX #5  — Volume MA shifted by 1 so current bar is excluded
    FIX #9  — Stochastic window=5 for 4H crypto sensitivity
               (daily charts keep default window=14)
    """
    try:
        df  = df.copy()
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']

        df['rsi']       = ta.momentum.RSIIndicator(c, window=14).rsi()
        macd            = ta.trend.MACD(c)
        df['macd_diff'] = macd.macd_diff()
        df['macd_line'] = macd.macd()
        df['macd']      = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        bb              = ta.volatility.BollingerBands(c)
        df['bb_upper']  = bb.bollinger_hband()
        df['bb_lower']  = bb.bollinger_lband()
        df['bb_mid']    = bb.bollinger_mavg()
        # FIX #BB — bandwidth = (upper - lower) / mid; squeeze = bw at 20-period low
        df['bb_bw']     = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_bw_min'] = df['bb_bw'].rolling(20).min().shift(1)  # shift so current bar not included
        df['ema20']     = ta.trend.EMAIndicator(c, window=20).ema_indicator()
        df['ema50']     = ta.trend.EMAIndicator(c, window=50).ema_indicator()
        df['atr']       = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

        # FIX #5 — shift(1) excludes the current forming candle from the baseline
        df['volume_ma'] = v.rolling(20).mean().shift(1)

        # FIX #COR — Close Location Value (CLV): genuinely independent of RSI/MACD/EMA.
        # Measures where within the candle range the close settled each bar.
        #   CLV = (2*close - high - low) / (high - low)
        #   +1.0 = close at the very top (buyers fully absorbed the candle)
        #   -1.0 = close at the very bottom (sellers fully absorbed)
        # Rolling 5-bar mean catches sustained absorption without noise.
        hl_range     = (h - l).replace(0, float('nan'))
        df['clv']    = ((2 * c - h - l) / hl_range).fillna(0)
        df['clv_ma'] = df['clv'].rolling(5).mean().shift(1)   # shift: exclude current bar

        # FIX #9 �� use window=5 for 4H (crypto moves fast); daily stays at 14
        stoch_window = 5 if timeframe == '4h' else 14
        stoch        = ta.momentum.StochasticOscillator(h, l, c, window=stoch_window, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        df = df.dropna()

        # FIX #2 — Pivot S/R (computed after dropna so indices are clean)
        if len(df) >= 10:
            df['support']    = find_pivot_support(df)
            df['resistance'] = find_pivot_resistance(df)
        else:
            df['support']    = l.rolling(20).min()
            df['resistance'] = h.rolling(20).max()

        # FIX #DIV — RSI and MACD divergence detection
        # Bullish divergence: price makes lower low but RSI makes higher low → trend exhaustion reversal signal
        # Bearish divergence: price makes higher high but RSI makes lower high → hidden weakness
        # Uses a 5-bar lookback to find the prior swing low/high
        if len(df) >= 6:
            close_s = df['close']
            rsi_s   = df['rsi']
            macd_s  = df['macd_diff']
            # Rolling 5-bar prior swing for comparison
            price_lo5 = close_s.shift(1).rolling(5).min()
            price_hi5 = close_s.shift(1).rolling(5).max()
            rsi_lo5   = rsi_s.shift(1).rolling(5).min()
            rsi_hi5   = rsi_s.shift(1).rolling(5).max()
            macd_lo5  = macd_s.shift(1).rolling(5).min()
            macd_hi5  = macd_s.shift(1).rolling(5).max()

            # Bullish RSI divergence: price lower low, RSI higher low
            df['rsi_bull_div'] = (close_s < price_lo5) & (rsi_s > rsi_lo5) & (rsi_s < 50)
            # Bearish RSI divergence: price higher high, RSI lower high
            df['rsi_bear_div'] = (close_s > price_hi5) & (rsi_s < rsi_hi5) & (rsi_s > 50)
            # Bullish MACD divergence: price lower low, MACD histogram higher low
            df['macd_bull_div'] = (close_s < price_lo5) & (macd_s > macd_lo5) & (macd_s < 0)
            # Bearish MACD divergence: price higher high, MACD histogram lower high
            df['macd_bear_div'] = (close_s > price_hi5) & (macd_s < macd_hi5) & (macd_s > 0)
        else:
            for col in ['rsi_bull_div', 'rsi_bear_div', 'macd_bull_div', 'macd_bear_div']:
                df[col] = False

        return df
    except Exception as e:
        logger.warning("add_indicators error (timeframe=%s): %s", timeframe, e)
        return None


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
# ──────────────���──────────────────────────────
# ── BTC spot price cache (for regime invalidation) ────────────────────────────
_btc_price_cache: dict = {}   # {'price': float, 'time': datetime}
_BTC_PRICE_TTL = 60           # refresh every 60 s

def _get_btc_price_cached() -> float:
    """Return cached BTC/USDT price, refreshing every 60 s."""
    global _btc_price_cache
    now = datetime.now()
    if _btc_price_cache and (now - _btc_price_cache['time']).total_seconds() < _BTC_PRICE_TTL:
        return _btc_price_cache['price']
    price = 0.0
    try:
        for fn, sym in [
            (lambda s: bybit_fetch_ohlcv(s, '1', 2),  'BTCUSDT'),
            (lambda s: binance_fetch_ohlcv(s, '1m', 2), 'BTCUSDT'),
            (lambda s: mexc_fetch_ohlcv(s, '1m', 2),    'BTCUSDT'),
        ]:
            df = fn(sym)
            if df is not None and len(df) >= 1:
                price = float(df['close'].iloc[-1])
                break
    except Exception:
        pass
    _btc_price_cache = {'price': price, 'time': now}
    return price


def get_btc_regime():
    """
    Determine the current BTC market regime using 4H OHLCV data.
    Tries Bybit BTCUSDT first, falls back to Binance, then MEXC.
    Returns one of: 'STRONG_BULL', 'BULL', 'NEUTRAL', 'BEAR', 'STRONG_BEAR'.
    Caches the result for _btc_regime_cache_ttl seconds.
    """
    global _btc_regime_cache

    # Serve cache if still fresh
    if _btc_regime_cache is not None:
        age = (datetime.now() - _btc_regime_cache['time']).total_seconds()
        if age < _btc_regime_cache_ttl:
            return _btc_regime_cache['regime']

    df = None
    for fetch_fn, symbol in [
        (lambda s: bybit_fetch_ohlcv(s, '240', 150),   'BTCUSDT'),
        (lambda s: binance_fetch_ohlcv(s, '4h', 150),  'BTCUSDT'),
        (lambda s: mexc_fetch_ohlcv(s, '4h', 150),     'BTCUSDT'),
    ]:
        try:
            df = fetch_fn(symbol)
            if df is not None and len(df) >= 80:
                break
            df = None
        except Exception:
            df = None

    if df is None:
        logger.warning("BTC regime: could not fetch OHLCV — defaulting NEUTRAL")
        _btc_regime_cache = {'regime': 'NEUTRAL', 'time': datetime.now()}
        return 'NEUTRAL'

    df = add_indicators(df, timeframe='4h')
    if df is None or len(df) < 20:
        _btc_regime_cache = {'regime': 'NEUTRAL', 'time': datetime.now()}
        return 'NEUTRAL'

    # ── Factor 1 — EMA alignment ───────────────────────────────────────
    last  = df.iloc[-1]
    ema20 = last['ema20']
    ema50 = last['ema50']
    rsi   = last['rsi']
    bull_ema = ema20 > ema50
    bear_ema = ema20 < ema50

    # ── Factor 2 — ADX trend strength ─────────────────────────────────
    # ADX measures trend strength regardless of direction.
    # < 20: no trend (ranging/choppy)
    # 20-25: weak trend forming
    # > 25: established trend
    try:
        adx_ind = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        adx = float(adx_ind.adx().iloc[-1])
    except Exception:
        adx = 0.0
    trend_present  = adx > 20
    trend_strong   = adx > 25

    # ── Factor 3 — EMA gap slope (trend accelerating or dying?) ────────
    # Compare the EMA20-EMA50 gap now vs 3 candles ago.
    # Widening gap = trend strengthening. Narrowing = trend exhausting.
    try:
        gap_now  = abs(df['ema20'].iloc[-1] - df['ema50'].iloc[-1])
        gap_prev = abs(df['ema20'].iloc[-4] - df['ema50'].iloc[-4])
        gap_widening  = gap_now > gap_prev * 1.02   # gap grew by >2%
        gap_narrowing = gap_now < gap_prev * 0.98   # gap shrank by >2%
    except Exception:
        gap_widening = gap_narrowing = False

    # ── Factor 4 — EMA crossover recency ───────────────────────────────
    # Walk back up to 50 candles to find the last time EMA20 crossed EMA50.
    # fresh_cross  (< 8 candles): early in trend, higher false-signal risk
    # established  (8–40 candles): most reliable regime zone
    # old_cross   (> 40 candles): trend may be maturing / reversal risk
    cross_age = 99  # default: no cross found in lookback
    try:
        ema20_s = df['ema20']
        ema50_s = df['ema50']
        for i in range(1, min(51, len(df))):
            curr_above = ema20_s.iloc[-i]   > ema50_s.iloc[-i]
            prev_above = ema20_s.iloc[-i-1] > ema50_s.iloc[-i-1]
            if curr_above != prev_above:
                cross_age = i
                break
    except Exception:
        cross_age = 99
    fresh_cross      = cross_age <= 4    # very new — cautious
    established_cross = 4 < cross_age <= 40  # sweet spot
    # cross_age > 40: old trend, may be late

    # ── Factor 5 — RSI (tightened, non-overlapping bands) ──────────────
    # Old: BULL >45, BEAR <55 → overlap 45–55 = ambiguous 25% of the time.
    # New: BULL >52, BEAR <48 → clean gap, no overlap.
    rsi_bull = rsi > 52
    rsi_bear = rsi < 48
    rsi_neutral = not rsi_bull and not rsi_bear  # 48–52: genuinely ambiguous

    # ── Classify regime ────────────────────────────────────────────────
    if bull_ema:
        if trend_strong and gap_widening and rsi_bull:
            regime = 'STRONG_BULL'
        elif trend_present and rsi_bull and not gap_narrowing:
            regime = 'BULL'
        elif (fresh_cross or rsi_neutral or gap_narrowing or not trend_present):
            # EMA says bull but evidence is weak or mixed — could still reverse
            regime = 'NEUTRAL'
        else:
            regime = 'BULL'
    elif bear_ema:
        if trend_strong and gap_widening and rsi_bear:
            regime = 'STRONG_BEAR'
        elif trend_present and rsi_bear and not gap_narrowing:
            regime = 'BEAR'
        elif (fresh_cross or rsi_neutral or gap_narrowing or not trend_present):
            regime = 'NEUTRAL'
        else:
            regime = 'BEAR'
    else:
        regime = 'NEUTRAL'

    _btc_regime_cache = {'regime': regime, 'time': datetime.now()}
    logger.info(
        "BTC regime: %s (EMA20=%.0f EMA50=%.0f RSI=%.1f ADX=%.1f "
        "gap_trend=%s cross_age=%d candles)",
        regime, ema20, ema50, rsi, adx,
        'widening' if gap_widening else ('narrowing' if gap_narrowing else 'flat'),
        cross_age
    )
    return regime


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
def session_context() -> dict:
    """
    Returns the current trading session and a score modifier for the signal bias.

    Returns:
        {
          'session':  str    — 'ASIAN' | 'LONDON' | 'OVERLAP' | 'NY' | 'DEAD'
          'modifier': float  — quality bonus/penalty fed into ig_l/ig_s (+1 to -1)
          'note':     str    — human-readable context for signal card
          'bias_warning': bool — True if the session typically fakes in the signal direction
        }

    Score modifiers:
        OVERLAP  +1.0  — NY+London dual liquidity; real trend entries
        NY       +0.5  — strong continuation session
        LONDON   +0.25 — trend initiating but manipulation-wick risk
        ASIAN    -0.5  — consolidation; breakouts often reversed at London open
        DEAD     -1.0  — extremely thin; signals here have lowest follow-through
    """
    hour = datetime.utcnow().hour   # 0–23

    if 12 <= hour < 17:
        session, modifier = 'OVERLAP', 1.0
        note = "NY+London overlap — highest real liquidity window ✅"
        bias_warning = False
    elif 13 <= hour < 21:
        session, modifier = 'NY', 0.5
        note = "NY session — trend continuation favoured ✅"
        bias_warning = False
    elif 7 <= hour < 12:
        session, modifier = 'LONDON', 0.25
        note = "London open — trend initiation, watch for manipulation wicks ⚠️"
        bias_warning = True   # London open wicks can flush before trend direction
    elif 0 <= hour < 8:
        session, modifier = 'ASIAN', -0.5
        note = "Asian session — low volatility, breakouts frequently reversed at London open ⚠️"
        bias_warning = True
    else:   # 21–23
        session, modifier = 'DEAD', -1.0
        note = "Dead zone (21–00 UTC) — thin market, low follow-through ❌"
        bias_warning = True

    return {
        'session':      session,
        'modifier':     modifier,
        'note':         note,
        'bias_warning': bias_warning,
    }


# ─────────────────────────────────────────────
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

def fib_confluence_score(df, price: float, support: float, resistance: float,
                          atr: float, bias: str) -> tuple:
    """
    FIX #FIB — Fibonacci Retracement Confluence

    Finds the last significant swing high and swing low in the lookback window,
    computes the standard Fibonacci retracement levels between them, then checks
    whether the pivot S/R level used by score_pair coincides with any Fib level.

    A confluence is defined as: |pivot_level - fib_level| / price < tolerance.

    Returns (score: int, fib_levels: list[float], confluence_note: str)
        score          — 0 (no confluence) or +1 (confluence) or +2 (tight confluence)
        fib_levels     — list of computed fib prices for signal card display
        confluence_note — human-readable description for reasons list
    """
    _FIB_TOLERANCE = 0.005   # 0.5% — tighter than ATR-based check; genuinely confluent

    try:
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n      = len(highs)

        if n < 20:
            return 0, [], ""

        # Look back over last 60 bars for the significant swing high and low.
        # Use a 3-bar swing definition consistent with find_pivot_support/resistance.
        lookback   = min(60, n - 1)
        swing      = 3
        start      = max(swing, n - lookback)

        swing_highs = []
        swing_lows  = []
        for i in range(start, n - swing):
            if all(highs[i] >= highs[i - j] for j in range(1, swing + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, swing + 1)):
                swing_highs.append((i, highs[i]))
            if all(lows[i] <= lows[i - j] for j in range(1, swing + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, swing + 1)):
                swing_lows.append((i, lows[i]))

        if not swing_highs and not swing_lows:
            return 0, [], ""

        # Use the most recent significant swing high and swing low.
        # Fall back to rolling window extremes when a swing side has no pivots
        # (e.g. strong trending market with no opposing swing).
        if swing_highs:
            sh_idx, sh_price = max(swing_highs, key=lambda x: x[1])
        else:
            sh_idx  = n - 1
            sh_price = float(df['high'].iloc[max(0, n - lookback):].max())

        if swing_lows:
            sl_idx, sl_price = min(swing_lows,  key=lambda x: x[1])
        else:
            sl_idx  = n - 1
            sl_price = float(df['low'].iloc[max(0, n - lookback):].min())

        swing_range = sh_price - sl_price
        if swing_range < atr * 0.5:
            # Swing range too small to be meaningful — skip
            return 0, [], ""

        # Compute Fibonacci levels (retracement from the swing extremes)
        # Standard retracements measured from the bottom of the move (sl_price)
        fib_levels = {
            ratio: sl_price + ratio * swing_range
            for ratio in _FIB_RATIOS
        }

        # Check pivot level against each fib level
        # For LONG: relevant pivot is support (floor), should be near 0.382/0.5/0.618
        # For SHORT: relevant pivot is resistance (ceiling), should be near 0.236/0.382/0.5
        pivot_level  = support  if bias == 'LONG' else resistance
        fib_priority = (
            [0.618, 0.500, 0.382, 0.786, 0.236] if bias == 'LONG'
            else [0.236, 0.382, 0.500, 0.618, 0.786]
        )

        best_score = 0
        best_note  = ""
        for ratio in fib_priority:
            fib_price = fib_levels[ratio]
            deviation = abs(pivot_level - fib_price) / price if price > 0 else 1.0
            if deviation < _FIB_TOLERANCE / 2:          # < 0.25% — very tight
                best_score = 2
                best_note  = (f"🎯 Fib {ratio:.3f} confluence ({deviation*100:.2f}% deviation) "
                              f"at ${fib_price:.4f} — high-probability structural zone")
                break
            elif deviation < _FIB_TOLERANCE:             # < 0.5% — confluent
                if best_score < 1:
                    best_score = 1
                    best_note  = (f"Fib {ratio:.3f} near pivot ({deviation*100:.2f}% deviation) "
                                  f"at ${fib_price:.4f} — structural confluence")

        return best_score, list(fib_levels.values()), best_note

    except Exception as _fib_e:
        logger.debug("fib_confluence_score error: %s", _fib_e)
        return 0, [], ""


# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────
def candle_quality_score(candle, bias: str) -> tuple:
    """
    FIX #CQ — Candle Quality Scoring
    ───��─────────────────────────────
    Scores the signal candle on three independent structural properties.
    A doji and a full-body engulfing candle are NOT the same signal —
    this function quantifies that difference and feeds it into confidence.

    Returns (score: float, label: str, detail: str)

    score > 0.0  → quality bonus  (max +1.0 fed to quality_bonus)
    score < 0.0  → quality penalty (min -1.0 applied as confidence demotion)
    score = 0.0  → neutral / indeterminate candle

    Three sub-scores (each –1 to +1), averaged then clamped to [–1, +1]:

    1. BODY RATIO  — body / total range
       A candle whose body fills > 60 % of its range committed to a direction.
       A body < 25 % (doji zone) is structural indecision.

    2. CLOSE POSITION — where close sits in the range [0 = low, 1 = high]
       For LONG:  close near the top (> 0.65) is bullish conviction.
                  close near the bottom (< 0.35) negates the signal.
       For SHORT: mirrored.

    3. WICK ASYMMETRY — upper_wick vs lower_wick ratio in the signal direction
       For LONG:  lower wick > upper wick → rejection of lows, bullish.
                  upper wick > 2× lower wick → sellers absorbed the move.
       For SHORT: upper wick > lower wick → rejection of highs, bearish.
    """
    try:
        o, h, l, c = float(candle['open']), float(candle['high']), float(candle['low']), float(candle['close'])
        total_range = h - l
        if total_range < 1e-12:
            return 0.0, "NEUTRAL", "zero-range candle — skip"

        body       = abs(c - o)
        body_ratio = body / total_range                          # [0, 1]

        close_pos  = (c - l) / total_range                      # [0, 1]; 0=closed at low, 1=at high

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        wick_sum   = upper_wick + lower_wick

        # ── sub-score 1: body ratio ─────────────────────────────────────
        if body_ratio >= 0.70:     body_sub =  1.0
        elif body_ratio >= 0.50:   body_sub =  0.5
        elif body_ratio >= 0.35:   body_sub =  0.0
        elif body_ratio >= 0.20:   body_sub = -0.5
        else:                      body_sub = -1.0   # doji

        # Doji override: body < 10% is structural indecision — close position and
        # wick asymmetry are meaningless noise on a candle with no body conviction.
        # Short-circuit and return POOR immediately.
        if body_ratio < 0.10:
            return -0.80, "POOR", (f"Doji / pin bar — body={body_ratio:.0%}, no directional conviction")

        # ── sub-score 2: close position (direction-aware) ────────────────
        if bias == 'LONG':
            if close_pos >= 0.70:    close_sub =  1.0
            elif close_pos >= 0.50:  close_sub =  0.5
            elif close_pos >= 0.35:  close_sub =  0.0
            elif close_pos >= 0.20:  close_sub = -0.5
            else:                    close_sub = -1.0
        else:  # SHORT
            if close_pos <= 0.30:    close_sub =  1.0
            elif close_pos <= 0.50:  close_sub =  0.5
            elif close_pos <= 0.65:  close_sub =  0.0
            elif close_pos <= 0.80:  close_sub = -0.5
            else:                    close_sub = -1.0

        # ── sub-score 3: wick asymmetry (direction-aware) ───────────────
        if wick_sum < 1e-12:
            wick_sub = 0.0          # no wicks at all — body-only candle, neutral here
        elif bias == 'LONG':
            # Bullish: lower wick absorbs selling, upper wick is healthy extension
            # Penalise: large upper wick means the rally was rejected hard
            ratio = lower_wick / (upper_wick + 1e-12)
            if ratio >= 2.0:       wick_sub =  1.0   # strong lower wick / rejection
            elif ratio >= 1.0:     wick_sub =  0.5
            elif ratio >= 0.5:     wick_sub =  0.0
            elif ratio >= 0.25:    wick_sub = -0.5
            else:                  wick_sub = -1.0   # upper wick >> lower wick, bullish move rejected
        else:  # SHORT
            ratio = upper_wick / (lower_wick + 1e-12)
            if ratio >= 2.0:       wick_sub =  1.0
            elif ratio >= 1.0:     wick_sub =  0.5
            elif ratio >= 0.5:     wick_sub =  0.0
            elif ratio >= 0.25:    wick_sub = -0.5
            else:                  wick_sub = -1.0

        raw   = (body_sub + close_sub + wick_sub) / 3.0
        score = max(-1.0, min(1.0, raw))

        # Human-readable label and detail
        if score >= 0.55:
            label  = "STRONG"
            detail = (f"body={body_ratio:.0%}, close_pos={close_pos:.0%}, "
                      f"body_sub={body_sub:+.1f} close_sub={close_sub:+.1f} wick_sub={wick_sub:+.1f}")
        elif score >= 0.15:
            label  = "GOOD"
            detail = (f"body={body_ratio:.0%}, close_pos={close_pos:.0%}")
        elif score >= -0.15:
            label  = "NEUTRAL"
            detail = (f"body={body_ratio:.0%} — mid-range close or equal wicks")
        elif score >= -0.55:
            label  = "WEAK"
            detail = (f"body={body_ratio:.0%}, close_pos={close_pos:.0%} — indecisive candle")
        else:
            label  = "POOR"
            detail = (f"body={body_ratio:.0%} doji/spinning top, close_pos={close_pos:.0%} — "
                      "signal candle shows no directional conviction")

        return score, label, detail

    except Exception as _cq_e:
        logger.debug("candle_quality_score error: %s", _cq_e)
        return 0.0, "NEUTRAL", "calc error"


def score_pair(df4h, df1d, funding, symbol, user_requested: bool = False):
    """
    Scoring engine — all bias/accuracy fixes applied:
    FIX #1  — Confidence formula: pure ratio-based, no raw-score inflation
    FIX #3  — Daily reads use iloc[-2] (last CLOSED candle, not forming one)
    FIX #4  — MACD crossover confirmed over 2 bars + histogram growth required
    FIX #6  — Funding thresholds raised to meaningful levels (±0.03% / ±0.07%)
    FIX #7  — T1/T2 capped against nearest pivot S/R level
    FIX #8  — Counter-trend veto: opposing daily EMA raises qualification bar
    FIX #RG — BTC market regime gate: suppresses contra-regime signals below conf 9
    FIX #MS — Minimum absolute score gate: downgrade conf when winning score too thin
               (prevents RSI alone from producing a 10/10 signal on 4:0 ratio)
    FIX #EZ — Entry zone freshness flag: tags signal if price was outside zone at
               scan time so format_signal can warn the user before they enter
    FIX #VA — Volatility-aware targets: T1/T2/T3 multipliers and entry/SL width
               scale with ATR% regime (RANGING/LOW/MEDIUM/HIGH/EXTREME) so targets
               are realistic for both stagnant and explosive coins
    FIX #TP — Guaranteed T1 < T2 < T3 ordering (LONG) / T1 > T2 > T3 (SHORT)
               via sequential clamping — prior approach could produce inversions
               when pivot resistance cap pushed T1 past T2_raw
    FIX #BB — BB squeeze breakout scoring: bandwidth expanding from 20-bar low
               adds +2 score in the breakout direction — consolidation breakouts
               are high-probability entries not captured by RSI/MACD alone
    FIX #OI — Outcome resolver: same-candle SL+target now correctly preserves
               prior confirmed target as sl_after_target=1 instead of always
               overwriting with sl_hit, eliminating a systematic win undercount
    FIX #SR — Scan sort tiebreaker: confidence+score ties resolved by vol regime
               (MEDIUM > HIGH > LOW > EXTREME > RANGING) for better signal ordering
    FIX #CQ — Candle Quality Scoring: signal candle scored on body ratio, close
               position within range, and wick asymmetry. STRONG/GOOD candles get
               a confidence bonus (max +1); WEAK/POOR candles get a penalty (−1
               or −2) that can drop confidence below the gate and suppress the
               signal entirely.  Filters out dojis and spinning tops that the
               existing logic accepted as valid breakout candles.
    FIX #SESSION — Session Awareness: UTC session tagged and scored. OVERLAP and NY
               sessions add to ig (uncapped). ASIAN and DEAD sessions penalise ig —
               signals in thin/consolidation windows require the rest of the stack
               to be stronger to compensate.  Session note surfaced in signal card.
    FIX #FIB — Fibonacci Confluence: 0.236/0.382/0.5/0.618/0.786 Fibonacci
               retracement levels computed from last major swing H/L in the 60-bar
               lookback. When pivot S/R coincides with a Fib level within 0.5%,
               the zone scores +1 to vg (structure); within 0.25% scores +2.
               Displayed in signal card details. Purely additive — never penalises.
    """
    try:
        # Warning tags — set below if signal passes with caveats.
        # Always initialized so score_pair always attaches them to result.
        low_conf_warning = None
        regime_warning   = None
        btc_regime       = 'UNKNOWN'

        L   = df4h.iloc[-1]   # current closed 4H candle
        P   = df4h.iloc[-2]   # previous 4H candle
        P2  = df4h.iloc[-3]   # two 4H candles ago (2-bar MACD confirmation)

        # FIX #3 — use the last CLOSED daily candle, never the forming one
        # iloc[-1] is still forming mid-day; its MACD/RSI values are noise.
        LD  = df1d.iloc[-2]   # last closed daily candle
        PD  = df1d.iloc[-3]   # previous closed daily candle

        price      = L['close']
        rsi4       = L['rsi']
        macd_diff  = L['macd_diff']; prev_diff  = P['macd_diff']; prev2_diff = P2['macd_diff']
        bb_upper   = L['bb_upper'];  bb_lower   = L['bb_lower']
        ema20      = L['ema20'];     ema50      = L['ema50']
        support    = float(L['support']); resistance = float(L['resistance'])
        atr        = L['atr']
        vol        = L['volume'];    vol_ma     = L['volume_ma']
        stoch_k    = L['stoch_k'];   stoch_d    = L['stoch_d']

        # FIX #3 — daily indicators from the closed candle
        rsi_d   = LD['rsi']
        macd_d  = LD['macd_diff']; prev_d = PD['macd_diff']
        ema20_d = LD['ema20'];     ema50_d = LD['ema50']

        # ── FIX #COR / FIX #SI — Grouped scoring accumulators ─────────
        # Six independent buckets instead of four, addressing ceiling #1:
        # the old mg (momentum) and tg (trend) buckets each contained two
        # sub-families that fire simultaneously from the same price event.
        #
        # SPLIT: momentum → osc_g + mtf_g
        #   osc_g = Oscillator group  : RSI 4H + Stochastic
        #           Both react to 4H price, but via different formulas.
        #           Cap = 3 so max contribution from "price is low" = 3.
        #   mtf_g = Multi-TF momentum : RSI Daily only
        #           Daily RSI is structurally the same formula as 4H RSI —
        #           they share the same underlying price stream, so they are
        #           NOT independent.  Treated as a cross-timeframe confirmation
        #           multiplier (max +2) rather than an additive peer.
        #
        # SPLIT: trend → cross_g + pos_g
        #   cross_g = Crossover group : MACD 4H + MACD Daily
        #             MACD is derived from price, but a *crossover event* on
        #             two different timeframes is two different signals — a
        #             4H cross is a 16-bar event; a daily cross is a 365-bar
        #             event.  Cap = 3 so a single timeframe MACD cross tops out.
        #   pos_g   = Position group  : EMA 4H + EMA Daily
        #             "Price above EMA" is structural positioning, not momentum.
        #             Cap = 3.  Aligns with cross_g but is a separate question
        #             (where you are vs where you're going).
        #
        #   vg = Structure group: BB band, BB squeeze, Pivot S/R  (cap 4)
        #   ig = Independent    : Volume, Funding, CLV (uncapped)
        #
        # Net effect: a single strong 4H candle can max osc_g (cap 3) OR
        # cross_g (cap 3), but not both to their old combined cap of 4+5=9.
        # Reaching high total score now genuinely requires 4+ distinct signals.
        osc_g_l,  osc_g_s  = 0, 0   # oscillators  (RSI 4H, Stochastic)
        mtf_g_l,  mtf_g_s  = 0, 0   # cross-TF     (RSI Daily)
        cross_g_l, cross_g_s = 0, 0 # MACD crossovers (4H + Daily)
        pos_g_l,  pos_g_s  = 0, 0   # EMA position  (4H + Daily)
        vg_l, vg_s = 0, 0
        ig_l, ig_s = 0, 0
        lr, sr = [], []

        # ── RSI 4H → oscillator bucket ─────────────────────────────────
        if rsi4 < 30:    osc_g_l+=3; lr.append(f"RSI 4H extremely oversold ({rsi4:.1f})")
        elif rsi4 < 40:  osc_g_l+=2; lr.append(f"RSI 4H oversold ({rsi4:.1f})")
        elif rsi4 < 48:  osc_g_l+=1; lr.append(f"RSI 4H leaning oversold ({rsi4:.1f})")
        elif rsi4 > 70:  osc_g_s+=3; sr.append(f"RSI 4H extremely overbought ({rsi4:.1f})")
        elif rsi4 > 60:  osc_g_s+=2; sr.append(f"RSI 4H overbought ({rsi4:.1f})")
        elif rsi4 > 52:  osc_g_s+=1; sr.append(f"RSI 4H leaning overbought ({rsi4:.1f})")
        rsi4_prev = P['rsi'] if 'rsi' in P.index else rsi4
        if 48 <= rsi4 <= 52:
            if rsi4 > rsi4_prev:   osc_g_l+=1; lr.append(f"RSI 4H crossing 50 upward — bullish momentum ({rsi4:.1f})")
            elif rsi4 < rsi4_prev: osc_g_s+=1; sr.append(f"RSI 4H crossing 50 downward — bearish momentum ({rsi4:.1f})")

        # Daily RSI → cross-TF confirmation bucket (structurally same formula,
        # different timeframe — treated as confirming multiplier, cap=2)
        if rsi_d < 35:   mtf_g_l+=2; lr.append(f"RSI Daily oversold ({rsi_d:.1f}) [closed candle]")
        elif rsi_d < 45: mtf_g_l+=1; lr.append(f"RSI Daily weak ({rsi_d:.1f}) [closed candle]")
        elif rsi_d > 65: mtf_g_s+=2; sr.append(f"RSI Daily overbought ({rsi_d:.1f}) [closed candle]")
        elif rsi_d > 55: mtf_g_s+=1; sr.append(f"RSI Daily elevated ({rsi_d:.1f}) [closed candle]")

        # ── FIX #4 — MACD → crossover bucket ──────────────────────────
        # MACD crossovers go to cross_g (cap=3). Sustained momentum without
        # a fresh cross goes to cross_g at reduced weight (+1) — still a
        # crossover-family signal but weaker evidence.
        macd_cross_bull_4h = (macd_diff > 0 and prev_diff <= 0 and
                               abs(macd_diff) > abs(prev_diff))
        macd_cross_bear_4h = (macd_diff < 0 and prev_diff >= 0 and
                               abs(macd_diff) > abs(prev_diff))
        macd_sust_bull_4h  = (macd_diff > 0 and prev_diff > 0 and
                               prev2_diff <= 0 and macd_diff > prev_diff)
        macd_sust_bear_4h  = (macd_diff < 0 and prev_diff < 0 and
                               prev2_diff >= 0 and macd_diff < prev_diff)

        if macd_sust_bull_4h:
            cross_g_l+=3; lr.append("MACD confirmed bullish crossover 4H ⚡ (2-candle hold)")
        elif macd_cross_bull_4h:
            cross_g_l+=2; lr.append("MACD tentative bullish crossover 4H (1 candle — unconfirmed)")
        elif macd_sust_bear_4h:
            cross_g_s+=3; sr.append("MACD confirmed bearish crossover 4H ⚡ (2-candle hold)")
        elif macd_cross_bear_4h:
            cross_g_s+=2; sr.append("MACD tentative bearish crossover 4H (1 candle — unconfirmed)")
        elif macd_diff > 0:
            cross_g_l+=1; lr.append("MACD bullish momentum 4H")
        elif macd_diff < 0:
            cross_g_s+=1; sr.append("MACD bearish momentum 4H")

        # FIX #4 + FIX #3 — daily MACD from closed candle, confirmed crossover
        macd_cross_bull_1d = (macd_d > 0 and prev_d <= 0 and abs(macd_d) > abs(prev_d))
        macd_cross_bear_1d = (macd_d < 0 and prev_d >= 0 and abs(macd_d) > abs(prev_d))

        if macd_cross_bull_1d:
            cross_g_l+=3; lr.append("MACD confirmed bullish crossover Daily ⚡ [closed candle]")
        elif macd_cross_bear_1d:
            cross_g_s+=3; sr.append("MACD confirmed bearish crossover Daily ⚡ [closed candle]")
        elif macd_d > 0:
            cross_g_l+=1; lr.append("MACD bullish Daily [closed candle]")
        elif macd_d < 0:
            cross_g_s+=1; sr.append("MACD bearish Daily [closed candle]")

        # ── EMA → position bucket ───────────────────────────────��───��───
        if price > ema20 > ema50:   pos_g_l+=2; lr.append("Bullish EMA stack 4H")
        elif price < ema20 < ema50: pos_g_s+=2; sr.append("Bearish EMA stack 4H")
        elif price > ema20:         pos_g_l+=1; lr.append("Price above EMA20 4H")
        elif price < ema20:         pos_g_s+=1; sr.append("Price below EMA20 4H")

        # Daily EMA from closed candle (FIX #3) — same pos_g bucket
        if price > ema20_d > ema50_d:   pos_g_l+=2; lr.append("Bullish EMA stack Daily [closed]")
        elif price < ema20_d < ema50_d: pos_g_s+=2; sr.append("Bearish EMA stack Daily [closed]")

        # ── Bollinger Bands ────────────────────────────────────────────
        if price <= bb_lower:   vg_l+=2; lr.append("Price at/below lower Bollinger Band")
        elif price >= bb_upper: vg_s+=2; sr.append("Price at/above upper Bollinger Band")

        # FIX #BB — BB squeeze breakout
        bb_bw     = L.get('bb_bw',     None) if hasattr(L, 'get') else L['bb_bw']     if 'bb_bw'     in L.index else None
        bb_bw_min = L.get('bb_bw_min', None) if hasattr(L, 'get') else L['bb_bw_min'] if 'bb_bw_min' in L.index else None
        if bb_bw is not None and bb_bw_min is not None and bb_bw_min > 0:
            squeeze_expanding = (bb_bw > bb_bw_min * 1.05)
            if squeeze_expanding:
                if price > P['close']:  vg_l+=2; lr.append(f"BB squeeze breakout BULLISH (bw expanding from floor)")
                else:                   vg_s+=2; sr.append(f"BB squeeze breakout BEARISH (bw expanding from floor)")

        # ── Stochastic → oscillator bucket (same osc_g as RSI 4H) ────────
        # Stochastic and RSI both measure price momentum on the 4H timeframe.
        # Both fire on the same candle, so they share osc_g's cap.
        if stoch_k < 20 and stoch_d < 20:   osc_g_l+=2; lr.append(f"Stochastic oversold (K:{stoch_k:.1f})")
        elif stoch_k > 80 and stoch_d > 80: osc_g_s+=2; sr.append(f"Stochastic overbought (K:{stoch_k:.1f})")
        if stoch_k > stoch_d and stoch_k < 45:   osc_g_l+=1; lr.append(f"Stochastic bullish cross low zone (K:{stoch_k:.1f})")
        elif stoch_k < stoch_d and stoch_k > 55: osc_g_s+=1; sr.append(f"Stochastic bearish cross high zone (K:{stoch_k:.1f})")

        # ── FIX #DIV — RSI/MACD Divergence Detection ──────────────────
        # Divergence is independent of the oscillator level (RSI<30 and
        # RSI-divergence are different signals from the same indicator family
        # but carry different information — level vs structural reversal).
        # Placed in vg (structure group) since divergence is a structural
        # break in indicator behaviour, not a momentum threshold crossing.
        _rsi_bull_div  = bool(L.get('rsi_bull_div',  False) if hasattr(L, 'get') else (L['rsi_bull_div']  if 'rsi_bull_div'  in L.index else False))
        _rsi_bear_div  = bool(L.get('rsi_bear_div',  False) if hasattr(L, 'get') else (L['rsi_bear_div']  if 'rsi_bear_div'  in L.index else False))
        _macd_bull_div = bool(L.get('macd_bull_div', False) if hasattr(L, 'get') else (L['macd_bull_div'] if 'macd_bull_div' in L.index else False))
        _macd_bear_div = bool(L.get('macd_bear_div', False) if hasattr(L, 'get') else (L['macd_bear_div'] if 'macd_bear_div' in L.index else False))

        if _rsi_bull_div:
            vg_l += 2; lr.append(f"🔀 Bullish RSI divergence — price lower low, RSI higher low (reversal signal)")
        if _macd_bull_div:
            vg_l += 2; lr.append(f"🔀 Bullish MACD divergence — price lower low, MACD histogram recovering")
        if _rsi_bear_div:
            vg_s += 2; sr.append(f"🔀 Bearish RSI divergence — price higher high, RSI lower high (exhaustion)")
        if _macd_bear_div:
            vg_s += 2; sr.append(f"🔀 Bearish MACD divergence — price higher high, MACD histogram weakening")

        # ── Volume (vol_ma already shifted by 1 — FIX #5) ─────────────
        if vol_ma and vol_ma > 0:
            ratio = vol / vol_ma
            if ratio > 1.8:
                if price > P['close']: ig_l+=2; lr.append(f"Strong volume bullish candle ({ratio:.1f}x avg)")
                else:                  ig_s+=2; sr.append(f"Strong volume bearish candle ({ratio:.1f}x avg)")
            elif ratio > 1.3:
                if price > P['close']: ig_l+=1; lr.append(f"Above avg volume bullish ({ratio:.1f}x)")
                else:                  ig_s+=1; sr.append(f"Above avg volume bearish ({ratio:.1f}x)")

        # ── Pivot S/R proximity (FIX #2 — real pivot levels) ──────────
        if atr > 0:
            if (price - support) < atr * 0.5:    vg_l+=2; lr.append(f"Near pivot support (${support:.4f})")
            if (resistance - price) < atr * 0.5: vg_s+=2; sr.append(f"Near pivot resistance (${resistance:.4f})")

        # ── FIX #FIB — Fibonacci retracement confluence ────────────────
        # Compute for BOTH directions now; apply the directional score to
        # vg_l / vg_s, and store levels for the result dict regardless.
        # Purely additive — never penalises, never blocks.
        _fib_l_score, _fib_levels, _fib_l_note = fib_confluence_score(df4h, price, support, resistance, atr, 'LONG')
        _fib_s_score, _fib_levels_s, _fib_s_note = fib_confluence_score(df4h, price, support, resistance, atr, 'SHORT')
        if _fib_l_score > 0 and _fib_l_note:
            vg_l += _fib_l_score; lr.append(_fib_l_note)
        if _fib_s_score > 0 and _fib_s_note:
            vg_s += _fib_s_score; sr.append(_fib_s_note)
        # Use LONG levels for display (they span the full swing range — same prices)
        _fib_display_levels = _fib_levels if _fib_levels else _fib_levels_s

        # ── FIX #6 — Funding thresholds at meaningful levels ───────────
        if funding != 0:
            fp = funding * 100
            if funding < -0.0007:   ig_l+=3; lr.append(f"Extreme negative funding ({fp:.4f}%) — strong short squeeze risk")
            elif funding < -0.0003: ig_l+=2; lr.append(f"Negative funding ({fp:.4f}%) — shorts paying longs")
            elif funding < 0:       ig_l+=1; lr.append(f"Slightly negative funding ({fp:.4f}%)")
            elif funding > 0.0007:  ig_s+=3; sr.append(f"Extreme positive funding ({fp:.4f}%) — strong long squeeze risk")
            elif funding > 0.0003:  ig_s+=2; sr.append(f"Positive funding ({fp:.4f}%) — longs paying shorts")
            elif funding > 0:       ig_s+=1; sr.append(f"Slightly positive funding ({fp:.4f}%)")

        # ── FIX #COR — Close Location Value (CLV): independent pressure signal ─
        # Where the candle closed within its own range is genuinely uncorrelated
        # with RSI/MACD/EMA because those are derived from close prices across
        # many bars, while CLV measures buyer/seller dominance within a single
        # candle's range.  A sustained rising CLV (5-bar avg) while price falls
        # is real absorption evidence — a different data source entirely.
        clv    = L.get('clv',    None) if hasattr(L, 'get') else (L['clv']    if 'clv'    in L.index else None)
        clv_ma = L.get('clv_ma', None) if hasattr(L, 'get') else (L['clv_ma'] if 'clv_ma' in L.index else None)
        if clv is not None and clv_ma is not None and not (clv != clv):   # not NaN
            if clv > 0.15 and clv > clv_ma:
                ig_l+=2; lr.append(f"CLV bullish: buyers absorbing within candle range ({clv:+.2f})")
            elif clv < -0.15 and clv < clv_ma:
                ig_s+=2; sr.append(f"CLV bearish: sellers dominating within candle range ({clv:+.2f})")

        # ── CONVICTION LAYER — CVD / OI / VWAP ─────────────────────────
        # Three genuinely independent data sources that are NOT correlated
        # with the existing RSI/MACD/EMA/BB buckets.  They go into ig (the
        # uncapped independent group) so their contribution is bounded only
        # by the quality of evidence, not by an artificial cap.
        #
        # Called AFTER bias is decided (CVD/OI/VWAP scoring is directional).
        # Called BEFORE the final ig sum so the points are included in ls/ss.
        #
        # At this point `bias` is already determined from the ls/ss comparison
        # above, but ig_l and ig_s are still being accumulated — the final
        # sum happens below at "Apply group caps, then sum".
        _conv = {}
        if _CONVICTION_AVAILABLE:
            try:
                _binance_ok = sakz_exchanges.BINANCE_AVAILABLE if sakz_exchanges.BINANCE_AVAILABLE is not None else False
                _conv = conviction_scores(symbol, df4h, bias, _binance_ok)
                ig_l += _conv.get('ig_long_bonus',  0)
                ig_s += _conv.get('ig_short_bonus', 0)
                # Surface conviction notes in the relevant reason list
                for _cn in _conv.get('display_lines', []):
                    if bias == 'LONG':
                        lr.append(_cn)
                    else:
                        sr.append(_cn)
            except Exception as _conv_e:
                logger.debug("conviction_scores error for %s: %s", symbol, _conv_e)

        # ── FIX #SESSION — Session awareness ───────────────────────────
        # Session modifier goes into ig (uncapped independent group).
        # Positive: OVERLAP (+1) and NY (+0.5) — add in the signal direction.
        # Negative: ASIAN (-0.5) and DEAD (-1) — subtract from the signal direction.
        #   (Penalising the direction bucket reduces winning score, which can widen
        #    or narrow the gap, eventually hitting the GAP_BLOCK or LOW_CONF gates.)
        # LONDON (+0.25) adds a small bonus but is flagged as manipulation-wick risk.
        _sess = session_context()
        _sess_mod = _sess['modifier']
        if _sess_mod > 0:
            ig_l += _sess_mod; lr.append(f"📅 Session: {_sess['note']}")
            ig_s += _sess_mod; sr.append(f"📅 Session: {_sess['note']}")
        elif _sess_mod < 0:
            # Penalise by reducing the winning direction's ig score
            # (floor at 0 — never drive ig negative from session alone)
            ig_l = max(0.0, ig_l + _sess_mod)
            ig_s = max(0.0, ig_s + _sess_mod)
            lr.append(f"📅 Session: {_sess['note']}")
            sr.append(f"📅 Session: {_sess['note']}")

        # ── FIX #COR / FIX #SI — Apply group caps, then sum ───────────
        # Six buckets, tighter caps per bucket:
        #   osc_g   cap=3  : RSI 4H + Stochastic (same 4H price, different formula)
        #   mtf_g   cap=2  : RSI Daily (same formula, different TF — confirmation only)
        #   cross_g cap=3  : MACD crossovers 4H + Daily (event-based, two TFs can stack)
        #   pos_g   cap=3  : EMA position 4H + Daily (structural, two TFs can stack)
        #   vg      cap=4  : BB + S/R structure (unchanged)
        #   ig      uncapped: Volume, Funding, CLV (genuinely different sources)
        #
        # Old max from momentum-derivatives alone: mg_cap(4) + tg_cap(5) = 9 points
        # New max from momentum-derivatives alone: osc(3) + mtf(2) + cross(3) + pos(3) = 11 raw
        # but only a market that has BOTH a fresh MACD cross AND EMA alignment AND
        # oversold RSI AND daily confirmation can approach that — four distinct facts,
        # not one strong candle firing all oscillators simultaneously.
        OSC_CAP   = 3   # RSI 4H + Stochastic
        MTF_CAP   = 2   # RSI Daily (confirmation multiplier)
        CROSS_CAP = 3   # MACD crossovers (4H + Daily)
        POS_CAP   = 3   # EMA position (4H + Daily)
        STRUCTURE_CAP = 4   # BB + S/R (unchanged)

        ls = (min(osc_g_l,  OSC_CAP)   + min(mtf_g_l,  MTF_CAP) +
              min(cross_g_l, CROSS_CAP) + min(pos_g_l,  POS_CAP) +
              min(vg_l, STRUCTURE_CAP)  + ig_l)
        ss = (min(osc_g_s,  OSC_CAP)   + min(mtf_g_s,  MTF_CAP) +
              min(cross_g_s, CROSS_CAP) + min(pos_g_s,  POS_CAP) +
              min(vg_s, STRUCTURE_CAP)  + ig_s)

        # ── Bias determination ─────────────────────────────────────────
        # NEUTRAL / very-weak evidence handling.
        #   • Auto-scan (user_requested=False): drop directionless noise so the
        #     alert feed only carries real signals.
        #   • Manual /scan or /chart (user_requested=True): NEVER drop. The user
        #     explicitly asked about this pair, so always return a best-effort
        #     read. The gap / counter-trend / regime / abs-score gates are all
        #     soft for user_requested, so it surfaces as a low-confidence
        #     "weak signal" (e.g. 2/10-4/10) with warnings instead of nothing.
        if ls == ss or (ls < 3 and ss < 3):
            if not user_requested:
                return None
            logger.debug("WEAK SOFT-WARN (user_requested): %s ls=%d ss=%d - "
                         "showing weak signal instead of dropping", symbol, ls, ss)
        if ls >= ss:
            bias, score, reasons = "LONG",  ls, lr
            winning, losing      = ls, ss
        else:
            bias, score, reasons = "SHORT", ss, sr
            winning, losing      = ss, ls

        # ── FIX #GAP — Minimum score gap ───────────────────────────────
        # Winning side must beat losing side by at least 2 points.
        # A 1-point edge (e.g. 5 vs 4) in a ranging market is noise —
        # the next candle can easily flip it, producing alternating signals.
        # When user_requested=True, convert to soft warning so analysis shows.
        if (winning - losing) < 2:
            gap_detail = f"gap={winning-losing} (winning={winning} losing={losing})"
            if user_requested:
                logger.debug("GAP SOFT-WARN (user_requested): %s %s %s", symbol, bias, gap_detail)
                ig_l.append(f"⚠️ Narrow gap: {gap_detail} — choppy market, trade carefully")
            else:
                logger.debug("GAP BLOCK: %s %s gap=%d (winning=%d losing=%d) — too close",
                             symbol, bias, winning - losing, winning, losing)
                return ScanFailure(REASON_GAP_BLOCK, detail=gap_detail)

        # ── FIX #FLIP — Direction flip cooldown ────────────────────────
        # If the last fired signal for this symbol was the OPPOSITE direction
        # within the cooldown window, require confidence ≥ _FLIP_MIN_CONF
        # to override it.  Prevents rapid SHORT→LONG→SHORT churn on auto-scan.
        #
        # When user_requested=True (manual /scan SYMBOL), the hard block is
        # converted to a soft warning so the full analysis is always shown.
        # The regime gate never fired for the user because the flip block was
        # returning before it — this is why it appeared the ADX gate was
        # "too strong".  It wasn't — it never ran.
        _FLIP_WINDOW_H = 6          # was 8; 6h = 1.5 × 4H candles
        _FLIP_MIN_CONF = 9
        last = _last_signal_bias.get(symbol)
        if last and last['bias'] != bias:
            hours_since = (datetime.now() - last['time']).total_seconds() / 3600
            if hours_since < _FLIP_WINDOW_H:
                tentative_ratio = (winning / (winning + losing)) * 10
                if round(tentative_ratio) < _FLIP_MIN_CONF:
                    flip_detail = (
                        f"tried {last['bias']}→{bias} after {hours_since:.1f}h, "
                        f"conf≈{round(tentative_ratio)} (need {_FLIP_MIN_CONF} to auto-flip)"
                    )
                    if user_requested:
                        # ── SOFT WARN: show analysis anyway for manual queries ──
                        # The user explicitly asked — they want the current picture.
                        logger.debug("FLIP SOFT-WARN (user_requested): %s %s", symbol, flip_detail)
                        ig_l.append(f"⚠️ Flip cooldown: {flip_detail} — showing analysis anyway")
                    else:
                        # ── HARD BLOCK: auto-scan / background jobs ────────────
                        logger.debug(
                            "FLIP BLOCK: %s tried to flip %s→%s after %.1fh (conf≈%d < %d)",
                            symbol, last['bias'], bias, hours_since,
                            round(tentative_ratio), _FLIP_MIN_CONF
                        )
                        return ScanFailure(REASON_FLIP_BLOCK, detail=flip_detail)

        # ── FIX #8 — Counter-trend veto ────────────────────────────────
        # If daily EMA stack strongly opposes the bias, require higher
        # minimum score — prevents weak counter-trend signals qualifying.
        daily_bearish = ema20_d < ema50_d
        daily_bullish = ema20_d > ema50_d
        counter_trend = (bias == 'LONG' and daily_bearish) or \
                        (bias == 'SHORT' and daily_bullish)
        min_score_req = 6 if counter_trend else 3

        if winning < min_score_req:
            ct_detail = f"winning={winning} < min_score_req={min_score_req} ({bias} vs daily EMA)"
            if user_requested:
                logger.debug("COUNTER_TREND SOFT-WARN (user_requested): %s %s", symbol, ct_detail)
                ig_l.append(f"⚠️ Counter-trend: {ct_detail} — signal opposes daily EMA, higher risk")
            else:
                return ScanFailure(REASON_COUNTER_TREND, detail=ct_detail)

        # ── FIX #1 — Confidence: pure ratio, capped quality bonus ──────
        # Old: raw_conf + score * 0.12  → inflated by up to +3 points
        # New: ratio × 10, plus max +1.0 quality bonus for timeframe/volume agreement
        ratio_conf    = (winning / (winning + losing)) * 10
        quality_bonus = 0.0
        if (bias == 'LONG' and daily_bullish) or (bias == 'SHORT' and daily_bearish):
            quality_bonus += 0.5   # both timeframes agree
        vol_ratio_check = (vol / vol_ma) if (vol_ma and vol_ma > 0) else 1.0
        if vol_ratio_check > 1.3 and ((bias == 'LONG' and price > P['close']) or
                                       (bias == 'SHORT' and price < P['close'])):
            quality_bonus += 0.5   # volume confirms direction

        # ── FIX #CQ — Candle Quality gate ──────────────────────────────
        # Score the signal candle on body ratio, close position, and wick
        # asymmetry.  A doji and a full-body engulfing candle are NOT the
        # same signal — this quantifies that difference.
        #
        # quality_score range: [–1.0, +1.0]
        #   STRONG (≥ 0.55) → +0.5 quality_bonus (stacks with TF/vol bonuses)
        #   GOOD   (≥ 0.15) → +0.25 quality_bonus
        #   NEUTRAL          → no change
        #   WEAK   (≥ −0.55) → −1 confidence demotion (post-round, min 4 to pass)
        #   POOR   (< −0.55) → −2 confidence demotion; most false breakouts land here
        #
        # Note: the demotion is applied AFTER rounding so the gate is always
        # a whole-number comparison against the existing LOW_CONF threshold.
        cq_score, cq_label, cq_detail = candle_quality_score(L, bias)
        if cq_score >= 0.55:
            quality_bonus += 0.5
            reasons.append(f"✅ Strong signal candle ({cq_label}): {cq_detail}")
        elif cq_score >= 0.15:
            quality_bonus += 0.25
            reasons.append(f"✔ Good signal candle ({cq_label}): {cq_detail}")
        elif cq_score <= -0.55:
            reasons.append(f"⚠️ Poor signal candle ({cq_label}): {cq_detail} — confidence penalised")
        elif cq_score <= -0.15:
            reasons.append(f"⚠️ Weak signal candle ({cq_label}): {cq_detail} — confidence penalised")

        confidence = min(10, round(ratio_conf + quality_bonus))

        # Apply candle quality demotion AFTER rounding
        if cq_score <= -0.55:
            confidence -= 2   # POOR candle: doji / spinning top / close mid-range + bad wicks
        elif cq_score <= -0.15:
            confidence -= 1   # WEAK candle: indecisive but not a clear doji

        # Low confidence: tag warning but continue — user requested this pair explicitly
        low_conf_warning = None
        if confidence < 4:
            low_conf_warning = f"Very weak signal: conf={confidence}/10 (winning={winning} losing={losing})"
            logger.debug("LOW CONF WARN: %s conf=%d — passing with warning", symbol, confidence)

        # ── FIX #MS — Minimum absolute score gate ──────────────────────
        # A ratio of 4:0 gives confidence=10 but ls=4 is weak evidence —
        # only 4 indicator points touched the entire stack.  Enforce a
        # minimum raw winning score so high confidence requires substance:
        #   conf 9–10 → winning ≥ 9  (very strong absolute signal)
        #   conf 7–8  → winning ≥ 7
        #   conf 4–6  → winning ≥ 5  (existing min_score_req already ≥ 3/6)
        # This prevents a coin with one extreme RSI reading from scoring 10/10.
        # FIX #CR — Confidence Recalibration: raise the absolute score bar for
        # high-conf signals. Previously conf=8 only needed winning≥7, which
        # let moderate-evidence signals inflate to 8/10.
        # New floors: conf 9-10 → winning≥10, conf 8 → winning≥9
        # This reduces the number of high-conf signals but improves their accuracy.
        abs_score_floor = {10: 10, 9: 10, 8: 9, 7: 7, 6: 5, 5: 5, 4: 5}.get(confidence, 5)
        if winning < abs_score_floor:
            # Downgrade confidence rather than drop the signal — it may still
            # be valid but at a more honest conviction level.
            old_conf   = confidence
            confidence = confidence - 1
            if confidence < 4:
                low_conf_warning = (low_conf_warning or "") + f" Abs-score gate: conf {old_conf}→{confidence}"
                logger.debug("ABS SCORE GATE WARN: %s conf %d→%d (winning=%d floor=%d) — passing with warning",
                             symbol, old_conf, confidence, winning, abs_score_floor)
            else:
                logger.debug("ABS SCORE GATE: %s conf %d→%d (winning=%d floor=%d)",
                             symbol, old_conf, confidence, winning, abs_score_floor)

        # ── FIX #RG2 — BTC Market Regime Gate (5-level) ───────────────
        # New get_btc_regime() returns: STRONG_BULL, BULL, NEUTRAL, BEAR, STRONG_BEAR
        #
        # Gate logic — what each regime blocks:
        #
        #   STRONG_BULL → SHORTs blocked below conf 9 (extreme vol: 7)
        #   BULL        → SHORTs blocked below conf 8 (extreme vol: 7)
        #   NEUTRAL     → ALL signals blocked below conf 8
        #                 (previously: -1 conf penalty only — far too weak)
        #                 Rationale: choppy markets are where false signals
        #                 cluster. If the regime is truly neutral, there is
        #                 no directional edge and low-conf signals should not fire.
        #   BEAR        → LONGs  blocked below conf 8 (extreme vol: 7)
        #   STRONG_BEAR → LONGs  blocked below conf 9 (extreme vol: 7)
        #
        # Exception: EXTREME vol regime (ATR% >= 5.5%) lowers all floors by 1.
        # In panic/euphoria conditions reversals are genuinely more likely.
        btc_regime     = get_btc_regime()
        is_btc         = symbol.upper().replace('_USDT', 'USDT') in ('BTCUSDT', 'BTCPERP')
        is_extreme_vol = (((atr / price) * 100) >= 5.5)

        # regime_warning is set when the signal is below the recommended floor
        # for the current BTC regime.  Signal is NOT hard-blocked — it passes
        # through with a warning so /scan always shows the full analysis.
        regime_warning = None

        # ���─ FIX #BTCD — BTC Dominance Filter ──────────────────────────
        # BTC.D rising = capital rotating from alts to BTC → alt LONGs face headwind.
        # Apply a confidence penalty of −1 for altcoin LONGs when BTC.D is rising.
        # Attach a warning note; never hard-block (user may know why they're trading).
        _btcd_info = _get_btc_dominance()
        _btcd_note = None
        if not is_btc and _btcd_info['trend'] != 'flat' and _btcd_info['btcd'] > 0:
            if _btcd_info['trend'] == 'rising' and bias == 'LONG':
                confidence = max(1, confidence - 1)
                _btcd_note = (f"⚠️ BTC.D rising ({_btcd_info['btcd']:.1f}%) — alt LONG headwind "
                              f"(conf docked −1 → {confidence})")
                reasons.append(_btcd_note)
                logger.debug("BTCD: %s LONG penalised -1 conf (BTC.D rising %.1f%%)",
                             symbol, _btcd_info['btcd'])
            elif _btcd_info['trend'] == 'falling' and bias == 'SHORT':
                _btcd_note = (f"ℹ️ BTC.D falling ({_btcd_info['btcd']:.1f}%) — alt SHORT headwind "
                              f"(capital returning to alts)")
                reasons.append(_btcd_note)

        if not is_btc:
            if btc_regime in ('STRONG_BULL', 'BULL') and bias == 'SHORT':
                floor = (9 if btc_regime == 'STRONG_BULL' else 8)
                if is_extreme_vol: floor = max(floor - 1, 7)
                if confidence < floor:
                    regime_warning = f"Counter-regime SHORT: conf={confidence} below {btc_regime} floor ({floor})"
                    logger.debug("REGIME WARN: %s SHORT conf=%d below floor=%d (%s)",
                                 symbol, confidence, floor, btc_regime)
                else:
                    reasons.append(f"⚠️ Counter-regime SHORT in BTC {btc_regime} — high-conf only ({confidence}/10)")

            elif btc_regime in ('STRONG_BEAR', 'BEAR') and bias == 'LONG':
                floor = (9 if btc_regime == 'STRONG_BEAR' else 8)
                if is_extreme_vol: floor = max(floor - 1, 7)
                if confidence < floor:
                    regime_warning = f"Counter-regime LONG: conf={confidence} below {btc_regime} floor ({floor})"
                    logger.debug("REGIME WARN: %s LONG conf=%d below floor=%d (%s)",
                                 symbol, confidence, floor, btc_regime)
                else:
                    reasons.append(f"⚠️ Counter-regime LONG in BTC {btc_regime} — high-conf only ({confidence}/10)")

            elif btc_regime == 'NEUTRAL':
                neutral_floor = 7 if is_extreme_vol else 8
                if confidence < neutral_floor:
                    regime_warning = f"{bias} conf={confidence} below NEUTRAL floor ({neutral_floor})"
                    logger.debug("REGIME WARN: %s %s conf=%d below neutral floor=%d",
                                 symbol, bias, confidence, neutral_floor)
                else:
                    reasons.append(f"ℹ️ BTC NEUTRAL regime — only high-conf signals pass ({confidence}/10)")


        # ── FIX #VA — Volatility-aware target multipliers ──────────────
        # FIX RR — Risk:Reward restructure.
        # OLD structure had SL > T1 distance in every regime (e.g. MEDIUM:
        # T1=1.5*ATR, SL=2.5*ATR → R:R=0.6). This means you need a >63%
        # win rate just to break even at T1 — mathematically impossible for
        # most technical systems. A 4.3% win rate against this structure
        # produces catastrophic negative EV.
        #
        # NEW structure: SL is always TIGHTER than T1, giving R:R >= 1.3:1
        # at T1. This means a system with even 45% win rate has positive EV.
        # T2 and T3 extend further to reward letting winners run.
        #
        # Regime    | OLD T1/SL      | NEW T1/SL      | R:R (T1)
        # RANGING   | 0.8 / 1.5 ATR  | 1.2 / 0.7 ATR  | 1.71
        # LOW       | 1.2 / 2.0 ATR  | 1.6 / 1.0 ATR  | 1.60
        # MEDIUM    | 1.5 / 2.5 ATR  | 2.0 / 1.3 ATR  | 1.54
        # HIGH      | 1.8 / 3.0 ATR  | 2.5 / 1.6 ATR  | 1.56
        # EXTREME   | 2.2 / 3.5 ATR  | 3.0 / 2.0 ATR  | 1.50
        #
        # Note: SL tightening is safe because Fix SL1 now requires a CLOSE
        # below the SL level (not a wick), so tighter SLs no longer fire on
        # normal candle noise.
        atr_pct_va = (atr / price) * 100

        # FIX #EW — Wider entry zones (+30%) to reduce missed entries.
        # entry_w_lo = lower pullback bound, entry_w_hi = upper chase allowance.
        # Wider zones mean price is more likely to trade through the zone
        # before the signal expires, reducing missed-entry count significantly.
        if atr_pct_va < 1.0:
            vol_regime = 'RANGING'
            t1_mult, t2_mult, t3_mult = 1.2,  1.9,  3.0
            sl_mult                   = 0.7
            entry_w_lo, entry_w_hi    = 0.22, 0.15   # was 0.15 / 0.10
        elif atr_pct_va < 2.0:
            vol_regime = 'LOW'
            t1_mult, t2_mult, t3_mult = 1.6,  2.5,  3.8
            sl_mult                   = 1.0
            entry_w_lo, entry_w_hi    = 0.35, 0.18   # was 0.25 / 0.12
        elif atr_pct_va < 3.5:
            vol_regime = 'MEDIUM'
            t1_mult, t2_mult, t3_mult = 2.0,  3.0,  4.8
            sl_mult                   = 1.3
            entry_w_lo, entry_w_hi    = 0.50, 0.25   # was 0.35 / 0.18
        elif atr_pct_va < 5.5:
            vol_regime = 'HIGH'
            t1_mult, t2_mult, t3_mult = 2.5,  3.8,  5.8
            sl_mult                   = 1.6
            entry_w_lo, entry_w_hi    = 0.65, 0.32   # was 0.45 / 0.22
        else:
            vol_regime = 'EXTREME'
            t1_mult, t2_mult, t3_mult = 3.0,  4.5,  7.0
            sl_mult                   = 2.0
            entry_w_lo, entry_w_hi    = 0.80, 0.42   # was 0.55 / 0.28

        if bias == "LONG":
            # ── FIX #TL (Timing Lag) — Pullback-anchored entry zone ────────
            # The 4H signal fires at candle CLOSE, meaning price has already
            # travelled the full move that triggered the signal.  Setting
            # entry_high = price + X ATR means we're asking the user to chase
            # above the close — the worst possible entry.
            #
            # Instead, anchor the zone BELOW the close price:
            #   entry_high = close - 0.05 ATR  (just under close — don't chase)
            #   entry_low  = close - entry_w_lo ATR  (deeper pullback, still valid)
            #
            # The zone now represents a realistic retracement entry:
            # "I want to buy the dip after the signal candle" not
            # "I want to buy at whatever price I see when I open Telegram".
            #
            # Targets are still calculated from the signal price (close) so
            # R:R is correctly measured from the breakout level, not the entry.
            entry_high = price - atr * 0.05                  # just under close — minimum pullback
            entry_low  = price - atr * (entry_w_lo + entry_w_hi)  # max pullback still in structure
            # Safety floor: entry_low must stay above stop_loss
            stop_loss  = max(support - atr * 0.3, price - atr * sl_mult)
            # Direction safety: SL must always be BELOW price for a LONG.
            # find_pivot_support can return a swing-low that is above the current
            # price (if the market has fallen since that pivot), which pushes SL
            # above entry — nonsensical and dangerous.  Fall back to ATR-based SL.
            if stop_loss >= price:
                stop_loss = price - atr * sl_mult

            # ── FIX #TP — Guaranteed T1 < T2 < T3 ordering ─────────────
            # Step 1: compute raw targets from price using vol-regime multipliers
            t1_raw = price + atr * t1_mult
            t2_raw = price + atr * t2_mult
            t3_raw = price + atr * t3_mult

            # Step 2: anchor T1 below nearest resistance (or use raw if no wall)
            if resistance > entry_high:
                t1 = min(t1_raw, resistance * 0.998)
                # Floor: T1 must always be above entry zone (tradeable signal)
                t1 = max(t1, entry_high * 1.001)
            else:
                t1 = t1_raw

            # Step 3: T2 must be strictly above T1 — use raw if it already is,
            # otherwise step it up by the gap between t1_mult and t2_mult
            t2 = max(t2_raw, t1 + atr * (t2_mult - t1_mult))

            # Step 4: T3 must be strictly above T2 — same pattern
            t3 = max(t3_raw, t2 + atr * (t3_mult - t2_mult))

        else:
            # ── FIX #TL — SHORT pullback-anchored entry zone ──────��─────
            # Mirror of the LONG fix: price fell to create the SHORT signal,
            # so entering at the close or below it is chasing the dump.
            # Set the zone to a realistic dead-cat bounce level:
            #   entry_low  = close + 0.05 ATR  (just above close — minimum bounce)
            #   entry_high = close + (entry_w_lo + entry_w_hi) ATR  (fuller bounce still valid)
            entry_low  = price + atr * 0.05
            entry_high = price + atr * (entry_w_lo + entry_w_hi)
            stop_loss  = min(resistance + atr * 0.3, price + atr * sl_mult)
            # Direction safety: SL must always be ABOVE price for a SHORT.
            if stop_loss <= price:
                stop_loss = price + atr * sl_mult

            # ── FIX #TP — Guaranteed T1 > T2 > T3 ordering (SHORT) ─────
            t1_raw = price - atr * t1_mult
            t2_raw = price - atr * t2_mult
            t3_raw = price - atr * t3_mult

            # Anchor T1 above nearest support
            if support < entry_low:
                t1 = max(t1_raw, support * 1.002)
                # Ceiling: T1 must always be below entry zone
                t1 = min(t1, entry_low * 0.999)
            else:
                t1 = t1_raw

            # T2 must be strictly below T1
            t2 = min(t2_raw, t1 - atr * (t2_mult - t1_mult))

            # T3 must be strictly below T2
            t3 = min(t3_raw, t2 - atr * (t3_mult - t2_mult))

        # ── DURATION ENGINE ─────────────────────────────────────────────
        # FIX #3 applied throughout — daily EMA/MACD from closed candle.
        # FIX #8 applied — counter-trend trades get shorter holds.

        # ── FIX #ATR-SL — ATR Stop Distance Validation ─────────────────
        # A stop that is less than 1.0× ATR from entry will be stopped out
        # on normal candle noise — the stop is not at a meaningful level.
        # If the computed stop is too tight, widen it to 1.2× ATR minimum.
        # Widen silently (don't fail — the signal can still be valid).
        _atr_sl_min_mult = 1.0   # stop must be at least 1.0× ATR from price
        _sl_distance = abs(price - stop_loss)
        if atr > 0 and _sl_distance < atr * _atr_sl_min_mult:
            _old_sl = stop_loss
            if bias == 'LONG':
                stop_loss = price - atr * _atr_sl_min_mult
                reasons.append(f"⚠️ SL widened to {_atr_sl_min_mult}× ATR — original was {(_sl_distance/atr):.2f}× ATR (noise-stop risk)")
            else:
                stop_loss = price + atr * _atr_sl_min_mult
                reasons.append(f"⚠️ SL widened to {_atr_sl_min_mult}× ATR — original was {(_sl_distance/atr):.2f}× ATR (noise-stop risk)")
            logger.debug("ATR-SL FIX: %s %s SL widened from %.6f to %.6f (ATR=%.6f)",
                         symbol, bias, _old_sl, stop_loss, atr)

        dur_score = 0
        dur_notes = []

        if bias == "LONG":
            tf_aligned = (price > ema20 > ema50) and (ema20_d > ema50_d)
        else:
            tf_aligned = (price < ema20 < ema50) and (ema20_d < ema50_d)

        if tf_aligned:
            dur_score += 3; dur_notes.append("Both 4H & Daily EMAs aligned — trend has legs")
        elif counter_trend:
            dur_score -= 2; dur_notes.append("Counter-trend — daily opposes, shorter hold recommended")
        else:
            dur_notes.append("EMA structure mixed — trend may be short-lived")

        # FIX #4 — confirmed crossovers only for duration scoring
        if macd_cross_bull_1d or macd_cross_bear_1d:
            dur_score += 3; dur_notes.append("Daily MACD confirmed cross [closed] — early in move, hold longer")
        elif macd_cross_bull_4h or macd_cross_bear_4h or macd_sust_bull_4h or macd_sust_bear_4h:
            dur_score += 2; dur_notes.append("4H MACD confirmed cross — move is fresh")
        elif (macd_diff > 0 and macd_d > 0) or (macd_diff < 0 and macd_d < 0):
            dur_score += 1; dur_notes.append("MACD momentum sustained both timeframes")
        else:
            dur_notes.append("MACD diverging across timeframes — move may be late")

        atr_pct = (atr / price) * 100
        if atr_pct >= 5.0:
            dur_score -= 2; dur_notes.append(f"ATR {atr_pct:.1f}% — very high volatility, exit quickly")
        elif atr_pct >= 3.0:
            dur_score -= 1; dur_notes.append(f"ATR {atr_pct:.1f}% — elevated volatility, targets hit faster")
        elif atr_pct >= 1.5:
            dur_score += 1; dur_notes.append(f"ATR {atr_pct:.1f}% — moderate volatility, medium hold suits")
        else:
            dur_score += 2; dur_notes.append(f"ATR {atr_pct:.1f}% — low volatility, needs more time to move")

        rsi_extreme = (rsi4 < 25 or rsi4 > 75) or (rsi_d < 30 or rsi_d > 70)
        if rsi_extreme:
            dur_score -= 1; dur_notes.append("RSI at extreme — snap-back likely, not a sustained trend")
        else:
            dur_score += 1; dur_notes.append("RSI in trend zone — sustainable directional momentum")

        vol_ratio = (vol / vol_ma) if (vol_ma and vol_ma > 0) else 1.0
        if vol_ratio > 2.5:
            dur_score -= 2; dur_notes.append(f"Volume {vol_ratio:.1f}x avg — possible climax candle, exit sooner")
        elif vol_ratio > 2.0:
            dur_score -= 1; dur_notes.append(f"Volume {vol_ratio:.1f}x avg — watch for exhaustion")
        elif vol_ratio > 1.3:
            dur_score += 1; dur_notes.append(f"Volume {vol_ratio:.1f}x avg — healthy participation")
        else:
            dur_notes.append("Volume below average — weak follow-through expected")

        if 40 <= rsi_d <= 60:
            dur_score += 2; dur_notes.append("Daily RSI in 40–60 range — trend has room to run [closed candle]")

        # Hard cap: very volatile coins cannot get multi-day holds
        if atr_pct >= 5.0 and dur_score > 5:
            dur_score = 5

        if dur_score >= 9:   hold_hours, hold = 72, "~3 days";   tf_note = f"Strong structural trend — hold ~3 days. {dur_notes[0]}"
        elif dur_score >= 7: hold_hours, hold = 48, "~2 days";   tf_note = f"Confirmed multi-TF trend — hold ~2 days. {dur_notes[0]}"
        elif dur_score >= 5: hold_hours, hold = 24, "~1 day";    tf_note = f"4H trend intact — hold ~24 hours. {dur_notes[0]}"
        elif dur_score >= 3: hold_hours, hold = 10, "~10 hours"; tf_note = f"Short-lived setup — exit within 10 hours. {dur_notes[0]}"
        elif dur_score >= 1: hold_hours, hold = 4,  "~4 hours";  tf_note = f"Quick move expected — target within 4 hours. {dur_notes[0]}"
        else:                hold_hours, hold = 1,  "~1 hour";   tf_note = f"Weak setup — scalp only. {dur_notes[0]}"

        # FIX #SE — Signal Expiry: cap hold_hours based on vol_regime so
        # high-volatility signals expire faster and don't stay open 3 days.
        # A 3-day EXTREME-vol signal is almost always stale within hours.
        _hold_caps = {'RANGING': 72, 'LOW': 48, 'MEDIUM': 24, 'HIGH': 12, 'EXTREME': 6}
        _cap = _hold_caps.get(vol_regime, 24)
        if hold_hours > _cap:
            hold_hours = _cap
            hold = f"~{_cap}h (vol-capped)"
            tf_note = tf_note + f" [hold capped at {_cap}h for {vol_regime} volatility]" 

        lev = calculate_leverage(price, entry_low, entry_high, stop_loss, atr, confidence, bias)

        # ── FIX #LS — Leverage-Safe Spread Cap ─────────────────────────
        # At high leverage a wide entry zone becomes dangerous: entering at
        # the bottom vs the top of the zone can shift liquidation price by
        # (spread% * leverage)%.  We cap the zone so that the spread never
        # exceeds (20 / suggested_leverage)% of price, with a floor of 0.4%
        # and ceiling of 3.0% (same as old fixed cap but now leverage-aware).
        if lev and lev['suggested'] and lev['suggested'] > 0:
            _sugg_lev = lev['suggested']
            _max_spread_pct = max(0.4, min(3.0, 20.0 / _sugg_lev)) / 100.0
            _max_spread_abs = price * _max_spread_pct
            _current_spread = abs(entry_high - entry_low)
            if _current_spread > _max_spread_abs and _max_spread_abs > 0:
                _entry_mid = (entry_low + entry_high) / 2
                entry_low  = _entry_mid - _max_spread_abs / 2
                entry_high = _entry_mid + _max_spread_abs / 2
                # Re-enforce direction safety after tightening
                if bias == 'LONG':
                    entry_low = max(entry_low, stop_loss * 1.001)
                else:
                    entry_high = min(entry_high, stop_loss * 0.999)
                # Recompute leverage from the tightened zone
                lev = calculate_leverage(price, entry_low, entry_high, stop_loss, atr, confidence, bias)

        # ── FIX #EZ — Entry zone freshness flag ────────────────────────
        # The entry zone is computed at scan time.  By the time a user
        # reads the signal, price may have already broken through the zone.
        # Tag the signal so format_signal can warn the user.
        #   entry_in_zone=True  → price still within entry_low..entry_high
        #   entry_in_zone=False → price has moved out of the zone
        entry_in_zone = (entry_low <= price <= entry_high)

        # ── FIX #FLIP — record this signal for cooldown tracking ───────
        _last_signal_bias[symbol] = {'bias': bias, 'time': datetime.now()}
        db_save_signal_bias(symbol, bias)  # FIX #PERSIST-BIAS — write to DB so cooldown survives restarts

        result = {
            'symbol': symbol, 'bias': bias, 'confidence': confidence,
            'score': score, 'price': price,
            'entry_low': entry_low, 'entry_high': entry_high,
            'stop_loss': stop_loss,
            't1': t1, 't2': t2, 't3': t3,
            'hold': hold, 'tf_note': tf_note,
            'hold_hours': hold_hours, 'dur_score': dur_score, 'dur_reasons': dur_notes,
            'funding': funding * 100,
            'rsi4': rsi4, 'rsi_d': rsi_d,
            'stoch_k': stoch_k, 'atr': atr,
            'reasons': reasons, 'leverage': lev,
            'counter_trend': counter_trend,
            'btc_regime': btc_regime,
            'vol_regime': vol_regime,
            'entry_in_zone': entry_in_zone,
            # FIX #CQ — candle quality metadata (surfaced in signal cards + ML features)
            'candle_quality': cq_label,
            'candle_quality_score': round(cq_score, 3),
            # FIX #SESSION — session context at signal time
            'session':          _sess['session'],
            'session_note':     _sess['note'],
            'session_warning':  _sess['bias_warning'],
            # FIX #FIB — Fibonacci retracement levels (for signal card details)
            'fib_levels':       _fib_display_levels,
            'fib_confluence':   max(_fib_l_score, _fib_s_score),
            'scan_time': datetime.now(),
            # FIX #RI — store BTC price at scan time for regime invalidation
            'btc_price_at_scan': _get_btc_price_cached(),
            # CONVICTION LAYER — CVD / OI / VWAP payload (empty dict if disabled)
            'conviction': _conv,
        }

        # ── ML SCORING — inject win-probability scores if models are loaded ──
        if _XGB_AVAILABLE:
            try:
                result['ml_score'] = xgb_predict_signal(result)
            except Exception as _ml_e:
                logger.debug("XGB predict failed for %s: %s", symbol, _ml_e)
        if _RF_AVAILABLE:
            try:
                result['rf_score'] = rf_predict_signal(result)
                rf_meta = rf_model_meta()
                if result.get('ml_score') is not None and result.get('rf_score') is not None:
                    consensus = (result['ml_score'] + result['rf_score']) / 2
                    result['consensus_score']   = consensus
                    result['consensus_verdict'] = (
                        "Strong edge" if consensus >= 0.70 else
                        "Moderate edge" if consensus >= 0.55 else
                        "Weak edge"
                    )
            except Exception as _ml_e:
                logger.debug("RF predict failed for %s: %s", symbol, _ml_e)

        # Attach warning tags so display layer can rate the risk
        result['regime_warning']   = regime_warning    # set above if regime floor missed
        result['low_conf_warning'] = low_conf_warning  # set above if conf < 4
        result['btc_regime']       = btc_regime
        result['confidence']       = confidence        # may have been downgraded above

        return result
    except Exception as e:
        logger.warning("score_pair error for %s: %s", symbol, e)
        return None




# ─────────────────────────────────────────────
# MULTI-TIMEFRAME ENGINE
# ─────────────────────────────────────────────
# TF_CONFIGS defines every supported timeframe.
# Each entry specifies:
#   primary      — candle interval fed to score_pair as df4h (the "fast" TF)
#   confirm      — confirmation candle interval fed as df1d (the "slow" TF)
#   bybit_pri    — Bybit API interval string for primary
#   bybit_con    — Bybit API interval string for confirmation
#   mexc_pri     — MEXC interval string for primary
#   mexc_con     — MEXC interval string for confirmation
#   binance_pri  — Binance interval string for primary
#   binance_con  — Binance interval string for confirmation
#   min_candles  — minimum closed candles required on the primary TF
#   label        — human-readable label shown in signals
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


# ── NEW-LISTING DYNAMIC SCAN ──────────────────────────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────

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
            # ── LIQUIDITY FILTER — skip for user-requested scans ──────────────
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
#   • Volume bars    (panel 2, coloured by candle direction)
#   • RSI with 30/70 levels (panel 3)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# ANALYZE FUNCTIONS
# ─────────────────────────────────────────────
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
        result['exchange'] = 'MEXC'
        return result
    except Exception as e:
        logger.warning("MEXC %s analyze error: %s", symbol, e)
        return None


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
    global _btc_regime_cache

    results = []

    bybit_check_available()
    binance_check_available()

    # Warm regime cache — same gate logic as full scan
    _btc_regime_cache = None
    regime = get_btc_regime()
    logger.info("=== MID SCAN START | BTC Regime: %s | ranks %d–%d ===",
                regime, rank_from, rank_to)

    def _process(r):
        if not r:
            return
        r['tier'] = 'MID'   # tag so UI can badge these signals
        results.append(r)

    for sym in bybit_get_mid_symbols(rank_from, rank_to):
        _process(analyze_bybit(sym))
        time.sleep(0.15)

    for sym in mexc_get_mid_symbols(rank_from, rank_to):
        _process(analyze_mexc(sym))
        time.sleep(0.15)

    for sym in binance_get_mid_symbols(rank_from, rank_to):
        _process(analyze_binance(sym))
        time.sleep(0.15)

    _vol_rank = {"MEDIUM": 4, "HIGH": 3, "LOW": 2, "EXTREME": 1, "RANGING": 0}
    results.sort(
        key=lambda x: (x["confidence"], x["score"],
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
# ══════════════════════════════════════════════════════════════════════════════
# LIQUIDITY FILTER
# ──────────────────────────────────────────────────────────────────────────────
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
# ═��════════════════════════════════════════════════════════════════════════════

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
            r    = requests.get(
                f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}",
                headers=HEADERS, timeout=8
            )
            data = r.json()
            if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                vol = float(data['result']['list'][0].get('turnover24h', 0) or 0)

        elif exchange == 'MEXC':
            _sym_c = symbol.upper().replace('_USDT', 'USDT'); futures_sym = (_sym_c[:-4] + '_USDT') if _sym_c.endswith('USDT') else (_sym_c + '_USDT')
            r    = requests.get(
                f"https://contract.mexc.com/api/v1/contract/ticker?symbol={futures_sym}",
                headers=HEADERS, timeout=8
            )
            data = r.json()
            if data.get('success') and data.get('data'):
                d   = data['data']
                vol = float(d.get('amount24', 0) or d.get('volume24', 0) or 0)

        elif exchange == 'BINANCE':
            r    = requests.get(
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
    global last_scan_results, previous_scan_results, last_scan_time, price_history
    global _btc_regime_cache, _scan_durations
    _scan_t0 = time.time()

    previous_scan_results = last_scan_results.copy()
    results = []

    # Check exchange availability once per scan
    bybit_check_available()
    binance_check_available()

    # FIX #RG — Invalidate regime cache so the first score_pair call
    # this cycle fetches a fresh BTC read.  All subsequent calls within
    # the scan will hit the newly populated cache (TTL = 15 min).
    _btc_regime_cache = None
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
                r    = requests.get("https://api.bybit.com/v5/market/tickers?category=linear",
                                    headers=HEADERS, timeout=15)
                data = r.json()
                if data.get('retCode') == 0:
                    for t in data['result']['list']:
                        sym = t.get('symbol', '')
                        if sym in symbols:
                            vol = float(t.get('turnover24h', 0) or 0)
                            _vol_cache[f"BYBIT_{sym}"] = (vol, now)
            elif exchange == 'MEXC':
                r    = requests.get("https://contract.mexc.com/api/v1/contract/ticker",
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
                r    = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
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
        if key not in price_history:
            price_history[key] = []
        price_history[key].append({'time': datetime.now(), 'price': r['price'],
                                   'exchange': r['exchange'], 'symbol': r['symbol']})
        cutoff = datetime.now() - timedelta(hours=24)
        price_history[key] = [p for p in price_history[key] if p['time'] > cutoff]

    bybit_syms   = bybit_get_top_symbols(50)
    mexc_syms    = mexc_get_top_symbols(50)
    binance_syms = binance_get_top_symbols(50) if sakz_exchanges.BINANCE_AVAILABLE else []

    # Warm volume cache with a single ticker request per exchange
    _warm_vol_cache(set(bybit_syms),   'BYBIT')
    _warm_vol_cache(set(mexc_syms),    'MEXC')
    if binance_syms:
        _warm_vol_cache(set(binance_syms), 'BINANCE')

    for sym in bybit_syms:
        _process(analyze_bybit(sym), 'BYBIT')
        time.sleep(0.2)

    for sym in mexc_syms:
        _process(analyze_mexc(sym), 'MEXC')
        time.sleep(0.2)

    for sym in binance_syms:
        _process(analyze_binance(sym), 'BINANCE')
        time.sleep(0.2)

    _vol_rank = {"MEDIUM": 4, "HIGH": 3, "LOW": 2, "EXTREME": 1, "RANGING": 0}
    results.sort(
        key=lambda x: (x["confidence"], x["score"], _vol_rank.get(x.get("vol_regime", "MEDIUM"), 2)),
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
    global _last_portfolio_summary
    _last_portfolio_summary = portfolio_summary

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

    last_scan_results = results
    last_scan_time    = datetime.now()

    db_save_scan(results)
    for r in results[:20]:
        db_register_outcome(r)

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
    _scan_durations.append(time.time() - _scan_t0)
    if len(_scan_durations) > 50:
        _scan_durations = _scan_durations[-50:]   # keep last 50
    return results


async def get_scan_results(force=False):
    """
    Async wrapper that:
    1. Returns cached results if within TTL and not forced
    2. Otherwise runs a fresh scan in SCAN_EXECUTOR (up to 3 concurrent)
    3. Uses a lock to prevent cache stampede (multiple users triggering
       simultaneous full scans at the exact same moment)
    """
    global _scan_cache

    # Serve cache if fresh enough
    if not force and _scan_cache:
        age = (datetime.now() - _scan_cache['time']).total_seconds()
        if age < CACHE_TTL_SECS:
            logger.info("Serving cached scan results (%.0fs old)", age)
            return _scan_cache['results'], True  # (results, from_cache)

    # Use cache lock only to avoid stampede — other users can still run
    # their own scans via SCAN_EXECUTOR independently
    async with _scan_cache_lock:
        # Re-check inside lock in case another coroutine just populated cache
        if not force and _scan_cache:
            age = (datetime.now() - _scan_cache['time']).total_seconds()
            if age < CACHE_TTL_SECS:
                return _scan_cache['results'], True

        loop    = asyncio.get_event_loop()
        results = await loop.run_in_executor(SCAN_EXECUTOR, run_full_scan)

        _scan_cache = {'results': results, 'time': datetime.now()}
        return results, False  # (results, from_cache)


# ─────────────────────────────────────────────
# IMPROVEMENT #3 — SIGNAL OUTCOME TRACKER
# Background job that checks signal outcomes at
# 4h, 8h, 24h, 48h intervals and updates the DB.
# ─────────────────────────────────────────────
def _fetch_ohlc_since(exchange, symbol, scan_time):
    """
    FIX A2 — Fetch 1H OHLC candles covering the period from scan_time to now.
    Returns a list of dicts {ts, open, high, low, close} sorted oldest-first.
    Uses each exchange's existing futures kline endpoint (perps only).
    Limit 60 = up to 60 hours of 1h candles — more than enough for the 48h window.
    """
    try:
        if exchange == 'BYBIT':
            df = bybit_fetch_ohlcv(symbol, '60', 60)          # '60' = 1h on Bybit
        elif exchange == 'BINANCE':
            df = binance_fetch_ohlcv(symbol, '1h', 60)
        else:
            df = mexc_fetch_ohlcv(symbol, '1h', 60)

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
# ─────────────────────────────────────────────

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


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    args = context.args
    hours_filter = None
    label = "ALL TIME"

    if args:
        try:
            minutes, label, mode = _parse_stats_arg(args[0])
        except ValueError:
            await update.message.reply_text(
                "⚠️ Invalid time format. Examples:\n"
                "/stats 5m       → last 5 minutes (live)\n"
                "/stats 30m      → last 30 minutes (live)\n"
                "/stats 1h       → last 1 hour (live)\n"
                "/stats 24       → last 24 hours (DB outcomes)\n"
                "/stats 168      → last 7 days\n"
                "/stats 720      → last 30 days"
            )
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
    bar        = "█" * bar_filled + "░" * bar_empty

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
#   /backtest 168      → last 7 days only
# ─────────────────────────────────────────────
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

    await update.message.reply_text("🔬 Running backtest analysis…")

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

    # ── Win rate by confidence band ────────────────────────────────────
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
        recommendations.append("🔴 LONG win rate < 40% — regime gate may need tightening (lower threshold from conf≥9 to conf≥8 in BEAR).")
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
        f"🔬 BACKTEST ANALYSIS — {label}\n"
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
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
    global last_scan_results, last_scan_time
    if not last_scan_results:
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

    filtered = [r for r in last_scan_results
                if r['confidence'] >= min_conf and (bias is None or r['bias'] == bias)]

    if not filtered:
        await update.message.reply_text(
            f"⚠️ No signals match: bias={bias or 'ANY'}, min conf={min_conf}/10\n"
            f"Try lowering the confidence threshold."
        )
        return

    age  = datetime.now() - last_scan_time
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

# Dedup cache: { "EXCHANGE_SYMBOL_BIAS": datetime_last_sent }
_autoscan_sent: dict = {}
_AUTOSCAN_COOLDOWN_H = 4    # hours before the same signal can fire again
_AUTOSCAN_MIN_CONF   = 8    # minimum confidence to push a signal

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
    """Return True if this signal key has NOT been sent within the cooldown window."""
    sent_at = _autoscan_sent.get(key)
    if not sent_at:
        return True
    return (datetime.now() - sent_at).total_seconds() > _AUTOSCAN_COOLDOWN_H * 3600


def _autoscan_mark_sent(key: str):
    _autoscan_sent[key] = datetime.now()
    # Prune stale entries so the dict does not grow forever
    cutoff = datetime.now() - timedelta(hours=_AUTOSCAN_COOLDOWN_H * 2)
    for k in list(_autoscan_sent):
        if _autoscan_sent[k] < cutoff:
            del _autoscan_sent[k]


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

    high_conf = [r for r in results if r['confidence'] >= _AUTOSCAN_MIN_CONF]
    if not high_conf:
        return

    for r in high_conf:
        exch      = r.get('exchange', 'MEXC')
        sym       = r['symbol']
        bias      = r['bias']
        sig_tf    = r.get('timeframe', '4h')
        dedup_key = f"{exch}_{sym}_{bias}_{sig_tf}"

        if not _autoscan_is_fresh(dedup_key):
            continue

        _autoscan_mark_sent(dedup_key)

        # FIX #RESTART-FLOOD — within the post-restart grace window, the
        # mark-sent above seeds the (wiped) dedup table, but we skip the actual
        # push so a restart doesn't re-blast every currently-live signal.
        if (datetime.now() - _BOT_START_TS).total_seconds() < _AUTOSCAN_STARTUP_GRACE_SECS:
            continue

        trade_type, tt_emoji = _autoscan_trade_type(r)
        conf       = r['confidence']
        conf_bar   = _conf_bar_emoji(conf)
        bias_emoji = "🟢" if bias == 'LONG' else "🔴"
        now_str    = datetime.now().strftime('%H:%M')

        header = (
            f"{tt_emoji} AUTOSCAN — {trade_type.upper()}\n"
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
            # Filter: if subscriber chose a specific TF, skip signals not on that TF
            if tf_pref and sig_tf != tf_pref:
                continue
            try:
                await context.bot.send_message(chat_id=chat_id, text=header)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=format_signal_primary(r, 1),
                    reply_markup=keyboard,
                )
                _safemode_store_signals(chat_id, [r])
                await asyncio.sleep(0.15)
            except Exception as e:
                logger.warning("Autoscan push failed for %s: %s", chat_id, e)

        await asyncio.sleep(0.5)


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
    global _scan_cache
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
    _scan_cache = {'results': merged, 'time': datetime.now()}
    logger.info("mid_scan_job merged %d new mid-tier signals into scan cache (%d total)",
                added, len(merged))


async def autoscan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    chat_id = update.effective_chat.id

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
            "Or *reply with a custom timeframe* — e.g. `1h`, `30m`, `1w`",
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
# ─────────────────────────────────────────────
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


def _rr_ratio(r):
    """Compute R:R ratio from signal fields. Returns float or None."""
    try:
        price     = r['price']
        stop_loss = r['stop_loss']
        t1        = r['t1']
        risk      = abs(price - stop_loss)
        reward    = abs(t1 - price)
        if risk > 0:
            return reward / risk
    except Exception:
        pass
    return None


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
    except Exception:
        pass
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
    cached = _signal_card_cache.get(cache_key)
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
        _signal_card_cache[cache_key] = result
        return result
    except Exception:
        return {}


def format_signal_primary(r, rank):
    """
    Req #15 — Unified signal panel used everywhere:
      /scan number tap, /cscan, /scalp, auto-signal detection.

    Layout:
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ⚡️ SWING | SIGNAL
        CONFIDENCE: 🟩🟩🟩🟩🟩 10/10
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      📊 Token: XMRUSDT
      🟢 Direction: LONG
      💰 Entry: $x – $x
      ⚡️ Leverage: 8x
      📐 R:R: 1:2.3
      🎯 TP1: $x  (+X% profit) 💰
      🎯 TP2: $x
      🎯 TP3: $x
      🛑 Stop Loss: $x

    Buttons: [🔄 Refresh]  [📋 Details]
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
    rr_str  = f"1:{rr:.1f}" if rr else "N/A"

    # counter-trend warning prefix
    ct_line = "⚠️ COUNTER-TREND — Higher risk\n" if r.get('counter_trend') else ""

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
        f"🎯 TP1: ${r['t1']:.4f}  (+{t1_pct:.1f}% profit) 💰",
        f"🎯 TP2: ${r['t2']:.4f}  (+{t2_pct:.1f}%)",
        f"🎯 TP3: ${r['t3']:.4f}  (+{t3_pct:.1f}%)",
        f"🛑 Stop Loss: ${r['stop_loss']:.4f}",
    ]

    # ML line if available
    consensus_score   = r.get('consensus_score')
    consensus_verdict = r.get('consensus_verdict', '')
    ml_score          = r.get('ml_score')
    rf_score          = r.get('rf_score')
    if consensus_score is not None:
        pct = int(round(consensus_score * 100))
        lines.append(f"🤖 ML: {pct}% win prob — {consensus_verdict}")
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
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
    _signal_card_cache[key] = _card_entry
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
    entry = _signal_card_cache.get(key)
    if not entry:
        await query.answer("Signal data expired. Re-run the scan.", show_alert=True)
        return
    r    = entry['signal']
    rank = entry['rank']
    link = get_exchange_link(r.get('exchange', ''), r['symbol'])
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
    entry = _signal_card_cache.get(key)
    if not entry:
        await query.answer("Signal data expired. Re-run the scan.", show_alert=True)
        return
    r    = entry['signal']
    rank = entry['rank']
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
    entry = _signal_card_cache.get(key)
    if not entry:
        await query.answer("Signal data expired. Please re-run the scan.", show_alert=True)
        return
    r    = entry['signal']
    rank = entry['rank']
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


        age_min = int((datetime.now() - scan_time).total_seconds() / 60)
        age_str = f"{age_min}m ago" if age_min < 60 else f"{age_min // 60}h {age_min % 60}m ago"
    else:
        age_str = "unknown"

    # Zone status using live price
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

    lines = [
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"#{rank}  {exchange} | {r['symbol']}  [{tf_label}]{' 🏷[MID]' if r.get('tier') == 'MID' else ''}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"{bias_emoji} BIAS: {r['bias']}",
        f"⭐ CONVICTION:  {conf_bar}",
    ]

    # FIX #8 — surface counter-trend warning prominently in signal card
    if r.get('counter_trend'):
        lines.append(f"⚠️  COUNTER-TREND — Daily EMA opposes this trade. Higher risk.")

    # FIX #RG — show BTC regime context so user knows the macro backdrop
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
    lines.append(f"{regime_emoji} BTC REGIME: {regime}"
                 + (" — regime-aligned ✅" if is_aligned
                    else " — regime-divergent ⚠️" if is_divergent
                    else " — choppy/transitioning ⚠️"))

    # FIX #VA — show volatility regime so user understands target sizing context
    vr = r.get('vol_regime', 'MEDIUM')
    vr_emoji = {'RANGING': '😴', 'LOW': '🐢', 'MEDIUM': '⚖️', 'HIGH': '⚡', 'EXTREME': '🌪️'}.get(vr, '⚖️')
    lines.append(f"{vr_emoji} VOL REGIME: {vr} — targets sized accordingly")

    lines += [
        f"",
        f"⏱ HOLD DURATION: {r['hold']}",
        f"📌 {r['tf_note']}",
        f"",
        f"💰 SIGNAL PRICE: ${scan_price:.6f}  (scanned {age_str})",
        f"💰 LIVE PRICE:   ${display_price:.6f}" + (" ⚠️ [fetch failed — using scan price]" if live_price == 0 else ""),
        f"📥 ENTRY ZONE:   ${entry_low:.6f} → ${entry_high:.6f}",
        f"   {zone_status}",
        # Legacy stale-zone warning removed — zone_status covers this case

        f"🛑 STOP LOSS:  ${r['stop_loss']:.6f}",
        f"🎯 TARGET 1:   ${r['t1']:.6f}",
        f"🎯 TARGET 2:   ${r['t2']:.6f}",
        f"🎯 TARGET 3:   ${r['t3']:.6f}",
        f"",
        f"📊 RSI 4H:    {r['rsi4']:.1f}",
        f"📊 RSI Daily: {r['rsi_d']:.1f}  (closed candle)",
        f"📊 Stoch K:   {r['stoch_k']:.1f}",
        f"📊 Funding:   {r['funding']:.4f}%",
        f"📊 ATR:       ${r['atr']:.6f}",
    ]

    if lev and conf >= 8:
        note = {10: "★ 10/10 — Full leverage authorized",
                9:  "★ 9/10 — Near-max leverage authorized"}.get(conf,
                    "★ 8/10 — Conservative leverage applied")
        lines += [
            f"",
            f"⚡ LEVERAGE (Isolated Margin)",
            f"   Suggested:        {lev['suggested']}x",
            f"   Max Safe:         {lev['max_safe']}x",
            f"   Volatility:       {lev['vol_label']} ({lev['atr_pct']:.2f}% ATR)",
            f"   SL Distance:      {lev['sl_dist']:.2f}%",
            f"   Liq Distance:     ~{lev['liq_dist']:.2f}%",
            f"   Fluctuation Room: {lev['fluct']:.2f}%",
            f"   {note}",
        ]
    elif conf < 8:
        lines += [f"", f"⚠️ LEVERAGE: Not recommended (confidence < 8/10)"]

    lines += [f"", f"✅ CONVICTION REASONS:"]
    for reason in r['reasons']:
        lines.append(f"   • {reason}")

    if dur_reasons:
        lines += [f"", f"🕐 DURATION ANALYSIS:"]
        for reason in dur_reasons:
            lines.append(f"   • {reason}")

    # FIX #DD — Portfolio correlation context
    corr_type    = r.get('corr_type', 'independent')
    corr_slot    = r.get('corr_slot', 1)
    corr_flagged = r.get('corr_flagged', False)
    cap          = 3   # CORRELATED_SLOT_CAP
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
            lines += [
                f"",
                f"📊 PORTFOLIO SLOT — Correlated {direction} #{corr_slot}/{cap}",
            ]
    else:
        lines += [f"", f"📊 PORTFOLIO SLOT — Independent signal (counter-regime or NEUTRAL)"]

    # IMPROVEMENT #8 — Exchange deeplink
    lines += [f"", f"🔗 Trade on {exchange}: {link}"]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# TRADE REMINDER JOB
# ─────────────────────────────────────────────
async def send_trade_update(context: ContextTypes.DEFAULT_TYPE):
    chat_id  = context.job.chat_id
    trade_id = (context.job.data or {}).get('trade_id')
    trades   = user_tracking.get(chat_id, {})
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
        if current <= signal['t3']: alerts.append("🎯🎯🎯 TARGET 3 HIT!")
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
        f"🎯 T1:      ${signal['t1']:.6f}\n\n"
        f"{pnl_emoji} PnL: {pnl_pct:+.2f}%\n"
        f"⏱ Open: {hours:.1f} hours\n"
    )
    if alerts:
        msg += f"\n{''.join(alerts)}\n"

    max_hold   = signal.get('hold_hours', 24)
    hold_label = signal.get('hold', f'{max_hold}h')
    warn_at    = max_hold * 0.85
    if hours >= max_hold:
        msg += f"\n🚨 RECOMMENDED HOLD ({hold_label}) REACHED. Close this trade now!\n"
    elif hours >= warn_at:
        msg += f"\n⚠️ Approaching recommended hold duration ({hold_label}). Consider closing soon.\n"

    stop_cb = f"stop_trade|{chat_id}|{trade_id}" if trade_id else "stop_trade_all"
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🛑 Stop Tracking", callback_data=stop_cb),
        InlineKeyboardButton("🔗 Trade Now", url=link),
    ]])
    msg += f"\n🔗 {link}"
    await context.bot.send_message(chat_id=chat_id, text=msg, reply_markup=keyboard)


# ─────────────────────────────────────────────
# CONVERSATION: PICK → REMINDER → INTERVAL
# ─────────────────────────────────────────────
async def pick_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    if not last_scan_results:
        await update.message.reply_text("⚠️ No scan data yet. Run /scan first.")
        return ConversationHandler.END

    total = len(last_scan_results)
    lines = [f"📊 LAST SCAN — {total} signals\n"]
    for i, r in enumerate(last_scan_results[:20], 1):
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
        if n < 1 or n > len(last_scan_results):
            await update.message.reply_text(f"⚠️ Enter a number between 1 and {len(last_scan_results)}.")
            return PICK_TRADE
        signal = last_scan_results[n - 1]
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
        if chat_id not in user_tracking:
            user_tracking[chat_id] = {}
        user_tracking[chat_id][trade_id] = {
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
        user_tracking[chat_id][trade_id]['job'] = job
        active_count = len(user_tracking[chat_id])

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
    trades  = user_tracking.get(chat_id, {})

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
        trades  = user_tracking.pop(chat_id, {})
        for tid, data in trades.items():
            try:
                if data.get('job'): data['job'].schedule_removal()
            except Exception: pass
            db_remove_trade(tid)
            # FIX: clear flip-cooldown so stopped pairs can be re-scanned immediately
            stopped_sym = data.get('signal', {}).get('symbol', '')
            if stopped_sym:
                _last_signal_bias.pop(stopped_sym, None)
        await query.edit_message_text(f"✅ All {len(trades)} trade(s) stopped.")

    elif parts[0] == "stop_trade":
        chat_id  = int(parts[1])
        trade_id = parts[2]
        trades   = user_tracking.get(chat_id, {})
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
            if stopped_sym and stopped_sym in _last_signal_bias:
                _last_signal_bias.pop(stopped_sym, None)
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
    trades  = user_tracking.get(chat_id, {})

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
# ─────────────────────────────────────────────

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
# ─────────────────────────────────────────────
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 SCENARIO ANALYSIS",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"🛑 Stop Loss   ${sl:.6f}",
        f"   Move: {sl_pct:+.2f}%  →  {'🔴 -$' if sl_pnl < 0 else '🟢 +$'}{abs(sl_pnl):,.2f}",
        f"",
        f"🎯 Target 1    ${t1:.6f}",
        f"   Move: {t1_pct:+.2f}%  →  🟢 +${t1_pnl:,.2f}",
        f"",
        f"🎯 Target 2    ${t2:.6f}",
        f"   Move: {t2_pct:+.2f}%  →  🟢 +${t2_pnl:,.2f}",
        f"",
        f"🎯 Target 3    ${t3:.6f}",
        f"   Move: {t3_pct:+.2f}%  →  🟢 +${t3_pnl:,.2f}",
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if bot_lev and not custom:
        lines.append(f"ℹ️ Bot suggested {bot_lev}x based on ATR & confidence.")
    elif custom:
        if bot_lev:
            lines.append(f"ℹ️ Bot suggested leverage: {bot_lev}x (you used {leverage}x).")
        lines.append(f"⚠️ Always use isolated margin with custom leverage.")

    lines.append(f"\n🔗 {get_exchange_link(exchange, symbol)}")
    return "\n".join(lines)


async def pnl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entry point — user picks a signal to calculate PnL for."""
    _track(update)
    global last_scan_results
    if not last_scan_results:
        await update.message.reply_text("⚠️ No scan data yet. Run /scan first.")
        return

    total = len(last_scan_results)
    lines = [f"💰 PNL CALCULATOR\n\nPick a signal:\n"]
    for i, r in enumerate(last_scan_results[:20], 1):
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
            if n < 1 or n > len(last_scan_results):
                await update.message.reply_text(f"⚠️ Enter 1–{len(last_scan_results)}.")
                return
            context.user_data['pnl_signal'] = last_scan_results[n - 1]
            context.user_data['pnl_step']   = 'capital'
            await update.message.reply_text(
                "💵 How much capital are you trading with? (USDT)\n"
                "Examples: 50  100  500  1000"
            )
        except ValueError:
            await update.message.reply_text("⚠️ Reply with a number only.")

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
# ─────────────────────────────────────────────
async def best_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    if not last_scan_results:
        await update.message.reply_text("⚠️ No scan data yet. Run /scan first.")
        return
    best = last_scan_results[0]
    age  = datetime.now() - last_scan_time
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
# ─────────────────────────────────────────────
async def tg_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _track(update)
    ph = db_load_price_history() if not price_history else price_history
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
    top = [g for g in gainers if g['change_pct'] > 0][:10]
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
    ph = db_load_price_history() if not price_history else price_history
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
    top = [g for g in losers if g['change_pct'] < 0][:10]
    if not top:
        await update.message.reply_text("📊 No losses recorded yet. Run /scan more times.")
        return
    lines = [f"📉 TOP LOSSES — Last 24 Hours\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]
    for i, g in enumerate(top, 1):
        lines.append(f"🔴 #{i} {g['exchange']} | {g['symbol']}\n"
                     f"   Change: {g['change_pct']:.2f}%\n"
                     f"   Then: ${g['price_then']:.6f}  Now: ${g['price_now']:.6f}\n")
    await update.message.reply_text("\n".join(lines))


# ─────────────────────────────────────────────
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
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "• *Full Scan* — analyses top 50 pairs on Bybit, MEXC & Binance\n"
            "  ⏱ Runs on the *4H timeframe* — swing trade signals only\n\n"
            "• *Custom Pair Scan* — analyse any coin on *your preferred timeframe*\n"
            "  `/cscan ZEC`       auto-detects the strongest timeframe\n"
            "  `/cscan ZEC 15m`   15-min chart → scalp signals (mins–2h)\n"
            "  `/cscan ZEC 1h`    1H chart → intraday signals (30min–8h)\n"
            "  `/cscan ZEC 4h`    4H chart → swing signals (4h–3 days)\n"
            "  `/cscan ZEC 1d`    Daily chart → position signals (1d–2wks)\n\n"
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
            "• *Broadcast* — auto-post signals to your group or channel\n"
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
            "• Auto-calculates suggested leverage per signal\n"
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


# ─────────────────────────────────────────────
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
        "best":        "/best — fetching best signal now...",
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
    tracking = user_tracking.get(chat_id)

    if last_scan_time:
        age  = datetime.now() - last_scan_time
        mins = int(age.total_seconds() // 60)
        longs  = sum(1 for r in last_scan_results if r['bias'] == 'LONG')
        shorts = sum(1 for r in last_scan_results if r['bias'] == 'SHORT')
        scan_info = (
            f"🕐 Last scan: {last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"⏱ {mins} minutes ago\n"
            f"📊 Signals: {len(last_scan_results)}  🟢{longs}  🔴{shorts}"
        )
    else:
        scan_info = "⚠️ No scan run yet."

    # FIX #SCANTIME — Scan duration stats
    if _scan_durations:
        recent   = _scan_durations[-10:]
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
    regime = _btc_regime_cache['regime'] if _btc_regime_cache else "unknown"
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
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# COMPACT SIGNAL CARDS — inline button display
# ─────────────────────────────────────────────
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
        pool = last_scan_results
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

    if not last_scan_results or idx >= len(last_scan_results):
        await query.message.reply_text("⚠️ Signal expired. Run /scan again.")
        return

    signal = last_scan_results[idx]
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
        if _scan_cache:
            age_secs = (datetime.now() - _scan_cache['time']).total_seconds()
            if age_secs < CACHE_TTL_SECS:
                mins_old = int(age_secs // 60)
                await update.message.reply_text(
                    f"⚡ Serving cached scan ({mins_old} min old — refreshes every 15 min)\n"
                    f"Use /scan again after 15 min for a fresh scan."
                )
                results = _scan_cache['results']
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

        longs        = sum(1 for r in results if r['bias'] == 'LONG')
        shorts       = sum(1 for r in results if r['bias'] == 'SHORT')
        high_conf    = sum(1 for r in results if r['confidence'] >= 8)
        bybit_count  = sum(1 for r in results if r.get('exchange') == 'BYBIT')
        mexc_count   = sum(1 for r in results if r.get('exchange') == 'MEXC')
        bin_count    = sum(1 for r in results if r.get('exchange') == 'BINANCE')
        bybit_note   = f"BYBIT: {bybit_count}" if sakz_exchanges.BYBIT_AVAILABLE else "BYBIT: skipped (blocked)"
        binance_note = f"BINANCE: {bin_count}"  if sakz_exchanges.BINANCE_AVAILABLE else "BINANCE: skipped (blocked)"
        cache_note   = "⚡ cached" if from_cache else "🔄 fresh"

        # Summary folded into send_signal_cards title
        pass

        if not from_cache:
            await notify_alerts(results, context.bot)
            await post_broadcast(results, context.bot)

        await send_signal_cards(
            update.message, results,
            title=f"📊 TOP SIGNALS — {last_scan_time.strftime('%H:%M')}",
            max_show=20,
            chat_id=chat_id,
            source="scan"
        )


# ─────────────────────────────────────────────
# /top[n]
# ─────────────────────────────────────────────
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

    if not last_scan_results:
        await update.message.reply_text("⏳ No scan yet. Fetching now...")
        results, _ = await get_scan_results()
    else:
        results = last_scan_results
        age     = datetime.now() - last_scan_time
        mins    = int(age.total_seconds() // 60)
        cache_note = "⚡ cached" if mins < 15 else "🕐 old"
        await update.message.reply_text(
            f"📊 Scan from {mins} min ago ({cache_note})\n"
            f"Showing top {min(n, len(results))} of {len(results)} signals…"
        )

    if not results:
        await update.message.reply_text("⚠️ No signals. Try /scan first.")
        return

    chat_id  = update.effective_chat.id
    actual_n = min(n, len(results))
    signals  = results[:actual_n]
    now_str  = datetime.now().strftime('%H:%M:%S')
    age_min  = int((datetime.now() - last_scan_time).total_seconds() // 60) if last_scan_time else 0

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


# ─────────────────────────────────────────────
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
                f"• Wait ~1–2 hours and try again — new listings fill up fast\n"
                f"• Try /cscan {sym_base} 15m once more candles accumulate\n"
                f"• Check the pair exists as a perpetual on MEXC/Bybit futures"
            )
        elif dominant_reason == REASON_REGIME_BLOCK:
            btc_regime = get_btc_regime()
            regime_details = next((f.detail for f in failures
                                   if f.reason == REASON_REGIME_BLOCK), "")
            # Show what the signal would look like anyway, with a clear warning
            diag_msg = (
                f"⚠️ {symbol} — Signal below BTC Regime floor\n\n"
                f"BTC Regime: {btc_regime}\n"
                f"Detail: {regime_details}\n\n"
                f"The signal was found but scored below the recommended confidence\n"
                f"floor for the current regime. It is shown below with a risk warning.\n\n"
                f"💡 Trade with reduced size and tighter risk management.\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            await update.message.reply_text(diag_msg)
            # Fall through — don't return, let the code below try to display the signal
            # by re-running with regime_warning attached to result
            # We need to force a re-scan that returns the result regardless of regime
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
                    'title':    f"📡 {best_r['symbol']}  [{tf_display}]",
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
                f"⚠️ {symbol} — Low confidence signal\n\n"
                f"The scoring engine found a signal but confidence is below the\n"
                f"recommended threshold ({best_detail or 'indicators disagreed'}).\n\n"
                f"Shown below with a risk warning. Treat as informational only.\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            await update.message.reply_text(diag_msg)
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
    _chat_scan_ctx[chat_id] = {
        'source':   'custom',
        'results':  results,
        'title':    f"📡 {best['symbol']}",
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
        [InlineKeyboardButton("🔄 Refresh (live price + PnL)",
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
            f"📡 CSCAN — {exchange} | {symbol}\n"
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
# ─────────────────────────────────────────────
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
        if not last_scan_results:
            await query.edit_message_text("⚠️ No scan data. Run /scan first.")
            return
        signals_pool = last_scan_results
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
    age_min = int((datetime.now() - last_scan_time).total_seconds() // 60) if last_scan_time and source == 'scan' else 0

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
# ─────────────────────────────────────────────
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
            for r in last_scan_results:
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
    if not last_scan_results:
        await update.message.reply_text("⚠️ No scan yet. Run /scan first.")
        return

    age   = datetime.now() - last_scan_time
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

    for i, r in enumerate(last_scan_results[:15], 1):
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

    if not last_scan_results:
        try:
            await query.edit_message_text("⚠️ No scan data. Run /scan first.")
        except Exception:
            pass
        return

    age   = datetime.now() - last_scan_time
    mins  = int(age.total_seconds() // 60)
    now_s = datetime.now().strftime('%H:%M:%S')

    lines     = [f"📊 FULL COMPARE — {now_s}\n🕐 Scan: {mins} min ago\n{'━'*30}\n"]
    winners   = 0
    losers    = 0
    best_pnl  = ('', -9999)
    worst_pnl = ('', 9999)

    parts   = query.data.split('|')
    chat_id = int(parts[1])

    for i, r in enumerate(last_scan_results[:15], 1):
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
    lines.append(f"🟢 Winning: {winners}  🔴 Losing: {losers}  📊 Tracked: {total_done}")
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
# ─────────────────────────────────────────────
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
# ─────────────────────────────────────────────
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
            lines = ["👁 YOUR WATCHLIST\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]
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

    if chat_id in safemode_users:
        # Turn OFF
        safemode_users.discard(chat_id)
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
        safemode_users.add(chat_id)
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
    if not last_scan_results:
        return
    all_watches = db_get_all_watchlist()
    if not all_watches:
        return

    # Build symbol map from last scan (exchange-agnostic)
    sym_map = {}
    for r in last_scan_results:
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
            except Exception:
                pass

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
            except Exception:
                pass

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


# ─────────────────────────────────────────────
# BTC VOLATILITY MONITOR — Smart re-scan trigger
# Checks BTC price every 15 min.
# If BTC moved >3% in 1 hour → emergency re-scan
# ─────────────────────────────────────────────
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
        f"🔄 Triggered emergency market scan\n"
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
# ─────────────────────────────────────────────
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
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


# ─────────────────────────────────────────────
# /leaderboard [hours] — Best performing pairs
# Usage: /leaderboard          → all time
#        /leaderboard 24       → last 24 hours
#        /leaderboard 168      → last 7 days
# ─────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════
# 🐌  S N A I L   M O D E  — HIDDEN PREMIUM FEATURE
# ═══════════════���═══════════════════════════════════════════════
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
                score += 15; reasons.append("✅ Daily EMA stack bearish — macro trend aligned")
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

        # ── 4. VOLUME CONVICTION ─────────────────────────────
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

        # ── 5. CANDLE PATTERN QUALITY ────────────────────────
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
                score += 8; reasons.append(f"✅ Wide runway to resistance — {r_space:.1f}x ATR of clear space")
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

        # ── 7. FUNDING RATE ANALYSIS ─────────────────────────
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
            cg_resp = requests.get(cg_url, timeout=8, headers=HEADERS)
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
    bias_e = "🟢" if r['bias'] == 'LONG' else "🔴"
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
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
    elite_candidates = [r for r in last_scan_results if r.get('confidence', 0) == 10]
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
                    for cached_r in last_scan_results:
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
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"No signal reached SNAIL standards today.\n"
                    f"Requirements: 10/10 confidence + Snail Score ≥ 80 + Low manipulation risk\n\n"
                    f"📊 Scanned: {len(last_scan_results)} pairs\n"
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
        f"━━━━━━━━━━━━━━━━━━���━━━━━━━━━━━",
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
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
    snail_unlocked.add(chat_id)
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
    if chat_id not in snail_unlocked and not db_snail_is_unlocked(chat_id):
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Status: {status_str}\n\n"
        f"Goal: 2x per day | Criteria: 10/10 + Snail Score ≥ 80\n"
        f"Use /snailvault to view your signal log.\n",
        reply_markup=keyboard
    )


async def snailvault_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/snailvault — completely silent if not unlocked."""
    _track(update)
    chat_id = update.effective_chat.id
    if chat_id not in snail_unlocked and not db_snail_is_unlocked(chat_id):
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

    lines = ["🐌 SNAIL VAULT — Signal Log\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]
    outcome_map = {'win_2x': '🎯 WIN 2x', 'stopped': '🛑 Stopped', 'pending': '⏳ Open'}
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

    if chat_id not in snail_unlocked and not db_snail_is_unlocked(chat_id):
        return

    if data == "snail_activate":
        db_snail_start_session(chat_id)
        snail_active[chat_id] = True
        snail_unlocked.add(chat_id)
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





# ═══════════════════════════════════════════════════════════════
# ── FEATURE BLOCK — v5 additions ────────────────────────────────
# 1. Timeframe argument for /scan and /cscan  (e.g. /scan t15m)
# 2. Trend-dying notification job
# 3. Snail 2x target tied to actual leverage
# 4. /chart — generate TA chart image
# 5. /lb alias for /leaderboard
# 6. Admin user-count tracking (/admin)
# ═══════════════════════════════════════════════════════════════

# ─── TIMEFRAME HELPERS ────────────────────────────────────────
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
                    r_spot    = requests.get(
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


# ─── PATCHED scan_command (timeframe support) ─────────────────
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

    # ── /scan [pair] or /scan [pair] [tf] ─────────────────────
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
            # ── Real reason diagnosis ─────────────────────────────────────
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
                    f"🟠 *{symbol}* — Signal too weak to publish\n\n"
                    f"A signal was found but confidence is below the minimum threshold.\n\n"
                    f"Detail: `{lc_d}`\n\n"
                    f"💡 Options:\n"
                    f"• Try `/scan {sym_arg} 1h` — shorter TF may be stronger\n"
                    f"• Market may be consolidating — check back after next candle\n"
                    f"• Use `/chart {sym_arg} {actual_tf}` to review manually"
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
            else:
                # NEUTRAL or unknown — market has no clear directional bias
                msg = (
                    f"😐 *{symbol}* — Market too neutral on {tf_label}\n\n"
                    f"Indicators are balanced — no clear LONG or SHORT edge right now.\n"
                    f"This is normal in sideways / consolidating markets.\n\n"
                    f"💡 Try:\n"
                    f"• `/scan {sym_arg} 1h` — shorter TF may be trending\n"
                    f"• `/chart {sym_arg} {actual_tf}` — read the chart yourself\n"
                    f"• `/scan` — find other pairs with active momentum\n"
                    f"• Check back after the next candle close"
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
            banner_lines.append(f"⚠️ Regime caution: {regime_warn}")
        if low_conf_warn:
            banner_lines.append(f"⚠️ Low confidence: {low_conf_warn}")
        if regime_warn or low_conf_warn:
            banner_lines.append("")
            banner_lines.append("_Signal shown as requested. Trade with caution and manage risk accordingly._")

        await update.message.reply_text("\n".join(banner_lines), parse_mode="Markdown")

        # Store for refresh / compare
        cscan_results[chat_id] = results
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
        r    = requests.get("https://contract.mexc.com/api/v1/contract/ticker",
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
                r2 = requests.get(
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
        r    = requests.get("https://api.bybit.com/v5/market/instruments-info",
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
      /scan new 24h   — pairs listed in last 24 hours
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


# ─── TREND-DYING MONITOR ──────────────────────────────────────
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
                    footer = "🛡️ Safe Mode alert — trend detected from your recent scan.\nUse /safemode to disable."
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
    for chat_id, trades in list(user_tracking.items()):
        for trade_id, tdata in list(trades.items()):
            try:
                sig = tdata.get('signal', {})
                await _check_and_notify(chat_id, sig, source_label='tracked')
            except Exception:
                continue

    # ── Pool 3: Safe Mode users — any signal from any scan ────────────────────
    for chat_id in list(safemode_users):
        signals = safemode_last_signals.get(chat_id, [])
        for sig in signals:
            try:
                await _check_and_notify(chat_id, sig, source_label='safemode')
            except Exception:
                continue


# ─── SNAIL 2x TARGET — leverage-aware ─────────────────────────
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


# ─── /chart COMMAND ───────────────────────────────────────────
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

    if not last_scan_results or idx >= len(last_scan_results):
        await query.message.reply_text("⚠️ Signal data expired. Run /scan again.")
        return

    signal = last_scan_results[idx]
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

    png = await loop.run_in_executor(SCAN_EXECUTOR, lambda: generate_chart(signal, df4h))
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

    try:
        _, symbol, tf = query.data.split('|')
    except ValueError:
        await query.message.reply_text("⚠️ Could not parse chart refresh data.")
        return

    await query.message.reply_text(
        f"📊 Refreshing {symbol} chart on {_tf_display(tf)}...\n⏳ Please wait..."
    )

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
        await query.message.reply_photo(photo=buf, caption=caption, reply_markup=kb)

    except Exception as e:
        logger.error("chart_tf_refresh_callback %s %s: %s", symbol, tf, e)
        await query.message.reply_text(f"❌ Chart refresh failed: {e}")


# ─── USER TRACKING ────────────────────────────────────────────
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


# ── Top-level activity middleware (registered in main with group=-1) ──
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
        f"   🆕 New today:            {len(new_today)}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 CURRENTLY ACTIVE ({len(active_now)})\n"
        f"{active_block}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🆕 NEW TODAY ({len(new_today)})\n"
        f"{new_block}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 TOP USERS BY ACTIVITY\n"
        f"{top_block}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 BOT STATUS\n"
        f"   Last scan signals: {len(last_scan_results)}\n"
        f"   Tracking active:   {len(user_tracking)}\n"
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
            f"🆕 NEW TODAY ({len(new_today)})\n"
            f"{new_block}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏆 TOP USERS BY ACTIVITY\n"
            f"{top_block}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📡 BOT STATUS\n"
            f"   Last scan signals: {len(last_scan_results)}\n"
            f"   Tracking active:   {len(user_tracking)}\n"
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
# ─���───────────────────────────────────────────
def _fetch_fgi():
    """Fetch Fear & Greed Index from alternative.me. Returns dict or None."""
    try:
        r    = requests.get("https://api.alternative.me/fng/?limit=7",
                            headers=HEADERS, timeout=10)
        data = r.json()
        return data.get('data', [])
    except Exception as e:
        logger.warning("FGI fetch error: %s", e)
        return None

def _fgi_bar(value):
    """Visual bar for FGI value 0-100."""
    filled = int(value / 10)
    empty  = 10 - filled
    return "█" * filled + "░" * empty

def _fgi_interpretation(value, classification):
    """Return trading guidance based on FGI value."""
    if value <= 10:
        return (
            "🔴 EXTREME FEAR\n\n"
            "📌 Market is in extreme panic. Historically this is when\n"
            "   the best LONG opportunities appear.\n\n"
            "✅ LONG bias favoured — but confirm with /scan first.\n"
            "⚠️ Catching a falling knife is risky — wait for reversal signals."
        )
    elif value <= 25:
        return (
            "🔴 FEAR\n\n"
            "📌 Market sentiment is bearish. Sellers dominate.\n\n"
            "✅ Cautious LONG opportunities may exist on strong support.\n"
            "✅ SHORT signals in this zone tend to be late — avoid chasing.\n"
            "🎯 Best approach: /filter LONG 8 to find high-conviction longs."
        )
    elif value <= 45:
        return (
            "🟡 INDECISIVE / NEUTRAL (Leaning Fear)\n\n"
            "📌 Market is uncertain. No clear directional dominance.\n\n"
            "⚠️ This is a choppy zone — signals have higher failure rate.\n"
            "🎯 Stick to signals with confidence ≥ 8/10 and short hold durations.\n"
            "💡 /filter LONG 8 or /filter SHORT 8 recommended."
        )
    elif value <= 55:
        return (
            "🟡 NEUTRAL\n\n"
            "📌 Market has no strong bias either way.\n\n"
            "⚠️ Indecisive conditions — best to trade only the highest conviction signals.\n"
            "🎯 Use /best and only take 9–10/10 confidence signals.\n"
            "💡 Both LONG and SHORT can work — let the TA decide."
        )
    elif value <= 70:
        return (
            "🟠 GREED\n\n"
            "📌 Market participants are optimistic. Buyers dominate.\n\n"
            "✅ LONG signals in this zone can ride the momentum well.\n"
            "⚠️ Approaching reversal territory — avoid over-leveraged longs.\n"
            "🎯 Consider LONG entries but tighten stop losses."
        )
    elif value <= 90:
        return (
            "🔴 EXTREME GREED\n\n"
            "📌 Market is overheated. Euphoria is high.\n\n"
            "⚠️ LONG entries here are late and risky — smart money is selling.\n"
            "✅ SHORT bias favoured — look for distribution signals.\n"
            "🎯 Use /filter SHORT 8 for high-confidence short signals."
        )
    else:
        return (
            "🔴 MAXIMUM GREED\n\n"
            "📌 Market is at peak euphoria. Contrarian SHORT setups are highest probability.\n\n"
            "✅ Strong SHORT bias — this is historically a major top zone.\n"
            "⚠️ Do NOT open new longs here without strong TA confirmation.\n"
            "🎯 /filter SHORT 9 recommended."
        )

async def fgi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /fgi    — current Fear & Greed Index + 7-day history + trading guidance
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

    # Current reading
    current   = data[0]
    value     = int(current['value'])
    classif   = current['value_classification']
    ts        = datetime.fromtimestamp(int(current['timestamp']))
    guidance  = _fgi_interpretation(value, classif)
    bar       = _fgi_bar(value)

    # 7-day history trend
    history_lines = []
    for d in data[:7]:
        v     = int(d['value'])
        c     = d['value_classification']
        dt    = datetime.fromtimestamp(int(d['timestamp'])).strftime('%b %d')
        hbar  = "█" * int(v/10) + "░" * (10-int(v/10))
        emoji = "🟢" if v <= 40 else ("🟡" if v <= 60 else "🔴")
        history_lines.append(f"{emoji} {dt}  {hbar}  {v:>3}  {c}")

    # Trend direction
    if len(data) >= 2:
        delta = value - int(data[1]['value'])
        trend = f"📈 +{delta} (improving)" if delta > 0 else (f"📉 {delta} (worsening)" if delta < 0 else "➡️ unchanged")
    else:
        trend = "N/A"

    msg = (
        f"😨 FEAR & GREED INDEX\n"
        f"{'━'*30}\n\n"
        f"📅 {ts.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"{bar}  {value}/100\n"
        f"Classification: {classif}\n"
        f"24h Change: {trend}\n\n"
        f"{'━'*30}\n"
        f"📊 7-DAY HISTORY\n"
        f"{'━'*30}\n"
        f"{chr(10).join(history_lines)}\n\n"
        f"{'━'*30}\n"
        f"🧭 TRADING GUIDANCE\n"
        f"{'━'*30}\n"
        f"{guidance}\n\n"
        f"{'━'*30}\n"
        f"ℹ️ Data: alternative.me/fng\n"
        f"Updates every ~1 hour."
    )

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔄 Refresh FGI",  callback_data="fgi_refresh"),
        InlineKeyboardButton("🔍 Scan Now",     callback_data="menu_run|scan")
    ]])
    await update.message.reply_text(msg, reply_markup=keyboard)


async def fgi_refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Refresh button on FGI card."""
    query = update.callback_query
    await query.answer("Fetching latest FGI...")
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_fgi)
    if not data:
        await query.answer("⚠️ FGI API unavailable", show_alert=True)
        return
    current  = data[0]
    value    = int(current['value'])
    classif  = current['value_classification']
    ts       = datetime.fromtimestamp(int(current['timestamp'])).strftime('%H:%M:%S')
    bar      = _fgi_bar(value)
    guidance = _fgi_interpretation(value, classif)
    msg = (
        f"😨 FEAR & GREED — REFRESHED [{ts}]\n"
        f"{'━'*30}\n\n"
        f"{bar}  {value}/100\n"
        f"{classif}\n\n"
        f"{'━'*30}\n"
        f"{guidance}"
    )
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔄 Refresh again", callback_data="fgi_refresh"),
        InlineKeyboardButton("🔍 Scan Now",      callback_data="menu_run|scan")
    ]])
    try:
        await query.edit_message_text(msg, reply_markup=keyboard)
    except Exception:
        await query.message.reply_text(msg, reply_markup=keyboard)


# ─────────────────────────────────────────────────────────────────────────────
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
        f"⏳ Please wait 60–90 seconds…"
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
        title=f"🔍 MID-TIER SIGNALS (rank {rank_from}–{rank_to})",
        max_show=15, chat_id=chat_id, source="scan"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CEILING #5 — EMPIRICAL PARAMETER CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────

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

    # ── Section 1 — ATR T1 multiplier sweep (at conf≥5, baseline) ─────────────
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

    # ── Section 3 — Vol regime breakdown (current params, conf≥5) ─────────────
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
            f"📈 WIDER T1 (×1.25) has higher EV ({wide_ev:+.3f} vs {cur_ev:+.3f}) — "
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
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
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "/pick                  Track a trade with auto-reminders\n"
        "/stoptrade             Stop tracking your current trade\n"
        "/check BTC LONG 98000 95000\n"
        "                       Validate an open trade vs live data\n"
        "/pnl                   PnL calculator\n"
    )

    analytics = (
        "\n📊  A N A L Y T I C S\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
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


# ─────────────────────────────────────────────
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
        f"⏳ Please wait ~60 seconds..."
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 {len(deduped)} signals  🟢 {longs}L  🔴 {shorts}S\n"
        f"⏱ 15M: {tf_15}   1H: {tf_1h}\n\n"
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
# ─────────────────────────────────────────────
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


# ──���──────────────────────────────────────────
# /check PAIR DIR ENTRY SL — Validate open trade
# Example: /check BTCUSDT LONG 98000 95000
# ─────────────────────────────────────────────
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

    # Get live price
    live_price = 0
    try:
        live_price = _get_live_price(symbol, exchange)
    except Exception:
        pass

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
# ─────────────────────────────────────────────
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
    except Exception:
        pass

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


# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
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
# ─────────────────────────────────────────────
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
        "🌳 Starting Random Forest training...\n"
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


# ─────────────────────────────────────────────
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
        lines = ["📋 *Paper Trading — Last 20 Closed*", ""]
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


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    global last_scan_results, last_scan_time, price_history, user_tracking

    print("=" * 55)
    print("  SAKZ SCAN BOT v2 — Starting")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # MEMORY — restore user data from JSON backup before DB init
    sakz_memory.restore_on_startup()

    # IMPROVEMENT #2 — init DB and restore state from last run
    db_init()
    db_init_user_tracking()
    _load_best_params()   # load calibrated params from best_params.json if present

    # AUTO PAPER TRADING — initialise paper DB tables if module is available
    if _PAPER_AVAILABLE:
        paper_init_db(db_connect)
        logger.info("Paper trading DB tables ready")
    restored, ts = db_load_last_scan()
    if restored:
        last_scan_results = restored
        last_scan_time    = ts
        logger.info("Restored %d signals from DB (scan at %s)", len(restored), ts)
    price_history = db_load_price_history()
    logger.info("Loaded %d price history entries from DB", len(price_history))

    # FIX #PERSIST-BIAS — restore flip-cooldown state from DB
    global _last_signal_bias
    _last_signal_bias = db_load_signal_bias()
    logger.info("Restored %d signal bias entries from DB", len(_last_signal_bias))

    # FIX #PERSIST-CACHE — restore signal card cache from DB (last 24h only)
    global _signal_card_cache
    _signal_card_cache = db_load_card_cache()
    logger.info("Restored %d signal card cache entries from DB", len(_signal_card_cache))

    # SAFE MODE — restore users who had it enabled before restart
    global safemode_users
    safemode_users = db_safemode_load()
    logger.info("Restored %d safemode users from DB", len(safemode_users))

    app = Application.builder().token(TELEGRAM_TOKEN).build()

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
        user_tracking[chat_id] = trades
        for trade_id, data in trades.items():
            job = app.job_queue.run_repeating(
                send_trade_update,
                interval=data['interval'] * 60,
                first=data['interval'] * 60,
                chat_id=chat_id,
                name=f"trade_{chat_id}_{trade_id}",
                data={'trade_id': trade_id}
            )
            user_tracking[chat_id][trade_id]['job'] = job
        logger.info("Restored %d trade(s) for chat_id %s", len(trades), chat_id)

    # Price-level alert checker — every 3 minutes
    app.job_queue.run_repeating(
        price_alert_job,
        interval=180,
        first=60,
        name="price_alert_checker"
    )

    # ── 🐌 SNAIL MODE — restore unlocked users from DB ───────────────────────
    global snail_unlocked
    snail_unlocked = db_snail_load_unlocked()
    logger.info("Restored %d snail-unlocked users", len(snail_unlocked))

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
    app.add_handler(CallbackQueryHandler(view_signal_callback,             pattern=r'^view_signal\|'))
    app.add_handler(CallbackQueryHandler(pnl_from_signal_callback,         pattern=r'^pnl_from_signal\|'))
    app.add_handler(CallbackQueryHandler(pnl_from_cscan_callback,          pattern=r'^pnl_from_cscan\|'))
    app.add_handler(CallbackQueryHandler(cscan_refresh_callback,           pattern=r'^cscan_refresh\|'))
    app.add_handler(CallbackQueryHandler(cscan_tf_callback,                pattern=r'^cscan_tf\|'))
    app.add_handler(CallbackQueryHandler(feed_refresh_callback,            pattern=r'^feed_refresh\|'))
    app.add_handler(CallbackQueryHandler(autoscan_tf_callback,             pattern=r'^autoscan_tf\|'))
    app.add_handler(CallbackQueryHandler(stats_time_callback,              pattern=r'^stats_time\|'))
    app.add_handler(CallbackQueryHandler(bt_time_callback,                 pattern=r'^bt_time\|'))
    app.add_handler(CallbackQueryHandler(leaderboard_time_callback,        pattern=r'^lb_time\|'))
    app.add_handler(CallbackQueryHandler(custom_compare_refresh_callback,  pattern=r'^custom_compare_refresh\|'))
    app.add_handler(CallbackQueryHandler(compare_refresh_callback,         pattern=r'^cmp_refresh\|'))
    app.add_handler(CallbackQueryHandler(compare_full_refresh_callback,    pattern=r'^cmp_full_refresh\|'))
    app.add_handler(CallbackQueryHandler(fgi_refresh_callback,             pattern=r'^fgi_refresh'))
    # 🐌 SNAIL callbacks — handles activate/status/report/stop buttons
    app.add_handler(CallbackQueryHandler(snail_callback_handler,    pattern=r'^snail_'))
    # 📋 MENU interactive button callbacks
    app.add_handler(CallbackQueryHandler(menu_callback_handler,     pattern=r'^menu\|'))
    app.add_handler(CallbackQueryHandler(menu_run_callback,         pattern=r'^menu_run\|'))
    app.add_handler(MessageHandler(filters.Regex(r'^/top\d+') & filters.TEXT, top_command))
    # 🐌 SNAIL commands — visible only to unlocked users
    app.add_handler(CommandHandler("snail",      snail_command))
    app.add_handler(CommandHandler("snailvault", snailvault_command))

    # Admin password entry — must come BEFORE pnl_message_handler
    async def _admin_pw_gate(update: Update, context: ContextTypes.DEFAULT_TYPE):
        consumed = await admin_password_handler(update, context)
        if not consumed:
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
