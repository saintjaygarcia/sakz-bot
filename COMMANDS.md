# Sakz Scan Bot — Complete Command Reference

This is the full list of every command registered in the bot, organised by category.
It covers regular user commands, hidden/secret commands, diagnostics, operator
commands, and admin/maintenance commands.

---

## How admin access works

A command marked **🔒 ADMIN** only runs for an authenticated admin. You become an
admin in either of two ways:

1. **By chat ID** — your Telegram chat ID is listed in the `ADMIN_IDS` environment
   variable (comma-separated). These users are allowed silently.
2. **By password** — run `/admin` and enter the admin password (`ADMIN_PASSWORD`,
   currently `Sakazuki01`). Once authenticated, your session is remembered.

Commands marked **🕵️ HIDDEN** are intentionally unlisted (not shown in `/menu` or
`/manual`). They still work for anyone who knows them, except `/snail` and
`/snailvault`, which stay completely silent unless your chat is unlocked.

---

## 1. Getting started & help

| Command | What it does |
|---|---|
| `/start` | Welcome message and quick intro to the bot. |
| `/menu` | Interactive button menu to reach features without typing commands. |
| `/manual` | Public user manual (excludes admin and hidden commands). |
| `/pro` | Show the full public command guide. `/pro on`\|`off` subscribe/unsubscribe to PRO alerts. `/pro status` shows live tracking. |
| `/status` | Bot status: last scan age, what you're currently tracking. |
| `/cancel` | Cancel the current interactive flow / conversation. |

---

## 2. Market scanning

| Command | What it does |
|---|---|
| `/scan` | Full market scan (top 50 by volume, 4H default, cached). `/scan ETH` scans a specific pair; `/scan ETH 1h` sets the timeframe. |
| `/scannew [window]` | Scan newly-listed pairs. Examples: `/scan new 24h`, `/scan new 30m`, `/scan new 7d`, `/scan new 2w`. |
| `/scanmid [from] [to]` | Scan the mid-tier universe (default ranks 51–200 by 24h volume). Same scoring engine as `/scan`. |
| `/scalp [PAIR]` | Scalp scan of the top 30 pairs on 15m + 1h. `/scalp ETH` for one pair. |
| `/swing [PAIR]` | Swing scan of the top 30 pairs on 4h + 1d. `/swing BTC` for one pair. |
| `/cscan PAIR [tf]` | Scan a specific pair on a specific timeframe (e.g. `/cscan BTC t15m`). Defaults to 4h. |
| `/best` | Show the single best signal from the last scan. |
| `/top`, `/top5`, `/top10` | Show the top N signals from the last scan (add a number, e.g. `/top 3`). |
| `/pick` | List the last scan's signals and pick one to drill into. |
| `/filter` | Filter the last scan's results by your criteria. |
| `/compare [PAIR]` | Compare all last-scan signals vs current prices. `/compare ZEC` for a full live compare of one pair. |
| `/analyse [PAIR] [tf]` | Raw market analysis with **no** regime gate or confidence floor — always returns an indicator snapshot, even in bad conditions. |

---

## 3. Single-trade checks

| Command | What it does |
|---|---|
| `/check PAIR DIR ENTRY SL` | Validate whether an open trade is still valid against current indicators. |
| `/confirm PAIR` | Re-run full analysis and compare to the last stored signal — "is this still good?". |

---

## 4. Trade tracking & PnL

| Command | What it does |
|---|---|
| `/trades` | Live status panel for all active tracked trades. |
| `/stoptrade` | Show all active tracked trades with inline **Stop** buttons. |
| `/pnl [rank\|symbol]` | PnL card. Pick a signal interactively, or look one up directly (e.g. `/pnl 3` or `/pnl ETH`). |
| `/safemode` | Toggle automatic dying-trend alerts. When ON, the bot monitors every signal you receive from any source and warns when momentum fades. |

---

## 5. Alerts & watchlist

| Command | What it does |
|---|---|
| `/alert` | Open the coin-alerts menu. |
| `/unalert PAIR` | Remove a coin alert (e.g. `/unalert BTCUSDT`). |
| `/palert PAIR PRICE [below]` | Price alert (e.g. `/palert BTCUSDT 95000`). Direction is auto-detected vs the live price; add `below` to override. |
| `/unpalert [PAIR]` | Remove a price alert, or run alone to list and pick. |
| `/watch` | Show your watchlist. |
| `/unwatch PAIR` | Remove a pair from your watchlist. |

---

## 6. Autoscan & subscriptions

| Command | What it does |
|---|---|
| `/autoscan [tf\|always\|off]` | Subscribe to auto-pushed signals. `/autoscan 4h` for a timeframe, `/autoscan always` for all, `/autoscan off` to stop. No argument shows status + buttons. |
| `/prime` | 🕵️ **HIDDEN** — secret high-conviction subscription. Subscribe, set your timezone via buttons, and view the live best-pick dashboard. |

---

## 7. Stats, backtesting & leaderboard

| Command | What it does |
|---|---|
| `/stats [window]` | Real performance stats over a time window (the unit decides the window). |
| `/backtest` | Run a backtest (see in-command usage). |
| `/btfull SYMBOL [tf] [bars]` | Full historical walk-forward backtest (e.g. `/btfull POWER 4h 500`). |
| `/paper [history]` | Paper-trading: open positions + 7-day summary. `/paper history` shows the last 20 closed positions. |
| `/leaderboard [window]`, `/lb` | Performance leaderboard. `/lb` is the short alias. |
| `/calibrate [symbol] [exchange]` | Empirically validate the bot's ATR multiplier parameters against real historical data. Heavy/slow. |

---

## 8. Charts & market sentiment

| Command | What it does |
|---|---|
| `/chart PAIR` | Generate a TA chart image (candles + EMA20/50 + RSI + MACD + Bollinger Bands). |
| `/fgi` | Current Fear & Greed Index as a branded image card. |
| `/tg` | Top **gainers** from accumulated price history (run `/scan` a few times first). |
| `/tl` | Top **losers** from accumulated price history (run `/scan` a few times first). |

---

## 9. Operator / broadcast

| Command | What it does |
|---|---|
| `/broadcast on\|off` | Toggle this chat as a broadcast channel that receives operator broadcasts. No argument shows current status. |

---

## 10. Admin & maintenance 🔒

These require admin authentication (see top of file).

| Command | What it does |
|---|---|
| `/admin` | 🔒 Admin dashboard. With `ADMIN_IDS` set, shows interaction stats; password-gated mode shows the full user-activity dashboard. |
| `/dbcheck` | 🔒 Inspect DB tables and user-activity rows. Helps diagnose why the dashboard shows 0 users. |
| `/optimize` | 🔒 Run a Hyperopt parameter search and persist `best_params.json`. |
| `/xgtrain` | 🔒 Retrain the XGBoost win-probability model from historical outcomes. |
| `/rftrain` | 🔒 Retrain the Random Forest win-probability model from historical outcomes. |

---

## 11. Diagnostics & secret commands 🕵️

| Command | What it does |
|---|---|
| `/autoscandiag` | 🕵️ **HIDDEN** — autoscan diagnostic. Reports subscriber count, whether you're subscribed, uptime + startup-grace status, last scan size, how many signals clear the confidence floor, the top signals, and a plain-English verdict on why autoscan is or isn't pushing. |
| `/snail` | 🕵️ **HIDDEN** — completely silent (no reply) unless your chat is unlocked. |
| `/snailvault` | 🕵️ **HIDDEN** — completely silent unless your chat is unlocked. |

---

## Quick admin checklist

- **Authenticate:** `/admin` → enter `Sakazuki01` (or be in `ADMIN_IDS`).
- **Health check:** `/status`, then `/autoscandiag` if autoscan seems quiet.
- **Data check:** `/dbcheck` to confirm users/outcomes are being recorded.
- **Retrain models:** `/xgtrain`, `/rftrain`, `/optimize` (after enough outcomes accumulate).
- **Broadcast:** `/broadcast on` in any channel you want operator messages delivered to.

_Generated from the live handler registrations in `sakz_bot.py`._
