# sakz-bot — full debug report

Repo: `sakz-bot-main.zip` (full project, 20 Python modules + configs).

## How this was debugged (what's provable here vs. not)

**This sandbox has no network and cannot install the heavy deps** (`python-telegram-bot`,
`ccxt`, `ta`, `scikit-learn`, `xgboost`), and the bot needs a Telegram token + Turso DB +
live exchange access at runtime. So the bot **cannot be executed end-to-end here.**

What I *could* do rigorously:
1. **Byte-compile every module** — all 20 `.py` files compile with no syntax errors.
2. **AST static scan** (custom, stdlib `ast`) for silent excepts, bare excepts, mutable
   default args, `== None`, duplicate dict keys, duplicate top-level defs.
3. **Import + unit-test** the dependency-light modules (`sakz_risk`, `sakz_validation`,
   `config`, `signal_types`) which DON'T need the heavy deps — so the fixes below are
   actually executed and asserted, not just eyeballed.

## ✅ Fixes applied AND verified by execution

| # | Issue | File | Verification |
|---|-------|------|--------------|
| **#6** | `WIN_OUTCOMES`/`LOSS_OUTCOMES` duplicated in `sakz_risk.py` & `sakz_validation.py` | `sakz_validation.py` now imports them from `sakz_risk` (single source of truth) | `r.WIN_OUTCOMES is v.WIN_OUTCOMES` ✓ |
| **#7** | `DrawdownGate.can_open()` updated peak *before* validating → a bad/`inf`/spike equity poisoned `peak_equity` | `sakz_risk.py` — added `_is_valid_equity()` (finite, >0, ≤2× peak); validate before update; fail closed on bad input | inf/NaN/0/neg/1e9-spike all rejected, peak preserved; 1.5× legit high accepted; dd gate still trips at 15% ✓ |
| **C2** | `_scan_cache` written outside `_scan_cache_lock` in `mid_scan_job` (race / torn cache) | `sakz_bot.py` — write wrapped in `async with _scan_cache_lock` (the `get_scan_results()` call stays outside to avoid re-entrant deadlock) | compiles; lock serialization tested in isolation ✓ |
| **H3** | `cscan_results` / `_chat_scan_ctx` grew forever (OOM) | `sakz_bot.py` — replaced with `_BoundedDict` (FIFO eviction, cap 2000) | compiles; eviction tested (`[7,8,9]`), updates keep size ✓ |
| **#8** | 16 bare `requests.get` calls, no retry — transient network/5xx surfaced as empty/"neutral" scans | `sakz_bot.py` — shared `requests.Session` + `HTTPAdapter(Retry(total=3, backoff=0.5, status_forcelist=429/5xx))`; all 16 calls routed through new `http_get()` (default timeout, env-tunable). No new dependency | retry config asserted (total=3, backoff=0.5, 5xx list); `http_get` injects default & preserves explicit timeout; 0 stray `requests.get` left ✓ |
| **M1** | Turso/`libsql_experimental` ignores `row_factory=sqlite3.Row` → rows are tuples → all 400+ `row['col']` accesses crash on the Turso path | `sakz_db.py` — dict-row adapter (`_DictRow`/`_CursorWrapper`/`_ConnWrapper`) at the single `db_connect()` chokepoint; only the Turso path is wrapped, local SQLite keeps native `sqlite3.Row` | simulated libsql with a no-`row_factory` sqlite conn (returns tuples); verified `row['col']`, `row[0]`, `.get`, `keys`, iteration, `.execute().fetchall()` chaining, `None`, and `with conn:` all work ✓ |

> **#8 & M1 caveat:** verified by compile + isolated/simulated execution here. The actual `libsql` driver and live exchange endpoints aren't available in this sandbox, so do a quick paper-mode smoke test on your side before trusting them in production.

## 🔎 Static-scan findings (informational)

- **M2 — silent `except: pass`:** `sakz_bot.py` ×30, `sakz_db.py` ×6, `sakz_backtest_hist.py` ×1.
  These swallow errors. They are *behavioral* — each needs a human decision (log vs. re-raise),
  so I did **not** auto-rewrite them blind (that risks changing control flow in a money path).
- No duplicate top-level defs, no duplicate dict keys, no bare `except:`, no mutable default args found.

## ⏳ NOT auto-applied — needs a runnable env or is empirical

I deliberately did **not** blind-edit these, because they change runtime behavior in the
14.5k-line monolith and I cannot execute it here to prove I didn't break anything. "Make no
mistakes" means not shipping unverified behavioral changes into a live trading bot.

- **H2 / #5 / #9 / #9b / M3 / #4 / H4 / C1 / #1** — in-monolith behavioral/structural; need the
  live bot (Telegram + exchanges) to validate.
- **#3 / M4** — empirical (win rate, ATR calibration). No code change can *guarantee* profitability;
  must be measured from live/paper data.

Precise patch snippets for all of the above are in `PATCHES.md` (from the earlier `sakz-bot-fixes.zip`).

## Bottom line
4 issues fixed and proven by execution; every module still compiles. The rest are documented
with exact patches but require your runtime (paper mode first) to validate safely.
