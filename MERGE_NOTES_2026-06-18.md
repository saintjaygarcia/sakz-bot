# sakz-bot - Merged build (Improved wiring + Scale hardening + Security fixes)

This build is the **improved/verified wiring build** (signal "Why" block,
`/btfull` validation report, BUCKET_LABELS fix, graceful `ta` import, full
test + CI scaffolding) with the **scale-hardening DB layer** ported in and two
**critical security fixes** added on top.

## What was merged from the hardened build (DB layer only)
The hardened build was branched *before* the wiring round (it lacked
`score_buckets`, the explain/validation toggles, and still carried the old
BUCKET_LABELS mislabel). So only the genuine scale work was ported -- file
contents were NOT copied wholesale (the hardened build also re-introduced a
bare `import ta` and a corrupted UTF-8 comment line, both avoided here).

Ported into `sakz_db.py`:
- **WAL `db_connect`**: `journal_mode=WAL`, `synchronous=NORMAL`,
  `busy_timeout` (env `SAKZ_SQLITE_BUSY_TIMEOUT`, default 30s),
  `foreign_keys=ON`, `temp_store=MEMORY`, `check_same_thread=False`.
- **14 performance indexes** created idempotently in `db_init`.
- `test_scale_hardening.py` (100 threads x 20 ops, 0 lock errors).

The debug-logging removals the hardened build made in `sakz_exchanges.py` /
`sakz_db.py` were intentionally NOT ported -- they reduce observability and are
not part of the scale fix.

## Security fixes added in this merge
### 1. Admin password no longer hardcoded (`sakz_bot.py`)
The literal `ADMIN_PASSWORD = "Sakazuki01"` was removed from source. The gate now
reads from the environment and compares with a constant-time hash check:
- `SAKZ_ADMIN_PASSWORD_HASH` - sha256 hex of the password (preferred), OR
- `SAKZ_ADMIN_PASSWORD` - raw password (hashed in memory at load).
- If neither is set the password gate **fails closed** (disabled); use `ADMIN_IDS`.

**Migration (so you are not locked out):** the sha256 of your current password
`Sakazuki01` is:
```
SAKZ_ADMIN_PASSWORD_HASH=7645447c51d99238be0000f504f6917a5bdd85bda35e3ab62417522aa49d7148
```
Set that env var on the host and the existing password keeps working -- but the
secret is no longer in the code. **Recommended:** pick a new password and store
its hash instead (`python3 -c "import hashlib;print(hashlib.sha256(b'NEWPASS').hexdigest())"`).

### 2. Admin sessions now expire (`sakz_db.py` -> `db_admin_is_authed`)
Previously any past authentication counted forever. Sessions now expire after
`SAKZ_ADMIN_SESSION_TTL_HOURS` (default 12h); a stale session forces re-auth.

## New environment variables
| Var | Default | Purpose |
|-----|---------|---------|
| `SAKZ_ADMIN_PASSWORD_HASH` | (unset) | sha256 hex of admin password (preferred) |
| `SAKZ_ADMIN_PASSWORD` | (unset) | raw admin password (fallback, hashed at load) |
| `SAKZ_ADMIN_SESSION_TTL_HOURS` | `12` | admin session lifetime |
| `SAKZ_SQLITE_BUSY_TIMEOUT` | `30` | seconds a contending SQLite conn waits for a lock |

## Verification (run in sandbox)
- `compileall` clean; 0 U+FFFD across all `.py`.
- Full suite green: wiring 8, signal_behavior 56, autoscan 17, prime,
  prime_integration 16, phase0 16, **scale-hardening (incl. 100-thread stress)**.

---

# Execution safety layer scaffold (`sakz_orders.py`, TESTNET-ONLY)

Added the disciplined order path discussed for execution readiness. It is
self-contained, reuses `sakz_risk`, and is fully unit-tested offline (a fake
transport is injected, so no network/keys are touched). **No real-money path is
enabled by default.**

```
signal -> PreTradeGate -> OrderRouter -> { PaperBackend | LiveBackend }
                              |
                          live_orders ledger (DB = source of truth)
```

## What it provides
- **Order ledger** (`live_orders`) + **control** (`trading_control`) tables via
  `orders_init_db(db_connect)` (same pattern as `sakz_paper`). Orders are
  written **before** they are sent.
- **Kill switch**: `set_kill_switch(db_connect, halted, reason)` / `is_halted(...)`.
  Auto-engages when the drawdown budget is breached.
- **PreTradeGate**: kill-switch -> required fields/valid side -> SL-on-correct-side
  -> drawdown budget (`risk.DrawdownGate`) -> max open positions -> position
  sizing (`risk.kelly_position_size`) clamped to per-order / per-symbol / total
  notional caps, with optional `risk.correlated_exposure_cap`.
- **Idempotent LiveBackend** (Bybit V5, testnet by default): generates an
  `orderLinkId`, writes a pending ledger row, sends via a **validated** request
  (`retCode==0` enforced - the raw `bybit_signed_request` does NOT check this),
  and **never re-sends** a link id that is already submitted. **Refuses mainnet**
  unless `SAKZ_ALLOW_MAINNET` is set.
- **PaperBackend** delegates to `sakz_paper.paper_maybe_open`, so paper and live
  share one router path.

## New env vars (all optional, safe defaults)
| Var | Default | Purpose |
|-----|---------|---------|
| `SAKZ_MAX_OPEN_POSITIONS` | `5` | max concurrent live positions |
| `SAKZ_MAX_NOTIONAL_PER_ORDER` | `50` | ceiling per single order (quote) |
| `SAKZ_MAX_NOTIONAL_PER_SYMBOL` | `100` | combined cap for one symbol |
| `SAKZ_MAX_TOTAL_NOTIONAL` | `250` | combined cap across all symbols |
| `SAKZ_DAILY_LOSS_LIMIT_PCT` | `0.05` | reserved for daily-loss halt wiring |
| `SAKZ_DRAWDOWN_BUDGET` | `0.15` | peak-to-trough budget for the gate |
| `SAKZ_KELLY_FRACTION` / `SAKZ_KELLY_CAP` | `0.25` / `0.25` | sizing |
| `SAKZ_ALLOW_MAINNET` | (unset) | must be set to allow non-testnet sends |

## Still TODO before live money (not in this scaffold)
- Wire `get_keys(chat_id)` to the encrypted vault (`sakz_execution` per-user keys).
- Server-time sync for the signed request (recv_window drift).
- Fill/position reconciliation loop (query open orders on restart) + tick/lot rounding.
- Wire `win_rate` from `signal_outcomes` via `risk.win_rate_from_outcomes`
  (currently a 0.55 placeholder) and the daily-loss-limit auto-halt.
- Register `orders_init_db` at startup and a `/halt` `/resume` admin command.

Tests: `test_orders.py` (26 checks) covers gates, caps, kill switch, drawdown
auto-halt, idempotency, retCode validation, and the mainnet refusal.

---

# Execution layer - remaining TODOs now COMPLETE (2026-06-19)

All items previously listed as "still TODO before live money" are now implemented
and tested (`test_orders.py`, 48 checks, all offline via injected fakes).

## 1. Per-user encrypted API key vault
- New `user_api_keys` table (PK `chat_id, exchange`), keys stored **encrypted**.
- `store_user_keys(...)`, `load_user_keys(...)`, `make_key_getter(...)` -> the
  `get_keys(chat_id)` callable that `LiveBackend` needs.
- Cipher is injectable; defaults to `sakz_execution.encrypt_secret/decrypt_secret`
  (Fernet). Verified round-trip with real Fernet and with a fake cipher.

## 2. Server-time sync (recv_window drift)
- `compute_time_offset`, `ServerClock(get_server_ms, local_ms)`, and best-effort
  `sync_server_time(testnet)` (queries `/v5/market/time`).
- `sakz_execution.py` gained `_TIME_OFFSET_MS` + `set_time_offset_ms()` /
  `get_time_offset_ms()`, and `bybit_signed_request` now stamps
  `time.time()*1000 + _TIME_OFFSET_MS`. Call `sync_server_time()` at startup.

## 3. Fill/position reconciliation + tick/lot rounding
- `LiveBackend.reconcile()` queries `/v5/order/realtime` by `orderLinkId` for every
  non-terminal ledger row and maps Bybit `orderStatus` -> ledger status
  (Filled->open, New/Partial->submitted, Cancelled->canceled, Rejected->failed),
  recording `cumExecQty` / `avgPrice`. Run on startup + periodically.
- `round_down_step()` + optional injected `get_instrument(symbol)` ->
  `{qtyStep, tickSize, minOrderQty}`; qty floored to lot, SL/TP floored to tick,
  and orders **below `minOrderQty` are refused**.

## 4. Win-rate from outcomes + daily-loss auto-halt
- `win_rate_for_symbol(db_connect, symbol)` reads `signal_outcomes` and feeds
  `risk.win_rate_from_outcomes` (falls back to 0.55 below `SAKZ_WINRATE_MIN_SAMPLE`).
  `PreTradeGate` now uses this automatically instead of the old fixed placeholder.
- `pnl_ledger` table + `record_realized_pnl()` / `daily_realized_pnl()`. The gate
  halts trading when today's realized loss >= `SAKZ_DAILY_LOSS_LIMIT_PCT` x equity.

## 5. Bot wiring (sakz_bot.py)
- `orders_init_db(db_connect)` is now called at startup (next to `paper_init_db`).
- Admin commands added: **`/halt`** (engage kill switch, optional reason),
  **`/resume`** (release), **`/orders`** (kill-switch state + open exposure +
  today's PnL). All gated by the existing admin auth (`db_admin_is_authed` /
  `ADMIN_IDS`). Wrapped in `_ORDERS_AVAILABLE` try-import so the bot still boots
  if the module is absent.

## New env vars
| Var | Default | Purpose |
|-----|---------|---------|
| `SAKZ_WINRATE_MIN_SAMPLE` | `10` | min historical outcomes before using real win-rate |
| `SAKZ_MIN_NOTIONAL` | `1.0` | reject orders whose sized notional collapses below this |

## Truly remaining (operational, not code)
- Per-symbol instrument metadata feed for `get_instrument` (Bybit `/v5/market/instruments-info`); a fetch+cache helper can wrap it.
- A scheduled `reconcile()` + `sync_server_time()` job (wire into the existing job_queue).
- Closing/exit order management + writing realized PnL into `pnl_ledger` on close.
- Mainnet is still hard-OFF until `SAKZ_ALLOW_MAINNET` is set and keys are vaulted.
