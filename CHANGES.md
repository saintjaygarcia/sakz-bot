# Sakz Bot — Refactor, Round 2

This round builds on Round 1 (which extracted the data layer into `sakz_db.py`
and the raw-requests exchange layer into `sakz_exchanges.py`, with verified
re-exports so existing call sites and `sakz_backtest_hist.py` keep working).

**Guiding rule this round:** because the strategy core cannot be executed in
this environment (no `ta`, `ccxt`, `telegram`, or market data), I only made
changes I could *prove* safe by running them. Anything I could not verify is
documented as a blueprint rather than applied blindly.

---

## On the "0 dangling references" critique (addressed)

You were right: an AST scope check confirms names resolve, but it does **not**
prove the runtime import contract that `sakz_backtest_hist.py` depends on:

```python
from sakz_bot import add_indicators, score_pair
from sakz_bot import (bybit_fetch_ohlcv, binance_fetch_ohlcv, mexc_fetch_ohlcv)
```

This round adds **real tests** instead of relying on a scope claim:

- `tests/test_import_contract.py` — parses `sakz_bot.py` and asserts each name
  the backtest imports is exposed at module scope (defined or re-imported).
  Since `sakz_bot.py` itself can't be imported here (heavy deps), this is the
  honest structural guard for that contract.
- `tests/test_sakz_exchanges.py` — actually **runs** the extracted fetchers with
  mocked HTTP and asserts parsing/gating behaviour, including a regression guard
  that the MEXC kline URL is well-formed (see below).
- `tests/test_sakz_db.py` — runs `db_connect()` / `db_init()` against a temp
  SQLite file and asserts the schema (24 tables) initialises.

`python -m unittest discover -s tests -t .` → **22 passing.**

---

## The suspected MEXC URL bug — investigated and REJECTED

While auditing the exchange layer the MEXC futures kline URL *appeared* to be
built with doubled braces (`f"{...}"`-style), which would have produced a
malformed URL and silently broken MEXC OHLCV. I verified it at the byte/hex
level: the real source is

```python
f"https://contract.mexc.com/api/v1/contract/kline/{futures_sym}"
```

i.e. a single, correct pair of f-string braces. The doubled braces were a
**display artifact** of how text is rendered to the assistant, not the file
content. **No change was made.** `tests/test_sakz_exchanges.py` now asserts the
requested URL equals `.../kline/BTC_USDT` with no stray braces, so any real
regression here would now fail a test.

---

## Implemented this round

### #5 — Centralised config (`config.py`)
Single source of truth, loaded once via `python-dotenv`:
- Database/Turso settings (`TURSO_URL`, `TURSO_TOKEN`, `_USE_TURSO`, `DB_PATH`,
  `ACTIVE_WINDOW_MIN`) — moved **verbatim** out of `sakz_db.py` (which now
  imports and re-exports them, so `from sakz_db import DB_PATH` still works).
- HTTP `HEADERS` (now imported by `sakz_exchanges.py` instead of redefined),
  `HTTP_TIMEOUT`, and per-venue mid/top minimum-volume constants.

### #2 — Exchange adapter layer (`exchange_adapters.py`)
A uniform interface — `get_top_symbols`, `get_mid_symbols`, `fetch_ohlcv`,
`fetch_funding`, `get_current_price`, `check_available`, `available` — across
`BybitAdapter`, `MexcAdapter`, `BinanceAdapter`, plus an `ADAPTERS` registry and
`get_adapter(name)`. Call sites can stop branching on exchange name.

Design choice (accuracy-first): adapters **delegate** to the existing, proven
`sakz_exchanges` functions rather than re-implementing venue logic. The
availability flags stay owned by `sakz_exchanges` and are read live via
attribute access, preserving the exact runtime-flag contract. MEXC has no
availability probe or funding endpoint in the legacy code, so its adapter
reports `available=True` and `fetch_funding=0` (documented parity).

### #6 — Typed `Signal` (`signal_types.py`)
A `Signal` dataclass with lossless `from_dict` / `to_dict` (unknown keys land in
`extra`, so round-trips are faithful). Deliberately **non-invasive**: it is a
scaffold for incremental adoption, not wired into the hot path (wiring it into
`score_pair` can't be runtime-verified here — see #1).

### #4 — Logging (partial, safe subset)
Converted the three post-logger module-load `print(...)` notices in
`sakz_bot.py` to `logger.info(...)`. The four pre-logger bootstrap prints (above
the `logger = …` definition) and the deliberate `main()` startup banner were
intentionally left as prints — converting them is either unsafe (NameError
before the logger exists) or a user-visible behaviour change.

### #7 — Test suite (`tests/`)
`test_sakz_db.py`, `test_sakz_exchanges.py`, `test_exchange_adapters.py`,
`test_signal_types.py`, `test_import_contract.py`. 22 tests, all passing.

---

## #1 — `score_pair` decomposition: blueprint (NOT yet applied)

`score_pair(df4h, df1d, funding, symbol, user_requested=False)` is ~917 lines
(lines 2170–3086 in the refactored `sakz_bot.py`). I am **not** chopping it
blindly: it cannot be executed here, so I cannot prove an equivalent rewrite —
exactly the accuracy risk you raised. Doing it safely requires a test harness
with recorded candle fixtures. The structure, mapped for that future work:

**Phase A — straight-line indicator accumulation (~2170–2510, no early returns).**
Mutates paired long/short buckets: `osc_g_l/s`, `mtf_g_l/s`, `cross_g_l/s`,
`pos_g_l/s`, `vg_l/s`, `ig_l/s` and counters `ls/ss`, `lr/sr`. Sub-blocks: RSI,
MACD, EMA stack, Bollinger, Stochastic, divergence, volume, support/resistance,
Fibonacci, funding, CLV.
→ Extract as `_accumulate_indicator_scores(...) -> ScoreBuckets` (pure;
depends only on the frames + funding).

**Phase B — preliminary bias (~2510–2522).** Compares `ls`/`ss`; first early
`return None` lives here (`if ls == ss or (ls < 3 and ss < 3)`).
→ `_determine_bias(buckets) -> Optional[str]`.

**Phase C — conviction + directional scoring (~2467–2700).** Reads `bias`,
applies caps (`OSC_CAP=3`, `MTF_CAP=2`, `CROSS_CAP=3`, `POS_CAP=3`,
`STRUCTURE_CAP=4`; `ig` uncapped) and the `conviction_scores` delegate.
→ `_score_direction(buckets, bias, conviction) -> DirectionalScore`.

**Phase D — gates / early exits.** `REASON_GAP_BLOCK`, `FLIP_BLOCK`,
`COUNTER_TREND`, each returning a `ScanFailure`.
→ `_apply_gates(...) -> Optional[ScanFailure]` (ordered, short-circuiting).

**Phase E — sizing + result assembly.** `confidence = min(10, round(ratio_conf
+ quality_bonus))`, entry/SL/TP levels, `abs_score_floor` map → result dict.
→ `_build_signal(...)`, ideally returning a `signal_types.Signal`.

**Safe sequence when fixtures exist:** capture inputs/outputs for a set of real
symbols → add a characterization test asserting byte-identical results → extract
Phase A first (largest, side-effect-free) → re-run → proceed phase by phase,
re-running the characterization test after each. Already-delegated helpers
(`conviction_scores`, `session_context`, `candle_quality_score`,
`get_btc_regime`) stay as-is.

---

## Suggested follow-ups

- Add recorded-candle fixtures so #1 (and wiring `Signal` into `score_pair`) can
  be done test-guarded.
- Migrate scanner call sites onto `exchange_adapters.get_adapter(...)` to retire
  the per-exchange branching, then optionally fold shared HTTP handling into a
  single `_get_json` helper.
- Finish the logging migration in `sakz_conviction.py` / `sakz_memory.py` once
  there's a smoke test for them.


---

# TODO list implementation (sections 1–8)

Implemented from `sakz_bot_opus_todo.html`. Wiring details for each item live in
`INTEGRATION_GUIDE.md`. All new logic is self-contained and unit-tested
(`tests/test_strategy_modules.py`, `tests/test_ml_outcomes.py`); the live
monolith cannot be executed in this environment, so monolith wiring is provided
as reviewed-but-unrun call-site instructions.

## 1. Critical ML training bug — FIXED + verified
- `xgboost_train.py` / `rf_train.py` queried `outcome IN ('win','loss')`, but the
  resolver writes `t1_hit/t2_hit/t3_hit/sl_hit`. The query matched **zero** rows,
  so both models trained on no data.
- Fixed: parameterized `WHERE outcome IN (?,?,?,?)` with
  `WIN_OUTCOMES=('t1_hit','t2_hit','t3_hit')`, `LOSS_OUTCOMES=('sl_hit',)`;
  `label = 1 if outcome in WIN_OUTCOMES else 0`. `rf_train` shares the loader.
- `tools/verify_outcomes.py` + `tests/test_ml_outcomes.py` confirm the loader now
  picks up resolved rows and the old query returned 0.

## 2. Refactor / hygiene — DONE
- DB → `sakz_db.py`, exchanges → `sakz_exchanges.py` / `exchange_adapters.py`
  (re-exported; call sites unchanged). Current `sakz_bot.py`: 0 bare `except:`,
  single `start_command`, no `flip_patch` monkey-patch. `score_pair` split is a
  documented, fixture-gated follow-up.

## 3. T2/T3 management — `sakz_trade_mgmt.py`
- `move_sl_to_breakeven`, `atr_trailing_stop` (both tighten-only, direction-aware),
  `cap_targets_for_regime` (caps T2≤2·ATR / T3≤3.5·ATR in LOW/RANGING only),
  `manage_after_t1` (combined).

## 4. Signal explanation — `sakz_explain.py`
- `bucket_breakdown`, `regime_tag` (BTC regime + counter-trend note),
  `dominant_reasons` / `reason_summary` (weighted), `explanation_block`.

## 5. Order flow — `sakz_orderflow.py`
- `delta_divergence` (price/CVD), `is_absorption`/`absorption_score` at S/R,
  `liquidation_cluster_risk` (pure; needs a live liquidation feed to populate).

## 6. Portfolio sizing — `sakz_risk.py`
- `fractional_kelly` (¼-Kelly, 25% cap), `kelly_position_size`,
  `correlated_exposure_cap`, `DrawdownGate` (15% max-drawdown pause).

## 7. Statistical validation — `sakz_validation.py`
- `walk_forward_splits` (rolling/anchored), `regime_split_report`,
  `monte_carlo_edge_test` (sign-permutation p-value), `max_drawdown`.

## 8. Cross-asset correlation — `sakz_correlation.py`
- `rolling_correlation_matrix` (20-bar returns), `is_btc_dependent`,
  `eth_btc_ratio_regime` (ALT_RISK_OFF/ON/NEUTRAL), `SECTOR_MAP` + `sector_of`,
  `follow_on_confidence` (correlated-signal damping).

**Tests:** `python -m unittest discover -s tests -t .` → 57 passing.


---

# Code-review response (load_dotenv / HTTP_TIMEOUT / get_adapter / wiring)

## 1. `HTTP_TIMEOUT` now actually used — FIXED
`sakz_exchanges.py` added `_TIMEOUT = HTTP_TIMEOUT` (imported from config) and the
**11** `requests.get(..., timeout=15)` calls now use `timeout=_TIMEOUT`. The few
lightweight ticker/price endpoints that intentionally used shorter `timeout=8` /
`timeout=10` were left as-is on purpose (documented in a comment). Changing
`HTTP_TIMEOUT` in config now propagates.

## 2. Redundant `load_dotenv()` removed — FIXED
`sakz_db.py` no longer calls `load_dotenv()`; it relies on `config.py` (the single
source of truth, which loads dotenv once). Replaced with a comment explaining why.

## 3. `get_adapter()` error message — FIXED
Unknown names now raise
`ValueError("Unknown exchange 'KRAKEN'; valid: ['BYBIT', 'MEXC', 'BINANCE']")`
instead of a bare `KeyError`. Covered by `tests/test_adapter_equivalence.py`.

## 4. Stateless-singleton note — DOCUMENTED
Added a comment at the `ADAPTERS` definition: the module-level singletons are
safe *because* adapters are stateless delegators; if one ever gains mutable
per-instance state, construct fresh per call (or lock) instead.

## 5. score_pair characterization harness — BUILT (the real remaining work)
score_pair can't be split safely without a regression net. Added:
- `tools/record_fixtures.py` — run in the **live** env (needs `ta` + network) to
  capture real `(df4h, df1d, funding) -> score_pair output` golden fixtures into
  `tests/fixtures/score_pair/`. Mirrors the exact production pipeline
  (`fetch_ohlcv -> add_indicators -> score_pair`).
- `tests/test_score_pair_characterization.py` — replays fixtures and asserts
  byte-identical output; **skips** when no fixtures exist or `sakz_bot` can't be
  imported (e.g. CI without `ta`).
- `tools/synth_ohlcv.py` + `tests/test_synth_ohlcv.py` — deterministic synthetic
  OHLCV generator with verified OHLC invariants, as an offline fixture source.

**You must run `python tools/record_fixtures.py BTCUSDT ETHUSDT ... ` once in the
live environment** to populate the goldens — this environment has no `ta`/network
so the goldens cannot be generated here. After that, the score_pair refactor can
proceed test-guarded.

## 6. Adapter migration — STARTED + proven safe
- `tests/test_adapter_equivalence.py` proves every adapter method delegates to the
  exact legacy `sakz_exchanges` function with identical args (incl. default
  interval, config volume default, MEXC funding=0 parity, live availability flags).
- First real call site migrated: the `_NL_FETCH` dispatch table in `sakz_bot.py`
  (used by `analyze_symbol_new_listing`) now calls `get_adapter(name).fetch_ohlcv`.
  This is behaviour-identical and was chosen because it's self-contained.
- `sakz_bot.py` now imports `get_adapter` — the adapter layer is no longer unused.

### Remaining call-site migration checklist (tracked)
These live branches still dispatch by exchange name. Migrate one at a time, each
guarded by recorded fixtures / a paper-trading smoke test, since the monolith
can't be executed in this environment:
- [ ] `scan_*` OHLCV fetch branches ~ lines 423–431 and 527–532
- [ ] price/funding helpers ~ lines 1465–1489
- [ ] `score_pair` feeder branches ~ lines 3208–3219, 3963–3983, 4044–4065
- [ ] outcome-resolver fetches ~ lines 4288–4290
- [ ] misc dispatch ~ 5653, 8630, 10052, 11435, 11879
- [x] `_NL_FETCH` new-listing dispatch (~3268) — DONE

## Scaffolding tracker (infrastructure not yet wired into live paths)
- [ ] `signal_types.Signal` — defined + tested, not yet returned by `score_pair`.
  Wire in as part of the score_pair split (Phase E in the score_pair plan above),
  gated by the characterization fixtures.
- [ ] Full `get_adapter` migration — see checklist above.

**Tests:** `python -m unittest discover -s tests -t .` → 68 tests, 67 pass + 1 skip
(the characterization test, until fixtures are recorded).


---

# Signal-coverage fixes (repetition / new listings / FETUSDT)

## Problem 2 (HIGH) — New listings silently dropped — FIXED
`_run_new_listings_scan()` correctly detected fresh pairs, then handed them to
`analyze_bybit()` / `analyze_mexc()`, which hard-require 30x 4H + 15x 1D candles.
A pair listed 24h ago has ~6 4H candles → every new listing returned `None` and
was dropped. Now it calls `analyze_symbol_new_listing()` (purpose-built for
3+ candles, walks 5m→15m→1h→4h, self-confirms on the primary frame). Both
exchange lists are merged into one de-duplicated set so a pair listed on both is
scored once, and `ScanFailure`/`None` returns are filtered out.

## Problem 1 (MED) — Same pairs repeating — FIXED (mid-tier rotation now scheduled)
`run_mid_scan()` (ranks 51–200) existed but was **never scheduled**, so the
auto-push pipeline only ever saw the top-50 (BTC/ETH/SOL/...). Added
`mid_scan_job` and scheduled it every 4h, offset 2h from the full scan. It
merges its mid-tier signals into the live scan cache by `exchange+symbol+bias`,
and the existing `continuous_scan_job` delivers them through the **unchanged**
confidence gate + dedup + per-subscriber TF filter. No change to scoring or push
logic — the mid-tier universe simply now participates.

## Problem 1 ("Also") — Restart re-send flood — FIXED (startup grace)
`_autoscan_sent` is in-memory and is wiped on restart, so the first
`continuous_scan_job` tick after a restart treated every live signal as
"never sent" and could flood subscribers. Added a 15-minute startup grace
(`_AUTOSCAN_STARTUP_GRACE_SECS`): during the window the dedup table is still
*seeded* (mark-sent) but the actual push is suppressed, so only genuinely new
signals fire once the bot has settled. Chosen over DB persistence as the
lower-risk option (no schema/DB-path surface).

## FETUSDT "not found on MEXC" — IMPROVED (single-pair cross-exchange fallback)
The single-pair `/scan` (`_cscan_pair_tf`) historically queried **MEXC only**
(Bybit/Binance had been removed with a "geo-blocked on this hosting region"
comment), so pairs that trade on Bybit/Binance perps but not MEXC (e.g.
FETUSDT) returned a misleading "not found on MEXC". Added a fallback: when MEXC
yields nothing, it now calls `analyze_symbol_new_listing()`, which probes
BYBIT → MEXC → BINANCE, each gated by its `*_AVAILABLE` flag.

**Honest caveat:** this only recovers the pair where Bybit/Binance are actually
reachable from your host. Where they are geo-blocked, the result is unchanged
(MEXC has no FETUSDT perp). The error wording was updated from "not found on
MEXC" to "not found on any reachable exchange" to stop misattributing it to MEXC.

## Problem 3 (LOWER) — Active universe rotation — DESIGNED, NOT YET BUILT
Genuine variety (rotating a different ~50-coin slice each cycle, or weighting by
momentum-change instead of raw 24h volume) is a real architecture change to
`run_full_scan` + the cache. Per the project's safety-first philosophy (don't
rewrite logic that can't be executed/tested here), this is deliberately **not**
rushed. Proposed design for when it can be validated against live data:
  - Maintain a rotating offset cursor over the ranked universe (e.g. top-200),
    scanning a sliding 50-coin window per cycle and persisting the cursor.
  - Add a momentum-change score (rank delta / 24h % move) as an alternative
    sort key, blended with volume.
  - Keep the top-N "anchors" (BTC/ETH) always-in so majors aren't missed.
  - Gate behind a config flag so it can be A/B'd against the current behaviour.

### Tracked follow-ups (scan coverage)
- [x] Schedule `run_mid_scan` as `mid_scan_job` (every 4h, +2h offset)
- [x] New-listings scan uses `analyze_symbol_new_listing`
- [x] Restart re-send flood guard (startup grace)
- [x] Single-pair `/scan` cross-exchange fallback
- [ ] Active universe rotation (Problem 3) — design above, needs live validation
- [ ] Optional: per-pair daily send cap (e.g. "max 2 sends/pair/day") on top of
      the 4h cooldown, if repetition persists even with mid-tier rotation
- [ ] Optional: persist `_autoscan_sent` to DB to also survive long outages

**Tests:** full suite still 68 tests, 67 pass + 1 skip. `sakz_bot.py` compiles.
The scan changes touch the un-runnable monolith, so verify `/scan new 24h`,
`/scan fet`, and mid-tier auto-pushes on a paper/testnet run before going live.

---

## User-requested pairs: always return an analysis for any MEXC pair (/scan + /chart)

**Problem:** A manual `/scan <PAIR>` or `/chart <PAIR>` could come back empty / "blocked" for valid MEXC pairs. Bybit and Binance APIs are geo-blocked, so MEXC is the only reachable source, and the scoring gates were dropping legitimate user queries.

**Root cause:** `score_pair()` already converted GAP / FLIP / COUNTER_TREND / REGIME / abs-score gates to *soft warnings* when `user_requested=True`, but two hard blocks remained:
1. The neutral / very-weak bias check (`if ls == ss or (ls < 3 and ss < 3): return None`) dropped the pair entirely, even for manual queries.
2. `/chart` scored the pair WITHOUT `user_requested=True`, and if scoring returned a `ScanFailure` object it was treated as a signal (`if sig:`), crashing on `sig['bias']`.

**Fix:**
- `score_pair()`: when `user_requested=True`, the neutral/weak branch no longer returns `None`. It logs a soft warning and falls through to produce a best-effort read. Tie (`ls == ss`) now resolves deterministically to LONG. Auto-scan (`user_requested=False`) behavior is unchanged — it still drops directionless noise so the alert feed stays clean.
- The remaining gates (gap, flip, counter-trend, regime, abs-score, BTC.D) were already soft for `user_requested`, so weak pairs now surface as low-confidence signals (e.g. 2/10–4/10) with explicit warning notes instead of being suppressed.
- `/chart` `_get_sig()` now scores with `user_requested=True`, so the bias overlay renders even for weak pairs.
- `/chart` normalises a `ScanFailure` result to `None`, so when MEXC genuinely has no data the chart still renders with a "no strong signal" note instead of crashing.
- Only genuine data-availability failures (no contract / insufficient history on MEXC) still produce a "not found" outcome.

**Scope:** Affects manual `/scan <pair>` and `/chart <pair>` only. Background auto-scan / push alerts keep their stricter gates.

**Verification:** `python -m py_compile sakz_bot.py` OK; full suite 68 tests (67 pass, 1 skip).
