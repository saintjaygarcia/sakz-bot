# Integration guide — TODO sections 1–8

This guide shows exactly where each new module plugs into `sakz_bot.py`. The new
modules are **self-contained and unit-tested** (`tests/test_strategy_modules.py`,
`tests/test_ml_outcomes.py`). The monolith itself can't be executed in this
environment (no `ta` / `ccxt` / `telegram` / `libsql` / network), so the wiring
steps below are written against the real call sites but were **not run end-to-end
here**. Wire one section at a time and confirm on paper/testnet.

---

## Section 1 — Critical ML training bug (DONE + verified)

**Bug:** `xgboost_train.py` and `rf_train.py` queried
`WHERE outcome IN ('win','loss')`, but the resolver writes
`t1_hit / t2_hit / t3_hit / sl_hit` (see `sakz_bot.py` outcome resolver and the
`signal_outcomes.outcome` column). The query matched **zero** rows, so the models
had been training on an empty / stale label set.

**Fix (already applied):** both trainers now use
```python
WIN_OUTCOMES  = ("t1_hit", "t2_hit", "t3_hit")
LOSS_OUTCOMES = ("sl_hit",)
RESOLVED_OUTCOMES = WIN_OUTCOMES + LOSS_OUTCOMES
# SELECT ... WHERE outcome IN (?,?,?,?)
label = 1 if outcome in WIN_OUTCOMES else 0
```
`rf_train.py` imports the shared loader from `xgboost_train.py` (with an inline
fallback that has the same fix).

**Verify against your real DB before retraining:**
```bash
python tools/verify_outcomes.py            # uses $SAKZ_DB_PATH or sakz_data.db
```
It prints counts per outcome and exits non-zero unless there are ≥30 resolved
rows with both classes present. Only retrain once this passes.

---

## Section 2 — Refactor / hygiene (DONE)

- DB layer extracted to `sakz_db.py`; exchange layer consolidated in
  `sakz_exchanges.py` (+ `exchange_adapters.py`). Call sites unchanged via
  re-exports.
- Current `sakz_bot.py` already has **no bare `except:`**, a **single**
  `start_command` definition, and **no leftover monkey-patches** (`flip_patch`
  removed). The remaining large item — splitting `score_pair` — is documented as
  a phased plan at the end of `CHANGES.md` and intentionally deferred until
  recorded-candle fixtures exist (it must be characterization-tested first).

---

## Section 3 — T2/T3 hit rate → `sakz_trade_mgmt.py`

Wire into the trade-tracking / resolve loop (the job that scans
`signal_outcomes WHERE outcome='pending'`). After T1 is hit for a signal:

```python
import sakz_trade_mgmt as tm

# once best_target_hit reaches 't1':
new_sl = tm.manage_after_t1(bias, current_price, atr, current_sl,
                            trail_mult=1.0)        # breakeven + ATR trail
db_update_signal_sl(signal_id, new_sl)            # persist via sakz_db
```

At **signal creation** (in/after `score_pair`, before storing T2/T3), clamp
extended targets in chop:
```python
t2, t3 = tm.cap_targets_for_regime(entry, atr, bias, t2, t3, btc_regime)
# only LOW / RANGING regimes are capped (t2≤2·ATR, t3≤3.5·ATR); others untouched
```
All helpers are **tighten-only** (never loosen a stop) and direction-aware.

---

## Section 4 — Signal explanation layer → `sakz_explain.py`

Inside `format_signal(...)`, after the score buckets and BTC regime are known:

```python
import sakz_explain as explain

scores = dict(osc_g=osc_g, mtf_g=mtf_g, cross_g=cross_g,
              pos_g=pos_g, vg=vg, ig=ig)
card += "\n" + explain.bucket_breakdown(scores)            # Osc:3 MTF:2 MACD:2 …
card += "\n" + explain.regime_tag(btc_regime, counter_trend_active)
card += "\nWhy: " + explain.reason_summary(reasons, top_n=2)
# or one call: explain.explanation_block(scores, btc_regime, reasons, counter_trend_active)
```
`reasons` may be a list of strings or `(text, weight)` pairs; weighted input is
sorted so the strongest drivers surface first.

---

## Section 5 — Order flow → `sakz_orderflow.py`

Uses CVD/volume that `sakz_conviction.py` already gathers. In `score_pair`'s
volume/structure block:

```python
import sakz_orderflow as of

div = of.delta_divergence(closes_4h, cvd_4h, lookback=10)
if (bias == 'LONG'  and div == of.BEARISH_DIVERGENCE) or \
   (bias == 'SHORT' and div == of.BULLISH_DIVERGENCE):
    vg -= 1                      # flow disagrees with the trade

vg += of.absorption_score(close_prev, close_now, last_vol, avg_vol,
                          pivot_levels, atr)   # +1 if absorption at S/R
```

**Liquidation data** needs a live feed (Bybit/Binance liquidation stream — add a
fetch in `sakz_ws.py` / `sakz_ccxt.py`). Once you have recent liquidation price
levels, gate the stop:
```python
if of.liquidation_cluster_risk(stop_loss, liq_levels, atr):
    # SL sits in a liquidation magnet — widen, skip, or flag the signal
```
`liquidation_cluster_risk` is pure/tested; only the feed is external.

---

## Section 6 — Portfolio sizing → `sakz_risk.py`

At position sizing (near `calculate_leverage`):

```python
import sakz_risk as risk

win_rate = risk.win_rate_from_outcomes(recent_outcomes)   # from sakz_db history
size = risk.kelly_position_size(equity, win_rate, payoff_ratio,
                                fraction=0.25)            # ¼-Kelly, capped 25%
size = risk.correlated_exposure_cap(symbol, open_symbols, corr_matrix, size)
```

Keep one process-wide drawdown gate and check it before every new entry:
```python
GATE = risk.DrawdownGate(threshold=0.15)   # module-level singleton
...
if not GATE.can_open(current_equity):      # pauses new entries past 15% DD
    return  # skip new signals; existing trades still managed
```
`corr_matrix` comes from Section 8.

---

## Section 7 — Statistical validation → `sakz_validation.py`

Use from `sakz_backtest_hist.py` (its `score_pair` contract is unchanged):

```python
import sakz_validation as val

for train_years, test_years in val.walk_forward_splits(all_years, 1, 1):
    fit_on(train_years); evaluate_on(test_years)     # no look-ahead

print(val.regime_split_report(trades))               # win rate/pnl per BTC regime
edge = val.monte_carlo_edge_test(trade_returns)      # permutation p-value
if not edge['has_edge']:
    print('WARNING: edge not distinguishable from luck (p=%.3f)' % edge['p_value'])
```

---

## Section 8 — Cross-asset correlation → `sakz_correlation.py`

Once per scan cycle, after fetching closes for the scanned universe:

```python
import sakz_correlation as corr

matrix = corr.rolling_correlation_matrix(closes_by_symbol, window=20)

# (a) demote BTC-beta 'breakouts'
if corr.is_btc_dependent(symbol, matrix):
    confidence -= 1   # it's a BTC move, not an independent setup

# (b) altcoin risk regime from ETH/BTC ratio
if corr.eth_btc_ratio_regime(eth_closes, btc_closes) == 'ALT_RISK_OFF':
    # tighten alt filters / require higher confidence

# (c) sector clustering — damp correlated follow-ons within a cycle
confidence = corr.follow_on_confidence(confidence, symbol, already_signalled)
```
Feed `matrix` into `sakz_risk.correlated_exposure_cap` (Section 6) to avoid
stacking correlated positions.

---

## Test & verify

```bash
cd <project root>
python -m unittest discover -s tests -t .      # 57 tests, all green
python -m py_compile *.py tests/*.py tools/*.py
python tools/verify_outcomes.py                # against your real DB
```
