"""Unit tests for the newly-wired dormant modules.

Covers:
  * sakz_explain.explanation_block  (wired into format_signal_details)
  * sakz_validation.{monte_carlo_edge_test, regime_split_report,
    walk_forward_splits, max_drawdown}  (wired into the /btfull backtest)
  * The exact outcome->label normalization the backtest uses so that
    regime_split_report counts wins/losses correctly.

Pure-logic only: no network, no exchange, no telegram deps required.
"""
import random

from sakz_explain import explanation_block, BUCKET_LABELS, reason_summary, regime_tag
from sakz_validation import (
    walk_forward_splits,
    regime_split_report,
    max_drawdown,
    monte_carlo_edge_test,
)

# Mirror of the map used in sakz_backtest_hist._aggregate
_OUTCOME_MAP = {'T1': 't1_hit', 'T2': 't2_hit', 'T3': 't3_hit',
                'SL': 'sl_hit', 'TIMEOUT': 'timeout'}
_PNL = {'T1': 1.2, 'T2': 2.4, 'T3': 3.6, 'SL': -1.5, 'TIMEOUT': -0.2}


def _sample_trades(n=40, seed=1):
    random.seed(seed)
    trades = []
    for i in range(n):
        oc = random.choice(['T1', 'T2', 'T3', 'SL', 'TIMEOUT'])
        trades.append({
            'bar': i, 'bias': random.choice(['LONG', 'SHORT']),
            'outcome': oc, 'pnl_pct': _PNL[oc], 'rr': 1.5,
            'btc_regime': random.choice(['BULL', 'BEAR', 'NEUTRAL']),
        })
    return trades


# ---------------------------------------------------------------- explain

def test_bucket_labels_are_accurate():
    # vg must read as Structure and ig as Independent (the fix we shipped).
    d = dict(BUCKET_LABELS)
    assert d['vg'] == 'Struct'
    assert d['ig'] == 'Indep'


def test_explanation_block_basic():
    buckets = {'osc_g': 3, 'mtf_g': 2, 'cross_g': 1, 'pos_g': 3, 'vg': 4, 'ig': 2}
    out = explanation_block(
        buckets, 'BULL',
        ['EMA stack aligned', 'RSI reset from oversold', 'MACD cross up'],
        counter_trend_active=False, top_n=2,
    )
    assert isinstance(out, str) and out.strip()
    # Buckets render with the corrected labels.
    assert 'Struct:4' in out
    assert 'Indep:2' in out
    # Top reasons surface in the summary.
    assert 'EMA stack aligned' in out


def test_explanation_block_handles_empty_inputs():
    # The wired call passes {} when score_buckets is missing; must not raise.
    assert isinstance(explanation_block({}, 'NEUTRAL', [], top_n=2), str)
    assert isinstance(reason_summary([], top_n=2), str)
    assert isinstance(regime_tag('NEUTRAL'), str)


# ---------------------------------------------------------------- validation

def test_outcome_map_matches_win_loss_vocab():
    # The normalized labels must line up with sakz_risk's win/loss vocab so
    # regime_split_report counts correctly.
    from sakz_risk import WIN_OUTCOMES, LOSS_OUTCOMES
    assert _OUTCOME_MAP['T1'] in WIN_OUTCOMES
    assert _OUTCOME_MAP['T2'] in WIN_OUTCOMES
    assert _OUTCOME_MAP['T3'] in WIN_OUTCOMES
    assert _OUTCOME_MAP['SL'] in LOSS_OUTCOMES


def test_regime_split_report_counts_wins():
    trades = _sample_trades()
    vt = [dict(t, outcome=_OUTCOME_MAP[t['outcome']], pnl=t['pnl_pct']) for t in trades]
    rs = regime_split_report(vt, regime_key='btc_regime',
                             outcome_key='outcome', pnl_key='pnl')
    assert rs, 'expected at least one regime bucket'
    total = sum(v['trades'] for v in rs.values())
    assert total == len(trades)
    for v in rs.values():
        assert 0.0 <= v['win_rate'] <= 1.0
        assert v['wins'] + v['losses'] <= v['trades']


def test_monte_carlo_edge_test_shape():
    trades = _sample_trades()
    mc = monte_carlo_edge_test([t['pnl_pct'] for t in trades])
    for k in ('real_total', 'p_value', 'mean_random', 'pct95_random',
              'n_iter', 'has_edge'):
        assert k in mc
    assert 0.0 <= mc['p_value'] <= 1.0
    assert isinstance(mc['has_edge'], bool)


def test_walk_forward_splits_are_out_of_sample():
    splits = walk_forward_splits(list(range(4)), train_span=1, test_span=1)
    assert splits == [([0], [1]), ([1], [2]), ([2], [3])]
    # train and test folds never overlap.
    for train, test in splits:
        assert not (set(train) & set(test))


def test_max_drawdown_positive_fraction():
    eq = []
    e = 1.0
    for t in _sample_trades():
        e *= (1.0 + t['pnl_pct'] / 100.0)
        eq.append(e)
    dd = max_drawdown(eq)
    assert dd >= 0.0
    # A monotonically rising curve has zero drawdown.
    assert max_drawdown([1.0, 1.1, 1.2, 1.3]) == 0.0


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f'PASS {fn.__name__}')
    print(f'\n{passed}/{len(fns)} wiring tests passed')
