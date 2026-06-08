"""Phase 0 — signal surfacing threshold tests."""
import unittest, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from sakz_scanner import (
    SCAN_DISPLAY_CONF_MIN,
    AUTOSCAN_DISPLAY_CONF_MIN,
    _conf_of,
    passes_display_floor,
    risk_reasons,
)

def _sig(**kw):
    base = dict(
        confidence=7, bias='LONG', price=1.0, stop_loss=0.9,
        entry_low=0.95, entry_high=1.05, t1=1.2, t2=1.35, t3=1.5,
        quote_volume=5_000_000, btc_regime='neutral',
    )
    base.update(kw)
    return base


class TestDisplayFloor(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(SCAN_DISPLAY_CONF_MIN, 7.0)
        self.assertGreaterEqual(AUTOSCAN_DISPLAY_CONF_MIN, SCAN_DISPLAY_CONF_MIN)

    def test_conf_of_normal(self):
        self.assertEqual(_conf_of(_sig(confidence=8)), 8.0)

    def test_conf_of_missing(self):
        self.assertEqual(_conf_of({}), 0.0)

    def test_conf_of_bad(self):
        self.assertEqual(_conf_of({'confidence': None}), 0.0)

    def test_passes_at_floor(self):
        self.assertTrue(passes_display_floor(_sig(confidence=7)))

    def test_passes_above_floor(self):
        self.assertTrue(passes_display_floor(_sig(confidence=9)))

    def test_fails_below_floor(self):
        self.assertFalse(passes_display_floor(_sig(confidence=4)))

    def test_custom_floor(self):
        self.assertTrue(passes_display_floor(_sig(confidence=8), floor=8))
        self.assertFalse(passes_display_floor(_sig(confidence=7), floor=8))


class TestRiskReasons(unittest.TestCase):
    def test_low_conf_flagged(self):
        reasons = risk_reasons(_sig(confidence=4))
        self.assertTrue(any("confidence" in r.lower() for r in reasons))

    def test_high_conf_no_conf_reason(self):
        reasons = risk_reasons(_sig(confidence=9))
        self.assertFalse(any("confidence" in r.lower() for r in reasons))

    def test_tight_stop_flagged(self):
        # stop 0.1% away from entry — noise territory
        reasons = risk_reasons(_sig(entry_low=1.0, stop_loss=0.999, confidence=4))
        self.assertTrue(any("tight" in r.lower() for r in reasons))

    def test_wide_stop_flagged(self):
        # stop 20% away
        reasons = risk_reasons(_sig(entry_low=1.0, stop_loss=0.80, confidence=4))
        self.assertTrue(any("wide" in r.lower() for r in reasons))

    def test_thin_liquidity_flagged(self):
        reasons = risk_reasons(_sig(confidence=4, quote_volume=500_000))
        self.assertTrue(any("liquidity" in r.lower() for r in reasons))

    def test_bear_regime_long_flagged(self):
        reasons = risk_reasons(_sig(confidence=4, btc_regime='bear_market', bias='LONG'))
        self.assertTrue(any("bearish" in r.lower() for r in reasons))

    def test_bull_regime_short_flagged(self):
        reasons = risk_reasons(_sig(confidence=4, btc_regime='bull_trend', bias='SHORT'))
        self.assertTrue(any("bullish" in r.lower() for r in reasons))

    def test_never_empty(self):
        # marginal but passes floor — must still return something
        reasons = risk_reasons(_sig(confidence=9))
        self.assertTrue(len(reasons) > 0)


if __name__ == '__main__':
    unittest.main()
