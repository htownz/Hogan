"""Tests for ML confidence-based position sizing."""
from __future__ import annotations

import pytest

from hogan_bot.decision import apply_ml_filter, ml_confidence, ml_probability_sizer
from hogan_bot.risk import calculate_position_size


# ---------------------------------------------------------------------------
# ml_confidence()
# ---------------------------------------------------------------------------


class TestMlConfidence:
    def test_half_probability_is_zero(self):
        assert ml_confidence(0.5) == pytest.approx(0.0)

    def test_certain_bull_is_one(self):
        assert ml_confidence(1.0) == pytest.approx(1.0)

    def test_certain_bear_is_one(self):
        assert ml_confidence(0.0) == pytest.approx(1.0)

    def test_midpoint_values(self):
        assert ml_confidence(0.75) == pytest.approx(0.5)
        assert ml_confidence(0.25) == pytest.approx(0.5)

    def test_slightly_above_half(self):
        assert ml_confidence(0.6) == pytest.approx(0.2)

    def test_output_clamped_to_one(self):
        # Probabilities outside [0,1] are technically invalid but should not
        # produce a scale > 1.
        assert ml_confidence(1.5) == pytest.approx(1.0)

    def test_returns_float(self):
        result = ml_confidence(0.7)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# calculate_position_size() — confidence_scale integration
# ---------------------------------------------------------------------------


class TestPositionSizeConfidenceScale:
    def _base_size(self):
        return calculate_position_size(
            equity_usd=10_000,
            price=50_000,
            stop_distance_pct=0.02,
            max_risk_per_trade=0.03,
            max_allocation_pct=0.75,
        )

    def test_default_scale_unchanged(self):
        base = self._base_size()
        scaled = calculate_position_size(
            equity_usd=10_000,
            price=50_000,
            stop_distance_pct=0.02,
            max_risk_per_trade=0.03,
            max_allocation_pct=0.75,
            confidence_scale=1.0,
        )
        assert scaled == pytest.approx(base)

    def test_half_scale(self):
        base = self._base_size()
        half = calculate_position_size(
            equity_usd=10_000,
            price=50_000,
            stop_distance_pct=0.02,
            max_risk_per_trade=0.03,
            max_allocation_pct=0.75,
            confidence_scale=0.5,
        )
        assert half == pytest.approx(base * 0.5)

    def test_zero_scale_returns_zero(self):
        size = calculate_position_size(
            equity_usd=10_000,
            price=50_000,
            stop_distance_pct=0.02,
            max_risk_per_trade=0.03,
            max_allocation_pct=0.75,
            confidence_scale=0.0,
        )
        assert size == pytest.approx(0.0)

    def test_scale_clamped_above_ceiling(self):
        base = self._base_size()
        above = calculate_position_size(
            equity_usd=10_000,
            price=50_000,
            stop_distance_pct=0.02,
            max_risk_per_trade=0.03,
            max_allocation_pct=0.75,
            confidence_scale=2.0,
        )
        assert above == pytest.approx(base * 1.5)

    def test_scale_above_one_scales_up(self):
        base = self._base_size()
        boosted = calculate_position_size(
            equity_usd=10_000,
            price=50_000,
            stop_distance_pct=0.02,
            max_risk_per_trade=0.03,
            max_allocation_pct=0.75,
            confidence_scale=1.3,
        )
        assert boosted == pytest.approx(base * 1.3)

    def test_scale_clamped_below_zero(self):
        size = calculate_position_size(
            equity_usd=10_000,
            price=50_000,
            stop_distance_pct=0.02,
            max_risk_per_trade=0.03,
            max_allocation_pct=0.75,
            confidence_scale=-0.5,
        )
        assert size == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# apply_ml_filter + ml_confidence round-trip
# ---------------------------------------------------------------------------


class TestConfidenceFilterRoundTrip:
    def test_high_confidence_buy_passes_filter(self):
        gd = apply_ml_filter("buy", 0.8, buy_threshold=0.55, sell_threshold=0.45)
        assert gd.action == "buy"
        assert ml_confidence(0.8) == pytest.approx(0.6)

    def test_low_confidence_buy_vetoed(self):
        gd = apply_ml_filter("buy", 0.4, buy_threshold=0.55, sell_threshold=0.45)
        assert gd.action == "hold"
        assert ml_confidence(0.4) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Regression: conf_scale must NOT multiply pipeline confidence into ML sizer
# ---------------------------------------------------------------------------


class TestConfScaleNoDoubleCounting:
    """Guard against the bug where conf_scale = ml_sizer * pipeline_confidence.

    The ML probability sizer output (0.30-1.50) IS the sizing scale.
    Pipeline confidence is checked by quality gate thresholds, not sizing.
    Multiplying them together crushes positions to ~10% of intended size.
    """

    def test_ml_sizer_output_is_independent_of_pipeline_confidence(self):
        """ml_probability_sizer should not depend on any external confidence."""
        scale_buy = ml_probability_sizer("buy", 0.55)
        assert scale_buy == pytest.approx(1.15, abs=0.01)
        scale_sell = ml_probability_sizer("sell", 0.45)
        assert scale_sell == pytest.approx(1.15, abs=0.01)

    def test_ml_sizer_floor_and_ceiling(self):
        assert ml_probability_sizer("buy", 0.0) == pytest.approx(0.30)
        assert ml_probability_sizer("buy", 1.0) == pytest.approx(1.50)
        assert ml_probability_sizer("sell", 1.0) == pytest.approx(0.30)
        assert ml_probability_sizer("sell", 0.0) == pytest.approx(1.50)

    def test_hold_returns_unity(self):
        assert ml_probability_sizer("hold", 0.7) == pytest.approx(1.0)

    def test_none_prob_returns_unity(self):
        assert ml_probability_sizer("buy", None) == pytest.approx(1.0)

    def test_typical_ml_sizer_not_crushed_by_confidence(self):
        """The bug was: conf_scale = ml_sizer(0.41) * pipeline_conf(0.15) = 0.10.

        Correct: conf_scale = ml_sizer(0.41) = 0.73.  This test ensures
        the sizer output is in a reasonable range on its own.
        """
        sizer_output = ml_probability_sizer("buy", 0.41)
        assert sizer_output > 0.5, (
            f"ML sizer for buy@0.41 should be >0.5, got {sizer_output}. "
            "If this fails, pipeline confidence may be leaking into conf_scale."
        )
