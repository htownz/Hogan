"""Tests for ML confidence-based position sizing."""
from __future__ import annotations

import pytest

from hogan_bot.decision import apply_ml_filter, ml_confidence
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

    def test_scale_clamped_above_one(self):
        base = self._base_size()
        above = calculate_position_size(
            equity_usd=10_000,
            price=50_000,
            stop_distance_pct=0.02,
            max_risk_per_trade=0.03,
            max_allocation_pct=0.75,
            confidence_scale=2.0,
        )
        assert above == pytest.approx(base)

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
        # prob=0.8 > buy_threshold=0.55 → action stays "buy"
        action = apply_ml_filter("buy", 0.8, buy_threshold=0.55, sell_threshold=0.45)
        assert action == "buy"
        assert ml_confidence(0.8) == pytest.approx(0.6)

    def test_low_confidence_buy_vetoed(self):
        action = apply_ml_filter("buy", 0.4, buy_threshold=0.55, sell_threshold=0.45)
        assert action == "hold"
        # Scale is computed regardless of filter outcome
        assert ml_confidence(0.4) == pytest.approx(0.2)
