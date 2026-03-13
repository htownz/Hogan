"""Dedicated tests for hogan_bot.risk — position sizing and drawdown guard."""
from __future__ import annotations

import pytest

from hogan_bot.risk import DrawdownGuard, calculate_position_size


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestCalculatePositionSize:
    def test_basic_sizing(self):
        size = calculate_position_size(
            equity_usd=10_000, price=100.0, stop_distance_pct=0.02,
            max_risk_per_trade=0.01, max_allocation_pct=0.20,
        )
        # Risk budget = 100, stop at 2% → risk-based size = 100/(100*0.02) = 50
        # Allocation cap = 2000/100 = 20
        assert size == pytest.approx(20.0)

    def test_risk_limited_not_allocation(self):
        size = calculate_position_size(
            equity_usd=10_000, price=100.0, stop_distance_pct=0.02,
            max_risk_per_trade=0.001, max_allocation_pct=0.50,
        )
        # Risk budget = 10, stop at 2% → 10/(100*0.02) = 5
        # Allocation cap = 5000/100 = 50
        assert size == pytest.approx(5.0)

    def test_zero_equity_returns_zero(self):
        assert calculate_position_size(0, 100, 0.02, 0.01, 0.2) == 0.0

    def test_zero_price_returns_zero(self):
        assert calculate_position_size(10_000, 0, 0.02, 0.01, 0.2) == 0.0

    def test_zero_stop_returns_zero(self):
        assert calculate_position_size(10_000, 100, 0, 0.01, 0.2) == 0.0

    def test_confidence_scales_down(self):
        full = calculate_position_size(10_000, 100, 0.02, 0.01, 0.20, confidence_scale=1.0)
        half = calculate_position_size(10_000, 100, 0.02, 0.01, 0.20, confidence_scale=0.5)
        assert half == pytest.approx(full * 0.5)

    def test_confidence_zero_gives_zero(self):
        assert calculate_position_size(10_000, 100, 0.02, 0.01, 0.20, confidence_scale=0.0) == 0.0

    def test_confidence_clamped_above_one(self):
        size = calculate_position_size(10_000, 100, 0.02, 0.01, 0.20, confidence_scale=1.5)
        full = calculate_position_size(10_000, 100, 0.02, 0.01, 0.20, confidence_scale=1.0)
        assert size == pytest.approx(full)

    def test_fee_aware_scaling(self):
        # Stop at 0.2%, fee_rate at 0.1% → stop < 3*fee → scale down
        size_no_fee = calculate_position_size(10_000, 100, 0.002, 0.01, 0.2, fee_rate=0.0)
        size_fee = calculate_position_size(10_000, 100, 0.002, 0.01, 0.2, fee_rate=0.001)
        assert size_fee < size_no_fee

    def test_wide_stop_no_fee_penalty(self):
        # Stop at 3%, fee_rate at 0.1% → stop > 3*fee → no scaling
        size_no_fee = calculate_position_size(10_000, 100, 0.03, 0.01, 0.2, fee_rate=0.0)
        size_fee = calculate_position_size(10_000, 100, 0.03, 0.01, 0.2, fee_rate=0.001)
        assert size_fee == pytest.approx(size_no_fee)

    def test_negative_equity_returns_zero(self):
        assert calculate_position_size(-100, 100, 0.02, 0.01, 0.2) == 0.0


# ---------------------------------------------------------------------------
# Drawdown guard
# ---------------------------------------------------------------------------

class TestDrawdownGuard:
    def test_no_drawdown_ok(self):
        g = DrawdownGuard(10_000, 0.10)
        assert g.update_and_check(10_000) is True
        assert g.update_and_check(11_000) is True

    def test_mild_drawdown_ok(self):
        g = DrawdownGuard(10_000, 0.10)
        g.update_and_check(10_000)
        assert g.update_and_check(9_500) is True

    def test_max_drawdown_breached(self):
        g = DrawdownGuard(10_000, 0.10)
        g.update_and_check(10_000)
        assert g.update_and_check(8_900) is False

    def test_peak_tracks_new_highs(self):
        g = DrawdownGuard(10_000, 0.10)
        g.update_and_check(12_000)
        assert g.peak_equity == 12_000
        # Now 10% of 12000 = 1200, so 10800 is boundary
        assert g.update_and_check(10_800) is True
        assert g.update_and_check(10_700) is False

    def test_zero_equity_fails(self):
        g = DrawdownGuard(10_000, 0.10)
        assert g.update_and_check(0) is False

    def test_recovery_after_drawdown(self):
        g = DrawdownGuard(10_000, 0.20)
        assert g.update_and_check(9_000) is True  # 10% dd
        assert g.update_and_check(10_500) is True  # recovery
        assert g.peak_equity == 10_500
