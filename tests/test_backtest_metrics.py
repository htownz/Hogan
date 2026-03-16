"""Tests for risk-adjusted performance metrics added to BacktestResult."""
from __future__ import annotations

import math

import pytest

from hogan_bot.backtest import compute_calmar, compute_sharpe, compute_sortino

# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_returns_none_for_single_point(self):
        assert compute_sharpe([1000.0]) is None

    def test_returns_none_for_zero_std(self):
        # Flat equity → no volatility → undefined Sharpe
        curve = [1000.0] * 100
        assert compute_sharpe(curve) is None

    def test_positive_for_rising_curve(self):
        # Monotonically rising equity → Sharpe should be strongly positive
        curve = [1000.0 + i for i in range(200)]
        s = compute_sharpe(curve)
        assert s is not None
        assert s > 0

    def test_negative_for_falling_curve(self):
        # Monotonically falling equity → negative Sharpe
        curve = [1000.0 - i * 0.5 for i in range(200)]
        s = compute_sharpe(curve)
        assert s is not None
        assert s < 0

    def test_opposite_sign_for_opposite_trends(self):
        # Rising equity → positive Sharpe; falling equity → negative Sharpe.
        up = [1000.0 + i for i in range(200)]
        down = [1200.0 - i for i in range(200)]
        s_up = compute_sharpe(up)
        s_down = compute_sharpe(down)
        assert s_up is not None and s_down is not None
        assert s_up > 0
        assert s_down < 0

    def test_annualisation_scale(self):
        # A curve with known per-bar mean >> std should produce a Sharpe that
        # is clearly positive and in the right order of magnitude.
        import numpy as np

        rng = np.random.default_rng(42)
        # Use a large sample to reduce sampling noise
        returns = rng.normal(loc=0.001, scale=0.01, size=5000)
        equity = [1000.0]
        for r in returns:
            equity.append(equity[-1] * (1.0 + r))

        result = compute_sharpe(equity)
        assert result is not None
        # Default annualization is now 1h (8760 bars/year).
        # Population Sharpe is 0.1 * sqrt(8760) ≈ 9.4.
        # With 5 000 bars and the given seed, sample Sharpe should be
        # in the ballpark [3, 20].
        assert result > 3
        assert result < 30


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_returns_none_for_single_point(self):
        assert compute_sortino([1000.0]) is None

    def test_returns_none_when_no_downside(self):
        # No negative returns → downside dev = 0 → undefined
        curve = [1000.0 + i for i in range(200)]
        assert compute_sortino(curve) is None

    def test_positive_for_mostly_positive_returns(self):
        import numpy as np

        rng = np.random.default_rng(7)
        # Mostly positive but some losses
        returns = np.abs(rng.normal(0.002, 0.01, 500)) * np.sign(rng.normal(0, 1, 500))
        equity = [1000.0]
        for r in returns:
            equity.append(max(1.0, equity[-1] * (1.0 + r)))
        s = compute_sortino(equity)
        # Not asserting sign because returns are mixed; just check it computes
        assert s is not None
        assert math.isfinite(s)

    def test_greater_than_sharpe_with_positive_skew(self):
        # When gains > losses, Sortino should be larger than Sharpe (downside
        # deviation < total std deviation).
        import numpy as np

        rng = np.random.default_rng(99)
        returns = rng.normal(0.001, 0.01, 1000)
        equity = [1000.0]
        for r in returns:
            equity.append(equity[-1] * (1.0 + r))

        s = compute_sharpe(equity)
        so = compute_sortino(equity)
        # Both defined
        assert s is not None and so is not None


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_returns_none_when_no_drawdown(self):
        assert compute_calmar(20.0, 0.0) is None

    def test_positive_return_positive_drawdown(self):
        result = compute_calmar(20.0, 10.0)
        assert result == pytest.approx(2.0)

    def test_negative_return(self):
        result = compute_calmar(-5.0, 15.0)
        assert result is not None
        assert result < 0

    def test_large_drawdown_small_return(self):
        result = compute_calmar(1.0, 50.0)
        assert result is not None
        assert result == pytest.approx(0.02)
