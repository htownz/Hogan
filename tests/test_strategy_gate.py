"""Strategy-gate smoke tests — CI-enforced invariants for the backtest
engine.

These are **not** edge tests (no claim about Sharpe > X on real data); they
are *regression guards* that fail the CI if a change introduces any of:

* catastrophic loss (equity collapses below a floor);
* drawdown exceeding the configured risk budget (the risk engine is
  supposed to enforce this);
* NaN/Inf in a core backtest metric;
* a pipeline wiring regression (a deterministic trending tape that used to
  generate trades suddenly produces zero);
* a feature-registry / champion-mode schema drift that ``run_backtest``
  no longer cleanly tolerates.

The goal: merging a PR that silently breaks the strategy floor should be
impossible without explicitly changing the thresholds in this file.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hogan_bot.backtest import run_backtest_on_candles


def _synthetic_trending_candles(n: int = 400, seed: int = 11) -> pd.DataFrame:
    """Deterministic trending OHLCV suitable for pipeline smoke tests.

    We stitch together short regime blocks (up-trend → pullback → up-trend
    → down-trend) so the regime classifier spends a meaningful fraction of
    bars in a directional regime — enough for the entry gate to fire at
    least a handful of trades. We also pump volume on trend bars so the
    volume-threshold filter doesn't silently kill every setup.
    """
    rng = np.random.RandomState(seed)
    base = 50_000.0

    # regime blocks (drift_per_bar, vol, rel_volume_mean, length)
    blocks = [
        (+0.0025, 0.006, 2.0, int(n * 0.30)),   # clean up-trend
        (-0.0005, 0.004, 1.0, int(n * 0.10)),   # chop / pause
        (+0.0030, 0.006, 2.2, int(n * 0.25)),   # strong up-trend
        (-0.0025, 0.006, 1.8, int(n * 0.20)),   # clean down-trend (short opps)
        (+0.0015, 0.005, 1.5, n - int(n * 0.85)),  # finish
    ]
    returns = np.concatenate([
        rng.normal(drift, vol, length) for (drift, vol, _, length) in blocks
    ])[:n]
    vol_mult = np.concatenate([
        np.full(length, rel) for (_, _, rel, length) in blocks
    ])[:n]

    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.002, 0.010, n))
    low = close * (1 - rng.uniform(0.002, 0.010, n))
    open_ = close + rng.randn(n) * close * 0.003
    volume = rng.uniform(400, 1_000, n) * vol_mult
    ts_ms = np.arange(n) * 3_600_000 + 1_700_000_000_000
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "ts_ms": ts_ms,
    })
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df


_BT_KWARGS = dict(
    symbol="BTC/USD",
    starting_balance_usd=10_000.0,
    aggressive_allocation=0.30,
    max_risk_per_trade=0.02,
    max_drawdown=0.20,
    short_ma_window=8,
    long_ma_window=21,
    volume_window=20,
    volume_threshold=1.0,
    fee_rate=0.001,
    use_policy_core=True,
)


class TestStrategyGateStructuralSanity:
    """Floor contract for the backtest engine — these must hold for ANY run."""

    @pytest.fixture(scope="class")
    def result(self):
        candles = _synthetic_trending_candles(400, seed=11)
        return run_backtest_on_candles(candles, **_BT_KWARGS)

    def test_runs_to_completion(self, result):
        assert result is not None

    def test_equity_curve_is_finite(self, result):
        assert result.start_equity > 0
        for eq in result.equity_curve:
            assert math.isfinite(eq), "equity_curve contains non-finite value"

    def test_end_equity_above_floor(self, result):
        """Soft catastrophic-loss floor: losing >80% of starting capital on a
        trending tape is a pipeline defect, not strategy variance."""
        floor = result.start_equity * 0.20
        assert result.end_equity > floor, (
            f"end_equity={result.end_equity:.2f} collapsed below 20% floor "
            f"({floor:.2f}) on a trending tape — check risk wiring."
        )

    def test_drawdown_respects_budget(self, result):
        """The engine-enforced DD cap is 20% (max_drawdown=0.20). A margin
        above 100% of budget allows for single-bar spikes between peak and
        trough; anything larger means the guard failed."""
        budget_pct = _BT_KWARGS["max_drawdown"] * 100.0
        assert result.max_drawdown_pct <= budget_pct * 2.0, (
            f"max_drawdown_pct={result.max_drawdown_pct:.2f} exceeds 2× budget "
            f"({budget_pct * 2:.2f}) — DrawdownGuard may have broken."
        )

    def test_trade_counters_are_consistent(self, result):
        assert result.trades >= 0
        assert 0.0 <= result.win_rate <= 1.0
        # If any trades fired, we must have recorded them
        if result.trades > 0:
            assert len(result.closed_trades) > 0

    def test_sharpe_sortino_are_sane(self, result):
        for name in ("sharpe_ratio", "sortino_ratio", "calmar_ratio"):
            val = getattr(result, name)
            if val is None:
                continue
            assert math.isfinite(val), f"{name} is non-finite: {val!r}"


class TestStrategyGatePipelineLiveness:
    """If this test starts failing on a trending tape, the gating pipeline
    has regressed to silent-skip mode."""

    def test_trending_tape_produces_trades(self):
        candles = _synthetic_trending_candles(500, seed=11)
        result = run_backtest_on_candles(candles, **_BT_KWARGS)
        assert result.trades > 0, (
            "Zero trades on a 500-bar uptrending synthetic tape means the "
            "decision pipeline has regressed — either signals aren't firing, "
            "the agent pipeline is rejecting every bar, or position sizing "
            "is flooring to zero."
        )


class TestStrategyGateDeterminism:
    """Two successive backtests on the same candles must produce identical
    aggregate metrics. Non-determinism here usually means module-level
    randomness leaking into the decision path."""

    def test_same_inputs_same_outputs(self):
        candles = _synthetic_trending_candles(300, seed=11)
        r1 = run_backtest_on_candles(candles, **_BT_KWARGS)
        r2 = run_backtest_on_candles(candles, **_BT_KWARGS)
        assert r1.trades == r2.trades
        assert r1.end_equity == pytest.approx(r2.end_equity, rel=1e-9)
        assert r1.max_drawdown_pct == pytest.approx(r2.max_drawdown_pct, rel=1e-9)
        assert r1.win_rate == pytest.approx(r2.win_rate, rel=1e-9)
