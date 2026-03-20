"""Dedicated tests for hogan_bot.decision — edge gate, entry quality, ML filter, spread estimation, adaptive confidence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hogan_bot.decision import (
    AdaptiveConfidence,
    QualityComponents,
    apply_ml_filter,
    compute_quality_components,
    edge_gate,
    entry_quality_gate,
    estimate_spread_from_candles,
    estimate_spread_from_order_book,
    ml_confidence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candles(n: int = 50, base: float = 100.0, noise: float = 0.005) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    close = base + np.cumsum(rng.randn(n) * noise * base)
    high = close * (1 + rng.uniform(0.001, 0.004, n))
    low = close * (1 - rng.uniform(0.001, 0.004, n))
    return pd.DataFrame({
        "open": close + rng.randn(n) * 0.01,
        "high": high, "low": low, "close": close,
        "volume": rng.uniform(100, 1000, n),
    })


# ---------------------------------------------------------------------------
# ML filter
# ---------------------------------------------------------------------------

class TestApplyMlFilter:
    def test_buy_passes_above_threshold(self):
        assert apply_ml_filter("buy", 0.65, 0.55, 0.45).action == "buy"

    def test_buy_blocked_below_threshold(self):
        gd = apply_ml_filter("buy", 0.50, 0.55, 0.45)
        assert gd.action == "hold"
        assert gd.blocked_by == "ml_filter_buy"

    def test_sell_passes_below_threshold(self):
        assert apply_ml_filter("sell", 0.30, 0.55, 0.45).action == "sell"

    def test_sell_blocked_above_threshold(self):
        gd = apply_ml_filter("sell", 0.50, 0.55, 0.45)
        assert gd.action == "hold"
        assert gd.blocked_by == "ml_filter_sell"

    def test_hold_passes_through(self):
        assert apply_ml_filter("hold", 0.99, 0.55, 0.45).action == "hold"

    def test_nan_passes_through(self):
        assert apply_ml_filter("buy", float("nan"), 0.55, 0.45).action == "buy"

    def test_none_passes_through(self):
        assert apply_ml_filter("sell", None, 0.55, 0.45).action == "sell"


# ---------------------------------------------------------------------------
# ML confidence
# ---------------------------------------------------------------------------

class TestMlConfidence:
    def test_half_gives_zero(self):
        assert ml_confidence(0.5) == 0.0

    def test_extremes_give_one(self):
        assert ml_confidence(0.0) == pytest.approx(1.0)
        assert ml_confidence(1.0) == pytest.approx(1.0)

    def test_moderate_prob(self):
        assert 0.0 < ml_confidence(0.65) < 1.0

    def test_nan_gives_zero(self):
        assert ml_confidence(float("nan")) == 0.0

    def test_none_gives_zero(self):
        assert ml_confidence(None) == 0.0

    def test_clamped_above_one(self):
        assert ml_confidence(1.5) == pytest.approx(1.0)

    def test_clamped_below_zero(self):
        assert ml_confidence(-0.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Edge gate
# ---------------------------------------------------------------------------

class TestEdgeGate:
    def test_hold_passes_through(self):
        assert edge_gate("hold", 0.01, 0.05, 0.001).action == "hold"

    def test_sufficient_edge_passes(self):
        gd = edge_gate("buy", atr_pct=0.02, take_profit_pct=0.05,
                        fee_rate=0.001, estimated_spread=0.0005)
        assert gd.action == "buy"
        assert gd.blocked_by is None

    def test_low_atr_blocked(self):
        gd = edge_gate("buy", atr_pct=0.001, take_profit_pct=0.05,
                        fee_rate=0.002, estimated_spread=0.001)
        assert gd.action == "hold"
        assert gd.blocked_by == "edge_gate_atr_low"

    def test_tp_below_friction_blocked(self):
        gd = edge_gate("buy", atr_pct=0.05, take_profit_pct=0.002,
                        fee_rate=0.001, min_edge_multiple=1.5,
                        estimated_spread=0.001)
        assert gd.action == "hold"
        assert gd.blocked_by == "edge_gate_tp_low"

    def test_forecast_below_friction_blocked(self):
        gd = edge_gate("buy", atr_pct=0.05, take_profit_pct=0.05,
                        fee_rate=0.002, forecast_expected_return=0.001,
                        estimated_spread=0.001)
        assert gd.action == "hold"
        assert gd.blocked_by == "edge_gate_forecast_low"

    def test_high_spread_blocked(self):
        gd = edge_gate("buy", atr_pct=0.012, take_profit_pct=0.05,
                        fee_rate=0.0005, estimated_spread=0.005)
        assert gd.action == "hold"
        assert gd.blocked_by is not None

    def test_sell_also_gated(self):
        gd = edge_gate("sell", atr_pct=0.001, take_profit_pct=0.05,
                        fee_rate=0.002)
        assert gd.action == "hold"

    def test_zero_spread_passes_other_checks(self):
        gd = edge_gate("buy", atr_pct=0.02, take_profit_pct=0.05,
                        fee_rate=0.001, estimated_spread=0.0)
        assert gd.action == "buy"


# ---------------------------------------------------------------------------
# Entry quality gate
# ---------------------------------------------------------------------------

class TestEntryQualityGate:
    def test_hold_passes_through(self):
        gd = entry_quality_gate("hold", final_confidence=0.9)
        assert gd.action == "hold"
        assert gd.size_scale == 1.0

    def test_high_confidence_passes(self):
        gd = entry_quality_gate("buy", final_confidence=0.8,
                                tech_confidence=0.7,
                                regime_confidence=0.6)
        assert gd.action == "buy"
        assert gd.size_scale == 1.0

    def test_low_final_confidence_blocks(self):
        gd = entry_quality_gate("buy", final_confidence=0.1)
        assert gd.action == "hold"
        assert gd.blocked_by == "quality_gate_final_conf"

    def test_low_tech_confidence_blocks(self):
        gd = entry_quality_gate("buy", final_confidence=0.8,
                                tech_confidence=0.05)
        assert gd.action == "hold"
        assert gd.blocked_by == "quality_gate_tech_conf"

    def test_low_regime_confidence_blocks(self):
        gd = entry_quality_gate("buy", final_confidence=0.8,
                                tech_confidence=0.8,
                                regime_confidence=0.3)
        assert gd.action == "hold"
        assert gd.blocked_by == "quality_gate_regime_conf"

    def test_whipsaw_reduces_scale(self):
        gd = entry_quality_gate("buy", final_confidence=0.8,
                                recent_whipsaw_count=3)
        assert gd.action == "buy"
        assert gd.size_scale < 1.0

    def test_many_whipsaws_blocks(self):
        gd = entry_quality_gate("buy", final_confidence=0.8,
                                recent_whipsaw_count=10)
        assert gd.action == "hold"
        assert gd.blocked_by == "quality_gate_whipsaw"

    def test_none_confidences_pass(self):
        gd = entry_quality_gate("buy")
        assert gd.action == "buy"
        assert gd.size_scale == 1.0


# ---------------------------------------------------------------------------
# QualityComponents
# ---------------------------------------------------------------------------

class TestQualityComponents:
    def test_compute_returns_dataclass(self):
        qc = compute_quality_components(
            final_confidence=0.5,
            tech_confidence=0.6,
            regime_confidence=0.7,
            up_prob=0.75,
        )
        assert isinstance(qc, QualityComponents)
        assert 0.0 <= qc.overall <= 1.0

    def test_all_high_inputs_give_high_overall(self):
        qc = compute_quality_components(
            final_confidence=0.6,
            tech_confidence=0.7,
            regime_confidence=0.75,
            up_prob=0.9,
            estimated_spread=0.0001,
            atr_pct=0.02,
            recent_whipsaw_count=0,
        )
        assert qc.overall > 0.7

    def test_poor_inputs_give_low_overall(self):
        qc = compute_quality_components(
            final_confidence=0.05,
            tech_confidence=0.05,
            regime_confidence=0.1,
            up_prob=0.51,
            estimated_spread=0.005,
            atr_pct=0.01,
            recent_whipsaw_count=5,
        )
        assert qc.overall < 0.3

    def test_ranging_scale_dampens(self):
        qc_full = compute_quality_components(final_confidence=0.5, ranging_scale=1.0)
        qc_damp = compute_quality_components(final_confidence=0.5, ranging_scale=0.5)
        assert qc_damp.overall < qc_full.overall

    def test_to_json(self):
        qc = compute_quality_components(final_confidence=0.5)
        j = qc.to_json()
        assert '"overall"' in j

    def test_freshness_penalty(self):
        qc_fresh = compute_quality_components(final_confidence=0.5, freshness_summary=None)
        qc_stale = compute_quality_components(
            final_confidence=0.5,
            freshness_summary={"stale_count": 3, "critical_stale_count": 1},
        )
        assert qc_stale.freshness_penalty < qc_fresh.freshness_penalty


# ---------------------------------------------------------------------------
# Spread estimation
# ---------------------------------------------------------------------------

class TestSpreadEstimation:
    def test_candles_returns_positive(self):
        candles = _candles(50)
        spread = estimate_spread_from_candles(candles)
        assert 0.0 < spread < 0.01

    def test_none_candles_returns_default(self):
        assert estimate_spread_from_candles(None) == pytest.approx(0.0005)

    def test_short_candles_returns_positive(self):
        candles = _candles(2)
        spread = estimate_spread_from_candles(candles)
        assert spread == pytest.approx(0.0005)

    def test_clamped_below_max(self):
        spread = estimate_spread_from_candles(_candles(50))
        assert spread <= 0.005

    def test_clamped_above_min(self):
        spread = estimate_spread_from_candles(_candles(50))
        assert spread >= 0.0001

    def test_order_book_normal(self):
        bids = [[100.0, 10], [99.9, 5]]
        asks = [[100.1, 10], [100.2, 5]]
        spread = estimate_spread_from_order_book(bids, asks)
        expected = (100.1 - 100.0) / (2 * 100.05)
        assert spread == pytest.approx(expected, rel=0.01)

    def test_order_book_empty(self):
        assert estimate_spread_from_order_book([], []) == 0.001


# ---------------------------------------------------------------------------
# Adaptive confidence
# ---------------------------------------------------------------------------

class TestAdaptiveConfidence:
    def test_fresh_instance_returns_base(self):
        ac = AdaptiveConfidence()
        result = ac.compute(0.75)
        assert result == pytest.approx(ml_confidence(0.75))

    def test_good_track_record_boosts_recency(self):
        ac = AdaptiveConfidence()
        for _ in range(30):
            ac.record_outcome(0.7, 1)
        assert ac._recency_accuracy_factor() > 1.0

    def test_good_track_record_vs_bad(self):
        ac_good = AdaptiveConfidence()
        ac_bad = AdaptiveConfidence()
        for _ in range(30):
            ac_good.record_outcome(0.7, 1)
            ac_bad.record_outcome(0.7, 0)
        assert ac_good.compute(0.7) > ac_bad.compute(0.7)

    def test_bad_track_record_dampens(self):
        ac = AdaptiveConfidence()
        for _ in range(30):
            ac.record_outcome(0.7, 0)
        dampened = ac.compute(0.7)
        base = ml_confidence(0.7)
        assert dampened <= base

    def test_agreement_boosts(self):
        ac = AdaptiveConfidence()
        agree = ac.compute(0.75, forecast_bias=0.10)
        neutral = ac.compute(0.75, forecast_bias=0.0)
        assert agree >= neutral

    def test_disagreement_dampens(self):
        ac = AdaptiveConfidence()
        disagree = ac.compute(0.75, forecast_bias=-0.10)
        neutral = ac.compute(0.75, forecast_bias=0.0)
        assert disagree <= neutral

    def test_volatile_regime_reduces(self):
        ac = AdaptiveConfidence()
        vol = ac.compute(0.75, regime="volatile")
        normal = ac.compute(0.75, regime=None)
        assert vol < normal

    def test_diagnostics_empty(self):
        ac = AdaptiveConfidence()
        d = ac.get_diagnostics()
        assert d["n_outcomes"] == 0
        assert d["recency_factor"] == 1.0

    def test_diagnostics_populated(self):
        ac = AdaptiveConfidence()
        for _ in range(25):
            ac.record_outcome(0.6, 1)
        d = ac.get_diagnostics()
        assert d["n_outcomes"] == 25
        assert d["overall_accuracy"] > 0

    def test_max_history_eviction(self):
        ac = AdaptiveConfidence(max_history=10)
        for i in range(20):
            ac.record_outcome(0.6, 1)
        assert len(ac._predictions) == 10

    def test_output_always_in_unit_interval(self):
        ac = AdaptiveConfidence()
        for _ in range(50):
            ac.record_outcome(0.9, 1)
        for prob in [0.0, 0.1, 0.5, 0.9, 1.0]:
            result = ac.compute(prob, forecast_bias=0.3, regime="volatile")
            assert 0.0 <= result <= 1.0
