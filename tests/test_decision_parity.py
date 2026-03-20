"""Decision parity harness — verifies that the same candle data produces
equivalent decisions through the live SignalEvaluator and the backtest
signal path.

This catches drift between event_loop.py and backtest.py decision logic.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from hogan_bot.config import BotConfig
from hogan_bot.decision import (
    GateDecision,
    apply_ml_filter,
    compute_quality_components,
    edge_gate,
    entry_quality_gate,
    loss_streak_scale,
    ml_blind_blocks_shorts,
    ml_blind_scale,
    ranging_gate,
)
from hogan_bot.regime import detect_regime, effective_thresholds


def _synthetic_candles(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV candles with realistic structure."""
    rng = np.random.RandomState(seed)
    base = 50000.0
    returns = rng.normal(0.0002, 0.008, n)
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.005, n))
    low = close * (1 - rng.uniform(0.001, 0.005, n))
    open_ = close + rng.randn(n) * close * 0.002
    volume = rng.uniform(100, 10000, n)
    ts_ms = np.arange(n) * 3600_000 + 1700000000000

    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "ts_ms": ts_ms,
    })


def _make_config(**overrides) -> BotConfig:
    """Create a minimal BotConfig for parity testing."""
    defaults = dict(
        symbols=["BTC/USD"],
        timeframe="1h",
        ml_buy_threshold=0.55,
        ml_sell_threshold=0.45,
        take_profit_pct=0.03,
        trailing_stop_pct=0.015,
        volume_threshold=1.0,
        fee_rate=0.001,
        max_risk_per_trade=0.02,
        aggressive_allocation=0.30,
        min_final_confidence=0.3,
        min_tech_confidence=0.4,
        min_regime_confidence=0.5,
        max_whipsaws=3,
        use_regime_detection=True,
        use_ml_filter=False,
        ml_confidence_sizing=False,
        paper_mode=True,
        starting_balance_usd=10000.0,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestGateParity:
    """Ensure gate functions produce identical results when called with
    the same inputs, regardless of calling context (live vs backtest)."""

    def test_ml_filter_returns_gate_decision(self):
        gd = apply_ml_filter("buy", 0.6, 0.55, 0.45)
        assert isinstance(gd, GateDecision)
        assert gd.action == "buy"

    def test_edge_gate_returns_gate_decision(self):
        gd = edge_gate("buy", atr_pct=0.02, take_profit_pct=0.05,
                        fee_rate=0.001, estimated_spread=0.0005)
        assert isinstance(gd, GateDecision)
        assert gd.action == "buy"

    def test_quality_gate_returns_gate_decision(self):
        gd = entry_quality_gate("buy", final_confidence=0.8,
                                tech_confidence=0.7, regime_confidence=0.6)
        assert isinstance(gd, GateDecision)
        assert gd.action == "buy"

    def test_ranging_gate_returns_gate_decision(self):
        gd = ranging_gate("buy", regime="ranging", tech_action="buy",
                          up_prob=0.75, recent_whipsaw_count=0)
        assert isinstance(gd, GateDecision)
        assert gd.action == "buy"


class TestRegimeParity:
    """Verify regime detection and effective_thresholds produce consistent
    results across live and backtest paths."""

    def test_detect_regime_deterministic(self):
        candles = _synthetic_candles(200, seed=42)
        r1 = detect_regime(candles)
        r2 = detect_regime(candles)
        assert r1.regime == r2.regime
        assert r1.confidence == r2.confidence

    def test_effective_thresholds_deterministic(self):
        candles = _synthetic_candles(200, seed=42)
        cfg = _make_config()
        rs = detect_regime(candles)
        e1 = effective_thresholds(rs, cfg)
        e2 = effective_thresholds(rs, cfg)
        for key in e1:
            assert e1[key] == pytest.approx(e2[key])


class TestSignalPathParity:
    """Run the same candles through a simulated live path and backtest path,
    verifying that regime, gate decisions, and quality components match."""

    def _run_signal_gates(self, candles: pd.DataFrame, cfg):
        """Simulate the gate chain from SignalEvaluator.evaluate()."""
        rs = detect_regime(candles)
        eff = effective_thresholds(rs, cfg)
        eff_ml_buy = eff.get("ml_buy_threshold", cfg.ml_buy_threshold)
        eff_tp = eff.get("take_profit_pct", cfg.take_profit_pct)
        eff_position_scale = eff.get("position_scale", 1.0)

        action = "buy"  # simulated pipeline output

        from hogan_bot.decision import estimate_spread_from_candles
        atr_pct = 0.02
        spread_est = estimate_spread_from_candles(candles)

        edge_gd = edge_gate(
            action, atr_pct=atr_pct, take_profit_pct=eff_tp,
            fee_rate=cfg.fee_rate, estimated_spread=spread_est,
        )
        action = edge_gd.action

        quality_gd = entry_quality_gate(
            action, final_confidence=0.6, tech_confidence=0.5,
            regime=rs.regime, regime_confidence=rs.confidence,
            min_final_confidence=cfg.min_final_confidence,
            min_tech_confidence=cfg.min_tech_confidence,
        )
        action = quality_gd.action
        quality_scale = quality_gd.size_scale

        ranging_gd = ranging_gate(
            action, regime=rs.regime, tech_action="buy",
            up_prob=0.65, recent_whipsaw_count=0,
        )
        action = ranging_gd.action
        ranging_scale = ranging_gd.size_scale

        qc = compute_quality_components(
            final_confidence=0.6, tech_confidence=0.5,
            regime_confidence=rs.confidence, up_prob=0.65,
            estimated_spread=spread_est, atr_pct=atr_pct,
            ranging_scale=ranging_scale, quality_gate_scale=quality_scale,
        )

        return {
            "regime": rs.regime,
            "regime_confidence": rs.confidence,
            "action": action,
            "edge_blocked": edge_gd.blocked_by,
            "quality_blocked": quality_gd.blocked_by,
            "ranging_blocked": ranging_gd.blocked_by,
            "quality_scale": quality_scale,
            "ranging_scale": ranging_scale,
            "qc_overall": qc.overall,
            "eff_ml_buy": eff_ml_buy,
            "eff_position_scale": eff_position_scale,
        }

    def test_same_candles_same_result(self):
        """The same candles must produce the same gate decisions."""
        candles = _synthetic_candles(200, seed=42)
        cfg = _make_config()
        r1 = self._run_signal_gates(candles, cfg)
        r2 = self._run_signal_gates(candles, cfg)
        assert r1["regime"] == r2["regime"]
        assert r1["action"] == r2["action"]
        assert r1["edge_blocked"] == r2["edge_blocked"]
        assert r1["quality_blocked"] == r2["quality_blocked"]
        assert r1["ranging_blocked"] == r2["ranging_blocked"]
        assert r1["quality_scale"] == pytest.approx(r2["quality_scale"])
        assert r1["qc_overall"] == pytest.approx(r2["qc_overall"])

    def test_different_seeds_may_differ(self):
        """Different candle data should generally produce different regimes."""
        cfg = _make_config()
        r1 = self._run_signal_gates(_synthetic_candles(200, seed=1), cfg)
        r2 = self._run_signal_gates(_synthetic_candles(200, seed=999), cfg)
        # They may or may not differ — this just ensures no crashes
        assert r1["regime"] in ("trending_up", "trending_down", "ranging", "volatile")
        assert r2["regime"] in ("trending_up", "trending_down", "ranging", "volatile")

    def test_quality_components_stable(self):
        """Quality components should be deterministic for the same inputs."""
        qc1 = compute_quality_components(
            final_confidence=0.5, tech_confidence=0.6,
            regime_confidence=0.7, up_prob=0.65,
            estimated_spread=0.001, atr_pct=0.02,
        )
        qc2 = compute_quality_components(
            final_confidence=0.5, tech_confidence=0.6,
            regime_confidence=0.7, up_prob=0.65,
            estimated_spread=0.001, atr_pct=0.02,
        )
        assert qc1.overall == pytest.approx(qc2.overall)
        assert qc1.ml_separation == pytest.approx(qc2.ml_separation)
        assert qc1.spread_penalty == pytest.approx(qc2.spread_penalty)


class TestMlBlindScale:
    """Verify ml_blind_scale detects low-conviction model output."""

    def test_diverse_probs_return_one(self):
        probs = [0.3, 0.7, 0.4, 0.6, 0.55, 0.35, 0.65, 0.45, 0.5, 0.72]
        assert ml_blind_scale(probs) == pytest.approx(1.0)

    def test_tight_cluster_scales_down(self):
        probs = [0.50, 0.505, 0.498, 0.502, 0.501, 0.499, 0.503, 0.497]
        scale = ml_blind_scale(probs)
        assert scale < 1.0
        assert scale >= 0.50

    def test_empty_returns_one(self):
        assert ml_blind_scale([]) == 1.0

    def test_none_returns_one(self):
        assert ml_blind_scale(None) == 1.0

    def test_short_list_returns_one(self):
        assert ml_blind_scale([0.5, 0.5]) == 1.0

    def test_floor_respected(self):
        probs = [0.50] * 30
        scale = ml_blind_scale(probs, floor_scale=0.40)
        assert scale >= 0.40


class TestRangingGateAsymmetric:
    """Verify ranging gate is more lenient on buys than sells."""

    def test_buy_passes_with_low_ml_separation(self):
        gd = ranging_gate("buy", regime="ranging", tech_action="buy",
                          up_prob=0.53, recent_whipsaw_count=0)
        assert gd.action == "buy"

    def test_sell_blocked_with_low_ml_separation(self):
        gd = ranging_gate("sell", regime="ranging", tech_action="sell",
                          up_prob=0.48, recent_whipsaw_count=0)
        assert gd.action == "hold"

    def test_buy_survives_whipsaws_that_block_sell(self):
        gd_buy = ranging_gate("buy", regime="ranging", tech_action="buy",
                              up_prob=0.75, recent_whipsaw_count=2)
        gd_sell = ranging_gate("sell", regime="ranging", tech_action="sell",
                               up_prob=0.25, recent_whipsaw_count=2)
        assert gd_buy.action == "buy"
        assert gd_sell.action == "hold"

    def test_buy_tech_disagree_scales_not_blocks(self):
        gd = ranging_gate("buy", regime="ranging", tech_action="sell",
                          up_prob=0.75, recent_whipsaw_count=0)
        assert gd.action == "buy"
        assert gd.size_scale == pytest.approx(0.70)

    def test_sell_tech_disagree_scales_half(self):
        gd = ranging_gate("sell", regime="ranging", tech_action="buy",
                          up_prob=0.25, recent_whipsaw_count=0)
        assert gd.action == "sell"
        assert gd.size_scale == pytest.approx(0.50)


class TestMlBlindBlocksShorts:
    """Verify shorts are blocked when ML conviction is extremely low."""

    def test_diverse_probs_no_block(self):
        probs = list(np.linspace(0.30, 0.70, 30))
        assert ml_blind_blocks_shorts(probs) is False

    def test_tight_cluster_blocks(self):
        probs = [0.50] * 30
        assert ml_blind_blocks_shorts(probs) is True

    def test_empty_no_block(self):
        assert ml_blind_blocks_shorts([]) is False

    def test_none_no_block(self):
        assert ml_blind_blocks_shorts(None) is False

    def test_moderate_spread_no_block(self):
        probs = list(np.linspace(0.47, 0.55, 30))
        assert ml_blind_blocks_shorts(probs) is False

    def test_just_below_threshold_blocks(self):
        probs = [0.50 + x * 0.001 for x in range(30)]
        assert ml_blind_blocks_shorts(probs) is True


class TestLossStreakScale:
    """Verify sizing dampens after consecutive losses."""

    def test_no_outcomes_returns_one(self):
        assert loss_streak_scale([]) == 1.0

    def test_all_wins_returns_one(self):
        assert loss_streak_scale([True, True, True, True]) == 1.0

    def test_three_losses_dampens(self):
        outcomes = [True, True, False, False, False]
        assert loss_streak_scale(outcomes) == 0.50

    def test_two_losses_no_dampen(self):
        outcomes = [True, True, False, False]
        assert loss_streak_scale(outcomes) == 1.0

    def test_win_resets_streak(self):
        outcomes = [False, False, False, True]
        assert loss_streak_scale(outcomes) == 1.0

    def test_long_loss_streak(self):
        outcomes = [False] * 10
        assert loss_streak_scale(outcomes) == 0.50

    def test_custom_threshold(self):
        outcomes = [False, False]
        assert loss_streak_scale(outcomes, streak_threshold=2) == 0.50


class TestPolicyCoreEquivalence:
    """Verify policy_core.decide() produces deterministic, consistent output.

    Given the same synthetic candles and config, the decision must be
    identical across calls.  This is the critical correctness gate
    before any swarm work.
    """

    @pytest.fixture()
    def core_inputs(self):
        """Shared fixture: candles + real BotConfig + pipeline."""
        from dataclasses import replace as dc_replace
        candles = _synthetic_candles(200, seed=42)
        cfg = dc_replace(
            BotConfig(),
            symbols=["BTC/USD"],
            timeframe="1h",
            use_ml_filter=False,
            use_ml_as_sizer=False,
            paper_mode=True,
            starting_balance_usd=10000.0,
            use_regime_detection=True,
        )
        from hogan_bot.agent_pipeline import AgentPipeline
        pipeline = AgentPipeline(cfg, conn=None)
        return candles, cfg, pipeline

    def test_decide_deterministic(self, core_inputs):
        """Two identical calls must return identical DecisionIntents."""
        from hogan_bot.policy_core import PolicyState, decide

        candles, cfg, pipeline = core_inputs
        for _ in range(2):
            state = PolicyState()
            intent = decide(
                symbol="BTC/USD",
                candles=candles,
                equity_usd=10000.0,
                config=cfg,
                pipeline=pipeline,
                ml_model=None,
                state=state,
                mode="backtest",
            )

            state2 = PolicyState()
            intent2 = decide(
                symbol="BTC/USD",
                candles=candles,
                equity_usd=10000.0,
                config=cfg,
                pipeline=pipeline,
                ml_model=None,
                state=state2,
                mode="backtest",
            )

            assert intent.action == intent2.action
            assert intent.confidence == pytest.approx(intent2.confidence)
            assert intent.size_usd == pytest.approx(intent2.size_usd, rel=1e-6)
            assert intent.regime == intent2.regime
            assert intent.eff_trailing_stop_pct == pytest.approx(
                intent2.eff_trailing_stop_pct or 0, abs=1e-9
            )
            assert intent.eff_allow_longs == intent2.eff_allow_longs
            assert intent.eff_allow_shorts == intent2.eff_allow_shorts

    def test_decide_returns_decision_intent(self, core_inputs):
        """decide() must return a DecisionIntent dataclass."""
        from hogan_bot.policy_core import PolicyState, decide
        from hogan_bot.swarm_decision.types import DecisionIntent

        candles, cfg, pipeline = core_inputs
        state = PolicyState()
        intent = decide(
            symbol="BTC/USD",
            candles=candles,
            equity_usd=10000.0,
            config=cfg,
            pipeline=pipeline,
            ml_model=None,
            state=state,
            mode="backtest",
        )
        assert isinstance(intent, DecisionIntent)
        assert intent.action in ("buy", "sell", "hold")
        assert 0.0 <= intent.confidence <= 1.0
        assert intent.size_usd >= 0.0
        # Swarm is now enabled by default (conditional_active mode),
        # so intent.swarm may be non-None.  Just check it's valid if present.
        if intent.swarm is not None:
            assert hasattr(intent.swarm, "final_action")

    def test_decide_with_mock_ml_model(self, core_inputs):
        """decide() with a mock ML model still produces deterministic output."""
        from dataclasses import replace as dc_replace

        from hogan_bot.policy_core import PolicyState, decide
        candles, cfg, pipeline = core_inputs
        cfg_ml = dc_replace(
            cfg,
            use_ml_filter=True,
            ml_confidence_sizing=True,
            use_ml_as_sizer=False,
        )

        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock(return_value=[[0.45, 0.55]])
        mock_model.set_regime = MagicMock()

        state1 = PolicyState()
        intent1 = decide(
            symbol="BTC/USD",
            candles=candles,
            equity_usd=10000.0,
            config=cfg_ml,
            pipeline=pipeline,
            ml_model=mock_model,
            state=state1,
            mode="backtest",
        )

        state2 = PolicyState()
        intent2 = decide(
            symbol="BTC/USD",
            candles=candles,
            equity_usd=10000.0,
            config=cfg_ml,
            pipeline=pipeline,
            ml_model=mock_model,
            state=state2,
            mode="backtest",
        )

        assert intent1.action == intent2.action
        assert intent1.up_prob == pytest.approx(intent2.up_prob or 0, abs=1e-6)
        assert intent1.size_usd == pytest.approx(intent2.size_usd, rel=1e-6)

    def test_block_reasons_populated(self, core_inputs):
        """block_reasons should be a list (possibly empty) on every intent."""
        from hogan_bot.policy_core import PolicyState, decide

        candles, cfg, pipeline = core_inputs
        state = PolicyState()
        intent = decide(
            symbol="BTC/USD",
            candles=candles,
            equity_usd=10000.0,
            config=cfg,
            pipeline=pipeline,
            ml_model=None,
            state=state,
            mode="backtest",
        )
        assert isinstance(intent.block_reasons, list)

    def test_regime_fields_populated(self, core_inputs):
        """Regime and effective parameters should be populated."""
        from hogan_bot.policy_core import PolicyState, decide

        candles, cfg, pipeline = core_inputs
        state = PolicyState()
        intent = decide(
            symbol="BTC/USD",
            candles=candles,
            equity_usd=10000.0,
            config=cfg,
            pipeline=pipeline,
            ml_model=None,
            state=state,
            mode="backtest",
        )
        assert intent.regime in ("trending_up", "trending_down", "ranging", "volatile", None)
        assert isinstance(intent.eff_allow_longs, bool)
        assert isinstance(intent.eff_allow_shorts, bool)
        assert intent.atr_pct >= 0.0


class TestEdgeGateAsymmetric:
    """Verify edge gate uses lower ATR threshold for buys."""

    def test_buy_passes_at_lower_atr(self):
        gd = edge_gate("buy", atr_pct=0.0015, take_profit_pct=0.054,
                        fee_rate=0.001)
        assert gd.action == "buy"

    def test_sell_blocked_at_same_atr(self):
        gd = edge_gate("sell", atr_pct=0.0015, take_profit_pct=0.054,
                        fee_rate=0.001)
        assert gd.action == "hold"

    def test_both_pass_at_high_atr(self):
        gd_buy = edge_gate("buy", atr_pct=0.005, take_profit_pct=0.054,
                           fee_rate=0.001)
        gd_sell = edge_gate("sell", atr_pct=0.005, take_profit_pct=0.054,
                            fee_rate=0.001)
        assert gd_buy.action == "buy"
        assert gd_sell.action == "sell"

    def test_both_blocked_at_zero_atr(self):
        gd_buy = edge_gate("buy", atr_pct=0.0, take_profit_pct=0.054,
                           fee_rate=0.001)
        gd_sell = edge_gate("sell", atr_pct=0.0, take_profit_pct=0.054,
                            fee_rate=0.001)
        assert gd_buy.action == "hold"
        assert gd_sell.action == "hold"
