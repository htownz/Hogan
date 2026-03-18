"""Tests for swarm vote balance fix — signal-aware safety agents.

Validates that the structural vote imbalance (hold always wins) is
resolved by making safety agents adopt the pipeline direction when
they have no objection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from hogan_bot.swarm_decision.agents._utils import get_baseline_action
from hogan_bot.swarm_decision.agents.risk_steward import RiskStewardAgent
from hogan_bot.swarm_decision.agents.data_guardian import DataGuardianAgent
from hogan_bot.swarm_decision.agents.execution_cost import ExecutionCostAgent
from hogan_bot.swarm_decision.controller import SwarmController
from hogan_bot.swarm_decision.types import AgentVote


def _make_candles(n: int = 100, price: float = 50_000.0) -> pd.DataFrame:
    """Generate synthetic candle data for testing."""
    import numpy as np
    ts = [1_700_000_000_000 + i * 3_600_000 for i in range(n)]
    closes = [price + np.random.normal(0, price * 0.001) for _ in range(n)]
    return pd.DataFrame({
        "ts_ms": ts,
        "open": closes,
        "high": [c * 1.002 for c in closes],
        "low": [c * 0.998 for c in closes],
        "close": closes,
        "volume": [100.0] * n,
    })


def _make_shared_context(action: str = "buy", confidence: float = 0.8) -> dict:
    signal = SimpleNamespace(
        action=action,
        confidence=confidence,
        stop_distance_pct=0.02,
        forecast=None,
    )
    return {
        "regime": "trending",
        "regime_state": {},
        "equity_usd": 10_000.0,
        "peak_equity_usd": 10_000.0,
        "atr_pct": 0.015,
        "hist_vol_20": 0.01,
        "take_profit_pct": 0.03,
        "pipeline_signal": signal,
        "up_prob": 0.65,
    }


# ---------------------------------------------------------------------------
# get_baseline_action utility
# ---------------------------------------------------------------------------

class TestGetBaselineAction:
    def test_returns_buy_from_pipeline(self):
        ctx = _make_shared_context("buy")
        assert get_baseline_action(ctx) == "buy"

    def test_returns_sell_from_pipeline(self):
        ctx = _make_shared_context("sell")
        assert get_baseline_action(ctx) == "sell"

    def test_returns_hold_from_pipeline(self):
        ctx = _make_shared_context("hold")
        assert get_baseline_action(ctx) == "hold"

    def test_returns_hold_when_no_signal(self):
        assert get_baseline_action({}) == "hold"

    def test_returns_hold_when_signal_action_none(self):
        ctx = {"pipeline_signal": SimpleNamespace(action=None)}
        assert get_baseline_action(ctx) == "hold"


# ---------------------------------------------------------------------------
# RiskSteward — signal-aware
# ---------------------------------------------------------------------------

class TestRiskStewardSignalAware:
    def test_no_objection_adopts_pipeline_buy(self):
        agent = RiskStewardAgent()
        candles = _make_candles()
        ctx = _make_shared_context("buy")
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "buy"
        assert vote.veto is False
        assert vote.confidence > 0.5

    def test_no_objection_adopts_pipeline_sell(self):
        agent = RiskStewardAgent()
        candles = _make_candles()
        ctx = _make_shared_context("sell")
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "sell"
        assert vote.veto is False

    def test_drawdown_veto_forces_hold(self):
        agent = RiskStewardAgent(max_drawdown_pct=0.10)
        candles = _make_candles()
        ctx = _make_shared_context("buy")
        ctx["equity_usd"] = 8_000.0
        ctx["peak_equity_usd"] = 10_000.0
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "hold"
        assert vote.veto is True
        assert vote.confidence == 0.0
        assert any("drawdown" in r for r in vote.block_reasons)

    def test_vol_spike_veto_forces_hold(self):
        agent = RiskStewardAgent(vol_veto_threshold=4.0)
        candles = _make_candles()
        ctx = _make_shared_context("buy")
        ctx["atr_pct"] = 0.05
        ctx["hist_vol_20"] = 0.01
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "hold"
        assert vote.veto is True
        assert any("vol_spike" in r for r in vote.block_reasons)

    def test_drawdown_warning_still_adopts_direction(self):
        agent = RiskStewardAgent(max_drawdown_pct=0.10)
        candles = _make_candles()
        ctx = _make_shared_context("buy")
        ctx["equity_usd"] = 9_400.0
        ctx["peak_equity_usd"] = 10_000.0
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "buy"
        assert vote.veto is False
        assert vote.size_scale < 1.0
        assert any("drawdown_warning" in r for r in vote.block_reasons)

    def test_configurable_thresholds(self):
        agent = RiskStewardAgent(
            max_drawdown_pct=0.20,
            vol_scale_threshold=5.0,
            vol_veto_threshold=8.0,
        )
        candles = _make_candles()
        ctx = _make_shared_context("buy")
        ctx["atr_pct"] = 0.05
        ctx["hist_vol_20"] = 0.01
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "buy"
        assert vote.veto is False

    def test_confidence_proportional_to_size_scale(self):
        agent = RiskStewardAgent()
        candles = _make_candles()
        ctx = _make_shared_context("buy")
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        expected_conf = 0.5 + 0.5 * vote.size_scale
        assert abs(vote.confidence - expected_conf) < 0.01


# ---------------------------------------------------------------------------
# DataGuardian — signal-aware
# ---------------------------------------------------------------------------

class TestDataGuardianSignalAware:
    def test_good_data_adopts_pipeline_buy(self):
        agent = DataGuardianAgent()
        candles = _make_candles(100)
        ctx = _make_shared_context("buy")
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "buy"
        assert vote.veto is False
        assert vote.confidence > 0.5

    def test_insufficient_bars_veto(self):
        agent = DataGuardianAgent(min_bars_required=50)
        candles = _make_candles(10)
        ctx = _make_shared_context("buy")
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "hold"
        assert vote.veto is True

    def test_stale_data_veto(self):
        agent = DataGuardianAgent(max_stale_hours=2.0)
        candles = _make_candles(100)
        ctx = _make_shared_context("buy")
        latest_ts = int(candles["ts_ms"].iloc[-1])
        as_of = latest_ts + int(3 * 3600 * 1000)
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=as_of, shared_context=ctx)
        assert vote.action == "hold"
        assert vote.veto is True
        assert any("stale" in r for r in vote.block_reasons)

    def test_configurable_min_bars(self):
        agent = DataGuardianAgent(min_bars_required=10)
        candles = _make_candles(15)
        ctx = _make_shared_context("buy")
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.veto is False
        assert vote.action == "buy"


# ---------------------------------------------------------------------------
# ExecutionCost — signal-aware
# ---------------------------------------------------------------------------

class TestExecutionCostSignalAware:
    def test_sufficient_edge_adopts_pipeline_buy(self):
        agent = ExecutionCostAgent(fee_rate=0.001, min_edge_over_cost=1.5)
        candles = _make_candles(100)
        ctx = _make_shared_context("buy")
        ctx["atr_pct"] = 0.02
        ctx["take_profit_pct"] = 0.03
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "buy"
        assert vote.veto is False
        assert vote.confidence >= 0.5

    def test_negative_edge_vetoes(self):
        agent = ExecutionCostAgent(fee_rate=0.05, min_edge_over_cost=1.5)
        candles = _make_candles(100)
        ctx = _make_shared_context("buy")
        ctx["atr_pct"] = 0.001
        ctx["take_profit_pct"] = 0.001
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "hold"
        assert vote.veto is True

    def test_insufficient_edge_data_adopts_baseline(self):
        agent = ExecutionCostAgent()
        candles = _make_candles(100)
        ctx = _make_shared_context("sell")
        ctx["atr_pct"] = 0.0
        ctx["take_profit_pct"] = 0.0
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "sell"
        assert vote.veto is False

    def test_configurable_fee_rate_and_edge_ratio(self):
        agent = ExecutionCostAgent(fee_rate=0.0001, min_edge_over_cost=1.0)
        candles = _make_candles(100)
        ctx = _make_shared_context("buy")
        ctx["atr_pct"] = 0.005
        ctx["take_profit_pct"] = 0.005
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.veto is False
        assert vote.action == "buy"


# ---------------------------------------------------------------------------
# Controller end-to-end — directional decisions now possible
# ---------------------------------------------------------------------------

class _MockPipelineAgent:
    agent_id = "pipeline_v1"

    def __init__(self, action: str = "buy", confidence: float = 0.8):
        self._action = action
        self._confidence = confidence

    def vote(self, *, symbol, candles, as_of_ms, shared_context) -> AgentVote:
        return AgentVote(
            agent_id=self.agent_id,
            action=self._action,
            confidence=self._confidence,
            size_scale=1.0,
            veto=False,
        )


class TestControllerEndToEnd:
    def _make_config(self, **overrides):
        defaults = {
            "swarm_min_agreement": 0.35,
            "swarm_min_vote_margin": 0.05,
            "swarm_max_entropy": 1.20,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_buy_wins_with_clean_conditions(self):
        config = self._make_config()
        candles = _make_candles(100)
        ctx = _make_shared_context("buy", 0.8)

        agents = [
            _MockPipelineAgent("buy", 0.8),
            RiskStewardAgent(),
            DataGuardianAgent(),
            ExecutionCostAgent(fee_rate=0.001),
        ]
        weights = {
            "pipeline_v1": 0.40,
            "risk_steward_v1": 0.20,
            "data_guardian_v1": 0.20,
            "execution_cost_v1": 0.20,
        }
        ctrl = SwarmController(agents=agents, weights=weights, config=config)
        decision = ctrl.decide(symbol="BTC/USD", candles=candles, shared_context=ctx)

        assert decision.final_action == "buy"
        assert decision.vetoed is False
        assert decision.agreement > 0.5

    def test_sell_wins_with_clean_conditions(self):
        config = self._make_config()
        candles = _make_candles(100)
        ctx = _make_shared_context("sell", 0.8)

        agents = [
            _MockPipelineAgent("sell", 0.8),
            RiskStewardAgent(),
            DataGuardianAgent(),
            ExecutionCostAgent(fee_rate=0.001),
        ]
        weights = {
            "pipeline_v1": 0.40,
            "risk_steward_v1": 0.20,
            "data_guardian_v1": 0.20,
            "execution_cost_v1": 0.20,
        }
        ctrl = SwarmController(agents=agents, weights=weights, config=config)
        decision = ctrl.decide(symbol="BTC/USD", candles=candles, shared_context=ctx)

        assert decision.final_action == "sell"
        assert decision.vetoed is False

    def test_veto_forces_hold_despite_directional_consensus(self):
        config = self._make_config()
        candles = _make_candles(100)
        ctx = _make_shared_context("buy", 0.9)
        ctx["equity_usd"] = 8_000.0
        ctx["peak_equity_usd"] = 10_000.0

        agents = [
            _MockPipelineAgent("buy", 0.9),
            RiskStewardAgent(max_drawdown_pct=0.10),
            DataGuardianAgent(),
            ExecutionCostAgent(fee_rate=0.001),
        ]
        ctrl = SwarmController(agents=agents, config=config)
        decision = ctrl.decide(symbol="BTC/USD", candles=candles, shared_context=ctx)

        assert decision.final_action == "hold"
        assert decision.vetoed is True

    def test_hold_pipeline_produces_hold(self):
        config = self._make_config()
        candles = _make_candles(100)
        ctx = _make_shared_context("hold", 0.5)

        agents = [
            _MockPipelineAgent("hold", 0.5),
            RiskStewardAgent(),
            DataGuardianAgent(),
            ExecutionCostAgent(fee_rate=0.001),
        ]
        ctrl = SwarmController(agents=agents, config=config)
        decision = ctrl.decide(symbol="BTC/USD", candles=candles, shared_context=ctx)

        assert decision.final_action == "hold"
        assert decision.vetoed is False


# ---------------------------------------------------------------------------
# hist_vol NaN handling
# ---------------------------------------------------------------------------

class TestHistVolNaN:
    def test_risk_steward_handles_nan_hist_vol(self):
        agent = RiskStewardAgent()
        candles = _make_candles()
        ctx = _make_shared_context("buy")
        ctx["hist_vol_20"] = float("nan")
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.veto is False
        assert vote.action == "buy"

    def test_risk_steward_handles_zero_hist_vol(self):
        agent = RiskStewardAgent()
        candles = _make_candles()
        ctx = _make_shared_context("buy")
        ctx["hist_vol_20"] = 0.0
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.veto is False
        assert vote.action == "buy"

    def test_risk_steward_handles_inf_hist_vol(self):
        agent = RiskStewardAgent()
        candles = _make_candles()
        ctx = _make_shared_context("buy")
        ctx["hist_vol_20"] = float("inf")
        vote = agent.vote(symbol="BTC/USD", candles=candles, as_of_ms=None, shared_context=ctx)
        assert vote.action == "buy"
        assert vote.veto is False
