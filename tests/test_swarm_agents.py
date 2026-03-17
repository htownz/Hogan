"""Tests for individual swarm agents — valid output, veto logic, edge cases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hogan_bot.swarm_decision.types import AgentVote


def _synthetic_candles(n: int = 200, seed: int = 42) -> pd.DataFrame:
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


class TestDataGuardian:
    """data_guardian_v1 should veto on bad data and pass on good data."""

    def test_valid_candles_no_veto(self):
        from hogan_bot.swarm_decision.agents.data_guardian import DataGuardianAgent
        agent = DataGuardianAgent()
        candles = _synthetic_candles(200)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={},
        )
        assert isinstance(vote, AgentVote)
        assert vote.agent_id == "data_guardian_v1"
        assert vote.veto is False
        assert vote.size_scale > 0

    def test_insufficient_bars_vetoes(self):
        from hogan_bot.swarm_decision.agents.data_guardian import DataGuardianAgent
        agent = DataGuardianAgent(min_bars_required=100)
        candles = _synthetic_candles(30)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={},
        )
        assert vote.veto is True
        assert any("insufficient_bars" in r for r in vote.block_reasons)

    def test_gap_detection_vetoes(self):
        from hogan_bot.swarm_decision.agents.data_guardian import DataGuardianAgent
        agent = DataGuardianAgent(max_gap_bars=2)
        candles = _synthetic_candles(100)
        ts = candles["ts_ms"].copy()
        ts.iloc[50] = ts.iloc[49] + 3600_000 * 10
        for i in range(51, len(ts)):
            ts.iloc[i] = ts.iloc[i - 1] + 3600_000
        candles["ts_ms"] = ts
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={},
        )
        assert vote.veto is True
        assert any("candle_gap" in r for r in vote.block_reasons)

    def test_stale_candles_veto(self):
        from hogan_bot.swarm_decision.agents.data_guardian import DataGuardianAgent
        agent = DataGuardianAgent(max_stale_hours=1.0)
        candles = _synthetic_candles(100)
        last_ts = int(candles["ts_ms"].iloc[-1])
        stale_as_of = last_ts + int(3600_000 * 3)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=stale_as_of,
            shared_context={},
        )
        assert vote.veto is True
        assert any("stale_candles" in r for r in vote.block_reasons)
        assert vote.freshness is not None
        assert vote.freshness.is_stale is True


class TestRiskSteward:
    """risk_steward_v1 should scale down or veto during drawdowns."""

    def test_no_drawdown_no_veto(self):
        from hogan_bot.swarm_decision.agents.risk_steward import RiskStewardAgent
        agent = RiskStewardAgent(max_drawdown_pct=0.10)
        candles = _synthetic_candles(50)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={"equity_usd": 10000, "peak_equity_usd": 10000},
        )
        assert vote.veto is False
        assert vote.size_scale == pytest.approx(1.0)

    def test_deep_drawdown_vetoes(self):
        from hogan_bot.swarm_decision.agents.risk_steward import RiskStewardAgent
        agent = RiskStewardAgent(max_drawdown_pct=0.10)
        candles = _synthetic_candles(50)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={"equity_usd": 8500, "peak_equity_usd": 10000},
        )
        assert vote.veto is True
        assert any("drawdown" in r for r in vote.block_reasons)

    def test_moderate_drawdown_scales_down(self):
        from hogan_bot.swarm_decision.agents.risk_steward import RiskStewardAgent
        agent = RiskStewardAgent(max_drawdown_pct=0.10)
        candles = _synthetic_candles(50)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={"equity_usd": 9300, "peak_equity_usd": 10000},
        )
        assert vote.veto is False
        assert vote.size_scale < 1.0
        assert any("warning" in r for r in vote.block_reasons)

    def test_vol_spike_vetoes(self):
        from hogan_bot.swarm_decision.agents.risk_steward import RiskStewardAgent
        agent = RiskStewardAgent(vol_veto_threshold=3.0)
        candles = _synthetic_candles(50)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={
                "equity_usd": 10000, "peak_equity_usd": 10000,
                "atr_pct": 0.06, "hist_vol_20": 0.01,
            },
        )
        assert vote.veto is True
        assert any("vol_spike" in r for r in vote.block_reasons)


class TestExecutionCost:
    """execution_cost_v1 should veto when costs exceed edge."""

    def test_high_edge_no_veto(self):
        from hogan_bot.swarm_decision.agents.execution_cost import ExecutionCostAgent
        agent = ExecutionCostAgent(fee_rate=0.001)
        candles = _synthetic_candles(100)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={"atr_pct": 0.03, "take_profit_pct": 0.05},
        )
        assert vote.veto is False
        assert vote.expected_edge_bps is not None

    def test_zero_edge_marks_insufficient(self):
        from hogan_bot.swarm_decision.agents.execution_cost import ExecutionCostAgent
        agent = ExecutionCostAgent(fee_rate=0.001)
        candles = _synthetic_candles(100)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={"atr_pct": 0.0, "take_profit_pct": 0.0},
        )
        assert any("insufficient" in r for r in vote.block_reasons)

    def test_returns_valid_agent_vote(self):
        from hogan_bot.swarm_decision.agents.execution_cost import ExecutionCostAgent
        agent = ExecutionCostAgent()
        candles = _synthetic_candles(100)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={"atr_pct": 0.02, "take_profit_pct": 0.03},
        )
        assert isinstance(vote, AgentVote)
        assert vote.agent_id == "execution_cost_v1"
        assert 0.0 <= vote.size_scale <= 1.0


class TestPipelineAgent:
    """pipeline_v1 should wrap AgentPipeline output as AgentVote."""

    def test_returns_valid_vote(self):
        from unittest.mock import MagicMock
        from hogan_bot.swarm_decision.agents.pipeline_agent import PipelineAgent

        mock_pipeline = MagicMock()
        mock_signal = MagicMock()
        mock_signal.action = "buy"
        mock_signal.confidence = 0.75
        mock_signal.stop_distance_pct = 0.02
        mock_signal.forecast = None
        mock_signal.volume_ratio = 1.5
        mock_pipeline.run.return_value = mock_signal

        agent = PipelineAgent(mock_pipeline)
        candles = _synthetic_candles(100)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={"regime": "trending_up"},
        )
        assert isinstance(vote, AgentVote)
        assert vote.agent_id == "pipeline_v1"
        assert vote.action == "buy"
        assert vote.confidence == pytest.approx(0.75)
        assert vote.veto is False

    def test_hold_signal_maps_correctly(self):
        from unittest.mock import MagicMock
        from hogan_bot.swarm_decision.agents.pipeline_agent import PipelineAgent

        mock_pipeline = MagicMock()
        mock_signal = MagicMock()
        mock_signal.action = "hold"
        mock_signal.confidence = 0.3
        mock_signal.stop_distance_pct = 0.01
        mock_signal.forecast = None
        mock_pipeline.run.return_value = mock_signal

        agent = PipelineAgent(mock_pipeline)
        candles = _synthetic_candles(100)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={},
        )
        assert vote.action == "hold"
        assert vote.confidence == pytest.approx(0.3)
