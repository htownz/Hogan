"""Tests for SwarmController — determinism, veto, agreement, entropy."""
from __future__ import annotations

import pytest

from hogan_bot.swarm_decision.controller import SwarmController
from hogan_bot.swarm_decision.types import AgentVote, SwarmDecision


class _StubAgent:
    """Minimal agent for testing that returns a pre-configured vote."""

    def __init__(self, agent_id: str, vote: AgentVote) -> None:
        self.agent_id = agent_id
        self._vote = vote

    def vote(self, *, symbol, candles, as_of_ms, shared_context) -> AgentVote:
        return self._vote


def _make_vote(
    agent_id: str = "test",
    action: str = "buy",
    confidence: float = 0.8,
    size_scale: float = 1.0,
    veto: bool = False,
    block_reasons: list[str] | None = None,
    expected_edge_bps: float | None = None,
) -> AgentVote:
    return AgentVote(
        agent_id=agent_id,
        action=action,
        confidence=confidence,
        size_scale=size_scale,
        veto=veto,
        block_reasons=block_reasons or [],
        expected_edge_bps=expected_edge_bps,
    )


class TestDeterminism:
    """Fixed votes + fixed weights must produce identical SwarmDecision."""

    def test_identical_across_runs(self):
        agents = [
            _StubAgent("a", _make_vote("a", "buy", 0.8)),
            _StubAgent("b", _make_vote("b", "buy", 0.7)),
            _StubAgent("c", _make_vote("c", "hold", 0.5)),
        ]
        weights = {"a": 0.4, "b": 0.35, "c": 0.25}
        import pandas as pd
        candles = pd.DataFrame({"close": [100.0] * 10})

        results = []
        for _ in range(5):
            ctrl = SwarmController(agents=agents, weights=weights)
            d = ctrl.decide(symbol="BTC/USD", candles=candles)
            results.append(d)

        for d in results[1:]:
            assert d.final_action == results[0].final_action
            assert d.final_confidence == pytest.approx(results[0].final_confidence)
            assert d.final_size_scale == pytest.approx(results[0].final_size_scale)
            assert d.agreement == pytest.approx(results[0].agreement)
            assert d.entropy == pytest.approx(results[0].entropy)

    def test_returns_swarm_decision_type(self):
        agents = [_StubAgent("a", _make_vote("a", "buy", 0.9))]
        import pandas as pd
        ctrl = SwarmController(agents=agents)
        d = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1.0]}))
        assert isinstance(d, SwarmDecision)


class TestVeto:
    """One veto agent must force hold."""

    def test_single_veto_forces_hold(self):
        agents = [
            _StubAgent("bull", _make_vote("bull", "buy", 0.95)),
            _StubAgent("guard", _make_vote("guard", "hold", 0.0, veto=True,
                                           block_reasons=["stale_data"])),
        ]
        import pandas as pd
        ctrl = SwarmController(agents=agents)
        d = ctrl.decide(symbol="BTC/USD", candles=pd.DataFrame({"close": [50000.0]}))
        assert d.final_action == "hold"
        assert d.vetoed is True
        assert any("stale_data" in r for r in d.block_reasons)
        assert d.final_size_scale == 0.0

    def test_no_veto_allows_trading(self):
        agents = [
            _StubAgent("bull", _make_vote("bull", "buy", 0.9)),
            _StubAgent("guard", _make_vote("guard", "hold", 0.5, veto=False)),
        ]
        import pandas as pd
        ctrl = SwarmController(agents=agents)
        d = ctrl.decide(symbol="BTC/USD", candles=pd.DataFrame({"close": [50000.0]}))
        assert d.vetoed is False

    def test_multiple_vetoes_all_reasons_captured(self):
        agents = [
            _StubAgent("g1", _make_vote("g1", "hold", 0.0, veto=True,
                                        block_reasons=["gap"])),
            _StubAgent("g2", _make_vote("g2", "hold", 0.0, veto=True,
                                        block_reasons=["stale"])),
        ]
        import pandas as pd
        ctrl = SwarmController(agents=agents)
        d = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1.0]}))
        assert d.vetoed is True
        assert len(d.block_reasons) == 2


class TestAgreementThreshold:
    """Low agreement should suppress trades."""

    def test_strong_agreement_trades(self):
        agents = [
            _StubAgent("a", _make_vote("a", "buy", 0.9)),
            _StubAgent("b", _make_vote("b", "buy", 0.8)),
            _StubAgent("c", _make_vote("c", "buy", 0.7)),
        ]
        import pandas as pd
        ctrl = SwarmController(agents=agents, config=type("C", (), {
            "swarm_min_agreement": 0.60,
            "swarm_min_vote_margin": 0.10,
            "swarm_max_entropy": 0.95,
        })())
        d = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1.0]}))
        assert d.final_action == "buy"

    def test_split_vote_forces_hold(self):
        agents = [
            _StubAgent("a", _make_vote("a", "buy", 0.5)),
            _StubAgent("b", _make_vote("b", "sell", 0.5)),
            _StubAgent("c", _make_vote("c", "hold", 0.5)),
        ]
        import pandas as pd
        ctrl = SwarmController(
            agents=agents,
            weights={"a": 1.0, "b": 1.0, "c": 1.0},
            config=type("C", (), {
                "swarm_min_agreement": 0.60,
                "swarm_min_vote_margin": 0.10,
                "swarm_max_entropy": 0.95,
            })(),
        )
        d = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1.0]}))
        assert d.final_action == "hold"
        assert any("agreement" in r or "margin" in r for r in d.block_reasons)


class TestEntropyThreshold:
    """High entropy should suppress trades."""

    def test_unanimous_low_entropy(self):
        agents = [
            _StubAgent("a", _make_vote("a", "sell", 0.9)),
            _StubAgent("b", _make_vote("b", "sell", 0.85)),
        ]
        import pandas as pd
        ctrl = SwarmController(agents=agents, config=type("C", (), {
            "swarm_min_agreement": 0.50,
            "swarm_min_vote_margin": 0.05,
            "swarm_max_entropy": 0.95,
        })())
        d = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1.0]}))
        assert d.entropy < 0.5

    def test_votes_field_populated(self):
        agents = [
            _StubAgent("a", _make_vote("a", "buy", 0.8)),
            _StubAgent("b", _make_vote("b", "sell", 0.6)),
        ]
        import pandas as pd
        ctrl = SwarmController(agents=agents)
        d = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1.0]}))
        assert len(d.votes) == 2
        assert d.votes[0].agent_id == "a"
        assert d.votes[1].agent_id == "b"


class TestWeightManagement:
    """Verify weight normalization and updates."""

    def test_weights_normalize_to_one(self):
        agents = [
            _StubAgent("a", _make_vote("a")),
            _StubAgent("b", _make_vote("b")),
        ]
        ctrl = SwarmController(agents=agents, weights={"a": 3.0, "b": 7.0})
        assert ctrl.weights["a"] == pytest.approx(0.3)
        assert ctrl.weights["b"] == pytest.approx(0.7)

    def test_set_weights_renormalizes(self):
        agents = [_StubAgent("a", _make_vote("a"))]
        ctrl = SwarmController(agents=agents)
        ctrl.set_weights({"a": 5.0})
        assert ctrl.weights["a"] == pytest.approx(1.0)

    def test_weights_used_in_decision(self):
        import pandas as pd
        agents = [
            _StubAgent("a", _make_vote("a", "buy", 0.8)),
            _StubAgent("b", _make_vote("b", "sell", 0.8)),
        ]
        ctrl = SwarmController(
            agents=agents,
            weights={"a": 0.9, "b": 0.1},
            config=type("C", (), {
                "swarm_min_agreement": 0.0,
                "swarm_min_vote_margin": 0.0,
                "swarm_max_entropy": 10.0,
            })(),
        )
        d = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1.0]}))
        assert d.final_action == "buy"


class TestSerialization:
    """Verify to_dict produces valid output."""

    def test_to_dict_complete(self):
        agents = [
            _StubAgent("a", _make_vote("a", "buy", 0.8)),
        ]
        import pandas as pd
        ctrl = SwarmController(agents=agents)
        d = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1.0]}))
        out = d.to_dict()
        assert "final_action" in out
        assert "votes" in out
        assert isinstance(out["votes"], list)
        assert "weights_used" in out
