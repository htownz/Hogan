"""Tests for agent_quarantine module — mode control and enforcement."""
from __future__ import annotations

import sqlite3

from hogan_bot.agent_quarantine import (
    get_agent_mode,
    get_agent_state,
    get_all_agent_modes,
    get_mode_history,
    is_agent_advisory_only,
    is_agent_quarantined,
    is_agent_veto_enabled,
    set_agent_mode,
)
from hogan_bot.swarm_decision.controller import SwarmController
from hogan_bot.swarm_decision.types import AgentVote


def _make_conn():
    conn = sqlite3.connect(":memory:")
    from hogan_bot.storage import _create_schema
    _create_schema(conn)
    return conn


class TestAgentModeBasics:
    def test_default_is_active(self):
        conn = _make_conn()
        assert get_agent_mode("risk_steward_v1", conn) == "active"
        conn.close()

    def test_set_and_get_mode(self):
        conn = _make_conn()
        set_agent_mode("risk_steward_v1", "no_veto", "operator_a", "too many vetoes", conn)
        assert get_agent_mode("risk_steward_v1", conn) == "no_veto"
        conn.close()

    def test_set_quarantined(self):
        conn = _make_conn()
        set_agent_mode("data_guardian_v1", "quarantined", "op", "broken data", conn)
        assert is_agent_quarantined("data_guardian_v1", conn)
        assert not is_agent_veto_enabled("data_guardian_v1", conn)
        conn.close()

    def test_advisory_only(self):
        conn = _make_conn()
        set_agent_mode("risk_steward_v1", "advisory_only", "op", "test", conn)
        assert is_agent_advisory_only("risk_steward_v1", conn)
        assert not is_agent_veto_enabled("risk_steward_v1", conn)
        conn.close()

    def test_veto_enabled_only_when_active(self):
        conn = _make_conn()
        assert is_agent_veto_enabled("x", conn) is True
        set_agent_mode("x", "no_veto", "op", "test", conn)
        assert is_agent_veto_enabled("x", conn) is False
        conn.close()

    def test_mode_state_includes_reason(self):
        conn = _make_conn()
        set_agent_mode("agent_a", "quarantined", "ben", "test reason", conn)
        state = get_agent_state("agent_a", conn)
        assert state.mode == "quarantined"
        assert state.reason == "test reason"
        assert state.operator == "ben"
        conn.close()

    def test_get_all_agent_modes(self):
        conn = _make_conn()
        set_agent_mode("a1", "active", "op", "r1", conn)
        set_agent_mode("a2", "no_veto", "op", "r2", conn)
        modes = get_all_agent_modes(conn)
        assert len(modes) == 2
        assert modes["a2"].mode == "no_veto"
        conn.close()

    def test_mode_history(self):
        conn = _make_conn()
        set_agent_mode("a1", "active", "op", "init", conn)
        set_agent_mode("a1", "no_veto", "op", "tune", conn)
        set_agent_mode("a1", "quarantined", "op", "broken", conn)
        hist = get_mode_history("a1", conn)
        assert len(hist) == 3
        assert hist[0]["mode"] == "quarantined"
        conn.close()


class _FakeAgent:
    def __init__(self, agent_id: str, action: str = "buy", conf: float = 0.8, veto: bool = False):
        self.agent_id = agent_id
        self._action = action
        self._conf = conf
        self._veto = veto

    def vote(self, **kwargs) -> AgentVote:
        return AgentVote(
            agent_id=self.agent_id, action=self._action,
            confidence=self._conf, veto=self._veto,
            block_reasons=["test_reason"] if self._veto else [],
        )


class TestControllerModeEnforcement:
    def test_quarantined_agent_excluded(self):
        import pandas as pd
        agents = [_FakeAgent("a", "buy", 0.8), _FakeAgent("b", "buy", 0.8)]
        ctrl = SwarmController(agents=agents)
        dec = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1]}),
                          agent_modes={"b": "quarantined"})
        agent_ids_in_votes = [v.agent_id for v in dec.votes]
        assert "b" not in agent_ids_in_votes

    def test_no_veto_strips_veto(self):
        import pandas as pd
        agents = [_FakeAgent("pipeline", "buy", 0.9), _FakeAgent("risk", "hold", 0.0, veto=True)]
        ctrl = SwarmController(agents=agents)
        dec = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1]}),
                          agent_modes={"risk": "no_veto"})
        assert dec.vetoed is False

    def test_advisory_only_excluded_from_fusion(self):
        import pandas as pd
        agents = [_FakeAgent("pipeline", "buy", 0.9), _FakeAgent("advisor", "sell", 0.9)]
        ctrl = SwarmController(agents=agents, weights={"pipeline": 0.5, "advisor": 0.5})
        dec = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1]}),
                          agent_modes={"advisor": "advisory_only"})
        assert dec.final_action == "buy"

    def test_advisory_veto_ignored(self):
        import pandas as pd
        agents = [_FakeAgent("pipeline", "buy", 0.9), _FakeAgent("risk", "hold", 0.0, veto=True)]
        ctrl = SwarmController(agents=agents)
        dec = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1]}),
                          agent_modes={"risk": "advisory_only"})
        assert dec.vetoed is False


class TestPreVetoCapture:
    def test_pre_veto_fields_populated_on_veto(self):
        import pandas as pd
        agents = [_FakeAgent("pipeline", "buy", 0.9), _FakeAgent("risk", "hold", 0.0, veto=True)]
        ctrl = SwarmController(agents=agents)
        dec = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1]}))
        assert dec.vetoed is True
        assert dec.pre_veto_action is not None
        assert dec.dominant_veto_agent == "risk"
        assert dec.veto_count >= 1

    def test_pre_veto_fields_populated_on_non_veto(self):
        import pandas as pd
        agents = [_FakeAgent("pipeline", "buy", 0.9)]
        ctrl = SwarmController(agents=agents)
        dec = ctrl.decide(symbol="X", candles=pd.DataFrame({"close": [1]}))
        assert dec.vetoed is False
        assert dec.pre_veto_action is not None
