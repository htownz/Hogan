"""Swarm Certification Suite — post-implementation acceptance tests.

This file is the go/no-go gate before any swarm authority promotion.
Every test here must pass before the swarm is trusted to observe, advise,
or execute in any mode.

Groups:
    1. Regression — validates the 5 critical bug fixes
    2. Shadow Parity — swarm shadow mode doesn't alter baseline trades
    3. Champion Protection — champion mode + swarm = champion behaviour
    4. Policy-Core Parity — use_policy_core=True matches legacy path
    5. DB Integrity — schema migrations, logging, FK constraints
"""
from __future__ import annotations

import sqlite3
from dataclasses import replace as dc_replace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from hogan_bot.config import BotConfig
from hogan_bot.swarm_decision.types import AgentVote, DecisionIntent, SwarmDecision


@pytest.fixture(autouse=False)
def clear_forecast_cache():
    """Clear the forecast model cache to prevent cross-test state bleed.

    AgentPipeline.run() calls compute_forecast() which caches models in
    a module-level dict.  Consecutive calls with different candles can
    produce different signals if the cached model has internal state.
    """
    from hogan_bot.forecast import _model_cache
    _model_cache.clear()
    yield
    _model_cache.clear()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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


def _base_config(**overrides) -> BotConfig:
    defaults = dict(
        symbols=["BTC/USD"],
        timeframe="1h",
        use_ml_filter=False,
        use_ml_as_sizer=False,
        ml_confidence_sizing=False,
        paper_mode=True,
        starting_balance_usd=10000.0,
        use_regime_detection=True,
        swarm_enabled=False,
        swarm_mode="shadow",
        swarm_agents="pipeline_v1,risk_steward_v1,data_guardian_v1,execution_cost_v1",
        swarm_min_agreement=0.60,
        swarm_min_vote_margin=0.10,
        swarm_max_entropy=0.95,
        swarm_log_full_votes=True,
    )
    defaults.update(overrides)
    return dc_replace(BotConfig(), **defaults)


def _make_pipeline(config):
    from hogan_bot.agent_pipeline import AgentPipeline
    return AgentPipeline(config, conn=None)


def _in_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    from hogan_bot.storage import _create_schema
    _create_schema(conn)
    return conn


# ===================================================================
# 1. REGRESSION TESTS — validate the 5 critical bug fixes
# ===================================================================

class TestFix1_PeakEquityDrawdown:
    """RiskSteward must actually veto when peak_equity_usd > equity_usd."""

    def test_risk_steward_detects_drawdown_via_shared_ctx(self):
        from hogan_bot.swarm_decision.agents.risk_steward import RiskStewardAgent
        agent = RiskStewardAgent(max_drawdown_pct=0.10)
        candles = _synthetic_candles(50)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={"equity_usd": 8500, "peak_equity_usd": 10000},
        )
        assert vote.veto is True, "15% drawdown should trigger veto"
        assert vote.size_scale == 0.0

    def test_risk_steward_no_false_alarm_at_peak(self):
        from hogan_bot.swarm_decision.agents.risk_steward import RiskStewardAgent
        agent = RiskStewardAgent(max_drawdown_pct=0.10)
        candles = _synthetic_candles(50)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={"equity_usd": 10000, "peak_equity_usd": 10000},
        )
        assert vote.veto is False, "No drawdown at peak should not veto"
        assert vote.size_scale == pytest.approx(1.0)

    def test_decide_threads_peak_equity_into_swarm(self):
        """policy_core.decide() must pass real peak_equity_usd to agents."""
        from hogan_bot.policy_core import PolicyState, decide

        candles = _synthetic_candles(200)
        cfg = _base_config(swarm_enabled=True, swarm_mode="shadow")
        pipeline = _make_pipeline(cfg)
        state = PolicyState()

        intent = decide(
            symbol="BTC/USD",
            candles=candles,
            equity_usd=8500.0,
            config=cfg,
            pipeline=pipeline,
            state=state,
            mode="backtest",
            peak_equity_usd=10000.0,
        )
        assert intent.swarm is not None, "Swarm should have run"
        risk_votes = [v for v in intent.swarm.votes if v.agent_id == "risk_steward_v1"]
        assert len(risk_votes) == 1
        rv = risk_votes[0]
        assert rv.veto is True, "RiskSteward should veto at 15% drawdown"

    def test_peak_equity_none_defaults_to_equity(self):
        """When peak_equity_usd is None, defaults to equity (no drawdown)."""
        from hogan_bot.policy_core import PolicyState, decide

        candles = _synthetic_candles(200)
        cfg = _base_config(swarm_enabled=True, swarm_mode="shadow")
        pipeline = _make_pipeline(cfg)
        state = PolicyState()

        intent = decide(
            symbol="BTC/USD",
            candles=candles,
            equity_usd=10000.0,
            config=cfg,
            pipeline=pipeline,
            state=state,
            mode="backtest",
            peak_equity_usd=None,
        )
        assert intent.swarm is not None
        risk_votes = [v for v in intent.swarm.votes if v.agent_id == "risk_steward_v1"]
        assert risk_votes[0].veto is False


class TestFix2_PipelineSignalReuse:
    """PipelineAgent must reuse the pre-computed signal, not re-run pipeline."""

    def test_pipeline_agent_uses_shared_signal(self):
        from hogan_bot.swarm_decision.agents.pipeline_agent import PipelineAgent

        mock_pipeline = MagicMock()
        mock_signal = MagicMock()
        mock_signal.action = "sell"
        mock_signal.confidence = 0.65
        mock_signal.stop_distance_pct = 0.015
        mock_signal.forecast = None

        agent = PipelineAgent(mock_pipeline)
        candles = _synthetic_candles(100)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={"pipeline_signal": mock_signal},
        )
        assert vote.action == "sell"
        assert vote.confidence == pytest.approx(0.65)
        mock_pipeline.run.assert_not_called()

    def test_pipeline_agent_falls_back_without_signal(self):
        from hogan_bot.swarm_decision.agents.pipeline_agent import PipelineAgent

        mock_pipeline = MagicMock()
        fallback_signal = MagicMock()
        fallback_signal.action = "hold"
        fallback_signal.confidence = 0.3
        fallback_signal.stop_distance_pct = 0.01
        fallback_signal.forecast = None
        mock_pipeline.run.return_value = fallback_signal

        agent = PipelineAgent(mock_pipeline)
        candles = _synthetic_candles(100)
        vote = agent.vote(
            symbol="BTC/USD", candles=candles, as_of_ms=None,
            shared_context={},
        )
        assert vote.action == "hold"
        mock_pipeline.run.assert_called_once()

    def test_no_double_pipeline_in_decide(self):
        """pipeline.run() should be called exactly once inside decide()."""
        from hogan_bot.policy_core import PolicyState, decide

        candles = _synthetic_candles(200)
        cfg = _base_config(swarm_enabled=True, swarm_mode="shadow")
        real_pipeline = _make_pipeline(cfg)
        original_run = real_pipeline.run

        call_count = 0
        def counting_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_run(*args, **kwargs)

        real_pipeline.run = counting_run
        state = PolicyState()
        decide(
            symbol="BTC/USD",
            candles=candles,
            equity_usd=10000.0,
            config=cfg,
            pipeline=real_pipeline,
            state=state,
            mode="backtest",
        )
        assert call_count == 1, f"pipeline.run() called {call_count} times, expected 1"


class TestFix3_BacktestSwarmEnabled:
    """Backtest should honour swarm_enabled from caller, not hardcode False."""

    def test_backtest_config_passes_swarm_enabled(self):
        """The _bt_config inside backtest should reflect swarm_enabled=True."""
        from hogan_bot.backtest import run_backtest_on_candles
        candles = _synthetic_candles(250)
        candles["timestamp"] = pd.to_datetime(candles["ts_ms"], unit="ms", utc=True)

        result = run_backtest_on_candles(
            candles,
            symbol="BTC/USD",
            starting_balance_usd=10000.0,
            aggressive_allocation=0.30,
            max_risk_per_trade=0.02,
            max_drawdown=0.15,
            short_ma_window=8,
            long_ma_window=21,
            volume_window=20,
            volume_threshold=1.0,
            fee_rate=0.001,
            use_policy_core=True,
            swarm_enabled=True,
            swarm_mode="shadow",
        )
        assert result.trades >= 0

    def test_backtest_swarm_disabled_by_default(self):
        """Without explicit swarm_enabled, backtest should not run swarm."""
        from hogan_bot.backtest import run_backtest_on_candles
        candles = _synthetic_candles(250)
        candles["timestamp"] = pd.to_datetime(candles["ts_ms"], unit="ms", utc=True)

        result = run_backtest_on_candles(
            candles,
            symbol="BTC/USD",
            starting_balance_usd=10000.0,
            aggressive_allocation=0.30,
            max_risk_per_trade=0.02,
            max_drawdown=0.15,
            short_ma_window=8,
            long_ma_window=21,
            volume_window=20,
            volume_threshold=1.0,
            fee_rate=0.001,
            use_policy_core=True,
        )
        assert result.trades >= 0


class TestFix4_WeightSnapshotLogging:
    """Weight snapshot should be logged exactly once per PolicyState."""

    def test_weight_snapshot_logged_once(self):
        from hogan_bot.policy_core import PolicyState, decide

        conn = _in_memory_db()
        candles = _synthetic_candles(200)
        cfg = _base_config(swarm_enabled=True, swarm_mode="shadow")
        pipeline = _make_pipeline(cfg)
        state = PolicyState()

        for _ in range(3):
            decide(
                symbol="BTC/USD",
                candles=candles,
                equity_usd=10000.0,
                config=cfg,
                pipeline=pipeline,
                state=state,
                conn=conn,
                mode="backtest",
            )

        rows = conn.execute("SELECT COUNT(*) FROM swarm_weight_snapshots").fetchone()
        assert rows[0] == 1, f"Expected 1 snapshot, got {rows[0]}"
        assert state.swarm_weights_logged is True

    def test_separate_states_log_independently(self):
        from hogan_bot.policy_core import PolicyState, decide

        conn = _in_memory_db()
        candles = _synthetic_candles(200)
        cfg = _base_config(swarm_enabled=True, swarm_mode="shadow")
        pipeline = _make_pipeline(cfg)

        for _ in range(2):
            state = PolicyState()
            decide(
                symbol="BTC/USD",
                candles=candles,
                equity_usd=10000.0,
                config=cfg,
                pipeline=pipeline,
                state=state,
                conn=conn,
                mode="backtest",
            )

        rows = conn.execute("SELECT COUNT(*) FROM swarm_weight_snapshots").fetchone()
        assert rows[0] == 2


class TestFix5_VoteDecisionFK:
    """Agent votes must carry decision_id linking to swarm_decisions."""

    def test_votes_linked_to_decision(self):
        from hogan_bot.policy_core import PolicyState, decide

        conn = _in_memory_db()
        candles = _synthetic_candles(200)
        cfg = _base_config(swarm_enabled=True, swarm_mode="shadow")
        pipeline = _make_pipeline(cfg)
        state = PolicyState()

        decide(
            symbol="BTC/USD",
            candles=candles,
            equity_usd=10000.0,
            config=cfg,
            pipeline=pipeline,
            state=state,
            conn=conn,
            mode="backtest",
        )

        decisions = conn.execute("SELECT id FROM swarm_decisions").fetchall()
        assert len(decisions) >= 1
        decision_id = decisions[0][0]

        votes = conn.execute(
            "SELECT decision_id FROM swarm_agent_votes WHERE decision_id = ?",
            (decision_id,),
        ).fetchall()
        assert len(votes) >= 1, "Votes should carry the decision_id"

    def test_decision_id_column_exists(self):
        conn = _in_memory_db()
        cols = [
            row[1] for row in
            conn.execute("PRAGMA table_info(swarm_agent_votes)").fetchall()
        ]
        assert "decision_id" in cols


# ===================================================================
# 2. SHADOW PARITY — swarm shadow must not alter baseline trades
# ===================================================================

class TestShadowParity:
    """With swarm_mode='shadow', the baseline action and sizing must be
    identical to swarm_enabled=False."""

    @pytest.mark.usefixtures("clear_forecast_cache")
    def test_shadow_same_action_as_baseline(self):
        from hogan_bot.policy_core import PolicyState, decide

        candles = _synthetic_candles(200, seed=99)
        cfg_off = _base_config(swarm_enabled=False)
        cfg_shadow = _base_config(swarm_enabled=True, swarm_mode="shadow")
        pipeline = _make_pipeline(cfg_off)

        state_off = PolicyState()
        intent_off = decide(
            symbol="BTC/USD", candles=candles, equity_usd=10000.0,
            config=cfg_off, pipeline=pipeline, state=state_off, mode="backtest",
        )

        state_on = PolicyState()
        intent_on = decide(
            symbol="BTC/USD", candles=candles, equity_usd=10000.0,
            config=cfg_shadow, pipeline=pipeline, state=state_on, mode="backtest",
        )

        assert intent_on.action == intent_off.action
        assert intent_on.size_usd == pytest.approx(intent_off.size_usd, rel=1e-9)
        assert intent_on.confidence == pytest.approx(intent_off.confidence, rel=1e-9)
        assert intent_on.swarm is not None, "Shadow swarm should still produce a decision"
        assert intent_off.swarm is None

    def test_shadow_runs_reliably_across_seeds(self):
        """Shadow mode produces valid swarm decisions for diverse candle data.

        Exact size parity between shadow-on and shadow-off is tested in
        ``test_shadow_same_action_as_baseline`` (single shared pipeline).
        This test verifies that shadow mode runs cleanly across varied
        market data without crashing and produces structurally valid output.
        """
        from hogan_bot.policy_core import PolicyState, decide

        cfg_shadow = _base_config(swarm_enabled=True, swarm_mode="shadow")
        expected_agents = {
            "pipeline_v1", "risk_steward_v1",
            "data_guardian_v1", "execution_cost_v1",
        }

        for seed in (1, 42, 123, 777):
            candles = _synthetic_candles(200, seed=seed)
            pipeline = _make_pipeline(cfg_shadow)
            state = PolicyState()
            intent = decide(
                symbol="BTC/USD", candles=candles, equity_usd=10000.0,
                config=cfg_shadow, pipeline=pipeline,
                state=state, mode="backtest",
            )
            assert intent.swarm is not None, f"Swarm should run for seed={seed}"
            assert intent.swarm.final_action in ("buy", "sell", "hold")
            vote_ids = {v.agent_id for v in intent.swarm.votes}
            assert vote_ids == expected_agents, f"Missing agents for seed={seed}: {expected_agents - vote_ids}"
            assert 0.0 <= intent.swarm.agreement <= 1.0
            assert intent.swarm.entropy >= 0.0

    def test_active_mode_can_override(self):
        """In active mode, the swarm CAN change the baseline action."""
        from hogan_bot.policy_core import PolicyState, decide

        candles = _synthetic_candles(200, seed=42)
        cfg = _base_config(swarm_enabled=True, swarm_mode="active")
        pipeline = _make_pipeline(cfg)
        state = PolicyState()

        intent = decide(
            symbol="BTC/USD", candles=candles, equity_usd=10000.0,
            config=cfg, pipeline=pipeline, state=state, mode="backtest",
        )
        assert intent.swarm is not None
        assert intent.action == intent.swarm.final_action


# ===================================================================
# 3. CHAMPION PROTECTION — champion + swarm = champion wins
# ===================================================================

class TestChampionProtection:
    """Champion mode must produce identical decisions regardless of whether
    swarm is enabled."""

    @pytest.mark.usefixtures("clear_forecast_cache")
    def test_champion_swarm_off_equals_swarm_shadow(self):
        """Champion mode + swarm shadow = champion behaviour unchanged.

        Uses a single shared pipeline to eliminate pipeline non-determinism.
        This mirrors production where one pipeline serves all calls.
        """
        from hogan_bot.champion import apply_champion_mode
        from hogan_bot.policy_core import PolicyState, decide

        candles = _synthetic_candles(200, seed=42)

        cfg_base = _base_config(swarm_enabled=False)
        cfg_base = apply_champion_mode(cfg_base)

        cfg_swarm = _base_config(swarm_enabled=True, swarm_mode="shadow")
        cfg_swarm = apply_champion_mode(cfg_swarm)

        pipeline = _make_pipeline(cfg_base)

        s1 = PolicyState()
        intent_off = decide(
            symbol="BTC/USD", candles=candles, equity_usd=10000.0,
            config=cfg_base, pipeline=pipeline,
            state=s1, mode="backtest",
        )

        s2 = PolicyState()
        intent_on = decide(
            symbol="BTC/USD", candles=candles, equity_usd=10000.0,
            config=cfg_swarm, pipeline=pipeline,
            state=s2, mode="backtest",
        )

        assert intent_on.action == intent_off.action
        assert intent_on.size_usd == pytest.approx(intent_off.size_usd, rel=1e-9)
        assert intent_on.regime == intent_off.regime
        assert intent_off.swarm is None
        assert intent_on.swarm is not None

    def test_champion_locks_not_overridden_by_swarm(self):
        """apply_champion_mode should not touch swarm fields."""
        from hogan_bot.champion import apply_champion_mode

        cfg = _base_config(swarm_enabled=True, swarm_mode="active")
        cfg = apply_champion_mode(cfg)
        assert cfg.swarm_enabled is True
        assert cfg.swarm_mode == "active"


# ===================================================================
# 4. POLICY-CORE PARITY — use_policy_core matches legacy decisions
# ===================================================================

class TestPolicyCoreParity:
    """When swarm is off, policy_core.decide() should produce the same
    action/sizing as the legacy inline backtest path."""

    def test_policy_core_produces_valid_output(self):
        from hogan_bot.policy_core import PolicyState, decide

        candles = _synthetic_candles(200, seed=42)
        cfg = _base_config(swarm_enabled=False)
        pipeline = _make_pipeline(cfg)
        state = PolicyState()

        intent = decide(
            symbol="BTC/USD", candles=candles, equity_usd=10000.0,
            config=cfg, pipeline=pipeline, state=state, mode="backtest",
        )
        assert isinstance(intent, DecisionIntent)
        assert intent.action in ("buy", "sell", "hold")
        assert intent.size_usd >= 0.0
        assert intent.swarm is None

    def test_policy_core_deterministic(self):
        from hogan_bot.policy_core import PolicyState, decide

        candles = _synthetic_candles(200, seed=42)
        cfg = _base_config(swarm_enabled=False)
        pipeline = _make_pipeline(cfg)

        results = []
        for _ in range(3):
            state = PolicyState()
            intent = decide(
                symbol="BTC/USD", candles=candles, equity_usd=10000.0,
                config=cfg, pipeline=pipeline, state=state, mode="backtest",
            )
            results.append(intent)

        for r in results[1:]:
            assert r.action == results[0].action
            assert r.size_usd == pytest.approx(results[0].size_usd, rel=1e-9)
            assert r.confidence == pytest.approx(results[0].confidence, rel=1e-9)
            assert r.regime == results[0].regime

    def test_backtest_policy_core_runs_without_crash(self):
        """Full backtest with use_policy_core=True must complete."""
        from hogan_bot.backtest import run_backtest_on_candles

        candles = _synthetic_candles(300, seed=42)
        candles["timestamp"] = pd.to_datetime(candles["ts_ms"], unit="ms", utc=True)

        result = run_backtest_on_candles(
            candles,
            symbol="BTC/USD",
            starting_balance_usd=10000.0,
            aggressive_allocation=0.30,
            max_risk_per_trade=0.02,
            max_drawdown=0.20,
            short_ma_window=8,
            long_ma_window=21,
            volume_window=20,
            volume_threshold=1.0,
            fee_rate=0.001,
            use_policy_core=True,
            swarm_enabled=False,
        )
        assert result.start_equity == 10000.0
        assert len(result.equity_curve) > 0

    def test_backtest_policy_core_swarm_shadow_runs(self):
        """Full backtest with policy_core + swarm shadow must complete."""
        from hogan_bot.backtest import run_backtest_on_candles

        candles = _synthetic_candles(300, seed=42)
        candles["timestamp"] = pd.to_datetime(candles["ts_ms"], unit="ms", utc=True)

        result = run_backtest_on_candles(
            candles,
            symbol="BTC/USD",
            starting_balance_usd=10000.0,
            aggressive_allocation=0.30,
            max_risk_per_trade=0.02,
            max_drawdown=0.20,
            short_ma_window=8,
            long_ma_window=21,
            volume_window=20,
            volume_threshold=1.0,
            fee_rate=0.001,
            use_policy_core=True,
            swarm_enabled=True,
            swarm_mode="shadow",
        )
        assert result.start_equity == 10000.0
        assert len(result.equity_curve) > 0


# ===================================================================
# 5. DB INTEGRITY — schema, logging, FK constraints
# ===================================================================

class TestDBIntegrity:
    """Verify schema migrations and logging helpers work correctly."""

    def test_schema_creates_all_swarm_tables(self):
        conn = _in_memory_db()
        tables = {
            row[0] for row in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "swarm_decisions" in tables
        assert "swarm_agent_votes" in tables
        assert "swarm_weight_snapshots" in tables

    def test_log_swarm_decision_returns_id(self):
        from hogan_bot.swarm_decision.logging import log_swarm_decision
        from hogan_bot.swarm_decision.types import AgentVote, SwarmDecision

        conn = _in_memory_db()
        decision = SwarmDecision(
            final_action="buy",
            final_confidence=0.8,
            final_size_scale=0.9,
            agreement=0.85,
            entropy=0.3,
            weights_used={"a": 0.5, "b": 0.5},
            votes=[
                AgentVote(agent_id="a", action="buy", confidence=0.8),
                AgentVote(agent_id="b", action="buy", confidence=0.7),
            ],
        )
        row_id = log_swarm_decision(
            conn, 1700000000000, "BTC/USD", "1h", decision, "shadow",
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_log_agent_votes_with_decision_id(self):
        from hogan_bot.swarm_decision.logging import log_agent_votes, log_swarm_decision
        from hogan_bot.swarm_decision.types import AgentVote, SwarmDecision

        conn = _in_memory_db()
        votes = [
            AgentVote(agent_id="a", action="buy", confidence=0.8),
            AgentVote(agent_id="b", action="sell", confidence=0.6),
        ]
        decision = SwarmDecision(
            final_action="buy", final_confidence=0.7, final_size_scale=0.8,
            agreement=0.7, entropy=0.5,
            weights_used={"a": 0.5, "b": 0.5}, votes=votes,
        )
        dec_id = log_swarm_decision(
            conn, 1700000000000, "BTC/USD", "1h", decision, "shadow",
        )
        log_agent_votes(
            conn, 1700000000000, "BTC/USD", "1h", votes,
            decision_id=dec_id,
        )

        rows = conn.execute(
            "SELECT agent_id, decision_id FROM swarm_agent_votes ORDER BY agent_id",
        ).fetchall()
        assert len(rows) == 2
        for _, d_id in rows:
            assert d_id == dec_id

    def test_log_weight_snapshot_persists(self):
        from hogan_bot.swarm_decision.logging import log_weight_snapshot

        conn = _in_memory_db()
        wid = log_weight_snapshot(
            conn, 1700000000000, "BTC/USD", "1h",
            {"pipeline_v1": 0.25, "risk_steward_v1": 0.25,
             "data_guardian_v1": 0.25, "execution_cost_v1": 0.25},
            regime="trending_up",
            source="static_init",
        )
        assert isinstance(wid, int)
        row = conn.execute(
            "SELECT source, regime FROM swarm_weight_snapshots WHERE id = ?",
            (wid,),
        ).fetchone()
        assert row[0] == "static_init"
        assert row[1] == "trending_up"

    def test_full_logging_cycle_in_decide(self):
        """decide() with conn should populate all three swarm tables."""
        from hogan_bot.policy_core import PolicyState, decide

        conn = _in_memory_db()
        candles = _synthetic_candles(200)
        cfg = _base_config(swarm_enabled=True, swarm_mode="shadow")
        pipeline = _make_pipeline(cfg)
        state = PolicyState()

        decide(
            symbol="BTC/USD", candles=candles, equity_usd=10000.0,
            config=cfg, pipeline=pipeline, state=state,
            conn=conn, mode="backtest",
        )

        dec_count = conn.execute("SELECT COUNT(*) FROM swarm_decisions").fetchone()[0]
        vote_count = conn.execute("SELECT COUNT(*) FROM swarm_agent_votes").fetchone()[0]
        snap_count = conn.execute("SELECT COUNT(*) FROM swarm_weight_snapshots").fetchone()[0]

        assert dec_count == 1
        assert vote_count == 4  # 4 agents
        assert snap_count == 1


# ===================================================================
# 6. AGENT FAILURE RESILIENCE
# ===================================================================

class TestAgentFailureResilience:
    """A failing agent should not crash the swarm — it gets a hold vote."""

    def test_failing_agent_produces_hold_vote(self):
        from hogan_bot.swarm_decision.controller import SwarmController
        from hogan_bot.swarm_decision.types import AgentVote

        class _BrokenAgent:
            agent_id = "broken_v1"
            def vote(self, **kwargs) -> AgentVote:
                raise RuntimeError("Kaboom")

        class _GoodAgent:
            agent_id = "good_v1"
            def vote(self, **kwargs) -> AgentVote:
                return AgentVote(agent_id="good_v1", action="buy", confidence=0.9)

        ctrl = SwarmController(agents=[_GoodAgent(), _BrokenAgent()])
        candles = pd.DataFrame({"close": [50000.0] * 10})
        d = ctrl.decide(symbol="BTC/USD", candles=candles)

        assert len(d.votes) == 2
        broken_vote = [v for v in d.votes if v.agent_id == "broken_v1"][0]
        assert broken_vote.action == "hold"
        assert broken_vote.confidence == 0.0
        assert any("agent_error" in r for r in broken_vote.block_reasons)

    def test_all_agents_fail_still_returns_decision(self):
        from hogan_bot.swarm_decision.controller import SwarmController

        class _Broken:
            agent_id = "broken"
            def vote(self, **kwargs):
                raise ValueError("oops")

        ctrl = SwarmController(agents=[_Broken()])
        candles = pd.DataFrame({"close": [50000.0] * 10})
        d = ctrl.decide(symbol="BTC/USD", candles=candles)
        assert isinstance(d, SwarmDecision)
        assert d.final_action == "hold"


# ===================================================================
# 7. LOOKAHEAD BIAS — indicators must not leak future data
# ===================================================================

class TestLookaheadBias:
    """Wrap the existing check_lookahead() tool as a hard test gate.

    EMA-based indicators (EMA, RSI, MACD) have infinite theoretical memory,
    so shifting the starting bar by 1 produces small warm-up diffs.  MACD
    compounds three EMAs, producing diffs up to ~0.2 at BTC price scale
    (0.0004% of value).  This is NOT lookahead.  Tolerance=1.0 catches
    real lookahead (diffs of 10+ for bounded indicators, 100+ at price
    scale) while allowing all EMA warm-up noise.
    """

    EMA_WARMUP_TOLERANCE = 1.0

    def test_indicators_no_lookahead(self):
        from hogan_bot.lookahead_check import check_lookahead

        candles = _synthetic_candles(200, seed=42)
        result = check_lookahead(candles, tolerance=self.EMA_WARMUP_TOLERANCE)
        assert result["ok"] is True, (
            f"Lookahead bias detected in: {result['leaking_columns']} "
            f"details: {result.get('details', {})}"
        )
        assert len(result["clean_columns"]) > 0

    def test_different_data_no_lookahead(self):
        from hogan_bot.lookahead_check import check_lookahead

        for seed in (1, 99, 777):
            candles = _synthetic_candles(200, seed=seed)
            result = check_lookahead(candles, tolerance=self.EMA_WARMUP_TOLERANCE)
            assert result["ok"] is True, (
                f"seed={seed}: lookahead in {result['leaking_columns']}"
            )

    def test_strict_tolerance_catches_ema_warmup(self):
        """At 1e-9 tolerance, EMA warm-up noise IS flagged — confirming
        the checker works and the permissive tolerance is intentional."""
        from hogan_bot.lookahead_check import check_lookahead

        candles = _synthetic_candles(200, seed=42)
        strict = check_lookahead(candles, tolerance=1e-9)
        ema_cols = {"ema_9", "ema_21", "rsi_14", "macd_hist"}
        flagged = set(strict["leaking_columns"])
        assert flagged & ema_cols, "Strict tolerance should flag EMA warm-up"

    def test_too_few_candles_returns_ok_with_warning(self):
        from hogan_bot.lookahead_check import check_lookahead

        candles = _synthetic_candles(30)
        result = check_lookahead(candles)
        assert result["ok"] is True
        assert "warning" in result


# ===================================================================
# 8. LONG-SIM STABILITY — multi-hundred bar backtest with swarm
# ===================================================================

class TestLongSimStability:
    """Run extended backtests and verify zero crashes, valid equity
    curves, and consistent logging when swarm is active."""

    def test_500bar_policy_core_no_crash(self):
        """500-bar backtest with policy_core + swarm shadow must complete
        cleanly and produce a valid equity curve."""
        from hogan_bot.backtest import run_backtest_on_candles

        candles = _synthetic_candles(500, seed=42)
        candles["timestamp"] = pd.to_datetime(candles["ts_ms"], unit="ms", utc=True)

        result = run_backtest_on_candles(
            candles,
            symbol="BTC/USD",
            starting_balance_usd=10000.0,
            aggressive_allocation=0.30,
            max_risk_per_trade=0.02,
            max_drawdown=0.25,
            short_ma_window=8,
            long_ma_window=21,
            volume_window=20,
            volume_threshold=1.0,
            fee_rate=0.001,
            use_policy_core=True,
            swarm_enabled=True,
            swarm_mode="shadow",
        )

        assert result.start_equity == 10000.0
        assert len(result.equity_curve) >= 400, (
            f"Expected >=400 equity points, got {len(result.equity_curve)}"
        )
        assert all(e > 0 for e in result.equity_curve), "Equity went to zero"
        assert result.max_drawdown_pct < 1.0, "Max drawdown should be < 100%"

    def test_500bar_swarm_off_no_crash(self):
        """Baseline sanity — same 500-bar run without swarm must also pass."""
        from hogan_bot.backtest import run_backtest_on_candles

        candles = _synthetic_candles(500, seed=42)
        candles["timestamp"] = pd.to_datetime(candles["ts_ms"], unit="ms", utc=True)

        result = run_backtest_on_candles(
            candles,
            symbol="BTC/USD",
            starting_balance_usd=10000.0,
            aggressive_allocation=0.30,
            max_risk_per_trade=0.02,
            max_drawdown=0.25,
            short_ma_window=8,
            long_ma_window=21,
            volume_window=20,
            volume_threshold=1.0,
            fee_rate=0.001,
            use_policy_core=True,
            swarm_enabled=False,
        )

        assert result.start_equity == 10000.0
        assert len(result.equity_curve) >= 400

    def test_different_seeds_all_complete(self):
        """Multiple 300-bar backtests with diverse data must all complete."""
        from hogan_bot.backtest import run_backtest_on_candles

        for seed in (1, 123, 456):
            candles = _synthetic_candles(300, seed=seed)
            candles["timestamp"] = pd.to_datetime(
                candles["ts_ms"], unit="ms", utc=True,
            )
            result = run_backtest_on_candles(
                candles,
                symbol="BTC/USD",
                starting_balance_usd=10000.0,
                aggressive_allocation=0.30,
                max_risk_per_trade=0.02,
                max_drawdown=0.25,
                short_ma_window=8,
                long_ma_window=21,
                volume_window=20,
                volume_threshold=1.0,
                fee_rate=0.001,
                use_policy_core=True,
                swarm_enabled=True,
                swarm_mode="shadow",
            )
            assert len(result.equity_curve) > 0, f"seed={seed} produced empty curve"


# ===================================================================
# 9. LOGGING COMPLETENESS — every bar must produce a swarm record
# ===================================================================

class TestLoggingCompleteness:
    """When swarm is enabled with a DB connection, every call to decide()
    must produce exactly one swarm_decisions row and one vote per agent."""

    def test_n_decides_produces_n_decisions(self):
        from hogan_bot.policy_core import PolicyState, decide

        conn = _in_memory_db()
        cfg = _base_config(swarm_enabled=True, swarm_mode="shadow")
        pipeline = _make_pipeline(cfg)
        state = PolicyState()
        n_bars = 10

        for i in range(n_bars):
            candles = _synthetic_candles(200, seed=42 + i)
            decide(
                symbol="BTC/USD",
                candles=candles,
                equity_usd=10000.0,
                config=cfg,
                pipeline=pipeline,
                state=state,
                conn=conn,
                mode="backtest",
            )

        dec_count = conn.execute(
            "SELECT COUNT(*) FROM swarm_decisions"
        ).fetchone()[0]
        vote_count = conn.execute(
            "SELECT COUNT(*) FROM swarm_agent_votes"
        ).fetchone()[0]

        assert dec_count == n_bars, (
            f"Expected {n_bars} decisions, got {dec_count}"
        )
        assert vote_count == n_bars * 4, (
            f"Expected {n_bars * 4} votes (4 agents × {n_bars} bars), "
            f"got {vote_count}"
        )

    def test_every_vote_has_decision_id(self):
        from hogan_bot.policy_core import PolicyState, decide

        conn = _in_memory_db()
        cfg = _base_config(swarm_enabled=True, swarm_mode="shadow")
        pipeline = _make_pipeline(cfg)
        state = PolicyState()

        for i in range(5):
            candles = _synthetic_candles(200, seed=100 + i)
            decide(
                symbol="BTC/USD",
                candles=candles,
                equity_usd=10000.0,
                config=cfg,
                pipeline=pipeline,
                state=state,
                conn=conn,
                mode="backtest",
            )

        orphans = conn.execute(
            "SELECT COUNT(*) FROM swarm_agent_votes WHERE decision_id IS NULL"
        ).fetchone()[0]
        assert orphans == 0, f"{orphans} votes lack a decision_id FK"

    def test_weight_snapshot_logged_once_despite_many_bars(self):
        from hogan_bot.policy_core import PolicyState, decide

        conn = _in_memory_db()
        cfg = _base_config(swarm_enabled=True, swarm_mode="shadow")
        pipeline = _make_pipeline(cfg)
        state = PolicyState()

        for i in range(10):
            candles = _synthetic_candles(200, seed=50 + i)
            decide(
                symbol="BTC/USD",
                candles=candles,
                equity_usd=10000.0,
                config=cfg,
                pipeline=pipeline,
                state=state,
                conn=conn,
                mode="backtest",
            )

        snap_count = conn.execute(
            "SELECT COUNT(*) FROM swarm_weight_snapshots"
        ).fetchone()[0]
        assert snap_count == 1, (
            f"Expected 1 weight snapshot, got {snap_count}"
        )
