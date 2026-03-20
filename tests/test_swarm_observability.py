"""Tests for swarm_observability, swarm_metrics, and swarm_replay modules.

Uses in-memory SQLite with synthetic data, following the certification
suite pattern.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

from hogan_bot.storage import _create_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _in_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    _create_schema(conn)
    return conn


def _seed_data(conn: sqlite3.Connection, n: int = 30) -> None:
    """Insert n synthetic shadow decisions, votes, baseline decisions, and some outcomes."""
    for i in range(n):
        ts_ms = 1700000000000 + i * 3600_000
        vetoed = 1 if i % 5 == 0 else 0
        action = "hold" if vetoed else ("buy" if i % 3 == 0 else "sell")
        regime = ["trending", "ranging", "volatile", "risk_off"][i % 4]
        decision_json = json.dumps({"regime": regime})
        block_reasons = json.dumps(["stale_data"] if vetoed else [])
        weights = json.dumps({"pipeline_v1": 0.4, "risk_steward_v1": 0.25,
                              "data_guardian_v1": 0.2, "execution_cost_v1": 0.15})

        cur = conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms, symbol, timeframe, as_of_ms, mode, final_action,
                final_conf, final_scale, agreement, entropy, vetoed,
                block_reasons_json, weights_json, decision_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts_ms, "BTC/USD", "1h", ts_ms, "shadow", action,
             0.55 + i * 0.01, 1.0, 0.65 + (i % 10) * 0.03, 0.3, vetoed,
             block_reasons, weights, decision_json),
        )
        dec_id = cur.lastrowid

        for agent in ["pipeline_v1", "risk_steward_v1", "data_guardian_v1", "execution_cost_v1"]:
            a_veto = 1 if (vetoed and agent == "data_guardian_v1") else 0
            a_reasons = json.dumps(["stale_data"] if a_veto else [])
            vote_json = json.dumps({"agent_id": agent, "action": action})
            conn.execute(
                """INSERT INTO swarm_agent_votes
                   (ts_ms, symbol, timeframe, as_of_ms, agent_id, action,
                    confidence, expected_edge_bps, size_scale, veto,
                    block_reasons_json, vote_json, decision_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts_ms, "BTC/USD", "1h", ts_ms, agent, action,
                 0.7, 5.0, 1.0, a_veto, a_reasons, vote_json, dec_id),
            )

        # Baseline decision_log entry
        realized_pnl = -0.5 if vetoed else 1.2
        conn.execute(
            """INSERT INTO decision_log
               (ts_ms, symbol, final_action, final_confidence, position_size, realized_pnl)
               VALUES (?,?,?,?,?,?)""",
            (ts_ms, "BTC/USD", action, 0.6, 100.0, realized_pnl),
        )

        # Outcome for every 3rd decision
        if i % 3 == 0:
            fwd = -20.0 if vetoed else 30.0
            conn.execute(
                """INSERT INTO swarm_outcomes
                   (decision_id, forward_60m_bps, mae_bps, mfe_bps,
                    was_trade_taken, was_veto_correct, outcome_label, updated_ms)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (dec_id, fwd, abs(fwd) * 0.5, abs(fwd) * 1.2,
                 0 if vetoed else 1,
                 1 if vetoed else 0,
                 "loss" if fwd < 0 else "win",
                 ts_ms + 3600_000),
            )

    conn.commit()


# ===================================================================
# swarm_observability tests
# ===================================================================

class TestSwarmObservability:
    def test_load_latest_decision_empty(self):
        from hogan_bot.swarm_observability import load_latest_swarm_decision
        conn = _in_memory_db()
        df = load_latest_swarm_decision(conn)
        assert df.empty
        conn.close()

    def test_load_latest_decision_seeded(self):
        from hogan_bot.swarm_observability import load_latest_swarm_decision
        conn = _in_memory_db()
        _seed_data(conn, n=10)
        df = load_latest_swarm_decision(conn, symbol="BTC/USD")
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "BTC/USD"
        conn.close()

    def test_load_swarm_votes_by_decision_id(self):
        from hogan_bot.swarm_observability import load_swarm_votes
        conn = _in_memory_db()
        _seed_data(conn, n=5)
        votes = load_swarm_votes(conn, decision_id=1)
        assert len(votes) == 4  # 4 agents per decision
        conn.close()

    def test_load_swarm_votes_bulk(self):
        from hogan_bot.swarm_observability import load_swarm_votes
        conn = _in_memory_db()
        _seed_data(conn, n=10)
        votes = load_swarm_votes(conn, symbol="BTC/USD", limit=100)
        assert len(votes) == 40  # 10 * 4 agents
        conn.close()

    def test_load_outcomes(self):
        from hogan_bot.swarm_observability import load_swarm_outcomes
        conn = _in_memory_db()
        _seed_data(conn, n=10)
        outcomes = load_swarm_outcomes(conn, symbol="BTC/USD")
        assert len(outcomes) > 0
        assert "forward_60m_bps" in outcomes.columns
        conn.close()

    def test_load_weight_history(self):
        from hogan_bot.swarm_observability import load_swarm_weight_history
        conn = _in_memory_db()
        # Insert a weight snapshot
        import time
        now_ms = int(time.time() * 1000)
        conn.execute(
            """INSERT INTO swarm_weight_snapshots
               (ts_ms, symbol, timeframe, regime, weights_json, source, notes)
               VALUES (?,?,?,?,?,?,?)""",
            (now_ms, "BTC/USD", "1h", "trending",
             json.dumps({"a": 0.5, "b": 0.5}), "static", "test"),
        )
        conn.commit()
        df = load_swarm_weight_history(conn, symbol="BTC/USD", days=1)
        assert len(df) == 1
        conn.close()

    def test_load_promotion_status_empty(self):
        from hogan_bot.swarm_observability import load_swarm_promotion_status
        conn = _in_memory_db()
        df = load_swarm_promotion_status(conn)
        assert df.empty
        conn.close()

    def test_load_decision_detail(self):
        from hogan_bot.swarm_observability import load_decision_detail
        conn = _in_memory_db()
        _seed_data(conn, n=5)
        detail = load_decision_detail(conn, decision_id=1)
        assert not detail["decision"].empty
        assert len(detail["votes"]) == 4
        conn.close()

    def test_load_veto_ledger(self):
        from hogan_bot.swarm_observability import load_veto_ledger
        conn = _in_memory_db()
        _seed_data(conn, n=10)
        df = load_veto_ledger(conn)
        assert len(df) > 0
        assert "reasons" in df.columns
        conn.close()

    def test_load_loss_clusters(self):
        from hogan_bot.swarm_observability import load_swarm_loss_clusters
        conn = _in_memory_db()
        _seed_data(conn, n=10)
        df = load_swarm_loss_clusters(conn, symbol="BTC/USD", days=36500)
        assert len(df) > 0
        conn.close()

    def test_schema_has_new_tables(self):
        conn = _in_memory_db()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "swarm_outcomes" in tables
        assert "swarm_promotion_reports" in tables
        conn.close()


# ===================================================================
# swarm_metrics tests
# ===================================================================

class TestSwarmMetrics:
    def test_compute_veto_precision_empty(self):
        from hogan_bot.swarm_metrics import compute_veto_precision
        result = compute_veto_precision(pd.DataFrame(), pd.DataFrame())
        assert result["total_vetoes"] == 0

    def test_compute_no_trade_rate(self):
        from hogan_bot.swarm_metrics import compute_no_trade_rate
        decisions = pd.DataFrame({
            "final_action": ["buy", "sell", "hold", "hold", "buy"],
        })
        result = compute_no_trade_rate(decisions)
        assert result["would_trade"] == 3
        assert result["would_hold"] == 2
        assert 0.39 < result["no_trade_rate"] < 0.41

    def test_compute_no_trade_rate_with_baseline(self):
        from hogan_bot.swarm_metrics import compute_no_trade_rate
        decisions = pd.DataFrame({"final_action": ["buy", "hold", "hold"]})
        baseline = pd.DataFrame({"final_action": ["buy", "sell", "buy"]})
        result = compute_no_trade_rate(decisions, baseline)
        assert result["baseline_would_trade"] == 3
        assert result["skipped_vs_baseline"] == 2

    def test_compute_trade_density(self):
        from hogan_bot.swarm_metrics import compute_trade_density
        decisions = pd.DataFrame({
            "ts_ms": [1700000000000 + i * 3600_000 for i in range(48)],
            "final_action": ["buy" if i % 2 == 0 else "hold" for i in range(48)],
        })
        density = compute_trade_density(decisions, bucket_hours=24)
        assert not density.empty
        assert "trades" in density.columns

    def test_compute_agent_leaderboard(self):
        from hogan_bot.swarm_metrics import compute_agent_leaderboard
        votes = pd.DataFrame({
            "agent_id": ["a", "a", "b", "b"],
            "action": ["buy", "sell", "buy", "hold"],
            "confidence": [0.8, 0.6, 0.7, 0.5],
            "veto": [0, 1, 0, 0],
        })
        lb = compute_agent_leaderboard(votes)
        assert len(lb) == 2
        assert "veto_rate" in lb.columns

    def test_compute_opportunity_monotonicity_empty(self):
        from hogan_bot.swarm_metrics import compute_opportunity_monotonicity
        result = compute_opportunity_monotonicity(pd.DataFrame())
        assert result["monotonic"] is False

    def test_compute_disagreement_stats(self):
        from hogan_bot.swarm_metrics import compute_disagreement_stats
        decisions = pd.DataFrame({
            "agreement": [0.8, 0.3, 0.9, 0.4, 0.7],
            "entropy": [0.2, 0.8, 0.1, 0.7, 0.3],
        })
        stats = compute_disagreement_stats(decisions)
        assert stats["count"] == 5
        assert 0 < stats["mean_agreement"] < 1
        assert stats["high_disagreement_pct"] > 0


# ===================================================================
# swarm_replay tests
# ===================================================================

class TestSwarmReplay:
    def test_render_decision_story_buy(self):
        from hogan_bot.swarm_replay import render_decision_story
        decision = {
            "symbol": "BTC/USD", "final_action": "buy", "final_conf": 0.75,
            "agreement": 0.85, "entropy": 0.2, "vetoed": 0, "mode": "shadow",
            "block_reasons_json": "[]", "decision_json": '{"regime": "trending"}',
        }
        story = render_decision_story(decision)
        assert "buy" in story.lower()
        assert "BTC/USD" in story
        assert "shadow" in story

    def test_render_decision_story_veto(self):
        from hogan_bot.swarm_replay import render_decision_story
        decision = {
            "symbol": "BTC/USD", "final_action": "hold", "final_conf": 0.3,
            "agreement": 0.4, "entropy": 0.8, "vetoed": 1, "mode": "shadow",
            "block_reasons_json": '["stale_data", "high_cost"]',
            "decision_json": '{"regime": "ranging"}',
        }
        story = render_decision_story(decision)
        assert "veto" in story.lower()
        assert "stale_data" in story

    def test_render_decision_story_with_votes(self):
        from hogan_bot.swarm_replay import render_decision_story
        decision = {
            "symbol": "BTC/USD", "final_action": "buy", "final_conf": 0.7,
            "agreement": 0.9, "entropy": 0.1, "vetoed": 0, "mode": "shadow",
            "block_reasons_json": "[]", "decision_json": "{}",
        }
        votes = pd.DataFrame({
            "agent_id": ["pipeline_v1", "risk_steward_v1"],
            "action": ["buy", "buy"],
            "confidence": [0.8, 0.7],
            "veto": [0, 0],
            "block_reasons_json": ["[]", "[]"],
        })
        story = render_decision_story(decision, votes=votes)
        assert "pipeline_v1" in story
        assert "risk_steward_v1" in story

    def test_render_decision_story_with_baseline(self):
        from hogan_bot.swarm_replay import render_decision_story
        decision = {
            "symbol": "BTC/USD", "final_action": "buy", "final_conf": 0.7,
            "agreement": 0.9, "entropy": 0.1, "vetoed": 0, "mode": "shadow",
            "block_reasons_json": "[]", "decision_json": "{}",
        }
        baseline = {"final_action": "sell", "final_confidence": 0.6}
        story = render_decision_story(decision, baseline=baseline)
        assert "divergence" in story.lower()

    def test_build_replay_frame(self):
        from hogan_bot.swarm_replay import build_replay_frame
        decision = {
            "ts_ms": 1700000000000, "symbol": "BTC/USD",
            "final_action": "buy", "final_conf": 0.7,
            "agreement": 0.9, "entropy": 0.1, "vetoed": 0, "mode": "shadow",
            "block_reasons_json": "[]", "decision_json": "{}",
        }
        frame = build_replay_frame(decision)
        assert "decision" in frame
        assert "story" in frame
        assert isinstance(frame["story"], str)
        assert frame["outcome"] is None
        assert frame["outcome_summary"] == "Outcome not yet recorded."

    def test_build_replay_frame_with_outcome(self):
        from hogan_bot.swarm_replay import build_replay_frame
        decision = {
            "ts_ms": 1700000000000, "symbol": "BTC/USD",
            "final_action": "buy", "final_conf": 0.7,
            "agreement": 0.9, "entropy": 0.1, "vetoed": 0, "mode": "shadow",
            "block_reasons_json": "[]", "decision_json": "{}",
        }
        outcome = {"forward_60m_bps": 25.0, "mae_bps": 10.0, "mfe_bps": 35.0,
                    "outcome_label": "win", "was_veto_correct": None}
        frame = build_replay_frame(decision, outcome=outcome)
        assert "win" in frame["outcome_summary"]
        assert "25.0" in frame["outcome_summary"]

    def test_compute_baseline_vs_swarm_delta(self):
        from hogan_bot.swarm_replay import compute_baseline_vs_swarm_delta
        decisions = pd.DataFrame({
            "ts_ms": [100, 200, 300],
            "symbol": ["BTC/USD", "BTC/USD", "BTC/USD"],
            "final_action": ["buy", "sell", "hold"],
        })
        baseline = pd.DataFrame({
            "ts_ms": [100, 200, 300],
            "symbol": ["BTC/USD", "BTC/USD", "BTC/USD"],
            "final_action": ["buy", "buy", "hold"],
        })
        delta = compute_baseline_vs_swarm_delta(decisions, baseline)
        assert delta["compared"] == 3
        assert delta["match_count"] == 2
        assert delta["mismatch_count"] == 1

    def test_compute_baseline_vs_swarm_delta_empty(self):
        from hogan_bot.swarm_replay import compute_baseline_vs_swarm_delta
        delta = compute_baseline_vs_swarm_delta(pd.DataFrame(), pd.DataFrame())
        assert delta["compared"] == 0
