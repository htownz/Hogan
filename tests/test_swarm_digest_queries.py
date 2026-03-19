"""Tests for swarm_digest_queries — query contract validation."""
from __future__ import annotations

import json
import sqlite3

import pytest

from hogan_bot.swarm_digest_queries import (
    fetch_digest_window,
    fetch_swarm_counts,
    fetch_opportunity_stats,
    fetch_veto_stats,
    fetch_agent_vote_stats,
    fetch_divergence_stats,
    fetch_learning_drift_stats,
    fetch_replay_candidates,
)


def _create_schema(conn: sqlite3.Connection) -> None:
    """Minimal schema for digest queries."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS swarm_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            as_of_ms INTEGER,
            mode TEXT NOT NULL,
            final_action TEXT NOT NULL,
            final_conf REAL NOT NULL,
            final_scale REAL NOT NULL,
            agreement REAL NOT NULL,
            entropy REAL NOT NULL,
            vetoed INTEGER NOT NULL,
            block_reasons_json TEXT NOT NULL,
            weights_json TEXT NOT NULL,
            decision_json TEXT NOT NULL,
            regime TEXT
        );
        CREATE TABLE IF NOT EXISTS swarm_agent_votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            as_of_ms INTEGER,
            agent_id TEXT NOT NULL,
            action TEXT NOT NULL,
            confidence REAL NOT NULL,
            expected_edge_bps REAL,
            size_scale REAL NOT NULL,
            veto INTEGER NOT NULL,
            block_reasons_json TEXT NOT NULL,
            vote_json TEXT NOT NULL,
            decision_id INTEGER REFERENCES swarm_decisions(id)
        );
        CREATE TABLE IF NOT EXISTS swarm_outcomes (
            decision_id INTEGER PRIMARY KEY REFERENCES swarm_decisions(id),
            forward_5m_bps REAL,
            forward_15m_bps REAL,
            forward_30m_bps REAL,
            forward_60m_bps REAL,
            mae_bps REAL,
            mfe_bps REAL,
            was_trade_taken INTEGER,
            was_veto_correct INTEGER,
            was_skip_correct INTEGER,
            outcome_label TEXT
        );
        CREATE TABLE IF NOT EXISTS decision_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER,
            symbol TEXT,
            final_action TEXT,
            swarm_decision_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS swarm_weight_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            symbol TEXT,
            timeframe TEXT,
            weights_json TEXT
        );
    """)


def _seed_decisions(conn: sqlite3.Connection, n: int = 60, veto_count: int = 50) -> None:
    """Insert N synthetic decisions, veto_count of which are vetoed."""
    base_ts = 1_742_169_600_000  # 2025-03-17T00:00:00Z
    for i in range(n):
        vetoed = 1 if i < veto_count else 0
        action = "hold" if vetoed else ("buy" if i % 2 == 0 else "sell")
        regime = "trending" if i % 3 == 0 else ("ranging" if i % 3 == 1 else "volatile")
        conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms, symbol, timeframe, mode, final_action, final_conf,
                final_scale, agreement, entropy, vetoed, block_reasons_json,
                weights_json, decision_json, regime)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                base_ts + i * 60_000,
                "BTC/USD", "1h", "shadow",
                action, 0.0 if vetoed else 0.7,
                0.0 if vetoed else 0.8,
                1.0 if vetoed else 0.65,
                0.0 if vetoed else 0.5,
                vetoed,
                json.dumps(["risk_steward_v1:vol_spike_5.1x"] if vetoed else []),
                json.dumps({"pipeline_v1": 0.4}),
                json.dumps({"test": True}),
                regime,
            ),
        )

        for agent in ["pipeline_v1", "risk_steward_v1", "data_guardian_v1"]:
            is_veto = (agent == "risk_steward_v1" and vetoed)
            conn.execute(
                """INSERT INTO swarm_agent_votes
                   (ts_ms, symbol, timeframe, agent_id, action, confidence,
                    size_scale, veto, block_reasons_json, vote_json, decision_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    base_ts + i * 60_000,
                    "BTC/USD", "1h", agent,
                    "hold" if is_veto else action,
                    0.0 if is_veto else 0.7,
                    0.0 if is_veto else 1.0,
                    1 if is_veto else 0,
                    json.dumps(["vol_spike_5.1x"] if is_veto else []),
                    json.dumps({}),
                    i + 1,
                ),
            )
    conn.commit()


@pytest.fixture
def empty_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def seeded_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    _seed_decisions(conn)
    yield conn
    conn.close()


class TestFetchDigestWindow:
    def test_empty_db(self, empty_db):
        result = fetch_digest_window(empty_db, date="2025-03-17")
        assert result["decision_count"] == 0
        assert result["date"] == "2025-03-17"

    def test_seeded_db(self, seeded_db):
        result = fetch_digest_window(seeded_db, date="2025-03-17")
        assert result["decision_count"] == 60


class TestFetchSwarmCounts:
    def test_empty_db(self, empty_db):
        result = fetch_swarm_counts(empty_db, 0, 9_999_999_999_999)
        assert result["decision_count"] == 0
        assert result["veto_count"] == 0

    def test_seeded_db(self, seeded_db):
        result = fetch_swarm_counts(seeded_db, 0, 9_999_999_999_999)
        assert result["decision_count"] == 60
        assert result["veto_count"] == 50
        assert result["would_trade_count"] == 10
        assert result["distinct_regimes"] == 3

    def test_symbol_filter(self, seeded_db):
        result = fetch_swarm_counts(seeded_db, 0, 9_999_999_999_999, symbol="ETH/USD")
        assert result["decision_count"] == 0


class TestFetchVetoStats:
    def test_empty_db(self, empty_db):
        result = fetch_veto_stats(empty_db, 0, 9_999_999_999_999)
        assert result["veto_ratio"] == 0.0
        assert result["top_veto_reason"] is None

    def test_seeded_db(self, seeded_db):
        result = fetch_veto_stats(seeded_db, 0, 9_999_999_999_999)
        assert result["veto_ratio"] > 0.5
        assert result["top_veto_reason"] is not None
        assert result["top_veto_reason_share"] > 0.0


class TestFetchAgentVoteStats:
    def test_empty_db(self, empty_db):
        result = fetch_agent_vote_stats(empty_db, 0, 9_999_999_999_999)
        assert result["agents"] == []

    def test_seeded_db(self, seeded_db):
        result = fetch_agent_vote_stats(seeded_db, 0, 9_999_999_999_999)
        assert len(result["agents"]) == 3
        agent_ids = {a["agent_id"] for a in result["agents"]}
        assert "pipeline_v1" in agent_ids
        assert "risk_steward_v1" in agent_ids


class TestFetchReplayCandidates:
    def test_empty_db(self, empty_db):
        result = fetch_replay_candidates(empty_db, 0, 9_999_999_999_999)
        assert result == []

    def test_seeded_db_capped(self, seeded_db):
        result = fetch_replay_candidates(seeded_db, 0, 9_999_999_999_999, limit=5)
        assert len(result) <= 5
        assert all("decision_id" in r for r in result)
        assert result[0]["priority"] == 1


class TestFetchOpportunityStats:
    def test_empty_db(self, empty_db):
        result = fetch_opportunity_stats(empty_db, 0, 9_999_999_999_999)
        assert result["opportunity_score_mean"] is None
