"""Tests for swarm_weekly_review_queries — query contract validation."""
from __future__ import annotations

import json
import sqlite3

import pytest

from hogan_bot.swarm_weekly_review_queries import (
    fetch_review_window,
    fetch_week_over_week_stats,
    fetch_weekly_agent_scores,
    fetch_weekly_regime_stats,
    fetch_weekly_replay_candidates,
    fetch_weekly_swarm_counts,
    fetch_weekly_veto_stats,
)


def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS swarm_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            as_of_ms INTEGER, mode TEXT NOT NULL, final_action TEXT NOT NULL,
            final_conf REAL NOT NULL, final_scale REAL NOT NULL,
            agreement REAL NOT NULL, entropy REAL NOT NULL,
            vetoed INTEGER NOT NULL, block_reasons_json TEXT NOT NULL,
            weights_json TEXT NOT NULL, decision_json TEXT NOT NULL, regime TEXT
        );
        CREATE TABLE IF NOT EXISTS swarm_agent_votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            as_of_ms INTEGER, agent_id TEXT NOT NULL, action TEXT NOT NULL,
            confidence REAL NOT NULL, expected_edge_bps REAL, size_scale REAL NOT NULL,
            veto INTEGER NOT NULL, block_reasons_json TEXT NOT NULL,
            vote_json TEXT NOT NULL, decision_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS swarm_outcomes (
            decision_id INTEGER PRIMARY KEY, forward_5m_bps REAL,
            forward_15m_bps REAL, forward_30m_bps REAL, forward_60m_bps REAL,
            mae_bps REAL, mfe_bps REAL, was_trade_taken INTEGER,
            was_veto_correct INTEGER, was_skip_correct INTEGER, outcome_label TEXT
        );
        CREATE TABLE IF NOT EXISTS decision_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, symbol TEXT,
            final_action TEXT, swarm_decision_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS swarm_weight_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL,
            symbol TEXT, timeframe TEXT, weights_json TEXT
        );
        CREATE TABLE IF NOT EXISTS swarm_promotion_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_ms INTEGER,
            symbol TEXT, timeframe TEXT, phase TEXT, recommendation TEXT,
            summary TEXT, gates_json TEXT, blockers_json TEXT
        );
    """)


def _seed(conn: sqlite3.Connection, n: int = 80, veto_count: int = 60) -> None:
    base_ts = 1_742_169_600_000
    for i in range(n):
        vetoed = 1 if i < veto_count else 0
        action = "hold" if vetoed else ("buy" if i % 2 == 0 else "sell")
        regime = ["trending", "ranging", "volatile"][i % 3]
        conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms,symbol,timeframe,mode,final_action,final_conf,final_scale,
                agreement,entropy,vetoed,block_reasons_json,weights_json,decision_json,regime)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (base_ts + i * 60_000, "BTC/USD", "1h", "shadow", action,
             0.0 if vetoed else 0.7, 0.0 if vetoed else 0.8,
             1.0 if vetoed else 0.65, 0.0 if vetoed else 0.5, vetoed,
             json.dumps(["risk_steward_v1:vol_spike_5.1x"] if vetoed else []),
             json.dumps({}), json.dumps({}), regime),
        )
        for agent in ["pipeline_v1", "risk_steward_v1"]:
            is_veto = (agent == "risk_steward_v1" and vetoed)
            conn.execute(
                """INSERT INTO swarm_agent_votes
                   (ts_ms,symbol,timeframe,agent_id,action,confidence,size_scale,
                    veto,block_reasons_json,vote_json,decision_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (base_ts + i * 60_000, "BTC/USD", "1h", agent,
                 "hold" if is_veto else action, 0.0 if is_veto else 0.7,
                 0.0 if is_veto else 1.0, 1 if is_veto else 0,
                 json.dumps(["vol_spike_5.1x"] if is_veto else []),
                 json.dumps({}), i + 1),
            )
    conn.commit()


@pytest.fixture
def empty_db():
    conn = sqlite3.connect(":memory:")
    _schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def seeded_db():
    conn = sqlite3.connect(":memory:")
    _schema(conn)
    _seed(conn)
    yield conn
    conn.close()


class TestFetchReviewWindow:
    def test_empty(self, empty_db):
        r = fetch_review_window(empty_db, week_end="2025-03-17")
        assert r["decision_count"] == 0
        assert "W" in r["week_label"]

    def test_seeded(self, seeded_db):
        r = fetch_review_window(seeded_db, week_end="2025-03-17")
        assert r["decision_count"] == 80


class TestFetchWeeklyCounts:
    def test_empty(self, empty_db):
        r = fetch_weekly_swarm_counts(empty_db, 0, 9_999_999_999_999)
        assert r["decision_count"] == 0

    def test_seeded(self, seeded_db):
        r = fetch_weekly_swarm_counts(seeded_db, 0, 9_999_999_999_999)
        assert r["decision_count"] == 80
        assert r["veto_count"] == 60
        assert r["would_trade_count"] == 20
        assert r["distinct_regimes"] == 3

    def test_symbol_filter(self, seeded_db):
        r = fetch_weekly_swarm_counts(seeded_db, 0, 9_999_999_999_999, symbol="ETH/USD")
        assert r["decision_count"] == 0


class TestFetchVetoStats:
    def test_seeded(self, seeded_db):
        r = fetch_weekly_veto_stats(seeded_db, 0, 9_999_999_999_999)
        assert r["veto_ratio"] > 0.5
        assert r["dominant_veto_agent"] == "risk_steward_v1"
        assert r["dominant_veto_agent_share"] > 0.9


class TestFetchAgentScores:
    def test_empty(self, empty_db):
        r = fetch_weekly_agent_scores(empty_db, 0, 9_999_999_999_999)
        assert r == []

    def test_seeded(self, seeded_db):
        r = fetch_weekly_agent_scores(seeded_db, 0, 9_999_999_999_999)
        assert len(r) == 2
        ids = {a["agent_id"] for a in r}
        assert "pipeline_v1" in ids
        assert "risk_steward_v1" in ids


class TestFetchRegimeStats:
    def test_seeded(self, seeded_db):
        r = fetch_weekly_regime_stats(seeded_db, 0, 9_999_999_999_999)
        assert r["distinct_regimes"] == 3
        assert len(r["regime_counts"]) >= 3


class TestWowStats:
    def test_no_prior(self, seeded_db):
        r = fetch_week_over_week_stats(
            seeded_db, 0, 9_999_999_999_999,
            9_999_999_999_999, 19_999_999_999_999,
        )
        assert r["prior_week_available"] is False

    def test_same_window(self, seeded_db):
        r = fetch_week_over_week_stats(
            seeded_db, 0, 9_999_999_999_999, 0, 9_999_999_999_999,
        )
        assert r["decision_count_wow_delta"] == 0


class TestReplayCandidates:
    def test_empty(self, empty_db):
        r = fetch_weekly_replay_candidates(empty_db, 0, 9_999_999_999_999)
        assert r == []

    def test_seeded_capped(self, seeded_db):
        r = fetch_weekly_replay_candidates(seeded_db, 0, 9_999_999_999_999, limit=5)
        assert len(r) <= 5
        assert all("category" in c for c in r)
        assert r[0]["priority"] == 1
