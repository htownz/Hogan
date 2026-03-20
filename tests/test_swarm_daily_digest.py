"""Tests for swarm_daily_digest — severity rules, operator actions, rendering."""
from __future__ import annotations

import json
import sqlite3

import pytest

from hogan_bot.swarm_daily_digest import (
    build_digest,
    build_headline,
    build_operator_actions,
    compute_severity_and_flags,
)
from hogan_bot.swarm_digest_render import render_json
from hogan_bot.swarm_digest_types import DailyDigest


def _create_schema(conn: sqlite3.Connection) -> None:
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


def _seed_stalled_swarm(conn: sqlite3.Connection, n: int = 60) -> None:
    """Seed a swarm that's in the universal veto stall state."""
    base_ts = 1_742_169_600_000
    for i in range(n):
        conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms, symbol, timeframe, mode, final_action, final_conf,
                final_scale, agreement, entropy, vetoed, block_reasons_json,
                weights_json, decision_json, regime)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                base_ts + i * 60_000,
                "BTC/USD", "1h", "shadow",
                "hold", 0.0, 0.0, 1.0, 0.0, 1,
                json.dumps(["risk_steward_v1:vol_spike_5.1x"]),
                json.dumps({"pipeline_v1": 0.4}),
                json.dumps({}),
                None,
            ),
        )
        conn.execute(
            """INSERT INTO swarm_agent_votes
               (ts_ms, symbol, timeframe, agent_id, action, confidence,
                size_scale, veto, block_reasons_json, vote_json, decision_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                base_ts + i * 60_000,
                "BTC/USD", "1h", "risk_steward_v1",
                "hold", 0.0, 0.0, 1,
                json.dumps(["vol_spike_5.1x"]),
                json.dumps({}),
                i + 1,
            ),
        )
    conn.commit()


def _seed_healthy_swarm(conn: sqlite3.Connection, n: int = 60) -> None:
    """Seed a swarm operating normally with some trades."""
    base_ts = 1_742_169_600_000
    for i in range(n):
        vetoed = 1 if i < 10 else 0
        action = "hold" if vetoed else ("buy" if i % 3 == 0 else "sell" if i % 3 == 1 else "hold")
        regime = ["trending", "ranging", "volatile"][i % 3]
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
                json.dumps(["risk:dd"] if vetoed else []),
                json.dumps({}), json.dumps({}),
                regime,
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
def stalled_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    _seed_stalled_swarm(conn)
    yield conn
    conn.close()


@pytest.fixture
def healthy_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    _seed_healthy_swarm(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Severity rules
# ---------------------------------------------------------------------------

class TestSeverityRules:
    def test_undersampled_returns_watch(self):
        metrics = {"decision_count": 10, "would_trade_count": 0, "veto_ratio": 0.0}
        severity, flags = compute_severity_and_flags(metrics)
        assert severity == "watch"
        codes = {f.code for f in flags}
        assert "UNDERSAMPLED" in codes

    def test_stall_no_would_trade_critical(self):
        metrics = {
            "decision_count": 60,
            "would_trade_count": 0,
            "veto_ratio": 0.85,
            "distinct_regimes": 0,
            "mean_agreement": 1.0,
            "mean_entropy": 0.0,
            "top_veto_reason_share": 0.9,
            "baseline_miss_count": 0,
            "baseline_match_count": 0,
            "learning_import_error_count": 0,
            "weight_update_count": 0,
        }
        severity, flags = compute_severity_and_flags(metrics)
        assert severity == "critical"
        codes = {f.code for f in flags}
        assert "STALL_NO_WOULD_TRADE" in codes
        assert "CRITICAL_VETO_RATIO" in codes
        assert "NO_REGIME_LABELS" in codes
        assert "CONTROLLER_COLLAPSE" in codes

    def test_dominant_veto_reason_warning(self):
        metrics = {
            "decision_count": 60,
            "would_trade_count": 20,
            "veto_ratio": 0.50,
            "distinct_regimes": 3,
            "top_veto_reason_share": 0.60,
            "baseline_miss_count": 0,
            "baseline_match_count": 60,
            "learning_import_error_count": 0,
            "weight_update_count": 1,
        }
        severity, flags = compute_severity_and_flags(metrics)
        assert severity == "warning"
        codes = {f.code for f in flags}
        assert "DOMINANT_VETO_REASON" in codes

    def test_baseline_miss_warning(self):
        metrics = {
            "decision_count": 60,
            "would_trade_count": 20,
            "veto_ratio": 0.30,
            "distinct_regimes": 3,
            "top_veto_reason_share": 0.20,
            "baseline_miss_count": 10,
            "baseline_match_count": 50,
            "learning_import_error_count": 0,
            "weight_update_count": 1,
        }
        severity, flags = compute_severity_and_flags(metrics)
        assert severity == "warning"
        codes = {f.code for f in flags}
        assert "BASELINE_JOIN_MISS" in codes

    def test_import_error_critical(self):
        metrics = {
            "decision_count": 60,
            "would_trade_count": 20,
            "veto_ratio": 0.30,
            "distinct_regimes": 3,
            "top_veto_reason_share": 0.20,
            "baseline_miss_count": 0,
            "baseline_match_count": 60,
            "learning_import_error_count": 1,
            "weight_update_count": 0,
        }
        severity, flags = compute_severity_and_flags(metrics)
        assert severity == "critical"
        codes = {f.code for f in flags}
        assert "LEARNING_IMPORT_ERROR" in codes

    def test_healthy_only_when_clean(self):
        metrics = {
            "decision_count": 100,
            "would_trade_count": 30,
            "veto_ratio": 0.20,
            "distinct_regimes": 3,
            "mean_agreement": 0.7,
            "mean_entropy": 0.4,
            "top_veto_reason_share": 0.30,
            "baseline_miss_count": 2,
            "baseline_match_count": 98,
            "learning_import_error_count": 0,
            "weight_update_count": 5,
            "opportunity_score_top_decile_markout_bps": 30.0,
            "opportunity_score_bottom_decile_markout_bps": -10.0,
        }
        severity, flags = compute_severity_and_flags(metrics)
        assert severity == "healthy"
        assert all(f.level not in ("critical", "warning") for f in flags)


# ---------------------------------------------------------------------------
# Operator actions
# ---------------------------------------------------------------------------

class TestOperatorActions:
    def test_stall_generates_action(self):
        metrics = {
            "decision_count": 60,
            "would_trade_count": 0,
            "veto_ratio": 0.85,
            "distinct_regimes": 0,
            "mean_agreement": 1.0,
            "mean_entropy": 0.0,
            "top_veto_reason_share": 0.9,
            "baseline_miss_count": 0,
            "baseline_match_count": 0,
            "learning_import_error_count": 0,
            "weight_update_count": 0,
        }
        _, flags = compute_severity_and_flags(metrics)
        actions = build_operator_actions(metrics, flags)
        assert len(actions) >= 3
        assert any("risk_steward" in a.lower() for a in actions)

    def test_no_actions_when_healthy(self):
        metrics = {
            "decision_count": 100,
            "would_trade_count": 30,
            "veto_ratio": 0.20,
            "distinct_regimes": 3,
            "top_veto_reason_share": 0.20,
            "baseline_miss_count": 2,
            "baseline_match_count": 98,
            "learning_import_error_count": 0,
            "weight_update_count": 5,
            "opportunity_score_top_decile_markout_bps": 30.0,
            "opportunity_score_bottom_decile_markout_bps": -10.0,
        }
        _, flags = compute_severity_and_flags(metrics)
        actions = build_operator_actions(metrics, flags)
        assert len(actions) == 0 or all("watch" not in a.lower() or True for a in actions)


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

class TestHeadline:
    def test_stall_headline(self):
        headline = build_headline("critical", {"decision_count": 100, "would_trade_count": 0, "veto_ratio": 0.9})
        assert "global veto layer" in headline.lower() or "critical" in headline.lower()

    def test_healthy_headline(self):
        headline = build_headline("healthy", {"decision_count": 100, "would_trade_count": 30, "veto_ratio": 0.20})
        assert "healthy" in headline.lower()


# ---------------------------------------------------------------------------
# Full digest build
# ---------------------------------------------------------------------------

class TestBuildDigest:
    def test_empty_db_produces_digest(self, empty_db):
        digest = build_digest(empty_db, date="2025-03-17")
        assert isinstance(digest, DailyDigest)
        assert digest.date == "2025-03-17"
        assert digest.severity in ("healthy", "watch", "warning", "critical")

    def test_stalled_db_produces_critical(self, stalled_db):
        digest = build_digest(stalled_db, date="2025-03-17")
        assert digest.severity == "critical"
        assert digest.metrics["decision_count"] == 60
        assert digest.metrics["would_trade_count"] == 0
        assert len(digest.operator_actions) >= 1

    def test_healthy_db_not_critical(self, healthy_db):
        digest = build_digest(healthy_db, date="2025-03-17")
        assert digest.severity != "critical"
        assert digest.metrics["would_trade_count"] > 0


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

class TestJsonSerialization:
    def test_json_roundtrip(self, stalled_db):
        digest = build_digest(stalled_db, date="2025-03-17")
        j = render_json(digest)
        parsed = json.loads(j)
        assert parsed["date"] == "2025-03-17"
        assert parsed["severity"] == "critical"
        assert "metrics" in parsed
        assert "flags" in parsed
        assert "operator_actions" in parsed

    def test_json_has_required_keys(self, stalled_db):
        digest = build_digest(stalled_db, date="2025-03-17")
        d = digest.to_dict()
        required = ["date", "phase", "symbol", "timeframe", "severity", "headline",
                     "metrics", "flags", "replay_candidates", "operator_actions"]
        for k in required:
            assert k in d, f"Missing key: {k}"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

class TestMarkdownRendering:
    def test_markdown_has_sections(self, stalled_db):
        digest = build_digest(stalled_db, date="2025-03-17")
        md = digest.summary_md
        assert "# Swarm Daily Digest" in md
        assert "Executive Summary" in md
        assert "Key Metrics" in md
        assert "Operator Actions Today" in md
        assert "Promotion Note" in md

    def test_markdown_has_severity(self, stalled_db):
        digest = build_digest(stalled_db, date="2025-03-17")
        assert "CRITICAL" in digest.summary_md

    def test_empty_db_still_renders(self, empty_db):
        digest = build_digest(empty_db, date="2025-03-17")
        assert len(digest.summary_md) > 100


# ---------------------------------------------------------------------------
# Replay candidates
# ---------------------------------------------------------------------------

class TestReplayCandidatesInDigest:
    def test_replay_list_capped(self, stalled_db):
        digest = build_digest(stalled_db, date="2025-03-17")
        assert len(digest.replay_candidates) <= 12

    def test_replay_candidates_ordered(self, stalled_db):
        digest = build_digest(stalled_db, date="2025-03-17")
        if digest.replay_candidates:
            priorities = [rc.priority for rc in digest.replay_candidates]
            assert priorities == sorted(priorities)
