"""Tests for the Swarm Weekly Review engine — severity, recommendations,
actions, rendering, and full build."""
from __future__ import annotations

import json
import sqlite3

import pytest

from hogan_bot.swarm_weekly_review_types import WeeklyFlag, WeeklyReview
from hogan_bot.swarm_weekly_review import (
    build_cursor_actions,
    build_headline,
    build_operator_actions,
    build_weekly_review,
    compute_recommendation,
    compute_severity_and_flags,
)
from hogan_bot.swarm_weekly_review_render import render_json, render_markdown


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
            action TEXT, swarm_decision_id INTEGER
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


def _seed_stalled(conn: sqlite3.Connection, n: int = 80) -> None:
    base_ts = 1_742_169_600_000
    for i in range(n):
        conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms,symbol,timeframe,mode,final_action,final_conf,final_scale,
                agreement,entropy,vetoed,block_reasons_json,weights_json,decision_json,regime)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (base_ts + i * 60_000, "BTC/USD", "1h", "shadow", "hold",
             0.0, 0.0, 1.0, 0.0, 1, json.dumps(["vol_spike"]),
             json.dumps({}), json.dumps({}), None),
        )
        conn.execute(
            """INSERT INTO swarm_agent_votes
               (ts_ms,symbol,timeframe,agent_id,action,confidence,size_scale,
                veto,block_reasons_json,vote_json,decision_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (base_ts + i * 60_000, "BTC/USD", "1h", "risk_steward_v1",
             "hold", 0.0, 0.0, 1, json.dumps(["vol_spike"]),
             json.dumps({}), i + 1),
        )
    conn.commit()


def _seed_healthy(conn: sqlite3.Connection, n: int = 100) -> None:
    base_ts = 1_742_169_600_000
    for i in range(n):
        vetoed = 1 if i < 20 else 0
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
             json.dumps([]), json.dumps({}), json.dumps({}), regime),
        )
        for agent in ["pipeline_v1", "risk_steward_v1"]:
            conn.execute(
                """INSERT INTO swarm_agent_votes
                   (ts_ms,symbol,timeframe,agent_id,action,confidence,size_scale,
                    veto,block_reasons_json,vote_json,decision_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (base_ts + i * 60_000, "BTC/USD", "1h", agent,
                 action, 0.7, 1.0, 0, json.dumps([]),
                 json.dumps({}), i + 1),
            )
        conn.execute(
            "INSERT INTO decision_log (ts_ms, symbol, action, swarm_decision_id) VALUES (?,?,?,?)",
            (base_ts + i * 60_000, "BTC/USD", action, i + 1),
        )
    conn.commit()


@pytest.fixture
def empty_db():
    conn = sqlite3.connect(":memory:")
    _schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def stalled_db():
    conn = sqlite3.connect(":memory:")
    _schema(conn)
    _seed_stalled(conn)
    yield conn
    conn.close()


@pytest.fixture
def healthy_db():
    conn = sqlite3.connect(":memory:")
    _schema(conn)
    _seed_healthy(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Severity & flag rules
# ---------------------------------------------------------------------------

class TestSeverityFlags:
    def test_stall_zero_would_trade(self):
        sev, flags = compute_severity_and_flags({
            "decision_count": 100, "would_trade_count": 0, "veto_ratio": 0.90,
            "distinct_regimes": 0, "baseline_match_count": 0, "baseline_miss_count": 100,
            "mean_agreement": 1.0, "mean_entropy": 0.0,
            "dominant_veto_agent": "risk_steward_v1", "dominant_veto_agent_share": 0.91,
        })
        assert sev == "critical"
        codes = {f.code for f in flags}
        assert "STALL_ZERO_WOULD_TRADE" in codes
        assert "CRITICAL_VETO_RATIO" in codes
        assert "REGIME_LOGGING_MISSING" in codes
        assert "PRE_VETO_CONSENSUS_MISSING" in codes

    def test_healthy(self):
        sev, flags = compute_severity_and_flags({
            "decision_count": 500, "would_trade_count": 200,
            "veto_ratio": 0.20, "distinct_regimes": 5,
            "baseline_match_count": 497, "baseline_miss_count": 3,
            "weight_update_count": 3,
            "opportunity_score_top_decile_markout_bps": 50,
            "opportunity_score_bottom_decile_markout_bps": -20,
        })
        assert sev in ("healthy", "watch")

    def test_undersampled(self):
        sev, flags = compute_severity_and_flags({"decision_count": 10})
        assert sev == "watch"
        assert any(f.code == "UNDERSAMPLED" for f in flags)

    def test_warning_veto_ratio(self):
        sev, flags = compute_severity_and_flags({
            "decision_count": 200, "would_trade_count": 50,
            "veto_ratio": 0.65, "distinct_regimes": 3,
        })
        assert sev == "warning"
        assert any(f.code == "HIGH_VETO_RATIO" for f in flags)

    def test_dominant_agent_warning(self):
        sev, flags = compute_severity_and_flags({
            "decision_count": 100, "would_trade_count": 30,
            "veto_ratio": 0.55, "distinct_regimes": 3,
            "dominant_veto_agent": "risk_steward_v1",
            "dominant_veto_agent_share": 0.75,
        })
        assert any(f.code == "DOMINANT_VETO_AGENT" for f in flags)

    def test_learning_import_error(self):
        sev, flags = compute_severity_and_flags({
            "decision_count": 100, "would_trade_count": 50,
            "veto_ratio": 0.20, "distinct_regimes": 3,
            "learning_import_error_count": 1,
        })
        assert sev == "critical"
        assert any(f.code == "LEARNING_PANEL_BROKEN" for f in flags)

    def test_baseline_join_failure(self):
        sev, flags = compute_severity_and_flags({
            "decision_count": 100, "would_trade_count": 50,
            "veto_ratio": 0.20, "distinct_regimes": 3,
            "baseline_match_count": 5, "baseline_miss_count": 95,
        })
        assert any(f.code == "BASELINE_JOIN_FAILURE" for f in flags)


# ---------------------------------------------------------------------------
# Recommendation routing
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_fix_instrumentation(self):
        flags = [WeeklyFlag(level="critical", code="BASELINE_JOIN_FAILURE", message="")]
        rec = compute_recommendation("critical", flags, {})
        assert rec == "fix_instrumentation"

    def test_tune_thresholds(self):
        flags = [WeeklyFlag(level="critical", code="STALL_ZERO_WOULD_TRADE", message="")]
        rec = compute_recommendation("critical", flags, {"decision_count": 100})
        assert rec == "tune_thresholds"

    def test_quarantine_agent(self):
        flags = [WeeklyFlag(level="critical", code="DOMINANT_VETO_AGENT", message="")]
        rec = compute_recommendation("critical", flags, {"dominant_veto_agent_share": 0.90})
        assert rec == "quarantine_agent"

    def test_hold(self):
        flags = [WeeklyFlag(level="warning", code="HIGH_VETO_RATIO", message="")]
        rec = compute_recommendation("warning", flags, {"decision_count": 200})
        assert rec == "hold"

    def test_promote(self):
        rec = compute_recommendation("healthy", [], {"decision_count": 500, "would_trade_count": 200})
        assert rec == "promote"

    def test_insufficient_data(self):
        rec = compute_recommendation("watch", [], {"decision_count": 10})
        assert rec == "insufficient_data"


# ---------------------------------------------------------------------------
# Action generation
# ---------------------------------------------------------------------------

class TestActionGeneration:
    def test_operator_actions_from_flags(self):
        flags = [
            WeeklyFlag(level="critical", code="STALL_ZERO_WOULD_TRADE", message="",
                       action="Audit risk_steward thresholds."),
        ]
        actions = build_operator_actions(flags, {"decision_count": 100, "would_trade_count": 0})
        assert any("thresholds" in a.lower() for a in actions)
        assert any("Replay" in a for a in actions)

    def test_cursor_actions_from_flags(self):
        flags = [
            WeeklyFlag(level="critical", code="LEARNING_PANEL_BROKEN", message=""),
            WeeklyFlag(level="critical", code="REGIME_LOGGING_MISSING", message=""),
        ]
        actions = build_cursor_actions(flags, {})
        assert any("import" in a.lower() for a in actions)
        assert any("regime" in a.lower() for a in actions)


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

class TestHeadline:
    def test_no_decisions(self):
        h = build_headline("watch", "insufficient_data", {"decision_count": 0})
        assert "No swarm" in h

    def test_tune(self):
        h = build_headline("critical", "tune_thresholds", {
            "decision_count": 100, "would_trade_count": 0, "veto_ratio": 0.85,
        })
        assert "suppressed" in h.lower() or "tuning" in h.lower()

    def test_promote(self):
        h = build_headline("healthy", "promote", {
            "decision_count": 500, "would_trade_count": 200, "veto_ratio": 0.15,
        })
        assert "promotion" in h.lower() or "Candidate" in h


# ---------------------------------------------------------------------------
# Full build
# ---------------------------------------------------------------------------

class TestBuildWeeklyReview:
    def test_empty_db(self, empty_db):
        review = build_weekly_review(empty_db, week_end="2025-03-17")
        assert isinstance(review, WeeklyReview)
        assert review.severity in ("watch", "healthy")
        assert review.recommendation == "insufficient_data"
        assert len(review.summary_md) > 0

    def test_stalled_db(self, stalled_db):
        review = build_weekly_review(stalled_db, week_end="2025-03-17")
        assert review.severity == "critical"
        codes = {f.code for f in review.flags}
        assert "STALL_ZERO_WOULD_TRADE" in codes
        assert review.recommendation in ("tune_thresholds", "fix_instrumentation", "quarantine_agent")
        assert len(review.operator_actions) > 0
        assert len(review.summary_md) > 0

    def test_healthy_db(self, healthy_db):
        review = build_weekly_review(healthy_db, week_end="2025-03-17")
        assert review.severity in ("watch", "healthy", "warning")
        assert len(review.agent_scores) > 0
        assert len(review.summary_md) > 0

    def test_stalled_dominant_agent(self, stalled_db):
        review = build_weekly_review(stalled_db, week_end="2025-03-17")
        codes = {f.code for f in review.flags}
        assert "DOMINANT_VETO_AGENT" in codes or review.metrics.get("dominant_veto_agent") == "risk_steward_v1"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_markdown_sections(self, stalled_db):
        review = build_weekly_review(stalled_db, week_end="2025-03-17")
        md = review.summary_md
        for section in [
            "Weekly Review", "Executive Summary", "Health and Readiness",
            "Veto Review", "Divergence Review", "Learning and Drift",
            "Week-over-Week", "Promotion Outlook",
        ]:
            assert section in md, f"Missing section: {section}"

    def test_json_roundtrip(self, stalled_db):
        review = build_weekly_review(stalled_db, week_end="2025-03-17")
        j = render_json(review)
        parsed = json.loads(j)
        assert parsed["severity"] == "critical"
        assert "flags" in parsed
        assert "metrics" in parsed
        assert "recommendation" in parsed

    def test_json_all_keys(self, healthy_db):
        review = build_weekly_review(healthy_db, week_end="2025-03-17")
        j = render_json(review)
        parsed = json.loads(j)
        for key in ["week_label", "phase", "severity", "headline", "recommendation",
                     "metrics", "flags", "agent_scores", "replay_candidates",
                     "operator_actions", "cursor_actions"]:
            assert key in parsed, f"Missing key in JSON: {key}"

    def test_markdown_render_function(self):
        md = render_markdown(
            week_label="2025-W12", phase="shadow", symbol="BTC/USD", timeframe="1h",
            severity="healthy", headline="Test run.", recommendation="hold",
            metrics={"decision_count": 100, "would_trade_count": 50, "veto_count": 20, "veto_ratio": 0.2, "distinct_regimes": 3},
            flags=[], agent_scores=[], replay_candidates=[],
            operator_actions=["Review replay candidates."],
            cursor_actions=["Add logging."],
        )
        assert "2025-W12" in md
        assert "HEALTHY" in md
        assert "Review replay" in md
        assert "Add logging" in md
