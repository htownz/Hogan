"""Tests for stall_detection module."""
from __future__ import annotations

import sqlite3

import pytest

from hogan_bot.stall_detection import (
    evaluate_stall_state,
    persist_stall_alerts,
    get_latest_stall_alerts,
    compute_stall_summary,
)
from hogan_bot.threshold_types import StallAlert


class TestEvaluateStallState:
    def test_critical_stall_zero_trades(self):
        alerts = evaluate_stall_state({
            "decision_count": 100, "would_trade_count": 0,
            "veto_ratio": 0.85, "top_veto_agent_share": 0.0,
            "distinct_regimes": 3,
        })
        codes = {a.code for a in alerts}
        assert "CRITICAL_STALL" in codes

    def test_severe_stall_low_ratio(self):
        alerts = evaluate_stall_state({
            "decision_count": 200, "would_trade_count": 5,
            "veto_ratio": 0.50, "top_veto_agent_share": 0.0,
            "distinct_regimes": 3,
        })
        codes = {a.code for a in alerts}
        assert "SEVERE_STALL" in codes

    def test_over_veto_warning(self):
        alerts = evaluate_stall_state({
            "decision_count": 100, "would_trade_count": 20,
            "veto_ratio": 0.75, "top_veto_agent_share": 0.0,
            "distinct_regimes": 3,
        })
        codes = {a.code for a in alerts}
        assert "OVER_VETO_WARNING" in codes

    def test_dominant_veto_agent(self):
        alerts = evaluate_stall_state({
            "decision_count": 100, "would_trade_count": 20,
            "veto_ratio": 0.50, "top_veto_agent_share": 0.65,
            "dominant_veto_agent": "risk_steward_v1",
            "distinct_regimes": 3,
        })
        codes = {a.code for a in alerts}
        assert "DOMINANT_VETO_AGENT" in codes

    def test_regime_blindness(self):
        alerts = evaluate_stall_state({
            "decision_count": 100, "would_trade_count": 20,
            "veto_ratio": 0.30, "top_veto_agent_share": 0.0,
            "distinct_regimes": 0,
        })
        codes = {a.code for a in alerts}
        assert "REGIME_BLINDNESS" in codes

    def test_baseline_join_failure(self):
        alerts = evaluate_stall_state({
            "decision_count": 100, "would_trade_count": 20,
            "veto_ratio": 0.30, "top_veto_agent_share": 0.0,
            "distinct_regimes": 3, "baseline_join_match_ratio": 0.50,
        })
        codes = {a.code for a in alerts}
        assert "BASELINE_JOIN_FAILURE" in codes

    def test_healthy_no_alerts(self):
        alerts = evaluate_stall_state({
            "decision_count": 200, "would_trade_count": 80,
            "veto_ratio": 0.30, "top_veto_agent_share": 0.20,
            "distinct_regimes": 4,
        })
        assert len(alerts) == 0

    def test_undersampled_no_critical(self):
        alerts = evaluate_stall_state({
            "decision_count": 10, "would_trade_count": 0,
            "veto_ratio": 0.0, "top_veto_agent_share": 0.0,
            "distinct_regimes": 0,
        })
        codes = {a.code for a in alerts}
        assert "CRITICAL_STALL" not in codes


class TestPersistAndRetrieve:
    def test_persist_and_get(self):
        conn = sqlite3.connect(":memory:")
        from hogan_bot.storage import _create_schema
        _create_schema(conn)

        alerts = [
            StallAlert(code="CRITICAL_STALL", severity="critical",
                       metric_name="would_trade_count", actual=0, threshold=1),
        ]
        persist_stall_alerts(alerts, conn)
        retrieved = get_latest_stall_alerts(conn)
        assert len(retrieved) == 1
        assert retrieved[0]["code"] == "CRITICAL_STALL"
        conn.close()

    def test_stall_summary_critical(self):
        conn = sqlite3.connect(":memory:")
        from hogan_bot.storage import _create_schema
        _create_schema(conn)

        alerts = [
            StallAlert(code="CRITICAL_STALL", severity="critical",
                       metric_name="would_trade_count", actual=0, threshold=1),
        ]
        persist_stall_alerts(alerts, conn)
        assert compute_stall_summary(conn) == "critical"
        conn.close()

    def test_stall_summary_healthy(self):
        conn = sqlite3.connect(":memory:")
        from hogan_bot.storage import _create_schema
        _create_schema(conn)
        assert compute_stall_summary(conn) == "healthy"
        conn.close()
