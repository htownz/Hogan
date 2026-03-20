"""Tests for threshold_apply — bundle creation, activation, dry-run, ack gating."""
from __future__ import annotations

import sqlite3

from hogan_bot.threshold_apply import apply_threshold
from hogan_bot.threshold_registry import get_active_bundle, save_bundle
from hogan_bot.threshold_types import ThresholdBundle


def _make_conn():
    conn = sqlite3.connect(":memory:")
    from hogan_bot.storage import _create_schema
    _create_schema(conn)
    return conn


class TestDryRun:
    def test_dry_run_no_mutation(self):
        conn = _make_conn()
        result = apply_threshold(
            conn, "agent_a", "relaxed", {"x": 5.0}, "test", "op",
            dry_run=True, require_ack=False,
        )
        assert result["dry_run"] is True
        assert result["activated"] is False
        assert get_active_bundle("agent_a", conn) is None
        conn.close()


class TestAckGating:
    def test_blocked_without_ack(self):
        conn = _make_conn()
        result = apply_threshold(
            conn, "agent_a", "relaxed", {"x": 5.0}, "test", "op",
            require_ack=True, provided_ack=None,
        )
        assert result["activated"] is False
        assert "blocked" in result["message"].lower()
        conn.close()

    def test_succeeds_with_correct_ack(self):
        conn = _make_conn()
        result = apply_threshold(
            conn, "agent_a", "relaxed", {"x": 5.0}, "test", "op",
            require_ack=True, provided_ack="I_APPROVE_THRESHOLD_CHANGE",
        )
        assert result["activated"] is True
        active = get_active_bundle("agent_a", conn)
        assert active is not None
        assert active.bundle_id == "relaxed"
        conn.close()

    def test_succeeds_without_require_ack(self):
        conn = _make_conn()
        result = apply_threshold(
            conn, "agent_a", "relaxed", {"x": 5.0}, "test", "op",
            require_ack=False,
        )
        assert result["activated"] is True
        conn.close()


class TestApplyWithExistingBundle:
    def test_version_increments(self):
        conn = _make_conn()
        save_bundle(ThresholdBundle("default", "a1", 1, {"x": 1}, active=True), conn)
        result = apply_threshold(
            conn, "a1", "default", {"x": 2}, "update", "op",
            require_ack=False,
        )
        assert result["new_version"] == 2
        assert result["activated"] is True
        conn.close()

    def test_diff_logged(self):
        conn = _make_conn()
        save_bundle(ThresholdBundle("default", "a1", 1, {"x": 1, "y": 2}, active=True), conn)
        result = apply_threshold(
            conn, "a1", "default", {"x": 3, "y": 2}, "update", "op",
            require_ack=False,
        )
        assert result["changes_count"] >= 1
        changed_fields = {c["field"] for c in result["changes"]}
        assert "x" in changed_fields
        conn.close()


class TestApplyMetadata:
    def test_result_has_reason_and_operator(self):
        conn = _make_conn()
        result = apply_threshold(
            conn, "a1", "b1", {"x": 1}, "my reason", "ben",
            require_ack=False,
        )
        assert result["reason"] == "my reason"
        assert result["operator"] == "ben"
        conn.close()
