"""Tests for threshold_registry — bundle CRUD, activation, diff."""
from __future__ import annotations

import sqlite3

from hogan_bot.threshold_registry import (
    activate_bundle,
    diff_bundles,
    get_active_bundle,
    get_change_history,
    list_bundles,
    log_threshold_changes,
    save_bundle,
)
from hogan_bot.threshold_types import ThresholdBundle


def _make_conn():
    conn = sqlite3.connect(":memory:")
    from hogan_bot.storage import _create_schema
    _create_schema(conn)
    return conn


class TestBundleCRUD:
    def test_save_and_get(self):
        conn = _make_conn()
        b = ThresholdBundle(
            bundle_id="default", agent_id="risk_steward_v1", version=1,
            values={"vol_spike_multiple": 5.0}, active=True,
        )
        save_bundle(b, conn)
        loaded = get_active_bundle("risk_steward_v1", conn)
        assert loaded is not None
        assert loaded.bundle_id == "default"
        assert loaded.values["vol_spike_multiple"] == 5.0
        conn.close()

    def test_no_active_returns_none(self):
        conn = _make_conn()
        assert get_active_bundle("nonexistent", conn) is None
        conn.close()

    def test_list_bundles(self):
        conn = _make_conn()
        save_bundle(ThresholdBundle("default", "agent_a", 1, {"x": 1}, active=True), conn)
        save_bundle(ThresholdBundle("relaxed", "agent_a", 1, {"x": 2}), conn)
        bundles = list_bundles("agent_a", conn)
        assert len(bundles) == 2
        conn.close()


class TestActivation:
    def test_activate_swaps(self):
        conn = _make_conn()
        save_bundle(ThresholdBundle("default", "a1", 1, {"x": 1}, active=True), conn)
        save_bundle(ThresholdBundle("relaxed", "a1", 1, {"x": 2}), conn)
        activate_bundle("a1", "relaxed", 1, "op", "test", conn)
        active = get_active_bundle("a1", conn)
        assert active is not None
        assert active.bundle_id == "relaxed"

        old = list_bundles("a1", conn)
        active_count = sum(1 for b in old if b.active)
        assert active_count == 1
        conn.close()

    def test_activation_logs_change(self):
        conn = _make_conn()
        save_bundle(ThresholdBundle("default", "a1", 1, {"x": 1}, active=True), conn)
        save_bundle(ThresholdBundle("relaxed", "a1", 1, {"x": 2}), conn)
        activate_bundle("a1", "relaxed", 1, "ben", "tuning", conn)
        history = get_change_history("a1", conn)
        assert len(history) >= 1
        assert history[0]["operator"] == "ben"
        conn.close()


class TestDiff:
    def test_diff_detects_changes(self):
        old = ThresholdBundle("b", "a", 1, {"x": 1, "y": 2})
        new = ThresholdBundle("b", "a", 2, {"x": 3, "y": 2, "z": 4})
        changes = diff_bundles(old, new)
        fields = {c.field_name for c in changes}
        assert "x" in fields
        assert "z" in fields
        assert "y" not in fields

    def test_diff_empty_when_same(self):
        b = ThresholdBundle("b", "a", 1, {"x": 1})
        assert diff_bundles(b, b) == []

    def test_diff_detects_removal(self):
        old = ThresholdBundle("b", "a", 1, {"x": 1, "y": 2})
        new = ThresholdBundle("b", "a", 2, {"x": 1})
        changes = diff_bundles(old, new)
        fields = {c.field_name for c in changes}
        assert "y" in fields


class TestChangeHistory:
    def test_log_and_retrieve(self):
        conn = _make_conn()
        from hogan_bot.threshold_types import ThresholdChange
        changes = [
            ThresholdChange(ts="2025-01-01", agent_id="a1", bundle_id="b1",
                            field_name="x", old_value=1, new_value=2,
                            reason="tune", operator="op"),
        ]
        log_threshold_changes(changes, conn)
        history = get_change_history("a1", conn)
        assert len(history) == 1
        assert history[0]["field_name"] == "x"
        conn.close()
