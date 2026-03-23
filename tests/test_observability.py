"""Tests for hogan_bot.observability module."""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from hogan_bot.observability import (
    aggregate_all_block_reasons,
    aggregate_block_reasons_by_regime,
    check_agent_mode_staleness,
    get_db_table_stats,
    observability_health_report,
    prune_old_rows,
    vacuum_db,
)


@pytest.fixture
def db():
    """In-memory DB with decision_log and swarm_agent_modes tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE decision_log (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER,
            symbol TEXT,
            regime TEXT,
            final_action TEXT,
            block_reasons_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE swarm_agent_modes (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER,
            agent_id TEXT,
            mode TEXT,
            reason TEXT,
            operator TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE swarm_decisions (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE swarm_agent_votes (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE swarm_weight_snapshots (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE swarm_stall_alerts (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE equity_snapshots (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER
        )
    """)
    conn.commit()
    yield conn
    conn.close()


def _seed_decisions(conn, n=10):
    now = int(time.time() * 1000)
    for i in range(n):
        reasons = []
        action = "hold"
        if i % 3 == 0:
            reasons = ["ml_filter"]
        elif i % 3 == 1:
            reasons = ["edge_gate", "macro_sitout"]
            action = "hold"
        else:
            action = "buy"

        regime = ["trending_up", "volatile", "ranging"][i % 3]
        conn.execute(
            "INSERT INTO decision_log (ts_ms, symbol, regime, final_action, block_reasons_json) VALUES (?, ?, ?, ?, ?)",
            (now - i * 60_000, "BTC/USD", regime, action, json.dumps(reasons) if reasons else None),
        )
    conn.commit()


def _seed_agent_modes(conn):
    now = int(time.time() * 1000)
    modes = [
        ("pipeline_v1", "active", "auto_promote", "system", now - 3600_000),
        ("sentiment_v1", "quarantined", "accuracy_below_threshold", "auto", now - 200 * 3600_000),
        ("risk_steward", "advisory_only", "manual_review", "operator", now - 3600_000),
    ]
    for agent_id, mode, reason, operator, ts in modes:
        conn.execute(
            "INSERT INTO swarm_agent_modes (ts_ms, agent_id, mode, reason, operator) VALUES (?, ?, ?, ?, ?)",
            (ts, agent_id, mode, reason, operator),
        )
    conn.commit()


class TestAggregateAllBlockReasons:
    def test_empty_db(self, db) -> None:
        r = aggregate_all_block_reasons(db)
        assert r["rows_scanned"] == 0
        assert r["counts"] == {}

    def test_counts_all_reasons(self, db) -> None:
        _seed_decisions(db)
        r = aggregate_all_block_reasons(db)
        assert r["rows_scanned"] > 0
        assert "ml_filter" in r["counts"]
        assert "edge_gate" in r["counts"]
        assert "macro_sitout" in r["counts"]

    def test_action_distribution(self, db) -> None:
        _seed_decisions(db)
        r = aggregate_all_block_reasons(db)
        assert "hold" in r["action_distribution"]

    def test_symbol_filter(self, db) -> None:
        _seed_decisions(db)
        r = aggregate_all_block_reasons(db, symbol="ETH/USD")
        assert r["rows_scanned"] == 0

    def test_since_ms_filter(self, db) -> None:
        _seed_decisions(db)
        future_ms = int((time.time() + 3600) * 1000)
        r = aggregate_all_block_reasons(db, since_ms=future_ms)
        assert r["rows_scanned"] == 0


class TestAggregateByRegime:
    def test_empty(self, db) -> None:
        r = aggregate_block_reasons_by_regime(db)
        assert r == {}

    def test_groups_by_regime(self, db) -> None:
        _seed_decisions(db)
        r = aggregate_block_reasons_by_regime(db)
        assert len(r) > 0
        for _regime, counts in r.items():
            assert isinstance(counts, dict)
            assert all(isinstance(v, int) for v in counts.values())


class TestAgentModeStaleness:
    def test_empty_db(self, db) -> None:
        r = check_agent_mode_staleness(db)
        assert r == []

    def test_detects_stale(self, db) -> None:
        _seed_agent_modes(db)
        r = check_agent_mode_staleness(db, stale_threshold_hours=168)
        stale = [a for a in r if a.is_stale]
        assert len(stale) == 1
        assert stale[0].agent_id == "sentiment_v1"
        assert stale[0].mode == "quarantined"

    def test_active_never_stale(self, db) -> None:
        _seed_agent_modes(db)
        r = check_agent_mode_staleness(db)
        active = [a for a in r if a.mode == "active"]
        assert all(not a.is_stale for a in active)


class TestDbTableStats:
    def test_returns_stats(self, db) -> None:
        _seed_decisions(db)
        stats = get_db_table_stats(db)
        names = [t.name for t in stats]
        assert "decision_log" in names
        dl = next(t for t in stats if t.name == "decision_log")
        assert dl.row_count == 10

    def test_empty_tables(self, db) -> None:
        stats = get_db_table_stats(db)
        for t in stats:
            assert t.row_count == 0


class TestPruneOldRows:
    def test_dry_run(self, db) -> None:
        _seed_decisions(db)
        deleted = prune_old_rows(db, retain_days=0, dry_run=True,
                                 tables={"decision_log": "ts_ms"})
        assert deleted["decision_log"] > 0
        count = db.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0]
        assert count == 10

    def test_execute(self, db) -> None:
        _seed_decisions(db, n=5)
        old_ms = int((time.time() - 200 * 86400) * 1000)
        db.execute("INSERT INTO decision_log (ts_ms, symbol, regime, final_action) VALUES (?, 'X', 'r', 'hold')", (old_ms,))
        db.commit()

        deleted = prune_old_rows(db, retain_days=90, dry_run=False,
                                 tables={"decision_log": "ts_ms"})
        assert deleted["decision_log"] == 1
        count = db.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0]
        assert count == 5


class TestVacuum:
    def test_vacuum_succeeds(self, db) -> None:
        assert vacuum_db(db) is True


class TestHealthReport:
    def test_empty_db(self, db) -> None:
        rpt = observability_health_report(db)
        assert "block_reasons" in rpt
        assert "agent_modes" in rpt
        assert "table_stats" in rpt
        assert "alerts" in rpt

    def test_with_data(self, db) -> None:
        _seed_decisions(db)
        _seed_agent_modes(db)
        rpt = observability_health_report(db)
        assert rpt["block_reasons"]["rows_scanned"] > 0
        assert rpt["stale_agent_count"] == 1
        assert any(a["code"] == "stale_agent_mode" for a in rpt["alerts"])

    def test_json_serializable(self, db) -> None:
        _seed_decisions(db)
        _seed_agent_modes(db)
        rpt = observability_health_report(db)
        s = json.dumps(rpt, default=str)
        assert isinstance(s, str)


class TestDbHygieneCli:
    def test_help(self) -> None:
        import subprocess
        result = subprocess.run(
            ["python", "scripts/db_hygiene.py", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "hygiene" in result.stdout.lower() or "prune" in result.stdout.lower()


class TestMetricsExist:
    def test_block_reasons_counter(self) -> None:
        from hogan_bot.metrics import BLOCK_REASONS, HOLD_NO_REASON
        assert BLOCK_REASONS is not None
        assert HOLD_NO_REASON is not None


class TestDocsExist:
    def test_observability_docs(self) -> None:
        import pathlib
        assert pathlib.Path("docs/EXECUTION_RUNBOOK.md").exists()
