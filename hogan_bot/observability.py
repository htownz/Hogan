"""Observability utilities — decision transparency, agent mode hygiene, DB health.

Answers "why didn't we trade?" across ALL gate layers (not just swarm), detects
stale agent modes, and provides DB table stats for retention planning.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision transparency — full block-reason histogram
# ---------------------------------------------------------------------------


def aggregate_all_block_reasons(
    conn: sqlite3.Connection,
    *,
    since_ms: int | None = None,
    symbol: str | None = None,
    limit: int = 50_000,
) -> dict[str, Any]:
    """Count ALL block reasons from ``decision_log.block_reasons_json``.

    Unlike ``aggregate_swarm_policy_block_reasons`` (which only counts
    ``swarm_*`` tags), this counts every reason: ML, edge, quality,
    ranging, pullback, macro_sitout, freshness, etc.

    Returns::

        {
            "counts": {"ml_filter": 42, "edge_gate": 30, ...},
            "rows_scanned": 5000,
            "rows_with_blocks": 1200,
            "total_block_events": 2400,
            "hold_with_no_reason": 150,
            "action_distribution": {"hold": 3000, "buy": 1500, "sell": 500},
        }
    """
    where_parts = ["block_reasons_json IS NOT NULL"]
    params: list[Any] = []
    if since_ms is not None:
        where_parts.append("ts_ms >= ?")
        params.append(since_ms)
    if symbol:
        where_parts.append("symbol = ?")
        params.append(symbol)

    where = " AND ".join(where_parts)
    sql = f"""
        SELECT block_reasons_json, final_action
        FROM decision_log
        WHERE {where}
        ORDER BY ts_ms DESC
        LIMIT ?
    """
    params.append(limit)

    counts: Counter[str] = Counter()
    rows_scanned = 0
    rows_with_blocks = 0
    hold_no_reason = 0
    action_dist: Counter[str] = Counter()

    try:
        for row in conn.execute(sql, params):
            rows_scanned += 1
            raw, action = row
            action_dist[action or "unknown"] += 1

            reasons = _parse_reasons(raw)
            if reasons:
                rows_with_blocks += 1
                for r in reasons:
                    counts[r] += 1
            elif (action or "").lower() == "hold":
                hold_no_reason += 1
    except Exception as exc:
        logger.warning("aggregate_all_block_reasons failed: %s", exc)

    return {
        "counts": dict(counts.most_common()),
        "rows_scanned": rows_scanned,
        "rows_with_blocks": rows_with_blocks,
        "total_block_events": int(sum(counts.values())),
        "hold_with_no_reason": hold_no_reason,
        "action_distribution": dict(action_dist.most_common()),
    }


def aggregate_block_reasons_by_regime(
    conn: sqlite3.Connection,
    *,
    since_ms: int | None = None,
    limit: int = 50_000,
) -> dict[str, dict[str, int]]:
    """Block reason counts grouped by regime.

    Returns ``{"trending_up": {"ml_filter": 10, ...}, ...}``.
    """
    where_parts = ["block_reasons_json IS NOT NULL"]
    params: list[Any] = []
    if since_ms is not None:
        where_parts.append("ts_ms >= ?")
        params.append(since_ms)

    where = " AND ".join(where_parts)
    sql = f"""
        SELECT regime, block_reasons_json
        FROM decision_log
        WHERE {where}
        ORDER BY ts_ms DESC
        LIMIT ?
    """
    params.append(limit)

    by_regime: dict[str, Counter[str]] = {}
    try:
        for row in conn.execute(sql, params):
            regime, raw = row
            regime = regime or "unknown"
            reasons = _parse_reasons(raw)
            if not reasons:
                continue
            if regime not in by_regime:
                by_regime[regime] = Counter()
            for r in reasons:
                by_regime[regime][r] += 1
    except Exception as exc:
        logger.warning("aggregate_block_reasons_by_regime failed: %s", exc)

    return {regime: dict(c.most_common()) for regime, c in sorted(by_regime.items())}


def _parse_reasons(raw: str | None) -> list[str]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s == "null":
        return []
    try:
        arr = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(arr, list):
        return []
    return [x for x in arr if isinstance(x, str) and x.strip()]


# ---------------------------------------------------------------------------
# Agent mode hygiene — staleness detection
# ---------------------------------------------------------------------------


@dataclass
class AgentModeStatus:
    agent_id: str
    mode: str
    reason: str
    operator: str
    ts_ms: int
    age_hours: float
    is_stale: bool


def check_agent_mode_staleness(
    conn: sqlite3.Connection,
    *,
    stale_threshold_hours: float = 168.0,  # 7 days
) -> list[AgentModeStatus]:
    """Check all agent modes and flag those that haven't been updated recently.

    An agent stuck in ``advisory_only`` or ``quarantined`` for >threshold
    hours may be silently degrading system behavior.
    """
    sql = """
        SELECT agent_id, mode, reason, operator, ts_ms
        FROM swarm_agent_modes
        WHERE id IN (SELECT MAX(id) FROM swarm_agent_modes GROUP BY agent_id)
        ORDER BY agent_id
    """
    now_ms = int(time.time() * 1000)
    results: list[AgentModeStatus] = []

    try:
        for row in conn.execute(sql):
            agent_id, mode, reason, operator, ts_ms = row
            age_hours = (now_ms - (ts_ms or 0)) / 3_600_000
            is_stale = (
                mode in ("advisory_only", "quarantined", "no_veto")
                and age_hours > stale_threshold_hours
            )
            results.append(AgentModeStatus(
                agent_id=agent_id or "unknown",
                mode=mode or "active",
                reason=reason or "",
                operator=operator or "",
                ts_ms=ts_ms or 0,
                age_hours=round(age_hours, 1),
                is_stale=is_stale,
            ))
    except Exception as exc:
        logger.warning("check_agent_mode_staleness failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# DB hygiene — table stats, prune, vacuum
# ---------------------------------------------------------------------------


@dataclass
class TableStats:
    name: str
    row_count: int
    oldest_ts_ms: int | None = None
    newest_ts_ms: int | None = None


def get_db_table_stats(conn: sqlite3.Connection) -> list[TableStats]:
    """Report row counts and age range for key tables."""
    tables_with_ts = {
        "decision_log": "ts_ms",
        "swarm_decisions": "ts_ms",
        "swarm_agent_votes": "ts_ms",
        "swarm_weight_snapshots": "ts_ms",
        "swarm_agent_modes": "ts_ms",
        "swarm_stall_alerts": "ts_ms",
        "swarm_outcomes": "updated_ms",
        "paper_trades": "open_ts_ms",
        "equity_snapshots": "ts_ms",
        "candles": "ts_ms",
        "orders": "ts_ms",
        "fills": "ts_ms",
    }

    results: list[TableStats] = []
    for table, ts_col in tables_with_ts.items():
        try:
            row = conn.execute(f"SELECT COUNT(*), MIN({ts_col}), MAX({ts_col}) FROM {table}").fetchone()
            if row:
                results.append(TableStats(
                    name=table,
                    row_count=row[0] or 0,
                    oldest_ts_ms=row[1],
                    newest_ts_ms=row[2],
                ))
        except Exception as exc:
            logger.debug("get_db_table_stats: %s query failed: %s", table, exc)

    return results


def prune_old_rows(
    conn: sqlite3.Connection,
    *,
    retain_days: int = 90,
    tables: dict[str, str] | None = None,
    dry_run: bool = True,
) -> dict[str, int]:
    """Delete rows older than ``retain_days`` from high-volume tables.

    Returns ``{table: rows_deleted}`` (or rows that would be deleted
    if ``dry_run=True``).
    """
    if tables is None:
        tables = {
            "decision_log": "ts_ms",
            "swarm_decisions": "ts_ms",
            "swarm_agent_votes": "ts_ms",
            "swarm_weight_snapshots": "ts_ms",
            "swarm_stall_alerts": "ts_ms",
        }

    cutoff_ms = int((time.time() - retain_days * 86400) * 1000)
    deleted: dict[str, int] = {}

    for table, ts_col in tables.items():
        try:
            count_row = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {ts_col} < ?", (cutoff_ms,)
            ).fetchone()
            n = count_row[0] if count_row else 0

            if dry_run:
                deleted[table] = n
            else:
                conn.execute(f"DELETE FROM {table} WHERE {ts_col} < ?", (cutoff_ms,))
                conn.commit()
                deleted[table] = n
                if n > 0:
                    logger.info("Pruned %d rows from %s (older than %d days)", n, table, retain_days)
        except Exception as exc:
            logger.warning("prune_old_rows failed for %s: %s", table, exc)
            deleted[table] = -1

    return deleted


def vacuum_db(conn: sqlite3.Connection) -> bool:
    """Run VACUUM to reclaim space after pruning.  Returns True on success."""
    try:
        conn.execute("VACUUM")
        return True
    except Exception as exc:
        logger.warning("VACUUM failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Combined health report
# ---------------------------------------------------------------------------


def observability_health_report(
    conn: sqlite3.Connection,
    *,
    since_hours: float = 24.0,
    symbol: str | None = None,
) -> dict[str, Any]:
    """One-call observability health check for dashboards and CI."""
    since_ms = int((time.time() - since_hours * 3600) * 1000)

    block_reasons = aggregate_all_block_reasons(conn, since_ms=since_ms, symbol=symbol)
    block_by_regime = aggregate_block_reasons_by_regime(conn, since_ms=since_ms)
    agent_modes = check_agent_mode_staleness(conn)
    table_stats = get_db_table_stats(conn)

    stale_agents = [a for a in agent_modes if a.is_stale]

    return {
        "period_hours": since_hours,
        "block_reasons": block_reasons,
        "block_reasons_by_regime": block_by_regime,
        "agent_modes": [
            {
                "agent_id": a.agent_id,
                "mode": a.mode,
                "age_hours": a.age_hours,
                "is_stale": a.is_stale,
            }
            for a in agent_modes
        ],
        "stale_agent_count": len(stale_agents),
        "stale_agent_ids": [a.agent_id for a in stale_agents],
        "table_stats": [
            {"table": t.name, "rows": t.row_count}
            for t in table_stats
        ],
        "alerts": _compute_observability_alerts(block_reasons, stale_agents, table_stats),
    }


def _compute_observability_alerts(
    block_reasons: dict,
    stale_agents: list[AgentModeStatus],
    table_stats: list[TableStats],
) -> list[dict]:
    alerts: list[dict] = []

    if block_reasons.get("hold_with_no_reason", 0) > 50:
        alerts.append({
            "level": "warning",
            "code": "unexplained_holds",
            "message": f"{block_reasons['hold_with_no_reason']} holds with no block reason — possible logging gap",
        })

    for agent in stale_agents:
        alerts.append({
            "level": "warning",
            "code": "stale_agent_mode",
            "message": f"Agent '{agent.agent_id}' stuck in '{agent.mode}' for {agent.age_hours:.0f}h — consider reset",
        })

    for t in table_stats:
        if t.row_count > 500_000:
            alerts.append({
                "level": "info",
                "code": "large_table",
                "message": f"Table '{t.name}' has {t.row_count:,} rows — consider pruning",
            })

    return alerts
