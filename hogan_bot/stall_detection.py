"""Stall detection — evaluate whether the swarm is effectively not trading.

Produces a list of StallAlert objects that can be persisted, surfaced
in the dashboard, and consumed by the weekly/daily review.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from hogan_bot.threshold_types import StallAlert


def evaluate_stall_state(
    metrics: dict[str, Any],
    *,
    stall_zero_trade_min: int = 50,
    stall_low_trade_min: int = 100,
    stall_low_trade_ratio: float = 0.05,
    over_veto_ratio_warn: float = 0.70,
    single_agent_veto_share_warn: float = 0.60,
    regime_min: int = 2,
) -> list[StallAlert]:
    alerts: list[StallAlert] = []

    dec = metrics.get("decision_count", 0)
    wt = metrics.get("would_trade_count", 0)
    veto_ratio = metrics.get("veto_ratio", 0.0)
    top_agent_share = metrics.get("top_veto_agent_share", 0.0)
    top_agent = metrics.get("dominant_veto_agent", "")
    regimes = metrics.get("distinct_regimes", 0)
    bl_match_ratio = metrics.get("baseline_join_match_ratio")

    wt_ratio = wt / dec if dec > 0 else 0.0

    if dec >= stall_zero_trade_min and wt == 0:
        alerts.append(StallAlert(
            code="CRITICAL_STALL", severity="critical",
            metric_name="would_trade_count", actual=wt,
            threshold=1,
            notes=f"Swarm is active ({dec} decisions) but zero would-trades.",
        ))

    if dec >= stall_low_trade_min and 0 < wt_ratio < stall_low_trade_ratio:
        alerts.append(StallAlert(
            code="SEVERE_STALL", severity="critical",
            metric_name="would_trade_ratio", actual=round(wt_ratio, 4),
            threshold=stall_low_trade_ratio,
            notes=f"Only {wt} would-trades from {dec} decisions ({wt_ratio:.1%}).",
        ))

    if veto_ratio > over_veto_ratio_warn:
        alerts.append(StallAlert(
            code="OVER_VETO_WARNING", severity="warn",
            metric_name="veto_ratio", actual=round(veto_ratio, 4),
            threshold=over_veto_ratio_warn,
            notes=f"Veto ratio {veto_ratio:.1%} exceeds {over_veto_ratio_warn:.0%} threshold.",
        ))

    if top_agent_share > single_agent_veto_share_warn:
        alerts.append(StallAlert(
            code="DOMINANT_VETO_AGENT", severity="warn",
            metric_name="top_veto_agent_share", actual=round(top_agent_share, 4),
            threshold=single_agent_veto_share_warn,
            notes=f"{top_agent or 'unknown'} accounts for {top_agent_share:.0%} of all vetoes.",
        ))

    if dec >= stall_zero_trade_min and regimes < regime_min:
        alerts.append(StallAlert(
            code="REGIME_BLINDNESS", severity="warn",
            metric_name="distinct_regimes", actual=regimes, threshold=regime_min,
            notes=f"Only {regimes} distinct regimes after {dec} decisions.",
        ))

    if bl_match_ratio is not None and bl_match_ratio < 0.90 and dec >= stall_zero_trade_min:
        alerts.append(StallAlert(
            code="BASELINE_JOIN_FAILURE", severity="warn",
            metric_name="baseline_join_match_ratio",
            actual=round(bl_match_ratio, 4), threshold=0.90,
            notes=f"Baseline match ratio {bl_match_ratio:.1%} — joins are unreliable.",
        ))

    return alerts


def persist_stall_alerts(
    alerts: list[StallAlert], conn: sqlite3.Connection,
) -> None:
    ts_ms = int(time.time() * 1000)
    rows = [
        (ts_ms, a.code, a.severity, a.metric_name, float(a.actual), float(a.threshold), a.notes)
        for a in alerts
    ]
    conn.executemany(
        """INSERT INTO swarm_stall_alerts (ts_ms, code, severity, metric_name, actual, threshold, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def get_latest_stall_alerts(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT ts_ms, code, severity, metric_name, actual, threshold, notes FROM swarm_stall_alerts ORDER BY ts_ms DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return []
    return [
        {"ts_ms": r[0], "code": r[1], "severity": r[2], "metric_name": r[3],
         "actual": r[4], "threshold": r[5], "notes": r[6]}
        for r in rows
    ]


def compute_stall_summary(conn: sqlite3.Connection, window_ms: int | None = None) -> str:
    """Return a one-line stall status string for dashboard badges."""
    alerts = get_latest_stall_alerts(conn, limit=10)
    if not alerts:
        return "healthy"
    severities = [a["severity"] for a in alerts]
    if "critical" in severities:
        return "critical"
    if "warn" in severities:
        return "warning"
    return "info"
