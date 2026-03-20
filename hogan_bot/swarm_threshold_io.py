"""Notifier event helpers for threshold tuning and agent quarantine.

Emits structured events to the notifier when stall alerts fire,
agents get quarantined, or threshold bundles are activated.
"""
from __future__ import annotations

import logging

from hogan_bot.threshold_types import StallAlert

logger = logging.getLogger(__name__)


def emit_stall_alert(notifier, alerts: list[StallAlert]) -> None:
    if not alerts or notifier is None:
        return
    for a in alerts:
        try:
            notifier.notify("swarm_stall_alert", {
                "code": a.code,
                "severity": a.severity,
                "metric": a.metric_name,
                "actual": a.actual,
                "threshold": a.threshold,
                "notes": a.notes,
            })
        except Exception as exc:
            logger.warning("Notifier stall alert failed: %s", exc)


def emit_agent_quarantined(notifier, agent_id: str, mode: str, reason: str, operator: str) -> None:
    if notifier is None:
        return
    try:
        notifier.notify("swarm_agent_quarantined", {
            "agent_id": agent_id,
            "mode": mode,
            "reason": reason,
            "operator": operator,
        })
    except Exception as exc:
        logger.warning("Notifier quarantine event failed: %s", exc)


def emit_bundle_activated(
    notifier, agent_id: str, bundle_id: str, version: int,
    reason: str, operator: str,
) -> None:
    if notifier is None:
        return
    try:
        notifier.notify("swarm_threshold_bundle_activated", {
            "agent_id": agent_id,
            "bundle_id": bundle_id,
            "version": version,
            "reason": reason,
            "operator": operator,
        })
    except Exception as exc:
        logger.warning("Notifier bundle event failed: %s", exc)


def emit_dominant_veto_agent(notifier, agent_id: str, share: float, notes: str = "") -> None:
    if notifier is None:
        return
    try:
        notifier.notify("swarm_dominant_veto_agent", {
            "agent_id": agent_id,
            "share": round(share, 4),
            "notes": notes,
        })
    except Exception as exc:
        logger.warning("Notifier dominant veto event failed: %s", exc)
