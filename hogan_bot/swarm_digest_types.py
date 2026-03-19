"""Structured data contracts for the Swarm Daily Digest."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DigestSeverity = Literal["healthy", "watch", "warning", "critical"]


@dataclass
class DigestMetric:
    """A single named metric value with optional annotation."""

    name: str
    value: float | int | str | bool | None
    notes: str = ""


@dataclass
class DigestFlag:
    """A severity-tagged flag raised by the digest rules engine."""

    level: DigestSeverity
    code: str
    message: str
    action: str = ""


@dataclass
class ReplayCandidate:
    """A decision flagged for manual review."""

    decision_id: int
    symbol: str
    ts_iso: str
    reason: str
    priority: int


@dataclass
class DailyDigest:
    """Complete daily digest output."""

    date: str
    phase: str
    symbol: str | None
    timeframe: str | None
    severity: DigestSeverity
    headline: str
    metrics: dict[str, Any]
    flags: list[DigestFlag] = field(default_factory=list)
    replay_candidates: list[ReplayCandidate] = field(default_factory=list)
    operator_actions: list[str] = field(default_factory=list)
    summary_md: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "phase": self.phase,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "severity": self.severity,
            "headline": self.headline,
            "metrics": self.metrics,
            "flags": [
                {"level": f.level, "code": f.code, "message": f.message, "action": f.action}
                for f in self.flags
            ],
            "replay_candidates": [
                {
                    "decision_id": rc.decision_id,
                    "symbol": rc.symbol,
                    "ts_iso": rc.ts_iso,
                    "reason": rc.reason,
                    "priority": rc.priority,
                }
                for rc in self.replay_candidates
            ],
            "operator_actions": self.operator_actions,
        }
