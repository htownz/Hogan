"""Structured data contracts for the Swarm Weekly Review."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ReviewSeverity = Literal["healthy", "watch", "warning", "critical"]
ReviewRecommendation = Literal[
    "promote",
    "hold",
    "rollback",
    "tune_thresholds",
    "fix_instrumentation",
    "quarantine_agent",
    "insufficient_data",
]


@dataclass
class WeeklyFlag:
    """A severity-tagged flag raised by the weekly review rules engine."""

    level: ReviewSeverity
    code: str
    message: str
    action: str = ""


@dataclass
class AgentWeeklyScore:
    """Per-agent weekly scorecard."""

    agent_id: str
    decisions: int
    vetoes: int
    hold_rate: float
    mean_confidence: float | None
    mean_edge_bps: float | None
    contribution_score: float | None
    notes: str = ""


@dataclass
class WeeklyReplayCandidate:
    """A decision flagged for weekly manual review, with category label."""

    decision_id: int
    symbol: str
    ts_iso: str
    category: str
    reason: str
    priority: int


@dataclass
class WeeklyReview:
    """Complete weekly review output."""

    week_label: str
    phase: str
    symbol: str | None
    timeframe: str | None
    severity: ReviewSeverity
    headline: str
    metrics: dict[str, Any]
    flags: list[WeeklyFlag] = field(default_factory=list)
    agent_scores: list[AgentWeeklyScore] = field(default_factory=list)
    replay_candidates: list[WeeklyReplayCandidate] = field(default_factory=list)
    operator_actions: list[str] = field(default_factory=list)
    cursor_actions: list[str] = field(default_factory=list)
    recommendation: ReviewRecommendation = "hold"
    summary_md: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "week_label": self.week_label,
            "phase": self.phase,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "severity": self.severity,
            "headline": self.headline,
            "recommendation": self.recommendation,
            "metrics": self.metrics,
            "flags": [
                {"level": f.level, "code": f.code, "message": f.message, "action": f.action}
                for f in self.flags
            ],
            "agent_scores": [
                {
                    "agent_id": a.agent_id, "decisions": a.decisions, "vetoes": a.vetoes,
                    "hold_rate": round(a.hold_rate, 4), "mean_confidence": a.mean_confidence,
                    "mean_edge_bps": a.mean_edge_bps, "contribution_score": a.contribution_score,
                    "notes": a.notes,
                }
                for a in self.agent_scores
            ],
            "replay_candidates": [
                {
                    "decision_id": rc.decision_id, "symbol": rc.symbol, "ts_iso": rc.ts_iso,
                    "category": rc.category, "reason": rc.reason, "priority": rc.priority,
                }
                for rc in self.replay_candidates
            ],
            "operator_actions": self.operator_actions,
            "cursor_actions": self.cursor_actions,
        }
