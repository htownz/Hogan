"""Structured data contracts for threshold tuning and agent quarantine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AgentMode = Literal["active", "advisory_only", "no_veto", "quarantined"]
Severity = Literal["info", "warn", "critical"]
TuningRecommendation = Literal[
    "hold", "relax_thresholds", "disable_veto_only", "advisory_only", "quarantine",
]


@dataclass
class ThresholdBundle:
    """A named, versioned set of threshold values for one agent."""

    bundle_id: str
    agent_id: str
    version: int
    values: dict[str, float | int | bool | str]
    notes: str = ""
    active: bool = False


@dataclass
class ThresholdChange:
    """Audit record for a single threshold field change."""

    ts: str
    agent_id: str
    bundle_id: str
    field_name: str
    old_value: Any
    new_value: Any
    reason: str
    operator: str


@dataclass
class AgentQuarantineState:
    """Persisted mode state for one swarm agent."""

    agent_id: str
    mode: AgentMode
    reason: str
    operator: str
    changed_at: str


@dataclass
class StallAlert:
    """A single stall-detection alert."""

    code: str
    severity: Severity
    metric_name: str
    actual: float | int
    threshold: float | int
    notes: str = ""


@dataclass
class ThresholdReviewResult:
    """Output of a threshold review for a single agent."""

    agent_id: str
    window_hours: int
    decision_count: int
    would_trade_count: int
    veto_ratio: float
    top_veto_reasons: list[dict[str, Any]]
    dominant_veto_agent: str | None
    dominant_veto_agent_share: float
    pre_veto_would_trade_count: int
    post_veto_would_trade_count: int
    pre_veto_agreement_mean: float | None
    post_veto_agreement_mean: float | None
    pre_veto_confidence_mean: float | None
    post_veto_confidence_mean: float | None
    recommendation: TuningRecommendation
    stall_alerts: list[StallAlert] = field(default_factory=list)
    notes: str = ""
