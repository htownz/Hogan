"""Structured data contracts for the swarm decision layer.

Every agent and the controller produce typed, deterministic outputs.
These dataclasses are the canonical interchange format — no free-form
strings in the decision path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FreshnessInfo:
    """Point-in-time freshness metadata for a data source."""

    as_of_ms: int
    latest_source_ts_ms: int | None = None
    latest_source_date: str | None = None
    age_seconds: float = 0.0
    is_stale: bool = False
    coverage_score: float = 1.0


@dataclass
class AgentVote:
    """A single expert's structured opinion on one bar.

    ``size_scale`` defaults to 1.0 (no adjustment).
    A veto agent sets ``veto=True`` and ``size_scale=0.0``.
    """

    agent_id: str
    action: str  # "buy" | "sell" | "hold"
    confidence: float  # [0, 1]
    expected_edge_bps: float | None = None
    stop_distance_pct: float | None = None
    size_scale: float = 1.0  # 0.0 = veto
    veto: bool = False
    block_reasons: list[str] = field(default_factory=list)
    features_used: list[str] = field(default_factory=list)
    freshness: FreshnessInfo | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "agent_id": self.agent_id,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "size_scale": round(self.size_scale, 4),
            "veto": self.veto,
            "block_reasons": self.block_reasons,
        }
        if self.expected_edge_bps is not None:
            d["expected_edge_bps"] = round(self.expected_edge_bps, 2)
        if self.stop_distance_pct is not None:
            d["stop_distance_pct"] = round(self.stop_distance_pct, 6)
        if self.features_used:
            d["features_used"] = self.features_used
        return d


def _compute_entropy(action_weights: dict[str, float]) -> float:
    """Shannon entropy of the action distribution (bits)."""
    total = sum(action_weights.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for w in action_weights.values():
        p = w / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


@dataclass
class SwarmDecision:
    """Fused output of the SwarmController for one bar.

    Contains the final action, the individual votes, and diagnostics
    needed for audit, dashboarding, and weight-update learning loops.
    """

    final_action: str  # "buy" | "sell" | "hold"
    final_confidence: float  # [0, 1]
    final_size_scale: float  # composite size multiplier from agents
    agreement: float  # vote-share of the winning action [0, 1]
    entropy: float  # Shannon entropy of action distribution (bits)
    weights_used: dict[str, float]
    votes: list[AgentVote]
    vetoed: bool = False
    block_reasons: list[str] = field(default_factory=list)

    # Pre-veto analytics (populated by controller before veto application)
    pre_veto_action: str | None = None
    pre_veto_confidence: float | None = None
    pre_veto_agreement: float | None = None
    pre_veto_entropy: float | None = None
    dominant_veto_agent: str | None = None
    veto_count: int = 0
    veto_agents: list[str] = field(default_factory=list)

    @property
    def n_agents(self) -> int:
        return len(self.votes)

    @property
    def n_vetoes(self) -> int:
        return sum(1 for v in self.votes if v.veto)

    def to_dict(self) -> dict:
        d: dict = {
            "final_action": self.final_action,
            "final_confidence": round(self.final_confidence, 4),
            "final_size_scale": round(self.final_size_scale, 4),
            "agreement": round(self.agreement, 4),
            "entropy": round(self.entropy, 4),
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "votes": [v.to_dict() for v in self.votes],
            "vetoed": self.vetoed,
            "block_reasons": self.block_reasons,
        }
        if self.pre_veto_action is not None:
            d["pre_veto_action"] = self.pre_veto_action
            d["pre_veto_confidence"] = round(self.pre_veto_confidence, 4) if self.pre_veto_confidence is not None else None
            d["pre_veto_agreement"] = round(self.pre_veto_agreement, 4) if self.pre_veto_agreement is not None else None
            d["pre_veto_entropy"] = round(self.pre_veto_entropy, 4) if self.pre_veto_entropy is not None else None
        if self.dominant_veto_agent:
            d["dominant_veto_agent"] = self.dominant_veto_agent
        if self.veto_count:
            d["veto_count"] = self.veto_count
            d["veto_agents"] = self.veto_agents
        return d


@dataclass
class DecisionIntent:
    """Final trade intent produced by policy_core.decide().

    This is the single, canonical output consumed by both the live event
    loop and the backtest engine.  It carries everything needed to decide
    *whether* and *how much* to trade, plus full audit metadata.
    """

    action: str  # "buy" | "sell" | "hold"
    confidence: float  # [0, 1]
    size_usd: float  # position size in quote currency
    stop_distance_pct: float  # from the signal pipeline
    up_prob: float | None = None
    regime: str | None = None
    regime_confidence: float | None = None
    atr_pct: float = 0.0

    # Effective regime-adjusted parameters (used by execution layer)
    eff_trailing_stop_pct: float | None = None
    eff_take_profit_pct: float | None = None
    eff_allow_longs: bool = True
    eff_allow_shorts: bool = True
    eff_long_size_scale: float = 1.0
    eff_short_size_scale: float = 1.0

    # Scale factors applied (for diagnostics)
    conf_scale: float = 1.0
    quality_scale: float = 1.0
    ranging_scale: float = 1.0
    pullback_scale: float = 1.0
    momentum_scale: float = 1.0
    freshness_scale: float = 1.0
    macro_scale: float = 1.0
    funding_scale: float = 1.0

    # Pipeline metadata
    explanation: str | None = None
    forecast_ret: float | None = None
    agent_weights: dict | None = None
    feature_freshness: dict | None = None
    vol_ratio: float = 1.0
    tech_confidence: float | None = None
    trade_quality_prob: float | None = None
    direction_score: float = 0.0
    quality_score: float = 0.0
    size_score: float = 0.0
    unified_score: float = 0.0
    block_reasons: list[str] = field(default_factory=list)

    # Raw signal tracking (for decision log diagnostics)
    raw_tech_action: str | None = None      # TechnicalAgent output before MetaWeigher
    pipeline_action: str | None = None      # MetaWeigher output before ML/gates

    # Swarm layer (None when swarm is off)
    swarm: SwarmDecision | None = None
    swarm_decision_id: int | None = None

    # Funnel counters for diagnostics (backtest populates these)
    funnel: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "size_usd": round(self.size_usd, 4),
            "stop_distance_pct": round(self.stop_distance_pct, 6),
            "regime": self.regime,
            "direction_score": round(self.direction_score, 4),
            "quality_score": round(self.quality_score, 4),
            "size_score": round(self.size_score, 4),
            "unified_score": round(self.unified_score, 4),
            "block_reasons": self.block_reasons,
            "raw_tech_action": self.raw_tech_action,
            "pipeline_action": self.pipeline_action,
        }
        if self.up_prob is not None:
            d["up_prob"] = round(self.up_prob, 4)
        if self.trade_quality_prob is not None:
            d["trade_quality_prob"] = round(self.trade_quality_prob, 4)
        if self.swarm is not None:
            d["swarm"] = self.swarm.to_dict()
        return d
