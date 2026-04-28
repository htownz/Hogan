"""Machine-readable Polymarket intelligence assessments.

The intelligence layer stays analysis-only. It turns normalized public data,
opportunity scoring, and after-cost edge checks into explicit recommendations
for the Alpha Lab shadow ledger and operator reports.
"""
from __future__ import annotations

from dataclasses import dataclass

from hogan_bot.fetch_polymarket import PolymarketMarketSnapshot, PolymarketOpportunity
from hogan_bot.polymarket_edge import EdgeAssessment


@dataclass(frozen=True)
class IntelligenceAssessment:
    market_id: str
    recommendation: str
    fair_value_source: str
    thesis: str
    evidence_score: float
    risk_flags: list[str]
    data_quality_score: float
    shadow_eligible: bool

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "recommendation": self.recommendation,
            "fair_value_source": self.fair_value_source,
            "thesis": self.thesis,
            "evidence_score": round(self.evidence_score, 4),
            "risk_flags": list(self.risk_flags),
            "data_quality_score": round(self.data_quality_score, 4),
            "shadow_eligible": self.shadow_eligible,
        }


def _fair_value_source(opportunity: PolymarketOpportunity) -> str:
    if opportunity.hogan_prob is None:
        if opportunity.safety_note:
            return "unavailable"
        return "market_implied_only"
    if opportunity.market_type == "price_target" and opportunity.horizon == "long_term":
        return "calibrated_long_horizon"
    return "hogan_short_term_ml"


def _thesis(opportunity: PolymarketOpportunity, edge: EdgeAssessment) -> str:
    if opportunity.safety_note:
        return f"Research-only: {opportunity.safety_note}."
    if edge.after_cost_ev > 0 and opportunity.hogan_prob is not None:
        return (
            f"Model/crowd disagreement with after-cost EV {edge.after_cost_ev:.4f}; "
            f"{opportunity.rationale}."
        )
    if opportunity.candidate_side == "research":
        return f"Informational market; {opportunity.rationale}."
    return f"No positive after-cost edge; {opportunity.rationale}."


def assess_intelligence(
    opportunity: PolymarketOpportunity,
    edge: EdgeAssessment,
    snapshot: PolymarketMarketSnapshot,
    *,
    min_data_quality: float = 0.55,
) -> IntelligenceAssessment:
    """Return a machine recommendation for one opportunity."""
    risk_flags = list(snapshot.quality_flags)
    if opportunity.safety_note:
        risk_flags.append(opportunity.safety_note)
    if opportunity.market_type == "unknown" or opportunity.horizon == "unknown":
        risk_flags.append("ambiguous_market_context")
    if snapshot.data_quality_score < min_data_quality:
        risk_flags.append("low_data_quality")
    if edge.after_cost_ev <= 0:
        risk_flags.append("non_positive_after_cost_ev")
    if _fair_value_source(opportunity) == "unavailable":
        risk_flags.append("fair_value_unavailable")
    if snapshot.eligibility == "blocked":
        risk_flags.append("snapshot_blocked")

    evidence = (
        opportunity.edge_score * 0.30
        + opportunity.liquidity_score * 0.20
        + opportunity.spread_score * 0.20
        + opportunity.catalyst_score * 0.10
        + snapshot.data_quality_score * 0.20
    )
    evidence = max(0.0, min(1.0, evidence))

    critical = {
        "fair_value_unavailable",
        "long_horizon_price_target_requires_calibrated_fair_value",
        "snapshot_blocked",
    }
    shadow_eligible = (
        edge.decision == "shadow_trade"
        and opportunity.candidate_side in ("buy_yes", "buy_no")
        and snapshot.eligibility == "shadow_candidate"
        and snapshot.data_quality_score >= min_data_quality
        and not critical.intersection(risk_flags)
    )
    if shadow_eligible:
        recommendation = "shadow_candidate"
    elif edge.decision == "reject" or "snapshot_blocked" in risk_flags:
        recommendation = "avoid"
    elif opportunity.candidate_side == "research" or opportunity.safety_note:
        recommendation = "research"
    else:
        recommendation = "monitor"

    return IntelligenceAssessment(
        market_id=opportunity.market_id,
        recommendation=recommendation,
        fair_value_source=_fair_value_source(opportunity),
        thesis=_thesis(opportunity, edge),
        evidence_score=evidence,
        risk_flags=sorted(set(risk_flags)),
        data_quality_score=snapshot.data_quality_score,
        shadow_eligible=shadow_eligible,
    )
