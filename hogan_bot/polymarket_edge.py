"""After-cost edge scoring for Polymarket opportunity candidates."""
from __future__ import annotations

from dataclasses import dataclass

from hogan_bot.fetch_polymarket import PolymarketOpportunity


@dataclass(frozen=True)
class EdgeAssessment:
    market_id: str
    side: str
    fair_probability: float
    market_probability: float
    expected_value: float
    after_cost_ev: float
    max_size_usd: float
    decision: str
    reject_reasons: list[str]


def _clip_prob(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def assess_opportunity_edge(
    opportunity: PolymarketOpportunity,
    *,
    calibrated_probability: float | None = None,
    fee_rate: float = 0.0,
    slippage_bps: float = 25.0,
    max_spread: float = 0.08,
    min_liquidity_score: float = 0.20,
    days_to_expiry: float | None = None,
    max_size_usd: float = 25.0,
) -> EdgeAssessment:
    """Estimate simple after-cost EV for a candidate YES/NO position."""
    fair = _clip_prob(
        calibrated_probability
        if calibrated_probability is not None
        else (opportunity.hogan_prob if opportunity.hogan_prob is not None else opportunity.crowd_prob)
    )
    market = _clip_prob(opportunity.crowd_prob)
    side = opportunity.candidate_side
    reject_reasons: list[str] = []

    if side == "buy_no":
        fair_position_prob = 1.0 - fair
        entry_price = 1.0 - market
    elif side == "buy_yes":
        fair_position_prob = fair
        entry_price = market
    else:
        fair_position_prob = fair
        entry_price = market
        reject_reasons.append("research_only_side")

    expected_value = fair_position_prob - entry_price
    spread_cost = max(0.0, 1.0 - opportunity.spread_score) * max_spread
    slippage_cost = slippage_bps / 10_000.0
    fee_cost = max(0.0, fee_rate) * entry_price * (1.0 - entry_price)
    expiry_penalty = 0.0
    if days_to_expiry is not None:
        if days_to_expiry <= 0:
            reject_reasons.append("expired_or_expiring")
        elif days_to_expiry < 1:
            expiry_penalty = 0.02
    after_cost_ev = expected_value - spread_cost - slippage_cost - fee_cost - expiry_penalty

    if opportunity.spread_score < 1.0 - max_spread / 0.10:
        reject_reasons.append("spread_too_wide")
    if opportunity.liquidity_score < min_liquidity_score:
        reject_reasons.append("low_liquidity")
    if after_cost_ev <= 0:
        reject_reasons.append("non_positive_ev")

    if reject_reasons:
        decision = "reject"
    elif after_cost_ev >= 0.05 and opportunity.total_score >= 0.60:
        decision = "shadow_trade"
    else:
        decision = "research"

    liquidity_scale = max(0.0, min(1.0, opportunity.liquidity_score))
    size = max_size_usd * liquidity_scale * max(0.0, min(1.0, after_cost_ev / 0.15))
    return EdgeAssessment(
        market_id=opportunity.market_id,
        side=side,
        fair_probability=fair_position_prob,
        market_probability=entry_price,
        expected_value=expected_value,
        after_cost_ev=after_cost_ev,
        max_size_usd=max(0.0, size),
        decision=decision,
        reject_reasons=reject_reasons,
    )
