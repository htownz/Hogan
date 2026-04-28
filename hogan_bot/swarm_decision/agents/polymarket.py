"""Polymarket swarm agent.

This agent is analysis-only. It consumes Polymarket metrics already attached
to the AgentPipeline signal and never talks to authenticated Polymarket APIs.
"""
from __future__ import annotations

import pandas as pd

from hogan_bot.swarm_decision.types import AgentVote


class PolymarketAgent:
    """Vote from prediction-market implied BTC/ETH and macro probabilities."""

    agent_id: str = "polymarket_v1"

    def __init__(
        self,
        min_signal: float = 0.20,
        veto_threshold: float = 0.65,
    ) -> None:
        self.min_signal = max(0.0, min(1.0, min_signal))
        self.veto_threshold = max(0.0, min(1.0, veto_threshold))

    def vote(
        self,
        *,
        symbol: str,
        candles: pd.DataFrame,
        as_of_ms: int | None,
        shared_context: dict,
    ) -> AgentVote:
        signal = shared_context.get("pipeline_signal")
        sent_details = getattr(getattr(signal, "sentiment", None), "details", {}) or {}
        macro_details = getattr(getattr(signal, "macro", None), "details", {}) or {}

        features_used: list[str] = []
        score = 0.0

        if "polymarket_bull" in sent_details:
            bull_prob = max(0.0, min(1.0, float(sent_details["polymarket_bull"])))
            score += (bull_prob - 0.5) * 2.0
            features_used.append("polymarket_bull")

        if "polymarket_crypto_risk" in sent_details:
            risk_support = max(0.0, min(1.0, float(sent_details["polymarket_crypto_risk"])))
            score += (risk_support - 0.5) * 0.8
            features_used.append("polymarket_crypto_risk")

        if "poly_macro_risk_prob" in macro_details:
            macro_risk = max(0.0, min(1.0, float(macro_details["poly_macro_risk_prob"])))
            score -= (macro_risk - 0.5) * 0.6
            features_used.append("poly_macro_risk_prob")

        if not features_used:
            return AgentVote(
                agent_id=self.agent_id,
                action="hold",
                confidence=0.0,
                size_scale=1.0,
                block_reasons=["polymarket_no_signal"],
            )

        score = max(-1.0, min(1.0, score))
        confidence = abs(score)
        if score >= self.min_signal:
            action = "buy"
        elif score <= -self.min_signal:
            action = "sell"
        else:
            action = "hold"

        pipeline_action = getattr(signal, "action", None)
        veto = (
            (pipeline_action == "buy" and score <= -self.veto_threshold)
            or (pipeline_action == "sell" and score >= self.veto_threshold)
        )
        reasons: list[str] = []
        size_scale = 1.0
        if veto:
            reasons.append("polymarket_strong_conflict")
            size_scale = 0.0
        elif pipeline_action in ("buy", "sell") and action not in (pipeline_action, "hold"):
            reasons.append("polymarket_direction_conflict")
            size_scale = 0.5

        return AgentVote(
            agent_id=self.agent_id,
            action=action,
            confidence=confidence,
            expected_edge_bps=score * 100.0,
            size_scale=size_scale,
            veto=veto,
            block_reasons=reasons,
            features_used=features_used,
        )
