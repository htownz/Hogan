"""Execution cost agent — veto/scale when edge < transaction cost.

Uses a Corwin-Schultz-inspired spread estimator from high/low prices
combined with the configured fee rate to estimate round-trip cost.
Vetoes when predicted edge after costs is negative.

When edge is sufficient, endorses the pipeline's direction
instead of defaulting to hold.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from hogan_bot.swarm_decision.agents._utils import get_baseline_action
from hogan_bot.swarm_decision.types import AgentVote


def _corwin_schultz_spread(candles: pd.DataFrame, window: int = 20) -> float:
    """Estimate effective spread from high-low prices.

    Based on the Corwin & Schultz (2012) principle that high/low
    prices embed both volatility and spread components.
    Returns spread as a fraction (e.g. 0.001 = 10 bps).
    """
    if len(candles) < window + 1:
        return 0.0

    high = candles["high"].values[-window:]
    low = candles["low"].values[-window:]
    close = candles["close"].values[-window:]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(high / np.maximum(low, 1e-12))
        beta = log_hl[:-1] ** 2 + log_hl[1:] ** 2
        gamma_arr = np.log(
            np.maximum(high[:-1], high[1:]) / np.maximum(np.minimum(low[:-1], low[1:]), 1e-12)
        ) ** 2

    beta_mean = np.nanmean(beta)
    gamma_mean = np.nanmean(gamma_arr)

    k = math.sqrt(2.0) - 1.0
    denom = 3.0 - 2.0 * math.sqrt(2.0)
    if denom == 0:
        return 0.0

    alpha = (math.sqrt(beta_mean) * k - math.sqrt(gamma_mean)) / denom
    alpha = max(alpha, 0.0)

    spread = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
    return max(0.0, spread)


class ExecutionCostAgent:
    """Vetoes when estimated round-trip cost exceeds predicted edge."""

    agent_id: str = "execution_cost_v1"

    def __init__(
        self,
        fee_rate: float = 0.0026,
        min_edge_over_cost: float = 1.5,
    ) -> None:
        self._fee_rate = fee_rate
        self._min_ratio = min_edge_over_cost

    def vote(
        self,
        *,
        symbol: str,
        candles: pd.DataFrame,
        as_of_ms: int | None,
        shared_context: dict,
    ) -> AgentVote:
        reasons: list[str] = []
        veto = False
        size_scale = 1.0

        spread = _corwin_schultz_spread(candles)
        round_trip_cost = 2.0 * self._fee_rate + spread
        cost_bps = round_trip_cost * 10_000

        atr_pct = shared_context.get("atr_pct", 0.0)
        tp_pct = shared_context.get("take_profit_pct", 0.0)
        edge_est = max(atr_pct, tp_pct)
        edge_bps = edge_est * 10_000

        if edge_bps <= 0 or cost_bps <= 0:
            baseline = get_baseline_action(shared_context)
            return AgentVote(
                agent_id=self.agent_id,
                action=baseline,
                confidence=0.5,
                expected_edge_bps=-cost_bps,
                size_scale=0.8,
                veto=False,
                block_reasons=["insufficient_edge_data"],
            )

        ratio = edge_bps / cost_bps
        if ratio < 1.0:
            veto = True
            size_scale = 0.0
            reasons.append(f"negative_edge_after_cost:ratio={ratio:.2f}")
        elif ratio < self._min_ratio:
            size_scale = ratio / self._min_ratio
            reasons.append(f"marginal_edge:ratio={ratio:.2f}")

        if veto:
            return AgentVote(
                agent_id=self.agent_id,
                action="hold",
                confidence=0.0,
                expected_edge_bps=edge_bps - cost_bps,
                size_scale=0.0,
                veto=True,
                block_reasons=reasons,
            )

        baseline = get_baseline_action(shared_context)
        confidence = min(1.0, max(0.5, ratio / self._min_ratio))
        return AgentVote(
            agent_id=self.agent_id,
            action=baseline,
            confidence=confidence,
            expected_edge_bps=edge_bps - cost_bps,
            size_scale=max(0.0, min(1.0, size_scale)),
            veto=False,
            block_reasons=reasons,
        )
