"""Pipeline agent — wraps AgentPipeline.run() output as an AgentVote.

This is the "composite expert" that reuses the existing multi-agent
intelligence (Technical + Sentiment + Macro) already in Hogan.
"""
from __future__ import annotations

import pandas as pd

from hogan_bot.swarm_decision.types import AgentVote


class PipelineAgent:
    """Wraps an existing ``AgentPipeline`` instance into a swarm agent."""

    agent_id: str = "pipeline_v1"

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    def vote(
        self,
        *,
        symbol: str,
        candles: pd.DataFrame,
        as_of_ms: int | None,
        shared_context: dict,
    ) -> AgentVote:
        regime = shared_context.get("regime")
        regime_state = shared_context.get("regime_state")

        signal = self._pipeline.run(
            candles,
            symbol=symbol,
            as_of_ms=as_of_ms,
            regime=regime,
            regime_state=regime_state,
        )

        action = signal.action or "hold"
        confidence = signal.confidence or 0.0
        stop_dist = signal.stop_distance_pct

        edge_bps: float | None = None
        if signal.forecast is not None:
            er = getattr(signal.forecast, "expected_return", None)
            if isinstance(er, dict) and er:
                edge_bps = max(abs(v) for v in er.values()) * 10_000
            elif isinstance(er, (int, float)):
                edge_bps = abs(float(er)) * 10_000

        return AgentVote(
            agent_id=self.agent_id,
            action=action,
            confidence=max(0.0, min(1.0, confidence)),
            expected_edge_bps=edge_bps,
            stop_distance_pct=stop_dist,
            size_scale=1.0,
            veto=False,
            block_reasons=[],
        )
