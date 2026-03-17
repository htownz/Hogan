"""SwarmController — multi-agent decision fusion.

Aggregates votes from N expert agents into a single ``SwarmDecision``
via weighted voting, hard safety vetoes, and conservative tie-breaking.

The controller is deterministic: given identical votes and weights it
always returns the same ``SwarmDecision``.
"""
from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd

from hogan_bot.swarm_decision.types import AgentVote, SwarmDecision, _compute_entropy

logger = logging.getLogger(__name__)


class DecisionAgent(Protocol):
    """Interface every swarm agent must implement."""

    agent_id: str

    def vote(
        self,
        *,
        symbol: str,
        candles: pd.DataFrame,
        as_of_ms: int | None,
        shared_context: dict,
    ) -> AgentVote: ...


class SwarmController:
    """Fuses agent votes into a single SwarmDecision.

    Fusion pipeline (per bar):
        1. Collect votes from all enabled agents
        2. Hard safety veto — any veto agent forces hold
        3. Weighted vote aggregation over {buy, sell, hold}
        4. Conservative tie-breaking — hold if margin too thin or entropy too high
        5. Composite size_scale from non-veto agents
        6. Build and return SwarmDecision
    """

    def __init__(
        self,
        *,
        agents: list[DecisionAgent],
        weights: dict[str, float] | None = None,
        config=None,
    ) -> None:
        self.agents = list(agents)
        self._config = config
        self._weights = weights or {a.agent_id: 1.0 for a in self.agents}
        _total = sum(self._weights.values()) or 1.0
        self._weights = {k: v / _total for k, v in self._weights.items()}

        self._min_agreement = getattr(config, "swarm_min_agreement", 0.60)
        self._min_vote_margin = getattr(config, "swarm_min_vote_margin", 0.10)
        self._max_entropy = getattr(config, "swarm_max_entropy", 0.95)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def set_weights(self, weights: dict[str, float]) -> None:
        _total = sum(weights.values()) or 1.0
        self._weights = {k: v / _total for k, v in weights.items()}

    def decide(
        self,
        *,
        symbol: str,
        candles: pd.DataFrame,
        as_of_ms: int | None = None,
        shared_context: dict | None = None,
    ) -> SwarmDecision:
        ctx = shared_context or {}
        votes: list[AgentVote] = []

        for agent in self.agents:
            try:
                v = agent.vote(
                    symbol=symbol,
                    candles=candles,
                    as_of_ms=as_of_ms,
                    shared_context=ctx,
                )
                votes.append(v)
            except Exception as exc:
                logger.warning("Agent %s failed: %s", agent.agent_id, exc)
                votes.append(AgentVote(
                    agent_id=agent.agent_id,
                    action="hold",
                    confidence=0.0,
                    veto=False,
                    block_reasons=[f"agent_error:{type(exc).__name__}"],
                ))

        # 1. Hard safety veto
        veto_reasons: list[str] = []
        for v in votes:
            if v.veto:
                for r in v.block_reasons:
                    veto_reasons.append(f"{v.agent_id}:{r}")

        if veto_reasons:
            return SwarmDecision(
                final_action="hold",
                final_confidence=0.0,
                final_size_scale=0.0,
                agreement=1.0,
                entropy=0.0,
                weights_used=dict(self._weights),
                votes=votes,
                vetoed=True,
                block_reasons=veto_reasons,
            )

        # 2. Weighted vote aggregation
        action_scores: dict[str, float] = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        for v in votes:
            w = self._weights.get(v.agent_id, 0.0)
            action_scores[v.action] += w * v.confidence

        total_score = sum(action_scores.values()) or 1e-9
        action_probs = {k: v / total_score for k, v in action_scores.items()}

        sorted_actions = sorted(action_probs.items(), key=lambda x: -x[1])
        best_action = sorted_actions[0][0]
        best_score = sorted_actions[0][1]
        runner_up_score = sorted_actions[1][1] if len(sorted_actions) > 1 else 0.0
        margin = best_score - runner_up_score
        entropy = _compute_entropy(action_probs)

        # 3. Conservative tie-breaking
        block_reasons: list[str] = []
        if best_score < self._min_agreement:
            best_action = "hold"
            block_reasons.append(f"low_agreement:{best_score:.3f}")
        elif margin < self._min_vote_margin:
            best_action = "hold"
            block_reasons.append(f"thin_margin:{margin:.3f}")
        elif entropy > self._max_entropy:
            best_action = "hold"
            block_reasons.append(f"high_entropy:{entropy:.3f}")

        # 4. Composite size_scale
        size_scales: list[float] = []
        for v in votes:
            if not v.veto and v.size_scale > 0:
                w = self._weights.get(v.agent_id, 0.0)
                size_scales.append(v.size_scale * w)
        final_size_scale = sum(size_scales) / (sum(
            self._weights.get(v.agent_id, 0.0)
            for v in votes if not v.veto and v.size_scale > 0
        ) or 1.0)
        final_size_scale = max(0.0, min(1.0, final_size_scale))

        # Weighted confidence
        conf_num = sum(
            v.confidence * self._weights.get(v.agent_id, 0.0)
            for v in votes
        )
        conf_den = sum(self._weights.get(v.agent_id, 0.0) for v in votes) or 1.0
        final_confidence = max(0.0, min(1.0, conf_num / conf_den))

        return SwarmDecision(
            final_action=best_action,
            final_confidence=final_confidence,
            final_size_scale=final_size_scale,
            agreement=best_score,
            entropy=entropy,
            weights_used=dict(self._weights),
            votes=votes,
            vetoed=False,
            block_reasons=block_reasons,
        )
