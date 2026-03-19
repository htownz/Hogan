"""Swarm Decision Layer — multi-agent decision fusion for Hogan.

Enable with ``HOGAN_SWARM_ENABLED=true`` in ``.env``.
Mode: ``HOGAN_SWARM_MODE=shadow`` (default) logs without affecting execution;
``HOGAN_SWARM_MODE=active`` replaces baseline decisions with swarm output.
"""
from hogan_bot.swarm_decision.types import (
    AgentVote,
    DecisionIntent,
    FreshnessInfo,
    SwarmDecision,
)
from hogan_bot.swarm_decision.controller import SwarmController

__all__ = [
    "AgentVote",
    "DecisionIntent",
    "FreshnessInfo",
    "SwarmController",
    "SwarmDecision",
]
