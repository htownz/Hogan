"""Shared utilities for swarm decision agents."""
from __future__ import annotations


def get_baseline_action(shared_context: dict) -> str:
    """Extract the pipeline's action from shared context.

    Safety agents call this to adopt the pipeline's direction when
    they have no objection, rather than always defaulting to hold.
    """
    sig = shared_context.get("pipeline_signal")
    if sig is not None:
        action = getattr(sig, "action", None)
        if action in ("buy", "sell", "hold"):
            return action
    return "hold"
