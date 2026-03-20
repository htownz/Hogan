"""Deterministic outcome classification and attribution for swarm decisions.

Every function here is a pure computation — no DB access, no LLM calls.
The caller provides the structured data; this module returns labels, scores,
and a human-readable learning note.

Attribution scores are in [-1, 1] where:
  +1 = component contributed positively to the outcome
  -1 = component detracted from the outcome
   0 = neutral / not applicable
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

POSITIVE_BPS = 10.0
NEGATIVE_BPS = -10.0
STRONG_POSITIVE_BPS = 20.0
STRONG_NEGATIVE_BPS = -20.0


def classify_outcome(
    decision: dict,
    outcome: dict,
    baseline_compare: dict | None = None,
    *,
    positive_bps: float = POSITIVE_BPS,
    negative_bps: float = NEGATIVE_BPS,
    strong_positive_bps: float = STRONG_POSITIVE_BPS,
    strong_negative_bps: float = STRONG_NEGATIVE_BPS,
) -> str:
    """Assign a deterministic outcome label to a swarm decision.

    Labels (in priority order):
      Saved by Veto, False Veto, Winner, Loser,
      Correct Skip, Missed Winner, Correct Downsize,
      Too Small, Too Large, Entry Too Early, Entry Too Late
    """
    vetoed = bool(decision.get("vetoed"))
    should_trade = decision.get("final_action", "hold") in ("buy", "sell")
    fwd_bps = outcome.get("forward_60m_bps") or outcome.get("forward_return_bps")

    baseline_would_trade = False
    if baseline_compare:
        bl_action = baseline_compare.get("final_action", "hold")
        baseline_would_trade = bl_action in ("buy", "sell")

    if fwd_bps is None:
        return "Pending"

    # --- Veto classifications ---
    if vetoed and baseline_would_trade:
        bl_action = baseline_compare.get("final_action", "hold") if baseline_compare else "hold"
        directed_return = fwd_bps if bl_action == "buy" else -fwd_bps
        if directed_return < negative_bps:
            return "Saved by Veto"
        elif directed_return > strong_positive_bps:
            return "False Veto"

    # --- Trade taken ---
    if should_trade and not vetoed:
        action = decision.get("final_action", "hold")
        directed_return = fwd_bps if action == "buy" else -fwd_bps
        if directed_return > positive_bps:
            return "Winner"
        elif directed_return < negative_bps:
            return "Loser"
        return "Scratch"

    # --- Skip / hold ---
    if not should_trade and not vetoed:
        if baseline_would_trade and baseline_compare:
            bl_action = baseline_compare.get("final_action", "hold")
            directed_return = fwd_bps if bl_action == "buy" else -fwd_bps
            if directed_return > strong_positive_bps:
                return "Missed Winner"
            elif directed_return < negative_bps:
                return "Correct Skip"
        return "Correct Skip"

    # Fallback for vetoed without baseline
    if vetoed:
        return "Saved by Veto" if fwd_bps < negative_bps else "Neutral Veto"

    return "Unclassified"


# ---------------------------------------------------------------------------
# Attribution scores
# ---------------------------------------------------------------------------

def compute_direction_attribution(
    decision: dict,
    outcome: dict,
) -> float:
    """Did the swarm pick the right direction?

    +1 if direction matched outcome, -1 if opposite, 0 for hold.
    """
    action = decision.get("final_action", "hold")
    fwd_bps = outcome.get("forward_60m_bps") or outcome.get("forward_return_bps")
    if action == "hold" or fwd_bps is None:
        return 0.0

    directed = fwd_bps if action == "buy" else -fwd_bps
    if directed > POSITIVE_BPS:
        return 1.0
    elif directed < NEGATIVE_BPS:
        return -1.0
    return 0.0


def compute_veto_attribution(
    decision: dict,
    outcome: dict,
    baseline_compare: dict | None = None,
) -> float:
    """Did the veto add or destroy value?

    +1 if veto prevented a loss, -1 if veto blocked a winner, 0 otherwise.
    """
    if not decision.get("vetoed"):
        return 0.0

    fwd_bps = outcome.get("forward_60m_bps") or outcome.get("forward_return_bps")
    if fwd_bps is None:
        return 0.0

    if baseline_compare:
        bl_action = baseline_compare.get("final_action", "hold")
        if bl_action in ("buy", "sell"):
            directed = fwd_bps if bl_action == "buy" else -fwd_bps
            if directed < NEGATIVE_BPS:
                return 1.0
            elif directed > STRONG_POSITIVE_BPS:
                return -1.0

    was_correct = outcome.get("was_veto_correct")
    if was_correct == 1:
        return 1.0
    elif was_correct == 0:
        return -1.0
    return 0.0


def compute_posture_attribution(
    decision: dict,
    outcome: dict,
) -> float:
    """Was the posture (full/reduced/probe/skip) appropriate?

    Compares size_scale against outcome magnitude.
    """
    size_scale = decision.get("final_scale", decision.get("size_multiplier", 1.0)) or 1.0
    fwd_bps = outcome.get("forward_60m_bps") or outcome.get("forward_return_bps")
    if fwd_bps is None:
        return 0.0

    action = decision.get("final_action", "hold")
    if action == "hold":
        return 0.0

    directed = fwd_bps if action == "buy" else -fwd_bps

    if directed > POSITIVE_BPS:
        # Winner: large size = good, small size = missed opportunity
        return min(1.0, size_scale)
    elif directed < NEGATIVE_BPS:
        # Loser: small size = good risk mgmt, large size = painful
        return max(-1.0, -(size_scale))
    return 0.0


def compute_entry_attribution(
    decision: dict,
    outcome: dict,
) -> float:
    """Was entry timing appropriate? Uses MAE/MFE ratio."""
    mae = outcome.get("mae_bps")
    mfe = outcome.get("mfe_bps")
    if mae is None or mfe is None:
        return 0.0

    if mfe <= 0:
        return -0.5  # never saw favorable movement

    ratio = mae / mfe if mfe > 0 else float("inf")
    # Low MAE/MFE ratio = good timing, high = bad timing
    if ratio < 0.3:
        return 0.8
    elif ratio < 0.6:
        return 0.3
    elif ratio > 1.5:
        return -0.8
    elif ratio > 1.0:
        return -0.3
    return 0.0


def compute_cost_attribution(
    decision: dict,
    outcome: dict,
) -> float:
    """Did execution costs materially affect the outcome?"""
    slippage = outcome.get("realized_slippage_bps") or outcome.get("slippage_bps")
    cost_drag = outcome.get("cost_drag_bps")
    fwd_bps = outcome.get("forward_60m_bps") or outcome.get("forward_return_bps")

    if fwd_bps is None:
        return 0.0

    total_cost = (slippage or 0.0) + (cost_drag or 0.0)
    if total_cost == 0:
        return 0.0

    if abs(fwd_bps) < 1:
        return -0.5 if total_cost > 5 else 0.0

    cost_ratio = total_cost / abs(fwd_bps) if fwd_bps != 0 else 0
    if cost_ratio > 0.5:
        return -0.8
    elif cost_ratio > 0.25:
        return -0.3
    return 0.0


def compute_disagreement_attribution(
    decision: dict,
    outcome: dict,
) -> float:
    """Did high disagreement predict poor outcomes?"""
    agreement = decision.get("agreement", 1.0) or 1.0
    fwd_bps = outcome.get("forward_60m_bps") or outcome.get("forward_return_bps")
    action = decision.get("final_action", "hold")

    if action == "hold" or fwd_bps is None:
        return 0.0

    directed = fwd_bps if action == "buy" else -fwd_bps

    if agreement < 0.5:
        # High disagreement — was the controller right to proceed?
        return 0.5 if directed > POSITIVE_BPS else -0.5
    return 0.0


# ---------------------------------------------------------------------------
# Full attribution bundle
# ---------------------------------------------------------------------------

def compute_full_attribution(
    decision: dict,
    outcome: dict,
    baseline_compare: dict | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Compute all attribution scores and outcome label."""
    label = classify_outcome(decision, outcome, baseline_compare, **kwargs)
    return {
        "outcome_label": label,
        "direction_attr": round(compute_direction_attribution(decision, outcome), 4),
        "veto_attr": round(compute_veto_attribution(decision, outcome, baseline_compare), 4),
        "posture_attr": round(compute_posture_attribution(decision, outcome), 4),
        "entry_attr": round(compute_entry_attribution(decision, outcome), 4),
        "cost_attr": round(compute_cost_attribution(decision, outcome), 4),
        "disagreement_attr": round(compute_disagreement_attribution(decision, outcome), 4),
    }


# ---------------------------------------------------------------------------
# Learning note — deterministic plain-English summary
# ---------------------------------------------------------------------------

def build_learning_note(
    decision: dict,
    votes: list[dict],
    outcome: dict,
    attribution: dict,
) -> str:
    """Build a deterministic, structured learning note from attribution data."""
    parts: list[str] = []
    label = attribution.get("outcome_label", "Unclassified")
    action = decision.get("final_action", "hold")
    agreement = decision.get("agreement", 0)
    regime = decision.get("regime", "unknown")

    parts.append(f"Outcome: {label}.")

    if label == "Saved by Veto":
        veto_agents = [v.get("agent_id", "?") for v in votes if v.get("veto")]
        parts.append(f"Veto by {', '.join(veto_agents) or 'unknown'} prevented a loss.")
        parts.append("Pattern: vetoes are adding value.")
    elif label == "False Veto":
        parts.append("Veto blocked a winning trade — review veto thresholds.")
        parts.append("Check if the vetoing agent is too conservative.")
    elif label == "Winner":
        _conf = decision.get("final_conf", decision.get("confidence", 0))
        parts.append(f"Correct {action} in {regime} regime ({_conf:.0%} confidence).")
        if attribution.get("entry_attr", 0) > 0.5:
            parts.append("Entry timing was good.")
        elif attribution.get("entry_attr", 0) < -0.3:
            parts.append("Entry timing could improve — high MAE relative to MFE.")
    elif label == "Loser":
        parts.append(f"Incorrect {action} in {regime} regime.")
        if attribution.get("disagreement_attr", 0) < -0.3:
            parts.append("High disagreement was a warning signal — consider tightening entropy gate.")
        if attribution.get("posture_attr", 0) < -0.5:
            parts.append("Position was oversized for a losing trade.")
    elif label == "Missed Winner":
        parts.append("Swarm correctly identified hold but baseline would have won.")
        parts.append("Review skip thresholds — may be too conservative.")
    elif label == "Correct Skip":
        parts.append("Correct to skip — forward movement was negative or flat.")

    # Disagreement note
    if agreement < 0.5:
        parts.append(f"Low agreement ({agreement:.0%}) — agents were divided.")

    # Dominant positive/negative attribution
    attrs = {k: v for k, v in attribution.items()
             if k.endswith("_attr") and isinstance(v, (int, float))}
    if attrs:
        best = max(attrs, key=lambda k: attrs[k])
        worst = min(attrs, key=lambda k: attrs[k])
        if attrs[best] > 0.3:
            parts.append(f"Strongest contributor: {best.replace('_attr', '')}.")
        if attrs[worst] < -0.3:
            parts.append(f"Weakest contributor: {worst.replace('_attr', '')}.")

    return " ".join(parts)
