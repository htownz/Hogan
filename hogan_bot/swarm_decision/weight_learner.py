"""Swarm agent weight learning — propose-then-promote pattern.

Computes proposed weight adjustments from outcome data and writes them
as ``shadow_update`` snapshots.  Promotion to live weights requires
explicit operator acknowledgment (no auto-promote).

Learning signal per agent:
  - Direction accuracy: did the agent's vote match the forward return?
  - Veto value: did vetoes prevent losses?
  - Confidence calibration: is the agent overconfident or underconfident?

The final proposal is bounded by ``swarm_weight_max_daily_shift`` so
no single update can drastically change the weight distribution.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WeightProposal:
    """A proposed weight update with evidence."""
    current_weights: dict[str, float]
    proposed_weights: dict[str, float]
    deltas: dict[str, float]
    evidence: dict[str, dict]
    min_trades_met: bool
    regime: str | None = None
    stable: bool = False
    notes: str = ""


def compute_agent_accuracy(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    min_outcomes: int = 30,
    days: int = 14,
) -> pd.DataFrame:
    """Per-agent direction accuracy from outcomes.

    Returns a DataFrame with: agent_id, total_votes, correct_direction,
    accuracy, veto_count, veto_correct, mean_confidence.
    """
    cutoff_ms = int((time.time() - days * 86400) * 1000)
    sym_filter = "AND sav.symbol = ?" if symbol else ""
    params: list = [cutoff_ms]
    if symbol:
        params.append(symbol)

    df = pd.read_sql_query(
        f"""SELECT sav.agent_id, sav.action AS agent_action,
                   sav.confidence, sav.veto,
                   so.forward_60m_bps, so.was_veto_correct
            FROM swarm_agent_votes sav
            JOIN swarm_outcomes so ON sav.decision_id = so.decision_id
            WHERE sav.ts_ms >= ?
              {sym_filter}
              AND so.forward_60m_bps IS NOT NULL""",
        conn, params=params,
    )

    if df.empty:
        return pd.DataFrame()

    # Hold is correct only when the market stayed flat (within scratch
    # territory).  Unconditionally treating hold as correct inflates
    # accuracy for passive agents and biases weight proposals.
    _HOLD_CORRECT_BPS = 10
    df["direction_correct"] = (
        ((df["agent_action"] == "buy") & (df["forward_60m_bps"] > 0)) |
        ((df["agent_action"] == "sell") & (df["forward_60m_bps"] < 0)) |
        ((df["agent_action"] == "hold") & (df["forward_60m_bps"].abs() < _HOLD_CORRECT_BPS))
    ).astype(int)

    agg = df.groupby("agent_id").agg(
        total_votes=("agent_action", "count"),
        correct_direction=("direction_correct", "sum"),
        veto_count=("veto", "sum"),
        veto_correct=("was_veto_correct", lambda s: s.fillna(0).sum()),
        mean_confidence=("confidence", "mean"),
    ).reset_index()

    agg["accuracy"] = (agg["correct_direction"] / agg["total_votes"]).round(4)
    return agg


def propose_weights(
    conn: sqlite3.Connection,
    current_weights: dict[str, float],
    symbol: str | None = None,
    min_trades: int = 50,
    max_daily_shift: float = 0.05,
    days: int = 14,
    regime: str | None = None,
) -> WeightProposal:
    """Propose new agent weights based on outcome accuracy.

    Uses a simple proportional reweighting:
      raw_score = accuracy * (1 + veto_value_bonus)
      proposed_weight ∝ raw_score

    Bounded by max_daily_shift per agent.
    """
    accuracy_df = compute_agent_accuracy(conn, symbol=symbol, days=days)

    if accuracy_df.empty:
        return WeightProposal(
            current_weights=current_weights,
            proposed_weights=current_weights,
            deltas={k: 0.0 for k in current_weights},
            evidence={},
            min_trades_met=False,
            regime=regime,
            notes="No outcome data available.",
        )

    total_outcomes = int(accuracy_df["total_votes"].max()) if not accuracy_df.empty else 0
    min_trades_met = total_outcomes >= min_trades

    evidence: dict[str, dict] = {}
    raw_scores: dict[str, float] = {}

    for _, row in accuracy_df.iterrows():
        aid = row["agent_id"]
        acc = float(row["accuracy"])
        veto_count = int(row["veto_count"])
        veto_correct = int(row["veto_correct"])
        veto_precision = veto_correct / veto_count if veto_count > 0 else 0.5

        # Veto bonus: agents with accurate vetoes get extra credit
        veto_bonus = 0.0
        if veto_count >= 5:
            veto_bonus = (veto_precision - 0.5) * 0.2

        raw_score = max(0.1, acc + veto_bonus)
        raw_scores[aid] = raw_score

        evidence[aid] = {
            "accuracy": acc,
            "total_votes": int(row["total_votes"]),
            "veto_count": veto_count,
            "veto_correct": veto_correct,
            "veto_precision": round(veto_precision, 4),
            "mean_confidence": round(float(row["mean_confidence"]), 4),
            "raw_score": round(raw_score, 4),
        }

    # Normalize raw scores to sum to 1.0
    total_raw = sum(raw_scores.values()) or 1.0
    ideal_weights = {k: v / total_raw for k, v in raw_scores.items()}

    # Include agents in current_weights that have no outcome data yet
    for aid in current_weights:
        if aid not in ideal_weights:
            ideal_weights[aid] = current_weights.get(aid, 1.0 / max(1, len(current_weights)))

    # Re-normalize
    total_ideal = sum(ideal_weights.values()) or 1.0
    ideal_weights = {k: v / total_ideal for k, v in ideal_weights.items()}

    # Bound the shift per agent — adaptive rate based on evidence strength
    proposed: dict[str, float] = {}
    for aid in current_weights:
        curr = current_weights.get(aid, 0.0)
        ideal = ideal_weights.get(aid, curr)
        delta = ideal - curr
        # Accelerate learning when evidence is strong
        agent_shift = max_daily_shift
        if aid in evidence:
            _acc = evidence[aid]["accuracy"]
            _votes = evidence[aid]["total_votes"]
            if _votes >= min_trades * 3 and (_acc < 0.35 or _acc > 0.70):
                agent_shift = max_daily_shift * 3.0
            elif _votes >= min_trades * 2 and (_acc < 0.40 or _acc > 0.65):
                agent_shift = max_daily_shift * 2.0
        clamped_delta = max(-agent_shift, min(agent_shift, delta))
        proposed[aid] = max(0.01, curr + clamped_delta)

    # Normalize proposed to sum to 1.0
    total_proposed = sum(proposed.values()) or 1.0
    proposed = {k: round(v / total_proposed, 6) for k, v in proposed.items()}

    deltas = {k: round(proposed[k] - current_weights.get(k, 0.0), 6)
              for k in proposed}

    return WeightProposal(
        current_weights=current_weights,
        proposed_weights=proposed,
        deltas=deltas,
        evidence=evidence,
        min_trades_met=min_trades_met,
        regime=regime,
        stable=all(abs(d) < 0.01 for d in deltas.values()),
        notes="Proposal ready for review." if min_trades_met else
              f"Insufficient data ({total_outcomes}/{min_trades} trades).",
    )


def log_weight_proposal(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    proposal: WeightProposal,
) -> int:
    """Write the proposed weights as a shadow_update snapshot."""
    from hogan_bot.swarm_decision.logging import log_weight_snapshot
    return log_weight_snapshot(
        conn,
        ts_ms=int(time.time() * 1000),
        symbol=symbol,
        timeframe=timeframe,
        weights=proposal.proposed_weights,
        regime=proposal.regime,
        source="shadow_update",
        notes=json.dumps({
            "deltas": proposal.deltas,
            "evidence_summary": {
                k: {kk: vv for kk, vv in v.items() if kk in ("accuracy", "veto_precision", "raw_score")}
                for k, v in proposal.evidence.items()
            },
            "min_trades_met": proposal.min_trades_met,
            "stable": proposal.stable,
        }),
    )


def promote_weights(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    proposal: WeightProposal,
) -> int:
    """Promote proposed weights to 'promoted' status.

    This should only be called after operator review.
    """
    from hogan_bot.swarm_decision.logging import log_weight_snapshot
    return log_weight_snapshot(
        conn,
        ts_ms=int(time.time() * 1000),
        symbol=symbol,
        timeframe=timeframe,
        weights=proposal.proposed_weights,
        regime=proposal.regime,
        source="promoted",
        notes=json.dumps({
            "from_weights": proposal.current_weights,
            "deltas": proposal.deltas,
        }),
    )
