"""Swarm decision replay and explanation utilities.

Provides deterministic renderers that turn stored swarm decision fields
into human-readable narratives and structured replay frames.  These are
pure functions — no LLM calls, no DB access.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Decision story — plain-English summary of a swarm decision
# ---------------------------------------------------------------------------

def render_decision_story(
    decision: dict | pd.Series,
    votes: pd.DataFrame | None = None,
    baseline: dict | pd.Series | None = None,
) -> str:
    """Render a stored swarm decision into a deterministic plain-English story.

    The story is generated entirely from structured DB fields — no LLM call.
    """
    d = dict(decision) if not isinstance(decision, dict) else decision
    lines: list[str] = []

    action = d.get("final_action", "hold")
    conf = d.get("final_conf", d.get("confidence", 0))
    agreement = d.get("agreement", 0)
    entropy = d.get("entropy", 0)
    vetoed = d.get("vetoed", 0)
    mode = d.get("mode", "shadow")
    symbol = d.get("symbol", "?")

    # Opening line
    if vetoed:
        lines.append(f"The swarm **vetoed** trading {symbol}.")
    elif action in ("buy", "sell"):
        lines.append(f"The swarm recommended **{action}** on {symbol} "
                      f"with {conf:.0%} confidence.")
    else:
        lines.append(f"The swarm recommended **hold** on {symbol}.")

    lines.append(f"Mode: {mode} | Agreement: {agreement:.0%} | Entropy: {entropy:.3f}")

    # Block reasons
    block_json = d.get("block_reasons_json", "[]")
    try:
        reasons = json.loads(block_json) if isinstance(block_json, str) else block_json
    except (json.JSONDecodeError, TypeError):
        reasons = []
    if reasons:
        lines.append(f"**Blockers:** {', '.join(reasons[:5])}")

    # Decision detail from decision_json
    detail_json = d.get("decision_json", "{}")
    try:
        detail = json.loads(detail_json) if isinstance(detail_json, str) else detail_json
    except (json.JSONDecodeError, TypeError):
        detail = {}

    regime = detail.get("regime", d.get("regime", "unknown"))
    lines.append(f"Regime: {regime}")

    # Per-agent breakdown
    if votes is not None and not votes.empty:
        lines.append("")
        lines.append("**Agent votes:**")
        for _, v in votes.iterrows():
            agent = v.get("agent_id", "?")
            a_action = v.get("action", "?")
            a_conf = v.get("confidence", 0)
            a_veto = v.get("veto", 0)
            veto_marker = " [VETO]" if a_veto else ""
            a_reasons_raw = v.get("block_reasons_json", "[]")
            try:
                a_reasons = json.loads(a_reasons_raw) if isinstance(a_reasons_raw, str) else a_reasons_raw
            except (json.JSONDecodeError, TypeError):
                a_reasons = []
            reason_str = f" — {a_reasons[0]}" if a_reasons else ""
            lines.append(
                f"  - {agent}: {a_action} ({a_conf:.0%}){veto_marker}{reason_str}"
            )

    # Baseline comparison
    if baseline is not None:
        b = dict(baseline) if not isinstance(baseline, dict) else baseline
        if b:
            bl_action = b.get("final_action", "?")
            bl_conf = b.get("final_confidence", 0)
            lines.append("")
            if bl_action == action:
                lines.append(f"Baseline agreed: {bl_action} ({bl_conf:.0%})")
            else:
                lines.append(
                    f"**Baseline divergence:** baseline={bl_action} ({bl_conf:.0%}), "
                    f"swarm={action} ({conf:.0%})"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Replay frame — structured context for a single decision
# ---------------------------------------------------------------------------

def build_replay_frame(
    decision: dict | pd.Series,
    votes: pd.DataFrame | None = None,
    baseline: dict | pd.Series | None = None,
    outcome: dict | pd.Series | None = None,
    candles: pd.DataFrame | None = None,
    candle_window: int = 20,
) -> dict[str, Any]:
    """Build a structured replay frame for a single swarm decision.

    Returns a dict suitable for rendering in a dashboard or serializing to JSON.
    """
    d = dict(decision) if not isinstance(decision, dict) else decision
    ts_ms = d.get("ts_ms", 0)

    frame: dict[str, Any] = {
        "decision": d,
        "story": render_decision_story(d, votes, baseline),
        "votes": votes.to_dict("records") if votes is not None and not votes.empty else [],
        "baseline": dict(baseline) if baseline is not None else None,
    }

    # Outcome
    if outcome is not None:
        o = dict(outcome) if not isinstance(outcome, dict) else outcome
        frame["outcome"] = o
        frame["outcome_summary"] = _summarize_outcome(o)
    else:
        frame["outcome"] = None
        frame["outcome_summary"] = "Outcome not yet recorded."

    # Candle window around decision
    if candles is not None and not candles.empty and ts_ms:
        mask = candles["ts_ms"] <= ts_ms
        pre = candles[mask].tail(candle_window)
        post = candles[candles["ts_ms"] > ts_ms].head(candle_window)
        frame["candles_before"] = pre.to_dict("records")
        frame["candles_after"] = post.to_dict("records")
    else:
        frame["candles_before"] = []
        frame["candles_after"] = []

    return frame


def _summarize_outcome(outcome: dict) -> str:
    """One-line summary of a decision's forward outcome."""
    fwd = outcome.get("forward_60m_bps")
    mae = outcome.get("mae_bps")
    mfe = outcome.get("mfe_bps")
    veto_correct = outcome.get("was_veto_correct")
    label = outcome.get("outcome_label", "unknown")

    parts: list[str] = [f"Label: {label}"]
    if fwd is not None:
        parts.append(f"60m: {fwd:+.1f}bps")
    if mae is not None:
        parts.append(f"MAE: {mae:.1f}bps")
    if mfe is not None:
        parts.append(f"MFE: {mfe:.1f}bps")
    if veto_correct is not None:
        parts.append(f"Veto correct: {'Yes' if veto_correct else 'No'}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Baseline vs swarm comparison
# ---------------------------------------------------------------------------

def compute_baseline_vs_swarm_delta(
    decisions: pd.DataFrame,
    baseline_decisions: pd.DataFrame,
) -> dict[str, Any]:
    """Compare swarm decisions against baseline on matching timestamps.

    Returns dict with match_count, mismatch_count, agreement_rate,
    swarm_upgrade_count, swarm_downgrade_count.
    """
    if decisions.empty or baseline_decisions.empty:
        return {
            "match_count": 0, "mismatch_count": 0, "agreement_rate": 0.0,
            "compared": 0,
        }

    merged = decisions.merge(
        baseline_decisions[["ts_ms", "symbol", "final_action"]].rename(
            columns={"final_action": "baseline_action"}
        ),
        on=["ts_ms", "symbol"],
        how="inner",
    )

    if merged.empty:
        return {
            "match_count": 0, "mismatch_count": 0, "agreement_rate": 0.0,
            "compared": 0,
        }

    match = (merged["final_action"] == merged["baseline_action"]).sum()
    total = len(merged)

    return {
        "compared": total,
        "match_count": int(match),
        "mismatch_count": total - int(match),
        "agreement_rate": round(match / total, 4) if total else 0.0,
    }
