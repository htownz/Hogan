"""Markdown and JSON rendering for the Swarm Daily Digest."""
from __future__ import annotations

import json
from typing import Any

from hogan_bot.swarm_digest_types import (
    DailyDigest,
    DigestFlag,
    DigestSeverity,
    ReplayCandidate,
)


_SEVERITY_BADGE = {
    "healthy": "HEALTHY",
    "watch": "WATCH",
    "warning": "WARNING",
    "critical": "CRITICAL",
}


def render_markdown(
    *,
    date: str,
    phase: str,
    symbol: str | None,
    timeframe: str | None,
    severity: DigestSeverity,
    headline: str,
    metrics: dict[str, Any],
    flags: list[DigestFlag],
    replay_candidates: list[ReplayCandidate],
    operator_actions: list[str],
) -> str:
    """Render a complete Markdown digest report."""
    lines: list[str] = []

    # -- Header --
    lines.append(f"# Swarm Daily Digest — {date}")
    lines.append("")
    lines.append(f"**Phase:** {phase}  ")
    if symbol:
        lines.append(f"**Symbol:** {symbol}  ")
    if timeframe:
        lines.append(f"**Timeframe:** {timeframe}  ")
    lines.append(f"**Severity:** {_SEVERITY_BADGE.get(severity, severity)}  ")
    lines.append("")
    lines.append(f"> {headline}")
    lines.append("")

    # -- Executive summary --
    lines.append("## Executive Summary")
    lines.append("")
    dec = metrics.get("decision_count", 0)
    wt = metrics.get("would_trade_count", 0)
    veto = metrics.get("veto_count", 0)
    vr = metrics.get("veto_ratio", 0.0)
    regimes = metrics.get("distinct_regimes", 0)
    lines.append(
        f"The swarm recorded **{dec}** decisions with **{wt}** would-trade signals "
        f"and **{veto}** vetoed decisions (veto ratio **{vr:.1%}**). "
        f"**{regimes}** distinct market regimes were observed."
    )
    lines.append("")

    # -- Key metrics --
    lines.append("## Key Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    key_fields = [
        ("decision_count", "Decisions"),
        ("would_trade_count", "Would-Trades"),
        ("veto_count", "Vetoes"),
        ("veto_ratio", "Veto Ratio"),
        ("distinct_regimes", "Distinct Regimes"),
        ("mean_agreement", "Mean Agreement"),
        ("mean_entropy", "Mean Entropy"),
        ("mean_confidence", "Mean Confidence"),
        ("agent_hold_dominance_ratio", "Agent Hold Dominance"),
    ]
    for key, label in key_fields:
        val = metrics.get(key)
        if val is not None:
            if isinstance(val, float):
                lines.append(f"| {label} | {val:.4f} |")
            else:
                lines.append(f"| {label} | {val} |")
    lines.append("")

    # -- Flags --
    critical_flags = [f for f in flags if f.level == "critical"]
    warning_flags = [f for f in flags if f.level == "warning"]
    watch_flags = [f for f in flags if f.level == "watch"]

    if critical_flags or warning_flags:
        lines.append("## What Is Broken or Blocking")
        lines.append("")
        for f in critical_flags:
            lines.append(f"- **[CRITICAL]** {f.message}")
        for f in warning_flags:
            lines.append(f"- **[WARNING]** {f.message}")
        lines.append("")

    if watch_flags:
        lines.append("## What to Watch")
        lines.append("")
        for f in watch_flags:
            lines.append(f"- {f.message}")
        lines.append("")

    if not critical_flags and not warning_flags and not watch_flags:
        lines.append("## What Improved")
        lines.append("")
        lines.append("- No critical or warning flags raised. Swarm operating normally.")
        lines.append("")

    # -- Top veto reasons --
    veto_reasons = metrics.get("veto_reasons_ranked", [])
    if veto_reasons:
        lines.append("## Top Veto Reasons")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("|--------|-------|")
        for vr_item in veto_reasons[:5]:
            lines.append(f"| {vr_item['reason']} | {vr_item['count']} |")
        lines.append("")

    # -- Opportunity quality --
    opp_mean = metrics.get("opportunity_score_mean")
    opp_top = metrics.get("opportunity_score_top_decile_markout_bps")
    opp_bot = metrics.get("opportunity_score_bottom_decile_markout_bps")
    if opp_mean is not None or opp_top is not None:
        lines.append("## Opportunity Quality")
        lines.append("")
        if opp_mean is not None:
            lines.append(f"- Mean opportunity score: **{opp_mean:.4f}**")
        if opp_top is not None and opp_bot is not None:
            lines.append(f"- Top decile markout: **{opp_top:+.1f} bps** vs bottom decile: **{opp_bot:+.1f} bps**")
            if opp_top > opp_bot:
                lines.append("- Score shows separation between good and bad outcomes.")
            else:
                lines.append("- Score does NOT separate good from bad outcomes.")
        tiers = ["tier_a_count", "tier_b_count", "tier_c_count", "tier_d_count", "tier_f_count"]
        tier_vals = {t: metrics.get(t, 0) for t in tiers}
        if any(tier_vals.values()):
            lines.append(f"- Tiers: A={tier_vals['tier_a_count']} B={tier_vals['tier_b_count']} "
                         f"C={tier_vals['tier_c_count']} D={tier_vals['tier_d_count']} F={tier_vals['tier_f_count']}")
        lines.append("")

    # -- Baseline divergence --
    bl_match = metrics.get("baseline_match_count", 0)
    bl_miss = metrics.get("baseline_miss_count", 0)
    div_count = metrics.get("divergence_count", 0)
    bl_ratio = metrics.get("baseline_match_ratio")
    lines.append("## Baseline Divergence")
    lines.append("")
    lines.append(f"- Baseline matches: **{bl_match}** | Misses: **{bl_miss}** | Divergences: **{div_count}**")
    if bl_ratio is not None:
        lines.append(f"- Match ratio: **{bl_ratio:.1%}**")
    if bl_miss > 5:
        lines.append("- Divergence analytics are suspect due to high baseline miss count.")
    lines.append("")

    # -- Learning & drift --
    wu = metrics.get("weight_update_count", 0)
    ie = metrics.get("learning_import_error_count", 0)
    lines.append("## Learning & Drift")
    lines.append("")
    lines.append(f"- Weight updates in window: **{wu}**")
    if ie > 0:
        lines.append(f"- Import errors: **{ie}** (fix before trusting drift views)")
    lines.append("")

    # -- Replay shortlist --
    if replay_candidates:
        lines.append("## Replay Shortlist")
        lines.append("")
        lines.append("| Priority | Decision | Symbol | Time | Reason |")
        lines.append("|----------|----------|--------|------|--------|")
        for rc in replay_candidates[:12]:
            lines.append(f"| {rc.priority} | {rc.decision_id} | {rc.symbol} | {rc.ts_iso} | {rc.reason} |")
        lines.append("")

    # -- Operator actions --
    if operator_actions:
        lines.append("## Operator Actions Today")
        lines.append("")
        for i, a in enumerate(operator_actions, 1):
            lines.append(f"{i}. {a}")
        lines.append("")

    # -- Promotion note --
    lines.append("## Promotion Note")
    lines.append("")
    if severity == "critical":
        lines.append("**Not ready.** Critical issues must be resolved before promotion can be considered.")
    elif severity == "warning":
        lines.append("**Collecting.** Warnings present; review and resolve before advancing.")
    elif severity == "watch":
        lines.append("**Collecting.** Insufficient data for promotion assessment.")
    else:
        lines.append("**Candidate improving.** Review manually for promotion readiness.")
    lines.append("")

    return "\n".join(lines)


def render_json(digest: DailyDigest) -> str:
    """Render the digest as a JSON string."""
    return json.dumps(digest.to_dict(), indent=2, default=str)
