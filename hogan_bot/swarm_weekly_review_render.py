"""Markdown and JSON rendering for the Swarm Weekly Review."""
from __future__ import annotations

import json
from typing import Any

from hogan_bot.swarm_weekly_review_types import (
    AgentWeeklyScore,
    ReviewRecommendation,
    ReviewSeverity,
    WeeklyFlag,
    WeeklyReplayCandidate,
    WeeklyReview,
)

_SEV_BADGE = {"healthy": "HEALTHY", "watch": "WATCH", "warning": "WARNING", "critical": "CRITICAL"}

_REC_LABEL = {
    "promote": "Promote", "hold": "Hold", "rollback": "Rollback",
    "tune_thresholds": "Tune Thresholds", "fix_instrumentation": "Fix Instrumentation",
    "quarantine_agent": "Quarantine Agent", "insufficient_data": "Insufficient Data",
}


def render_markdown(
    *,
    week_label: str,
    phase: str,
    symbol: str | None,
    timeframe: str | None,
    severity: ReviewSeverity,
    headline: str,
    recommendation: ReviewRecommendation,
    metrics: dict[str, Any],
    flags: list[WeeklyFlag],
    agent_scores: list[AgentWeeklyScore],
    replay_candidates: list[WeeklyReplayCandidate],
    operator_actions: list[str],
    cursor_actions: list[str],
) -> str:
    L: list[str] = []

    # -- Header --
    L.append(f"# Swarm Weekly Review — {week_label}")
    L.append("")
    L.append(f"**Phase:** {phase}  ")
    if symbol:
        L.append(f"**Symbol:** {symbol}  ")
    if timeframe:
        L.append(f"**Timeframe:** {timeframe}  ")
    L.append(f"**Severity:** {_SEV_BADGE.get(severity, severity)}  ")
    L.append(f"**Recommendation:** {_REC_LABEL.get(recommendation, recommendation)}  ")
    L.append("")
    L.append(f"> {headline}")
    L.append("")

    # -- Executive Summary --
    L.append("## Executive Summary")
    L.append("")
    dec = metrics.get("decision_count", 0)
    vc = metrics.get("veto_count", 0)
    vr = metrics.get("veto_ratio", 0.0)

    improved = []
    regressed = []
    wow_avail = metrics.get("prior_week_available", False)
    if wow_avail:
        if metrics.get("veto_ratio_wow_delta", 0) < -0.05:
            improved.append("veto ratio decreased")
        if metrics.get("would_trade_wow_delta", 0) > 0:
            improved.append("would-trade count increased")
        if metrics.get("veto_ratio_wow_delta", 0) > 0.05:
            regressed.append("veto ratio increased")
        if metrics.get("would_trade_wow_delta", 0) < 0:
            regressed.append("would-trade count decreased")

    L.append(f"- **What improved:** {', '.join(improved) if improved else 'N/A (first week or no prior data)'}")
    L.append(f"- **What regressed:** {', '.join(regressed) if regressed else 'Nothing notable'}")
    L.append(f"- **What to do next:** {operator_actions[0] if operator_actions else 'Continue monitoring'}")
    L.append("")

    # -- Health & Readiness --
    L.append("## Health and Readiness")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    for key, label in [
        ("decision_count", "Decisions"), ("would_trade_count", "Would-Trades"),
        ("veto_count", "Vetoes"), ("veto_ratio", "Veto Ratio"),
        ("distinct_regimes", "Distinct Regimes"),
        ("mean_agreement", "Mean Agreement"), ("mean_entropy", "Mean Entropy"),
        ("baseline_match_count", "Baseline Matches"), ("baseline_miss_count", "Baseline Misses"),
        ("learning_import_error_count", "Import Errors"),
    ]:
        val = metrics.get(key)
        if val is not None:
            L.append(f"| {label} | {val if not isinstance(val, float) else f'{val:.4f}'} |")
    L.append("")

    # -- Flags --
    crit = [f for f in flags if f.level == "critical"]
    warn = [f for f in flags if f.level == "warning"]
    watch = [f for f in flags if f.level == "watch"]

    if crit or warn:
        L.append("## What Is Broken or Blocking")
        L.append("")
        for f in crit:
            L.append(f"- **[CRITICAL]** {f.message}")
        for f in warn:
            L.append(f"- **[WARNING]** {f.message}")
        L.append("")
    if watch:
        L.append("## What to Watch")
        L.append("")
        for f in watch:
            L.append(f"- {f.message}")
        L.append("")
    if not crit and not warn and not watch:
        L.append("## Status")
        L.append("")
        L.append("- No critical or warning flags. Swarm operating normally.")
        L.append("")

    # -- Opportunity Quality --
    opp_mean = metrics.get("opportunity_score_mean")
    opp_top = metrics.get("opportunity_score_top_decile_markout_bps")
    opp_bot = metrics.get("opportunity_score_bottom_decile_markout_bps")
    ntr = metrics.get("no_trade_ratio")
    if opp_mean is not None or ntr is not None:
        L.append("## Opportunity Quality")
        L.append("")
        if opp_mean is not None:
            L.append(f"- Mean opportunity score: **{opp_mean:.4f}**")
        if opp_top is not None and opp_bot is not None:
            L.append(f"- Top decile markout: **{opp_top:+.1f} bps** vs bottom: **{opp_bot:+.1f} bps**")
        if ntr is not None:
            L.append(f"- No-trade ratio: **{ntr:.1%}**")
        tiers = {t: metrics.get(f"tier_{t}_count", 0) for t in "abcdf"}
        if any(tiers.values()):
            L.append(f"- Tiers: A={tiers['a']} B={tiers['b']} C={tiers['c']} D={tiers['d']} F={tiers['f']}")
        L.append("")

    # -- Veto Review --
    veto_reasons = metrics.get("veto_reasons_ranked", [])
    dom_agent = metrics.get("dominant_veto_agent")
    L.append("## Veto Review")
    L.append("")
    L.append(f"- Veto ratio: **{vr:.1%}** ({vc} vetoes / {dec} decisions)")
    if dom_agent:
        L.append(f"- Dominant veto agent: **{dom_agent}** ({metrics.get('dominant_veto_agent_share', 0):.0%} of vetoes)")
    if veto_reasons:
        L.append("- Top reasons:")
        for vr_item in veto_reasons[:5]:
            L.append(f"  - {vr_item['reason']} ({vr_item['count']})")
    cap = metrics.get("veto_bad_trade_capture_rate")
    blk = metrics.get("veto_blocked_winner_rate")
    if cap is not None:
        L.append(f"- Veto captured bad trades: **{cap:.0%}**")
    if blk is not None:
        L.append(f"- Veto blocked winners: **{blk:.0%}**")
    L.append("")

    # -- Agent Leaderboard --
    if agent_scores:
        L.append("## Agent Leaderboard")
        L.append("")
        L.append("| Agent | Decisions | Vetoes | Hold Rate | Confidence | Edge (bps) |")
        L.append("|-------|-----------|--------|-----------|------------|------------|")
        for a in sorted(agent_scores, key=lambda x: x.vetoes):
            conf_s = f"{a.mean_confidence:.2f}" if a.mean_confidence is not None else "—"
            edge_s = f"{a.mean_edge_bps:+.1f}" if a.mean_edge_bps is not None else "—"
            L.append(f"| {a.agent_id} | {a.decisions} | {a.vetoes} | {a.hold_rate:.0%} | {conf_s} | {edge_s} |")
        L.append("")

    # -- Divergence Review --
    L.append("## Divergence Review")
    L.append("")
    bl_match = metrics.get("baseline_match_count", 0)
    bl_miss = metrics.get("baseline_miss_count", 0)
    div_count = metrics.get("divergence_count", 0)
    bl_ratio = metrics.get("baseline_match_ratio")
    L.append(f"- Baseline matches: **{bl_match}** | Misses: **{bl_miss}** | Divergences: **{div_count}**")
    if bl_ratio is not None:
        L.append(f"- Match ratio: **{bl_ratio:.1%}**")
    if bl_miss > 5:
        L.append("- Divergence analytics are suspect due to baseline join failures.")
    L.append("")

    # -- Learning & Drift --
    L.append("## Learning and Drift")
    L.append("")
    L.append(f"- Weight updates: **{metrics.get('weight_update_count', 0)}**")
    ie = metrics.get("learning_import_error_count", 0)
    if ie > 0:
        L.append(f"- Import errors: **{ie}** (fix before trusting drift)")
    L.append(f"- Promotion report available: **{'Yes' if metrics.get('promotion_report_available') else 'No'}**")
    L.append("")

    # -- Week-over-Week --
    L.append("## Week-over-Week Deltas")
    L.append("")
    if wow_avail:
        L.append("| Metric | Delta |")
        L.append("|--------|-------|")
        for key, label in [
            ("decision_count_wow_delta", "Decisions"), ("would_trade_wow_delta", "Would-Trades"),
            ("veto_count_wow_delta", "Vetoes"), ("veto_ratio_wow_delta", "Veto Ratio"),
            ("distinct_regimes_wow_delta", "Regimes"),
        ]:
            v = metrics.get(key)
            if v is not None:
                sign = "+" if v > 0 else ""
                L.append(f"| {label} | {sign}{v if not isinstance(v, float) else f'{v:.4f}'} |")
    else:
        L.append("Week-over-week comparison unavailable — prior review window not found.")
    L.append("")

    # -- Replay Shortlist --
    if replay_candidates:
        L.append("## Replay Review Shortlist")
        L.append("")
        L.append("| # | Category | Decision | Symbol | Time | Reason |")
        L.append("|---|----------|----------|--------|------|--------|")
        for rc in replay_candidates[:20]:
            L.append(f"| {rc.priority} | {rc.category} | {rc.decision_id} | {rc.symbol} | {rc.ts_iso} | {rc.reason} |")
        L.append("")

    # -- Promotion Outlook --
    L.append("## Promotion Outlook")
    L.append("")
    if recommendation == "promote":
        L.append("**Candidate ready.** No blockers. Review manually for promotion.")
    elif recommendation == "hold":
        L.append("**Hold.** Evidence is mixed or incomplete. Continue collecting data.")
    elif recommendation == "fix_instrumentation":
        L.append("**Not ready.** Instrumentation issues must be resolved first.")
    elif recommendation == "tune_thresholds":
        L.append("**Not ready.** Threshold tuning needed to stop over-suppression.")
    elif recommendation == "quarantine_agent":
        L.append("**Not ready.** A dominant agent is collapsing opportunity. Quarantine or downweight.")
    elif recommendation == "rollback":
        L.append("**Rollback.** Recent authority change clearly worsened behavior.")
    else:
        L.append(f"**{_REC_LABEL.get(recommendation, recommendation)}.**")
    if crit:
        L.append("")
        L.append("Blockers:")
        for f in crit:
            L.append(f"- {f.message}")
    L.append("")

    # -- Operator Actions --
    if operator_actions:
        L.append("## Operator Actions This Week")
        L.append("")
        for i, a in enumerate(operator_actions, 1):
            L.append(f"{i}. {a}")
        L.append("")

    # -- Cursor Actions --
    if cursor_actions:
        L.append("## Cursor Actions This Week")
        L.append("")
        for i, a in enumerate(cursor_actions, 1):
            L.append(f"{i}. {a}")
        L.append("")

    return "\n".join(L)


def render_json(review: WeeklyReview) -> str:
    return json.dumps(review.to_dict(), indent=2, default=str)
