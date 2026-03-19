"""Swarm Weekly Review — CLI report generator and severity/recommendation engine.

Turns 7 days of swarm behavior into an operator-grade review packet with
deterministic severity classification, promotion recommendations,
operator + Cursor action lists, and categorized replay candidates.

Usage:
    python -m hogan_bot.swarm_weekly_review \
        --db data/hogan.db --week-end 2026-03-17 \
        --out-md reports/weekly/review_2026-W12.md \
        --out-json reports/weekly/review_2026-W12.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

from hogan_bot.swarm_weekly_review_types import (
    AgentWeeklyScore,
    ReviewRecommendation,
    ReviewSeverity,
    WeeklyFlag,
    WeeklyReplayCandidate,
    WeeklyReview,
)
from hogan_bot.swarm_weekly_review_queries import (
    fetch_review_window,
    fetch_week_over_week_stats,
    fetch_weekly_agent_scores,
    fetch_weekly_divergence_stats,
    fetch_weekly_learning_stats,
    fetch_weekly_opportunity_stats,
    fetch_weekly_regime_stats,
    fetch_weekly_replay_candidates,
    fetch_weekly_swarm_counts,
    fetch_weekly_veto_stats,
    _ts_range,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity and flag rules
# ---------------------------------------------------------------------------

def compute_severity_and_flags(
    metrics: dict,
    *,
    stall_min: int = 50,
    critical_veto_ratio: float = 0.80,
    warning_veto_ratio: float = 0.60,
    min_regime_coverage: int = 3,
    agent_dominance_warn: float = 0.70,
    baseline_miss_warn: float = 0.10,
    min_decisions: int = 300,
    min_would_trade: int = 100,
) -> tuple[ReviewSeverity, list[WeeklyFlag]]:
    flags: list[WeeklyFlag] = []

    dec = metrics.get("decision_count", 0)
    wt = metrics.get("would_trade_count", 0)
    vr = metrics.get("veto_ratio", 0.0)
    regimes = metrics.get("distinct_regimes", 0)
    bl_miss = metrics.get("baseline_miss_count", 0)
    bl_match = metrics.get("baseline_match_count", 0)
    bl_total = bl_match + bl_miss
    bl_ratio = bl_match / bl_total if bl_total > 0 else None
    import_err = metrics.get("learning_import_error_count", 0)
    dom_agent_share = metrics.get("dominant_veto_agent_share", 0.0)
    dom_agent = metrics.get("dominant_veto_agent")
    mean_agr = metrics.get("mean_agreement")
    mean_ent = metrics.get("mean_entropy")
    opp_top = metrics.get("opportunity_score_top_decile_markout_bps")
    opp_bot = metrics.get("opportunity_score_bottom_decile_markout_bps")
    regime_missing = metrics.get("regime_missing", False)

    # --- Critical ---
    if dec >= stall_min and wt == 0:
        flags.append(WeeklyFlag(
            level="critical", code="STALL_ZERO_WOULD_TRADE",
            message=f"Swarm is active ({dec} decisions) but not generating tradable opportunities.",
            action="Audit risk_steward thresholds, pre-veto consensus, and opportunity routing.",
        ))

    if vr >= critical_veto_ratio:
        flags.append(WeeklyFlag(
            level="critical", code="CRITICAL_VETO_RATIO",
            message=f"Weekly veto ratio {vr:.1%} exceeds critical threshold.",
            action="Audit agent veto thresholds and volatility normalization.",
        ))

    if dec >= stall_min and regimes == 0:
        flags.append(WeeklyFlag(
            level="critical", code="REGIME_LOGGING_MISSING",
            message=f"0 distinct regimes after {dec} decisions.",
            action="Fix regime logging or regime classifier wiring before trusting promotion readiness.",
        ))

    if bl_ratio is not None and bl_ratio < 0.90 and bl_total >= 20:
        flags.append(WeeklyFlag(
            level="critical", code="BASELINE_JOIN_FAILURE",
            message=f"Baseline match ratio {bl_ratio:.1%} — {bl_miss} decisions have no matching baseline.",
            action="Audit timestamp rounding, symbol normalization, and timeframe join keys.",
        ))

    if import_err > 0:
        flags.append(WeeklyFlag(
            level="critical", code="LEARNING_PANEL_BROKEN",
            message=f"{import_err} learning/drift import errors detected.",
            action="Fix import path / package execution context before trusting learning metrics.",
        ))

    if dom_agent and dom_agent_share >= agent_dominance_warn:
        flags.append(WeeklyFlag(
            level="critical" if dom_agent_share >= 0.85 else "warning",
            code="DOMINANT_VETO_AGENT",
            message=f"{dom_agent} accounts for {dom_agent_share:.0%} of all vetoes.",
            action=f"Review blocker thresholds and normalization for {dom_agent}.",
        ))

    if (dec >= stall_min and mean_agr == 1.0 and mean_ent == 0.0 and wt == 0):
        flags.append(WeeklyFlag(
            level="critical", code="PRE_VETO_CONSENSUS_MISSING",
            message="Post-veto holds dominate with agreement=1.0, entropy=0.0 — pre-veto consensus not visible.",
            action="Log and render pre-veto agreement, entropy, and confidence separately.",
        ))

    # --- Warning ---
    if warning_veto_ratio <= vr < critical_veto_ratio:
        flags.append(WeeklyFlag(
            level="warning", code="HIGH_VETO_RATIO",
            message=f"Weekly veto ratio {vr:.1%} exceeds warning threshold.",
            action="Review agent thresholds; veto rate may be too aggressive.",
        ))

    if 0 < regimes < min_regime_coverage and dec >= stall_min:
        flags.append(WeeklyFlag(
            level="warning", code="LOW_REGIME_COVERAGE",
            message=f"Only {regimes} distinct regimes (need {min_regime_coverage}).",
            action="Collect more samples across market regimes before trusting promotion.",
        ))

    if dec >= min_decisions and wt < min_would_trade and wt > 0:
        flags.append(WeeklyFlag(
            level="warning", code="LOW_WOULD_TRADE_VOLUME",
            message=f"Only {wt} would-trades from {dec} decisions.",
            action="Investigate excessive hold/veto ratio suppressing would-trade generation.",
        ))

    if opp_top is not None and opp_bot is not None and opp_top <= opp_bot:
        flags.append(WeeklyFlag(
            level="warning", code="OPP_SCORE_NO_SEPARATION",
            message="Opportunity score top decile does not outperform bottom decile.",
            action="Do not promote thresholds; score is not separating outcomes.",
        ))

    if bl_miss > 5 and (bl_ratio is None or bl_ratio >= 0.90):
        flags.append(WeeklyFlag(
            level="warning", code="BASELINE_JOIN_MISS",
            message=f"{bl_miss} decisions lack baseline matches.",
            action="Audit baseline join keys before trusting divergence analytics.",
        ))

    # --- Watch ---
    if dec < stall_min:
        flags.append(WeeklyFlag(level="watch", code="UNDERSAMPLED",
                                message=f"Only {dec} decisions — need {stall_min} for reliable analysis."))

    if metrics.get("weight_update_count", 0) == 0:
        flags.append(WeeklyFlag(level="watch", code="NO_WEIGHT_UPDATES",
                                message="No weight updates in this window."))

    if opp_top is None and opp_bot is None:
        flags.append(WeeklyFlag(level="watch", code="NO_OUTCOMES",
                                message="No outcome data for opportunity calibration."))

    levels = [f.level for f in flags]
    if "critical" in levels:
        severity: ReviewSeverity = "critical"
    elif "warning" in levels:
        severity = "warning"
    elif "watch" in levels:
        severity = "watch"
    else:
        severity = "healthy"
    return severity, flags


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def compute_recommendation(
    severity: ReviewSeverity,
    flags: list[WeeklyFlag],
    metrics: dict,
) -> ReviewRecommendation:
    codes = {f.code for f in flags}

    if "LEARNING_PANEL_BROKEN" in codes or "BASELINE_JOIN_FAILURE" in codes or "REGIME_LOGGING_MISSING" in codes:
        return "fix_instrumentation"

    if "STALL_ZERO_WOULD_TRADE" in codes or "CRITICAL_VETO_RATIO" in codes:
        return "tune_thresholds"

    if "DOMINANT_VETO_AGENT" in codes:
        share = metrics.get("dominant_veto_agent_share", 0.0)
        if share >= 0.85:
            return "quarantine_agent"
        return "tune_thresholds"

    if severity == "critical":
        return "fix_instrumentation"

    dec = metrics.get("decision_count", 0)
    if dec < 50:
        return "insufficient_data"

    if severity == "healthy":
        wt = metrics.get("would_trade_count", 0)
        if wt > 0 and not any(f.level in ("critical", "warning") for f in flags):
            return "promote"
        return "hold"

    return "hold"


# ---------------------------------------------------------------------------
# Action generation
# ---------------------------------------------------------------------------

def build_operator_actions(flags: list[WeeklyFlag], metrics: dict) -> list[str]:
    actions: list[str] = []
    seen: set[str] = set()
    for f in flags:
        if f.action and f.code not in seen:
            actions.append(f.action)
            seen.add(f.code)

    if metrics.get("would_trade_count", 0) == 0 and metrics.get("decision_count", 0) >= 50:
        actions.append("Replay the top 10 vetoed decisions to understand blocking patterns.")
    if metrics.get("distinct_regimes", 0) == 0:
        actions.append("Verify regime labels are being logged end-to-end.")

    return actions


def build_cursor_actions(flags: list[WeeklyFlag], metrics: dict) -> list[str]:
    actions: list[str] = []
    codes = {f.code for f in flags}

    if "LEARNING_PANEL_BROKEN" in codes:
        actions.append("Fix learning/drift module import path.")
    if "BASELINE_JOIN_FAILURE" in codes or "BASELINE_JOIN_MISS" in codes:
        actions.append("Fix baseline join key matching (timestamp rounding, symbol normalization).")
    if "REGIME_LOGGING_MISSING" in codes:
        actions.append("Wire regime labels into swarm_decisions.regime column.")
    if "STALL_ZERO_WOULD_TRADE" in codes or "CRITICAL_VETO_RATIO" in codes:
        actions.append("Audit risk_steward volatility normalization and veto thresholds.")
    if "PRE_VETO_CONSENSUS_MISSING" in codes:
        actions.append("Add pre-veto consensus logging (agreement/entropy before veto application).")
    if "DOMINANT_VETO_AGENT" in codes:
        agent = metrics.get("dominant_veto_agent", "unknown")
        actions.append(f"Review and potentially downweight {agent}.")

    return actions


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

def build_headline(severity: ReviewSeverity, rec: ReviewRecommendation, metrics: dict) -> str:
    dec = metrics.get("decision_count", 0)
    wt = metrics.get("would_trade_count", 0)
    vr = metrics.get("veto_ratio", 0.0)

    if dec == 0:
        return "No swarm decisions recorded this week."

    if rec == "fix_instrumentation":
        return "Instrumentation issues must be fixed before the swarm can be evaluated."
    if rec == "tune_thresholds":
        return f"Swarm is active but suppressed by veto dominance (veto ratio {vr:.0%}, {wt} would-trades). Threshold tuning required."
    if rec == "quarantine_agent":
        agent = metrics.get("dominant_veto_agent", "an agent")
        return f"Single agent ({agent}) is dominating and collapsing opportunity. Consider quarantine."
    if rec == "promote":
        return f"Healthy week: {dec} decisions, {wt} would-trades, veto ratio {vr:.0%}. Candidate for promotion."
    if rec == "insufficient_data":
        return f"Insufficient data ({dec} decisions). Continue collecting."

    return f"Mixed week: {dec} decisions, {wt} would-trades, veto ratio {vr:.0%}. Recommendation: {rec}."


# ---------------------------------------------------------------------------
# Full review builder
# ---------------------------------------------------------------------------

def build_weekly_review(
    conn: sqlite3.Connection,
    *,
    week_end: str | None = None,
    days: int = 7,
    symbol: str | None = None,
    timeframe: str | None = None,
    phase: str = "shadow",
    config=None,
    include_previous_week: bool = True,
) -> WeeklyReview:
    window = fetch_review_window(conn, week_end=week_end, days=days, symbol=symbol, timeframe=timeframe)
    start_ms, end_ms = window["start_ms"], window["end_ms"]
    week_label = window["week_label"]

    counts = fetch_weekly_swarm_counts(conn, start_ms, end_ms, symbol, timeframe)
    opp = fetch_weekly_opportunity_stats(conn, start_ms, end_ms, symbol, timeframe)
    veto = fetch_weekly_veto_stats(conn, start_ms, end_ms, symbol, timeframe)
    agent_rows = fetch_weekly_agent_scores(conn, start_ms, end_ms, symbol, timeframe)
    div = fetch_weekly_divergence_stats(conn, start_ms, end_ms, symbol, timeframe)
    learn = fetch_weekly_learning_stats(conn, start_ms, end_ms, symbol, timeframe)
    regime = fetch_weekly_regime_stats(conn, start_ms, end_ms, symbol, timeframe)

    wow = {}
    if include_previous_week:
        prev_end_ms = start_ms
        prev_start_ms = prev_end_ms - (end_ms - start_ms)
        wow = fetch_week_over_week_stats(conn, start_ms, end_ms, prev_start_ms, prev_end_ms, symbol, timeframe)

    replay_raw = fetch_weekly_replay_candidates(
        conn, start_ms, end_ms, symbol, timeframe,
        limit=_cfg(config, "swarm_weekly_review_max_replay_candidates", 20),
    )

    metrics = {**counts, **opp, **veto, **div, **learn, **regime, **wow}

    no_trade_ratio = None
    if counts["decision_count"] > 0:
        no_trade_ratio = round(1.0 - counts["would_trade_count"] / counts["decision_count"], 4)
    metrics["no_trade_ratio"] = no_trade_ratio

    severity, flags = compute_severity_and_flags(
        metrics,
        stall_min=_cfg(config, "swarm_weekly_review_stall_zero_trade_decision_min", 50),
        critical_veto_ratio=_cfg(config, "swarm_weekly_review_critical_veto_ratio", 0.80),
        warning_veto_ratio=_cfg(config, "swarm_weekly_review_warning_veto_ratio", 0.60),
        min_regime_coverage=_cfg(config, "swarm_weekly_review_min_regime_coverage", 3),
        agent_dominance_warn=_cfg(config, "swarm_weekly_review_agent_dominance_ratio_warn", 0.70),
        baseline_miss_warn=_cfg(config, "swarm_weekly_review_baseline_miss_ratio_warn", 0.10),
        min_decisions=_cfg(config, "swarm_weekly_review_min_decisions", 300),
        min_would_trade=_cfg(config, "swarm_weekly_review_min_would_trade", 100),
    )

    rec = compute_recommendation(severity, flags, metrics)
    headline = build_headline(severity, rec, metrics)
    op_actions = build_operator_actions(flags, metrics)
    cur_actions = build_cursor_actions(flags, metrics)

    agent_scores = [
        AgentWeeklyScore(
            agent_id=a["agent_id"], decisions=a["decisions"], vetoes=a["vetoes"],
            hold_rate=a["hold_rate"], mean_confidence=a["mean_confidence"],
            mean_edge_bps=a["mean_edge_bps"], contribution_score=None,
        )
        for a in agent_rows
    ]

    replay_candidates = [
        WeeklyReplayCandidate(
            decision_id=rc["decision_id"], symbol=rc["symbol"], ts_iso=rc["ts_iso"],
            category=rc["category"], reason=rc["reason"], priority=rc["priority"],
        )
        for rc in replay_raw
    ]

    from hogan_bot.swarm_weekly_review_render import render_markdown
    summary_md = render_markdown(
        week_label=week_label, phase=phase, symbol=symbol, timeframe=timeframe,
        severity=severity, headline=headline, recommendation=rec,
        metrics=metrics, flags=flags, agent_scores=agent_scores,
        replay_candidates=replay_candidates,
        operator_actions=op_actions, cursor_actions=cur_actions,
    )

    return WeeklyReview(
        week_label=week_label, phase=phase, symbol=symbol, timeframe=timeframe,
        severity=severity, headline=headline, metrics=metrics,
        flags=flags, agent_scores=agent_scores, replay_candidates=replay_candidates,
        operator_actions=op_actions, cursor_actions=cur_actions,
        recommendation=rec, summary_md=summary_md,
    )


def _cfg(config, key: str, default):
    if config is None:
        return default
    return getattr(config, key, default)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Swarm Weekly Review")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--week-end", default=None, help="End date (YYYY-MM-DD), default today UTC")
    parser.add_argument("--days", type=int, default=7, help="Window size in days")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--phase", default="shadow")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--include-previous-week", action="store_true", default=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not os.path.exists(args.db):
        logger.error("Database not found: %s", args.db)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    try:
        review = build_weekly_review(
            conn, week_end=args.week_end, days=args.days,
            symbol=args.symbol, timeframe=args.timeframe, phase=args.phase,
            include_previous_week=args.include_previous_week,
        )
    finally:
        conn.close()

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(review.summary_md, encoding="utf-8")
        logger.info("Markdown review written to %s", args.out_md)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(review.to_dict(), indent=2, default=str), encoding="utf-8",
        )
        logger.info("JSON review written to %s", args.out_json)

    if args.notify:
        try:
            from hogan_bot.notifier import make_notifier
            notifier = make_notifier()
            notifier.notify("swarm_weekly_review", {
                "week_label": review.week_label, "severity": review.severity,
                "headline": review.headline, "recommendation": review.recommendation,
                "top_metrics": {
                    k: review.metrics.get(k)
                    for k in ["decision_count", "would_trade_count", "veto_ratio",
                              "dominant_veto_agent", "dominant_veto_agent_share"]
                },
                "top_actions": review.operator_actions[:3],
                "report_path": args.out_md or args.out_json or "",
            })
            logger.info("Notification sent")
        except Exception as exc:
            logger.warning("Notification failed: %s", exc)

    print(f"\n{'='*60}")
    print(f"Swarm Weekly Review — {review.week_label}")
    print(f"Severity: {review.severity.upper()}")
    print(f"Recommendation: {review.recommendation}")
    print(f"Headline: {review.headline}")
    if review.operator_actions:
        print(f"\nOperator actions:")
        for i, a in enumerate(review.operator_actions, 1):
            print(f"  {i}. {a}")
    if review.cursor_actions:
        print(f"\nCursor actions:")
        for i, a in enumerate(review.cursor_actions, 1):
            print(f"  {i}. {a}")
    print(f"{'='*60}\n")

    if args.strict and review.severity == "critical":
        sys.exit(2)


if __name__ == "__main__":
    main()
