"""Swarm Daily Digest — CLI report generator and severity engine.

Converts raw swarm activity into an operator-readable daily report
with deterministic severity classification, operator actions, and
replay candidates.

Usage:
    python -m hogan_bot.swarm_daily_digest \
        --db data/hogan.db --date 2026-03-17 \
        --out-md reports/digests/digest_2026-03-17.md \
        --out-json reports/digests/digest_2026-03-17.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

from hogan_bot.swarm_digest_types import (
    DailyDigest,
    DigestFlag,
    DigestSeverity,
    ReplayCandidate,
)
from hogan_bot.swarm_digest_queries import (
    fetch_digest_window,
    fetch_swarm_counts,
    fetch_opportunity_stats,
    fetch_veto_stats,
    fetch_agent_vote_stats,
    fetch_divergence_stats,
    fetch_learning_drift_stats,
    fetch_replay_candidates,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity and flag rules engine
# ---------------------------------------------------------------------------

def compute_severity_and_flags(
    metrics: dict,
    *,
    stall_decision_min: int = 50,
    stall_would_trade_max: int = 0,
    critical_veto_ratio: float = 0.80,
    warning_veto_ratio: float = 0.60,
    min_regime_coverage: int = 3,
    max_baseline_miss_ratio: float = 0.10,
    max_import_error_count: int = 0,
) -> tuple[DigestSeverity, list[DigestFlag]]:
    """Apply tiered severity rules and return (severity, flags)."""
    flags: list[DigestFlag] = []
    severity: DigestSeverity = "healthy"

    dec = metrics.get("decision_count", 0)
    wt = metrics.get("would_trade_count", 0)
    vr = metrics.get("veto_ratio", 0.0)
    regimes = metrics.get("distinct_regimes", 0)
    mean_agr = metrics.get("mean_agreement")
    mean_ent = metrics.get("mean_entropy")
    bl_miss = metrics.get("baseline_miss_count", 0)
    bl_total = metrics.get("baseline_match_count", 0) + bl_miss
    import_err = metrics.get("learning_import_error_count", 0)
    top_veto_share = metrics.get("top_veto_reason_share", 0.0)
    opp_top = metrics.get("opportunity_score_top_decile_markout_bps")
    opp_bot = metrics.get("opportunity_score_bottom_decile_markout_bps")
    hold_dom = metrics.get("agent_hold_dominance_ratio", 0.0)
    weight_updates = metrics.get("weight_update_count", 0)

    # --- Critical flags ---
    if dec >= stall_decision_min and wt <= stall_would_trade_max:
        flags.append(DigestFlag(
            level="critical",
            code="STALL_NO_WOULD_TRADE",
            message=f"{dec} decisions recorded with {wt} would-trade events.",
            action="Inspect risk_steward thresholds and pre-veto opportunity distribution; swarm may be globally suppressed.",
        ))

    if vr >= critical_veto_ratio:
        flags.append(DigestFlag(
            level="critical",
            code="CRITICAL_VETO_RATIO",
            message=f"Veto ratio {vr:.1%} exceeds critical threshold {critical_veto_ratio:.0%}.",
            action="Audit agent veto thresholds and volatility normalization.",
        ))

    if dec >= stall_decision_min and regimes < 1:
        flags.append(DigestFlag(
            level="critical",
            code="NO_REGIME_LABELS",
            message=f"0 distinct regimes after {dec} decisions.",
            action="Fix regime logging or regime classifier wiring before trusting promotion readiness.",
        ))

    if (dec >= stall_decision_min
            and mean_agr == 1.0 and mean_ent == 0.0
            and wt == 0):
        flags.append(DigestFlag(
            level="critical",
            code="CONTROLLER_COLLAPSE",
            message="mean_agreement=1.0, mean_entropy=0.0, 0 would-trades — controller is collapsing all decisions into unanimous hold.",
            action="Log and display pre-veto consensus separately; review controller collapse-to-hold behavior.",
        ))

    if import_err > max_import_error_count:
        flags.append(DigestFlag(
            level="critical",
            code="LEARNING_IMPORT_ERROR",
            message=f"{import_err} learning/drift import errors detected.",
            action="Fix learning/drift module import path or packaging issue before trusting drift views.",
        ))

    # --- Warning flags ---
    if vr >= warning_veto_ratio and vr < critical_veto_ratio:
        flags.append(DigestFlag(
            level="warning",
            code="HIGH_VETO_RATIO",
            message=f"Veto ratio {vr:.1%} exceeds warning threshold {warning_veto_ratio:.0%}.",
            action="Review agent thresholds; veto rate may be too aggressive.",
        ))

    if 0 < regimes < min_regime_coverage and dec >= stall_decision_min:
        flags.append(DigestFlag(
            level="warning",
            code="LOW_REGIME_COVERAGE",
            message=f"Only {regimes} distinct regimes seen (need {min_regime_coverage}).",
            action="Collect more samples across market regimes before trusting promotion.",
        ))

    if top_veto_share > 0.50:
        flags.append(DigestFlag(
            level="warning",
            code="DOMINANT_VETO_REASON",
            message=f"Top veto reason accounts for {top_veto_share:.0%} of all vetoes.",
            action="Review top veto reason normalization and thresholding; one veto reason is dominating the swarm.",
        ))

    if bl_miss > 5:
        flags.append(DigestFlag(
            level="warning",
            code="BASELINE_JOIN_MISS",
            message=f"{bl_miss} decisions have no matching baseline entry.",
            action="Audit baseline join keys (symbol, timeframe, timestamp rounding/timezone) before using divergence analytics.",
        ))

    if opp_top is not None and opp_bot is not None and opp_top <= opp_bot:
        flags.append(DigestFlag(
            level="warning",
            code="OPP_SCORE_NO_SEPARATION",
            message="Opportunity score top decile does not outperform bottom decile.",
            action="Do not promote thresholds yet; opportunity score is not separating good from bad outcomes.",
        ))

    # --- Watch flags ---
    if dec < stall_decision_min:
        flags.append(DigestFlag(
            level="watch",
            code="UNDERSAMPLED",
            message=f"Only {dec} decisions in window (need {stall_decision_min} for reliable analysis).",
        ))

    if weight_updates == 0:
        flags.append(DigestFlag(
            level="watch",
            code="NO_WEIGHT_UPDATES",
            message="No weight updates recorded in this window.",
        ))

    if opp_top is None and opp_bot is None:
        flags.append(DigestFlag(
            level="watch",
            code="NO_OUTCOMES",
            message="No outcome data available for opportunity calibration.",
        ))

    # --- Determine overall severity ---
    levels = [f.level for f in flags]
    if "critical" in levels:
        severity = "critical"
    elif "warning" in levels:
        severity = "warning"
    elif "watch" in levels:
        severity = "watch"
    else:
        severity = "healthy"

    return severity, flags


# ---------------------------------------------------------------------------
# Operator action generation
# ---------------------------------------------------------------------------

def build_operator_actions(metrics: dict, flags: list[DigestFlag]) -> list[str]:
    """Generate plain-English action items from flags and metrics."""
    actions: list[str] = []
    seen_codes: set[str] = set()

    for f in flags:
        if f.action and f.code not in seen_codes:
            actions.append(f.action)
            seen_codes.add(f.code)

    return actions


# ---------------------------------------------------------------------------
# Headline generation
# ---------------------------------------------------------------------------

def build_headline(severity: DigestSeverity, metrics: dict) -> str:
    """Generate a concise headline for the digest."""
    dec = metrics.get("decision_count", 0)
    wt = metrics.get("would_trade_count", 0)
    vr = metrics.get("veto_ratio", 0.0)

    if dec == 0:
        return "No swarm decisions recorded in this window."

    if severity == "critical":
        if wt == 0 and dec >= 50:
            return (
                "Swarm is alive and logging, but it is currently behaving as a "
                "global veto layer rather than a tradable decision layer. "
                "Review risk_steward thresholds, pre-veto opportunity distribution, "
                "and controller collapse-to-hold behavior before collecting more shadow samples."
            )
        return f"Critical issues detected: {dec} decisions, {wt} would-trades, veto ratio {vr:.0%}."

    if severity == "warning":
        return f"Warnings present: {dec} decisions, {wt} would-trades, veto ratio {vr:.0%}."

    if severity == "watch":
        return f"Collecting data: {dec} decisions so far, {wt} would-trades."

    return f"Healthy: {dec} decisions, {wt} would-trades, veto ratio {vr:.0%}."


# ---------------------------------------------------------------------------
# Full digest builder
# ---------------------------------------------------------------------------

def build_digest(
    conn: sqlite3.Connection,
    *,
    date: str | None = None,
    hours: int = 24,
    symbol: str | None = None,
    timeframe: str | None = None,
    phase: str = "shadow",
    config=None,
) -> DailyDigest:
    """Build a complete DailyDigest from the database."""
    window = fetch_digest_window(conn, date=date, hours=hours, symbol=symbol, timeframe=timeframe)
    start_ms = window["start_ms"]
    end_ms = window["end_ms"]
    resolved_date = window["date"]

    counts = fetch_swarm_counts(conn, start_ms, end_ms, symbol, timeframe)
    opp = fetch_opportunity_stats(conn, start_ms, end_ms, symbol, timeframe)
    veto = fetch_veto_stats(conn, start_ms, end_ms, symbol, timeframe)
    agents = fetch_agent_vote_stats(conn, start_ms, end_ms, symbol, timeframe)
    div = fetch_divergence_stats(conn, start_ms, end_ms, symbol, timeframe)
    drift = fetch_learning_drift_stats(conn, start_ms, end_ms, symbol, timeframe)
    replay_raw = fetch_replay_candidates(conn, start_ms, end_ms, symbol, timeframe,
                                          limit=_cfg(config, "swarm_daily_digest_max_replay_candidates", 12))

    metrics = {**counts, **opp, **veto, **agents, **div, **drift}

    severity, flags = compute_severity_and_flags(
        metrics,
        stall_decision_min=_cfg(config, "swarm_daily_digest_stall_decision_min", 50),
        stall_would_trade_max=_cfg(config, "swarm_daily_digest_stall_would_trade_max", 0),
        critical_veto_ratio=_cfg(config, "swarm_daily_digest_critical_veto_ratio", 0.80),
        warning_veto_ratio=_cfg(config, "swarm_daily_digest_warning_veto_ratio", 0.60),
        min_regime_coverage=_cfg(config, "swarm_daily_digest_min_regime_coverage", 3),
        max_baseline_miss_ratio=_cfg(config, "swarm_daily_digest_max_baseline_miss_ratio", 0.10),
        max_import_error_count=_cfg(config, "swarm_daily_digest_max_import_error_count", 0),
    )

    headline = build_headline(severity, metrics)
    actions = build_operator_actions(metrics, flags)

    replay_candidates = [
        ReplayCandidate(
            decision_id=rc["decision_id"],
            symbol=rc["symbol"],
            ts_iso=rc["ts_iso"],
            reason=rc["reason"],
            priority=rc["priority"],
        )
        for rc in replay_raw
    ]

    from hogan_bot.swarm_digest_render import render_markdown
    summary_md = render_markdown(
        date=resolved_date,
        phase=phase,
        symbol=symbol,
        timeframe=timeframe,
        severity=severity,
        headline=headline,
        metrics=metrics,
        flags=flags,
        replay_candidates=replay_candidates,
        operator_actions=actions,
    )

    return DailyDigest(
        date=resolved_date,
        phase=phase,
        symbol=symbol,
        timeframe=timeframe,
        severity=severity,
        headline=headline,
        metrics=metrics,
        flags=flags,
        replay_candidates=replay_candidates,
        operator_actions=actions,
        summary_md=summary_md,
    )


def _cfg(config, key: str, default):
    if config is None:
        return default
    return getattr(config, key, default)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Swarm Daily Digest")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD), default today UTC")
    parser.add_argument("--hours", type=int, default=24, help="Window size in hours")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--phase", default="shadow")
    parser.add_argument("--out-md", default=None, help="Output Markdown file path")
    parser.add_argument("--out-json", default=None, help="Output JSON file path")
    parser.add_argument("--notify", action="store_true", help="Send notification")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on critical severity")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not os.path.exists(args.db):
        logger.error("Database not found: %s", args.db)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    try:
        digest = build_digest(
            conn,
            date=args.date,
            hours=args.hours,
            symbol=args.symbol,
            timeframe=args.timeframe,
            phase=args.phase,
        )
    finally:
        conn.close()

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(digest.summary_md, encoding="utf-8")
        logger.info("Markdown digest written to %s", args.out_md)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(digest.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("JSON digest written to %s", args.out_json)

    if args.notify:
        try:
            from hogan_bot.notifier import make_notifier
            notifier = make_notifier()
            notifier.notify("swarm_daily_digest", {
                "date": digest.date,
                "severity": digest.severity,
                "headline": digest.headline,
                "top_metrics": {
                    k: digest.metrics.get(k)
                    for k in ["decision_count", "would_trade_count", "veto_ratio"]
                },
                "top_actions": digest.operator_actions[:3],
                "report_path": args.out_md or args.out_json or "",
            })
            logger.info("Notification sent")
        except Exception as exc:
            logger.warning("Notification failed: %s", exc)

    print(f"\n{'='*60}")
    print(f"Swarm Daily Digest — {digest.date}")
    print(f"Severity: {digest.severity.upper()}")
    print(f"Headline: {digest.headline}")
    if digest.operator_actions:
        print(f"\nOperator actions:")
        for i, a in enumerate(digest.operator_actions, 1):
            print(f"  {i}. {a}")
    print(f"{'='*60}\n")

    if args.strict and digest.severity == "critical":
        sys.exit(2)


if __name__ == "__main__":
    main()
