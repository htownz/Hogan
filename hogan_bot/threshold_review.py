"""Threshold review CLI — per-agent threshold evaluation.

Evaluates an agent's veto behavior over a time window, computes pre-veto
vs post-veto tradeability, and recommends tuning actions.

Usage:
    python -m hogan_bot.threshold_review \
        --db data/hogan.db --window-hours 24 \
        --agent-id risk_steward_v1 \
        --out-json reports/risk_steward_review.json \
        --out-md reports/risk_steward_review.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hogan_bot.threshold_types import StallAlert, ThresholdReviewResult, TuningRecommendation
from hogan_bot.stall_detection import evaluate_stall_state

logger = logging.getLogger(__name__)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone()
    return bool(r and r[0])


def review_agent(
    conn: sqlite3.Connection,
    agent_id: str,
    window_hours: int = 24,
    end_ts_ms: int | None = None,
) -> ThresholdReviewResult:
    if end_ts_ms is None:
        end_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ts_ms = end_ts_ms - window_hours * 3600 * 1000

    decision_count = 0
    would_trade_count = 0
    veto_count = 0
    distinct_regimes = 0
    pre_veto_wt = 0
    post_veto_wt = 0
    pre_veto_agr_sum = 0.0
    post_veto_agr_sum = 0.0
    pre_veto_conf_sum = 0.0
    post_veto_conf_sum = 0.0
    pre_veto_n = 0
    post_veto_n = 0

    if _table_exists(conn, "swarm_decisions"):
        rows = conn.execute(
            """SELECT final_action, vetoed, agreement, final_conf, pre_veto_action,
                      pre_veto_agreement, pre_veto_confidence, regime
               FROM swarm_decisions WHERE ts_ms BETWEEN ? AND ?""",
            (start_ts_ms, end_ts_ms),
        ).fetchall()
        decision_count = len(rows)
        regimes_set: set[str] = set()
        for r in rows:
            action, vetoed, agr, conf, pv_act, pv_agr, pv_conf, regime = r
            if action in ("buy", "sell") and not vetoed:
                would_trade_count += 1
            if vetoed:
                veto_count += 1
            if regime:
                regimes_set.add(regime)
            if pv_act and pv_act in ("buy", "sell"):
                pre_veto_wt += 1
            if pv_agr is not None:
                pre_veto_agr_sum += pv_agr
                pre_veto_conf_sum += (pv_conf or 0.0)
                pre_veto_n += 1
            post_veto_agr_sum += agr
            post_veto_conf_sum += conf
            post_veto_n += 1
        distinct_regimes = len(regimes_set)

    veto_ratio = veto_count / decision_count if decision_count > 0 else 0.0

    # Agent-specific veto stats
    top_veto_reasons: list[dict] = []
    dom_agent = None
    dom_agent_share = 0.0
    if _table_exists(conn, "swarm_agent_votes"):
        agent_rows = conn.execute(
            """SELECT sav.agent_id, sav.block_reasons_json
               FROM swarm_agent_votes sav
               JOIN swarm_decisions sd ON sav.decision_id = sd.id
               WHERE sav.veto = 1 AND sd.ts_ms BETWEEN ? AND ?""",
            (start_ts_ms, end_ts_ms),
        ).fetchall()
        reason_counts: dict[str, int] = {}
        agent_veto_counts: dict[str, int] = {}
        for aid, br_json in agent_rows:
            agent_veto_counts[aid] = agent_veto_counts.get(aid, 0) + 1
            try:
                reasons = json.loads(br_json) if br_json else []
            except (json.JSONDecodeError, TypeError):
                reasons = []
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        ranked = sorted(reason_counts.items(), key=lambda x: -x[1])
        top_veto_reasons = [{"reason": r, "count": c} for r, c in ranked[:10]]
        if agent_veto_counts:
            total_vetos = sum(agent_veto_counts.values())
            dom_pair = max(agent_veto_counts.items(), key=lambda x: x[1])
            dom_agent = dom_pair[0]
            dom_agent_share = dom_pair[1] / total_vetos if total_vetos > 0 else 0.0

    # Stall alerts
    stall_metrics = {
        "decision_count": decision_count,
        "would_trade_count": would_trade_count,
        "veto_ratio": veto_ratio,
        "distinct_regimes": distinct_regimes,
        "top_veto_agent_share": dom_agent_share,
        "dominant_veto_agent": dom_agent or "",
    }
    stall_alerts = evaluate_stall_state(stall_metrics)

    # Recommendation
    rec = _compute_recommendation(
        agent_id=agent_id,
        decision_count=decision_count,
        would_trade_count=would_trade_count,
        veto_ratio=veto_ratio,
        dom_agent=dom_agent,
        dom_agent_share=dom_agent_share,
        pre_veto_wt=pre_veto_wt,
        stall_alerts=stall_alerts,
    )

    return ThresholdReviewResult(
        agent_id=agent_id,
        window_hours=window_hours,
        decision_count=decision_count,
        would_trade_count=would_trade_count,
        veto_ratio=round(veto_ratio, 4),
        top_veto_reasons=top_veto_reasons,
        dominant_veto_agent=dom_agent,
        dominant_veto_agent_share=round(dom_agent_share, 4),
        pre_veto_would_trade_count=pre_veto_wt,
        post_veto_would_trade_count=would_trade_count,
        pre_veto_agreement_mean=round(pre_veto_agr_sum / pre_veto_n, 4) if pre_veto_n > 0 else None,
        post_veto_agreement_mean=round(post_veto_agr_sum / post_veto_n, 4) if post_veto_n > 0 else None,
        pre_veto_confidence_mean=round(pre_veto_conf_sum / pre_veto_n, 4) if pre_veto_n > 0 else None,
        post_veto_confidence_mean=round(post_veto_conf_sum / post_veto_n, 4) if post_veto_n > 0 else None,
        recommendation=rec,
        stall_alerts=stall_alerts,
    )


def _compute_recommendation(
    *,
    agent_id: str,
    decision_count: int,
    would_trade_count: int,
    veto_ratio: float,
    dom_agent: str | None,
    dom_agent_share: float,
    pre_veto_wt: int,
    stall_alerts: list[StallAlert],
) -> TuningRecommendation:
    codes = {a.code for a in stall_alerts}

    if dom_agent == agent_id and dom_agent_share >= 0.80 and would_trade_count == 0:
        return "quarantine"

    if dom_agent == agent_id and dom_agent_share >= 0.60 and would_trade_count == 0:
        return "disable_veto_only"

    if "CRITICAL_STALL" in codes and pre_veto_wt > 0:
        return "relax_thresholds"

    if "CRITICAL_STALL" in codes or "SEVERE_STALL" in codes:
        return "disable_veto_only"

    if "OVER_VETO_WARNING" in codes and dom_agent == agent_id:
        return "relax_thresholds"

    if "DOMINANT_VETO_AGENT" in codes and dom_agent == agent_id:
        return "advisory_only"

    return "hold"


def render_review_json(result: ThresholdReviewResult) -> str:
    d = {
        "agent_id": result.agent_id,
        "window_hours": result.window_hours,
        "decision_count": result.decision_count,
        "would_trade_count": result.would_trade_count,
        "veto_ratio": result.veto_ratio,
        "top_veto_reasons": result.top_veto_reasons,
        "dominant_veto_agent": result.dominant_veto_agent,
        "dominant_veto_agent_share": result.dominant_veto_agent_share,
        "pre_veto_would_trade_count": result.pre_veto_would_trade_count,
        "post_veto_would_trade_count": result.post_veto_would_trade_count,
        "pre_veto_agreement_mean": result.pre_veto_agreement_mean,
        "post_veto_agreement_mean": result.post_veto_agreement_mean,
        "pre_veto_confidence_mean": result.pre_veto_confidence_mean,
        "post_veto_confidence_mean": result.post_veto_confidence_mean,
        "recommendation": result.recommendation,
        "stall_alerts": [
            {"code": a.code, "severity": a.severity, "metric_name": a.metric_name,
             "actual": a.actual, "threshold": a.threshold, "notes": a.notes}
            for a in result.stall_alerts
        ],
    }
    return json.dumps(d, indent=2)


def render_review_md(result: ThresholdReviewResult) -> str:
    L: list[str] = []
    L.append(f"# Threshold Review — {result.agent_id}")
    L.append("")
    L.append(f"**Window:** {result.window_hours}h  ")
    L.append(f"**Recommendation:** {result.recommendation}  ")
    L.append("")

    L.append("## Core Metrics")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Decisions | {result.decision_count} |")
    L.append(f"| Would-Trades | {result.would_trade_count} |")
    L.append(f"| Veto Ratio | {result.veto_ratio:.1%} |")
    L.append(f"| Dominant Veto Agent | {result.dominant_veto_agent or '—'} ({result.dominant_veto_agent_share:.0%}) |")
    L.append("")

    L.append("## Pre-Veto vs Post-Veto")
    L.append("")
    L.append("| Metric | Pre-Veto | Post-Veto |")
    L.append("|--------|----------|-----------|")
    L.append(f"| Would-Trades | {result.pre_veto_would_trade_count} | {result.post_veto_would_trade_count} |")
    pv_agr = f"{result.pre_veto_agreement_mean:.4f}" if result.pre_veto_agreement_mean is not None else "—"
    po_agr = f"{result.post_veto_agreement_mean:.4f}" if result.post_veto_agreement_mean is not None else "—"
    L.append(f"| Agreement Mean | {pv_agr} | {po_agr} |")
    pv_conf = f"{result.pre_veto_confidence_mean:.4f}" if result.pre_veto_confidence_mean is not None else "—"
    po_conf = f"{result.post_veto_confidence_mean:.4f}" if result.post_veto_confidence_mean is not None else "—"
    L.append(f"| Confidence Mean | {pv_conf} | {po_conf} |")
    L.append("")

    if result.top_veto_reasons:
        L.append("## Top Veto Reasons")
        L.append("")
        for vr in result.top_veto_reasons[:5]:
            L.append(f"- {vr['reason']} ({vr['count']})")
        L.append("")

    if result.stall_alerts:
        L.append("## Stall Alerts")
        L.append("")
        for a in result.stall_alerts:
            L.append(f"- **[{a.severity.upper()}]** {a.code}: {a.notes}")
        L.append("")

    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Threshold Review")
    parser.add_argument("--db", required=True)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.db):
        logger.error("DB not found: %s", args.db)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    try:
        result = review_agent(conn, args.agent_id, window_hours=args.window_hours)
    finally:
        conn.close()

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(render_review_json(result), encoding="utf-8")
        logger.info("JSON written to %s", args.out_json)

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(render_review_md(result), encoding="utf-8")
        logger.info("Markdown written to %s", args.out_md)

    print(f"\nAgent: {result.agent_id}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Decisions: {result.decision_count} | Would-Trades: {result.would_trade_count} | Veto Ratio: {result.veto_ratio:.1%}")
    if result.stall_alerts:
        print("Stall alerts:")
        for a in result.stall_alerts:
            print(f"  [{a.severity}] {a.code}: {a.notes}")


if __name__ == "__main__":
    main()
