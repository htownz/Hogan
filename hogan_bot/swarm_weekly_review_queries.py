"""Query layer for the Swarm Weekly Review.

Each function reads from swarm SQLite tables within a 7-day (or custom)
window and returns plain dicts suitable for the review rules engine.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_range(week_end: str | None = None, days: int = 7) -> tuple[int, int]:
    if week_end:
        dt = datetime.strptime(week_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_ms = int((dt + timedelta(days=1)).timestamp() * 1000)
    start_ms = int((dt + timedelta(days=1) - timedelta(days=days)).timestamp() * 1000)
    return start_ms, end_ms


def _week_label(week_end: str | None = None) -> str:
    if week_end:
        dt = datetime.strptime(week_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ).fetchone()
    return bool(row and row[0])


def _opt_where(symbol: str | None, timeframe: str | None, prefix: str = "") -> tuple[str, list]:
    col = f"{prefix}." if prefix else ""
    clauses: list[str] = []
    params: list = []
    if symbol:
        clauses.append(f"{col}symbol = ?")
        params.append(symbol)
    if timeframe:
        clauses.append(f"{col}timeframe = ?")
        params.append(timeframe)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


# ---------------------------------------------------------------------------
# 1. Review window
# ---------------------------------------------------------------------------

def fetch_review_window(
    conn: sqlite3.Connection,
    week_end: str | None = None,
    days: int = 7,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict:
    start_ms, end_ms = _ts_range(week_end, days)
    decision_count = 0
    if _table_exists(conn, "swarm_decisions"):
        extra, params = _opt_where(symbol, timeframe)
        row = conn.execute(
            f"SELECT COUNT(*) FROM swarm_decisions WHERE ts_ms BETWEEN ? AND ?{extra}",
            [start_ms, end_ms] + params,
        ).fetchone()
        decision_count = row[0] if row else 0
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "week_label": _week_label(week_end),
        "week_end": week_end or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "days": days,
        "decision_count": decision_count,
    }


# ---------------------------------------------------------------------------
# 2. Core counts
# ---------------------------------------------------------------------------

def fetch_weekly_swarm_counts(
    conn: sqlite3.Connection, start_ms: int, end_ms: int,
    symbol: str | None = None, timeframe: str | None = None,
) -> dict:
    out: dict = {
        "decision_count": 0, "veto_count": 0, "would_trade_count": 0,
        "distinct_regimes": 0, "mean_agreement": None, "mean_entropy": None,
        "mean_confidence": None, "baseline_match_count": 0, "baseline_miss_count": 0,
    }
    if not _table_exists(conn, "swarm_decisions"):
        return out
    extra, params = _opt_where(symbol, timeframe)
    bp = [start_ms, end_ms] + params
    row = conn.execute(
        f"""SELECT COUNT(*),
               SUM(CASE WHEN vetoed=1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN final_action IN ('buy','sell') AND vetoed=0 THEN 1 ELSE 0 END),
               COUNT(DISTINCT regime),
               AVG(agreement), AVG(entropy), AVG(final_conf)
            FROM swarm_decisions WHERE ts_ms BETWEEN ? AND ?{extra}""", bp,
    ).fetchone()
    if row and row[0]:
        out["decision_count"] = row[0]
        out["veto_count"] = row[1] or 0
        out["would_trade_count"] = row[2] or 0
        out["distinct_regimes"] = row[3] or 0
        out["mean_agreement"] = round(row[4], 4) if row[4] is not None else None
        out["mean_entropy"] = round(row[5], 4) if row[5] is not None else None
        out["mean_confidence"] = round(row[6], 4) if row[6] is not None else None

    if _table_exists(conn, "decision_log"):
        bl_extra, bl_params = _opt_where(symbol, timeframe, prefix="sd")
        bl = conn.execute(
            f"""SELECT SUM(CASE WHEN dl.id IS NOT NULL THEN 1 ELSE 0 END),
                       SUM(CASE WHEN dl.id IS NULL THEN 1 ELSE 0 END)
                FROM swarm_decisions sd
                LEFT JOIN decision_log dl ON dl.swarm_decision_id = sd.id
                WHERE sd.ts_ms BETWEEN ? AND ?{bl_extra}""",
            [start_ms, end_ms] + bl_params,
        ).fetchone()
        if bl:
            out["baseline_match_count"] = bl[0] or 0
            out["baseline_miss_count"] = bl[1] or 0
    return out


# ---------------------------------------------------------------------------
# 3. Opportunity stats
# ---------------------------------------------------------------------------

def fetch_weekly_opportunity_stats(
    conn: sqlite3.Connection, start_ms: int, end_ms: int,
    symbol: str | None = None, timeframe: str | None = None,
) -> dict:
    out: dict = {
        "opportunity_score_mean": None, "opportunity_score_std": None,
        "opportunity_score_top_decile_markout_bps": None,
        "opportunity_score_bottom_decile_markout_bps": None,
        "opportunity_monotonicity_score": None, "no_trade_ratio": None,
        "tier_a_count": 0, "tier_b_count": 0, "tier_c_count": 0,
        "tier_d_count": 0, "tier_f_count": 0,
    }
    if not _table_exists(conn, "swarm_outcomes"):
        return out
    extra, params = _opt_where(symbol, timeframe, prefix="sd")
    rows = conn.execute(
        f"""SELECT so.forward_60m_bps, sd.final_conf, so.outcome_label
            FROM swarm_outcomes so
            JOIN swarm_decisions sd ON so.decision_id = sd.id
            WHERE sd.ts_ms BETWEEN ? AND ?{extra}
            ORDER BY sd.final_conf""",
        [start_ms, end_ms] + params,
    ).fetchall()
    if not rows:
        return out
    markouts = [r[0] for r in rows if r[0] is not None]
    confs = [r[1] for r in rows if r[1] is not None]
    labels = [r[2] for r in rows if r[2] is not None]
    if confs:
        import statistics
        out["opportunity_score_mean"] = round(statistics.mean(confs), 4)
        out["opportunity_score_std"] = round(statistics.pstdev(confs), 4) if len(confs) > 1 else 0.0
    if markouts:
        n = len(markouts)
        d = max(1, n // 10)
        sm = sorted(markouts)
        out["opportunity_score_bottom_decile_markout_bps"] = round(sum(sm[:d]) / d, 2)
        out["opportunity_score_top_decile_markout_bps"] = round(sum(sm[-d:]) / d, 2)
    tier_map = {"a": 0, "b": 0, "c": 0, "d": 0, "f": 0}
    for lbl in labels:
        ll = (lbl or "").lower()
        if "winner" in ll or "saved" in ll:
            tier_map["a"] += 1
        elif "scratch" in ll:
            tier_map["b"] += 1
        elif "pending" in ll:
            tier_map["c"] += 1
        elif "loser" in ll:
            tier_map["d"] += 1
        elif "false" in ll or "missed" in ll:
            tier_map["f"] += 1
        else:
            tier_map["c"] += 1
    for t, c in tier_map.items():
        out[f"tier_{t}_count"] = c
    return out


# ---------------------------------------------------------------------------
# 4. Veto stats
# ---------------------------------------------------------------------------

def fetch_weekly_veto_stats(
    conn: sqlite3.Connection, start_ms: int, end_ms: int,
    symbol: str | None = None, timeframe: str | None = None,
) -> dict:
    out: dict = {
        "veto_ratio": 0.0, "top_veto_reason": None, "top_veto_reason_share": 0.0,
        "veto_reasons_ranked": [], "veto_bad_trade_capture_rate": None,
        "veto_blocked_winner_rate": None, "dominant_veto_agent": None,
        "dominant_veto_agent_share": 0.0,
    }
    if not _table_exists(conn, "swarm_agent_votes"):
        return out
    extra, params = _opt_where(symbol, timeframe, prefix="sd")
    bp = [start_ms, end_ms] + params

    rows = conn.execute(
        f"""SELECT sav.agent_id, sav.block_reasons_json
            FROM swarm_agent_votes sav
            JOIN swarm_decisions sd ON sav.decision_id = sd.id
            WHERE sav.veto = 1 AND sd.ts_ms BETWEEN ? AND ?{extra}""", bp,
    ).fetchall()

    total_row = conn.execute(
        f"""SELECT COUNT(*) FROM swarm_decisions
            WHERE ts_ms BETWEEN ? AND ?{_opt_where(symbol, timeframe)[0]}""",
        [start_ms, end_ms] + _opt_where(symbol, timeframe)[1],
    ).fetchone()
    total = total_row[0] if total_row else 0
    veto_count = len(rows)
    out["veto_ratio"] = round(veto_count / total, 4) if total > 0 else 0.0

    reason_counts: dict[str, int] = {}
    agent_veto_counts: dict[str, int] = {}
    for agent_id, reasons_json in rows:
        agent_veto_counts[agent_id] = agent_veto_counts.get(agent_id, 0) + 1
        try:
            reasons = json.loads(reasons_json) if reasons_json else []
        except (json.JSONDecodeError, TypeError):
            reasons = []
        for r in reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    if reason_counts:
        ranked = sorted(reason_counts.items(), key=lambda x: -x[1])
        out["veto_reasons_ranked"] = [{"reason": r, "count": c} for r, c in ranked[:10]]
        out["top_veto_reason"] = ranked[0][0]
        out["top_veto_reason_share"] = round(ranked[0][1] / veto_count, 4) if veto_count else 0.0

    if agent_veto_counts and veto_count > 0:
        dom = max(agent_veto_counts.items(), key=lambda x: x[1])
        out["dominant_veto_agent"] = dom[0]
        out["dominant_veto_agent_share"] = round(dom[1] / veto_count, 4)

    if _table_exists(conn, "swarm_outcomes"):
        vq = conn.execute(
            f"""SELECT SUM(CASE WHEN so.was_veto_correct=1 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN so.was_veto_correct=0 THEN 1 ELSE 0 END),
                       COUNT(*)
                FROM swarm_outcomes so
                JOIN swarm_decisions sd ON so.decision_id = sd.id
                WHERE sd.vetoed = 1 AND sd.ts_ms BETWEEN ? AND ?{extra}""", bp,
        ).fetchone()
        if vq and vq[2] and vq[2] > 0:
            out["veto_bad_trade_capture_rate"] = round((vq[0] or 0) / vq[2], 4)
            out["veto_blocked_winner_rate"] = round((vq[1] or 0) / vq[2], 4)
    return out


# ---------------------------------------------------------------------------
# 5. Agent scores
# ---------------------------------------------------------------------------

def fetch_weekly_agent_scores(
    conn: sqlite3.Connection, start_ms: int, end_ms: int,
    symbol: str | None = None, timeframe: str | None = None,
) -> list[dict]:
    if not _table_exists(conn, "swarm_agent_votes"):
        return []
    extra, params = _opt_where(symbol, timeframe, prefix="sd")
    rows = conn.execute(
        f"""SELECT sav.agent_id,
               COUNT(*) AS total,
               SUM(CASE WHEN sav.action='hold' THEN 1 ELSE 0 END) AS holds,
               SUM(CASE WHEN sav.veto=1 THEN 1 ELSE 0 END) AS vetoes,
               AVG(sav.confidence) AS avg_conf,
               AVG(sav.expected_edge_bps) AS avg_edge
            FROM swarm_agent_votes sav
            JOIN swarm_decisions sd ON sav.decision_id = sd.id
            WHERE sd.ts_ms BETWEEN ? AND ?{extra}
            GROUP BY sav.agent_id""",
        [start_ms, end_ms] + params,
    ).fetchall()
    agents = []
    for r in rows:
        hold_rate = r[2] / r[1] if r[1] > 0 else 0.0
        agents.append({
            "agent_id": r[0], "decisions": r[1], "vetoes": r[3],
            "hold_rate": round(hold_rate, 4),
            "mean_confidence": round(r[4], 4) if r[4] is not None else None,
            "mean_edge_bps": round(r[5], 2) if r[5] is not None else None,
        })
    return agents


# ---------------------------------------------------------------------------
# 6. Divergence stats
# ---------------------------------------------------------------------------

def fetch_weekly_divergence_stats(
    conn: sqlite3.Connection, start_ms: int, end_ms: int,
    symbol: str | None = None, timeframe: str | None = None,
) -> dict:
    out: dict = {
        "baseline_match_count": 0, "baseline_miss_count": 0,
        "baseline_match_ratio": None, "divergence_count": 0,
    }
    if not _table_exists(conn, "decision_log") or not _table_exists(conn, "swarm_decisions"):
        return out
    extra, params = _opt_where(symbol, timeframe, prefix="sd")
    rows = conn.execute(
        f"""SELECT sd.id, sd.final_action, dl.final_action
            FROM swarm_decisions sd
            LEFT JOIN decision_log dl ON dl.swarm_decision_id = sd.id
            WHERE sd.ts_ms BETWEEN ? AND ?{extra}""",
        [start_ms, end_ms] + params,
    ).fetchall()
    matches = misses = divergences = 0
    for _, swarm_act, bl_act in rows:
        if bl_act is None:
            misses += 1
        elif swarm_act == bl_act:
            matches += 1
        else:
            divergences += 1
    total = matches + misses + divergences
    out["baseline_match_count"] = matches
    out["baseline_miss_count"] = misses
    out["divergence_count"] = divergences
    out["baseline_match_ratio"] = round(matches / total, 4) if total > 0 else None
    return out


# ---------------------------------------------------------------------------
# 7. Learning / drift stats
# ---------------------------------------------------------------------------

def fetch_weekly_learning_stats(
    conn: sqlite3.Connection, start_ms: int, end_ms: int,
    symbol: str | None = None, timeframe: str | None = None,
) -> dict:
    out: dict = {
        "weight_update_count": 0, "latest_weight_update_ts": None,
        "learning_import_error_count": 0, "promotion_report_available": False,
    }
    if _table_exists(conn, "swarm_weight_snapshots"):
        extra, params = _opt_where(symbol, timeframe)
        row = conn.execute(
            f"""SELECT COUNT(*), MAX(ts_ms) FROM swarm_weight_snapshots
                WHERE ts_ms BETWEEN ? AND ?{extra}""",
            [start_ms, end_ms] + params,
        ).fetchone()
        if row:
            out["weight_update_count"] = row[0] or 0
            out["latest_weight_update_ts"] = row[1]
    if _table_exists(conn, "swarm_promotion_reports"):
        extra2, params2 = _opt_where(symbol, timeframe)
        pr = conn.execute(
            f"""SELECT COUNT(*) FROM swarm_promotion_reports
                WHERE created_ms BETWEEN ? AND ?{extra2}""",
            [start_ms, end_ms] + params2,
        ).fetchone()
        out["promotion_report_available"] = bool(pr and pr[0] > 0)
    try:
        from hogan_bot import swarm_metrics  # noqa: F401
    except ImportError:
        out["learning_import_error_count"] = 1
    return out


# ---------------------------------------------------------------------------
# 8. Regime stats
# ---------------------------------------------------------------------------

def fetch_weekly_regime_stats(
    conn: sqlite3.Connection, start_ms: int, end_ms: int,
    symbol: str | None = None, timeframe: str | None = None,
) -> dict:
    out: dict = {"regime_counts": {}, "distinct_regimes": 0, "regime_missing": False}
    if not _table_exists(conn, "swarm_decisions"):
        return out
    extra, params = _opt_where(symbol, timeframe)
    rows = conn.execute(
        f"""SELECT regime, COUNT(*) FROM swarm_decisions
            WHERE ts_ms BETWEEN ? AND ?{extra}
            GROUP BY regime""",
        [start_ms, end_ms] + params,
    ).fetchall()
    counts: dict[str, int] = {}
    for regime, cnt in rows:
        key = regime or "(none)"
        counts[key] = cnt
    out["regime_counts"] = counts
    real = {k for k in counts if k != "(none)"}
    out["distinct_regimes"] = len(real)
    out["regime_missing"] = "(none)" in counts and counts["(none)"] > 0
    return out


# ---------------------------------------------------------------------------
# 9. Week-over-week
# ---------------------------------------------------------------------------

def fetch_week_over_week_stats(
    conn: sqlite3.Connection,
    curr_start: int, curr_end: int,
    prev_start: int, prev_end: int,
    symbol: str | None = None, timeframe: str | None = None,
) -> dict:
    """Compute deltas between current and previous review windows."""
    def _counts(s: int, e: int) -> dict:
        extra, params = _opt_where(symbol, timeframe)
        if not _table_exists(conn, "swarm_decisions"):
            return {"decision_count": 0, "would_trade_count": 0, "veto_count": 0, "distinct_regimes": 0}
        r = conn.execute(
            f"""SELECT COUNT(*),
                   SUM(CASE WHEN vetoed=1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN final_action IN ('buy','sell') AND vetoed=0 THEN 1 ELSE 0 END),
                   COUNT(DISTINCT regime)
                FROM swarm_decisions WHERE ts_ms BETWEEN ? AND ?{extra}""",
            [s, e] + params,
        ).fetchone()
        return {
            "decision_count": r[0] or 0, "veto_count": r[1] or 0,
            "would_trade_count": r[2] or 0, "distinct_regimes": r[3] or 0,
        } if r else {"decision_count": 0, "would_trade_count": 0, "veto_count": 0, "distinct_regimes": 0}

    curr = _counts(curr_start, curr_end)
    prev = _counts(prev_start, prev_end)

    def _delta(key: str):
        c, p = curr.get(key, 0), prev.get(key, 0)
        return c - p

    def _ratio_delta(num_key: str, den_key: str):
        cn, cd = curr.get(num_key, 0), curr.get(den_key, 0)
        pn, pd = prev.get(num_key, 0), prev.get(den_key, 0)
        cr = cn / cd if cd > 0 else 0.0
        pr = pn / pd if pd > 0 else 0.0
        return round(cr - pr, 4)

    out = {
        "prior_week_available": prev["decision_count"] > 0,
        "decision_count_wow_delta": _delta("decision_count"),
        "would_trade_wow_delta": _delta("would_trade_count"),
        "veto_count_wow_delta": _delta("veto_count"),
        "veto_ratio_wow_delta": _ratio_delta("veto_count", "decision_count"),
        "distinct_regimes_wow_delta": _delta("distinct_regimes"),
    }
    return out


# ---------------------------------------------------------------------------
# 10. Replay candidates
# ---------------------------------------------------------------------------

def fetch_weekly_replay_candidates(
    conn: sqlite3.Connection, start_ms: int, end_ms: int,
    symbol: str | None = None, timeframe: str | None = None,
    limit: int = 20,
) -> list[dict]:
    candidates: list[dict] = []
    if not _table_exists(conn, "swarm_decisions"):
        return candidates
    extra, params = _opt_where(symbol, timeframe, prefix="sd")
    bp = [start_ms, end_ms] + params

    has_outcomes = _table_exists(conn, "swarm_outcomes")
    if has_outcomes:
        sql = f"""SELECT sd.id, sd.symbol, sd.ts_ms, sd.final_action, sd.vetoed,
                         sd.final_conf, so.forward_60m_bps, so.outcome_label
                  FROM swarm_decisions sd
                  LEFT JOIN swarm_outcomes so ON so.decision_id = sd.id
                  WHERE sd.ts_ms BETWEEN ? AND ?{extra}
                  ORDER BY ABS(COALESCE(so.forward_60m_bps, 0)) DESC
                  LIMIT ?"""
    else:
        sql = f"""SELECT sd.id, sd.symbol, sd.ts_ms, sd.final_action, sd.vetoed,
                         sd.final_conf, NULL, NULL
                  FROM swarm_decisions sd
                  WHERE sd.ts_ms BETWEEN ? AND ?{extra}
                  ORDER BY sd.ts_ms DESC LIMIT ?"""

    rows = conn.execute(sql, bp + [limit * 4]).fetchall()

    categories_found: dict[str, int] = {}
    priority = 1

    for row in rows:
        dec_id, sym, ts_ms, action, vetoed, conf, fwd_bps, label = row
        ts_iso = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat() if ts_ms else ""

        cat = _classify_replay_candidate(vetoed, action, conf, fwd_bps, label)
        if categories_found.get(cat, 0) >= 3:
            continue

        reason = _build_replay_reason(cat, vetoed, action, fwd_bps, label)
        candidates.append({
            "decision_id": dec_id, "symbol": sym or "", "ts_iso": ts_iso,
            "category": cat, "reason": reason, "priority": priority,
        })
        categories_found[cat] = categories_found.get(cat, 0) + 1
        priority += 1
        if len(candidates) >= limit:
            break

    return candidates


def _classify_replay_candidate(vetoed, action, conf, fwd_bps, label) -> str:
    if fwd_bps is not None and fwd_bps > 20 and not vetoed:
        return "best_opportunity"
    if vetoed and fwd_bps is not None and fwd_bps > 15:
        return "worst_veto"
    if vetoed:
        return "dominant_agent_case"
    if action == "hold" and conf is not None and conf > 0.5:
        return "high_confidence_hold"
    if label and "false" in (label or "").lower():
        return "strong_divergence"
    return "general"


def _build_replay_reason(cat, vetoed, action, fwd_bps, label) -> str:
    parts = {
        "best_opportunity": f"Strong forward move {fwd_bps:+.0f}bps on {action}" if fwd_bps else f"{action} with outcome",
        "worst_veto": f"Vetoed but market moved {fwd_bps:+.0f}bps" if fwd_bps else "Vetoed with significant move",
        "dominant_agent_case": "Vetoed decision — check dominant agent contribution",
        "high_confidence_hold": f"Hold at confidence {conf}" if (conf := None) is None else "High-confidence hold",
        "strong_divergence": f"Divergent outcome: {label}",
        "general": f"{action} decision",
    }
    return parts.get(cat, f"{action} decision")
