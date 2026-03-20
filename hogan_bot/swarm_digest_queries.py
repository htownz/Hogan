"""Query layer for the Swarm Daily Digest.

Each function reads from the swarm SQLite tables within a time window
and returns a plain dict suitable for the digest rules engine.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone


def _ts_range(date: str | None = None, hours: int = 24) -> tuple[int, int]:
    """Return (start_ms, end_ms) for a digest window."""
    if date:
        dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_ms = int((dt + timedelta(days=1)).timestamp() * 1000)
    start_ms = int((dt + timedelta(days=1) - timedelta(hours=hours)).timestamp() * 1000)
    return start_ms, end_ms


def _safe_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row and row[0])


def _opt_filters(symbol: str | None, timeframe: str | None, prefix: str = "") -> tuple[str, list]:
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
# 1. Digest window
# ---------------------------------------------------------------------------

def fetch_digest_window(
    conn: sqlite3.Connection,
    date: str | None = None,
    hours: int = 24,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict:
    """Return start/end timestamps and core row counts for the window."""
    start_ms, end_ms = _ts_range(date, hours)
    extra, params = _opt_filters(symbol, timeframe)

    decision_count = 0
    if _safe_table_exists(conn, "swarm_decisions"):
        row = conn.execute(
            f"SELECT COUNT(*) FROM swarm_decisions WHERE ts_ms BETWEEN ? AND ?{extra}",
            [start_ms, end_ms] + params,
        ).fetchone()
        decision_count = row[0] if row else 0

    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "hours": hours,
        "decision_count": decision_count,
    }


# ---------------------------------------------------------------------------
# 2. Core counts
# ---------------------------------------------------------------------------

def fetch_swarm_counts(
    conn: sqlite3.Connection,
    start_ms: int,
    end_ms: int,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict:
    """Return total decisions, vetoes, would-trades, distinct regimes, baseline matches."""
    extra, params = _opt_filters(symbol, timeframe)
    base_params = [start_ms, end_ms] + params

    out: dict = {
        "decision_count": 0,
        "veto_count": 0,
        "would_trade_count": 0,
        "distinct_regimes": 0,
        "mean_agreement": None,
        "mean_entropy": None,
        "mean_confidence": None,
        "baseline_match_count": 0,
        "baseline_miss_count": 0,
    }

    if not _safe_table_exists(conn, "swarm_decisions"):
        return out

    rows = conn.execute(
        f"""SELECT
                COUNT(*) AS cnt,
                SUM(CASE WHEN vetoed = 1 THEN 1 ELSE 0 END) AS vetoes,
                SUM(CASE WHEN final_action IN ('buy','sell') AND vetoed = 0 THEN 1 ELSE 0 END) AS would_trades,
                COUNT(DISTINCT regime) AS regimes,
                AVG(agreement) AS avg_agreement,
                AVG(entropy) AS avg_entropy,
                AVG(final_conf) AS avg_conf
            FROM swarm_decisions
            WHERE ts_ms BETWEEN ? AND ?{extra}""",
        base_params,
    ).fetchone()

    if rows and rows[0]:
        out["decision_count"] = rows[0]
        out["veto_count"] = rows[1] or 0
        out["would_trade_count"] = rows[2] or 0
        out["distinct_regimes"] = rows[3] or 0
        out["mean_agreement"] = round(rows[4], 4) if rows[4] is not None else None
        out["mean_entropy"] = round(rows[5], 4) if rows[5] is not None else None
        out["mean_confidence"] = round(rows[6], 4) if rows[6] is not None else None

    if _safe_table_exists(conn, "decision_log"):
        bl_extra, bl_params = _opt_filters(symbol, timeframe, prefix="sd")
        bl = conn.execute(
            f"""SELECT
                    SUM(CASE WHEN dl.id IS NOT NULL THEN 1 ELSE 0 END) AS matches,
                    SUM(CASE WHEN dl.id IS NULL THEN 1 ELSE 0 END) AS misses
                FROM swarm_decisions sd
                LEFT JOIN decision_log dl
                    ON dl.swarm_decision_id = sd.id
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

def fetch_opportunity_stats(
    conn: sqlite3.Connection,
    start_ms: int,
    end_ms: int,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict:
    """Return score distribution, tier counts, decile markouts."""
    out: dict = {
        "opportunity_score_mean": None,
        "opportunity_score_median": None,
        "opportunity_score_top_decile_markout_bps": None,
        "opportunity_score_bottom_decile_markout_bps": None,
        "opportunity_monotonicity_score": None,
        "tier_a_count": 0,
        "tier_b_count": 0,
        "tier_c_count": 0,
        "tier_d_count": 0,
        "tier_f_count": 0,
    }

    if not _safe_table_exists(conn, "swarm_outcomes"):
        return out

    extra, params = _opt_filters(symbol, timeframe, prefix="sd")
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
        sorted_c = sorted(confs)
        mid = len(sorted_c) // 2
        out["opportunity_score_mean"] = round(sum(confs) / len(confs), 4)
        out["opportunity_score_median"] = round(sorted_c[mid], 4)

    if markouts:
        n = len(markouts)
        decile = max(1, n // 10)
        sorted_m = sorted(markouts)
        out["opportunity_score_bottom_decile_markout_bps"] = round(sum(sorted_m[:decile]) / decile, 2)
        out["opportunity_score_top_decile_markout_bps"] = round(sum(sorted_m[-decile:]) / decile, 2)

    tier_map = {"a": 0, "b": 0, "c": 0, "d": 0, "f": 0}
    for lbl in labels:
        ll = lbl.lower() if lbl else ""
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

def fetch_veto_stats(
    conn: sqlite3.Connection,
    start_ms: int,
    end_ms: int,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict:
    """Return veto ratio, top reasons, capture quality."""
    out: dict = {
        "veto_ratio": 0.0,
        "top_veto_reason": None,
        "top_veto_reason_share": 0.0,
        "veto_reasons_ranked": [],
        "veto_bad_trade_capture_rate": None,
        "veto_blocked_winner_rate": None,
    }

    if not _safe_table_exists(conn, "swarm_agent_votes"):
        return out

    extra, params = _opt_filters(symbol, timeframe, prefix="sd")
    rows = conn.execute(
        f"""SELECT sav.block_reasons_json
            FROM swarm_agent_votes sav
            JOIN swarm_decisions sd ON sav.decision_id = sd.id
            WHERE sav.veto = 1 AND sd.ts_ms BETWEEN ? AND ?{extra}""",
        [start_ms, end_ms] + params,
    ).fetchall()

    total_decisions = conn.execute(
        f"""SELECT COUNT(*) FROM swarm_decisions
            WHERE ts_ms BETWEEN ? AND ?{extra.replace('sd.', '')}""",
        [start_ms, end_ms] + params,
    ).fetchone()
    total = total_decisions[0] if total_decisions else 0

    veto_count = len(rows)
    out["veto_ratio"] = round(veto_count / total, 4) if total > 0 else 0.0

    reason_counts: dict[str, int] = {}
    for (reasons_json,) in rows:
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

    if _safe_table_exists(conn, "swarm_outcomes"):
        vq = conn.execute(
            f"""SELECT
                    SUM(CASE WHEN so.was_veto_correct = 1 THEN 1 ELSE 0 END) AS good,
                    SUM(CASE WHEN so.was_veto_correct = 0 THEN 1 ELSE 0 END) AS bad,
                    COUNT(*) AS total
                FROM swarm_outcomes so
                JOIN swarm_decisions sd ON so.decision_id = sd.id
                WHERE sd.vetoed = 1 AND sd.ts_ms BETWEEN ? AND ?{extra}""",
            [start_ms, end_ms] + params,
        ).fetchone()
        if vq and vq[2] and vq[2] > 0:
            out["veto_bad_trade_capture_rate"] = round(vq[0] / vq[2], 4) if vq[0] is not None else None
            out["veto_blocked_winner_rate"] = round(vq[1] / vq[2], 4) if vq[1] is not None else None

    return out


# ---------------------------------------------------------------------------
# 5. Per-agent vote stats
# ---------------------------------------------------------------------------

def fetch_agent_vote_stats(
    conn: sqlite3.Connection,
    start_ms: int,
    end_ms: int,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict:
    """Return per-agent vote counts, veto counts, confidence means."""
    out: dict = {"agents": [], "agent_hold_dominance_ratio": 0.0}

    if not _safe_table_exists(conn, "swarm_agent_votes"):
        return out

    extra, params = _opt_filters(symbol, timeframe, prefix="sd")
    rows = conn.execute(
        f"""SELECT
                sav.agent_id,
                COUNT(*) AS total,
                SUM(CASE WHEN sav.action = 'hold' THEN 1 ELSE 0 END) AS holds,
                SUM(CASE WHEN sav.veto = 1 THEN 1 ELSE 0 END) AS vetoes,
                AVG(sav.confidence) AS avg_conf,
                AVG(sav.expected_edge_bps) AS avg_edge
            FROM swarm_agent_votes sav
            JOIN swarm_decisions sd ON sav.decision_id = sd.id
            WHERE sd.ts_ms BETWEEN ? AND ?{extra}
            GROUP BY sav.agent_id""",
        [start_ms, end_ms] + params,
    ).fetchall()

    total_votes = 0
    total_holds = 0
    agents = []
    for r in rows:
        agents.append({
            "agent_id": r[0],
            "total_votes": r[1],
            "hold_votes": r[2],
            "veto_count": r[3],
            "avg_confidence": round(r[4], 4) if r[4] is not None else None,
            "avg_edge_bps": round(r[5], 2) if r[5] is not None else None,
        })
        total_votes += r[1]
        total_holds += r[2]

    out["agents"] = agents
    out["agent_hold_dominance_ratio"] = round(total_holds / total_votes, 4) if total_votes > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# 6. Divergence stats
# ---------------------------------------------------------------------------

def fetch_divergence_stats(
    conn: sqlite3.Connection,
    start_ms: int,
    end_ms: int,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict:
    """Return swarm-vs-baseline match/miss rates."""
    out: dict = {
        "baseline_match_count": 0,
        "baseline_miss_count": 0,
        "baseline_match_ratio": None,
        "divergence_count": 0,
        "divergence_actionable_count": 0,
    }

    if not _safe_table_exists(conn, "decision_log") or not _safe_table_exists(conn, "swarm_decisions"):
        return out

    extra, params = _opt_filters(symbol, timeframe, prefix="sd")
    rows = conn.execute(
        f"""SELECT
                sd.id, sd.final_action, dl.final_action AS baseline_action
            FROM swarm_decisions sd
            LEFT JOIN decision_log dl ON dl.swarm_decision_id = sd.id
            WHERE sd.ts_ms BETWEEN ? AND ?{extra}""",
        [start_ms, end_ms] + params,
    ).fetchall()

    matches = 0
    misses = 0
    divergences = 0
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
    out["divergence_actionable_count"] = divergences
    return out


# ---------------------------------------------------------------------------
# 7. Learning / drift stats
# ---------------------------------------------------------------------------

def fetch_learning_drift_stats(
    conn: sqlite3.Connection,
    start_ms: int,
    end_ms: int,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict:
    """Return weight update history and drift indicators."""
    out: dict = {
        "weight_update_count": 0,
        "latest_weight_update_ts": None,
        "learning_import_error_count": 0,
        "calibration_drift_score": None,
    }

    if _safe_table_exists(conn, "swarm_weight_snapshots"):
        extra, params = _opt_filters(symbol, timeframe)
        row = conn.execute(
            f"""SELECT COUNT(*), MAX(ts_ms)
                FROM swarm_weight_snapshots
                WHERE ts_ms BETWEEN ? AND ?{extra}""",
            [start_ms, end_ms] + params,
        ).fetchone()
        if row:
            out["weight_update_count"] = row[0] or 0
            out["latest_weight_update_ts"] = row[1]

    try:
        from hogan_bot import swarm_metrics  # noqa: F401
    except ImportError:
        out["learning_import_error_count"] = 1

    return out


# ---------------------------------------------------------------------------
# 8. Replay candidates
# ---------------------------------------------------------------------------

def fetch_replay_candidates(
    conn: sqlite3.Connection,
    start_ms: int,
    end_ms: int,
    symbol: str | None = None,
    timeframe: str | None = None,
    limit: int = 12,
) -> list[dict]:
    """Return highest-priority decisions for manual review."""
    candidates: list[dict] = []

    if not _safe_table_exists(conn, "swarm_decisions"):
        return candidates

    extra, params = _opt_filters(symbol, timeframe, prefix="sd")
    base_params = [start_ms, end_ms] + params

    if _safe_table_exists(conn, "swarm_outcomes"):
        rows = conn.execute(
            f"""SELECT sd.id, sd.symbol, sd.ts_ms, sd.final_action, sd.vetoed,
                       so.forward_60m_bps, so.outcome_label
                FROM swarm_decisions sd
                LEFT JOIN swarm_outcomes so ON so.decision_id = sd.id
                WHERE sd.ts_ms BETWEEN ? AND ?{extra}
                ORDER BY ABS(COALESCE(so.forward_60m_bps, 0)) DESC
                LIMIT ?""",
            base_params + [limit * 3],
        ).fetchall()
    else:
        rows = conn.execute(
            f"""SELECT sd.id, sd.symbol, sd.ts_ms, sd.final_action, sd.vetoed,
                       NULL, NULL
                FROM swarm_decisions sd
                WHERE sd.ts_ms BETWEEN ? AND ?{extra}
                ORDER BY sd.ts_ms DESC
                LIMIT ?""",
            base_params + [limit * 3],
        ).fetchall()

    priority = 1
    for row in rows:
        dec_id, sym, ts_ms, action, vetoed, fwd_bps, label = row
        ts_iso = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat() if ts_ms else ""

        reason_parts = []
        if vetoed:
            reason_parts.append("Vetoed")
        if fwd_bps is not None and abs(fwd_bps) > 20:
            reason_parts.append(f"extreme move {fwd_bps:+.0f}bps after {'veto' if vetoed else action}")
        if label and "false" in label.lower():
            reason_parts.append(f"outcome={label}")
        if label and "missed" in label.lower():
            reason_parts.append(f"outcome={label}")
        if not reason_parts:
            reason_parts.append(f"{action} decision")

        candidates.append({
            "decision_id": dec_id,
            "symbol": sym or "",
            "ts_iso": ts_iso,
            "reason": "; ".join(reason_parts),
            "priority": priority,
        })
        priority += 1
        if len(candidates) >= limit:
            break

    return candidates
