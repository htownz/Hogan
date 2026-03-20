"""Replay query helpers — all dashboard-facing replay queries live here.

Keeps dashboard.py clean by centralising SQL, filtering, sorting, and
similar-event logic.  Every function accepts a raw ``sqlite3.Connection``.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Filter contract
# ---------------------------------------------------------------------------

@dataclass
class ReplayFilter:
    """Controls which decisions the replay selector shows."""
    symbol: str | None = None
    timeframe: str | None = None
    source: str | None = None       # all / swarm / vetoed / traded / skipped
    start_ts_ms: int | None = None
    end_ts_ms: int | None = None
    decision_id: int | None = None
    sort_by: str = "latest"         # latest / highest_opportunity / biggest_winner / biggest_loser / highest_disagreement / veto_events
    limit: int = 200


# ---------------------------------------------------------------------------
# Decision list (Zone A)
# ---------------------------------------------------------------------------

_SORT_MAP = {
    "latest": "sd.ts_ms DESC",
    "highest_opportunity": "sd.final_conf DESC, sd.ts_ms DESC",
    "biggest_winner": "COALESCE(so.forward_60m_bps, -99999) DESC",
    "biggest_loser": "COALESCE(so.forward_60m_bps, 99999) ASC",
    "highest_disagreement": "sd.entropy DESC, sd.ts_ms DESC",
    "veto_events": "sd.vetoed DESC, sd.ts_ms DESC",
}


def list_replay_decisions(
    conn: sqlite3.Connection,
    flt: ReplayFilter,
) -> list[dict[str, Any]]:
    """Return matching decisions for the replay selector."""
    clauses: list[str] = []
    params: list = []

    if flt.symbol:
        clauses.append("sd.symbol = ?")
        params.append(flt.symbol)
    if flt.timeframe:
        clauses.append("sd.timeframe = ?")
        params.append(flt.timeframe)
    if flt.start_ts_ms:
        clauses.append("sd.ts_ms >= ?")
        params.append(flt.start_ts_ms)
    if flt.end_ts_ms:
        clauses.append("sd.ts_ms <= ?")
        params.append(flt.end_ts_ms)
    if flt.decision_id:
        clauses.append("sd.id = ?")
        params.append(flt.decision_id)

    # Source filter
    if flt.source == "vetoed":
        clauses.append("sd.vetoed = 1")
    elif flt.source == "traded":
        clauses.append("sd.final_action IN ('buy','sell') AND sd.vetoed = 0")
    elif flt.source == "skipped":
        clauses.append("(sd.final_action = 'hold' OR sd.vetoed = 1)")
    elif flt.source == "swarm":
        clauses.append("sd.mode != 'baseline'")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order = _SORT_MAP.get(flt.sort_by, "sd.ts_ms DESC")

    sql = f"""
        SELECT sd.id, sd.ts_ms, sd.symbol, sd.timeframe,
               sd.mode, sd.final_action AS swarm_action,
               sd.final_conf, sd.agreement, sd.entropy,
               sd.vetoed, sd.block_reasons_json,
               so.forward_60m_bps, so.outcome_label,
               so.mae_bps, so.mfe_bps,
               sa.outcome_label AS attr_label, sa.learning_note,
               dl.final_action AS baseline_action
        FROM swarm_decisions sd
        LEFT JOIN swarm_outcomes so ON sd.id = so.decision_id
        LEFT JOIN swarm_attribution sa ON sd.id = sa.decision_id
        LEFT JOIN decision_log dl ON sd.ts_ms = dl.ts_ms AND sd.symbol = dl.symbol
        {where}
        ORDER BY {order}
        LIMIT {int(flt.limit)}
    """

    rows = conn.execute(sql, params).fetchall()
    cols = [d[0] for d in conn.execute(sql, params).description] if rows else []
    return [dict(zip(cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Single decision (full replay)
# ---------------------------------------------------------------------------

def get_replay_decision(
    conn: sqlite3.Connection,
    decision_id: int,
) -> dict[str, Any] | None:
    """Full replay bundle for a single decision."""
    dec = _fetch_one(conn, "SELECT * FROM swarm_decisions WHERE id = ?", (decision_id,))
    if not dec:
        return None

    votes = get_replay_votes(conn, decision_id)
    outcome = get_replay_outcome(conn, decision_id)
    attribution = get_replay_attribution(conn, decision_id)
    baseline = get_replay_baseline_compare(conn, decision_id)
    candles = get_replay_candles(
        conn, dec["symbol"], dec["timeframe"], dec["ts_ms"],
    )
    similar = get_replay_similar_events(conn, decision_id)

    return {
        "decision": dec,
        "votes": votes,
        "outcome": outcome,
        "attribution": attribution,
        "baseline_compare": baseline,
        "candles": candles,
        "similar_events": similar,
    }


def get_replay_votes(
    conn: sqlite3.Connection,
    decision_id: int,
) -> list[dict[str, Any]]:
    """All agent votes for a decision."""
    rows = conn.execute(
        "SELECT * FROM swarm_agent_votes WHERE decision_id = ? ORDER BY agent_id",
        (decision_id,),
    ).fetchall()
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM swarm_agent_votes WHERE decision_id = ?", (decision_id,),
    ).description] if rows else []
    return [dict(zip(cols, r)) for r in rows]


def get_replay_outcome(
    conn: sqlite3.Connection,
    decision_id: int,
) -> dict[str, Any] | None:
    """Outcome record for a decision."""
    return _fetch_one(conn, "SELECT * FROM swarm_outcomes WHERE decision_id = ?", (decision_id,))


def get_replay_attribution(
    conn: sqlite3.Connection,
    decision_id: int,
) -> dict[str, Any] | None:
    """Attribution record for a decision."""
    return _fetch_one(conn, "SELECT * FROM swarm_attribution WHERE decision_id = ?", (decision_id,))


def get_replay_candles(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    ts_ms: int,
    bars_before: int = 60,
    bars_after: int = 60,
) -> pd.DataFrame:
    """Candle window around a decision timestamp."""
    before = pd.read_sql_query(
        """SELECT * FROM candles
           WHERE symbol = ? AND timeframe = ? AND ts_ms <= ?
           ORDER BY ts_ms DESC LIMIT ?""",
        conn, params=(symbol, timeframe, ts_ms, bars_before),
    )
    after = pd.read_sql_query(
        """SELECT * FROM candles
           WHERE symbol = ? AND timeframe = ? AND ts_ms > ?
           ORDER BY ts_ms ASC LIMIT ?""",
        conn, params=(symbol, timeframe, ts_ms, bars_after),
    )
    combined = pd.concat([before.iloc[::-1], after], ignore_index=True)
    return combined


def get_replay_baseline_compare(
    conn: sqlite3.Connection,
    decision_id: int,
) -> dict[str, Any] | None:
    """Baseline decision_log entry matching the swarm decision timestamp."""
    dec = _fetch_one(conn, "SELECT ts_ms, symbol FROM swarm_decisions WHERE id = ?", (decision_id,))
    if not dec:
        return None
    return _fetch_one(
        conn,
        "SELECT * FROM decision_log WHERE ts_ms = ? AND symbol = ? LIMIT 1",
        (dec["ts_ms"], dec["symbol"]),
    )


# ---------------------------------------------------------------------------
# Similar events (Zone E, Tab 3)
# ---------------------------------------------------------------------------

def get_replay_similar_events(
    conn: sqlite3.Connection,
    decision_id: int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find top-N similar historical decisions using structured bucket matching."""
    dec = _fetch_one(conn, "SELECT * FROM swarm_decisions WHERE id = ?", (decision_id,))
    if not dec:
        return []

    symbol = dec["symbol"]
    timeframe = dec["timeframe"]

    # Parse decision_json for regime
    try:
        detail = json.loads(dec.get("decision_json", "{}") or "{}")
    except (json.JSONDecodeError, TypeError):
        detail = {}
    regime = detail.get("regime", "")

    conf = dec.get("final_conf", 0) or 0
    entropy = dec.get("entropy", 0) or 0
    action = dec.get("final_action", "hold")

    # Bucket boundaries
    conf_bucket = int(conf * 5)  # 0-4
    entropy_bucket = int(min(entropy, 1.5) * 3)  # 0-4

    candidates = pd.read_sql_query(
        """SELECT sd.*, so.forward_60m_bps, so.outcome_label,
                  sa.outcome_label AS attr_label
           FROM swarm_decisions sd
           LEFT JOIN swarm_outcomes so ON sd.id = so.decision_id
           LEFT JOIN swarm_attribution sa ON sd.id = sa.decision_id
           WHERE sd.symbol = ? AND sd.timeframe = ? AND sd.id != ?
           ORDER BY sd.ts_ms DESC LIMIT 500""",
        conn, params=(symbol, timeframe, decision_id),
    )

    if candidates.empty:
        return []

    scores: list[tuple[float, int]] = []
    for idx, row in candidates.iterrows():
        score = 0.0

        # Regime match
        try:
            c_detail = json.loads(row.get("decision_json", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            c_detail = {}
        if c_detail.get("regime") == regime:
            score += 3.0

        # Action match
        if row.get("final_action") == action:
            score += 2.0

        # Confidence bucket match
        c_conf = row.get("final_conf", 0) or 0
        if int(c_conf * 5) == conf_bucket:
            score += 2.0

        # Entropy bucket match
        c_entropy = row.get("entropy", 0) or 0
        if int(min(c_entropy, 1.5) * 3) == entropy_bucket:
            score += 2.0

        # Veto match
        if bool(row.get("vetoed")) == bool(dec.get("vetoed")):
            score += 1.0

        scores.append((score, idx))

    scores.sort(key=lambda x: -x[0])
    top_indices = [idx for _, idx in scores[:limit]]
    result = candidates.loc[top_indices]

    return result.to_dict("records")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_one(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple = (),
) -> dict[str, Any] | None:
    """Execute SQL and return the first row as a dict, or None."""
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))
