"""Reusable query helpers for the Hogan swarm decision layer.

These functions read from the SQLite DB and return DataFrames or dicts.
They are designed to be called from the Streamlit dashboard, CLI scripts,
or any other consumer that needs swarm data.

All functions accept a raw ``sqlite3.Connection`` so the caller controls
the connection lifecycle and can use in-memory DBs in tests.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd


# ---------------------------------------------------------------------------
# Latest decision
# ---------------------------------------------------------------------------

def load_latest_swarm_decision(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Return the most recent swarm decision row (1-row DF, or empty)."""
    clauses, params = _sym_tf_filter(symbol, timeframe)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return pd.read_sql_query(
        f"SELECT * FROM swarm_decisions {where} ORDER BY ts_ms DESC LIMIT 1",
        conn, params=params,
    )


# ---------------------------------------------------------------------------
# Agent votes
# ---------------------------------------------------------------------------

def load_swarm_votes(
    conn: sqlite3.Connection,
    decision_id: int | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    limit: int = 400,
) -> pd.DataFrame:
    """Load agent votes, optionally filtered by decision_id or symbol/tf."""
    if decision_id is not None:
        return pd.read_sql_query(
            "SELECT * FROM swarm_agent_votes WHERE decision_id = ?",
            conn, params=(decision_id,),
        )
    clauses, params = _sym_tf_filter(symbol, timeframe, prefix="sav")
    join = (
        "SELECT sav.*, sd.final_action AS decision_action, sd.agreement AS decision_agreement "
        "FROM swarm_agent_votes sav "
        "JOIN swarm_decisions sd ON sav.decision_id = sd.id"
    )
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return pd.read_sql_query(
        f"{join}{where} ORDER BY sav.ts_ms DESC LIMIT {int(limit)}",
        conn, params=params,
    )


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

def load_swarm_outcomes(
    conn: sqlite3.Connection,
    decision_id: int | None = None,
    symbol: str | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Load forward-outcome records."""
    if decision_id is not None:
        return pd.read_sql_query(
            "SELECT * FROM swarm_outcomes WHERE decision_id = ?",
            conn, params=(decision_id,),
        )
    sym_filter = "AND sd.symbol = ?" if symbol else ""
    params: tuple = (symbol,) if symbol else ()
    return pd.read_sql_query(
        f"""SELECT so.*, sd.ts_ms, sd.symbol, sd.final_action
            FROM swarm_outcomes so
            JOIN swarm_decisions sd ON so.decision_id = sd.id
            WHERE 1=1 {sym_filter}
            ORDER BY sd.ts_ms DESC LIMIT {int(limit)}""",
        conn, params=params,
    )


# ---------------------------------------------------------------------------
# Weight history
# ---------------------------------------------------------------------------

def load_swarm_weight_history(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    timeframe: str | None = None,
    days: int = 30,
) -> pd.DataFrame:
    """Load weight snapshots from the last N days."""
    import time
    cutoff_ms = int((time.time() - days * 86400) * 1000)
    clauses, params = _sym_tf_filter(symbol, timeframe)
    clauses.append("ts_ms >= ?")
    params.append(cutoff_ms)
    where = f"WHERE {' AND '.join(clauses)}"
    return pd.read_sql_query(
        f"SELECT * FROM swarm_weight_snapshots {where} ORDER BY ts_ms DESC",
        conn, params=params,
    )


# ---------------------------------------------------------------------------
# Promotion status
# ---------------------------------------------------------------------------

def load_swarm_promotion_status(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Load the latest promotion report row."""
    clauses, params = _sym_tf_filter(symbol, timeframe)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return pd.read_sql_query(
        f"SELECT * FROM swarm_promotion_reports {where} ORDER BY created_ms DESC LIMIT 1",
        conn, params=params,
    )


def load_swarm_promotion_history(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    limit: int = 50,
) -> pd.DataFrame:
    """Load recent promotion reports."""
    sym_filter = "WHERE symbol = ?" if symbol else ""
    params: tuple = (symbol,) if symbol else ()
    return pd.read_sql_query(
        f"SELECT * FROM swarm_promotion_reports {sym_filter} ORDER BY created_ms DESC LIMIT {int(limit)}",
        conn, params=params,
    )


# ---------------------------------------------------------------------------
# Decisions (bulk)
# ---------------------------------------------------------------------------

def load_swarm_decisions(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    timeframe: str | None = None,
    mode: str | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Load recent swarm decisions with optional filters."""
    clauses, params = _sym_tf_filter(symbol, timeframe)
    if mode:
        clauses.append("mode = ?")
        params.append(mode)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return pd.read_sql_query(
        f"SELECT * FROM swarm_decisions {where} ORDER BY ts_ms DESC LIMIT {int(limit)}",
        conn, params=params,
    )


# ---------------------------------------------------------------------------
# Veto ledger
# ---------------------------------------------------------------------------

def load_veto_ledger(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Load all veto events with agent and decision context."""
    sym_filter = "AND sd.symbol = ?" if symbol else ""
    params: tuple = (symbol,) if symbol else ()
    df = pd.read_sql_query(
        f"""SELECT sav.ts_ms, sav.agent_id, sav.block_reasons_json,
                   sd.final_action AS swarm_action, sd.agreement,
                   sd.vetoed AS decision_vetoed
            FROM swarm_agent_votes sav
            JOIN swarm_decisions sd ON sav.decision_id = sd.id
            WHERE sav.veto = 1 {sym_filter}
            ORDER BY sav.ts_ms DESC LIMIT {int(limit)}""",
        conn, params=params,
    )
    if not df.empty:
        df["reasons"] = df["block_reasons_json"].apply(
            lambda x: ", ".join(json.loads(x)) if x else ""
        )
    return df


# ---------------------------------------------------------------------------
# Decision detail (single decision drill-down)
# ---------------------------------------------------------------------------

def load_decision_detail(
    conn: sqlite3.Connection,
    decision_id: int,
) -> dict:
    """Full drill-down for a single swarm decision: decision + votes + baseline + outcome."""
    dec = pd.read_sql_query(
        "SELECT * FROM swarm_decisions WHERE id = ?", conn, params=(decision_id,),
    )
    votes = pd.read_sql_query(
        "SELECT * FROM swarm_agent_votes WHERE decision_id = ?",
        conn, params=(decision_id,),
    )
    baseline = pd.DataFrame()
    outcome = pd.DataFrame()
    if not dec.empty:
        ts_ms = int(dec.iloc[0]["ts_ms"])
        symbol = dec.iloc[0]["symbol"]
        baseline = pd.read_sql_query(
            "SELECT * FROM decision_log WHERE ts_ms = ? AND symbol = ? LIMIT 1",
            conn, params=(ts_ms, symbol),
        )
        outcome = pd.read_sql_query(
            "SELECT * FROM swarm_outcomes WHERE decision_id = ?",
            conn, params=(decision_id,),
        )
    return {
        "decision": dec,
        "votes": votes,
        "baseline": baseline,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Loss clusters
# ---------------------------------------------------------------------------

def load_swarm_loss_clusters(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    days: int = 30,
) -> pd.DataFrame:
    """Identify clusters of consecutive vetoed-correct or losing decisions."""
    import time
    cutoff_ms = int((time.time() - days * 86400) * 1000)
    sym_filter = "AND sd.symbol = ?" if symbol else ""
    params: tuple = (cutoff_ms,) + ((symbol,) if symbol else ())
    return pd.read_sql_query(
        f"""SELECT sd.ts_ms, sd.symbol, sd.final_action, sd.agreement,
                   sd.vetoed, so.outcome_label, so.forward_60m_bps, so.mae_bps
            FROM swarm_decisions sd
            LEFT JOIN swarm_outcomes so ON sd.id = so.decision_id
            WHERE sd.ts_ms >= ? {sym_filter}
            ORDER BY sd.ts_ms""",
        conn, params=params,
    )


# ---------------------------------------------------------------------------
# Score calibration
# ---------------------------------------------------------------------------

def load_swarm_score_calibration(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    days: int = 30,
) -> pd.DataFrame:
    """Load opportunity scores vs realized forward returns for calibration analysis."""
    import time
    cutoff_ms = int((time.time() - days * 86400) * 1000)
    sym_filter = "AND sd.symbol = ?" if symbol else ""
    params: tuple = (cutoff_ms,) + ((symbol,) if symbol else ())
    return pd.read_sql_query(
        f"""SELECT sd.ts_ms, sd.symbol, sd.final_action, sd.final_conf,
                   sd.agreement, sd.entropy, sd.vetoed,
                   so.forward_60m_bps, so.mae_bps, so.mfe_bps,
                   so.was_veto_correct, so.outcome_label
            FROM swarm_decisions sd
            LEFT JOIN swarm_outcomes so ON sd.id = so.decision_id
            WHERE sd.ts_ms >= ? {sym_filter}
            ORDER BY sd.ts_ms""",
        conn, params=params,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym_tf_filter(
    symbol: str | None,
    timeframe: str | None = None,
    prefix: str = "",
) -> tuple[list[str], list]:
    """Build WHERE clauses and params for optional symbol/timeframe filters."""
    col_prefix = f"{prefix}." if prefix else ""
    clauses: list[str] = []
    params: list = []
    if symbol:
        clauses.append(f"{col_prefix}symbol = ?")
        params.append(symbol)
    if timeframe:
        clauses.append(f"{col_prefix}timeframe = ?")
        params.append(timeframe)
    return clauses, params
