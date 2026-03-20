"""DB write helpers for swarm decision logging.

All functions accept a raw ``sqlite3.Connection`` and write to the
``swarm_decisions``, ``swarm_agent_votes``, and ``swarm_weight_snapshots``
tables defined in ``storage._create_schema()``.
"""
from __future__ import annotations

import json
import logging
import sqlite3

from hogan_bot.swarm_decision.types import AgentVote, SwarmDecision

logger = logging.getLogger(__name__)


def log_swarm_decision(
    conn: sqlite3.Connection,
    ts_ms: int,
    symbol: str,
    timeframe: str,
    decision: SwarmDecision,
    mode: str,
    as_of_ms: int | None = None,
    regime: str | None = None,
    stall_state: str | None = None,
) -> int:
    """Persist one SwarmDecision row.  Returns the new row id, or -1 on failure."""
    try:
        return _log_swarm_decision_inner(conn, ts_ms, symbol, timeframe, decision, mode, as_of_ms, regime, stall_state)
    except Exception as exc:
        logger.error("log_swarm_decision failed for %s at %d: %s", symbol, ts_ms, exc)
        return -1


def _log_swarm_decision_inner(conn, ts_ms, symbol, timeframe, decision, mode, as_of_ms, regime, stall_state) -> int:
    cur = conn.execute(
        """
        INSERT INTO swarm_decisions (
            ts_ms, symbol, timeframe, as_of_ms, mode,
            final_action, final_conf, final_scale,
            agreement, entropy, vetoed,
            block_reasons_json, weights_json, decision_json,
            pre_veto_action, pre_veto_confidence, pre_veto_agreement, pre_veto_entropy,
            dominant_veto_agent, veto_count, veto_agents_json, regime, stall_state
        ) VALUES (?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?,?, ?,?,?,?,?)
        """,
        (
            int(ts_ms),
            symbol,
            timeframe,
            int(as_of_ms) if as_of_ms is not None else None,
            mode,
            decision.final_action,
            round(decision.final_confidence, 6),
            round(decision.final_size_scale, 6),
            round(decision.agreement, 6),
            round(decision.entropy, 6),
            1 if decision.vetoed else 0,
            json.dumps(decision.block_reasons),
            json.dumps({k: round(v, 6) for k, v in decision.weights_used.items()}),
            json.dumps(decision.to_dict()),
            decision.pre_veto_action,
            round(decision.pre_veto_confidence, 6) if decision.pre_veto_confidence is not None else None,
            round(decision.pre_veto_agreement, 6) if decision.pre_veto_agreement is not None else None,
            round(decision.pre_veto_entropy, 6) if decision.pre_veto_entropy is not None else None,
            decision.dominant_veto_agent,
            decision.veto_count,
            json.dumps(decision.veto_agents) if decision.veto_agents else None,
            regime,
            stall_state,
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def log_agent_votes(
    conn: sqlite3.Connection,
    ts_ms: int,
    symbol: str,
    timeframe: str,
    votes: list[AgentVote],
    as_of_ms: int | None = None,
    decision_id: int | None = None,
) -> None:
    """Persist individual agent votes for a single bar."""
    if decision_id is not None and decision_id < 0:
        logger.warning("log_agent_votes skipped: invalid decision_id=%d for %s at %d", decision_id, symbol, ts_ms)
        return
    try:
        return _log_agent_votes_inner(conn, ts_ms, symbol, timeframe, votes, as_of_ms, decision_id)
    except Exception as exc:
        logger.error("log_agent_votes failed for %s at %d: %s", symbol, ts_ms, exc)


def _log_agent_votes_inner(conn, ts_ms, symbol, timeframe, votes, as_of_ms, decision_id) -> None:
    rows = []
    for v in votes:
        rows.append((
            int(ts_ms),
            symbol,
            timeframe,
            int(as_of_ms) if as_of_ms is not None else None,
            v.agent_id,
            v.action,
            round(v.confidence, 6),
            round(v.expected_edge_bps, 2) if v.expected_edge_bps is not None else None,
            round(v.size_scale, 6),
            1 if v.veto else 0,
            json.dumps(v.block_reasons),
            json.dumps(v.to_dict()),
            decision_id,
        ))
    conn.executemany(
        """
        INSERT INTO swarm_agent_votes (
            ts_ms, symbol, timeframe, as_of_ms,
            agent_id, action, confidence, expected_edge_bps,
            size_scale, veto, block_reasons_json, vote_json,
            decision_id
        ) VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?, ?)
        """,
        rows,
    )
    conn.commit()


def log_weight_snapshot(
    conn: sqlite3.Connection,
    ts_ms: int,
    symbol: str,
    timeframe: str,
    weights: dict[str, float],
    regime: str | None = None,
    source: str = "static",
    notes: str | None = None,
) -> int:
    """Persist a weight snapshot.  Returns the new row id, or -1 on failure."""
    try:
        return _log_weight_snapshot_inner(conn, ts_ms, symbol, timeframe, weights, regime, source, notes)
    except Exception as exc:
        logger.error("log_weight_snapshot failed for %s at %d: %s", symbol, ts_ms, exc)
        return -1


def _log_weight_snapshot_inner(conn, ts_ms, symbol, timeframe, weights, regime, source, notes) -> int:
    cur = conn.execute(
        """
        INSERT INTO swarm_weight_snapshots (
            ts_ms, symbol, timeframe, regime,
            weights_json, source, notes
        ) VALUES (?,?,?,?, ?,?,?)
        """,
        (
            int(ts_ms),
            symbol,
            timeframe,
            regime,
            json.dumps({k: round(v, 6) for k, v in weights.items()}),
            source,
            notes,
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]
