"""DB write helpers for swarm decision logging.

All functions accept a raw ``sqlite3.Connection`` and write to the
``swarm_decisions``, ``swarm_agent_votes``, and ``swarm_weight_snapshots``
tables defined in ``storage._create_schema()``.
"""
from __future__ import annotations

import json
import sqlite3
import time

from hogan_bot.swarm_decision.types import AgentVote, SwarmDecision


def log_swarm_decision(
    conn: sqlite3.Connection,
    ts_ms: int,
    symbol: str,
    timeframe: str,
    decision: SwarmDecision,
    mode: str,
    as_of_ms: int | None = None,
) -> int:
    """Persist one SwarmDecision row.  Returns the new row id."""
    cur = conn.execute(
        """
        INSERT INTO swarm_decisions (
            ts_ms, symbol, timeframe, as_of_ms, mode,
            final_action, final_conf, final_scale,
            agreement, entropy, vetoed,
            block_reasons_json, weights_json, decision_json
        ) VALUES (?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?)
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
    """Persist a weight snapshot.  Returns the new row id."""
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
