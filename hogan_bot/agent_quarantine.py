"""Agent quarantine control — set, retrieve, and enforce agent modes.

Modes:
- active: full influence (default)
- advisory_only: emits outputs, excluded from fusion weights and veto power
- no_veto: contributes scores/notes, veto=False enforced regardless of internal logic
- quarantined: excluded from runtime aggregation unless diagnostic flag is set
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone

from hogan_bot.threshold_types import AgentMode, AgentQuarantineState

_DEFAULT_MODE: AgentMode = "active"


def _table_exists(conn: sqlite3.Connection, name: str = "swarm_agent_modes") -> bool:
    r = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone()
    return bool(r and r[0])


def get_agent_mode(agent_id: str, conn: sqlite3.Connection) -> AgentMode:
    if not _table_exists(conn):
        return _DEFAULT_MODE
    row = conn.execute(
        "SELECT mode FROM swarm_agent_modes WHERE agent_id = ? ORDER BY ts_ms DESC LIMIT 1",
        (agent_id,),
    ).fetchone()
    return row[0] if row else _DEFAULT_MODE  # type: ignore[return-value]


def get_agent_state(agent_id: str, conn: sqlite3.Connection) -> AgentQuarantineState:
    if not _table_exists(conn):
        return AgentQuarantineState(agent_id=agent_id, mode="active", reason="default", operator="system", changed_at="")
    row = conn.execute(
        "SELECT mode, reason, operator, ts_ms FROM swarm_agent_modes WHERE agent_id = ? ORDER BY ts_ms DESC LIMIT 1",
        (agent_id,),
    ).fetchone()
    if not row:
        return AgentQuarantineState(agent_id=agent_id, mode="active", reason="default", operator="system", changed_at="")
    ts_iso = datetime.fromtimestamp(row[3] / 1000, tz=timezone.utc).isoformat() if row[3] else ""
    return AgentQuarantineState(agent_id=agent_id, mode=row[0], reason=row[1], operator=row[2], changed_at=ts_iso)


def set_agent_mode(
    agent_id: str, mode: AgentMode, operator: str, reason: str,
    conn: sqlite3.Connection,
) -> None:
    ts_ms = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO swarm_agent_modes (ts_ms, agent_id, mode, reason, operator) VALUES (?, ?, ?, ?, ?)",
        (ts_ms, agent_id, mode, reason, operator),
    )
    conn.commit()


def is_agent_veto_enabled(agent_id: str, conn: sqlite3.Connection) -> bool:
    mode = get_agent_mode(agent_id, conn)
    return mode == "active"


def is_agent_advisory_only(agent_id: str, conn: sqlite3.Connection) -> bool:
    mode = get_agent_mode(agent_id, conn)
    return mode == "advisory_only"


def is_agent_quarantined(agent_id: str, conn: sqlite3.Connection) -> bool:
    mode = get_agent_mode(agent_id, conn)
    return mode == "quarantined"


def get_all_agent_modes(conn: sqlite3.Connection) -> dict[str, AgentQuarantineState]:
    if not _table_exists(conn):
        return {}
    rows = conn.execute(
        """SELECT agent_id, mode, reason, operator, ts_ms
           FROM swarm_agent_modes
           WHERE id IN (SELECT MAX(id) FROM swarm_agent_modes GROUP BY agent_id)
           ORDER BY agent_id""",
    ).fetchall()
    result: dict[str, AgentQuarantineState] = {}
    for r in rows:
        ts_iso = datetime.fromtimestamp(r[4] / 1000, tz=timezone.utc).isoformat() if r[4] else ""
        result[r[0]] = AgentQuarantineState(
            agent_id=r[0], mode=r[1], reason=r[2], operator=r[3], changed_at=ts_iso,
        )
    return result


def get_mode_history(agent_id: str, conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT ts_ms, mode, reason, operator FROM swarm_agent_modes WHERE agent_id = ? ORDER BY ts_ms DESC LIMIT ?",
        (agent_id, limit),
    ).fetchall()
    return [
        {"ts_ms": r[0], "mode": r[1], "reason": r[2], "operator": r[3]}
        for r in rows
    ]
