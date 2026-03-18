"""Threshold bundle registry — versioned CRUD, activation, and diff.

Every threshold change is persisted so the operator has a full audit
trail of what was active at any point in time.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone

from hogan_bot.threshold_types import ThresholdBundle, ThresholdChange


def _ensure_tables(conn: sqlite3.Connection) -> None:
    from hogan_bot.storage import _create_schema
    _create_schema(conn)


def get_active_bundle(agent_id: str, conn: sqlite3.Connection) -> ThresholdBundle | None:
    row = conn.execute(
        """SELECT bundle_id, version, values_json, notes
           FROM swarm_threshold_bundles
           WHERE agent_id = ? AND active = 1
           ORDER BY ts_ms DESC LIMIT 1""",
        (agent_id,),
    ).fetchone()
    if not row:
        return None
    return ThresholdBundle(
        bundle_id=row[0], agent_id=agent_id, version=row[1],
        values=json.loads(row[2]), notes=row[3] or "", active=True,
    )


def save_bundle(bundle: ThresholdBundle, conn: sqlite3.Connection) -> int:
    ts_ms = int(time.time() * 1000)
    cur = conn.execute(
        """INSERT INTO swarm_threshold_bundles
           (ts_ms, agent_id, bundle_id, version, values_json, notes, active)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (ts_ms, bundle.agent_id, bundle.bundle_id, bundle.version,
         json.dumps(bundle.values), bundle.notes, 1 if bundle.active else 0),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def activate_bundle(
    agent_id: str, bundle_id: str, version: int,
    operator: str, reason: str, conn: sqlite3.Connection,
) -> None:
    ts_ms = int(time.time() * 1000)
    conn.execute(
        "UPDATE swarm_threshold_bundles SET active = 0 WHERE agent_id = ? AND active = 1",
        (agent_id,),
    )
    conn.execute(
        """UPDATE swarm_threshold_bundles SET active = 1
           WHERE agent_id = ? AND bundle_id = ? AND version = ?""",
        (agent_id, bundle_id, version),
    )
    row = conn.execute(
        "SELECT values_json FROM swarm_threshold_bundles WHERE agent_id = ? AND bundle_id = ? AND version = ?",
        (agent_id, bundle_id, version),
    ).fetchone()
    if row:
        conn.execute(
            """INSERT INTO swarm_threshold_changes
               (ts_ms, agent_id, bundle_id, field_name, old_value_json, new_value_json, reason, operator)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts_ms, agent_id, bundle_id, "__activation__", "null", row[0], reason, operator),
        )
    conn.commit()


def diff_bundles(old: ThresholdBundle, new: ThresholdBundle) -> list[ThresholdChange]:
    changes: list[ThresholdChange] = []
    ts = datetime.now(timezone.utc).isoformat()
    all_keys = set(old.values) | set(new.values)
    for key in sorted(all_keys):
        ov = old.values.get(key)
        nv = new.values.get(key)
        if ov != nv:
            changes.append(ThresholdChange(
                ts=ts, agent_id=new.agent_id, bundle_id=new.bundle_id,
                field_name=key, old_value=ov, new_value=nv,
                reason="bundle_diff", operator="system",
            ))
    return changes


def log_threshold_changes(changes: list[ThresholdChange], conn: sqlite3.Connection) -> None:
    ts_ms = int(time.time() * 1000)
    rows = [
        (ts_ms, c.agent_id, c.bundle_id, c.field_name,
         json.dumps(c.old_value), json.dumps(c.new_value),
         c.reason, c.operator)
        for c in changes
    ]
    conn.executemany(
        """INSERT INTO swarm_threshold_changes
           (ts_ms, agent_id, bundle_id, field_name, old_value_json, new_value_json, reason, operator)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def list_bundles(agent_id: str, conn: sqlite3.Connection) -> list[ThresholdBundle]:
    rows = conn.execute(
        """SELECT bundle_id, version, values_json, notes, active
           FROM swarm_threshold_bundles
           WHERE agent_id = ? ORDER BY ts_ms DESC""",
        (agent_id,),
    ).fetchall()
    return [
        ThresholdBundle(
            bundle_id=r[0], agent_id=agent_id, version=r[1],
            values=json.loads(r[2]), notes=r[3] or "", active=bool(r[4]),
        )
        for r in rows
    ]


def get_change_history(
    agent_id: str, conn: sqlite3.Connection, limit: int = 50,
) -> list[dict]:
    rows = conn.execute(
        """SELECT ts_ms, bundle_id, field_name, old_value_json, new_value_json, reason, operator
           FROM swarm_threshold_changes
           WHERE agent_id = ? ORDER BY ts_ms DESC LIMIT ?""",
        (agent_id, limit),
    ).fetchall()
    return [
        {
            "ts_ms": r[0], "bundle_id": r[1], "field_name": r[2],
            "old_value": json.loads(r[3]) if r[3] else None,
            "new_value": json.loads(r[4]) if r[4] else None,
            "reason": r[5], "operator": r[6],
        }
        for r in rows
    ]
