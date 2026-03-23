#!/usr/bin/env python3
"""Reset swarm agent modes in SQLite (e.g. after stale auto_quarantine).

Default: set every agent whose latest mode is not ``active`` back to ``active``.

Usage:
  python scripts/reset_swarm_agent_modes.py
  python scripts/reset_swarm_agent_modes.py --db data/hogan.db --dry-run
"""
from __future__ import annotations

import argparse
import sqlite3
import time


def main() -> None:
    p = argparse.ArgumentParser(description="Reset swarm_agent_modes rows to active")
    p.add_argument("--db", default="data/hogan.db", help="SQLite database path")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned resets only",
    )
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        rows = conn.execute(
            """SELECT agent_id, mode, reason, operator
               FROM swarm_agent_modes
               WHERE id IN (SELECT MAX(id) FROM swarm_agent_modes GROUP BY agent_id)
               ORDER BY agent_id"""
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"DB error (table missing?): {e}")
        return
    finally:
        conn.close()

    print("Latest agent modes:")
    for r in rows:
        print(f"  {r[0]:28}  mode={r[1]:15}  op={r[3]}  reason={r[2][:60]!r}")

    to_reset = [r for r in rows if r[1] != "active"]
    if not to_reset:
        print("\nNothing to reset (all active).")
        return

    print(f"\nWould reset {len(to_reset)} agent(s) to active.")
    if args.dry_run:
        return

    conn = sqlite3.connect(args.db)
    ts_ms = int(time.time() * 1000)
    for r in to_reset:
        agent_id = r[0]
        conn.execute(
            "INSERT INTO swarm_agent_modes (ts_ms, agent_id, mode, reason, operator) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                ts_ms,
                agent_id,
                "active",
                "manual reset via scripts/reset_swarm_agent_modes.py",
                "manual",
            ),
        )
        print(f"  RESET {agent_id} -> active")
    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
