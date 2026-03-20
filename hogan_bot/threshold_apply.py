"""Threshold apply CLI — create/activate threshold bundles with audit trail.

Supports dry-run mode and operator acknowledgement gating to prevent
accidental threshold escalation.

Usage:
    python -m hogan_bot.threshold_apply \
        --agent-id risk_steward_v1 \
        --bundle-id relaxed_shadow \
        --values '{"vol_spike_multiple": 7.0, "hard_veto": false}' \
        --reason "Reduce over-veto during swarm shadow proving" \
        --operator ben \
        --ack I_APPROVE_THRESHOLD_CHANGE
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

from hogan_bot.threshold_registry import (
    activate_bundle,
    diff_bundles,
    get_active_bundle,
    log_threshold_changes,
    save_bundle,
)
from hogan_bot.threshold_types import ThresholdBundle, ThresholdChange

logger = logging.getLogger(__name__)


def apply_threshold(
    conn: sqlite3.Connection,
    agent_id: str,
    bundle_id: str,
    values: dict,
    reason: str,
    operator: str,
    *,
    dry_run: bool = False,
    require_ack: bool = True,
    ack_phrase: str = "I_APPROVE_THRESHOLD_CHANGE",
    provided_ack: str | None = None,
    notes: str = "",
) -> dict:
    """Create and optionally activate a threshold bundle.

    Returns a dict summarizing the action taken (for both dry-run and real).
    """
    from hogan_bot.storage import _create_schema
    _create_schema(conn)

    old_bundle = get_active_bundle(agent_id, conn)
    old_version = old_bundle.version if old_bundle else 0
    new_version = old_version + 1

    new_bundle = ThresholdBundle(
        bundle_id=bundle_id, agent_id=agent_id, version=new_version,
        values=values, notes=notes, active=False,
    )

    changes: list[ThresholdChange] = []
    if old_bundle:
        changes = diff_bundles(old_bundle, new_bundle)

    result = {
        "agent_id": agent_id,
        "bundle_id": bundle_id,
        "old_version": old_version,
        "new_version": new_version,
        "changes_count": len(changes),
        "changes": [
            {"field": c.field_name, "old": c.old_value, "new": c.new_value}
            for c in changes
        ],
        "dry_run": dry_run,
        "activated": False,
        "reason": reason,
        "operator": operator,
    }

    if dry_run:
        result["message"] = "Dry run — no changes applied."
        return result

    if require_ack and provided_ack != ack_phrase:
        result["message"] = f"Activation blocked — provide --ack {ack_phrase}"
        result["activated"] = False
        save_bundle(new_bundle, conn)
        return result

    new_bundle.active = True
    save_bundle(new_bundle, conn)
    activate_bundle(agent_id, bundle_id, new_version, operator, reason, conn)

    if changes:
        for c in changes:
            c.reason = reason
            c.operator = operator
        log_threshold_changes(changes, conn)

    result["activated"] = True
    result["message"] = f"Bundle {bundle_id} v{new_version} activated for {agent_id}."
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply Threshold Bundle")
    parser.add_argument("--db", default="data/hogan.db")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--values", required=True, help="JSON dict of threshold values")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--operator", default="local_operator")
    parser.add_argument("--notes", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ack", default=None, help="Acknowledgement phrase for activation")
    parser.add_argument("--no-require-ack", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    try:
        values = json.loads(args.values)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON for --values: %s", e)
        sys.exit(1)

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    try:
        from hogan_bot.storage import _create_schema
        _create_schema(conn)

        result = apply_threshold(
            conn, args.agent_id, args.bundle_id, values,
            args.reason, args.operator,
            dry_run=args.dry_run,
            require_ack=not args.no_require_ack,
            provided_ack=args.ack,
            notes=args.notes,
        )
    finally:
        conn.close()

    print(json.dumps(result, indent=2))

    if result.get("changes"):
        print("\nChanges:")
        for c in result["changes"]:
            print(f"  {c['field']}: {c['old']} → {c['new']}")

    if not result["activated"] and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
