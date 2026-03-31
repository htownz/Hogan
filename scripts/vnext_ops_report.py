#!/usr/bin/env python3
"""Generate concise operator reports from observability primitives."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hogan_bot.observability import observability_health_report  # noqa: E402
from hogan_bot.storage import get_connection  # noqa: E402


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _to_markdown(report: dict, *, hours: float) -> str:
    lines: list[str] = []
    lines.append(f"# Hogan Ops Report ({hours:.0f}h)")
    lines.append("")
    lines.append(f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Alerts: **{len(report.get('alerts', []))}**")
    lines.append(f"- Stale agents: **{report.get('stale_agent_count', 0)}**")
    lines.append("")

    ex = report.get("execution", {})
    lines.append("## Execution")
    lines.append(
        f"- Orders: `{ex.get('total')}` | Failed: `{ex.get('failed')}` | "
        f"Fail rate: `{ex.get('fail_rate')}`"
    )
    lines.append("")

    bc = report.get("block_reasons", {})
    lines.append("## Decision Funnel")
    lines.append(
        f"- Rows scanned: `{bc.get('rows_scanned', 0)}` | "
        f"Rows with blocks: `{bc.get('rows_with_blocks', 0)}` | "
        f"Hold with no reason: `{bc.get('hold_with_no_reason', 0)}`"
    )
    top = list((bc.get("counts") or {}).items())[:10]
    for reason, count in top:
        lines.append(f"- `{reason}`: {count}")
    lines.append("")

    lines.append("## Integrity")
    integ = report.get("integrity", {})
    lines.append(f"- Overall OK: `{integ.get('ok')}`")
    lines.append(f"- sqlite_integrity_ok: `{integ.get('sqlite_integrity_ok')}`")
    lines.append(f"- orphan_fills: `{integ.get('orphan_fills')}`")
    lines.append(f"- orphan_decision_links: `{integ.get('orphan_decision_links')}`")
    lines.append(f"- invalid_trade_timestamps: `{integ.get('invalid_trade_timestamps')}`")
    lines.append("")

    lines.append("## Calendar / Drift")
    mc = report.get("macro_calendar", {})
    md = report.get("model_drift", {})
    sd = report.get("swarm_drift", {})
    lines.append(
        f"- Macro calendar latest event: `{mc.get('latest_event_date')}` "
        f"({mc.get('days_remaining')} days remaining)"
    )
    lines.append(
        f"- Model drift: `{md.get('model_path')}` exists=`{md.get('exists')}` "
        f"age_hours=`{md.get('age_hours')}`"
    )
    lines.append(
        f"- Shadow/active drift acceptable: `{sd.get('drift_acceptable')}` "
        f"warnings=`{sd.get('warnings', [])}`"
    )
    lines.append("")

    alerts = report.get("alerts", [])
    lines.append("## Alerts")
    if alerts:
        for a in alerts:
            lines.append(f"- **{a.get('level','info').upper()}** `{a.get('code')}`: {a.get('message')}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Hogan vNext ops report")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--output-dir", default="reports/validation")
    p.add_argument("--json", action="store_true", help="Write JSON alongside markdown")
    args = p.parse_args()

    conn = get_connection(args.db)
    try:
        rpt = observability_health_report(conn, since_hours=args.hours)
    finally:
        conn.close()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    md_path = out_dir / f"ops_report_{stamp}.md"
    md_path.write_text(_to_markdown(rpt, hours=args.hours), encoding="utf-8")
    if args.json:
        json_path = out_dir / f"ops_report_{stamp}.json"
        json_path.write_text(json.dumps(rpt, indent=2, default=str), encoding="utf-8")
        print(f"Wrote: {md_path}")
        print(f"Wrote: {json_path}")
    else:
        print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
