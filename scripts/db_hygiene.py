#!/usr/bin/env python3
"""DB hygiene CLI — table stats, decision transparency report, agent mode check, prune.

Usage::

    # Report table sizes and agent mode staleness
    python scripts/db_hygiene.py --db data/hogan.db

    # Full observability health report (last 24h)
    python scripts/db_hygiene.py --db data/hogan.db --report --hours 24

    # Dry-run prune (show what would be deleted)
    python scripts/db_hygiene.py --db data/hogan.db --prune --retain-days 90

    # Actually prune + vacuum
    python scripts/db_hygiene.py --db data/hogan.db --prune --retain-days 90 --execute

    # JSON output for CI
    python scripts/db_hygiene.py --db data/hogan.db --report --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hogan_bot.observability import (
    check_agent_mode_staleness,
    get_db_table_stats,
    observability_health_report,
    prune_old_rows,
    vacuum_db,
)


def _ts_to_str(ts_ms: int | None) -> str:
    if ts_ms is None or ts_ms <= 0:
        return "n/a"
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hogan DB hygiene and observability")
    parser.add_argument("--db", default="data/hogan.db", help="Path to SQLite database")
    parser.add_argument("--report", action="store_true", help="Full observability health report")
    parser.add_argument("--hours", type=float, default=24.0, help="Lookback hours for report")
    parser.add_argument("--symbol", default=None, help="Filter by symbol")
    parser.add_argument("--prune", action="store_true", help="Show/execute retention pruning")
    parser.add_argument("--retain-days", type=int, default=90, help="Days to retain when pruning")
    parser.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run)")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM after prune")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    if args.report:
        rpt = observability_health_report(conn, since_hours=args.hours, symbol=args.symbol)
        if args.json:
            print(json.dumps(rpt, indent=2, default=str))
        else:
            _print_report(rpt)
        conn.close()
        return

    if args.prune:
        deleted = prune_old_rows(conn, retain_days=args.retain_days, dry_run=not args.execute)
        action = "DELETED" if args.execute else "WOULD DELETE"
        if args.json:
            print(json.dumps({"action": action, "retain_days": args.retain_days, "rows": deleted}))
        else:
            print(f"\n{'=' * 50}")
            print(f"  PRUNE ({action}) — retain {args.retain_days} days")
            print(f"{'=' * 50}")
            for table, n in deleted.items():
                print(f"  {table:<30} {n:>8} rows")
        if args.execute and args.vacuum:
            ok = vacuum_db(conn)
            print(f"\n  VACUUM: {'OK' if ok else 'FAILED'}")
        conn.close()
        return

    stats = get_db_table_stats(conn)
    modes = check_agent_mode_staleness(conn)

    if args.json:
        output = {
            "tables": [{"name": t.name, "rows": t.row_count,
                         "oldest": _ts_to_str(t.oldest_ts_ms),
                         "newest": _ts_to_str(t.newest_ts_ms)} for t in stats],
            "agent_modes": [{"agent_id": m.agent_id, "mode": m.mode,
                             "age_hours": m.age_hours, "is_stale": m.is_stale} for m in modes],
        }
        print(json.dumps(output, indent=2))
    else:
        _print_stats(stats, modes)

    conn.close()


def _print_stats(stats: list, modes: list) -> None:
    print(f"\n{'=' * 70}")
    print("  DB TABLE STATS")
    print(f"{'=' * 70}")
    hdr = f"  {'Table':<30} {'Rows':>10} {'Oldest':>18} {'Newest':>18}"
    print(hdr)
    print(f"  {'-' * 66}")
    for t in stats:
        print(f"  {t.name:<30} {t.row_count:>10,} {_ts_to_str(t.oldest_ts_ms):>18} {_ts_to_str(t.newest_ts_ms):>18}")

    if modes:
        print(f"\n{'=' * 70}")
        print("  AGENT MODE STATUS")
        print(f"{'=' * 70}")
        for m in modes:
            stale_tag = " *** STALE ***" if m.is_stale else ""
            print(f"  {m.agent_id:<25} mode={m.mode:<15} age={m.age_hours:.0f}h{stale_tag}")

    print()


def _print_report(rpt: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  OBSERVABILITY HEALTH REPORT (last {rpt['period_hours']}h)")
    print(f"{'=' * 70}")

    br = rpt.get("block_reasons", {})
    print(f"\n  BLOCK REASONS ({br.get('rows_scanned', 0)} decisions scanned)")
    print(f"    Rows with blocks: {br.get('rows_with_blocks', 0)}")
    print(f"    Hold with no reason: {br.get('hold_with_no_reason', 0)}")
    print(f"    Action distribution: {br.get('action_distribution', {})}")
    counts = br.get("counts", {})
    if counts:
        print(f"\n    {'Reason':<40} {'Count':>6}")
        print(f"    {'-' * 48}")
        for reason, count in counts.items():
            print(f"    {reason:<40} {count:>6}")

    by_regime = rpt.get("block_reasons_by_regime", {})
    if by_regime:
        print("\n  BLOCK REASONS BY REGIME")
        for regime, reasons in by_regime.items():
            top3 = list(reasons.items())[:3]
            top_str = ", ".join(f"{r}={c}" for r, c in top3)
            print(f"    {regime:<16} {top_str}")

    modes = rpt.get("agent_modes", [])
    if modes:
        print(f"\n  AGENT MODES ({rpt.get('stale_agent_count', 0)} stale)")
        for m in modes:
            tag = " *** STALE ***" if m["is_stale"] else ""
            print(f"    {m['agent_id']:<25} {m['mode']:<15} {m['age_hours']:.0f}h{tag}")

    alerts = rpt.get("alerts", [])
    if alerts:
        print(f"\n  ALERTS ({len(alerts)})")
        for a in alerts:
            print(f"    [{a['level'].upper()}] {a['code']}: {a['message']}")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
