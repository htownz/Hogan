#!/usr/bin/env python3
"""Run walk-forward + optional swarm certification; archive outputs under reports/validation/.

Usage:
  python scripts/run_validation_battery.py
  python scripts/run_validation_battery.py --db data/hogan.db --skip-certification
  python scripts/run_validation_battery.py --wf-extra --ml-sizer --macro-sitout

Exits non-zero if walk-forward fails its promotion gate (walk-forward process exit code).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_cmd(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return int(p.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hogan validation battery (walk-forward + certification)")
    parser.add_argument("--db", default="data/hogan.db", help="SQLite DB path")
    parser.add_argument(
        "--output-dir",
        default="reports/validation",
        help="Directory for JSON reports, logs, and manifest",
    )
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--skip-certification", action="store_true")
    parser.add_argument("--cert-bars", type=int, default=3000, help="Bars for swarm_certification.py")
    parser.add_argument(
        "--wf-extra",
        action="store_true",
        help="Pass --ml-sizer --macro-sitout to walk-forward (canonical stress profile)",
    )
    args = parser.parse_args(argv)

    out_dir = (_PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    manifest: dict = {
        "stamp_utc": stamp,
        "db": args.db,
        "steps": [],
    }
    wf_rc = 0
    cert_rc = 0

    py = sys.executable

    if not args.skip_walk_forward:
        wf_json = out_dir / f"wf_{stamp}.json"
        wf_log = out_dir / f"wf_{stamp}.log"
        wf_cmd = [
            py, "-m", "hogan_bot.walk_forward",
            "--db", args.db,
            "--symbol", args.symbol,
            "--timeframe", args.timeframe,
            "--n-splits", str(args.n_splits),
            "--output", str(wf_json.relative_to(_PROJECT_ROOT)),
        ]
        if args.wf_extra:
            wf_cmd.extend(["--ml-sizer", "--macro-sitout"])
        wf_rc = _run_cmd(wf_cmd, wf_log, cwd=_PROJECT_ROOT)
        manifest["steps"].append({
            "name": "walk_forward",
            "cmd": wf_cmd,
            "exit_code": wf_rc,
            "report_json": str(wf_json.relative_to(_PROJECT_ROOT)),
            "log": str(wf_log.relative_to(_PROJECT_ROOT)),
        })

    if not args.skip_certification:
        cert_log = out_dir / f"cert_{stamp}.log"
        cert_cmd = [
            py,
            str(_PROJECT_ROOT / "scripts" / "swarm_certification.py"),
            "--db", args.db,
            "--symbol", args.symbol,
            "--bars", str(args.cert_bars),
            "--scratch-db",
        ]
        cert_rc = _run_cmd(cert_cmd, cert_log, cwd=_PROJECT_ROOT)
        manifest["steps"].append({
            "name": "swarm_certification",
            "cmd": cert_cmd,
            "exit_code": cert_rc,
            "log": str(cert_log.relative_to(_PROJECT_ROOT)),
        })

    man_path = out_dir / f"manifest_{stamp}.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written: {man_path.relative_to(_PROJECT_ROOT)}")
    for step in manifest["steps"]:
        print(f"  {step['name']}: exit {step['exit_code']}")

    # Walk-forward gate failure is the primary CI-style failure
    if wf_rc != 0:
        return wf_rc
    if cert_rc != 0:
        return cert_rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
