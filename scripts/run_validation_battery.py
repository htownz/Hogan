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
    parser.add_argument(
        "--single-wf",
        action="store_true",
        help="Run only one walk-forward scenario (legacy behavior).",
    )
    parser.add_argument(
        "--min-pass-scenarios",
        type=int,
        default=3,
        help="Minimum successful WF scenarios required in matrix mode.",
    )
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
    wf_matrix_results: list[dict] = []

    py = sys.executable

    if not args.skip_walk_forward:
        _wf_scenarios: list[tuple[str, list[str]]]
        if args.single_wf:
            extra = ["--ml-sizer", "--macro-sitout"] if args.wf_extra else []
            _wf_scenarios = [("single", extra)]
        else:
            _wf_scenarios = [
                ("baseline_no_ml", ["--no-ml"]),
                ("ml_filter", []),
                ("ml_sizer", ["--ml-sizer"]),
                ("ml_sizer_macro", ["--ml-sizer", "--macro-sitout"]),
                ("regime_models", ["--regime-models"]),
            ]

        for scen_name, scen_flags in _wf_scenarios:
            wf_json = out_dir / f"wf_{scen_name}_{stamp}.json"
            wf_log = out_dir / f"wf_{scen_name}_{stamp}.log"
            wf_cmd = [
                py, "-m", "hogan_bot.walk_forward",
                "--db", args.db,
                "--symbol", args.symbol,
                "--timeframe", args.timeframe,
                "--n-splits", str(args.n_splits),
                "--output", str(wf_json.relative_to(_PROJECT_ROOT)),
                *scen_flags,
            ]
            rc = _run_cmd(wf_cmd, wf_log, cwd=_PROJECT_ROOT)
            passes_gate = False
            if rc == 0 and wf_json.exists():
                try:
                    _wf_payload = json.loads(wf_json.read_text(encoding="utf-8"))
                    passes_gate = bool(_wf_payload.get("summary", {}).get("passes_gate", False))
                except Exception:
                    passes_gate = False
            scenario_ok = bool(rc == 0 and passes_gate)
            wf_matrix_results.append({
                "scenario": scen_name,
                "cmd": wf_cmd,
                "exit_code": rc,
                "passes_gate": passes_gate,
                "scenario_ok": scenario_ok,
                "report_json": str(wf_json.relative_to(_PROJECT_ROOT)),
                "log": str(wf_log.relative_to(_PROJECT_ROOT)),
            })
            manifest["steps"].append({
                "name": "walk_forward",
                "scenario": scen_name,
                "cmd": wf_cmd,
                "exit_code": rc,
                "passes_gate": passes_gate,
                "scenario_ok": scenario_ok,
                "report_json": str(wf_json.relative_to(_PROJECT_ROOT)),
                "log": str(wf_log.relative_to(_PROJECT_ROOT)),
            })
        _passes = sum(1 for r in wf_matrix_results if r.get("scenario_ok"))
        _required = 1 if args.single_wf else max(1, int(args.min_pass_scenarios))
        wf_rc = 0 if _passes >= _required else 2
        manifest["wf_matrix"] = {
            "single_wf": args.single_wf,
            "required_passes": _required,
            "actual_passes": _passes,
            "total": len(wf_matrix_results),
            "results": wf_matrix_results,
        }

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
