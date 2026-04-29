#!/usr/bin/env python3
"""Summarize a Hogan validation package for promotion review."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _latest_manifest(output_dir: Path) -> Path:
    manifests = sorted(output_dir.glob("manifest_*.json"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        raise FileNotFoundError(f"No manifest_*.json files found in {output_dir}")
    return manifests[-1]


def _scenario_summary(entry: dict[str, Any]) -> dict[str, Any]:
    report_path = entry.get("report_json")
    payload: dict[str, Any] = {}
    if report_path:
        path = _resolve_path(report_path)
        if path.exists():
            payload = _load_json(path)

    summary = payload.get("summary", {}) if payload else {}
    return {
        "scenario": entry.get("scenario", "unknown"),
        "exit_code": entry.get("exit_code"),
        "passes_gate": bool(entry.get("passes_gate", summary.get("passes_gate", False))),
        "report_json": report_path,
        "n_windows": summary.get("n_windows"),
        "n_positive": summary.get("n_positive"),
        "total_trades": summary.get("total_trades"),
        "mean_return_pct": summary.get("mean_return_pct"),
        "mean_sharpe": summary.get("mean_sharpe"),
        "mean_calmar": summary.get("mean_calmar"),
        "worst_calmar": summary.get("worst_calmar"),
        "worst_drawdown_pct": summary.get("worst_drawdown_pct"),
        "gate_config": summary.get("gate_config", {}),
    }


def summarize_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    matrix = manifest.get("wf_matrix", {})
    scenarios = [_scenario_summary(entry) for entry in matrix.get("results", [])]
    actual_passes = sum(1 for s in scenarios if s["passes_gate"])
    required_passes = int(matrix.get("required_passes", 1))

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "db": manifest.get("db"),
        "required_passes": required_passes,
        "actual_passes": actual_passes,
        "recommendation": "PASS" if actual_passes >= required_passes else "HOLD",
        "scenarios": scenarios,
        "notes": [
            "Calmar is an active promotion gate when min_calmar is 0.0.",
            "A HOLD recommendation means do not scale autonomy or live-like size from this package.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Hogan validation package results")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest path. Defaults to latest reports/validation/manifest_*.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/validation",
        help="Directory used when selecting the latest manifest.",
    )
    parser.add_argument(
        "--output",
        default="reports/validation/current_baseline_summary.json",
        help="Path for the summary JSON.",
    )
    args = parser.parse_args(argv)

    out_dir = _resolve_path(args.output_dir)
    manifest_path = _resolve_path(args.manifest) if args.manifest else _latest_manifest(out_dir)
    summary = summarize_manifest(manifest_path)

    out_path = _resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Summary written: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Recommendation: {summary['recommendation']} ({summary['actual_passes']}/{summary['required_passes']} scenarios passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
