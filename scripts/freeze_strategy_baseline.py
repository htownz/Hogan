#!/usr/bin/env python3
"""Freeze the current best Hogan strategy report as a baseline marker.

Records the path, metrics, active git commit, and a small slice of regime
position-scale knobs so the strategy search has a stable reference to beat.
Re-run whenever the baseline shifts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_subject(rev: str | None) -> str | None:
    if not rev:
        return None
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s", rev],
            cwd=PROJECT_ROOT, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_summary(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(f"Baseline report not found: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report payload is not an object: {report_path}")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Report missing 'summary': {report_path}")
    return summary


_REGIME_BLOCK_RE = re.compile(
    r"\"(trending_up|trending_down|ranging|volatile)\":\s*RegimeConfig\((?P<body>.*?)\),",
    re.DOTALL,
)
_LONG_SCALE_RE = re.compile(r"long_size_scale\s*=\s*([0-9.]+)")
_SHORT_SCALE_RE = re.compile(r"short_size_scale\s*=\s*([0-9.]+)")


def _capture_regime_scales(config_path: Path) -> dict[str, dict[str, float]]:
    text = config_path.read_text(encoding="utf-8")
    snapshot: dict[str, dict[str, float]] = {}
    for match in _REGIME_BLOCK_RE.finditer(text):
        regime = match.group(1)
        body = match.group("body")
        long_m = _LONG_SCALE_RE.search(body)
        short_m = _SHORT_SCALE_RE.search(body)
        if long_m and short_m:
            snapshot[regime] = {
                "long_size_scale": float(long_m.group(1)),
                "short_size_scale": float(short_m.group(1)),
            }
    return snapshot


def freeze_baseline(
    *,
    report_path: Path,
    output_path: Path,
    note: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    summary = _extract_summary(report_path)
    config_path = config_path or PROJECT_ROOT / "hogan_bot" / "config.py"
    rev = _git_rev()

    record = {
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": note,
        "baseline_report": str(report_path.relative_to(PROJECT_ROOT))
            if report_path.is_relative_to(PROJECT_ROOT) else str(report_path),
        "baseline_report_sha256": _file_sha256(report_path),
        "git_commit": rev,
        "git_subject": _git_subject(rev),
        "summary": summary,
        "regime_position_scales": _capture_regime_scales(config_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze a baseline strategy report")
    parser.add_argument(
        "--report",
        default="reports/validation/wf_volatile_size_020_tail_smoke.json",
        help="Path to the report we treat as current best",
    )
    parser.add_argument(
        "--output",
        default="reports/validation/strategy_search_baseline.json",
        help="Where to write the baseline marker",
    )
    parser.add_argument(
        "--note",
        default="Tail-smoke baseline after volatile/short_size_scale tweaks",
        help="Free-form note describing why this is the current best",
    )
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    record = freeze_baseline(
        report_path=report_path,
        output_path=output_path,
        note=args.note,
    )
    rel = output_path.relative_to(PROJECT_ROOT) if output_path.is_relative_to(PROJECT_ROOT) else output_path
    print(f"Baseline frozen: {rel}")
    summary = record["summary"]
    print(
        f"  return={summary.get('mean_return_pct')} "
        f"sharpe={summary.get('mean_sharpe')} "
        f"calmar={summary.get('mean_calmar')} "
        f"worst_dd={summary.get('worst_drawdown_pct')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
