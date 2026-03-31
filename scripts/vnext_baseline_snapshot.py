#!/usr/bin/env python3
"""Capture a reproducible baseline snapshot for Hogan vNext work.

Writes a JSON artifact under reports/validation/ with:
- git commit + dirty flag
- DB table counts and age ranges
- latest validation manifests/walk-forward report pointers
- key summary metrics from the most recent walk-forward JSON (if available)
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "hogan.db"
REPORT_DIR = PROJECT_ROOT / "reports" / "validation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hogan_bot.observability import get_db_table_stats  # noqa: E402
from hogan_bot.storage import get_connection  # noqa: E402


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception:
        return ""


def _latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def capture_baseline(db_path: Path) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    conn = get_connection(str(db_path))
    try:
        table_stats = [asdict(t) for t in get_db_table_stats(conn)]
    finally:
        conn.close()

    manifest_paths = list((PROJECT_ROOT / "reports" / "validation").glob("manifest_*.json"))
    wf_paths = list((PROJECT_ROOT / "diagnostics").glob("walk_forward*.json"))
    latest_manifest = _latest(manifest_paths)
    latest_wf = _latest(wf_paths)
    latest_manifest_json = _load_json(latest_manifest)
    latest_wf_json = _load_json(latest_wf)

    baseline = {
        "captured_at_utc": stamp,
        "git": {
            "branch": branch,
            "commit": commit,
            "dirty": dirty,
        },
        "db_path": str(db_path.relative_to(PROJECT_ROOT)) if db_path.exists() else str(db_path),
        "table_stats": table_stats,
        "latest_artifacts": {
            "manifest_path": str(latest_manifest.relative_to(PROJECT_ROOT)) if latest_manifest else None,
            "walk_forward_path": str(latest_wf.relative_to(PROJECT_ROOT)) if latest_wf else None,
        },
        "latest_manifest_summary": {
            "stamp_utc": latest_manifest_json.get("stamp_utc"),
            "steps": latest_manifest_json.get("steps", []),
        },
        "latest_walk_forward_summary": latest_wf_json.get("summary", {}),
        "latest_walk_forward_exit_lifecycle": latest_wf_json.get("exit_lifecycle", {}),
    }

    out_path = REPORT_DIR / f"vnext_baseline_{stamp}.json"
    out_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    out = capture_baseline(DEFAULT_DB)
    print(f"Baseline snapshot written: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
