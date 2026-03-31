#!/usr/bin/env python3
"""Build a canary/shadow promotion decision package for Hogan vNext."""
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


def _latest_manifest(path: Path) -> Path | None:
    files = list(path.glob("manifest_*.json"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _load(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _wf_summary(report_path: Path) -> dict:
    data = _load(report_path)
    return data.get("summary", {}) if isinstance(data, dict) else {}


def build_package(
    db_path: str,
    manifest_path: Path | None = None,
    cert_override: bool | None = None,
) -> tuple[dict, Path]:
    validation_dir = PROJECT_ROOT / "reports" / "validation"
    selected_manifest = (manifest_path.resolve() if manifest_path is not None else _latest_manifest(validation_dir))
    manifest = _load(selected_manifest)
    wf_matrix = manifest.get("wf_matrix", {})
    wf_results = wf_matrix.get("results", [])
    cert_steps = [s for s in manifest.get("steps", []) if s.get("name") == "swarm_certification"]
    cert_ok = all(int(s.get("exit_code", 1)) == 0 for s in cert_steps) if cert_steps else True
    if cert_override is not None:
        cert_ok = bool(cert_override)

    scenario_summaries: list[dict] = []
    scenario_gate_passes = 0
    for r in wf_results:
        report_rel = r.get("report_json")
        if not report_rel:
            continue
        rp = PROJECT_ROOT / report_rel
        sm = _wf_summary(rp)
        _gate_pass = bool(sm.get("passes_gate", False))
        if _gate_pass:
            scenario_gate_passes += 1
        scenario_summaries.append({
            "scenario": r.get("scenario"),
            "exit_code": r.get("exit_code"),
            "passes_gate": _gate_pass,
            "report_json": report_rel,
            "summary": sm,
        })

    conn = get_connection(db_path)
    try:
        obs = observability_health_report(conn, since_hours=24.0)
    finally:
        conn.close()

    required = int(wf_matrix.get("required_passes", 1))
    actual = int(scenario_gate_passes)
    matrix_ok = actual >= required

    critical_codes = {
        "storage_integrity",
        "execution_failure_spike",
        "swarm_shadow_active_drift",
    }
    critical_alerts = [a for a in obs.get("alerts", []) if a.get("code") in critical_codes]
    observability_ok = len(critical_alerts) == 0

    recommendation = "PROMOTE" if (matrix_ok and observability_ok and cert_ok) else "HOLD"
    rationale = []
    rationale.append(
        f"WF matrix passes {actual}/{required} ({'OK' if matrix_ok else 'FAIL'})"
    )
    rationale.append(
        f"Observability critical alerts: {len(critical_alerts)} ({'OK' if observability_ok else 'FAIL'})"
    )
    rationale.append(
        f"Swarm certification: {'PASS' if cert_ok else 'FAIL'}"
    )

    pkg = {
        "generated_at_utc": _ts(),
        "manifest_path": str(selected_manifest.relative_to(PROJECT_ROOT))
        if selected_manifest is not None else None,
        "wf_matrix": {
            "required_passes": required,
            "actual_passes": actual,
            "results": scenario_summaries,
        },
        "swarm_certification": {
            "present": bool(cert_steps),
            "passed": cert_ok,
            "steps": cert_steps,
        },
        "observability": {
            "alerts": obs.get("alerts", []),
            "critical_alerts": critical_alerts,
            "integrity": obs.get("integrity", {}),
            "swarm_drift": obs.get("swarm_drift", {}),
            "execution": obs.get("execution", {}),
        },
        "recommendation": recommendation,
        "rationale": rationale,
    }

    out = validation_dir / f"vnext_promotion_package_{_ts()}.json"
    out.write_text(json.dumps(pkg, indent=2, default=str), encoding="utf-8")
    return pkg, out


def main() -> int:
    p = argparse.ArgumentParser(description="Build Hogan vNext promotion package")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--manifest", default=None, help="Optional manifest path")
    p.add_argument(
        "--cert-passed",
        action="store_true",
        help="Override certification status to PASS (use when certification was run separately).",
    )
    args = p.parse_args()

    manifest = Path(args.manifest) if args.manifest else None
    cert_override = True if args.cert_passed else None
    pkg, out = build_package(args.db, manifest, cert_override=cert_override)
    print(f"Wrote: {out.relative_to(PROJECT_ROOT)}")
    print(f"Recommendation: {pkg['recommendation']}")
    for r in pkg["rationale"]:
        print(f"  - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
