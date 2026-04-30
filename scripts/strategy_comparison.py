#!/usr/bin/env python3
"""Run and rank a fixed Hogan strategy comparison matrix.

This is a research helper, not a promotion gate. It keeps the walk-forward
geometry identical across scenarios and emits a ranked manifest so we can
compare candidates without hand-copying metrics from terminal output.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StrategyScenario:
    name: str
    description: str
    flags: tuple[str, ...] = ()


DEFAULT_SCENARIOS: tuple[StrategyScenario, ...] = (
    StrategyScenario(
        name="technical_no_ml",
        description="Technical/agent pipeline without ML filter",
        flags=("--no-ml",),
    ),
    StrategyScenario(
        name="ml_filter",
        description="Default ML probability filter",
    ),
    StrategyScenario(
        name="ml_sizer",
        description="ML probability as continuous position sizer",
        flags=("--ml-sizer",),
    ),
    StrategyScenario(
        name="ml_sizer_macro",
        description="ML sizer plus macro sit-out overlay",
        flags=("--ml-sizer", "--macro-sitout"),
    ),
    StrategyScenario(
        name="regime_models",
        description="Per-regime ML model routing",
        flags=("--regime-models",),
    ),
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def scenario_by_name(names: list[str] | None = None) -> list[StrategyScenario]:
    scenarios = list(DEFAULT_SCENARIOS)
    if not names:
        return scenarios

    lookup = {scenario.name: scenario for scenario in scenarios}
    missing = [name for name in names if name not in lookup]
    if missing:
        valid = ", ".join(sorted(lookup))
        raise ValueError(f"Unknown scenario(s): {', '.join(missing)}. Valid scenarios: {valid}")
    return [lookup[name] for name in names]


def build_walk_forward_command(
    scenario: StrategyScenario,
    *,
    python_exe: str,
    db: str,
    symbol: str,
    timeframe: str,
    n_splits: int,
    min_train: int | None,
    min_test: int | None,
    min_calmar: float,
    output_path: Path,
) -> list[str]:
    cmd = [
        python_exe,
        "-m",
        "hogan_bot.walk_forward",
        "--db",
        db,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--n-splits",
        str(n_splits),
        "--min-calmar",
        str(min_calmar),
        "--output",
        display_path(output_path),
    ]
    if min_train is not None:
        cmd.extend(["--min-train", str(min_train)])
    if min_test is not None:
        cmd.extend(["--min-test", str(min_test)])
    cmd.extend(scenario.flags)
    return cmd


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return int(result.returncode)


def load_summary(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return payload.get("summary", {}) if isinstance(payload, dict) else {}


def summarize_result(
    scenario: StrategyScenario,
    *,
    command: list[str],
    exit_code: int | None,
    report_path: Path,
    log_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    summary = {} if dry_run else load_summary(report_path)
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "flags": list(scenario.flags),
        "cmd": command,
        "exit_code": exit_code,
        "passes_gate": bool(summary.get("passes_gate", False)),
        "n_windows": summary.get("n_windows"),
        "n_positive": summary.get("n_positive"),
        "total_trades": summary.get("total_trades"),
        "mean_return_pct": summary.get("mean_return_pct"),
        "mean_sharpe": summary.get("mean_sharpe"),
        "mean_calmar": summary.get("mean_calmar"),
        "worst_calmar": summary.get("worst_calmar"),
        "worst_drawdown_pct": summary.get("worst_drawdown_pct"),
        "report_json": display_path(report_path),
        "log": display_path(log_path),
    }


def _metric(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank by promotion status first, then risk-adjusted and return metrics."""
    return sorted(
        results,
        key=lambda row: (
            bool(row.get("passes_gate")),
            _metric(row.get("mean_calmar"), -999.0),
            _metric(row.get("mean_return_pct"), -999.0),
            _metric(row.get("mean_sharpe"), -999.0),
            -_metric(row.get("worst_drawdown_pct"), 999.0),
        ),
        reverse=True,
    )


def build_manifest(
    *,
    stamp: str,
    db: str,
    symbol: str,
    timeframe: str,
    n_splits: int,
    min_train: int | None,
    min_test: int | None,
    min_calmar: float,
    results: list[dict[str, Any]],
    dry_run: bool = False,
) -> dict[str, Any]:
    ranked = rank_results(results)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stamp": stamp,
        "dry_run": dry_run,
        "db": db,
        "symbol": symbol,
        "timeframe": timeframe,
        "comparison_geometry": {
            "n_splits": n_splits,
            "min_train": min_train,
            "min_test": min_test,
            "min_calmar": min_calmar,
        },
        "recommendation": "PASS" if any(row.get("passes_gate") for row in ranked) else "HOLD",
        "best_scenario": ranked[0]["scenario"] if ranked else None,
        "ranked_results": ranked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run and rank Hogan strategy comparison scenarios")
    parser.add_argument("--db", default="data/hogan.db", help="SQLite DB path")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--min-train", type=int, default=16000)
    parser.add_argument("--min-test", type=int, default=1000)
    parser.add_argument("--min-calmar", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        default="reports/validation/strategy_comparison",
        help="Directory for scenario reports, logs, and manifest",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.name for scenario in DEFAULT_SCENARIOS],
        help="Scenario to run. May be repeated. Defaults to the full matrix.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write manifest commands without running them")
    args = parser.parse_args(argv)

    scenarios = scenario_by_name(args.scenario)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        report_path = out_dir / f"wf_{scenario.name}_{stamp}.json"
        log_path = out_dir / f"wf_{scenario.name}_{stamp}.log"
        cmd = build_walk_forward_command(
            scenario,
            python_exe=sys.executable,
            db=args.db,
            symbol=args.symbol,
            timeframe=args.timeframe,
            n_splits=args.n_splits,
            min_train=args.min_train,
            min_test=args.min_test,
            min_calmar=args.min_calmar,
            output_path=report_path,
        )
        exit_code = None if args.dry_run else run_command(cmd, log_path)
        results.append(
            summarize_result(
                scenario,
                command=cmd,
                exit_code=exit_code,
                report_path=report_path,
                log_path=log_path,
                dry_run=args.dry_run,
            )
        )

    manifest = build_manifest(
        stamp=stamp,
        db=args.db,
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_splits=args.n_splits,
        min_train=args.min_train,
        min_test=args.min_test,
        min_calmar=args.min_calmar,
        results=results,
        dry_run=args.dry_run,
    )
    manifest_path = out_dir / f"strategy_comparison_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Strategy comparison written: {display_path(manifest_path)}")
    print(f"Recommendation: {manifest['recommendation']}")
    for idx, row in enumerate(manifest["ranked_results"], start=1):
        print(
            "{idx}. {scenario}: return={ret} sharpe={sharpe} calmar={calmar} "
            "worst_dd={dd} pass={passed}".format(
                idx=idx,
                scenario=row["scenario"],
                ret=row.get("mean_return_pct"),
                sharpe=row.get("mean_sharpe"),
                calmar=row.get("mean_calmar"),
                dd=row.get("worst_drawdown_pct"),
                passed=row.get("passes_gate"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
