#!/usr/bin/env python3
"""Summarize a Hogan strategy search across smoke + full + attribution outputs.

Reads the frozen baseline, the smoke and full strategy comparison manifests, and
optionally a leading-candidate loss attribution and emits a single consolidated
JSON. Intended for handoff so we can see at a glance whether tail loss actually
improved or whether further iteration is needed.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _summary_keys(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        return {}
    return {
        "n_windows": summary.get("n_windows"),
        "n_positive": summary.get("n_positive"),
        "total_trades": summary.get("total_trades"),
        "mean_return_pct": summary.get("mean_return_pct"),
        "mean_sharpe": summary.get("mean_sharpe"),
        "mean_calmar": summary.get("mean_calmar"),
        "worst_calmar": summary.get("worst_calmar"),
        "worst_drawdown_pct": summary.get("worst_drawdown_pct"),
        "passes_gate": summary.get("passes_gate"),
    }


def _short_results(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in manifest.get("ranked_results", []):
        rows.append({
            "scenario": entry.get("scenario"),
            "kind": entry.get("kind"),
            "passes_gate": entry.get("passes_gate"),
            "n_windows": entry.get("n_windows"),
            "n_positive": entry.get("n_positive"),
            "total_trades": entry.get("total_trades"),
            "mean_return_pct": entry.get("mean_return_pct"),
            "mean_sharpe": entry.get("mean_sharpe"),
            "mean_calmar": entry.get("mean_calmar"),
            "worst_drawdown_pct": entry.get("worst_drawdown_pct"),
        })
    return rows


def build_summary(
    *,
    baseline_path: Path,
    smoke_manifest_path: Path | None,
    full_manifest_path: Path | None,
    candidate_attribution_path: Path | None,
    baseline_attribution_path: Path | None,
) -> dict[str, Any]:
    baseline = _load(baseline_path)

    smoke_manifest = _load(smoke_manifest_path) if smoke_manifest_path else None
    full_manifest = _load(full_manifest_path) if full_manifest_path else None
    candidate_attr = _load(candidate_attribution_path) if candidate_attribution_path else None
    baseline_attr = _load(baseline_attribution_path) if baseline_attribution_path else None

    tail_loss_comparison: dict[str, Any] | None = None
    if candidate_attr and baseline_attr:
        cand_summary = candidate_attr.get("summary", {})
        base_summary = baseline_attr.get("summary", {})

        def _delta(key: str) -> float | None:
            try:
                return round(float(cand_summary[key]) - float(base_summary[key]), 4)
            except (KeyError, TypeError, ValueError):
                return None

        tail_loss_comparison = {
            "candidate_summary": cand_summary,
            "baseline_summary": base_summary,
            "deltas": {
                "trades": _delta("trades"),
                "loss_rate": _delta("loss_rate"),
                "total_pnl_pct": _delta("total_pnl_pct"),
                "loss_drag_pct": _delta("loss_drag_pct"),
                "worst_loss_pct": _delta("worst_loss_pct"),
            },
        }

    verdict_reasons: list[str] = []
    if tail_loss_comparison:
        deltas = tail_loss_comparison["deltas"]
        loss_drag_delta = deltas.get("loss_drag_pct")
        worst_loss_delta = deltas.get("worst_loss_pct")
        if loss_drag_delta is not None:
            if loss_drag_delta > 0:
                verdict_reasons.append(
                    f"tail loss drag improved (less negative by {loss_drag_delta:+.2f}pp)"
                )
            elif loss_drag_delta < 0:
                verdict_reasons.append(
                    f"tail loss drag worsened (more negative by {loss_drag_delta:+.2f}pp)"
                )
            else:
                verdict_reasons.append("tail loss drag unchanged")
        if worst_loss_delta is not None:
            if worst_loss_delta > 0:
                verdict_reasons.append(
                    f"worst tail loss improved (less negative by {worst_loss_delta:+.2f}pp)"
                )
            elif worst_loss_delta < 0:
                verdict_reasons.append(
                    f"worst tail loss worsened (more negative by {worst_loss_delta:+.2f}pp)"
                )
            else:
                verdict_reasons.append("worst tail loss unchanged")

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": {
            "path": _display(baseline_path),
            "frozen_at_utc": baseline.get("frozen_at_utc"),
            "summary": baseline.get("summary"),
            "regime_position_scales": baseline.get("regime_position_scales"),
            "git_commit": baseline.get("git_commit"),
        },
        "tail_smoke_matrix": (
            {
                "path": _display(smoke_manifest_path) if smoke_manifest_path else None,
                "geometry": smoke_manifest.get("comparison_geometry") if smoke_manifest else None,
                "best_scenario": smoke_manifest.get("best_scenario") if smoke_manifest else None,
                "ranked_results": _short_results(smoke_manifest) if smoke_manifest else [],
            }
            if smoke_manifest
            else None
        ),
        "full_validation": (
            {
                "path": _display(full_manifest_path) if full_manifest_path else None,
                "geometry": full_manifest.get("comparison_geometry") if full_manifest else None,
                "best_scenario": full_manifest.get("best_scenario") if full_manifest else None,
                "ranked_results": _short_results(full_manifest) if full_manifest else [],
            }
            if full_manifest
            else None
        ),
        "leading_candidate_attribution": (
            {
                "path": _display(candidate_attribution_path) if candidate_attribution_path else None,
                "summary": candidate_attr.get("summary") if candidate_attr else None,
            }
            if candidate_attr
            else None
        ),
        "baseline_attribution": (
            {
                "path": _display(baseline_attribution_path) if baseline_attribution_path else None,
                "summary": baseline_attr.get("summary") if baseline_attr else None,
            }
            if baseline_attr
            else None
        ),
        "tail_loss_comparison": tail_loss_comparison,
        "verdict_reasons": verdict_reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a Hogan strategy search")
    parser.add_argument(
        "--baseline",
        default="reports/validation/strategy_search_baseline.json",
    )
    parser.add_argument(
        "--smoke-manifest",
        default="reports/validation/strategy_search_tail_smoke/strategy_comparison_20260430T173215Z.json",
    )
    parser.add_argument(
        "--full-manifest",
        default="reports/validation/strategy_search_full/strategy_comparison_20260430T175413Z.json",
    )
    parser.add_argument(
        "--candidate-attribution",
        default="reports/validation/strategy_search_tail_smoke/loss_attribution_regime_models_tail.json",
    )
    parser.add_argument(
        "--baseline-attribution",
        default="reports/validation/loss_attribution_volatile_size_020_tail_smoke.json",
    )
    parser.add_argument(
        "--output",
        default="reports/validation/strategy_search_summary.json",
    )
    args = parser.parse_args(argv)

    summary = build_summary(
        baseline_path=_resolve(args.baseline),
        smoke_manifest_path=_resolve(args.smoke_manifest) if args.smoke_manifest else None,
        full_manifest_path=_resolve(args.full_manifest) if args.full_manifest else None,
        candidate_attribution_path=_resolve(args.candidate_attribution)
            if args.candidate_attribution else None,
        baseline_attribution_path=_resolve(args.baseline_attribution)
            if args.baseline_attribution else None,
    )

    out = _resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Strategy search summary: {_display(out)}")

    if summary["full_validation"]:
        ranked = summary["full_validation"]["ranked_results"]
        for row in ranked:
            print(
                "  {kind:<8} {scenario:<32} ret={ret} sharpe={sh} calmar={cal} dd={dd}".format(
                    kind=row.get("kind") or "",
                    scenario=row.get("scenario") or "",
                    ret=row.get("mean_return_pct"),
                    sh=row.get("mean_sharpe"),
                    cal=row.get("mean_calmar"),
                    dd=row.get("worst_drawdown_pct"),
                )
            )

    if summary["verdict_reasons"]:
        print("Verdict:")
        for reason in summary["verdict_reasons"]:
            print(f"  - {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
