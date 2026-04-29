#!/usr/bin/env python3
"""Aggregate Hogan walk-forward loss attribution by regime, side, and exit."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bucket() -> dict[str, float | int]:
    return {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl_pct": 0.0,
        "loss_drag_pct": 0.0,
        "worst_loss_pct": 0.0,
    }


def _add_trade(bucket: dict[str, float | int], pnl: float) -> None:
    bucket["trades"] = int(bucket["trades"]) + 1
    bucket["total_pnl_pct"] = float(bucket["total_pnl_pct"]) + pnl
    if pnl > 0:
        bucket["wins"] = int(bucket["wins"]) + 1
    else:
        bucket["losses"] = int(bucket["losses"]) + 1
        bucket["loss_drag_pct"] = float(bucket["loss_drag_pct"]) + pnl
        bucket["worst_loss_pct"] = min(float(bucket["worst_loss_pct"]), pnl)


def _finalize_bucket(bucket: dict[str, float | int]) -> dict[str, float | int]:
    trades = int(bucket["trades"])
    wins = int(bucket["wins"])
    return {
        "trades": trades,
        "wins": wins,
        "losses": int(bucket["losses"]),
        "win_rate": round(wins / trades, 4) if trades else 0.0,
        "total_pnl_pct": round(float(bucket["total_pnl_pct"]), 4),
        "avg_pnl_pct": round(float(bucket["total_pnl_pct"]) / trades, 4) if trades else 0.0,
        "loss_drag_pct": round(float(bucket["loss_drag_pct"]), 4),
        "worst_loss_pct": round(float(bucket["worst_loss_pct"]), 4),
    }


def _closed_trades(payload: dict[str, Any]) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    for window in payload.get("windows", []):
        for trade in window.get("closed_trades", []):
            t = dict(trade)
            t["window_idx"] = window.get("window_idx")
            trades.append(t)
    return trades


def _aggregate_funnel(payloads: list[dict[str, Any]]) -> dict[str, int | float]:
    funnel: dict[str, int | float] = defaultdict(int)
    for payload in payloads:
        for window in payload.get("windows", []):
            for key, value in window.get("signal_funnel", {}).items():
                if isinstance(value, (int, float)):
                    funnel[key] += value
    return dict(sorted(funnel.items()))


def build_report(paths: list[Path]) -> dict[str, Any]:
    payloads = [_load_json(path) for path in paths]
    by_regime: dict[str, dict[str, float | int]] = defaultdict(_bucket)
    by_side: dict[str, dict[str, float | int]] = defaultdict(_bucket)
    by_exit: dict[str, dict[str, float | int]] = defaultdict(_bucket)
    by_regime_side_exit: dict[str, dict[str, float | int]] = defaultdict(_bucket)

    all_trades: list[dict[str, Any]] = []
    for payload in payloads:
        all_trades.extend(_closed_trades(payload))

    for trade in all_trades:
        pnl = float(trade.get("pnl_pct", 0.0) or 0.0)
        regime = str(trade.get("regime") or trade.get("entry_regime") or "unknown")
        side = str(trade.get("side") or "unknown")
        exit_reason = str(trade.get("exit_reason") or trade.get("close_reason") or "unknown")

        _add_trade(by_regime[regime], pnl)
        _add_trade(by_side[side], pnl)
        _add_trade(by_exit[exit_reason], pnl)
        _add_trade(by_regime_side_exit[f"{regime}|{side}|{exit_reason}"], pnl)

    funnel = _aggregate_funnel(payloads)
    block_or_gate_counts = {
        key: value
        for key, value in funnel.items()
        if "block" in key or "gate" in key or "filter" in key
    }

    finalized_combo = {
        key: _finalize_bucket(value) for key, value in by_regime_side_exit.items()
    }
    top_loss_buckets = sorted(
        finalized_combo.items(),
        key=lambda item: item[1]["loss_drag_pct"],
    )[:20]

    total_pnl = sum(float(t.get("pnl_pct", 0.0) or 0.0) for t in all_trades)
    losses = [float(t.get("pnl_pct", 0.0) or 0.0) for t in all_trades if float(t.get("pnl_pct", 0.0) or 0.0) <= 0]

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": [str(path.relative_to(PROJECT_ROOT)) for path in paths],
        "summary": {
            "trades": len(all_trades),
            "losses": len(losses),
            "loss_rate": round(len(losses) / len(all_trades), 4) if all_trades else 0.0,
            "total_pnl_pct": round(total_pnl, 4),
            "loss_drag_pct": round(sum(losses), 4),
            "worst_loss_pct": round(min(losses), 4) if losses else 0.0,
        },
        "by_regime": {key: _finalize_bucket(value) for key, value in by_regime.items()},
        "by_side": {key: _finalize_bucket(value) for key, value in by_side.items()},
        "by_exit_reason": {key: _finalize_bucket(value) for key, value in by_exit.items()},
        "top_loss_buckets": [
            {"bucket": key, **value} for key, value in top_loss_buckets
        ],
        "gate_and_block_counts": block_or_gate_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a focused Hogan loss attribution report")
    parser.add_argument("reports", nargs="+", help="Walk-forward JSON report paths")
    parser.add_argument(
        "--output",
        default="reports/validation/loss_attribution_current.json",
        help="Output JSON path",
    )
    args = parser.parse_args(argv)

    paths = [_resolve_path(path) for path in args.reports]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing report(s): {', '.join(missing)}")

    report = build_report(paths)
    out_path = _resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"Loss attribution written: {out_path.relative_to(PROJECT_ROOT)}")
    print(
        "Trades={trades} loss_rate={loss_rate:.1%} total_pnl={total_pnl_pct:+.2f}% "
        "loss_drag={loss_drag_pct:+.2f}%".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
