"""Exit lifecycle analytics — tail losses, time-in-trade by regime, exit quality.

Pure functions that operate on closed-trade dicts (from backtest or
walk-forward windows).  No DB access.  The caller provides the data.

These metrics answer "where is the strategy bleeding?" and "are exits
leaving money on the table or holding too long?"
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def compute_tail_loss_metrics(
    closed_trades: list[dict],
    *,
    percentiles: tuple[float, ...] = (5, 10, 25),
) -> dict[str, Any]:
    """Characterize tail losses across a set of closed trades.

    Returns worst single loss, percentile breakpoints, fraction of
    trades exceeding various loss thresholds, and total tail-loss drag.
    """
    if not closed_trades:
        return {"n_trades": 0}

    pnls = [t.get("pnl_pct", 0.0) for t in closed_trades]
    losses = [p for p in pnls if p < 0]

    result: dict[str, Any] = {
        "n_trades": len(pnls),
        "n_losses": len(losses),
        "loss_rate": round(len(losses) / len(pnls), 4) if pnls else 0.0,
        "worst_loss_pct": round(min(pnls), 4) if pnls else 0.0,
        "mean_loss_pct": round(float(np.mean(losses)), 4) if losses else 0.0,
        "total_loss_drag_pct": round(sum(losses), 4),
    }

    if losses:
        for p in percentiles:
            result[f"p{int(p)}_loss_pct"] = round(
                float(np.percentile(losses, p)), 4
            )

    thresholds = [1.0, 2.0, 3.0, 5.0]
    for t in thresholds:
        count = sum(1 for loss in losses if loss < -t)
        result[f"losses_beyond_{t}pct"] = count

    return result


def compute_time_in_trade_by_regime(
    closed_trades: list[dict],
) -> dict[str, dict[str, Any]]:
    """Average bars held and PnL breakdown by entry regime.

    Highlights regimes where trades are held too long (high bars_held,
    negative PnL) or too short (low bars_held, missed upside).
    """
    if not closed_trades:
        return {}

    by_regime: dict[str, list[dict]] = defaultdict(list)
    for t in closed_trades:
        regime = t.get("entry_regime") or "unknown"
        by_regime[regime].append(t)

    result: dict[str, dict[str, Any]] = {}
    for regime, trades in sorted(by_regime.items()):
        bars = [t.get("bars_held", 0) for t in trades]
        pnls = [t.get("pnl_pct", 0.0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        n = len(trades)

        result[regime] = {
            "n_trades": n,
            "avg_bars_held": round(float(np.mean(bars)), 1) if bars else 0,
            "median_bars_held": round(float(np.median(bars)), 1) if bars else 0,
            "max_bars_held": int(max(bars)) if bars else 0,
            "avg_pnl_pct": round(float(np.mean(pnls)), 4) if pnls else 0.0,
            "total_pnl_pct": round(sum(pnls), 4),
            "win_rate": round(len(wins) / n, 4) if n else 0.0,
            "avg_win_pct": round(float(np.mean(wins)), 4) if wins else 0.0,
            "avg_loss_pct": round(float(np.mean(losses)), 4) if losses else 0.0,
            "worst_loss_pct": round(min(pnls), 4) if pnls else 0.0,
        }

    return result


def compute_exit_quality_by_regime(
    closed_trades: list[dict],
) -> dict[str, dict[str, Any]]:
    """Exit reason breakdown by regime — where are specific exit types hurting?

    Groups by (regime, exit_reason) and shows count, total PnL, avg PnL.
    Highlights which regime + exit combos are the biggest bleeders.
    """
    if not closed_trades:
        return {}

    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for t in closed_trades:
        regime = t.get("entry_regime") or "unknown"
        reason = t.get("close_reason") or t.get("exit_reason") or "unknown"
        pnl = t.get("pnl_pct", 0.0)
        buckets[regime][reason].append(pnl)

    result: dict[str, dict[str, Any]] = {}
    for regime in sorted(buckets):
        reasons = {}
        for reason, pnls in sorted(
            buckets[regime].items(), key=lambda x: sum(x[1])
        ):
            reasons[reason] = {
                "count": len(pnls),
                "total_pnl_pct": round(sum(pnls), 4),
                "avg_pnl_pct": round(float(np.mean(pnls)), 4),
            }
        result[regime] = reasons

    return result


def compute_hold_duration_vs_pnl(
    closed_trades: list[dict],
    *,
    duration_buckets: tuple[int, ...] = (3, 6, 12, 18, 24),
) -> dict[str, dict[str, Any]]:
    """Bucket trades by hold duration and show PnL stats per bucket.

    Answers: "are long holds profitable or should we exit earlier?"
    """
    if not closed_trades:
        return {}

    by_bucket: dict[str, list[float]] = defaultdict(list)
    for t in closed_trades:
        bars = t.get("bars_held", 0)
        pnl = t.get("pnl_pct", 0.0)
        assigned = False
        for b in duration_buckets:
            if bars <= b:
                by_bucket[f"0-{b}h"].append(pnl)
                assigned = True
                break
        if not assigned:
            by_bucket[f">{duration_buckets[-1]}h"].append(pnl)

    result: dict[str, dict[str, Any]] = {}
    for bucket in sorted(by_bucket, key=lambda x: int(x.split("-")[-1].rstrip("h")) if "-" in x else 999):
        pnls = by_bucket[bucket]
        n = len(pnls)
        wins = [p for p in pnls if p > 0]
        result[bucket] = {
            "n_trades": n,
            "avg_pnl_pct": round(float(np.mean(pnls)), 4) if pnls else 0.0,
            "total_pnl_pct": round(sum(pnls), 4),
            "win_rate": round(len(wins) / n, 4) if n else 0.0,
            "worst_pct": round(min(pnls), 4) if pnls else 0.0,
        }

    return result


def summarize_exit_lifecycle(
    closed_trades: list[dict],
) -> dict[str, Any]:
    """One-call summary combining all exit lifecycle analytics."""
    return {
        "tail_losses": compute_tail_loss_metrics(closed_trades),
        "time_in_trade_by_regime": compute_time_in_trade_by_regime(closed_trades),
        "exit_quality_by_regime": compute_exit_quality_by_regime(closed_trades),
        "hold_duration_vs_pnl": compute_hold_duration_vs_pnl(closed_trades),
    }
