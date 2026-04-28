"""Promotion gate for Polymarket shadow-trading evidence.

This gate is intentionally conservative. It evaluates hypothetical shadow
trades only and does not enable live Polymarket order placement.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolymarketPromotionResult:
    approved: bool
    reasons: list[str]
    metrics: dict[str, float]


def evaluate_polymarket_promotion(
    metrics: dict[str, float],
    *,
    min_trades: int = 50,
    min_total_pnl: float = 0.0,
    min_avg_pnl: float = 0.10,
    min_win_rate: float = 0.55,
) -> PolymarketPromotionResult:
    """Evaluate whether shadow results justify live-readiness work."""
    reasons: list[str] = []
    trades = float(metrics.get("trades", 0.0))
    total_pnl = float(metrics.get("total_pnl", 0.0))
    avg_pnl = float(metrics.get("avg_pnl", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0))

    if trades < min_trades:
        reasons.append(f"insufficient_shadow_trades:{trades:.0f}<{min_trades}")
    if total_pnl <= min_total_pnl:
        reasons.append(f"total_pnl_below_gate:{total_pnl:.2f}<={min_total_pnl:.2f}")
    if avg_pnl < min_avg_pnl:
        reasons.append(f"avg_pnl_below_gate:{avg_pnl:.2f}<{min_avg_pnl:.2f}")
    if win_rate < min_win_rate:
        reasons.append(f"win_rate_below_gate:{win_rate:.2f}<{min_win_rate:.2f}")

    return PolymarketPromotionResult(
        approved=not reasons,
        reasons=reasons,
        metrics={
            "trades": trades,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "win_rate": win_rate,
        },
    )


def evaluate_shadow_ledger(conn, **kwargs) -> PolymarketPromotionResult:
    """Evaluate promotion readiness from the local shadow ledger."""
    from hogan_bot.storage import summarize_polymarket_shadow_trades

    return evaluate_polymarket_promotion(
        summarize_polymarket_shadow_trades(conn),
        **kwargs,
    )
