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
    max_drawdown: float = 15.0,
    max_loss_streak: int = 5,
    min_market_type_coverage: int = 1,
    min_quality_weighted_pnl: float = 0.0,
) -> PolymarketPromotionResult:
    """Evaluate whether shadow results justify live-readiness work."""
    reasons: list[str] = []
    trades = float(metrics.get("trades", 0.0))
    total_pnl = float(metrics.get("total_pnl", 0.0))
    avg_pnl = float(metrics.get("avg_pnl", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    drawdown = float(metrics.get("max_drawdown", 0.0))
    loss_streak = float(metrics.get("worst_loss_streak", 0.0))
    coverage = float(metrics.get("market_type_coverage", 0.0))
    quality_pnl = float(metrics.get("data_quality_weighted_pnl", 0.0))

    if trades < min_trades:
        reasons.append(f"insufficient_shadow_trades:{trades:.0f}<{min_trades}")
    if total_pnl <= min_total_pnl:
        reasons.append(f"total_pnl_below_gate:{total_pnl:.2f}<={min_total_pnl:.2f}")
    if avg_pnl < min_avg_pnl:
        reasons.append(f"avg_pnl_below_gate:{avg_pnl:.2f}<{min_avg_pnl:.2f}")
    if win_rate < min_win_rate:
        reasons.append(f"win_rate_below_gate:{win_rate:.2f}<{min_win_rate:.2f}")
    if "max_drawdown" in metrics and drawdown > max_drawdown:
        reasons.append(f"drawdown_above_gate:{drawdown:.2f}>{max_drawdown:.2f}")
    if "worst_loss_streak" in metrics and loss_streak > max_loss_streak:
        reasons.append(f"loss_streak_above_gate:{loss_streak:.0f}>{max_loss_streak}")
    if "market_type_coverage" in metrics and coverage < min_market_type_coverage:
        reasons.append(f"market_type_coverage_below_gate:{coverage:.0f}<{min_market_type_coverage}")
    if "data_quality_weighted_pnl" in metrics and quality_pnl <= min_quality_weighted_pnl:
        reasons.append(f"quality_weighted_pnl_below_gate:{quality_pnl:.2f}<={min_quality_weighted_pnl:.2f}")

    return PolymarketPromotionResult(
        approved=not reasons,
        reasons=reasons,
        metrics={
            "trades": trades,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "win_rate": win_rate,
            "max_drawdown": drawdown,
            "worst_loss_streak": loss_streak,
            "market_type_coverage": coverage,
            "data_quality_weighted_pnl": quality_pnl,
        },
    )


def evaluate_shadow_ledger(conn, **kwargs) -> PolymarketPromotionResult:
    """Evaluate promotion readiness from the local shadow ledger."""
    from hogan_bot.storage import summarize_polymarket_shadow_trades

    return evaluate_polymarket_promotion(
        summarize_polymarket_shadow_trades(conn),
        **kwargs,
    )
