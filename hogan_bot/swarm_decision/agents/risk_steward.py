"""Risk steward agent — veto/scale based on drawdown and volatility.

Prevents the system from taking large positions during drawdowns or
extreme volatility, acting as a conservative safety net.
"""
from __future__ import annotations

import pandas as pd

from hogan_bot.swarm_decision.types import AgentVote


class RiskStewardAgent:
    """Outputs size_scale + optional veto based on portfolio risk state."""

    agent_id: str = "risk_steward_v1"

    def __init__(
        self,
        max_drawdown_pct: float = 0.10,
        vol_scale_threshold: float = 2.5,
        vol_veto_threshold: float = 4.0,
    ) -> None:
        self._max_dd = max_drawdown_pct
        self._vol_scale = vol_scale_threshold
        self._vol_veto = vol_veto_threshold

    def vote(
        self,
        *,
        symbol: str,
        candles: pd.DataFrame,
        as_of_ms: int | None,
        shared_context: dict,
    ) -> AgentVote:
        reasons: list[str] = []
        size_scale = 1.0
        veto = False

        equity = shared_context.get("equity_usd", 0.0)
        peak_equity = shared_context.get("peak_equity_usd", equity)
        if peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity
            if dd >= self._max_dd:
                veto = True
                size_scale = 0.0
                reasons.append(f"drawdown_{dd:.1%}")
            elif dd >= self._max_dd * 0.5:
                size_scale *= 0.5
                reasons.append(f"drawdown_warning_{dd:.1%}")

        atr_pct = shared_context.get("atr_pct", 0.0)
        hist_vol = shared_context.get("hist_vol_20", 0.0)
        if hist_vol > 0 and atr_pct > 0:
            vol_ratio = atr_pct / hist_vol
            if vol_ratio >= self._vol_veto:
                veto = True
                size_scale = 0.0
                reasons.append(f"vol_spike_{vol_ratio:.1f}x")
            elif vol_ratio >= self._vol_scale:
                size_scale *= max(0.3, 1.0 - (vol_ratio - self._vol_scale) * 0.2)
                reasons.append(f"high_vol_{vol_ratio:.1f}x")

        return AgentVote(
            agent_id=self.agent_id,
            action="hold",
            confidence=0.5,
            size_scale=max(0.0, min(1.0, size_scale)),
            veto=veto,
            block_reasons=reasons,
        )
