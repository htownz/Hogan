"""Volatility regime agent — votes based on volatility structure.

Uses ATR trends, Bollinger bandwidth, and vol-of-vol to classify the
current volatility regime and translate that into directional bias.

Key insight: volatility contraction precedes expansion (breakouts),
while volatility expansion at extremes precedes mean-reversion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hogan_bot.swarm_decision.agents._utils import get_baseline_action
from hogan_bot.swarm_decision.types import AgentVote


class VolatilityRegimeAgent:
    """Directional vote informed by volatility structure."""

    agent_id: str = "volatility_regime_v1"

    def __init__(
        self,
        atr_window: int = 14,
        bb_window: int = 20,
        bb_num_std: float = 2.0,
        squeeze_bw_pct: float = 0.03,
        expansion_bw_pct: float = 0.08,
    ) -> None:
        self._atr_window = atr_window
        self._bb_window = bb_window
        self._bb_std = bb_num_std
        self._squeeze_bw = squeeze_bw_pct
        self._expansion_bw = expansion_bw_pct

    def vote(
        self,
        *,
        symbol: str,
        candles: pd.DataFrame,
        as_of_ms: int | None,
        shared_context: dict,
    ) -> AgentVote:
        min_bars = max(self._atr_window, self._bb_window) + 10
        if len(candles) < min_bars:
            return self._neutral("insufficient_bars")

        close = candles["close"].astype(float)
        high = candles["high"].astype(float)
        low = candles["low"].astype(float)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self._atr_window).mean()

        sma = close.rolling(self._bb_window).mean()
        std = close.rolling(self._bb_window).std()
        bandwidth = (2 * self._bb_std * std) / sma.where(sma > 0, 1e-9)

        bw_now = float(bandwidth.iloc[-1]) if not np.isnan(bandwidth.iloc[-1]) else 0.0
        atr_now = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
        atr_prev = float(atr.iloc[-13]) if len(atr) > 12 and not np.isnan(atr.iloc[-13]) else atr_now
        px = float(close.iloc[-1])

        atr_pct = atr_now / max(px, 1e-9)
        atr_trend = (atr_now - atr_prev) / max(atr_prev, 1e-9)

        reasons: list[str] = []
        confidence = 0.50
        size_scale = 1.0

        is_squeeze = bw_now < self._squeeze_bw
        is_expansion = bw_now > self._expansion_bw
        atr_rising = atr_trend > 0.15
        atr_falling = atr_trend < -0.15

        if is_squeeze and atr_falling:
            action = get_baseline_action(shared_context)
            confidence = 0.65
            size_scale = 0.70
            reasons.append(f"vol_squeeze_bw={bw_now:.3f}")
        elif is_expansion and atr_rising:
            baseline = get_baseline_action(shared_context)
            price_above_sma = px > float(sma.iloc[-1]) if not np.isnan(sma.iloc[-1]) else True
            if baseline == "hold":
                action = "buy" if price_above_sma else "sell"
            else:
                action = baseline
            confidence = 0.60
            size_scale = 0.80
            reasons.append(f"vol_expansion_bw={bw_now:.3f}")
        elif is_expansion and atr_falling:
            action = get_baseline_action(shared_context)
            confidence = 0.55
            size_scale = 0.60
            reasons.append("vol_exhaustion")
        else:
            action = get_baseline_action(shared_context)
            confidence = 0.50
            size_scale = 1.0

        return AgentVote(
            agent_id=self.agent_id,
            action=action,
            confidence=max(0.0, min(1.0, confidence)),
            size_scale=max(0.0, min(1.0, size_scale)),
            veto=False,
            block_reasons=reasons,
        )

    def _neutral(self, reason: str) -> AgentVote:
        return AgentVote(
            agent_id=self.agent_id,
            action="hold",
            confidence=0.0,
            size_scale=1.0,
            veto=False,
            block_reasons=[reason],
        )
