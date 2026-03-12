"""ExitEvaluator — answers "is the thesis broken?" not "what would I do fresh?"

The key insight: the same pipeline that says "buy" at bar N can say "sell" at
bar N+1 with equal authority. That causes flippy signal-exit damage. A dedicated
exit model uses different criteria:

- **Trend persistence**: Has the trend reversed, or just pulled back?
- **Unrealized P/L trajectory**: Are we in drawdown or consolidating?
- **Time decay**: Has the position aged past its expected hold window?
- **Volatility expansion**: Has the regime changed since entry?

Usage::

    evaluator = ExitEvaluator()
    should_exit, reason = evaluator.should_exit(
        candles=candles,
        entry_price=100.0,
        current_price=98.5,
        bars_held=5,
        side="long",
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExitDecision:
    should_exit: bool = False
    reason: str = ""
    urgency: float = 0.0  # 0.0 = no urgency, 1.0 = immediate exit


class ExitEvaluator:
    """Decides whether to exit an open position based on thesis integrity."""

    def __init__(
        self,
        trend_reversal_threshold: float = 0.6,
        max_consolidation_bars: int = 12,
        drawdown_panic_pct: float = 0.03,
        volatility_expansion_threshold: float = 2.0,
    ):
        self._trend_reversal_threshold = trend_reversal_threshold
        self._max_consolidation_bars = max_consolidation_bars
        self._drawdown_panic_pct = drawdown_panic_pct
        self._vol_expansion_threshold = volatility_expansion_threshold

    def should_exit(
        self,
        candles: pd.DataFrame,
        entry_price: float,
        current_price: float,
        bars_held: int,
        side: str = "long",
        max_hold_bars: int = 24,
        entry_atr: float | None = None,
    ) -> ExitDecision:
        """Evaluate whether the current position should be exited.

        Returns an ExitDecision with should_exit, reason, and urgency.
        """
        if len(candles) < 20:
            logger.debug("EXIT_MODEL: insufficient candles (%d < 20), skipping", len(candles))
            return ExitDecision()

        close = candles["close"].astype(float)

        # Unrealized P/L
        if side == "long":
            upnl_pct = (current_price - entry_price) / entry_price
        else:
            upnl_pct = (entry_price - current_price) / entry_price

        # 1. Trend persistence check: has the trend actually reversed?
        trend_score = self._trend_persistence(close, side)
        if trend_score < -self._trend_reversal_threshold:
            logger.debug("EXIT_MODEL: trend reversed (score=%.2f)", trend_score)
            return ExitDecision(
                should_exit=True,
                reason="trend_reversal",
                urgency=min(1.0, abs(trend_score)),
            )

        # 2. Drawdown panic: significant unrealized loss
        if upnl_pct < -self._drawdown_panic_pct:
            atr_pct = self._current_atr_pct(candles)
            if abs(upnl_pct) > atr_pct * 1.5:
                logger.debug("EXIT_MODEL: drawdown panic (upnl=%.3f)", upnl_pct)
                return ExitDecision(
                    should_exit=True,
                    reason="drawdown_exceeded",
                    urgency=min(1.0, abs(upnl_pct) / self._drawdown_panic_pct),
                )

        # 3. Time decay: position has aged past expected hold window
        hold_ratio = bars_held / max(max_hold_bars, 1)
        if hold_ratio > 0.75 and upnl_pct < 0.005:
            logger.debug("EXIT_MODEL: time decay (held %.0f%%, upnl=%.3f)", hold_ratio * 100, upnl_pct)
            return ExitDecision(
                should_exit=True,
                reason="time_decay",
                urgency=hold_ratio,
            )

        # 4. Volatility expansion: regime changed dramatically
        if entry_atr is not None:
            current_atr = self._current_atr_pct(candles)
            vol_ratio = current_atr / max(entry_atr, 1e-9)
            if vol_ratio > self._vol_expansion_threshold:
                logger.debug("EXIT_MODEL: vol expansion (ratio=%.1f)", vol_ratio)
                return ExitDecision(
                    should_exit=True,
                    reason="volatility_expansion",
                    urgency=min(1.0, vol_ratio / self._vol_expansion_threshold - 0.5),
                )

        # 5. Stagnation: position is flat for too long
        if bars_held > self._max_consolidation_bars and abs(upnl_pct) < 0.002:
            logger.debug("EXIT_MODEL: stagnation (bars=%d, upnl=%.4f)", bars_held, upnl_pct)
            return ExitDecision(
                should_exit=True,
                reason="stagnation",
                urgency=0.3,
            )

        return ExitDecision()

    def _trend_persistence(self, close: pd.Series, side: str) -> float:
        """Score in [-1, 1]: positive = trend intact, negative = reversed.

        Uses a combination of short/long MA spread (normalized by ATR-like
        range to avoid unit mismatch) and recent momentum direction.
        """
        if len(close) < 20:
            return 0.0

        ma_fast = close.rolling(5).mean().iloc[-1]
        ma_slow = close.rolling(20).mean().iloc[-1]

        if pd.isna(ma_fast) or pd.isna(ma_slow) or ma_slow < 1e-9:
            return 0.0

        # Normalize spread as a z-score relative to rolling std
        rolling_std = close.rolling(20).std().iloc[-1]
        if pd.isna(rolling_std) or rolling_std < 1e-9:
            rolling_std = abs(ma_slow) * 0.01

        ma_spread_z = (ma_fast - ma_slow) / rolling_std

        recent_returns = close.pct_change().iloc[-5:].dropna()
        if recent_returns.empty:
            momentum_z = 0.0
        else:
            ret_mean = float(recent_returns.mean())
            ret_std = float(recent_returns.std())
            momentum_z = ret_mean / max(ret_std, 1e-9) if ret_std > 1e-9 else 0.0

        # Both components are now z-score-like, combine with equal weight
        raw = ma_spread_z * 0.6 + momentum_z * 0.4

        if side != "long":
            raw = -raw

        return float(max(-1.0, min(1.0, raw)))

    def _current_atr_pct(self, candles: pd.DataFrame) -> float:
        """Current ATR as a percentage of price."""
        if len(candles) < 15:
            return 0.01
        high = candles["high"].astype(float)
        low = candles["low"].astype(float)
        close = candles["close"].astype(float)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        if pd.isna(atr) or close.iloc[-1] < 1e-9:
            return 0.01
        return float(atr / close.iloc[-1])
