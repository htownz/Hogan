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
    """Decides whether to exit an open position based on thesis integrity.

    Short positions use tighter drawdown thresholds (unlimited upside risk),
    faster time decay (12h max hold vs 24h for longs), and a volatility
    *contraction* exit (shorts profit from vol expansion; contraction
    weakens the thesis).
    """

    def __init__(
        self,
        trend_reversal_threshold: float = 0.6,
        max_consolidation_bars: int = 12,
        drawdown_panic_pct: float = 0.03,
        volatility_expansion_threshold: float = 2.0,
        time_decay_threshold: float = 0.75,
        # Short-specific overrides — applied when side != "long"
        short_drawdown_panic_pct: float | None = None,
        short_time_decay_threshold: float | None = None,
        short_vol_contraction_threshold: float | None = None,
        short_max_consolidation_bars: int | None = None,
    ):
        self._trend_reversal_threshold = trend_reversal_threshold
        self._max_consolidation_bars = max_consolidation_bars
        self._drawdown_panic_pct = drawdown_panic_pct
        self._vol_expansion_threshold = volatility_expansion_threshold
        self._time_decay_threshold = time_decay_threshold

        # Short defaults: tighter drawdown (2% vs 3%), faster decay (0.60 vs 0.75),
        # vol contraction at 0.5x entry ATR, shorter stagnation window.
        self._short_drawdown_panic_pct = short_drawdown_panic_pct or (drawdown_panic_pct * 0.67)
        self._short_time_decay_threshold = short_time_decay_threshold or 0.60
        self._short_vol_contraction_threshold = short_vol_contraction_threshold or 0.50
        self._short_max_consolidation_bars = short_max_consolidation_bars or max(max_consolidation_bars - 4, 6)

    def should_exit(
        self,
        candles: pd.DataFrame,
        entry_price: float,
        current_price: float,
        bars_held: int,
        side: str = "long",
        max_hold_bars: int = 24,
        entry_atr: float | None = None,
        vol_ratio: float | None = None,
    ) -> ExitDecision:
        """Evaluate whether the current position should be exited.

        Returns an ExitDecision with should_exit, reason, and urgency.
        """
        if len(candles) < 20:
            logger.debug("EXIT_MODEL: insufficient candles (%d < 20), skipping", len(candles))
            return ExitDecision()

        is_short = side != "long"
        close = candles["close"].astype(float)

        if is_short:
            upnl_pct = (entry_price - current_price) / entry_price
        else:
            upnl_pct = (current_price - entry_price) / entry_price

        # Select side-appropriate thresholds
        dd_panic = self._short_drawdown_panic_pct if is_short else self._drawdown_panic_pct
        td_threshold = self._short_time_decay_threshold if is_short else self._time_decay_threshold
        stag_bars = self._short_max_consolidation_bars if is_short else self._max_consolidation_bars

        # 1. Trend persistence check: has the trend actually reversed?
        trend_score = self._trend_persistence(close, side)
        if trend_score < -self._trend_reversal_threshold:
            logger.debug("EXIT_MODEL: trend reversed (score=%.2f, side=%s)", trend_score, side)
            return ExitDecision(
                should_exit=True,
                reason="trend_reversal",
                urgency=min(1.0, abs(trend_score)),
            )

        # 2. Drawdown panic: significant unrealized loss
        if upnl_pct < -dd_panic:
            atr_pct = self._current_atr_pct(candles)
            atr_mult = 1.2 if is_short else 1.5
            if abs(upnl_pct) > atr_pct * atr_mult:
                logger.debug("EXIT_MODEL: drawdown panic (upnl=%.3f, side=%s)", upnl_pct, side)
                return ExitDecision(
                    should_exit=True,
                    reason="drawdown_exceeded",
                    urgency=min(1.0, abs(upnl_pct) / dd_panic),
                )

        # 3. Time decay: position has aged past expected hold window
        hold_ratio = bars_held / max(max_hold_bars, 1)
        if hold_ratio > td_threshold and upnl_pct < 0.005:
            logger.debug("EXIT_MODEL: time decay (held %.0f%%, upnl=%.3f, side=%s)",
                         hold_ratio * 100, upnl_pct, side)
            return ExitDecision(
                should_exit=True,
                reason="time_decay",
                urgency=hold_ratio,
            )

        # 4a. Volatility expansion (longs): regime changed dramatically
        # 4b. Volatility contraction (shorts): vol is dying, thesis weakening
        if entry_atr is not None:
            current_atr = self._current_atr_pct(candles)
            if is_short:
                atr_ratio = current_atr / max(entry_atr, 1e-9)
                if atr_ratio < self._short_vol_contraction_threshold:
                    logger.debug("EXIT_MODEL: vol contraction (ratio=%.2f, side=short)", atr_ratio)
                    return ExitDecision(
                        should_exit=True,
                        reason="volatility_contraction",
                        urgency=min(1.0, (self._short_vol_contraction_threshold - atr_ratio) / 0.3),
                    )
            else:
                atr_expansion = current_atr / max(entry_atr, 1e-9)
                if atr_expansion > self._vol_expansion_threshold:
                    logger.debug("EXIT_MODEL: vol expansion (ratio=%.1f)", atr_expansion)
                    return ExitDecision(
                        should_exit=True,
                        reason="volatility_expansion",
                        urgency=min(1.0, atr_expansion / self._vol_expansion_threshold - 0.5),
                    )

        # 5. Stagnation: position is flat for too long
        if bars_held > stag_bars and abs(upnl_pct) < 0.002:
            logger.debug("EXIT_MODEL: stagnation (bars=%d, upnl=%.4f, side=%s)",
                         bars_held, upnl_pct, side)
            return ExitDecision(
                should_exit=True,
                reason="stagnation",
                urgency=0.3,
            )

        # 6. Volume fade: volume dried up while position stalls in mild profit
        if vol_ratio is not None and vol_ratio < 0.5 and 0.0 <= upnl_pct < 0.005:
            logger.debug("EXIT_MODEL: volume fade (vol_ratio=%.2f, upnl=%.4f)", vol_ratio, upnl_pct)
            return ExitDecision(
                should_exit=True,
                reason="volume_fade",
                urgency=0.25,
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
