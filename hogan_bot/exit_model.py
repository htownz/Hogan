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
from dataclasses import dataclass

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
        # Trend-persistence MA periods — aligned with strategy's Ripster
        # cloud periods by default.  The previous 5/20 was too fast and
        # caused premature exits on normal pullbacks.
        trend_fast_ma: int = 8,
        trend_slow_ma: int = 34,
        stagnation_threshold: float = 0.003,
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
        self._trend_fast_ma = trend_fast_ma
        self._trend_slow_ma = trend_slow_ma
        self._stagnation_threshold = stagnation_threshold

        # Short defaults: tighter drawdown (2% vs 3%), faster decay (0.60 vs 0.75),
        # vol contraction at 0.5x entry ATR, shorter stagnation window.
        self._short_drawdown_panic_pct = short_drawdown_panic_pct or (drawdown_panic_pct * 0.67)
        self._short_time_decay_threshold = short_time_decay_threshold or 0.60
        self._short_vol_contraction_threshold = short_vol_contraction_threshold or 0.50
        self._short_max_consolidation_bars = short_max_consolidation_bars or max(max_consolidation_bars - 4, 6)

    # Per-regime exit parameter overrides.  Each key maps to a dict of
    # parameter → value.  Absent keys fall through to the base (side-aware)
    # defaults.  This replaces the old multiplier-tuple approach with
    # explicit values that are easier to reason about and sweep.
    _REGIME_EXIT_PARAMS: dict[str, dict[str, float]] = {
        "volatile": {
            "drawdown_panic_pct": 0.042,       # wider: tolerate vol noise
            "time_decay_threshold": 0.65,      # faster exit: vol can reverse
            "stagnation_bars_mult": 0.75,      # shorter patience in chop
            "trend_reversal_threshold": 0.48,  # easier reversal trigger
        },
        "trending_up": {
            "drawdown_panic_pct": 0.036,       # slightly wider than base
            "time_decay_threshold": 0.75,      # standard
            "stagnation_bars_mult": 1.00,
            "trend_reversal_threshold": 0.60,  # strict: stay in trend
        },
        "trending_down": {
            "drawdown_panic_pct": 0.030,       # tighter: protect capital
            "time_decay_threshold": 0.68,      # faster
            "stagnation_bars_mult": 0.85,
            "trend_reversal_threshold": 0.54,  # easier trigger
        },
        "ranging": {
            "drawdown_panic_pct": 0.038,       # wider: mean-reversion needs room for pullbacks
            "time_decay_threshold": 0.80,      # patient: let mean-reversion thesis fully develop
            "stagnation_bars_mult": 1.20,      # extended stagnation window for ranging trades
            "trend_reversal_threshold": 0.70,  # harder to call reversal in noise
        },
    }

    # Legacy compat alias (external code may reference this)
    _REGIME_EXIT_ADJUSTMENTS = _REGIME_EXIT_PARAMS

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
        regime: str | None = None,
        max_favorable_pct: float = 0.0,
    ) -> ExitDecision:
        """Evaluate whether the current position should be exited.

        Returns an ExitDecision with should_exit, reason, and urgency.

        Graduated urgency: as hold_ratio increases, thresholds loosen so
        that the ExitEvaluator becomes progressively easier to trigger.
        This converts max_hold_time from a cliff into a smooth ramp.
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
        trend_rev_thresh = self._trend_reversal_threshold

        # Apply regime-specific parameter overrides
        if regime and regime in self._REGIME_EXIT_PARAMS:
            _rp = self._REGIME_EXIT_PARAMS[regime]
            if "drawdown_panic_pct" in _rp:
                dd_panic = _rp["drawdown_panic_pct"]
                if is_short:
                    dd_panic *= 0.67
            if "time_decay_threshold" in _rp:
                td_threshold = _rp["time_decay_threshold"]
                if is_short:
                    td_threshold = min(td_threshold, self._short_time_decay_threshold)
            if "stagnation_bars_mult" in _rp:
                stag_bars = max(4, int(stag_bars * _rp["stagnation_bars_mult"]))
            if "trend_reversal_threshold" in _rp:
                trend_rev_thresh = _rp["trend_reversal_threshold"]

        # ── Graduated urgency: scale thresholds by hold_ratio ─────────
        # After 50% of max hold, progressively loosen all triggers.
        # At 100% hold, thresholds are halved (2x easier to exit).
        hold_ratio = bars_held / max(max_hold_bars, 1)
        _urgency_ramp = max(0.0, (hold_ratio - 0.5) * 2.0)  # 0 at 50%, 1 at 100%
        _ease = 1.0 - _urgency_ramp * 0.5  # 1.0 at 50%, 0.5 at 100%

        dd_panic *= _ease
        trend_rev_thresh *= _ease
        stag_bars = max(3, int(stag_bars * _ease))

        # 1. Trend persistence check: has the trend actually reversed?
        trend_score = self._trend_persistence(close, side)
        if trend_score < -trend_rev_thresh:
            logger.debug("EXIT_MODEL: trend reversed (score=%.2f, thresh=%.2f, side=%s, hold_ratio=%.2f)",
                         trend_score, trend_rev_thresh, side, hold_ratio)
            return ExitDecision(
                should_exit=True,
                reason="trend_reversal",
                urgency=min(1.0, abs(trend_score) + _urgency_ramp * 0.3),
            )

        # 2. Drawdown panic: significant unrealized loss
        _mfe_relief = min(0.50, max_favorable_pct * 5.0) if max_favorable_pct > 0 else 0.0
        _eff_dd_panic = dd_panic * (1.0 + _mfe_relief)
        if upnl_pct < -_eff_dd_panic:
            atr_pct = self._current_atr_pct(candles)
            atr_mult = (1.2 if is_short else 1.5) * _ease
            if abs(upnl_pct) > atr_pct * atr_mult:
                logger.debug(
                    "EXIT_MODEL: drawdown panic (upnl=%.3f, mfe=%.3f, eff_dd=%.3f, side=%s, hold_ratio=%.2f)",
                    upnl_pct, max_favorable_pct, _eff_dd_panic, side, hold_ratio,
                )
                return ExitDecision(
                    should_exit=True,
                    reason="drawdown_exceeded",
                    urgency=min(1.0, abs(upnl_pct) / max(_eff_dd_panic, 1e-9)),
                )

        # 3. Time decay: position has aged past expected hold window
        # With graduated urgency, this activates earlier for losing trades
        _eff_td = td_threshold * _ease
        if hold_ratio > _eff_td and upnl_pct < 0.0:
            logger.debug("EXIT_MODEL: time decay (held %.0f%%, upnl=%.3f, side=%s, eff_td=%.2f)",
                         hold_ratio * 100, upnl_pct, side, _eff_td)
            return ExitDecision(
                should_exit=True,
                reason="time_decay",
                urgency=min(1.0, hold_ratio + _urgency_ramp * 0.2),
            )

        # 3b. Position decay: after 75% hold, exit marginal-profit trades
        # Fee round-trip ~0.2% for BTC, so 0.8% threshold gives ~0.6% net.
        _decay_thresh = 0.008 - _urgency_ramp * 0.003  # tightens from 0.8% to 0.5% at max hold
        if hold_ratio > 0.75 and 0.0 <= upnl_pct < _decay_thresh:
            logger.debug("EXIT_MODEL: position decay (held %.0f%%, upnl=%.3f < %.3f — marginal profit, closing)",
                         hold_ratio * 100, upnl_pct, _decay_thresh)
            return ExitDecision(
                should_exit=True,
                reason="position_decay",
                urgency=hold_ratio,
            )

        # 3c. Give-back protection: if trade had a good run (MFE > 1.5%)
        # but has given back more than 60% of peak profit, exit to lock in gains.
        if max_favorable_pct > 0.015 and upnl_pct > 0:
            _giveback_ratio = 1.0 - (upnl_pct / max_favorable_pct)
            _giveback_thresh = max(0.40, 0.60 - _urgency_ramp * 0.20)  # 60% give-back at base, 40% at max urgency
            if _giveback_ratio > _giveback_thresh:
                logger.debug(
                    "EXIT_MODEL: give-back (mfe=%.3f, upnl=%.3f, given_back=%.0f%%, thresh=%.0f%%)",
                    max_favorable_pct, upnl_pct, _giveback_ratio * 100, _giveback_thresh * 100,
                )
                return ExitDecision(
                    should_exit=True,
                    reason="give_back",
                    urgency=min(1.0, _giveback_ratio),
                )

        # 4a. Volatility expansion (longs): regime changed dramatically
        # 4b. Volatility contraction (shorts): vol is dying, thesis weakening
        if entry_atr is not None:
            current_atr = self._current_atr_pct(candles)
            if is_short:
                _vol_thresh = self._short_vol_contraction_threshold * (1.0 + _urgency_ramp * 0.3)
                atr_ratio = current_atr / max(entry_atr, 1e-9)
                if atr_ratio < _vol_thresh:
                    logger.debug("EXIT_MODEL: vol contraction (ratio=%.2f, thresh=%.2f, side=short)", atr_ratio, _vol_thresh)
                    return ExitDecision(
                        should_exit=True,
                        reason="volatility_contraction",
                        urgency=min(1.0, (_vol_thresh - atr_ratio) / 0.3),
                    )
            else:
                _vol_exp_thresh = self._vol_expansion_threshold * _ease
                atr_expansion = current_atr / max(entry_atr, 1e-9)
                if atr_expansion > _vol_exp_thresh:
                    logger.debug("EXIT_MODEL: vol expansion (ratio=%.1f, thresh=%.1f)", atr_expansion, _vol_exp_thresh)
                    return ExitDecision(
                        should_exit=True,
                        reason="volatility_expansion",
                        urgency=min(1.0, atr_expansion / max(_vol_exp_thresh, 0.1) - 0.5),
                    )

        # 5. Stagnation: position is flat for too long
        if bars_held > stag_bars and abs(upnl_pct) < self._stagnation_threshold:
            logger.debug("EXIT_MODEL: stagnation (bars=%d, stag_thresh=%d, upnl=%.4f, side=%s)",
                         bars_held, stag_bars, upnl_pct, side)
            return ExitDecision(
                should_exit=True,
                reason="stagnation",
                urgency=min(1.0, 0.3 + _urgency_ramp * 0.4),
            )

        # 6. Volume fade: volume dried up while position stalls in mild profit
        _vol_fade_thresh = 0.5 + _urgency_ramp * 0.3
        if vol_ratio is not None and vol_ratio < _vol_fade_thresh and -0.002 <= upnl_pct < 0.001:
            logger.debug("EXIT_MODEL: volume fade (vol_ratio=%.2f, thresh=%.2f, upnl=%.4f)",
                         vol_ratio, _vol_fade_thresh, upnl_pct)
            return ExitDecision(
                should_exit=True,
                reason="volume_fade",
                urgency=0.25 + _urgency_ramp * 0.3,
            )

        # 7. Momentum exhaustion: price extending but momentum fading
        if bars_held >= 4 and upnl_pct > 0.003:
            _exh = self._momentum_exhaustion(close, side)
            _exh_thresh = max(0.25, 0.5 - _urgency_ramp * 0.25)
            if _exh > _exh_thresh:
                logger.debug(
                    "EXIT_MODEL: momentum exhaustion (score=%.2f, thresh=%.2f, upnl=%.3f, side=%s)",
                    _exh, _exh_thresh, upnl_pct, side,
                )
                return ExitDecision(
                    should_exit=True,
                    reason="momentum_exhaustion",
                    urgency=min(1.0, _exh),
                )

        return ExitDecision()

    def _trend_persistence(self, close: pd.Series, side: str) -> float:
        """Score in [-1, 1]: positive = trend intact, negative = reversed.

        Uses a combination of short/long MA spread (normalized by ATR-like
        range to avoid unit mismatch) and recent momentum direction.

        MA periods are configurable via ``trend_fast_ma`` / ``trend_slow_ma``
        to align with the strategy's actual indicator periods.
        """
        _min_bars = max(self._trend_slow_ma, 20)
        if len(close) < _min_bars:
            return 0.0

        ma_fast = close.rolling(self._trend_fast_ma).mean().iloc[-1]
        ma_slow = close.rolling(self._trend_slow_ma).mean().iloc[-1]

        if pd.isna(ma_fast) or pd.isna(ma_slow) or ma_slow < 1e-9:
            return 0.0

        rolling_std = close.rolling(self._trend_slow_ma).std().iloc[-1]
        if pd.isna(rolling_std) or rolling_std < 1e-9:
            rolling_std = abs(ma_slow) * 0.01

        ma_spread_z = (ma_fast - ma_slow) / rolling_std

        _mom_window = max(self._trend_fast_ma, 5)
        recent_returns = close.pct_change().iloc[-_mom_window:].dropna()
        if recent_returns.empty:
            momentum_z = 0.0
        else:
            ret_mean = float(recent_returns.mean())
            ret_std = float(recent_returns.std())
            momentum_z = ret_mean / max(ret_std, 0.001) if ret_std > 1e-9 else 0.0

        raw = ma_spread_z * 0.6 + momentum_z * 0.4

        if side != "long":
            raw = -raw

        return float(max(-1.0, min(1.0, raw)))

    def _momentum_exhaustion(self, close: pd.Series, side: str) -> float:
        """Detect price-momentum divergence (exhaustion).

        For longs: price making higher highs but RSI declining = bearish divergence.
        For shorts: price making lower lows but RSI rising = bullish divergence.

        Returns a score in [0, 1] where > 0.5 signals exhaustion.
        """
        _window = 14
        if len(close) < _window + 5:
            return 0.0

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(_window).mean()
        loss = (-delta.clip(upper=0)).rolling(_window).mean()
        rs = gain / loss.clip(lower=1e-9)
        rsi = 100 - (100 / (1 + rs))

        rsi_vals = rsi.iloc[-8:].dropna()
        close_vals = close.iloc[-8:]
        if len(rsi_vals) < 6 or len(close_vals) < 6:
            return 0.0

        half = len(close_vals) // 2
        price_first = float(close_vals.iloc[:half].mean())
        price_second = float(close_vals.iloc[half:].mean())
        rsi_first = float(rsi_vals.iloc[:half].mean())
        rsi_second = float(rsi_vals.iloc[half:].mean())

        if side == "long":
            price_rising = price_second > price_first
            rsi_falling = rsi_second < rsi_first - 2.0
            if price_rising and rsi_falling and rsi_second > 60:
                divergence = (rsi_first - rsi_second) / max(rsi_first, 1.0)
                return float(min(1.0, divergence * 3.0))
        else:
            price_falling = price_second < price_first
            rsi_rising = rsi_second > rsi_first + 2.0
            if price_falling and rsi_rising and rsi_second < 40:
                divergence = (rsi_second - rsi_first) / max(100 - rsi_first, 1.0)
                return float(min(1.0, divergence * 3.0))

        return 0.0

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
