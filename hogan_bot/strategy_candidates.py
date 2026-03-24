"""Entry families for the strategy matrix tournament.

Five entry families that produce StrategySignal with neutral sizing metadata
(confidence=1.0, volume_ratio=1.0). Stops/TP/trailing/max-hold come from
the exit pack, not from the entry; stop_distance_pct here is ATR-based
as a fallback default that the matrix runner overrides.

A. DonchianBreakout     -- symmetric 20-bar channel breakout
B. RSIPullbackReclaim   -- RSI reclaim with EMA trend filter
C. EMAPullbackTrend     -- pullback to EMA(21) inside aligned triple-EMA trend
D. BollingerSqueezeBreakout -- BB squeeze detection + directional breakout
E. StrippedBaseline     -- existing Hogan StrategyRouter, no overlays (CONTROL)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hogan_bot.indicators import compute_atr
from hogan_bot.strategy import StrategySignal

_HOLD = StrategySignal(action="hold", stop_distance_pct=0.01, confidence=1.0, volume_ratio=1.0)
_MIN_BARS = 55


def _atr_stop_pct(atr_val: float, price: float, mult: float = 2.0) -> float:
    if price <= 0:
        return 0.02
    raw = (atr_val * mult) / price
    return float(np.clip(raw, 0.004, 0.10))


# ── A. Donchian Breakout ────────────────────────────────────────────────────

class DonchianBreakout:
    """Symmetric 20-bar Donchian channel breakout.

    Long:  close > highest high of prior 20 bars
    Short: close < lowest low of prior 20 bars
    """
    name: str = "donchian_breakout"

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def generate_signal(self, candles, config=None, regime_state=None) -> StrategySignal:
        if len(candles) < self.lookback + 2:
            return _HOLD

        close = candles["close"].astype(float)
        high = candles["high"].astype(float)
        low = candles["low"].astype(float)

        px = float(close.iloc[-1])
        prior_high = high.iloc[-(self.lookback + 1):-1]
        prior_low = low.iloc[-(self.lookback + 1):-1]

        hhv = float(prior_high.max())
        llv = float(prior_low.min())

        atr_series = compute_atr(candles, window=14)
        atr_val = float(atr_series.iloc[-1])
        stop = _atr_stop_pct(atr_val, px)

        if px > hhv:
            return StrategySignal(action="buy", stop_distance_pct=stop, confidence=1.0, volume_ratio=1.0)
        if px < llv:
            return StrategySignal(action="sell", stop_distance_pct=stop, confidence=1.0, volume_ratio=1.0)
        return _HOLD


# ── B. RSI Pullback + Trend Filter (Reclaim) ────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


class RSIPullbackReclaim:
    """RSI reclaim trigger with dual-EMA trend filter.

    Long trend:  EMA(50) > EMA(200) and close > EMA(50)
    Long entry:  prior bar RSI(14) < 30 and current bar RSI(14) >= 35
    Short trend: EMA(50) < EMA(200) and close < EMA(50)
    Short entry: prior bar RSI(14) > 70 and current bar RSI(14) <= 65
    """
    name: str = "rsi_pullback_reclaim"

    def __init__(
        self,
        rsi_period: int = 14,
        ema_fast: int = 50,
        ema_slow: int = 200,
        long_dip: float = 30.0,
        long_reclaim: float = 35.0,
        short_rip: float = 70.0,
        short_reclaim: float = 65.0,
    ):
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.long_dip = long_dip
        self.long_reclaim = long_reclaim
        self.short_rip = short_rip
        self.short_reclaim = short_reclaim

    def generate_signal(self, candles, config=None, regime_state=None) -> StrategySignal:
        needed = max(self.ema_slow + 5, self.rsi_period + 5)
        if len(candles) < needed:
            return _HOLD

        close = candles["close"].astype(float)
        px = float(close.iloc[-1])

        ema_f = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.ema_slow, adjust=False).mean()

        rsi = _rsi(close, self.rsi_period)
        rsi_now = float(rsi.iloc[-1])
        rsi_prev = float(rsi.iloc[-2])

        atr_val = float(compute_atr(candles, window=14).iloc[-1])
        stop = _atr_stop_pct(atr_val, px)

        ema_f_now = float(ema_f.iloc[-1])
        ema_s_now = float(ema_s.iloc[-1])

        if ema_f_now > ema_s_now and px > ema_f_now:
            if rsi_prev < self.long_dip and rsi_now >= self.long_reclaim:
                return StrategySignal(action="buy", stop_distance_pct=stop, confidence=1.0, volume_ratio=1.0)

        if ema_f_now < ema_s_now and px < ema_f_now:
            if rsi_prev > self.short_rip and rsi_now <= self.short_reclaim:
                return StrategySignal(action="sell", stop_distance_pct=stop, confidence=1.0, volume_ratio=1.0)

        return _HOLD


# ── C. EMA Pullback Trend ───────────────────────────────────────────────────

class EMAPullbackTrend:
    """Triple-EMA pullback: trade pullbacks to EMA(21) inside an aligned trend.

    Long:  EMA(8) > EMA(21) > EMA(55), low touched/crossed below EMA(21),
           close finishes above EMA(21).
    Short: EMA(8) < EMA(21) < EMA(55), high touched/crossed above EMA(21),
           close finishes below EMA(21).
    """
    name: str = "ema_pullback_trend"

    def __init__(self, fast: int = 8, medium: int = 21, slow: int = 55):
        self.fast = fast
        self.medium = medium
        self.slow = slow

    def generate_signal(self, candles, config=None, regime_state=None) -> StrategySignal:
        if len(candles) < self.slow + 5:
            return _HOLD

        close = candles["close"].astype(float)
        high = candles["high"].astype(float)
        low = candles["low"].astype(float)

        ema_f = close.ewm(span=self.fast, adjust=False).mean()
        ema_m = close.ewm(span=self.medium, adjust=False).mean()
        ema_sl = close.ewm(span=self.slow, adjust=False).mean()

        ef = float(ema_f.iloc[-1])
        em = float(ema_m.iloc[-1])
        es = float(ema_sl.iloc[-1])
        px = float(close.iloc[-1])
        lo = float(low.iloc[-1])
        hi = float(high.iloc[-1])

        atr_val = float(compute_atr(candles, window=14).iloc[-1])
        stop = _atr_stop_pct(atr_val, px)

        if ef > em > es:
            if lo <= em and px > em:
                return StrategySignal(action="buy", stop_distance_pct=stop, confidence=1.0, volume_ratio=1.0)

        if ef < em < es:
            if hi >= em and px < em:
                return StrategySignal(action="sell", stop_distance_pct=stop, confidence=1.0, volume_ratio=1.0)

        return _HOLD


# ── D. Bollinger Squeeze Breakout ───────────────────────────────────────────

class BollingerSqueezeBreakout:
    """BB squeeze detection + directional breakout on release.

    Squeeze:  BB(20, 2.0) width < Keltner(20, 1.5) width
    Long:     squeeze active on prior bar, current close > upper BB
    Short:    squeeze active on prior bar, current close < lower BB
    """
    name: str = "bollinger_squeeze_breakout"

    def __init__(self, period: int = 20, bb_std: float = 2.0, kc_mult: float = 1.5):
        self.period = period
        self.bb_std = bb_std
        self.kc_mult = kc_mult

    def generate_signal(self, candles, config=None, regime_state=None) -> StrategySignal:
        if len(candles) < self.period + 5:
            return _HOLD

        close = candles["close"].astype(float)
        px = float(close.iloc[-1])

        sma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std(ddof=0)
        upper_bb = sma + self.bb_std * std
        lower_bb = sma - self.bb_std * std
        bb_width = upper_bb - lower_bb

        atr_series = compute_atr(candles, window=self.period)
        kc_width = 2.0 * self.kc_mult * atr_series

        squeeze = bb_width < kc_width
        squeeze_prev = bool(squeeze.iloc[-2])

        atr_val = float(atr_series.iloc[-1])
        stop = _atr_stop_pct(atr_val, px)

        if not squeeze_prev:
            return _HOLD

        ub = float(upper_bb.iloc[-1])
        lb = float(lower_bb.iloc[-1])

        if px > ub:
            return StrategySignal(action="buy", stop_distance_pct=stop, confidence=1.0, volume_ratio=1.0)
        if px < lb:
            return StrategySignal(action="sell", stop_distance_pct=stop, confidence=1.0, volume_ratio=1.0)
        return _HOLD


# ── E. Stripped Current Baseline (CONTROL) ──────────────────────────────────

class StrippedBaseline:
    """Existing Hogan StrategyRouter with all overlays removed.

    This is a CONTROL candidate. If it wins, a follow-up ablation is required
    before promotion — the right response is to identify *which piece* inside
    the baseline carries the edge.
    """
    name: str = "stripped_baseline"

    def __init__(self):
        from hogan_bot.strategy_router import StrategyRouter
        self._router = StrategyRouter()
        self._default_config = None

    def _get_config(self, config):
        if config is not None:
            return config
        if self._default_config is None:
            from hogan_bot.config import load_config
            self._default_config = load_config()
        return self._default_config

    def generate_signal(self, candles, config=None, regime_state=None) -> StrategySignal:
        cfg = self._get_config(config)
        sig = self._router.route(candles, cfg, regime_state)
        return StrategySignal(
            action=sig.action,
            stop_distance_pct=sig.stop_distance_pct,
            confidence=1.0,
            volume_ratio=1.0,
        )


# ── Registry ────────────────────────────────────────────────────────────────

ENTRY_FAMILIES: dict[str, type] = {
    "A_donchian": DonchianBreakout,
    "B_rsi_reclaim": RSIPullbackReclaim,
    "C_ema_pullback": EMAPullbackTrend,
    "D_bb_squeeze": BollingerSqueezeBreakout,
    "E_baseline": StrippedBaseline,
}


def get_entry_family(key: str):
    """Instantiate an entry family by its registry key."""
    cls = ENTRY_FAMILIES[key]
    return cls()
