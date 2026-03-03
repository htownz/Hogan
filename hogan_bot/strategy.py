from __future__ import annotations

from dataclasses import dataclass



@dataclass
class StrategySignal:
    action: str
    stop_distance_pct: float
    confidence: float
    volume_ratio: float


def generate_signal(
    candles,
    short_window: int,
    long_window: int,
    volume_window: int,
    volume_threshold: float,
) -> StrategySignal:
    """MA crossover with volume confirmation and ATR-like stop proxy."""
    min_len = max(long_window, volume_window) + 2
    if len(candles) < min_len:
        return StrategySignal("hold", 0.01, 0.0, 0.0)

    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    volume = candles["volume"].astype(float)

    short_ma = close.rolling(short_window).mean()
    long_ma = close.rolling(long_window).mean()
    avg_volume = volume.rolling(volume_window).mean()

    bullish_cross = short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]
    bearish_cross = short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]

    volume_ratio = float(volume.iloc[-1] / max(avg_volume.iloc[-1], 1e-9))
    volume_confirmed = volume_ratio >= volume_threshold

    # crude volatility proxy based on recent candle range
    candle_range_pct = float(((high.iloc[-1] - low.iloc[-1]) / max(close.iloc[-1], 1e-9)))
    stop_distance_pct = max(0.004, min(0.03, candle_range_pct * 1.5))

    confidence = min(1.0, max(0.0, (volume_ratio - 1.0) / 1.5)) if volume_confirmed else 0.0

    if bullish_cross and volume_confirmed:
        return StrategySignal("buy", stop_distance_pct, confidence, volume_ratio)
    if bearish_cross and volume_confirmed:
        return StrategySignal("sell", stop_distance_pct, confidence, volume_ratio)
    return StrategySignal("hold", stop_distance_pct, 0.0, volume_ratio)
