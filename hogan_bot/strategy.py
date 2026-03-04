from __future__ import annotations

from dataclasses import dataclass

from hogan_bot.indicators import (
    cloud_signal,
    compute_atr,
    detect_fvgs,
    fvg_entry_signal,
    ripster_ema_clouds,
)


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
    *,
    use_ema_clouds: bool = False,
    ema_fast_short: int = 8,
    ema_fast_long: int = 9,
    ema_slow_short: int = 34,
    ema_slow_long: int = 50,
    use_fvg: bool = False,
    fvg_min_gap_pct: float = 0.001,
    signal_mode: str = "any",
) -> StrategySignal:
    """MA crossover with volume confirmation, optionally combined with EMA clouds and FVGs.

    All parameters after the bare ``*`` are keyword-only and default to the
    behaviour of the original MA-only strategy, so existing callers require
    no changes.

    signal_mode
        ``"ma_only"`` — original MA crossover behaviour; new indicators ignored.
        ``"any"``     — buy/sell fires when *any* enabled signal agrees.
        ``"all"``     — buy/sell fires only when *all* enabled signals agree.
    """
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

    confidence = min(1.0, max(0.0, (volume_ratio - 1.0) / 1.5)) if volume_confirmed else 0.0

    # ATR-based stop distance (14-bar Wilder ATR replaces the single-bar range proxy)
    atr_series = compute_atr(candles, window=14)
    atr_stop = float(atr_series.iloc[-1] / max(close.iloc[-1], 1e-9)) * 1.5
    stop_distance_pct = max(0.004, min(0.03, atr_stop))

    if bullish_cross and volume_confirmed:
        ma_action = "buy"
    elif bearish_cross and volume_confirmed:
        ma_action = "sell"
    else:
        ma_action = "hold"

    if signal_mode == "ma_only" or (not use_ema_clouds and not use_fvg):
        return StrategySignal(ma_action, stop_distance_pct, confidence, volume_ratio)

    votes: list[str] = [ma_action]

    if use_ema_clouds:
        enriched = ripster_ema_clouds(
            candles,
            fast_short=ema_fast_short,
            fast_long=ema_fast_long,
            slow_short=ema_slow_short,
            slow_long=ema_slow_long,
        )
        latest_cloud = cloud_signal(enriched).iloc[-1]
        if latest_cloud == "bullish":
            votes.append("buy")
        elif latest_cloud == "bearish":
            votes.append("sell")
        else:
            votes.append("hold")

    if use_fvg:
        fvgs = detect_fvgs(candles, min_gap_pct=fvg_min_gap_pct)
        last_close = float(close.iloc[-1])
        votes.append(fvg_entry_signal(fvgs, last_close))

    if signal_mode == "all":
        if all(v == "buy" for v in votes):
            action = "buy"
        elif all(v == "sell" for v in votes):
            action = "sell"
        else:
            action = "hold"
    else:  # "any"
        if "buy" in votes:
            action = "buy"
        elif "sell" in votes:
            action = "sell"
        else:
            action = "hold"

    return StrategySignal(action, stop_distance_pct, confidence, volume_ratio)
