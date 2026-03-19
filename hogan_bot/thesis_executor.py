"""1h-thesis / 15m-execution engine.

The 1h timeframe establishes a directional thesis (long/short).
The 15m timeframe provides precise entry timing by waiting for a
pullback into the thesis direction before executing.

Diagnostic evidence shows that MA crossover signals fire after the
move has already happened, causing entries at local highs (avg
range_position 0.59, avg run-up +1.46%).  By requiring a 15m
pullback before entry, we convert lagging 1h signals into
precise 15m dip-buys or bounce-shorts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Thesis:
    """Active directional bias from the 1h timeframe."""
    direction: str             # "long" or "short"
    created_bar_1h: int        # 1h bar index where thesis was established
    confidence: float          # pipeline confidence at thesis creation
    creation_price: float = 0.0  # 1h close when thesis was created
    regime: str | None = None
    max_age_1h_bars: int = 4   # thesis expires after N 1h bars
    executed: bool = False
    expired: bool = False


@dataclass
class ExecutionTrigger:
    """15m entry signal within an active thesis."""
    bar_15m_idx: int
    entry_price: float
    direction: str
    trigger_reason: str  # "rsi_pullback", "bb_lower", "range_low", etc.
    range_position: float
    rsi: float | None = None


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Fast RSI for the last bar of a close array."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_bb_pct_b(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    """Bollinger Band %B for the last bar."""
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    sma = np.mean(window)
    std = np.std(window, ddof=1)
    if std < 1e-12:
        return 0.5
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = upper - lower
    if band_width < 1e-12:
        return 0.5
    return (closes[-1] - lower) / band_width


def _range_position(highs: np.ndarray, lows: np.ndarray, close: float) -> float:
    """Where the close sits in the [low, high] range.  0 = low, 1 = high."""
    local_high = np.max(highs)
    local_low = np.min(lows)
    rng = local_high - local_low
    if rng < 1e-12:
        return 0.5
    return (close - local_low) / rng


def check_15m_entry(
    candles_15m: pd.DataFrame,
    thesis: Thesis,
    *,
    lookback_15m: int = 16,
    min_pullback_pct: float = 1.2,
    long_rsi_ceiling: float = 45.0,
    short_rsi_floor: float = 55.0,
) -> ExecutionTrigger | None:
    """Check if the current 15m bar meets the entry condition for the thesis.

    Requires an actual pullback from the thesis creation price plus
    momentum confirmation via 15m RSI.  No fallback conditions —
    if the pullback doesn't materialise, the thesis expires.

    For a **long thesis**:
    - 15m close has pulled back at least ``min_pullback_pct`` % from
      the thesis creation price (signal at $100 → wait until ≤$99.20).
    - RSI_15m ≤ ``long_rsi_ceiling`` (not overbought on 15m).

    For a **short thesis**:
    - 15m close has bounced at least ``min_pullback_pct`` % above the
      thesis creation price.
    - RSI_15m ≥ ``short_rsi_floor`` (not oversold on 15m).
    """
    if len(candles_15m) < lookback_15m or thesis.creation_price <= 0:
        return None

    closes = candles_15m["close"].values
    highs = candles_15m["high"].values
    lows = candles_15m["low"].values
    current_close = float(closes[-1])

    rsi = _compute_rsi(closes, period=14)
    pct_from_thesis = (current_close - thesis.creation_price) / thesis.creation_price * 100

    recent_highs = highs[-lookback_15m:]
    recent_lows = lows[-lookback_15m:]
    range_pos = _range_position(recent_highs, recent_lows, current_close)

    if thesis.direction == "long":
        has_pullback = pct_from_thesis <= -min_pullback_pct
        momentum_ok = rsi <= long_rsi_ceiling
        if not (has_pullback and momentum_ok):
            return None
        reason = "price_pullback"
    elif thesis.direction == "short":
        has_bounce = pct_from_thesis >= min_pullback_pct
        momentum_ok = rsi >= short_rsi_floor
        if not (has_bounce and momentum_ok):
            return None
        reason = "price_bounce"
    else:
        return None

    return ExecutionTrigger(
        bar_15m_idx=len(candles_15m) - 1,
        entry_price=current_close,
        direction=thesis.direction,
        trigger_reason=reason,
        range_position=round(range_pos, 3),
        rsi=round(rsi, 1),
    )


def align_15m_to_1h(
    candles_1h: pd.DataFrame,
    candles_15m: pd.DataFrame,
) -> dict[int, list[int]]:
    """Build a mapping from 1h bar index to 15m bar indices within that hour.

    Each 1h bar covers a 1-hour window.  The 4 corresponding 15m bars
    fall within the same 1-hour bucket (aligned to the hour boundary).

    Returns ``{1h_bar_idx: [15m_idx_0, 15m_idx_1, ...]}``.
    """
    ts_1h = candles_1h["timestamp"].values.astype("int64") // 10**6  # ms
    ts_15m = candles_15m["timestamp"].values.astype("int64") // 10**6

    mapping: dict[int, list[int]] = {}

    _15m_ptr = 0
    for i, t1h in enumerate(ts_1h):
        hour_start = t1h
        hour_end = t1h + 3_600_000
        indices = []
        j = _15m_ptr
        while j < len(ts_15m) and ts_15m[j] < hour_start:
            j += 1
        while j < len(ts_15m) and ts_15m[j] < hour_end:
            indices.append(j)
            j += 1
        if indices:
            mapping[i] = indices
            _15m_ptr = indices[0]

    return mapping


def find_15m_entry_in_window(
    candles_15m: pd.DataFrame,
    bar_indices: list[int],
    thesis: Thesis,
    lookback_context: int = 16,
) -> ExecutionTrigger | None:
    """Scan a set of 15m bars for the first one meeting entry conditions.

    Uses a rolling window of ``lookback_context`` bars ending at each
    candidate 15m bar to compute RSI, BB %B, and range position.
    """
    for idx in bar_indices:
        start = max(0, idx - lookback_context + 1)
        window = candles_15m.iloc[start:idx + 1]
        trigger = check_15m_entry(window, thesis)
        if trigger is not None:
            trigger.bar_15m_idx = idx
            trigger.entry_price = float(candles_15m["close"].iloc[idx])
            return trigger
    return None
