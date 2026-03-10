"""Timeframe utilities for Hogan: parse, convert, and infer bar intervals.

Used by backtest (annualization), ML/ICT (previous-day features), retrain (horizon),
and config (hour-to-bar conversion).
"""

from __future__ import annotations

import pandas as pd

_TF_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "45m": 45,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
}


def parse_timeframe_to_minutes(timeframe: str) -> int:
    """Return the number of minutes per bar for the given timeframe string.

    Examples: '5m' -> 5, '1h' -> 60, '1d' -> 1440.
    Falls back to 5 if unrecognised.
    """
    return _TF_MINUTES.get(timeframe, 5)


def bars_per_day(timeframe: str) -> int:
    """Return the number of bars per UTC calendar day for the given timeframe."""
    minutes = parse_timeframe_to_minutes(timeframe)
    if minutes <= 0:
        return 288  # fallback to 5m-equivalent
    return 24 * 60 // minutes


def bars_per_year(timeframe: str, days: int = 365) -> float:
    """Return the number of bars per year for annualization (Sharpe, Sortino)."""
    return bars_per_day(timeframe) * days


def hours_to_bars(hours: float, timeframe: str) -> int:
    """Convert real-world hours to bar count for the given timeframe."""
    minutes = parse_timeframe_to_minutes(timeframe)
    if minutes <= 0:
        return max(1, int(hours * 12))  # 5m fallback: 12 bars/hour
    return max(1, int(hours * 60 / minutes))


def infer_timeframe_from_candles(candles: pd.DataFrame) -> str | None:
    """Infer timeframe from candle timestamps by sampling bar intervals.

    Expects a 'timestamp' or 'ts_ms' column. Returns a standard string like
    '5m', '1h', or None if inference fails.
    """
    if candles.empty or len(candles) < 2:
        return None

    ts_col = "ts_ms" if "ts_ms" in candles.columns else "timestamp"
    if ts_col not in candles.columns:
        return None

    ts = candles[ts_col].iloc[-5:]  # sample last few bars
    if ts_col == "ts_ms":
        diffs_ms = ts.diff().dropna()
        if diffs_ms.empty:
            return None
        median_diff_ms = diffs_ms.median()
        if pd.isna(median_diff_ms) or median_diff_ms <= 0:
            return None
        minutes = int(round(median_diff_ms / 60_000))
    else:
        ts_parsed = pd.to_datetime(ts, utc=True)
        diffs = ts_parsed.diff().dropna()
        if diffs.empty:
            return None
        median_diff = diffs.median()
        if pd.isna(median_diff):
            return None
        minutes = int(round(median_diff.total_seconds() / 60))

    if minutes <= 0:
        return None

    # Map minutes to standard timeframe string
    rev_map = {v: k for k, v in _TF_MINUTES.items()}
    return rev_map.get(minutes, f"{minutes}m")


def default_horizon_bars(timeframe: str, target_hours: float = 6.0) -> int:
    """Compute horizon_bars that targets *target_hours* of real time.

    Different timeframes need different bar counts for the same holding period.
    E.g. 6 hours = 72 bars at 5m, 12 bars at 30m, 6 bars at 1h.
    """
    return max(1, hours_to_bars(target_hours, timeframe))
