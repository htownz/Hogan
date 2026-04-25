"""Timeframe utilities for Hogan: parse, convert, and infer bar intervals.

Used by backtest (annualization), ML/ICT (previous-day features), retrain (horizon),
and config (hour-to-bar conversion).
"""

from __future__ import annotations

import math

import pandas as pd

_TF_SECONDS: dict[str, int] = {
    "1s": 1,
    "5s": 5,
    "10s": 10,
    "15s": 15,
    "30s": 30,
    "1m": 1,
    "5m": 5 * 60,
    "10m": 10 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "45m": 45 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}

_TF_MINUTES: dict[str, int] = {
    tf: max(1, seconds // 60) for tf, seconds in _TF_SECONDS.items()
    if seconds >= 60
}


def parse_timeframe_to_seconds(timeframe: str) -> int:
    """Return seconds per bar for strings like ``10s``, ``1m``, ``1h``.

    Raises ``ValueError`` instead of silently mapping unknown values to 5m.
    This is important for sub-minute ingestion: a typo like ``10sec`` should
    not accidentally become a 5-minute feature horizon.
    """
    tf = str(timeframe).strip().lower()
    if tf in _TF_SECONDS:
        return _TF_SECONDS[tf]
    if len(tf) < 2:
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    unit = tf[-1]
    try:
        value = int(tf[:-1])
    except ValueError as exc:
        raise ValueError(f"invalid timeframe: {timeframe!r}") from exc
    if value <= 0:
        raise ValueError(f"timeframe must be positive: {timeframe!r}")
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 60 * 60
    if unit == "d":
        return value * 24 * 60 * 60
    raise ValueError(f"unsupported timeframe unit in {timeframe!r}")


def parse_timeframe_to_minutes(timeframe: str) -> float:
    """Return the number of minutes per bar for the given timeframe string.

    Examples: '10s' -> 0.1666..., '5m' -> 5, '1h' -> 60.
    Legacy callers expect unknown values to fall back to 5 minutes; strict
    validation for new sub-minute paths lives in ``parse_timeframe_to_seconds``.
    """
    try:
        return parse_timeframe_to_seconds(timeframe) / 60.0
    except ValueError:
        return 5


def bars_per_day(timeframe: str) -> int:
    """Return the number of bars per UTC calendar day for the given timeframe."""
    seconds = parse_timeframe_to_seconds(timeframe)
    return int((24 * 60 * 60) // seconds)


def bars_per_year(timeframe: str, days: int = 365) -> float:
    """Return the number of bars per year for annualization (Sharpe, Sortino)."""
    return bars_per_day(timeframe) * days


def hours_to_bars(hours: float, timeframe: str) -> int:
    """Convert real-world hours to bar count for the given timeframe."""
    seconds = parse_timeframe_to_seconds(timeframe)
    return max(1, int(hours * 60 * 60 / seconds))


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
        seconds = int(round(median_diff_ms / 1000))
    else:
        ts_parsed = pd.to_datetime(ts, utc=True)
        diffs = ts_parsed.diff().dropna()
        if diffs.empty:
            return None
        median_diff = diffs.median()
        if pd.isna(median_diff):
            return None
        seconds = int(round(median_diff.total_seconds()))

    if seconds <= 0:
        return None

    rev_map = {v: k for k, v in _TF_SECONDS.items()}
    if seconds in rev_map:
        return rev_map[seconds]
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def default_horizon_bars(timeframe: str, target_hours: float = 6.0) -> int:
    """Compute horizon_bars that targets *target_hours* of real time.

    Different timeframes need different bar counts for the same holding period.
    E.g. 6 hours = 72 bars at 5m, 12 bars at 30m, 6 bars at 1h.
    Falls back to 12 for unrecognised timeframes (legacy retrain behavior).
    """
    try:
        seconds = parse_timeframe_to_seconds(timeframe)
    except ValueError:
        return 12
    return max(1, int(math.ceil(target_hours * 60 * 60 / seconds)))
