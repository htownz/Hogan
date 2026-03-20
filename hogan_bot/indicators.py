"""Ripster EMA clouds, ICT Fair-Value Gaps, and ATR for Hogan."""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder-smoothed Average True Range (ATR).

    True Range = max(high − low, |high − prev_close|, |low − prev_close|).
    Smoothed with an EWM whose span equals *window* (Wilder's method).
    Returns a Series aligned to *df*'s index.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    _missing = {"high", "low", "close"} - set(df.columns)
    if _missing:
        raise KeyError(f"compute_atr requires columns {{'high', 'low', 'close'}}; missing {_missing}")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).rename("hl"),
            (high - prev_close).abs().rename("hpc"),
            (low - prev_close).abs().rename("lpc"),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()


# ---------------------------------------------------------------------------
# Ripster EMA Clouds
# ---------------------------------------------------------------------------


def ripster_ema_clouds(
    df: pd.DataFrame,
    fast_short: int = 8,
    fast_long: int = 9,
    slow_short: int = 34,
    slow_long: int = 50,
) -> pd.DataFrame:
    """Return a copy of *df* with four Ripster EMA cloud columns added.

    Fast cloud : EMA(fast_short) and EMA(fast_long)   — default 8 and 9
    Slow cloud : EMA(slow_short) and EMA(slow_long)   — default 34 and 50

    Uses ``pandas.ewm(span=N, adjust=False)`` so no extra dependencies are
    needed beyond the base requirements.
    """
    out = df.copy()
    close = out["close"].astype(float)
    out["ema_fast_short"] = close.ewm(span=fast_short, adjust=False).mean()
    out["ema_fast_long"] = close.ewm(span=fast_long, adjust=False).mean()
    out["ema_slow_short"] = close.ewm(span=slow_short, adjust=False).mean()
    out["ema_slow_long"] = close.ewm(span=slow_long, adjust=False).mean()
    return out


def cloud_signal(df: pd.DataFrame) -> pd.Series:
    """Return a per-bar cloud direction Series.

    Requires ``ripster_ema_clouds()`` to have been called first so that
    *df* contains the four ``ema_*`` columns.

    'bullish' : fast cloud sits entirely above the slow cloud
                (ema_fast_short > ema_slow_long)
    'bearish' : fast cloud sits entirely below the slow cloud
                (ema_fast_long < ema_slow_short)
    'neutral' : clouds are overlapping or indeterminate
    """
    bull = df["ema_fast_short"] > df["ema_slow_long"]
    bear = df["ema_fast_long"] < df["ema_slow_short"]
    return pd.Series(
        np.select([bull, bear], ["bullish", "bearish"], default="neutral"),
        index=df.index,
    )


# ---------------------------------------------------------------------------
# ICT Fair-Value Gaps
# ---------------------------------------------------------------------------


def detect_fvgs(df: pd.DataFrame, min_gap_pct: float = 0.001) -> list[dict]:
    """Detect ICT Fair-Value Gaps across all bars in *df*.

    Bullish FVG at bar i : ``low[i] > high[i-2]``  (price gapped up)
    Bearish FVG at bar i : ``high[i] < low[i-2]``  (price gapped down)

    *min_gap_pct* — minimum gap size expressed as a fraction of the middle
    candle's close price (default 0.1 %).  Gaps smaller than this are
    discarded to eliminate noise.

    Each record in the returned list has the shape::

        {
            "direction": "bull" | "bear",
            "top":       float,   # upper boundary of the gap zone
            "bottom":    float,   # lower boundary of the gap zone
            "formed_at": int,     # iloc position (bar index) of formation
            "filled":    bool,    # True if any subsequent close entered the zone
        }

    The ``filled`` flag is set based on the entire history available in *df*
    (suitable for strategy use and backtesting).  For point-in-time ML
    features use :func:`fvg_features_frame` instead.
    """
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)

    fvgs: list[dict] = []
    for i in range(2, n):
        if low[i] > high[i - 2]:
            gap_bottom = float(high[i - 2])
            gap_top = float(low[i])
            if (gap_top - gap_bottom) / max(close[i - 1], 1e-9) >= min_gap_pct:
                fvgs.append(
                    {
                        "direction": "bull",
                        "top": gap_top,
                        "bottom": gap_bottom,
                        "formed_at": i,
                        "filled": False,
                    }
                )
        elif high[i] < low[i - 2]:
            gap_top = float(low[i - 2])
            gap_bottom = float(high[i])
            if (gap_top - gap_bottom) / max(close[i - 1], 1e-9) >= min_gap_pct:
                fvgs.append(
                    {
                        "direction": "bear",
                        "top": gap_top,
                        "bottom": gap_bottom,
                        "formed_at": i,
                        "filled": False,
                    }
                )

    for fvg in fvgs:
        start = fvg["formed_at"] + 1
        if start < n:
            subsequent = close[start:]
            fvg["filled"] = bool(
                ((subsequent >= fvg["bottom"]) & (subsequent <= fvg["top"])).any()
            )

    return fvgs


def active_fvgs(fvg_list: list[dict]) -> list[dict]:
    """Return only the unfilled FVG records from *fvg_list*."""
    return [g for g in fvg_list if not g["filled"]]


def fvg_entry_signal(fvg_list: list[dict], close_price: float) -> str:
    """Return ``'buy'``, ``'sell'``, or ``'hold'``.

    Checks whether *close_price* is inside an active (unfilled) FVG zone.
    When multiple zones overlap, the most recently formed gap wins (LIFO).
    A bullish zone triggers ``'buy'``; a bearish zone triggers ``'sell'``.
    """
    live = sorted(active_fvgs(fvg_list), key=lambda g: g["formed_at"], reverse=True)
    for gap in live:
        if gap["bottom"] <= close_price <= gap["top"]:
            return "buy" if gap["direction"] == "bull" else "sell"
    return "hold"


def fvg_features_frame(df: pd.DataFrame, min_gap_pct: float = 0.001) -> pd.DataFrame:
    """Compute point-in-time FVG features for every bar without look-ahead bias.

    Unlike :func:`detect_fvgs`, the ``filled`` status is only updated with
    information available *up to and including* each bar, making these
    features safe for ML training.

    Returns a :class:`~pandas.DataFrame` aligned to *df*'s index with
    four integer columns:

    ``fvg_bull_active`` — count of unfilled bullish FVGs at the start of each bar
    ``fvg_bear_active`` — count of unfilled bearish FVGs at the start of each bar
    ``in_bull_fvg``     — 1 if the bar's close is inside an active bullish zone
    ``in_bear_fvg``     — 1 if the bar's close is inside an active bearish zone

    Note: a gap is only counted as *active* at the bar it is formed; it
    cannot be *filled* by the same bar that forms it (the close of the
    formation candle lies outside the gap zone by construction).
    """
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)

    bull_active_arr = [0] * n
    bear_active_arr = [0] * n
    in_bull_arr = [0] * n
    in_bear_arr = [0] * n

    live: list[dict] = []

    for i in range(n):
        if i >= 2:
            if low[i] > high[i - 2]:
                gap_bottom = float(high[i - 2])
                gap_top = float(low[i])
                if (gap_top - gap_bottom) / max(close[i - 1], 1e-9) >= min_gap_pct:
                    live.append(
                        {"direction": "bull", "top": gap_top, "bottom": gap_bottom, "filled": False}
                    )
            elif high[i] < low[i - 2]:
                gap_top = float(low[i - 2])
                gap_bottom = float(high[i])
                if (gap_top - gap_bottom) / max(close[i - 1], 1e-9) >= min_gap_pct:
                    live.append(
                        {"direction": "bear", "top": gap_top, "bottom": gap_bottom, "filled": False}
                    )

        bull_active_arr[i] = sum(1 for g in live if g["direction"] == "bull" and not g["filled"])
        bear_active_arr[i] = sum(1 for g in live if g["direction"] == "bear" and not g["filled"])

        for gap in live:
            if not gap["filled"] and gap["bottom"] <= close[i] <= gap["top"]:
                if gap["direction"] == "bull":
                    in_bull_arr[i] = 1
                else:
                    in_bear_arr[i] = 1
                gap["filled"] = True

    return pd.DataFrame(
        {
            "fvg_bull_active": bull_active_arr,
            "fvg_bear_active": bear_active_arr,
            "in_bull_fvg": in_bull_arr,
            "in_bear_fvg": in_bear_arr,
        },
        index=df.index,
    )
