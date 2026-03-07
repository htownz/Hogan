"""Macro-asset feature builder for Hogan.

Reads OHLCV candles for SPY, QQQ, GLD, SLV, TLT, UUP, VIX, and TNX from the
``candles`` table and produces a 10-feature macro context vector:

  macro_spy_trend     SPY daily close > 20d SMA (0/1 — bull equity market)
  macro_spy_ret       SPY 1-day return ÷ 5, clipped [-1, +1]
  macro_vix_norm      VIX daily close ÷ 40, clipped [0, 1.5] (fear gauge)
  macro_vix_high      VIX > 20 threshold (binary fear flag)
  macro_gld_trend     GLD daily close > 20d SMA (0/1 — risk-off precious metals)
  macro_tlt_ret       TLT 1-day return ÷ 3, clipped [-1, +1] (bond direction)
  macro_uup_trend     UUP daily close > 20d SMA (0/1 — dollar strength)
  macro_tnx_norm      TNX daily close ÷ 7, clipped [0, 1] (10Y yield env)
  macro_risk_off      Composite: (VIX>20) + GLD_trend + TLT_trend - SPY_trend ÷ 3
  macro_qqq_spy_rel   QQQ 5d return − SPY 5d return (tech vs broad divergence)

Design goals
------------
* **No look-ahead leakage** — for each 5m bar timestamp, only macro candles
  BEFORE that timestamp are used (pd.merge_asof with ``direction="backward"``).
* **Training efficiency** — all macro candles are loaded into memory once and
  looked up via vectorised merge (not one DB query per row).
* **Graceful degradation** — returns zeros when macro data is unavailable,
  so the bot never crashes due to a missing macro source.

Usage
-----
    # Build macro features for a batch of 5m candles (training):
    from hogan_bot.macro_features import add_macro_features
    df_with_macro = add_macro_features(candles_5m, conn)

    # Single-row inference:
    from hogan_bot.macro_features import get_macro_feature_row
    row = get_macro_feature_row(conn, ts_ms=current_ts_ms)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names (order is authoritative — do not reorder)
# ---------------------------------------------------------------------------

MACRO_FEATURE_NAMES: list[str] = [
    "macro_spy_trend",    # SPY daily close > 20d SMA (0/1)
    "macro_spy_ret",      # SPY 1d return ÷ 5, clipped [-1, +1]
    "macro_vix_norm",     # VIX daily close ÷ 40, clipped [0, 1.5]
    "macro_vix_high",     # VIX > 20 threshold (binary)
    "macro_gld_trend",    # GLD daily close > 20d SMA (0/1)
    "macro_tlt_ret",      # TLT 1d return ÷ 3, clipped [-1, +1]
    "macro_uup_trend",    # UUP daily close > 20d SMA (dollar bull)
    "macro_tnx_norm",     # TNX daily close ÷ 7, clipped [0, 1]
    "macro_risk_off",     # Composite risk-off score ÷ 3 → [−0.33, 1]
    "macro_qqq_spy_rel",  # QQQ 5d return − SPY 5d return (tech divergence)
]

N_MACRO_FEATURES: int = len(MACRO_FEATURE_NAMES)  # 10

# Sentinel returned when data is unavailable
_ZEROS: list[float] = [0.0] * N_MACRO_FEATURES

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_daily_candles(conn: "sqlite3.Connection", symbol: str, limit: int = 300) -> pd.DataFrame:
    """Load the most recent *limit* daily candles for *symbol*."""
    rows = conn.execute(
        """
        SELECT ts_ms, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND timeframe = '1d'
        ORDER BY ts_ms DESC
        LIMIT ?
        """,
        (symbol, limit),
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df


def _compute_macro_table(conn: "sqlite3.Connection") -> pd.DataFrame:
    """Pre-compute all macro features aligned to daily timestamps.

    Returns a DataFrame indexed by ``ts_ms`` with one row per trading day,
    containing all 10 macro features.  Used by :func:`add_macro_features`
    to do a fast vectorised merge.
    """
    assets = {
        "SPY": _load_daily_candles(conn, "SPY/USD"),
        "QQQ": _load_daily_candles(conn, "QQQ/USD"),
        "GLD": _load_daily_candles(conn, "GLD/USD"),
        "TLT": _load_daily_candles(conn, "TLT/USD"),
        "UUP": _load_daily_candles(conn, "UUP/USD"),
        "VIX": _load_daily_candles(conn, "VIX/USD"),
        "TNX": _load_daily_candles(conn, "TNX/USD"),
    }

    # Need at least SPY and VIX to build anything useful
    if assets["SPY"].empty or assets["VIX"].empty:
        return pd.DataFrame()

    def _sma_trend(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """1 if close > SMA(window), else 0."""
        sma = df["close"].rolling(window, min_periods=window).mean()
        return (df["close"] > sma).astype(float)

    def _ret(df: pd.DataFrame, n: int = 1) -> pd.Series:
        return df["close"].pct_change(n)

    # Align all assets to SPY's timestamp index using merge_asof
    base_ts = assets["SPY"]["ts_ms"].copy()

    rows: dict[str, pd.Series] = {"ts_ms": base_ts}

    # SPY
    spy = assets["SPY"]
    spy_trend = _sma_trend(spy)
    spy_ret = _ret(spy, 1)
    spy_ret5 = _ret(spy, 5)
    rows["macro_spy_trend"] = spy_trend.values
    rows["macro_spy_ret"]   = (spy_ret / 5.0).clip(-1.0, 1.0).values

    # VIX
    vix = assets["VIX"]
    if not vix.empty:
        vix_aligned = pd.merge_asof(
            base_ts.to_frame("ts_ms"),
            vix[["ts_ms", "close"]].rename(columns={"close": "vix_close"}),
            on="ts_ms", direction="backward",
        )
        vix_c = vix_aligned["vix_close"].fillna(20.0)
        rows["macro_vix_norm"] = (vix_c / 40.0).clip(0.0, 1.5).values
        rows["macro_vix_high"] = (vix_c > 20.0).astype(float).values
    else:
        rows["macro_vix_norm"] = np.zeros(len(base_ts))
        rows["macro_vix_high"] = np.zeros(len(base_ts))

    # GLD
    gld = assets["GLD"]
    if not gld.empty:
        gld_aligned = pd.merge_asof(
            base_ts.to_frame("ts_ms"),
            gld[["ts_ms", "close"]].copy(),
            on="ts_ms", direction="backward",
        )
        gld_close = gld_aligned["close"].ffill().fillna(0.0)
        # trend: close > 20d SMA — need to compute on aligned series
        gld_sma = gld_close.rolling(20, min_periods=5).mean()
        rows["macro_gld_trend"] = (gld_close > gld_sma).astype(float).values
    else:
        rows["macro_gld_trend"] = np.zeros(len(base_ts))

    # TLT
    tlt = assets["TLT"]
    if not tlt.empty:
        tlt_aligned = pd.merge_asof(
            base_ts.to_frame("ts_ms"),
            tlt[["ts_ms", "close"]].copy(),
            on="ts_ms", direction="backward",
        )
        tlt_close = tlt_aligned["close"].ffill().fillna(0.0)
        tlt_ret = tlt_close.pct_change(1)
        tlt_sma = tlt_close.rolling(20, min_periods=5).mean()
        rows["macro_tlt_ret"]   = (tlt_ret / 3.0).clip(-1.0, 1.0).values
        tlt_trend = (tlt_close > tlt_sma).astype(float)
    else:
        rows["macro_tlt_ret"] = np.zeros(len(base_ts))
        tlt_trend = pd.Series(np.zeros(len(base_ts)))

    # UUP (dollar proxy)
    uup = assets["UUP"]
    if not uup.empty:
        uup_aligned = pd.merge_asof(
            base_ts.to_frame("ts_ms"),
            uup[["ts_ms", "close"]].copy(),
            on="ts_ms", direction="backward",
        )
        uup_close = uup_aligned["close"].ffill().fillna(0.0)
        uup_sma = uup_close.rolling(20, min_periods=5).mean()
        rows["macro_uup_trend"] = (uup_close > uup_sma).astype(float).values
    else:
        rows["macro_uup_trend"] = np.zeros(len(base_ts))

    # TNX (10Y yield)
    tnx = assets["TNX"]
    if not tnx.empty:
        tnx_aligned = pd.merge_asof(
            base_ts.to_frame("ts_ms"),
            tnx[["ts_ms", "close"]].copy(),
            on="ts_ms", direction="backward",
        )
        tnx_close = tnx_aligned["close"].ffill().fillna(4.0)
        rows["macro_tnx_norm"] = (tnx_close / 7.0).clip(0.0, 1.0).values
    else:
        rows["macro_tnx_norm"] = np.zeros(len(base_ts))

    # Composite risk-off: (VIX>20) + GLD_trend + TLT_trend - SPY_trend → ÷3
    risk_off = (
        pd.Series(rows["macro_vix_high"])
        + pd.Series(rows["macro_gld_trend"])
        + tlt_trend.values
        - pd.Series(rows["macro_spy_trend"])
    ) / 3.0
    rows["macro_risk_off"] = risk_off.values

    # QQQ vs SPY relative strength (5-day divergence)
    qqq = assets["QQQ"]
    if not qqq.empty:
        qqq_aligned = pd.merge_asof(
            base_ts.to_frame("ts_ms"),
            qqq[["ts_ms", "close"]].copy(),
            on="ts_ms", direction="backward",
        )
        qqq_close = qqq_aligned["close"].ffill().fillna(0.0)
        qqq_ret5  = qqq_close.pct_change(5)
        rows["macro_qqq_spy_rel"] = (qqq_ret5 - spy_ret5).clip(-0.2, 0.2).values
    else:
        rows["macro_qqq_spy_rel"] = np.zeros(len(base_ts))

    table = pd.DataFrame(rows)
    return table.dropna(subset=["ts_ms"]).sort_values("ts_ms").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API — batch (training)
# ---------------------------------------------------------------------------

def add_macro_features(
    candles: pd.DataFrame,
    conn: "sqlite3.Connection",
) -> pd.DataFrame:
    """Join macro features onto a 5m candle DataFrame (for training).

    Uses ``pd.merge_asof`` with ``direction="backward"`` so each 5m row
    gets the most recent macro state that was available BEFORE that bar —
    no look-ahead leakage.

    Parameters
    ----------
    candles : DataFrame
        Must contain a ``ts_ms`` column (Unix epoch milliseconds).
    conn : sqlite3.Connection
        Open connection to the Hogan DB.

    Returns
    -------
    DataFrame with 10 extra columns (``macro_*``).
    Fills zeros when macro data is unavailable for a given timestamp.
    """
    out = candles.copy()
    for col in MACRO_FEATURE_NAMES:
        out[col] = 0.0

    if conn is None:
        return out

    try:
        macro_table = _compute_macro_table(conn)
        if macro_table.empty:
            return out

        # Ensure ts_ms column exists in candles
        if "ts_ms" not in out.columns and "timestamp" in out.columns:
            out["ts_ms"] = (
                pd.to_datetime(out["timestamp"]).astype("int64") // 1_000_000
            )

        # merge_asof requires both sides sorted by key
        out_sorted = out.sort_values("ts_ms").copy()
        macro_sorted = macro_table[["ts_ms"] + MACRO_FEATURE_NAMES].sort_values("ts_ms")

        merged = pd.merge_asof(
            out_sorted,
            macro_sorted,
            on="ts_ms",
            direction="backward",
            suffixes=("", "_macro"),
        )
        # Fill any NaN macro values with 0
        for col in MACRO_FEATURE_NAMES:
            merged[col] = merged[col].fillna(0.0)

        # Restore original row order
        merged = merged.set_index(out_sorted.index).reindex(out.index)
        for col in MACRO_FEATURE_NAMES:
            out[col] = merged[col].values

    except Exception as exc:
        logger.warning("add_macro_features failed: %s — using zeros", exc)

    return out


# ---------------------------------------------------------------------------
# Public API — single row (live inference)
# ---------------------------------------------------------------------------

def get_macro_feature_row(
    conn: "sqlite3.Connection",
    ts_ms: int | None = None,
) -> list[float]:
    """Return the 10 macro features for the most recent available data.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open DB connection.
    ts_ms : int, optional
        Cutoff timestamp in milliseconds.  Uses current time if ``None``.

    Returns
    -------
    list of 10 floats (zeros on any failure).
    """
    if conn is None:
        return list(_ZEROS)

    try:
        import time
        cutoff = ts_ms or int(time.time() * 1000)

        macro_table = _compute_macro_table(conn)
        if macro_table.empty:
            return list(_ZEROS)

        # Find the last row before cutoff
        before = macro_table[macro_table["ts_ms"] <= cutoff]
        if before.empty:
            return list(_ZEROS)

        row = before.iloc[-1]
        return [float(row.get(col, 0.0)) for col in MACRO_FEATURE_NAMES]

    except Exception as exc:
        logger.warning("get_macro_feature_row failed: %s", exc)
        return list(_ZEROS)
