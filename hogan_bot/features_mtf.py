"""Multi-timeframe (MTF) and external feature builder for the Hogan RL agent.

Provides :func:`build_feature_row_extended` which returns a 44-element
feature vector:

    24 base 5m features  (from :func:`~hogan_bot.ml.build_feature_row`)
  + 14 MTF features      (7 from 1h + 7 from 15m, see table below)
  + 6 ext features       (derivatives / on-chain / macro, zeros when unavailable)
  = 44 ML features

Combined with 3 position-state scalars in :class:`~hogan_bot.rl_env.TradingEnv`
the final observation vector is 47-dimensional.

MTF feature table
-----------------
+--------------------+-------------------------------------------+
| h1_ret_1           | 1h close-to-close return                  |
| h1_rsi_14          | RSI(14) on 1h closes (normalised to [0,1]) |
| h1_atr_pct         | ATR(14) / close on 1h                     |
| h1_macd_hist       | MACD histogram / close on 1h              |
| h1_bb_pct_b        | Bollinger %B on 1h                        |
| h1_vol_ratio       | Volume vs. 20-bar 1h average              |
| h1_trend_up        | 1h close > 1h 20-bar MA  (0 / 1)         |
| m15_ret_1          | 15m return                                |
| m15_rsi_14         | RSI(14) on 15m closes (normalised)        |
| m15_atr_pct        | ATR(14) / close on 15m                   |
| m15_macd_hist      | MACD histogram / close on 15m            |
| m15_bb_pct_b       | Bollinger %B on 15m                      |
| m15_vol_ratio      | Volume ratio on 15m                      |
| m15_trend_up       | 15m close > 15m 20-bar MA (0 / 1)        |
+--------------------+-------------------------------------------+

External feature table (6 features — zeros when data unavailable)
-------------------------------------------------------------------
+---------------------------+--------------------------------------------+
| funding_rate              | Kraken Futures funding rate (clipped ±0.005)|
| open_interest_pct_change  | 1-day OI % change                          |
| mvrv_zscore               | MVRV z-score (365-day rolling)             |
| sopr                      | Spent Output Profit Ratio (clipped 0.95-1.05)|
| active_addr_ma7_pct_change| 7-day MA of active addresses % change      |
| spy_ret_1d                | Previous-day SPY return                    |
+---------------------------+--------------------------------------------+
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hogan_bot.ml import build_feature_row

# Names for the 6 extended (external) features — used for DB lookups
EXT_FEATURE_NAMES: list[str] = [
    "funding_rate",
    "open_interest_pct_change",
    "mvrv_zscore",
    "sopr",
    "active_addr_ma7_pct_change",
    "spy_ret_1d",
]

_MTF_MIN_BARS: int = 20   # minimum candles needed to compute MTF features


# ---------------------------------------------------------------------------
# Internal helpers — single-timeframe feature computation
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.clip(lower=1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=window - 1, min_periods=window, adjust=False).mean()


def _compute_tf_features(candles: pd.DataFrame) -> list[float] | None:
    """Return 7 features for the LAST bar in *candles*.

    Features:
        ret_1, rsi_14 (norm), atr_pct, macd_hist_pct, bb_pct_b,
        vol_ratio, trend_up
    """
    if candles is None or len(candles) < _MTF_MIN_BARS:
        return None

    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    volume = candles["volume"].astype(float)

    # 1. Close-to-close return
    ret_1 = float(close.pct_change(1).iloc[-1])

    # 2. RSI normalised to [0, 1]
    rsi_val = float(_rsi(close, 14).iloc[-1]) / 100.0

    # 3. ATR / close
    atr_val = float(_atr(high, low, close, 14).iloc[-1])
    atr_pct = atr_val / max(float(close.iloc[-1]), 1e-9)

    # 4. MACD histogram / close
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = float((macd_line - signal_line).iloc[-1]) / max(float(close.iloc[-1]), 1e-9)

    # 5. Bollinger %B
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower).clip(lower=1e-9)
    bb_pct_b = float(((close - bb_lower) / bb_width).iloc[-1])

    # 6. Volume ratio
    vol_avg = volume.rolling(20).mean().clip(lower=1e-9)
    vol_ratio = float((volume / vol_avg).iloc[-1])

    # 7. Trend: close > 20-bar MA (0 or 1)
    ma20 = float(close.rolling(20).mean().iloc[-1])
    trend_up = 1.0 if float(close.iloc[-1]) > ma20 else 0.0

    vals = [ret_1, rsi_val, atr_pct, macd_hist, bb_pct_b, vol_ratio, trend_up]

    # Return None if any value is NaN
    if any(np.isnan(v) for v in vals):
        return None
    return [float(v) for v in vals]


# ---------------------------------------------------------------------------
# External feature loader
# ---------------------------------------------------------------------------

def build_ext_features(
    current_ts: pd.Timestamp | None,
    conn=None,
    symbol: str = "BTC/USD",
) -> list[float]:
    """Return the 6 external features aligned to *current_ts*.

    Fetches from the ``derivatives_metrics`` and ``onchain_metrics`` SQLite
    tables (and ``candles`` for SPY).  Returns zeros when data is unavailable.

    Parameters
    ----------
    current_ts:
        The 5m bar timestamp for which to look up the external values.
    conn:
        An open SQLite connection.  When ``None`` no DB lookup is performed.
    symbol:
        The trading symbol, e.g. ``"BTC/USD"``.
    """
    zeros = [0.0] * len(EXT_FEATURE_NAMES)
    if conn is None or current_ts is None:
        return zeros

    try:
        ts_ms = int(pd.Timestamp(current_ts).timestamp() * 1000)
        result = list(zeros)  # mutable copy

        # ── Derivatives metrics (funding rate, OI pct change) ───────────
        for i, metric in enumerate(["funding_rate", "open_interest_pct_change"]):
            row = conn.execute(
                """
                SELECT value FROM derivatives_metrics
                WHERE symbol = ? AND metric = ? AND ts_ms <= ?
                ORDER BY ts_ms DESC LIMIT 1
                """,
                (symbol, metric, ts_ms),
            ).fetchone()
            if row:
                val = float(row[0])
                if metric == "funding_rate":
                    val = float(np.clip(val, -0.005, 0.005))
                result[i] = val

        # ── On-chain metrics (MVRV z-score, SOPR, active addresses) ────
        ts_date = pd.Timestamp(current_ts).strftime("%Y-%m-%d")
        onchain_map = {
            "mvrv_zscore": 2,
            "sopr": 3,
            "active_addr_ma7_pct_change": 4,
        }
        for metric, idx in onchain_map.items():
            row = conn.execute(
                """
                SELECT value FROM onchain_metrics
                WHERE symbol = ? AND metric = ? AND date <= ?
                ORDER BY date DESC LIMIT 1
                """,
                (symbol, metric, ts_date),
            ).fetchone()
            if row:
                val = float(row[0])
                if metric == "sopr":
                    val = float(np.clip(val, 0.95, 1.05))
                result[idx] = val

        # ── SPY previous-day return ──────────────────────────────────────
        spy_row = conn.execute(
            """
            SELECT close FROM candles
            WHERE symbol = 'SPY/USD' AND timeframe = '1d' AND ts_ms <= ?
            ORDER BY ts_ms DESC LIMIT 2
            """,
            (ts_ms,),
        ).fetchall()
        if len(spy_row) >= 2:
            prev_close, curr_close = float(spy_row[1][0]), float(spy_row[0][0])
            if prev_close > 0:
                result[5] = (curr_close - prev_close) / prev_close
        return result

    except Exception:
        return zeros


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_row_extended(
    candles_5m: pd.DataFrame,
    candles_1h: pd.DataFrame | None = None,
    candles_15m: pd.DataFrame | None = None,
    conn=None,
    symbol: str = "BTC/USD",
) -> list[float] | None:
    """Return the 44-element extended feature vector for the last bar.

    Computes:
    * 24 base 5m features  via :func:`~hogan_bot.ml.build_feature_row`
    * 14 MTF features       (7 from 1h + 7 from 15m; zeros when unavailable)
    * 6 ext features        (from DB; zeros when unavailable)

    Parameters
    ----------
    candles_5m:
        5-minute OHLCV window (oldest-first, sliced to current bar).
    candles_1h:
        1-hour OHLCV window up to current bar (or ``None``).
    candles_15m:
        15-minute OHLCV window up to current bar (or ``None``).
    conn:
        Open SQLite connection for ext feature lookup (or ``None``).
    symbol:
        Trading symbol for DB lookups.

    Returns
    -------
    list[float] of length 44, or ``None`` when base features cannot be computed.
    """
    # 24 base 5m features
    base = build_feature_row(candles_5m)
    if base is None:
        return None

    # 14 MTF features — 7 from 1h
    h1_feats = _compute_tf_features(candles_1h) or [0.0] * 7
    # 7 from 15m
    m15_feats = _compute_tf_features(candles_15m) or [0.0] * 7

    # 6 ext features
    ts = (
        candles_5m["timestamp"].iloc[-1]
        if "timestamp" in candles_5m.columns
        else None
    )
    ext = build_ext_features(ts, conn=conn, symbol=symbol)

    combined = base + h1_feats + m15_feats + ext  # 24 + 7 + 7 + 6 = 44
    return combined


# Feature column names for documentation / debugging
MTF_FEATURE_COLUMNS: list[str] = [
    "h1_ret_1", "h1_rsi_14", "h1_atr_pct", "h1_macd_hist",
    "h1_bb_pct_b", "h1_vol_ratio", "h1_trend_up",
    "m15_ret_1", "m15_rsi_14", "m15_atr_pct", "m15_macd_hist",
    "m15_bb_pct_b", "m15_vol_ratio", "m15_trend_up",
]

EXTENDED_FEATURE_COLUMNS: list[str] = (
    # base 24 — imported from ml.py at runtime to avoid circular import
    MTF_FEATURE_COLUMNS
    + EXT_FEATURE_NAMES
)
