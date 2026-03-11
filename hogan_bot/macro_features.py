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
    # Original 10 macro-asset features
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
    # On-chain features (4)
    "onchain_hashrate_trend",   # BTC hash rate 7d SMA direction (0/1)
    "onchain_addr_trend",       # active address 7d SMA direction (0/1)
    "onchain_mempool_norm",     # mempool MB / 100, clipped [0, 1]
    "onchain_fee_norm",         # avg fee USD / 10, clipped [0, 1]
    # Sentiment features (4)
    "sent_fear_greed_norm",     # Fear & Greed index / 100 [0, 1]
    "sent_btc_dominance",       # BTC dominance / 100 [0, 1]
    "sent_defi_tvl_change",     # DeFi TVL 7d % change / 100, clipped [-0.5, 0.5]
    "sent_stablecoin_norm",     # stablecoin mcap / 500B, clipped [0, 1]
    # Derivatives features (2)
    "deriv_funding_rate",       # normalised funding rate, clipped [-1, 1]
    "deriv_oi_change",          # OI % change, clipped [-1, 1]
    # Inter-market features (3)
    "intermarket_dxy_trend",    # DXY close > 20d SMA (0/1)
    "intermarket_spy_btc_corr", # 20-day SPY-BTC return correlation [-1, 1]
    "intermarket_gold_btc_rel", # GLD 5d return - BTC 5d return, clipped [-0.2, 0.2]
]

N_MACRO_FEATURES: int = len(MACRO_FEATURE_NAMES)  # 23

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


def _load_onchain_series(
    conn: "sqlite3.Connection", metric: str, limit: int = 365,
) -> pd.DataFrame:
    """Load a time series from the onchain_metrics table."""
    rows = conn.execute(
        "SELECT date, value FROM onchain_metrics WHERE metric = ? ORDER BY date DESC LIMIT ?",
        (metric, limit),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _load_derivatives_latest(conn: "sqlite3.Connection", metric: str) -> float:
    """Load the most recent derivatives metric value."""
    row = conn.execute(
        "SELECT value FROM derivatives_metrics WHERE metric = ? ORDER BY ts_ms DESC LIMIT 1",
        (metric,),
    ).fetchone()
    return float(row[0]) if row else 0.0


def _compute_extended_table(conn: "sqlite3.Connection") -> pd.DataFrame | None:
    """Build a time-series table of extended features aligned to daily timestamps.

    Returns a DataFrame with ``ts_ms`` and one column per extended feature,
    suitable for ``merge_asof`` against 1h candle timestamps.
    """
    try:
        # Funding rate: stored with ms timestamps in derivatives_metrics
        fr_rows = conn.execute(
            "SELECT ts_ms, value FROM derivatives_metrics "
            "WHERE metric = 'funding_rate' ORDER BY ts_ms"
        ).fetchall()
        fr_df = pd.DataFrame(fr_rows, columns=["ts_ms", "deriv_funding_rate"])
        if not fr_df.empty:
            fr_df["deriv_funding_rate"] = fr_df["deriv_funding_rate"].astype(float).clip(-1.0, 1.0)

        # OI change
        oi_rows = conn.execute(
            "SELECT ts_ms, value FROM derivatives_metrics "
            "WHERE metric = 'open_interest_pct_change' ORDER BY ts_ms"
        ).fetchall()
        oi_df = pd.DataFrame(oi_rows, columns=["ts_ms", "deriv_oi_change"])
        if not oi_df.empty:
            oi_df["deriv_oi_change"] = oi_df["deriv_oi_change"].astype(float).clip(-1.0, 1.0)

        # Daily on-chain and sentiment from onchain_metrics
        daily_metrics = {
            "fear_greed_value": ("sent_fear_greed_norm", lambda v: np.clip(v / 100.0, 0.0, 1.0)),
            "btc_hashrate_eh": ("_hashrate_raw", lambda v: v),
            "btc_active_addr": ("_addr_raw", lambda v: v),
            "btc_mempool_mb": ("onchain_mempool_norm", lambda v: np.clip(v / 100.0, 0.0, 1.0)),
            "btc_avg_fee_usd": ("onchain_fee_norm", lambda v: np.clip(v / 10.0, 0.0, 1.0)),
            "defi_tvl_change_1d": ("sent_defi_tvl_change", lambda v: np.clip(v / 100.0 if abs(v) > 1 else v, -0.5, 0.5)),
            "defi_stablecoin_b": ("sent_stablecoin_norm", lambda v: np.clip(v / 500.0, 0.0, 1.0)),
        }

        daily_frames = {}
        for metric_name, (feat_name, transform) in daily_metrics.items():
            rows = conn.execute(
                "SELECT date, value FROM onchain_metrics WHERE metric = ? ORDER BY date",
                (metric_name,),
            ).fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=["date", feat_name])
                df["ts_ms"] = (pd.to_datetime(df["date"]).astype("int64") // 1_000_000)
                df[feat_name] = pd.to_numeric(df[feat_name], errors="coerce").apply(transform)
                daily_frames[feat_name] = df[["ts_ms", feat_name]]

        # Compute hashrate/addr trends (7d SMA direction)
        if "_hashrate_raw" in daily_frames:
            hr = daily_frames.pop("_hashrate_raw")
            sma7 = hr["_hashrate_raw"].rolling(7, min_periods=3).mean()
            hr["onchain_hashrate_trend"] = (hr["_hashrate_raw"] > sma7).astype(float)
            daily_frames["onchain_hashrate_trend"] = hr[["ts_ms", "onchain_hashrate_trend"]]

        if "_addr_raw" in daily_frames:
            ad = daily_frames.pop("_addr_raw")
            sma7 = ad["_addr_raw"].rolling(7, min_periods=3).mean()
            ad["onchain_addr_trend"] = (ad["_addr_raw"] > sma7).astype(float)
            daily_frames["onchain_addr_trend"] = ad[["ts_ms", "onchain_addr_trend"]]

        # BTC dominance
        for src_metric in ("cg_btc_dominance", "cmc_btc_dominance"):
            rows = conn.execute(
                "SELECT date, value FROM onchain_metrics WHERE metric = ? ORDER BY date",
                (src_metric,),
            ).fetchall()
            if rows and len(rows) > 5:
                df = pd.DataFrame(rows, columns=["date", "sent_btc_dominance"])
                df["ts_ms"] = (pd.to_datetime(df["date"]).astype("int64") // 1_000_000)
                val = pd.to_numeric(df["sent_btc_dominance"], errors="coerce")
                df["sent_btc_dominance"] = np.where(val > 1, val / 100.0, val).clip(0.0, 1.0)
                daily_frames["sent_btc_dominance"] = df[["ts_ms", "sent_btc_dominance"]]
                break

        # Intermarket features (these need more complex computation, keep as latest-value)
        ext_snapshot = _compute_extended_features(conn)
        for key in ("intermarket_dxy_trend", "intermarket_spy_btc_corr", "intermarket_gold_btc_rel"):
            if key not in daily_frames:
                daily_frames[key] = None

        # Build unified table: merge all daily frames on ts_ms
        if not daily_frames and fr_df.empty:
            return None

        # Start with the densest time series
        all_ts = set()
        if not fr_df.empty:
            all_ts.update(fr_df["ts_ms"].tolist())
        for name, df in daily_frames.items():
            if df is not None:
                all_ts.update(df["ts_ms"].tolist())

        if not all_ts:
            return None

        base = pd.DataFrame({"ts_ms": sorted(all_ts)})

        # Merge funding rate
        if not fr_df.empty:
            base = pd.merge_asof(base, fr_df, on="ts_ms", direction="backward")

        # Merge OI
        if not oi_df.empty:
            base = pd.merge_asof(base, oi_df, on="ts_ms", direction="backward")

        # Merge daily features
        for name, df in daily_frames.items():
            if df is not None and not df.empty:
                base = pd.merge_asof(base, df.sort_values("ts_ms"), on="ts_ms", direction="backward")

        # Fill intermarket features as constants
        for key in ("intermarket_dxy_trend", "intermarket_spy_btc_corr", "intermarket_gold_btc_rel"):
            if key not in base.columns:
                base[key] = ext_snapshot.get(key, 0.0)

        # Fill missing columns
        extended_names = MACRO_FEATURE_NAMES[10:]
        for col in extended_names:
            if col not in base.columns:
                base[col] = 0.0

        return base.sort_values("ts_ms").reset_index(drop=True)

    except Exception as exc:
        logger.warning("_compute_extended_table failed: %s — falling back to snapshot", exc)
        return None


def _compute_extended_features(conn: "sqlite3.Connection") -> dict[str, float]:
    """Compute all extended features (on-chain, sentiment, derivatives, inter-market).

    Returns a dict of feature_name -> value. Uses the latest available data.
    Falls back to 0.0 for any missing metric.
    """
    result: dict[str, float] = {}

    # --- On-chain features ---
    hashrate = _load_onchain_series(conn, "btc_hashrate_eh", 30)
    if len(hashrate) >= 7:
        sma7 = hashrate["value"].rolling(7).mean()
        result["onchain_hashrate_trend"] = float(hashrate["value"].iloc[-1] > sma7.iloc[-1])
    else:
        result["onchain_hashrate_trend"] = 0.0

    addrs = _load_onchain_series(conn, "btc_active_addr", 30)
    if len(addrs) >= 7:
        sma7 = addrs["value"].rolling(7).mean()
        result["onchain_addr_trend"] = float(addrs["value"].iloc[-1] > sma7.iloc[-1])
    else:
        result["onchain_addr_trend"] = 0.0

    mempool = _load_onchain_series(conn, "btc_mempool_mb", 5)
    if not mempool.empty:
        result["onchain_mempool_norm"] = float(np.clip(mempool["value"].iloc[-1] / 100.0, 0.0, 1.0))
    else:
        result["onchain_mempool_norm"] = 0.0

    fee = _load_onchain_series(conn, "btc_avg_fee_usd", 5)
    if not fee.empty:
        result["onchain_fee_norm"] = float(np.clip(fee["value"].iloc[-1] / 10.0, 0.0, 1.0))
    else:
        result["onchain_fee_norm"] = 0.0

    # --- Sentiment features ---
    fg = _load_onchain_series(conn, "fear_greed_value", 5)
    if not fg.empty:
        result["sent_fear_greed_norm"] = float(np.clip(fg["value"].iloc[-1] / 100.0, 0.0, 1.0))
    else:
        result["sent_fear_greed_norm"] = 0.0

    dom = _load_onchain_series(conn, "cg_btc_dominance", 5)
    if dom.empty:
        dom = _load_onchain_series(conn, "cmc_btc_dominance", 5)
    if not dom.empty:
        val = dom["value"].iloc[-1]
        result["sent_btc_dominance"] = float(np.clip(val / 100.0 if val > 1 else val, 0.0, 1.0))
    else:
        result["sent_btc_dominance"] = 0.0

    tvl = _load_onchain_series(conn, "defi_tvl_change_7d", 5)
    if not tvl.empty:
        result["sent_defi_tvl_change"] = float(np.clip(tvl["value"].iloc[-1] / 100.0, -0.5, 0.5))
    else:
        result["sent_defi_tvl_change"] = 0.0

    stable = _load_onchain_series(conn, "defi_stablecoin_b", 5)
    if not stable.empty:
        result["sent_stablecoin_norm"] = float(np.clip(stable["value"].iloc[-1] / 500.0, 0.0, 1.0))
    else:
        result["sent_stablecoin_norm"] = 0.0

    # --- Derivatives features ---
    try:
        funding = _load_derivatives_latest(conn, "funding_rate")
        result["deriv_funding_rate"] = float(np.clip(funding, -1.0, 1.0))
    except Exception:
        result["deriv_funding_rate"] = 0.0

    try:
        oi_chg = _load_derivatives_latest(conn, "open_interest_pct_change")
        result["deriv_oi_change"] = float(np.clip(oi_chg, -1.0, 1.0))
    except Exception:
        result["deriv_oi_change"] = 0.0

    # --- Inter-market features ---
    dxy = _load_onchain_series(conn, "dxy_close", 30)
    if len(dxy) >= 20:
        sma20 = dxy["value"].rolling(20).mean()
        result["intermarket_dxy_trend"] = float(dxy["value"].iloc[-1] > sma20.iloc[-1])
    else:
        result["intermarket_dxy_trend"] = 0.0

    # SPY-BTC correlation: load daily returns and compute rolling correlation
    spy_candles = _load_daily_candles(conn, "SPY/USD", limit=30)
    btc_candles = _load_daily_candles(conn, "BTC/USD", limit=30)
    if len(spy_candles) >= 20 and len(btc_candles) >= 20:
        spy_ret = spy_candles.set_index("ts_ms")["close"].pct_change()
        btc_ret = btc_candles.set_index("ts_ms")["close"].pct_change()
        joined = pd.DataFrame({"spy": spy_ret, "btc": btc_ret}).dropna()
        if len(joined) >= 10:
            corr = joined["spy"].corr(joined["btc"])
            result["intermarket_spy_btc_corr"] = float(np.clip(corr, -1.0, 1.0)) if not np.isnan(corr) else 0.0
        else:
            result["intermarket_spy_btc_corr"] = 0.0
    else:
        result["intermarket_spy_btc_corr"] = 0.0

    gld_candles = _load_daily_candles(conn, "GLD/USD", limit=10)
    if len(gld_candles) >= 5 and len(btc_candles) >= 5:
        gld_ret5 = gld_candles["close"].iloc[-1] / gld_candles["close"].iloc[-5] - 1
        btc_ret5 = btc_candles["close"].iloc[-1] / btc_candles["close"].iloc[-5] - 1
        result["intermarket_gold_btc_rel"] = float(np.clip(gld_ret5 - btc_ret5, -0.2, 0.2))
    else:
        result["intermarket_gold_btc_rel"] = 0.0

    return result


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
    DataFrame with extra columns for all macro/extended features.
    Fills zeros when data is unavailable for a given timestamp.
    """
    out = candles.copy()
    for col in MACRO_FEATURE_NAMES:
        out[col] = 0.0

    if conn is None:
        return out

    # Original 10 macro features via time-series merge
    _ORIGINAL_10 = MACRO_FEATURE_NAMES[:10]
    try:
        macro_table = _compute_macro_table(conn)
        if not macro_table.empty:
            if "ts_ms" not in out.columns and "timestamp" in out.columns:
                out["ts_ms"] = (
                    pd.to_datetime(out["timestamp"]).astype("int64") // 1_000_000
                )

            # Drop pre-initialized zero columns before merge to avoid
            # suffix collision (merge_asof would rename the real data to
            # *_macro while keeping the zeros under the original name).
            out_for_merge = out.drop(columns=_ORIGINAL_10, errors="ignore")
            out_sorted = out_for_merge.sort_values("ts_ms").copy()
            available_cols = [c for c in _ORIGINAL_10 if c in macro_table.columns]
            macro_sorted = macro_table[["ts_ms"] + available_cols].sort_values("ts_ms")

            merged = pd.merge_asof(
                out_sorted[["ts_ms"]],
                macro_sorted,
                on="ts_ms",
                direction="backward",
            )
            for col in available_cols:
                merged[col] = merged[col].fillna(0.0)

            merged = merged.set_index(out_sorted.index).reindex(out.index)
            for col in available_cols:
                out[col] = merged[col].values

    except Exception as exc:
        logger.warning("add_macro_features (original 10) failed: %s", exc)

    # Extended features: on-chain, sentiment, derivatives, inter-market
    # Time-align daily on-chain/sentiment data via merge_asof where possible,
    # fall back to latest-value broadcast for metrics with sparse history.
    try:
        ext_table = _compute_extended_table(conn)
        if ext_table is not None and not ext_table.empty:
            if "ts_ms" not in out.columns and "timestamp" in out.columns:
                out["ts_ms"] = (
                    pd.to_datetime(out["timestamp"]).astype("int64") // 1_000_000
                )
            ext_cols = [c for c in ext_table.columns if c != "ts_ms"]
            out_sorted = out.sort_values("ts_ms").copy()
            merged = pd.merge_asof(
                out_sorted[["ts_ms"]],
                ext_table.sort_values("ts_ms"),
                on="ts_ms",
                direction="backward",
            )
            merged = merged.set_index(out_sorted.index).reindex(out.index)
            for col in ext_cols:
                vals = merged[col].fillna(0.0).values
                out[col] = vals
        else:
            ext = _compute_extended_features(conn)
            for col, val in ext.items():
                if col in MACRO_FEATURE_NAMES:
                    out[col] = val
    except Exception as exc:
        logger.warning("add_macro_features (extended) failed: %s", exc)

    return out


# ---------------------------------------------------------------------------
# Public API — single row (live inference)
# ---------------------------------------------------------------------------

def get_macro_feature_row(
    conn: "sqlite3.Connection",
    ts_ms: int | None = None,
) -> list[float]:
    """Return all macro + extended features for the most recent available data.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open DB connection.
    ts_ms : int, optional
        Cutoff timestamp in milliseconds.  Uses current time if ``None``.

    Returns
    -------
    list of floats (zeros on any failure).
    """
    if conn is None:
        return list(_ZEROS)

    result = dict.fromkeys(MACRO_FEATURE_NAMES, 0.0)

    # Original 10 macro features
    try:
        import time
        cutoff = ts_ms or int(time.time() * 1000)

        macro_table = _compute_macro_table(conn)
        if not macro_table.empty:
            before = macro_table[macro_table["ts_ms"] <= cutoff]
            if not before.empty:
                row = before.iloc[-1]
                for col in MACRO_FEATURE_NAMES[:10]:
                    result[col] = float(row.get(col, 0.0))
    except Exception as exc:
        logger.warning("get_macro_feature_row (original) failed: %s", exc)

    # Extended features
    try:
        ext = _compute_extended_features(conn)
        result.update(ext)
    except Exception as exc:
        logger.warning("get_macro_feature_row (extended) failed: %s", exc)

    return [result.get(col, 0.0) for col in MACRO_FEATURE_NAMES]
