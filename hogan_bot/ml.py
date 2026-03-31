"""Machine-learning pipeline for Hogan: feature engineering, training, and inference."""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hogan_bot.indicators import (
    cloud_signal,
    compute_atr,
    fvg_features_frame,
    ripster_ema_clouds,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    model: object
    feature_columns: list[str]
    # StandardScaler fitted on training data; None for tree-based models.
    # Use getattr(m, "scaler", None) when reading pickled artifacts that
    # pre-date this field to stay backward-compatible.
    scaler: object | None = field(default=None)


class RegimeModelRouter:
    """Routes ML predictions to regime-specific models.

    Holds a global fallback ``TrainedModel`` and optional per-regime models.
    When ``predict(candles, regime)`` is called, it uses the regime-specific
    model if available, otherwise falls back to the global model.

    Quacks like ``TrainedModel`` for backward compatibility: ``model``,
    ``feature_columns``, and ``scaler`` all delegate to the global model so
    ``predict_up_probability(candles, router)`` works transparently.
    """

    def __init__(
        self,
        global_model: TrainedModel,
        regime_models: dict[str, TrainedModel] | None = None,
    ):
        self._global = global_model
        self._regime_models = regime_models or {}
        self._current_regime: str | None = None

    @property
    def model(self):
        m = self._regime_models.get(self._current_regime) if self._current_regime else None
        return m.model if m else self._global.model

    @property
    def feature_columns(self) -> list[str]:
        m = self._regime_models.get(self._current_regime) if self._current_regime else None
        return m.feature_columns if m else self._global.feature_columns

    @property
    def scaler(self):
        m = self._regime_models.get(self._current_regime) if self._current_regime else None
        return m.scaler if m else self._global.scaler

    def set_regime(self, regime: str | None) -> None:
        """Set the current regime for subsequent predictions."""
        self._current_regime = regime

    @property
    def has_regime_models(self) -> bool:
        return len(self._regime_models) > 0

    @property
    def regime_names(self) -> list[str]:
        return list(self._regime_models.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-smoothed Relative Strength Index, returned on the same index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.clip(lower=1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (ADX, +DI, -DI) using Wilder's smoothing.

    All three are normalised to [0, 1] before being returned.
    """
    # True Range
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    # Raw directional movement
    up_move = high.diff()
    down_move = -(low.diff())  # positive when price falls

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=close.index,
    )

    alpha = 1.0 / window  # Wilder's exponential smoothing factor
    atr14 = tr.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean() / atr14.clip(lower=1e-9)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean() / atr14.clip(lower=1e-9)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-9) * 100.0
    adx = dx.ewm(alpha=alpha, min_periods=window, adjust=False).mean()

    return adx / 100.0, plus_di / 100.0, minus_di / 100.0


def _feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix.

    Returns a copy of *candles* with all feature columns appended.  Every
    column is computed from data available up to and including each bar —
    there is no look-ahead leakage.
    """
    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    open_ = candles["open"].astype(float)
    volume = candles["volume"].astype(float)

    frame = candles.copy()

    # ------------------------------------------------------------------
    # Price momentum
    # ------------------------------------------------------------------
    frame["ret_1"] = close.pct_change(1)
    frame["ret_3"] = close.pct_change(3)
    frame["ret_6"] = close.pct_change(6)
    frame["ret_12"] = close.pct_change(12)

    # ------------------------------------------------------------------
    # Moving-average spread
    # ------------------------------------------------------------------
    frame["ma_fast"] = close.rolling(12).mean()
    frame["ma_slow"] = close.rolling(48).mean()
    frame["ma_spread"] = (frame["ma_fast"] / frame["ma_slow"].clip(lower=1e-9)) - 1.0

    # ------------------------------------------------------------------
    # Realized volatility and oscillator
    # ------------------------------------------------------------------
    ret_1 = frame["ret_1"]
    frame["volatility_20"] = ret_1.rolling(20).std()
    rsi_raw = _rsi(close, window=14)
    frame["rsi_14"] = rsi_raw / 100.0  # normalize to [0, 1]

    # ------------------------------------------------------------------
    # ATR-based volatility
    # ------------------------------------------------------------------
    atr = compute_atr(frame, window=14)
    frame["atr_pct"] = atr / close.clip(lower=1e-9)

    # ------------------------------------------------------------------
    # MACD histogram
    # ------------------------------------------------------------------
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    frame["macd_hist_pct"] = (macd_line - signal_line) / close.clip(lower=1e-9)

    # ------------------------------------------------------------------
    # Bollinger %B — where is price within the bands?
    # ------------------------------------------------------------------
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower).clip(lower=1e-9)
    frame["bb_pct_b"] = (close - bb_lower) / bb_width  # ~[0,1], <0 oversold, >1 overbought

    # ------------------------------------------------------------------
    # Volatility regime — current vol relative to its 50-bar average
    # ------------------------------------------------------------------
    vol_short = ret_1.rolling(10).std()
    vol_long = ret_1.rolling(50).std().clip(lower=1e-9)
    frame["vol_regime"] = vol_short / vol_long  # >1 = elevated, <1 = subdued

    # ------------------------------------------------------------------
    # Candle microstructure
    # ------------------------------------------------------------------
    frame["range_pct"] = (high - low) / close.clip(lower=1e-9)
    frame["candle_body_pct"] = (close - open_).abs() / close.clip(lower=1e-9)

    body_top = np.maximum(open_.values, close.values)
    body_bot = np.minimum(open_.values, close.values)
    frame["upper_wick_pct"] = (
        np.maximum(high.values - body_top, 0.0) / close.clip(lower=1e-9).values
    )
    frame["lower_wick_pct"] = (
        np.maximum(body_bot - low.values, 0.0) / close.clip(lower=1e-9).values
    )

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------
    vol_avg = volume.rolling(20).mean().clip(lower=1e-9)
    frame["vol_ratio"] = volume / vol_avg
    frame["vol_spike"] = (frame["vol_ratio"] > 2.0).astype(int)

    # ------------------------------------------------------------------
    # Ripster EMA cloud features
    # ------------------------------------------------------------------
    enriched = ripster_ema_clouds(frame)
    sig = cloud_signal(enriched)
    frame["cloud_bull"] = (sig == "bullish").astype(int)
    frame["cloud_bear"] = (sig == "bearish").astype(int)
    frame["cloud_width_pct"] = (
        (enriched["ema_fast_short"] - enriched["ema_slow_long"]) / close.clip(lower=1e-9)
    )

    # ------------------------------------------------------------------
    # ICT FVG features (point-in-time, no look-ahead)
    # ------------------------------------------------------------------
    fvg_feats = fvg_features_frame(frame)
    frame["fvg_bull_active"] = fvg_feats["fvg_bull_active"].values
    frame["fvg_bear_active"] = fvg_feats["fvg_bear_active"].values
    frame["in_bull_fvg"] = fvg_feats["in_bull_fvg"].values
    frame["in_bear_fvg"] = fvg_feats["in_bear_fvg"].values

    # ==================================================================
    # NEW INDICATORS (12 additions — total base features: 24 → 36)
    # ==================================================================

    # ------------------------------------------------------------------
    # ADX — trend strength (normalised to [0,1]) and directional lines
    # ------------------------------------------------------------------
    adx_val, plus_di_val, minus_di_val = _adx(high, low, close, window=14)
    frame["adx_14"] = adx_val
    frame["plus_di"] = plus_di_val
    frame["minus_di"] = minus_di_val

    # ------------------------------------------------------------------
    # Stochastic RSI — RSI of RSI, K and D lines [0, 1]
    # ------------------------------------------------------------------
    rsi_min = rsi_raw.rolling(14).min()
    rsi_max = rsi_raw.rolling(14).max()
    stoch_rsi_k = (rsi_raw - rsi_min) / (rsi_max - rsi_min).clip(lower=1e-9)
    frame["stoch_rsi_k"] = stoch_rsi_k
    frame["stoch_rsi_d"] = stoch_rsi_k.rolling(3).mean()

    # ------------------------------------------------------------------
    # OBV z-score — directional volume flow, 20-bar normalised
    # ------------------------------------------------------------------
    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    obv_mean = obv.rolling(20).mean()
    obv_std = obv.rolling(20).std().clip(lower=1e-9)
    frame["obv_norm"] = (obv - obv_mean) / obv_std  # z-score ~[-3, 3]

    # ------------------------------------------------------------------
    # VWAP distance — % price above/below session VWAP (resets 00:00 UTC)
    # ------------------------------------------------------------------
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume

    # Determine day grouping from available timestamp columns (UTC calendar day)
    if "ts_ms" in candles.columns:
        day_key = pd.to_datetime(candles["ts_ms"], unit="ms", utc=True).dt.date
    elif "timestamp" in candles.columns:
        day_key = pd.to_datetime(candles["timestamp"], utc=True).dt.date
    else:
        from hogan_bot.timeframe_utils import bars_per_day, infer_timeframe_from_candles
        tf = infer_timeframe_from_candles(candles) or "1h"
        day_key = pd.Series(np.arange(len(close)) // bars_per_day(tf), index=close.index)

    cum_pv = pv.groupby(day_key).cumsum()
    cum_vol = volume.groupby(day_key).cumsum().clip(lower=1e-9)
    vwap = cum_pv / cum_vol
    frame["vwap_dist"] = (close - vwap) / vwap.clip(lower=1e-9)

    # ------------------------------------------------------------------
    # Keltner channel position — ATR-based band %B [0, 1] normally
    # ------------------------------------------------------------------
    kc_mid = close.ewm(span=20, adjust=False).mean()
    kc_upper = kc_mid + 2.0 * atr
    kc_lower = kc_mid - 2.0 * atr
    kc_width = (kc_upper - kc_lower).clip(lower=1e-9)
    frame["keltner_pos"] = (close - kc_lower) / kc_width

    # ------------------------------------------------------------------
    # CCI (20) — commodity channel index, clipped to ±2
    # ------------------------------------------------------------------
    tp_mean = typical_price.rolling(20).mean()
    tp_mad = typical_price.rolling(20).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    ).clip(lower=1e-9)
    frame["cci_20"] = ((typical_price - tp_mean) / (0.015 * tp_mad)).clip(-2.0, 2.0)

    # ------------------------------------------------------------------
    # MFI (14) — Money Flow Index, volume-weighted RSI, [0, 1]
    # ------------------------------------------------------------------
    tp_diff = typical_price.diff()
    pos_mf = (typical_price * volume).where(tp_diff > 0, 0.0)
    neg_mf = (typical_price * volume).where(tp_diff < 0, 0.0)
    pos_sum = pos_mf.rolling(14).sum()
    neg_sum = neg_mf.rolling(14).sum().clip(lower=1e-9)
    mfr = pos_sum / neg_sum
    frame["mfi_14"] = (100.0 - 100.0 / (1.0 + mfr)) / 100.0

    # ------------------------------------------------------------------
    # CMF (20) — Chaikin Money Flow [-1, 1]
    # ------------------------------------------------------------------
    hl_range = (high - low).clip(lower=1e-9)
    clv = ((close - low) - (high - close)) / hl_range  # Close Location Value
    frame["cmf_20"] = (clv * volume).rolling(20).sum() / volume.rolling(20).sum().clip(lower=1e-9)

    # ------------------------------------------------------------------
    # ROC (10) — Rate of Change, 10-bar price momentum
    # ------------------------------------------------------------------
    frame["roc_10"] = (close / close.shift(10).clip(lower=1e-9)) - 1.0

    # ==================================================================
    # ICT (Inner Circle Trading) structural features — 7 additions
    # All computed from the price series alone; no look-ahead.
    # ==================================================================

    # 1. Dealing-range position (Premium / Discount):
    #    Where is price within the rolling 20-bar high/low range?
    #    0 = at range low (discount), 1 = at range high (premium), 0.5 = equilibrium
    rolling_high_20 = high.rolling(20).max()
    rolling_low_20 = low.rolling(20).min()
    range_width = (rolling_high_20 - rolling_low_20).clip(lower=1e-9)
    frame["ict_pd_pct"] = (close - rolling_low_20) / range_width

    # 2 & 3. Binary premium / discount flags
    frame["ict_in_discount"] = (frame["ict_pd_pct"] < 0.5).astype(float)
    frame["ict_in_premium"]  = (frame["ict_pd_pct"] > 0.5).astype(float)

    # 4 & 5. Proximity to recent liquidity pools (swing high / low):
    #    % distance *below* the nearest swing high (buy-side liquidity above price)
    #    % distance *above* the nearest swing low  (sell-side liquidity below price)
    rolling_high_14 = high.rolling(14).max()
    rolling_low_14  = low.rolling(14).min()
    frame["ict_swing_high_dist"] = (rolling_high_14 - close) / close.clip(lower=1e-9)
    frame["ict_swing_low_dist"]  = (close - rolling_low_14) / close.clip(lower=1e-9)

    # 6 & 7. Previous-day high / low reference levels (PDH / PDL):
    #    Key ICT levels where buy-side and sell-side liquidity rest overnight.
    #    Reuses day_key from VWAP section above.

    # Per-day agg; map each bar to previous UTC calendar day's high/low
    day_high = high.groupby(day_key).max()
    day_low = low.groupby(day_key).min()
    prev_day_high_map = day_high.shift(1)
    prev_day_low_map = day_low.shift(1)
    prev_day_high = day_key.map(prev_day_high_map)
    prev_day_low = day_key.map(prev_day_low_map)
    # % distance below PDH (positive = price is below PDH, the "safe" zone)
    frame["ict_pdh_dist"] = (prev_day_high - close) / close.clip(lower=1e-9)
    # % distance above PDL (positive = price is above PDL, the "safe" zone)
    frame["ict_pdl_dist"] = (close - prev_day_low) / close.clip(lower=1e-9)

    return frame


# Single source of truth: feature_registry owns the canonical 59-column list.
from hogan_bot.feature_registry import (  # noqa: E402
    _FULL_FEATURE_COLUMNS as _FEATURE_COLUMNS,
)

# EXPERIMENTAL: ICT structural features — quarantined from champion path.
# Still computed in _feature_frame() but not included in default training/inference.
# Use _ALL_FEATURE_COLUMNS when explicitly opting in to ICT features.
_EXPERIMENTAL_FEATURES: list[str] = [
    "ict_pd_pct",
    "ict_in_discount",
    "ict_in_premium",
    "ict_swing_high_dist",
    "ict_swing_low_dist",
    "ict_pdh_dist",
    "ict_pdl_dist",
]

_ALL_FEATURE_COLUMNS: list[str] = _FEATURE_COLUMNS + _EXPERIMENTAL_FEATURES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_feature_row_extended(
    candles_5m: pd.DataFrame,
    candles_1h: pd.DataFrame | None = None,
    candles_15m: pd.DataFrame | None = None,
    candles_10m: pd.DataFrame | None = None,
    candles_30m: pd.DataFrame | None = None,
    candles_3h: pd.DataFrame | None = None,
    conn=None,
    symbol: str = "BTC/USD",
    extended_mtf: bool = False,
) -> list[float] | None:
    """Extended feature vector — delegates to
    :func:`hogan_bot.features_mtf.build_feature_row_extended`.

    When ``extended_mtf=False`` (default): same size as before (backward-compat).
    When ``extended_mtf=True``: adds 10m + 30m features (+14 total).
    """
    from hogan_bot.features_mtf import build_feature_row_extended as _ext
    return _ext(
        candles_5m,
        candles_1h=candles_1h,
        candles_15m=candles_15m,
        candles_10m=candles_10m,
        candles_30m=candles_30m,
        candles_3h=candles_3h,
        conn=conn,
        symbol=symbol,
        extended_mtf=extended_mtf,
    )


def build_feature_row(
    candles: pd.DataFrame,
    db_conn=None,
    use_champion_features: bool | None = None,
) -> list[float] | None:
    """Return the feature vector for the **last bar** in *candles*.

    When *db_conn* is provided, the 10 macro-asset features are populated
    from the DB (SPY/VIX/GLD etc.); otherwise they default to 0.

    When *use_champion_features* is True (or HOGAN_CHAMPION_MODE is set),
    returns only the 8-feature champion subset. Otherwise returns the full 59.

    Returns ``None`` when there is insufficient history (< 60 bars).
    """
    if len(candles) < 60:
        return None
    try:
        from hogan_bot.feature_registry import get_feature_columns
        from hogan_bot.macro_features import MACRO_FEATURE_NAMES

        frame = _feature_frame(candles)

        if db_conn is not None:
            from hogan_bot.macro_features import get_macro_feature_row
            ts_ms = None
            if "ts_ms" in frame.columns:
                ts_ms = int(frame["ts_ms"].iloc[-1])
            elif "timestamp" in frame.columns:
                import pandas as _pd
                ts_ms = int(_pd.Timestamp(frame["timestamp"].iloc[-1]).timestamp() * 1000)
            macro_vals = get_macro_feature_row(db_conn, ts_ms=ts_ms)
            for col, val in zip(MACRO_FEATURE_NAMES, macro_vals):
                frame[col] = val
        else:
            for col in MACRO_FEATURE_NAMES:
                frame[col] = 0.0

        cols = get_feature_columns(use_champion_features)
        last = frame[cols].iloc[-1]
        if last.isna().any():
            return None
        return [float(v) for v in last.values]
    except Exception as exc:
        logger.warning("build_feature_row failed (ML will be skipped this bar): %s", exc)
        return None


def build_feature_row_checked(
    candles: pd.DataFrame,
    db_conn=None,
    data_ages_hours: dict[str, float] | None = None,
    use_champion_features: bool | None = None,
):
    """Like :func:`build_feature_row` but returns a :class:`FeatureResult`
    with staleness metadata.

    Parameters
    ----------
    candles : pd.DataFrame
        OHLCV candle data.
    db_conn : optional
        SQLite connection for macro feature lookup.
    data_ages_hours : dict[str, float] | None
        Mapping of source name (e.g. ``"macro_db"``) to hours since last
        update.  Used to flag stale features.
    use_champion_features : bool | None
        When True, use champion feature subset. When None, follows
        HOGAN_CHAMPION_MODE.

    Returns
    -------
    FeatureResult | None
        ``None`` when insufficient data, otherwise a FeatureResult with
        the feature vector and staleness info.
    """
    from hogan_bot.feature_registry import (
        check_staleness,
        get_feature_columns,
    )

    values = build_feature_row(candles, db_conn=db_conn, use_champion_features=use_champion_features)
    if values is None:
        return None

    cols = get_feature_columns(use_champion_features)
    return check_staleness(cols, values, data_ages_hours)


def build_training_set(
    candles: pd.DataFrame,
    horizon_bars: int = 12,
    db_conn=None,
    fee_rate: float = 0.0026,
    label_mode: str = "fee_threshold",
    use_champion_features: bool | None = None,
) -> tuple[pd.DataFrame | None, pd.Series | None, list[str], pd.Series | None]:
    """Construct feature matrix *X* and label vector *y* from *candles*.

    Parameters
    ----------
    label_mode : str
        ``"fee_threshold"`` (default) — labels 1 when future return exceeds
        ``2 * fee_rate``, 0 when below ``-2 * fee_rate``, drops the ambiguous
        dead zone.

        ``"triple_barrier"`` — uses path-aware triple-barrier labels from
        ``ml_advanced.triple_barrier_labels()``.  Accounts for time decay
        and adverse excursion, better for trend-following strategies.

    When *db_conn* is provided, joins in 10 macro-asset features
    (SPY/QQQ/GLD/TLT/UUP/VIX/TNX) via :func:`~hogan_bot.macro_features.add_macro_features`.
    These are aligned by timestamp with no look-ahead leakage.

    Returns ``(None, None, feature_columns)`` when the dataset is empty after
    dropping NaN rows.
    """
    frame = _feature_frame(candles)

    if db_conn is not None:
        try:
            from hogan_bot.macro_features import MACRO_FEATURE_NAMES, add_macro_features
            frame = add_macro_features(frame, db_conn)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Macro feature join failed: %s — training without macro context", exc
            )
            for col in MACRO_FEATURE_NAMES:
                if col not in frame.columns:
                    frame[col] = 0.0
    else:
        from hogan_bot.macro_features import MACRO_FEATURE_NAMES
        for col in MACRO_FEATURE_NAMES:
            frame[col] = 0.0

    if label_mode == "enhanced_triple_barrier":
        from hogan_bot.ml_advanced import triple_barrier_labels_enhanced
        tb_result = triple_barrier_labels_enhanced(
            frame, horizon=horizon_bars, vol_span=100,
            k_up=2.0, k_dn=2.0, time_decay=True,
            regime_adaptive=True, fee_rate=fee_rate,
        )
        frame["target"] = tb_result["label"].values
        frame["meta_quality"] = tb_result["meta_quality"].values
        frame.loc[frame["target"].isna(), "target"] = np.nan
    elif label_mode == "triple_barrier":
        from hogan_bot.ml_advanced import triple_barrier_labels
        frame["target"] = triple_barrier_labels(
            frame["close"], horizon=horizon_bars, vol_span=100, k=2.0,
        ).values
        frame["target"] = frame["target"].astype(float)
        frame.loc[frame["target"].isna(), "target"] = np.nan
    else:
        future_ret = frame["close"].shift(-horizon_bars) / frame["close"] - 1.0
        min_move = 2.0 * fee_rate
        frame["target"] = np.where(
            future_ret > min_move, 1,
            np.where(future_ret < -min_move, 0, np.nan),
        )

    from hogan_bot.feature_registry import get_feature_columns
    feature_cols = get_feature_columns(use_champion_features)

    keep_cols = feature_cols + ["target"]
    has_quality = "meta_quality" in frame.columns
    if has_quality:
        keep_cols = keep_cols + ["meta_quality"]

    dataset = frame[keep_cols].dropna(subset=feature_cols + ["target"]).copy()
    dataset["target"] = dataset["target"].astype(int)
    if dataset.empty:
        return None, None, feature_cols, None

    quality = dataset["meta_quality"].values if has_quality else None
    return dataset[feature_cols], dataset["target"], feature_cols, quality


def make_paper_trade_labels(
    db_path: str,
    candles: pd.DataFrame,
    symbol: str,
    min_trades: int = 5,
    use_champion_features: bool | None = None,
) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """Extract feature-label pairs from closed paper trades stored in the DB.

    For each closed paper trade, the candle at the entry timestamp is located
    and features are extracted using *_feature_frame*.  Labels are assigned
    based on actual trade outcome (direction correctness):

    * ``long``  + ``realized_pnl > 0``  → label = 1  (price went up, correct)
    * ``long``  + ``realized_pnl ≤ 0``  → label = 0
    * ``short`` + ``realized_pnl > 0``  → label = 0  (price fell, correct for short)
    * ``short`` + ``realized_pnl ≤ 0``  → label = 1

    Returns ``(X_extra, y_extra)`` aligned with the active feature columns, or
    ``(None, None)`` when fewer than *min_trades* are available.
    """
    import sqlite3 as _sqlite3

    try:
        conn = _sqlite3.connect(db_path)
        try:
            import pandas as _pd
            trades_df = _pd.read_sql_query(
                """SELECT trade_id, symbol, side, realized_pnl, open_ts_ms
                   FROM paper_trades
                   WHERE exit_price IS NOT NULL AND symbol = ?
                   ORDER BY open_ts_ms""",
                conn,
                params=(symbol,),
            )
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("make_paper_trade_labels failed for %s: %s", symbol, exc)
        return None, None

    if len(trades_df) < min_trades:
        return None, None

    from hogan_bot.feature_registry import get_feature_columns
    _cols = get_feature_columns(use_champion_features)

    frame = _feature_frame(candles)
    candle_ts = candles["ts_ms"].values

    X_rows: list[list[float]] = []
    y_values: list[int] = []

    for _, trade in trades_df.iterrows():
        ts = int(trade["open_ts_ms"])
        idx = int(np.searchsorted(candle_ts, ts, side="right")) - 1
        if idx < 1 or idx >= len(frame):
            continue
        row = frame[_cols].iloc[idx]
        if row.isna().any():
            continue

        pnl = float(trade["realized_pnl"])
        raw_side = str(trade["side"]).lower().strip()
        side = "long" if raw_side in ("buy", "long") else "short"
        if side == "long":
            label = 1 if pnl > 0 else 0
        else:
            label = 0 if pnl > 0 else 1

        X_rows.append(row.tolist())
        y_values.append(label)

    if len(X_rows) < min_trades:
        return None, None

    X_extra = pd.DataFrame(X_rows, columns=_cols)
    y_extra = pd.Series(y_values, name="target", dtype=int)
    return X_extra, y_extra


def make_backtest_labels(
    result: object,
    candles: pd.DataFrame,
    symbol: str,
    min_trades: int = 5,
    db_conn=None,
    use_champion_features: bool | None = None,
) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """Extract feature-label pairs from backtest closed trades.

    For each closed trade in *result.closed_trades*, features are taken from
    the entry bar. Labels follow the same convention as paper trades:
    long + pnl > 0 → 1, long + pnl ≤ 0 → 0.

    When *db_conn* is provided, macro features are joined; otherwise zeros.

    Returns ``(X_extra, y_extra)`` aligned with the active feature columns, or
    ``(None, None)`` when fewer than *min_trades* are available.
    """
    closed = getattr(result, "closed_trades", None)
    if not closed or len(closed) < min_trades:
        return None, None

    from hogan_bot.feature_registry import get_feature_columns
    _cols = get_feature_columns(use_champion_features)

    frame = _feature_frame(candles)
    if db_conn is not None:
        try:
            from hogan_bot.macro_features import MACRO_FEATURE_NAMES, add_macro_features
            frame = add_macro_features(frame, db_conn)
        except Exception as exc:
            logger.warning("make_backtest_labels: macro feature join failed — using zeros: %s", exc)
            from hogan_bot.macro_features import MACRO_FEATURE_NAMES
            for col in MACRO_FEATURE_NAMES:
                if col not in frame.columns:
                    frame[col] = 0.0
    else:
        from hogan_bot.macro_features import MACRO_FEATURE_NAMES
        for col in MACRO_FEATURE_NAMES:
            frame[col] = 0.0
    X_rows: list[list[float]] = []
    y_values: list[int] = []

    for trade in closed:
        idx = int(trade.get("entry_bar_idx", -1))
        if idx < 1 or idx >= len(frame):
            continue
        row = frame[_cols].iloc[idx]
        if row.isna().any():
            continue

        pnl = float(trade.get("pnl_usd", 0.0))
        raw_side = str(trade.get("side", "long")).lower().strip()
        side = "long" if raw_side in ("buy", "long") else "short"
        if side == "long":
            label = 1 if pnl > 0 else 0
        else:
            label = 0 if pnl > 0 else 1

        X_rows.append(row.tolist())
        y_values.append(label)

    if len(X_rows) < min_trades:
        return None, None

    X_extra = pd.DataFrame(X_rows, columns=_cols)
    y_extra = pd.Series(y_values, name="target", dtype=int)
    return X_extra, y_extra


def _blend_paper_labels(
    x: pd.DataFrame,
    y: pd.Series,
    paper_labels: tuple[pd.DataFrame, pd.Series] | None,
    weight: float = 3.0,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray | None]:
    """Concatenate paper-trade labeled rows into (x, y) and build sample weights.

    Paper-trade rows are appended *after* the canonical rows so they fall into
    the training split (rows 0..80%).  Returns ``(x_merged, y_merged, weights)``
    where ``weights`` is ``None`` when no paper labels are provided.
    """
    if paper_labels is None:
        return x, y, None
    X_extra, y_extra = paper_labels
    if X_extra is None or len(X_extra) == 0:
        return x, y, None

    # Put extra rows at the *start* so they land in the 80% train split
    x_merged = pd.concat([X_extra, x], ignore_index=True)
    y_merged = pd.concat([y_extra, y], ignore_index=True)
    weights = np.concatenate([
        np.full(len(X_extra), weight),
        np.ones(len(x)),
    ])
    return x_merged, y_merged, weights


def select_features(
    x: pd.DataFrame,
    y: pd.Series,
    max_features: int = 25,
    min_importance: float = 0.005,
) -> list[str]:
    """Return the top features ranked by tree-based importance.

    Uses a quick Random Forest fit to rank features, then keeps up to
    *max_features* that exceed *min_importance*.  Also drops features
    with >50% NaN rate or near-zero variance.

    IMPORTANT: Callers must pass only training data to avoid data leakage.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return list(x.columns)

    keep_cols = []
    for col in x.columns:
        nan_rate = x[col].isna().mean()
        if nan_rate > 0.5:
            continue
        if x[col].std() < 1e-8:
            continue
        keep_cols.append(col)

    if len(keep_cols) <= max_features:
        return keep_cols

    x_clean = x[keep_cols].fillna(0.0)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    rf.fit(x_clean, y)

    importances = sorted(
        zip(keep_cols, rf.feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    )
    selected = [col for col, imp in importances[:max_features] if imp >= min_importance]
    return selected if selected else keep_cols[:max_features]


def train_logistic_regression(
    candles: pd.DataFrame,
    model_path: str,
    horizon_bars: int = 12,
    tune_hyperparams: bool = False,
    paper_labels: tuple[pd.DataFrame, pd.Series] | None = None,
    paper_labels_weight: float = 3.0,
    db_conn=None,
    prune_features: bool = True,
    max_features: int = 25,
    label_mode: str = "fee_threshold",
) -> dict[str, object]:
    """Fit a scaled logistic-regression classifier and pickle the artifact.

    When *tune_hyperparams* is ``True`` a small C grid search
    (0.01 / 0.1 / 1.0 / 10.0) is performed and the best model is saved.
    Returns an extended metrics dict including ``roc_auc``, ``precision``,
    ``recall``, ``f1``, and (if tuning) ``best_C``.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for training") from exc

    x, y, feature_columns, _mq = build_training_set(candles, horizon_bars=horizon_bars, db_conn=db_conn, label_mode=label_mode)
    if x is None or y is None or len(x) < 200:
        raise RuntimeError("Not enough training rows. Increase OHLCV history.")

    x, y, sample_weight = _blend_paper_labels(x, y, paper_labels, paper_labels_weight)

    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    w_train = sample_weight[:split] if sample_weight is not None else None

    from hogan_bot.champion import is_champion_mode as _is_champ
    if prune_features and len(x_train) >= 500 and not _is_champ():
        selected = select_features(x_train, y_train, max_features=max_features)
        x_train = x_train[selected]
        x_test = x_test[selected]
        feature_columns = selected

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

    if tune_hyperparams:
        best_C, best_model, best_acc = 1.0, None, -1.0
        for C in (0.01, 0.1, 1.0, 10.0):
            candidate = LogisticRegression(max_iter=500, C=C, class_weight="balanced")
            candidate.fit(x_train_sc, y_train, sample_weight=w_train)
            acc = float(accuracy_score(y_test, candidate.predict(x_test_sc)))
            if acc > best_acc:
                best_acc, best_C, best_model = acc, C, candidate
        model = best_model
    else:
        best_C = 1.0
        model = LogisticRegression(max_iter=500, C=best_C, class_weight="balanced")
        model.fit(x_train_sc, y_train, sample_weight=w_train)

    pred = model.predict(x_test_sc)
    proba = model.predict_proba(x_test_sc)[:, 1]

    auc = float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.5
    metrics: dict[str, object] = {
        "model_type": "logistic_regression",
        "best_C": best_C,
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": auc,
        "brier": float(brier_score_loss(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "features": len(feature_columns),
    }

    artifact = TrainedModel(model=model, feature_columns=feature_columns, scaler=scaler)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    return metrics


def train_random_forest(
    candles: pd.DataFrame,
    model_path: str,
    horizon_bars: int = 12,
    paper_labels: tuple[pd.DataFrame, pd.Series] | None = None,
    paper_labels_weight: float = 3.0,
    db_conn=None,
    prune_features: bool = True,
    max_features: int = 25,
    label_mode: str = "fee_threshold",
) -> dict[str, object]:
    """Fit a Random Forest classifier and pickle the artifact.

    Feature importances are included in the returned metrics under
    ``feature_importances``.  RF is invariant to feature scale so no
    ``StandardScaler`` is applied (``TrainedModel.scaler`` is ``None``).
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for training") from exc

    x, y, feature_columns, _mq = build_training_set(candles, horizon_bars=horizon_bars, db_conn=db_conn, label_mode=label_mode)
    if x is None or y is None or len(x) < 200:
        raise RuntimeError("Not enough training rows. Increase OHLCV history.")

    x, y, sample_weight = _blend_paper_labels(x, y, paper_labels, paper_labels_weight)

    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    w_train = sample_weight[:split] if sample_weight is not None else None

    from hogan_bot.champion import is_champion_mode as _is_champ
    if prune_features and len(x_train) >= 500 and not _is_champ():
        selected = select_features(x_train, y_train, max_features=max_features)
        x_train = x_train[selected]
        x_test = x_test[selected]
        feature_columns = selected

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]

    auc = float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.5
    importances = {
        col: float(imp) for col, imp in zip(feature_columns, model.feature_importances_)
    }
    metrics: dict[str, object] = {
        "model_type": "random_forest",
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": auc,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "features": len(feature_columns),
        "feature_importances": importances,
    }

    artifact = TrainedModel(model=model, feature_columns=feature_columns, scaler=None)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    return metrics


def train_xgboost(
    candles: pd.DataFrame,
    model_path: str,
    horizon_bars: int = 12,
    paper_labels: tuple[pd.DataFrame, pd.Series] | None = None,
    paper_labels_weight: float = 3.0,
    db_conn=None,
    prune_features: bool = True,
    max_features: int = 25,
    label_mode: str = "fee_threshold",
) -> dict[str, object]:
    """Fit an XGBoost gradient-boosted classifier and pickle the artifact.

    Requires the ``xgboost`` package (``pip install xgboost``).
    Returns metrics including ``feature_importances`` (gain-based).
    No feature scaling is applied — XGBoost is tree-based.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is not installed.  Run: pip install xgboost"
        ) from exc
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for metrics") from exc

    x, y, feature_columns, _mq = build_training_set(candles, horizon_bars=horizon_bars, db_conn=db_conn, label_mode=label_mode)
    if x is None or y is None or len(x) < 200:
        raise RuntimeError("Not enough training rows. Increase OHLCV history.")

    x, y, sample_weight = _blend_paper_labels(x, y, paper_labels, paper_labels_weight)

    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    w_train = sample_weight[:split] if sample_weight is not None else None

    from hogan_bot.champion import is_champion_mode as _is_champ
    if prune_features and len(x_train) >= 500 and not _is_champ():
        selected = select_features(x_train, y_train, max_features=max_features)
        x_train = x_train[selected]
        x_test = x_test[selected]
        feature_columns = selected

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]

    auc = float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.5
    raw_imp = model.get_booster().get_score(importance_type="gain")
    importances = {col: float(raw_imp.get(col, 0.0)) for col in feature_columns}
    metrics: dict[str, object] = {
        "model_type": "xgboost",
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": auc,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "features": len(feature_columns),
        "feature_importances": importances,
    }

    artifact = TrainedModel(model=model, feature_columns=feature_columns, scaler=None)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    return metrics


def train_lightgbm(
    candles: pd.DataFrame, model_path: str, horizon_bars: int = 12, db_conn=None,
    label_mode: str = "fee_threshold",
) -> dict[str, object]:
    """Fit a LightGBM gradient-boosted classifier and pickle the artifact.

    Requires the ``lightgbm`` package (``pip install lightgbm``).
    Returns metrics including ``feature_importances`` (split-based).
    No feature scaling is applied — LightGBM is tree-based.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm is not installed.  Run: pip install lightgbm"
        ) from exc
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for metrics") from exc

    x, y, feature_columns, _mq = build_training_set(candles, horizon_bars=horizon_bars, db_conn=db_conn, label_mode=label_mode)
    if x is None or y is None or len(x) < 200:
        raise RuntimeError("Not enough training rows. Increase OHLCV history.")

    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]

    auc = float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.5
    importances = {
        col: float(imp) for col, imp in zip(feature_columns, model.feature_importances_)
    }
    metrics: dict[str, object] = {
        "model_type": "lightgbm",
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": auc,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "features": len(feature_columns),
        "feature_importances": importances,
    }

    artifact = TrainedModel(model=model, feature_columns=feature_columns, scaler=None)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    return metrics


def calibrate_model(
    candles: pd.DataFrame,
    model_path: str,
    horizon_bars: int = 12,
    method: str = "sigmoid",
    calibration_fraction: float = 0.2,
    use_champion_features: bool | None = None,
    db_conn=None,
    label_mode: str = "fee_threshold",
) -> dict[str, object]:
    """Wrap an existing pickled model with probability calibration.

    Loads the artifact at *model_path*, uses the last *calibration_fraction*
    of the labelled dataset as a holdout calibration set (``cv="prefit"``),
    fits a ``CalibratedClassifierCV``, and overwrites the pickle.

    Returns a summary dict so the result can be logged to the model registry.

    Parameters
    ----------
    method:
        ``"sigmoid"`` (Platt scaling) or ``"isotonic"``.
    calibration_fraction:
        Fraction of labelled rows reserved for calibration (default 20 %).
    use_champion_features:
        When True, build only the champion 8-feature subset. When None,
        follows HOGAN_CHAMPION_MODE environment variable.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for calibration") from exc

    artifact = load_model(model_path)
    x, y, feature_columns, _mq = build_training_set(
        candles, horizon_bars=horizon_bars, use_champion_features=use_champion_features,
        db_conn=db_conn, label_mode=label_mode,
    )
    if x is None or y is None or len(x) < 100:
        raise RuntimeError("Not enough data for calibration. Increase OHLCV history.")

    cal_start = int(len(x) * (1.0 - calibration_fraction))
    x_cal = x.iloc[cal_start:]
    y_cal = y.iloc[cal_start:]

    scaler = getattr(artifact, "scaler", None)
    if scaler is not None:
        x_cal_arr = scaler.transform(x_cal)
    else:
        x_cal_arr = x_cal

    try:
        from sklearn.frozen import FrozenEstimator  # type: ignore[import-untyped]
        calibrated = CalibratedClassifierCV(FrozenEstimator(artifact.model), method=method)
    except (ImportError, ModuleNotFoundError):
        calibrated = CalibratedClassifierCV(artifact.model, cv="prefit", method=method)
    calibrated.fit(x_cal_arr, y_cal)

    new_artifact = TrainedModel(
        model=calibrated,
        feature_columns=feature_columns,
        scaler=scaler,
    )
    with open(model_path, "wb") as f:
        pickle.dump(new_artifact, f)

    return {
        "model_type": f"calibrated_{method}",
        "calibration_rows": len(x_cal),
        "total_rows": len(x),
        "features": len(feature_columns),
    }


def walk_forward_cv(
    candles: pd.DataFrame,
    horizon_bars: int = 12,
    n_splits: int = 5,
    model_type: str = "logreg",
    db_conn=None,
    fee_rate: float = 0.0026,
    label_mode: str = "enhanced_triple_barrier",
) -> list[dict[str, object]]:
    """Purged walk-forward time-series cross-validation.

    Supports ``model_type`` in ``{"logreg", "random_forest", "xgboost",
    "lightgbm", "histgb"}``.

    An embargo gap of ``horizon_bars`` rows is inserted between each
    train/test boundary to prevent information leakage from overlapping
    prediction horizons.

    Returns a list of per-fold dicts with ``fold``, ``train_rows``,
    ``test_rows``, ``accuracy``, ``roc_auc``.
    """
    try:
        from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for walk-forward CV") from exc

    x, y, _, _mq = build_training_set(
        candles, horizon_bars=horizon_bars, db_conn=db_conn,
        fee_rate=fee_rate, label_mode=label_mode,
    )
    if x is None or y is None or len(x) < 200:
        raise RuntimeError(
            "Not enough labelled rows for walk-forward CV. "
            "Increase OHLCV history or reduce n_splits."
        )

    n = len(x)
    min_train = max(200, n // (n_splits + 1))
    fold_size = (n - min_train) // n_splits
    if fold_size < 20:
        raise RuntimeError(
            "Not enough data for walk-forward CV with these settings. "
            "Increase OHLCV history or reduce n_splits."
        )

    embargo = horizon_bars

    results: list[dict[str, object]] = []
    for i in range(n_splits):
        train_end = min_train + fold_size * i
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)

        if test_start >= n or test_end - test_start < 10:
            continue

        x_train = x.iloc[:train_end]
        y_train = y.iloc[:train_end]
        x_test = x.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        if len(x_test) < 10:
            continue

        use_scaler = model_type == "logreg"
        scaler = None
        if use_scaler:
            scaler = StandardScaler()
            x_train_arr = scaler.fit_transform(x_train)
            x_test_arr = scaler.transform(x_test)
        else:
            x_train_arr = x_train
            x_test_arr = x_test

        model = _make_cv_model(model_type)
        model.fit(x_train_arr, y_train)
        pred = model.predict(x_test_arr)
        proba = model.predict_proba(x_test_arr)[:, 1]

        auc = float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.5
        results.append(
            {
                "fold": i + 1,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "accuracy": float(accuracy_score(y_test, pred)),
                "roc_auc": auc,
                "brier": float(brier_score_loss(y_test, proba)),
            }
        )

    return results


def _make_cv_model(model_type: str):
    """Instantiate a classifier for cross-validation."""
    if model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            class_weight="balanced_subsample", random_state=42,
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            use_label_encoder=False, random_state=42,
        )
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=42, verbose=-1,
        )
    elif model_type in ("histgb", "hist_gb"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=400, random_state=42,
        )
    else:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")


def load_model(model_path: str) -> TrainedModel:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, TrainedModel):
        raise RuntimeError("Invalid model artifact format")
    return model


def predict_up_probability(
    candles: pd.DataFrame,
    trained_model: TrainedModel,
    db_conn=None,
) -> float:
    """Return P(next move is up) for the most recent bar in *candles*.

    When *db_conn* is provided and the model was trained with macro features,
    the macro context (VIX, SPY, GLD, etc.) is fetched from the candles table
    so inference is consistent with training.  If unavailable, macro slots are
    filled with zeros (graceful degradation for models without macro features).
    """
    frame = _feature_frame(candles)

    # Inject macro features when the model expects them
    feature_cols = trained_model.feature_columns
    from hogan_bot.macro_features import MACRO_FEATURE_NAMES
    macro_cols_needed = [c for c in feature_cols if c in MACRO_FEATURE_NAMES]
    if macro_cols_needed:
        if db_conn is not None:
            try:
                from hogan_bot.macro_features import get_macro_feature_row
                ts_ms = None
                if "ts_ms" in frame.columns:
                    ts_ms = int(frame["ts_ms"].iloc[-1])
                elif "timestamp" in frame.columns:
                    ts_ms = int(
                        pd.Timestamp(frame["timestamp"].iloc[-1]).timestamp() * 1000
                    )
                macro_vals = get_macro_feature_row(db_conn, ts_ms=ts_ms)
                for col, val in zip(MACRO_FEATURE_NAMES, macro_vals):
                    frame[col] = val
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "predict_up_probability: macro fetch failed (%s) — using zeros", exc
                )
                for col in macro_cols_needed:
                    frame[col] = 0.0
        else:
            for col in macro_cols_needed:
                frame[col] = 0.0

    latest = frame[feature_cols].tail(1)
    if latest.empty or latest.isna().any(axis=1).iloc[0]:
        return 0.5
    scaler = getattr(trained_model, "scaler", None)
    if scaler is not None:
        latest_arr = scaler.transform(latest)
        proba = trained_model.model.predict_proba(latest_arr)[0][1]
    else:
        proba = trained_model.model.predict_proba(latest)[0][1]
    return float(min(max(proba, 0.0), 1.0))



def train_hist_gradient_boosting(
    candles: pd.DataFrame,
    model_path: str,
    horizon_bars: int = 12,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    max_iter: int = 400,
    db_conn=None,
    label_mode: str = "fee_threshold",
) -> dict[str, object]:
    """Train + save sklearn HistGradientBoostingClassifier (strong tabular baseline)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    X, y, feature_cols, meta_quality = build_training_set(
        candles, horizon_bars=horizon_bars, db_conn=db_conn, label_mode=label_mode,
    )
    if X is None or y is None or len(X) < 50:
        raise ValueError(f"Not enough training rows ({len(X) if X is not None else 0})")
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Combine class-balance weights with meta_quality from enhanced labeling
    class_counts = y_train.value_counts()
    n_samples = len(y_train)
    n_classes = len(class_counts)
    class_weight = y_train.map(
        {c: n_samples / (n_classes * cnt) for c, cnt in class_counts.items()}
    ).values
    if meta_quality is not None:
        mq_train = meta_quality[:split]
        mq_train = np.clip(mq_train, 0.1, 1.0)
        sample_weight = class_weight * mq_train
    else:
        sample_weight = class_weight

    clf = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "model_type": "hist_gb",
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.0,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    artifact = TrainedModel(model=clf, feature_columns=feature_cols, scaler=None)
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    return metrics


def build_feature_frame(candles: pd.DataFrame, db_conn=None) -> pd.DataFrame:
    """Return a feature DataFrame aligned to candles index for advanced models.

    When *db_conn* is provided, macro features are joined from the database.
    Otherwise macro/onchain/sentiment/derivatives/intermarket columns are
    filled with zeros so the output always matches ``_FEATURE_COLUMNS`` (59).
    """
    frame = _feature_frame(candles)
    if db_conn is not None:
        try:
            from hogan_bot.macro_features import add_macro_features
            frame = add_macro_features(frame, db_conn)
        except Exception as exc:
            logger.warning("build_feature_frame: macro feature join failed — using zeros: %s", exc)
    for col in _FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0
    out = frame[list(_FEATURE_COLUMNS)].copy()
    return out.replace([np.inf, -np.inf], np.nan)
