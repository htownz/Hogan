"""Machine-learning pipeline for Hogan: feature engineering, training, and inference."""
from __future__ import annotations

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


@dataclass
class TrainedModel:
    model: object
    feature_columns: list[str]
    # StandardScaler fitted on training data; None for tree-based models.
    # Use getattr(m, "scaler", None) when reading pickled artifacts that
    # pre-date this field to stay backward-compatible.
    scaler: object | None = field(default=None)


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

    # Determine day grouping from available timestamp columns
    if "ts_ms" in candles.columns:
        day_key = pd.to_datetime(candles["ts_ms"], unit="ms", utc=True).dt.date
    elif "timestamp" in candles.columns:
        day_key = pd.to_datetime(candles["timestamp"], utc=True).dt.date
    else:
        # Fallback: assume 288 5m bars per day
        day_key = pd.Series(np.arange(len(close)) // 288, index=close.index)

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
    if "ts_ms" in candles.columns:
        day_key = pd.to_datetime(candles["ts_ms"], unit="ms", utc=True).dt.date
    elif "timestamp" in candles.columns:
        day_key = pd.to_datetime(candles["timestamp"], utc=True).dt.date
    else:
        day_key = pd.Series(np.arange(len(close)) // 288, index=close.index)

    day_key_s = pd.Series(day_key.values, index=close.index)
    prev_day_high = high.groupby(day_key_s).transform("max").shift(288)
    prev_day_low  = low.groupby(day_key_s).transform("min").shift(288)
    # % distance below PDH (positive = price is below PDH, the "safe" zone)
    frame["ict_pdh_dist"] = (prev_day_high - close) / close.clip(lower=1e-9)
    # % distance above PDL (positive = price is above PDL, the "safe" zone)
    frame["ict_pdl_dist"] = (close - prev_day_low) / close.clip(lower=1e-9)

    return frame


# Feature column order is fixed — 43 features total (36 base + 7 ICT structural).
# Update _FEATURE_COLUMNS and tests together.
_FEATURE_COLUMNS: list[str] = [
    # momentum (4)
    "ret_1", "ret_3", "ret_6", "ret_12",
    # trend (1)
    "ma_spread",
    # volatility / oscillators (3)
    "volatility_20", "rsi_14", "atr_pct",
    # MACD (1)
    "macd_hist_pct",
    # Bollinger (1)
    "bb_pct_b",
    # regime (1)
    "vol_regime",
    # candle microstructure (4)
    "range_pct", "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
    # volume (2)
    "vol_ratio", "vol_spike",
    # EMA cloud (3)
    "cloud_bull", "cloud_bear", "cloud_width_pct",
    # FVG (4)
    "fvg_bull_active", "fvg_bear_active", "in_bull_fvg", "in_bear_fvg",
    # ADX — trend strength + directional lines (3)
    "adx_14", "plus_di", "minus_di",
    # Stochastic RSI (2)
    "stoch_rsi_k", "stoch_rsi_d",
    # OBV z-score (1)
    "obv_norm",
    # VWAP distance (1)
    "vwap_dist",
    # Keltner channel position (1)
    "keltner_pos",
    # CCI, MFI, CMF, ROC (4)
    "cci_20", "mfi_14", "cmf_20", "roc_10",
    # ICT structural features (7) — premium/discount, liquidity proximity, PDH/PDL
    "ict_pd_pct",           # dealing-range position [0=discount, 1=premium]
    "ict_in_discount",      # 1 if price < range midpoint
    "ict_in_premium",       # 1 if price > range midpoint
    "ict_swing_high_dist",  # % below nearest 14-bar swing high (buy-side liquidity)
    "ict_swing_low_dist",   # % above nearest 14-bar swing low (sell-side liquidity)
    "ict_pdh_dist",         # % below previous-day high (PDH)
    "ict_pdl_dist",         # % above previous-day low (PDL)
]

assert len(_FEATURE_COLUMNS) == 43, f"Expected 43 features, got {len(_FEATURE_COLUMNS)}"


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
    """73-dim extended feature vector (36 base + 14 MTF + 20 ext + 3 pos).

    Convenience re-export — delegates to
    :func:`hogan_bot.features_mtf.build_feature_row_extended`.
    """
    from hogan_bot.features_mtf import build_feature_row_extended as _ext
    return _ext(candles_5m, candles_1h, candles_15m, conn=conn, symbol=symbol)


def build_feature_row(candles: pd.DataFrame) -> list[float] | None:
    """Return the feature vector for the **last bar** in *candles*.

    Used by :class:`~hogan_bot.rl_env.TradingEnv` at every environment step
    to build the ML-feature portion of the observation vector.

    Returns ``None`` when there is insufficient history to compute all features
    (fewer than 60 bars).  The caller is expected to substitute zeros.
    ADX and Stochastic RSI both need ~28 bars of warmup; 60 is conservative.
    """
    if len(candles) < 60:
        return None
    try:
        frame = _feature_frame(candles)
        last = frame[_FEATURE_COLUMNS].iloc[-1]
        if last.isna().any():
            return None
        return [float(v) for v in last.values]
    except Exception:  # noqa: BLE001
        return None


def build_training_set(
    candles: pd.DataFrame, horizon_bars: int = 3
) -> tuple[pd.DataFrame | None, pd.Series | None, list[str]]:
    """Construct feature matrix *X* and label vector *y* from *candles*.

    Returns ``(None, None, feature_columns)`` when the dataset is empty after
    dropping NaN rows.
    """
    frame = _feature_frame(candles)
    future_ret = frame["close"].shift(-horizon_bars) / frame["close"] - 1.0
    frame["target"] = (future_ret > 0).astype(int)

    dataset = frame[_FEATURE_COLUMNS + ["target"]].dropna().copy()
    if dataset.empty:
        return None, None, _FEATURE_COLUMNS

    return dataset[_FEATURE_COLUMNS], dataset["target"], _FEATURE_COLUMNS


def train_logistic_regression(
    candles: pd.DataFrame,
    model_path: str,
    horizon_bars: int = 3,
    tune_hyperparams: bool = False,
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

    x, y, feature_columns = build_training_set(candles, horizon_bars=horizon_bars)
    if x is None or y is None or len(x) < 200:
        raise RuntimeError("Not enough training rows. Increase OHLCV history.")

    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

    if tune_hyperparams:
        best_C, best_model, best_acc = 1.0, None, -1.0
        for C in (0.01, 0.1, 1.0, 10.0):
            candidate = LogisticRegression(max_iter=500, C=C, class_weight="balanced")
            candidate.fit(x_train_sc, y_train)
            acc = float(accuracy_score(y_test, candidate.predict(x_test_sc)))
            if acc > best_acc:
                best_acc, best_C, best_model = acc, C, candidate
        model = best_model
    else:
        best_C = 1.0
        model = LogisticRegression(max_iter=500, C=best_C, class_weight="balanced")
        model.fit(x_train_sc, y_train)

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
    candles: pd.DataFrame, model_path: str, horizon_bars: int = 3
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

    x, y, feature_columns = build_training_set(candles, horizon_bars=horizon_bars)
    if x is None or y is None or len(x) < 200:
        raise RuntimeError("Not enough training rows. Increase OHLCV history.")

    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)
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
    candles: pd.DataFrame, model_path: str, horizon_bars: int = 3
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

    x, y, feature_columns = build_training_set(candles, horizon_bars=horizon_bars)
    if x is None or y is None or len(x) < 200:
        raise RuntimeError("Not enough training rows. Increase OHLCV history.")

    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

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
    model.fit(x_train, y_train)
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
    candles: pd.DataFrame, model_path: str, horizon_bars: int = 3
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

    x, y, feature_columns = build_training_set(candles, horizon_bars=horizon_bars)
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
    horizon_bars: int = 3,
    method: str = "sigmoid",
    calibration_fraction: float = 0.2,
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
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for calibration") from exc

    artifact = load_model(model_path)
    x, y, feature_columns = build_training_set(candles, horizon_bars=horizon_bars)
    if x is None or y is None or len(x) < 100:
        raise RuntimeError("Not enough data for calibration. Increase OHLCV history.")

    cal_start = int(len(x) * (1.0 - calibration_fraction))
    x_cal = x.iloc[cal_start:]
    y_cal = y.iloc[cal_start:]

    scaler = getattr(artifact, "scaler", None)
    if scaler is not None:
        x_cal_arr = scaler.transform(x_cal)
    else:
        x_cal_arr = x_cal.values

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
    candles: pd.DataFrame, horizon_bars: int = 3, n_splits: int = 5
) -> list[dict[str, object]]:
    """Walk-forward time-series cross-validation using logistic regression.

    Each fold trains on all data up to a cutpoint and tests on the next
    segment, mirroring live deployment.  Returns a list of per-fold dicts
    with ``fold``, ``train_rows``, ``test_rows``, ``accuracy``, ``roc_auc``.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for walk-forward CV") from exc

    x, y, _ = build_training_set(candles, horizon_bars=horizon_bars)
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

    results: list[dict[str, object]] = []
    for i in range(n_splits):
        train_end = min_train + fold_size * i
        test_end = min(train_end + fold_size, n)

        x_train = x.iloc[:train_end]
        y_train = y.iloc[:train_end]
        x_test = x.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        if len(x_test) < 10:
            continue

        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train)
        x_test_sc = scaler.transform(x_test)

        model = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
        model.fit(x_train_sc, y_train)
        pred = model.predict(x_test_sc)
        proba = model.predict_proba(x_test_sc)[:, 1]

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


def load_model(model_path: str) -> TrainedModel:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, TrainedModel):
        raise RuntimeError("Invalid model artifact format")
    return model


def predict_up_probability(candles: pd.DataFrame, trained_model: TrainedModel) -> float:
    frame = _feature_frame(candles)
    latest = frame[trained_model.feature_columns].dropna().tail(1)
    if latest.empty:
        return 0.5
    # Use getattr for backward compatibility with pre-scaler pickled artifacts.
    scaler = getattr(trained_model, "scaler", None)
    if scaler is not None:
        latest_arr = scaler.transform(latest)
        proba = trained_model.model.predict_proba(latest_arr)[0][1]
    else:
        proba = trained_model.model.predict_proba(latest)[0][1]
    return float(proba)



def train_hist_gradient_boosting(
    candles: pd.DataFrame,
    model_path: str,
    horizon_bars: int = 3,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    max_iter: int = 400,
) -> dict[str, object]:
    """Train + save sklearn HistGradientBoostingClassifier (strong tabular baseline)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

    X, y, feature_cols = make_feature_matrix(candles, horizon_bars=horizon_bars)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    clf = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42,
    )
    clf.fit(X_train, y_train)

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
    save_model(artifact, model_path)
    return metrics


def build_feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Return a feature DataFrame aligned to candles index for advanced models."""
    frame = _feature_frame(candles)
    cols = list(_FEATURE_COLUMNS)
    out = frame[cols].copy()
    return out.replace([np.inf, -np.inf], np.nan)
