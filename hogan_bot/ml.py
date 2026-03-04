"""Machine-learning pipeline for Hogan: feature engineering, training, and inference."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hogan_bot.indicators import cloud_signal, fvg_features_frame, ripster_ema_clouds


@dataclass
class TrainedModel:
    model: object
    feature_columns: list[str]
    # StandardScaler fitted on training data; None for models that don't need scaling
    # (e.g. RandomForest).  Use getattr(m, "scaler", None) when reading pickled
    # artifacts that pre-date this field.
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


def _feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix.  Returns a copy of *candles* with feature
    columns appended.  No look-ahead — every column is computable from data
    available up to and including each bar.
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
    # Realized volatility and momentum oscillator
    # ------------------------------------------------------------------
    frame["volatility_20"] = frame["ret_1"].rolling(20).std()
    frame["rsi_14"] = _rsi(close, window=14) / 100.0  # normalized to [0, 1]

    # ------------------------------------------------------------------
    # Candle microstructure
    # ------------------------------------------------------------------
    frame["range_pct"] = (high - low) / close.clip(lower=1e-9)
    frame["candle_body_pct"] = (close - open_).abs() / close.clip(lower=1e-9)

    body_top = np.maximum(open_.values, close.values)
    body_bot = np.minimum(open_.values, close.values)
    frame["upper_wick_pct"] = np.maximum(high.values - body_top, 0.0) / close.clip(lower=1e-9).values
    frame["lower_wick_pct"] = np.maximum(body_bot - low.values, 0.0) / close.clip(lower=1e-9).values

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
    # Signed normalized distance between fast and slow cloud tops
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

    return frame


# Feature column order is fixed; update build_training_set and tests together.
_FEATURE_COLUMNS: list[str] = [
    # momentum
    "ret_1", "ret_3", "ret_6", "ret_12",
    # trend
    "ma_spread",
    # volatility / oscillator
    "volatility_20", "rsi_14",
    # candle microstructure
    "range_pct", "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
    # volume
    "vol_ratio", "vol_spike",
    # EMA cloud
    "cloud_bull", "cloud_bear", "cloud_width_pct",
    # FVG
    "fvg_bull_active", "fvg_bear_active", "in_bull_fvg", "in_bear_fvg",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_training_set(
    candles: pd.DataFrame, horizon_bars: int = 3
) -> tuple[pd.DataFrame | None, pd.Series | None, list[str]]:
    """Construct feature matrix *X* and label vector *y* from *candles*.

    Returns ``(None, None, feature_columns)`` if the dataset is empty after
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
    candles: pd.DataFrame, model_path: str, horizon_bars: int = 3
) -> dict[str, object]:
    """Fit a scaled logistic-regression classifier and pickle the artifact.

    Returns a metrics dict that now includes ``roc_auc``, ``precision``,
    ``recall``, and ``f1`` in addition to ``accuracy``.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
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

    model = LogisticRegression(max_iter=500, C=1.0)
    model.fit(x_train_sc, y_train)
    pred = model.predict(x_test_sc)
    proba = model.predict_proba(x_test_sc)[:, 1]

    auc = float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.5
    metrics: dict[str, object] = {
        "model_type": "logistic_regression",
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": auc,
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
    ``feature_importances``.  Random Forest is invariant to feature scale so
    no ``StandardScaler`` is applied; the pickled ``TrainedModel.scaler`` is
    ``None``.
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


def walk_forward_cv(
    candles: pd.DataFrame, horizon_bars: int = 3, n_splits: int = 5
) -> list[dict[str, object]]:
    """Walk-forward time-series cross-validation using logistic regression.

    Splits the labelled dataset into *n_splits* expanding-window folds.  Each
    fold trains on all data up to a cutpoint and tests on the next segment,
    mirroring live deployment where the model never sees future data.

    Returns a list of per-fold metric dicts with ``fold``, ``train_rows``,
    ``test_rows``, ``accuracy``, and ``roc_auc``.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
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

        model = LogisticRegression(max_iter=500, C=1.0)
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
