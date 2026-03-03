from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainedModel:
    model: object
    feature_columns: list[str]


def _feature_frame(candles):
    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    volume = candles["volume"].astype(float)

    frame = candles.copy()
    frame["ret_1"] = close.pct_change(1)
    frame["ret_3"] = close.pct_change(3)
    frame["ret_6"] = close.pct_change(6)
    frame["ma_fast"] = close.rolling(12).mean()
    frame["ma_slow"] = close.rolling(48).mean()
    frame["ma_spread"] = (frame["ma_fast"] / frame["ma_slow"]) - 1.0
    frame["range_pct"] = (high - low) / close.clip(lower=1e-9)
    frame["vol_ratio"] = volume / volume.rolling(20).mean().clip(lower=1e-9)
    return frame


def build_training_set(candles, horizon_bars: int = 3):
    frame = _feature_frame(candles)
    future_ret = frame["close"].shift(-horizon_bars) / frame["close"] - 1.0
    frame["target"] = (future_ret > 0).astype(int)

    feature_columns = ["ret_1", "ret_3", "ret_6", "ma_spread", "range_pct", "vol_ratio"]
    dataset = frame[feature_columns + ["target"]].dropna().copy()
    if dataset.empty:
        return None, None, feature_columns

    x = dataset[feature_columns]
    y = dataset["target"]
    return x, y, feature_columns


def train_logistic_regression(candles, model_path: str, horizon_bars: int = 3) -> dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for training") from exc

    x, y, feature_columns = build_training_set(candles, horizon_bars=horizon_bars)
    if x is None or y is None or len(x) < 200:
        raise RuntimeError("Not enough training rows. Increase OHLCV history.")

    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, pred))

    artifact = TrainedModel(model=model, feature_columns=feature_columns)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    return {"accuracy": acc, "train_rows": float(len(x_train)), "test_rows": float(len(x_test))}


def load_model(model_path: str) -> TrainedModel:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, TrainedModel):
        raise RuntimeError("Invalid model artifact format")
    return model


def predict_up_probability(candles, trained_model: TrainedModel) -> float:
    frame = _feature_frame(candles)
    latest = frame[trained_model.feature_columns].dropna().tail(1)
    if latest.empty:
        return 0.5
    proba = trained_model.model.predict_proba(latest)[0][1]
    return float(proba)
