
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from hogan_bot.ml import build_feature_frame


# ----------------------------
# Triple-barrier labeling
# ----------------------------

def triple_barrier_labels(close: pd.Series, horizon: int = 48, vol_span: int = 100, k: float = 2.0) -> pd.Series:
    """Label each bar using a simple triple-barrier scheme.

    For each time t:
      - compute volatility estimate (EWMA of returns)
      - set upper/lower barriers = +/- k * vol * price
      - look ahead up to `horizon` bars to see which barrier is hit first

    Returns labels in {0,1} where 1 = up move wins, 0 = down move wins.
    """
    ret = close.pct_change().fillna(0.0)
    vol = ret.ewm(span=vol_span, adjust=False).std().fillna(method="bfill").clip(lower=1e-9)

    labels = np.full(len(close), np.nan)
    prices = close.values
    volv = vol.values

    for i in range(len(close) - horizon - 1):
        p0 = prices[i]
        up = p0 * (1 + k * volv[i])
        dn = p0 * (1 - k * volv[i])
        future = prices[i + 1 : i + 1 + horizon]
        hit = None
        for pj in future:
            if pj >= up:
                hit = 1
                break
            if pj <= dn:
                hit = 0
                break
        if hit is None:
            hit = 1 if future[-1] >= p0 else 0
        labels[i] = hit

    s = pd.Series(labels, index=close.index)
    return s.astype("float").fillna(method="ffill").fillna(0.0).astype(int)


# ----------------------------
# Regime detection (trend / mean-revert / chop proxy)
# ----------------------------

def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    ret = close.pct_change().fillna(0.0)
    out["vol_20"] = ret.rolling(20).std().fillna(0.0)
    out["trend_50"] = close.pct_change(50).fillna(0.0)
    out["autocorr_20"] = ret.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False).fillna(0.0)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def fit_regime_model(df: pd.DataFrame, n_regimes: int = 3, seed: int = 7) -> KMeans:
    X = regime_features(df).values
    km = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
    km.fit(X)
    return km


def infer_regime(km: KMeans, df: pd.DataFrame) -> pd.Series:
    X = regime_features(df).values
    lab = km.predict(X)
    return pd.Series(lab, index=df.index)


# ----------------------------
# Ensemble artifact
# ----------------------------

@dataclass
class AdvancedEnsembleArtifact:
    artifact_type: str
    feature_columns: List[str]
    regime_model: KMeans
    models_by_regime: Dict[int, object]
    calibration: bool
    horizon: int
    label_k: float


def train_advanced_ensemble(
    candles: pd.DataFrame,
    horizon: int = 48,
    label_k: float = 2.0,
    n_regimes: int = 3,
    seed: int = 7,
) -> Tuple[AdvancedEnsembleArtifact, Dict[str, float]]:
    """Train a regime-aware ensemble with calibrated probabilities."""
    df = candles.copy()
    feat = build_feature_frame(df).dropna()
    if feat.empty:
        raise ValueError("Not enough data to train")

    y = triple_barrier_labels(df.loc[feat.index, "close"], horizon=horizon, k=label_k)

    km = fit_regime_model(df.loc[feat.index], n_regimes=n_regimes, seed=seed)
    regimes = infer_regime(km, df.loc[feat.index])

    models: Dict[int, object] = {}
    aucs: List[float] = []
    for r in range(n_regimes):
        mask = regimes == r
        Xr = feat.loc[mask].values
        yr = y.loc[mask].values
        if len(yr) < 500 or len(np.unique(yr)) < 2:
            Xr = feat.values
            yr = y.values

        base = HistGradientBoostingClassifier(
            random_state=seed,
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
        )
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(Xr, yr)
        models[int(r)] = cal

        pr = cal.predict_proba(Xr)[:, 1]
        try:
            aucs.append(float(roc_auc_score(yr, pr)))
        except Exception:
            pass

    artifact = AdvancedEnsembleArtifact(
        artifact_type="advanced_ensemble_v1",
        feature_columns=list(feat.columns),
        regime_model=km,
        models_by_regime=models,
        calibration=True,
        horizon=horizon,
        label_k=label_k,
    )
    metrics = {
        "train_auc_mean": float(np.mean(aucs)) if aucs else 0.0,
        "train_auc_min": float(np.min(aucs)) if aucs else 0.0,
        "train_auc_max": float(np.max(aucs)) if aucs else 0.0,
        "n_rows": int(len(feat)),
    }
    return artifact, metrics


def save_artifact(artifact: AdvancedEnsembleArtifact, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def load_artifact(path: str) -> AdvancedEnsembleArtifact:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not getattr(obj, "artifact_type", "").startswith("advanced_ensemble"):
        raise TypeError("Not an advanced ensemble artifact")
    return obj


def predict_up_probability(artifact: AdvancedEnsembleArtifact, candles: pd.DataFrame) -> float:
    """Predict P(up) for the most recent bar."""
    feat = build_feature_frame(candles).dropna()
    if feat.empty:
        return 0.5
    X = feat.values.astype(float)
    reg = int(infer_regime(artifact.regime_model, candles.loc[feat.index]).iloc[-1])
    model = artifact.models_by_regime.get(reg) or list(artifact.models_by_regime.values())[0]
    p = float(model.predict_proba(X[-1:])[:, 1][0])
    return max(0.0, min(1.0, p))
