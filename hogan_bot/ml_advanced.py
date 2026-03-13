
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
    vol = ret.ewm(span=vol_span, adjust=False).std().bfill().clip(lower=1e-9)

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
    return s.astype("float").ffill().fillna(0.0).astype(int)


def triple_barrier_labels_enhanced(
    candles: pd.DataFrame,
    horizon: int = 48,
    vol_span: int = 100,
    k_up: float = 2.0,
    k_dn: float = 2.0,
    time_decay: bool = True,
    regime_adaptive: bool = True,
    fee_rate: float = 0.0026,
) -> pd.DataFrame:
    """Enhanced triple-barrier labeling with regime awareness and time decay.

    Improvements over the basic version:

    1. **Asymmetric barriers**: ``k_up`` and ``k_dn`` can differ, allowing
       wider profit targets and tighter stops (or vice versa).

    2. **Regime-adaptive barriers**: When ``regime_adaptive=True``, barriers
       widen in trending markets (let winners run) and tighten in ranging
       markets (take profits sooner).

    3. **Time decay**: When ``time_decay=True``, barriers shrink linearly
       toward expiry — penalizing trades that take too long to resolve,
       which is realistic for time-sensitive strategies.

    4. **Metalabel output**: Returns a DataFrame with columns:
       - ``label``: 1 (profitable) or 0 (not)
       - ``barrier_hit``: ``"upper"`` / ``"lower"`` / ``"timeout"``
       - ``bars_to_hit``: how many bars until barrier was touched
       - ``pnl_at_hit``: percentage P&L at the barrier hit
       - ``meta_quality``: [0, 1] quality score (clean hits score higher
         than timeouts; early hits score higher than late)

    Parameters
    ----------
    candles : DataFrame
        OHLCV data with ``close``, ``high``, ``low`` columns.
    horizon : int
        Maximum bars to look ahead.
    vol_span : int
        EWMA span for volatility estimate.
    k_up / k_dn : float
        Multipliers for upper/lower barriers (in units of vol).
    time_decay : bool
        Shrink barriers linearly toward expiry.
    regime_adaptive : bool
        Adjust barrier widths based on detected ADX regime.
    fee_rate : float
        Round-trip fee rate; labels account for friction.
    """
    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)

    ret = close.pct_change().fillna(0.0)
    vol = ret.ewm(span=vol_span, adjust=False).std().bfill().clip(lower=1e-9)

    regime_mult_up = np.ones(len(close))
    regime_mult_dn = np.ones(len(close))

    if regime_adaptive:
        adx_series = _quick_adx(high, low, close, period=14)
        ma_fast = close.rolling(20).mean()
        ma_slow = close.rolling(50).mean()
        trend_up = (ma_fast > ma_slow).astype(float).values
        adx_vals = adx_series.values

        for i in range(len(close)):
            adx = adx_vals[i] if not np.isnan(adx_vals[i]) else 20.0
            if adx > 25:
                if trend_up[i]:
                    regime_mult_up[i] = 1.5
                    regime_mult_dn[i] = 0.8
                else:
                    regime_mult_up[i] = 0.8
                    regime_mult_dn[i] = 1.5
            elif adx < 18:
                regime_mult_up[i] = 0.7
                regime_mult_dn[i] = 0.7

    n = len(close)
    labels = np.full(n, np.nan)
    barrier_hit = np.full(n, "", dtype=object)
    bars_to_hit = np.full(n, np.nan)
    pnl_at_hit = np.full(n, np.nan)
    meta_quality = np.full(n, np.nan)

    prices = close.values
    highs = high.values
    lows = low.values
    volv = vol.values

    for i in range(n - horizon - 1):
        p0 = prices[i]
        base_up = k_up * volv[i] * regime_mult_up[i]
        base_dn = k_dn * volv[i] * regime_mult_dn[i]

        hit_type = None
        hit_bar = horizon
        hit_pnl = 0.0

        for j in range(1, horizon + 1):
            idx = i + j
            if idx >= n:
                break

            if time_decay:
                decay = 1.0 - 0.4 * (j / horizon)
            else:
                decay = 1.0

            up_barrier = p0 * (1.0 + base_up * decay)
            dn_barrier = p0 * (1.0 - base_dn * decay)

            if highs[idx] >= up_barrier:
                hit_type = "upper"
                hit_bar = j
                hit_pnl = (up_barrier - p0) / p0 - fee_rate
                break

            if lows[idx] <= dn_barrier:
                hit_type = "lower"
                hit_bar = j
                hit_pnl = (dn_barrier - p0) / p0 - fee_rate
                break

        if hit_type is None:
            hit_type = "timeout"
            hit_bar = horizon
            hit_pnl = (prices[min(i + horizon, n - 1)] - p0) / p0 - fee_rate

        labels[i] = 1 if hit_pnl > 0 else 0
        barrier_hit[i] = hit_type
        bars_to_hit[i] = hit_bar
        pnl_at_hit[i] = hit_pnl

        if hit_type == "timeout":
            meta_quality[i] = 0.2
        else:
            speed_bonus = 1.0 - (hit_bar / horizon)
            pnl_bonus = min(1.0, abs(hit_pnl) / (3.0 * volv[i] + 1e-9))
            meta_quality[i] = 0.4 + 0.3 * speed_bonus + 0.3 * pnl_bonus

    result = pd.DataFrame({
        "label": labels,
        "barrier_hit": barrier_hit,
        "bars_to_hit": bars_to_hit,
        "pnl_at_hit": pnl_at_hit,
        "meta_quality": meta_quality,
    }, index=candles.index)

    return result


def _quick_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Lightweight ADX computation for regime-adaptive barriers."""
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -(low.diff())
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=close.index,
    )

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr.clip(lower=1e-9)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr.clip(lower=1e-9)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-9) * 100.0
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx


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
