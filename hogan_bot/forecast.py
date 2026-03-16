"""Forecast head — directional probability over multiple horizons.

Produces structured forward-looking estimates that the policy layer
uses to decide *whether there is edge*, independent of the technical
signal's crossover/trend logic.

Outputs
-------
ForecastResult
    direction_prob : dict[str, float]
        Probability of positive return at 4h, 12h, 24h horizons.
    expected_return : dict[str, float]
        Expected return (%) at each horizon (simple historical analog).
    trend_persistence : float
        Probability the current trend direction continues (0-1).
    confidence : float
        Overall confidence in the forecast (0-1), based on feature coverage.

Architecture (two modes)
------------------------
**Trained mode** (preferred): Loads calibrated horizon-specific LightGBM
models from ``models/forecast_{horizon}.pkl``.  Uses the full 59-feature
frame from ``ml._feature_frame()`` for inference.  Calibration is via
``CalibratedClassifierCV`` so output probabilities are reliable.

**Heuristic fallback**: When model files don't exist, uses hand-crafted
features with fixed sigmoid coefficients.  This is the original v1
implementation kept for cold-start and testing.

Calibration metrics tracked during training:
- Brier score, ECE (Expected Calibration Error), ROC-AUC
- Precision/recall at operational thresholds
"""
from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hogan_bot.indicators import compute_atr

logger = logging.getLogger(__name__)

_MODEL_DIR = Path("models")
_HORIZONS = {"4h": 4, "12h": 12, "24h": 24}

_model_cache: dict[str, object] = {}


@dataclass
class ForecastResult:
    """Structured forecast output for the policy layer."""
    direction_prob: dict[str, float] = field(default_factory=dict)
    expected_return: dict[str, float] = field(default_factory=dict)
    trend_persistence: float = 0.5
    confidence: float = 0.0

    @property
    def bullish_4h(self) -> float:
        return self.direction_prob.get("4h", 0.5)

    @property
    def bullish_12h(self) -> float:
        return self.direction_prob.get("12h", 0.5)

    @property
    def bullish_24h(self) -> float:
        return self.direction_prob.get("24h", 0.5)

    def summary(self) -> str:
        parts = []
        for h in ("4h", "12h", "24h"):
            p = self.direction_prob.get(h, 0.5)
            er = self.expected_return.get(h, 0.0)
            parts.append(f"{h}:{p:.0%}up/{er:+.2f}%")
        parts.append(f"persist={self.trend_persistence:.0%}")
        return " ".join(parts)


def _safe_sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _load_forecast_model(horizon: str):
    """Load a trained forecast model from disk (cached after first load)."""
    if horizon in _model_cache:
        return _model_cache[horizon]
    path = _MODEL_DIR / f"forecast_{horizon}.pkl"
    if not path.exists():
        _model_cache[horizon] = None
        return None
    try:
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        _model_cache[horizon] = model
        logger.info("Loaded forecast model: %s", path)
        return model
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        _model_cache[horizon] = None
        return None


def _ml_forecast(candles: pd.DataFrame) -> ForecastResult | None:
    """Try to produce a forecast using trained ML models.

    Returns None if any horizon model is missing, signaling fallback
    to the heuristic implementation.
    """
    models = {h: _load_forecast_model(h) for h in _HORIZONS}
    if any(m is None for m in models.values()):
        return None

    try:
        from hogan_bot.ml import _FEATURE_COLUMNS, _feature_frame
        frame = _feature_frame(candles)
        last_row = frame[_FEATURE_COLUMNS].iloc[[-1]]
        if last_row.isna().any(axis=1).iloc[0]:
            return None
    except Exception:
        return None

    direction_prob: dict[str, float] = {}
    expected_return: dict[str, float] = {}

    close = candles["close"].astype(float)
    ret_1 = close.pct_change(1)
    vol_20 = float(ret_1.rolling(20).std().iloc[-1]) if len(close) >= 20 else 0.01

    for horizon_name, horizon_bars in _HORIZONS.items():
        model = models[horizon_name]
        try:
            scaler = getattr(model, "scaler", None)
            clf = getattr(model, "model", model)
            X = last_row.values
            if scaler is not None:
                X = scaler.transform(X)
            prob = float(clf.predict_proba(X)[:, 1][0])
        except Exception:
            return None

        direction_prob[horizon_name] = round(prob, 4)
        typical_move = vol_20 * math.sqrt(horizon_bars) * 100.0
        expected_return[horizon_name] = round((2.0 * prob - 1.0) * typical_move, 4)

    ma_fast = close.rolling(12).mean()
    ma_slow = close.rolling(48).mean()
    ma_spread = float(((ma_fast - ma_slow) / ma_slow.clip(lower=1e-9)).iloc[-1])

    avg_prob = np.mean(list(direction_prob.values()))
    trend_persistence = round(float(avg_prob if ma_spread > 0 else 1.0 - avg_prob), 4)
    confidence = 0.85

    return ForecastResult(
        direction_prob=direction_prob,
        expected_return=expected_return,
        trend_persistence=trend_persistence,
        confidence=confidence,
    )


def compute_forecast(candles: pd.DataFrame) -> ForecastResult:
    """Compute multi-horizon directional forecast from OHLCV data.

    Tries trained ML models first; falls back to heuristic when
    model files don't exist.
    """
    min_bars = 80
    if candles is None or len(candles) < min_bars:
        return ForecastResult(confidence=0.0)

    ml_result = _ml_forecast(candles)
    if ml_result is not None:
        return ml_result

    return _heuristic_forecast(candles)


def _heuristic_forecast(candles: pd.DataFrame) -> ForecastResult:
    """Original heuristic forecast (v1 fallback)."""
    close = candles["close"].astype(float)
    volume = candles["volume"].astype(float)

    ret_1 = close.pct_change(1)

    ma_fast = close.rolling(12).mean()
    ma_slow = close.rolling(48).mean()
    ma_spread = ((ma_fast - ma_slow) / ma_slow.clip(lower=1e-9)).iloc[-1]

    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(com=13, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(com=13, min_periods=14, adjust=False).mean()
    rs = gain / loss.clip(lower=1e-9)
    rsi = (100.0 - (100.0 / (1.0 + rs))).iloc[-1] / 100.0

    vol_20 = float(ret_1.rolling(20).std().iloc[-1])
    vol_50 = float(ret_1.rolling(50).std().iloc[-1]) or vol_20
    vol_regime = vol_20 / max(vol_50, 1e-9)

    atr = compute_atr(candles, window=14)
    atr_pct = float(atr.iloc[-1] / max(close.iloc[-1], 1e-9))

    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1] / max(vol_avg, 1e-9))

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_pct_b = float((close.iloc[-1] - bb_lower.iloc[-1]) / max(bb_upper.iloc[-1] - bb_lower.iloc[-1], 1e-9))

    persistence_logit = (
        2.0 * abs(ma_spread) * 100.0
        + 0.5 * max(0, atr_pct * 100.0 - 1.0)
        - 1.0 * vol_regime
        + 0.3 * (0.5 - abs(rsi - 0.5)) * 2.0
    )
    trend_persistence = _safe_sigmoid(persistence_logit)

    direction_prob = {}
    expected_return = {}

    for horizon_name, horizon_bars in [("4h", 4), ("12h", 12), ("24h", 24)]:
        recent_ret = float(close.pct_change(min(horizon_bars, len(close) - 1)).iloc[-1])
        mr_pressure = -(bb_pct_b - 0.5) * 0.3
        trend_component = ma_spread * 20.0
        vol_boost = min(0.3, max(-0.1, (vol_ratio - 1.0) * 0.15))

        if horizon_bars <= 4:
            logit = 0.5 * recent_ret * 50.0 + 0.3 * mr_pressure + 0.2 * trend_component
        elif horizon_bars <= 12:
            logit = 0.3 * recent_ret * 30.0 + 0.25 * mr_pressure + 0.45 * trend_component
        else:
            logit = 0.15 * recent_ret * 20.0 + 0.2 * mr_pressure + 0.65 * trend_component

        logit += vol_boost
        prob_up = _safe_sigmoid(logit)
        direction_prob[horizon_name] = round(prob_up, 4)
        typical_move = vol_20 * math.sqrt(horizon_bars) * 100.0
        expected_return[horizon_name] = round((2.0 * prob_up - 1.0) * typical_move, 4)

    feature_count = 6
    nan_count = sum(1 for v in [ma_spread, rsi, vol_20, bb_pct_b, vol_ratio, atr_pct]
                    if v != v)
    data_confidence = (feature_count - nan_count) / feature_count
    stability = max(0.0, 1.0 - abs(vol_regime - 1.0))
    confidence = round(data_confidence * (0.5 + 0.5 * stability), 4)

    return ForecastResult(
        direction_prob=direction_prob,
        expected_return=expected_return,
        trend_persistence=round(trend_persistence, 4),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac = mask.sum() / len(y_true)
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += frac * abs(avg_acc - avg_conf)
    return ece


def train_forecast_models(
    candles: pd.DataFrame,
    db_conn=None,
    output_dir: str = "models",
    fee_rate: float = 0.0026,
    purge_bars: int = 24,
) -> dict:
    """Train calibrated forecast models for all horizons.

    Uses a purged walk-forward split::

        [--- train (60%) ---][purge][--- cal (15%) ---][purge][--- test (25%) ---]

    The base LightGBM is trained on the train window, then calibrated
    (isotonic regression) on the separate calibration window.  Metrics
    are computed on the unseen test window.  A JSON model card is saved
    alongside each ``.pkl`` with full provenance.

    Returns a metrics dict with per-horizon Brier, ECE, ROC-AUC,
    precision/recall at operational thresholds.
    """
    import json as _json
    from datetime import datetime, timezone

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        brier_score_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as LGBMClassifier

    from hogan_bot.ml import _FEATURE_COLUMNS, TrainedModel, _feature_frame

    frame = _feature_frame(candles)

    if db_conn is not None:
        try:
            from hogan_bot.macro_features import add_macro_features
            frame = add_macro_features(frame, db_conn)
        except Exception:
            pass
    from hogan_bot.macro_features import MACRO_FEATURE_NAMES
    for col in MACRO_FEATURE_NAMES:
        if col not in frame.columns:
            frame[col] = 0.0

    all_metrics: dict = {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for horizon_name, horizon_bars in _HORIZONS.items():
        future_ret = frame["close"].shift(-horizon_bars) / frame["close"] - 1.0
        frame[f"target_{horizon_name}"] = (future_ret > 0).astype(int)

        dataset = frame[_FEATURE_COLUMNS + [f"target_{horizon_name}"]].dropna().copy()
        if len(dataset) < 300:
            logger.warning("Horizon %s: only %d rows, need >=300, skipping", horizon_name, len(dataset))
            continue

        X = dataset[_FEATURE_COLUMNS]
        y = dataset[f"target_{horizon_name}"]
        n = len(X)

        # Purged walk-forward: train 60% | purge | cal 15% | purge | test 25%
        train_end = int(n * 0.60)
        cal_start = train_end + purge_bars
        cal_end = cal_start + int(n * 0.15)
        test_start = cal_end + purge_bars

        if test_start >= n or cal_end >= n:
            logger.warning("Horizon %s: not enough data for purged split (n=%d)", horizon_name, n)
            continue

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_cal, y_cal = X.iloc[cal_start:cal_end], y.iloc[cal_start:cal_end]
        X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_cal_sc = scaler.transform(X_cal)
        X_test_sc = scaler.transform(X_test)

        # Step 1: Train base classifier on train window
        try:
            base = LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1, n_jobs=-1,
            )
        except TypeError:
            base = LGBMClassifier(
                max_iter=200, max_depth=6, learning_rate=0.05,
                random_state=42,
            )
        base.fit(X_train_sc, y_train)

        # Step 2: Calibrate on separate held-out calibration window
        cal = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
        cal.fit(X_cal_sc, y_cal)

        # Step 3: Evaluate on unseen test window
        proba = cal.predict_proba(X_test_sc)[:, 1]

        brier = float(brier_score_loss(y_test, proba))
        ece = _expected_calibration_error(y_test.values, proba)
        try:
            auc = float(roc_auc_score(y_test, proba))
        except ValueError:
            auc = 0.0

        # Precision/recall at operational thresholds
        thresholds = [0.55, 0.60, 0.65]
        threshold_metrics: dict[str, dict[str, float]] = {}
        for t in thresholds:
            preds = (proba >= t).astype(int)
            if preds.sum() > 0:
                threshold_metrics[f"t{t:.2f}"] = {
                    "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
                    "n_triggered": int(preds.sum()),
                }

        artifact = TrainedModel(scaler=scaler, model=cal, feature_columns=_FEATURE_COLUMNS)
        model_path = out / f"forecast_{horizon_name}.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(artifact, fh)

        _model_cache.pop(horizon_name, None)

        h_metrics = {
            "brier": round(brier, 4),
            "ece": round(ece, 4),
            "roc_auc": round(auc, 4),
            "n_train": int(len(X_train)),
            "n_cal": int(len(X_cal)),
            "n_test": int(len(X_test)),
            "purge_bars": purge_bars,
            "thresholds": threshold_metrics,
        }
        all_metrics[horizon_name] = h_metrics

        # Save model card (JSON sidecar)
        card = {
            "horizon": horizon_name,
            "horizon_bars": horizon_bars,
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "split": {
                "train": f"0:{train_end}",
                "purge1": f"{train_end}:{cal_start}",
                "calibration": f"{cal_start}:{cal_end}",
                "purge2": f"{cal_end}:{test_start}",
                "test": f"{test_start}:{n}",
            },
            "metrics": h_metrics,
            "features": len(_FEATURE_COLUMNS),
            "calibration_method": "isotonic (CalibratedClassifierCV, cv=prefit)",
        }
        card_path = out / f"forecast_{horizon_name}_card.json"
        with open(card_path, "w", encoding="utf-8") as fh:
            _json.dump(card, fh, indent=2)

        logger.info(
            "Forecast %s — brier=%.4f ece=%.4f auc=%.4f (train=%d cal=%d test=%d purge=%d)",
            horizon_name, brier, ece, auc,
            len(X_train), len(X_cal), len(X_test), purge_bars,
        )

    return all_metrics
