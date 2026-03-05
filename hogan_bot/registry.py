"""Model registry for Hogan — append-only JSONL log + MLflow experiment tracking.

MLflow is optional: when ``mlflow`` is installed and ``MLFLOW_TRACKING_URI`` is
set (or defaults to ``./mlruns``), every training run is also logged as an
MLflow run with params, metrics, and model artifact.  The JSONL file is always
written as a lightweight local fallback.

Start the MLflow UI::

    mlflow server --host 0.0.0.0 --port 5000

Or just::

    mlflow ui
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_MLFLOW_AVAILABLE = False
try:
    import mlflow  # type: ignore
    _MLFLOW_AVAILABLE = True
except ImportError:
    pass


def _mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "mlruns")


class ModelRegistry:
    """Append-only registry that records every training run as a JSONL file.

    Each line is a self-contained JSON object describing the run::

        {
            "timestamp":     "2026-03-04T12:00:00+00:00",
            "model_path":    "models/hogan_logreg.pkl",
            "model_hash":    "ab12cd34ef56",
            "model_type":    "logistic_regression",
            "symbol":        "BTC/USD",
            "timeframe":     "5m",
            "horizon_bars":  3,
            "features":      24,
            "metrics": {
                "accuracy": 0.52,
                "roc_auc":  0.55,
                ...
            }
        }

    Usage::

        registry = ModelRegistry()
        registry.log(metrics, model_path="models/hogan_logreg.pkl",
                     symbol="BTC/USD", timeframe="5m", horizon_bars=3)
        best = registry.best(metric="roc_auc")
    """

    def __init__(
        self,
        registry_path: str = "models/registry.jsonl",
        experiment_name: str = "hogan",
        use_mlflow: bool = True,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self._use_mlflow = use_mlflow and _MLFLOW_AVAILABLE
        if self._use_mlflow:
            mlflow.set_tracking_uri(_mlflow_tracking_uri())
            mlflow.set_experiment(experiment_name)
            logger.debug("MLflow tracking at %s, experiment=%s", _mlflow_tracking_uri(), experiment_name)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log(
        self,
        metrics: dict,
        model_path: str,
        symbol: str = "BTC/USD",
        timeframe: str = "5m",
        horizon_bars: int = 3,
        params: dict | None = None,
    ) -> dict:
        """Append a training-run record to JSONL and optionally MLflow."""
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "model_path": str(model_path),
            "model_hash": self._hash_file(model_path),
            "model_type": metrics.get("model_type", "unknown"),
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon_bars": horizon_bars,
            "features": metrics.get("features", 0),
            "metrics": {
                k: v
                for k, v in metrics.items()
                if k not in ("model_type", "features", "feature_importances")
            },
        }
        with open(self.registry_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

        # MLflow logging (best-effort — never crash on MLflow error)
        if self._use_mlflow:
            try:
                with mlflow.start_run(run_name=f"{metrics.get('model_type','model')}-{symbol}-{timeframe}"):
                    mlflow.set_tags({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "horizon_bars": horizon_bars,
                        "model_type": metrics.get("model_type", "unknown"),
                    })
                    log_params = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "horizon_bars": horizon_bars,
                        "features": metrics.get("features", 0),
                    }
                    if params:
                        log_params.update(params)
                    mlflow.log_params(log_params)
                    log_metrics = {
                        k: float(v) for k, v in entry["metrics"].items()
                        if isinstance(v, (int, float)) and v == v  # exclude NaN
                    }
                    mlflow.log_metrics(log_metrics)
                    if Path(model_path).exists():
                        mlflow.log_artifact(model_path, artifact_path="model")
            except Exception as exc:
                logger.warning("MLflow log failed (non-fatal): %s", exc)

        return entry

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def load_all(self) -> list[dict]:
        """Return all registry entries in chronological order."""
        if not self.registry_path.exists():
            return []
        entries: list[dict] = []
        with open(self.registry_path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    try:
                        entries.append(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
        return entries

    def best(self, metric: str = "roc_auc") -> dict | None:
        """Return the entry with the highest value for *metric*.

        Looks inside the nested ``metrics`` dict for the metric key.
        Returns ``None`` when the registry is empty.
        """
        entries = self.load_all()
        if not entries:
            return None
        return max(
            entries,
            key=lambda e: e.get("metrics", {}).get(metric, float("-inf")),
        )

    def summary(self) -> list[dict]:
        """Return a flat list of {timestamp, model_type, symbol, accuracy,
        roc_auc, f1} suitable for tabular display."""
        rows = []
        for e in self.load_all():
            m = e.get("metrics", {})
            rows.append(
                {
                    "timestamp": e.get("timestamp", ""),
                    "model_type": e.get("model_type", ""),
                    "symbol": e.get("symbol", ""),
                    "timeframe": e.get("timeframe", ""),
                    "horizon_bars": e.get("horizon_bars", ""),
                    "accuracy": m.get("accuracy"),
                    "roc_auc": m.get("roc_auc"),
                    "f1": m.get("f1"),
                    "model_path": e.get("model_path", ""),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(path: str) -> str:
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                h.update(fh.read())
            return h.hexdigest()[:16]
        except (FileNotFoundError, OSError):
            return "unknown"
