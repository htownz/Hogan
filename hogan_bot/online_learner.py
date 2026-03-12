"""Online / Continuous Learning — Phase 4b.

Provides incremental model updates triggered by live trade outcomes without
requiring a full cold retrain.  Three complementary approaches:

1. **SGDClassifier** — sklearn logistic regression with ``partial_fit``.
   Updates weights after each batch of labeled trades (fast, minimal memory).

2. **HistGradientBoostingClassifier** — sklearn warm-started refit on a
   rolling window.  Better accuracy than SGD but heavier (~2-10 s on 10k rows).

3. **KL Divergence feature drift monitor** — compares the current live feature
   distribution against the training distribution.  Emits a ``FEATURE_DRIFT``
   Prometheus counter and logs a warning when drift exceeds threshold.

Usage::

    from hogan_bot.online_learner import OnlineLearner
    learner = OnlineLearner(db_path="data/hogan.db")

    # Call periodically (e.g. every 50 completed trades)
    learner.update(symbol="BTC/USD")

Run as a daemon::

    python -m hogan_bot.online_learner --db data/hogan.db --interval 3600
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = "models"
_DRIFT_KL_THRESHOLD = 0.1   # KL divergence per feature above this = drift alert
_MIN_BATCH_SIZE     = 50     # minimum labeled rows before partial_fit
_ROLLING_WINDOW     = 5000   # rows for HistGB warm refit


class OnlineLearner:
    """Manages incremental model updates and feature drift monitoring.

    Parameters
    ----------
    db_path: str
        Path to the Hogan SQLite database.
    model_dir: str
        Directory where online model checkpoints are saved.
    sgd_path: str
        Path for the SGD model pickle (auto-created on first call).
    histgb_path: str
        Path for the HistGB model pickle.
    drift_threshold: float
        KL divergence per-feature threshold for drift alerts.
    min_batch: int
        Minimum number of new labeled rows to trigger partial_fit.
    """

    def __init__(
        self,
        db_path: str = "data/hogan.db",
        model_dir: str = _DEFAULT_MODEL_DIR,
        drift_threshold: float = _DRIFT_KL_THRESHOLD,
        min_batch: int = _MIN_BATCH_SIZE,
    ) -> None:
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.sgd_path = self.model_dir / "online_sgd.pkl"
        self.histgb_path = self.model_dir / "online_histgb.pkl"
        self.train_stats_path = self.model_dir / "online_train_stats.pkl"
        self.drift_threshold = drift_threshold
        self.min_batch = min_batch

        self._sgd = self._load_sgd()
        self._histgb = self._load_histgb()
        self._train_stats: dict | None = self._load_train_stats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, symbol: str | None = None) -> dict:
        """Run one online update cycle.

        1. Fetches new labeled rows from ``online_training_buffer``.
        2. Checks for feature drift vs. training distribution.
        3. Partial-fits the SGD model.
        4. Optionally warm-restarts HistGB on the rolling window.
        5. Saves updated models to disk.

        Returns a summary dict.
        """
        from hogan_bot.labeler import get_labeled_dataset
        from hogan_bot.storage import get_connection

        conn = get_connection(self.db_path)
        dataset = get_labeled_dataset(conn, symbol=symbol, min_rows=self.min_batch)
        conn.close()

        if dataset is None:
            logger.info("OnlineLearner: insufficient labeled data for %s", symbol or "all")
            return {"status": "skipped", "reason": "insufficient_data"}

        X_all, y_all = dataset
        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.int32)

        # 2. Drift check
        drift_info = self._check_drift(X)

        # 3. SGD partial_fit
        sgd_updated = self._partial_fit_sgd(X, y)

        # 4. HistGB warm refit on rolling window
        histgb_updated = False
        if len(X) >= max(200, self.min_batch * 2):
            # Use last _ROLLING_WINDOW rows
            X_window = X[-_ROLLING_WINDOW:]
            y_window = y[-_ROLLING_WINDOW:]
            histgb_updated = self._warm_fit_histgb(X_window, y_window)

        # 5. Update training statistics for drift monitoring
        self._update_train_stats(X)

        summary = {
            "status": "updated",
            "symbol": symbol,
            "n_rows": len(X),
            "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()),
            "sgd_updated": sgd_updated,
            "histgb_updated": histgb_updated,
            "drift_features": drift_info,
        }
        logger.info(
            "OnlineLearner: symbol=%s rows=%d pos=%d neg=%d drift_cols=%d",
            symbol, len(X), int(y.sum()), int(len(y) - y.sum()), len(drift_info),
        )

        try:
            from hogan_bot.metrics import ONLINE_UPDATES
            ONLINE_UPDATES.labels(model_name="sgd").inc()
        except Exception:
            pass

        return summary

    def compute_shadow_weights(self, symbol: str | None = None, lookback: int = 100) -> dict | None:
        """Compute proposed MetaWeigher weights from recent decision outcomes.

        This is Phase A: shadow mode.  Proposed weights are logged but
        **never applied** to live decisions.  Returns None when
        insufficient data is available.

        The algorithm:
        1. Load recent closed decisions from decision_log (including regime)
        2. Compute per-regime and global signal-to-outcome correlations
        3. Translate correlations to proposed weights
        4. Apply floor/ceiling bounds per agent
        5. Log the full comparison (global + per-regime)
        """
        from hogan_bot.storage import get_connection

        conn = get_connection(self.db_path)
        try:
            query = """
                SELECT tech_action, tech_confidence,
                       sent_bias, sent_strength,
                       macro_regime, macro_risk_on,
                       final_action, realized_pnl,
                       regime
                FROM decision_log
                WHERE realized_pnl IS NOT NULL
            """
            params: list = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            query += " ORDER BY ts_ms DESC LIMIT ?"
            params.append(int(lookback))

            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        if len(rows) < 20:
            logger.debug("Shadow weights: only %d closed decisions, need 20+", len(rows))
            return None

        def _compute_correlations(subset):
            n = len(subset)
            tc = sc = mc = 0.0
            for tech_action, tech_conf, sent_bias, sent_str, _, macro_risk, _, pnl, _ in subset:
                outcome = 1.0 if (pnl or 0.0) > 0 else -1.0
                tc += {"buy": 1.0, "sell": -1.0}.get(tech_action, 0.0) * (tech_conf or 0.5) * outcome
                sc += {"bullish": 1.0, "bearish": -1.0}.get(sent_bias, 0.0) * (sent_str or 0.0) * outcome
                mc += (1.0 if macro_risk else -1.0) * outcome * 0.5
            return {"technical": tc / n, "sentiment": sc / n, "macro": mc / n}

        def _correlations_to_weights(corr):
            raw = {
                "technical": 0.55 + corr["technical"] * 0.3,
                "sentiment": 0.25 + corr["sentiment"] * 0.3,
                "macro": 0.20 + corr["macro"] * 0.3,
            }
            bounds = {"technical": (0.35, 0.75), "sentiment": (0.10, 0.35), "macro": (0.10, 0.30)}
            clamped = {k: max(lo, min(hi, raw[k])) for k, (lo, hi) in bounds.items()}
            total = sum(clamped.values())
            return {k: round(v / total, 4) for k, v in clamped.items()}

        # Global weights
        global_corr = _compute_correlations(rows)
        proposed = _correlations_to_weights(global_corr)

        # Per-regime weights (only for regimes with enough samples)
        regime_buckets: dict[str, list] = {}
        for r in rows:
            regime = r[8] or "unknown"
            regime_buckets.setdefault(regime, []).append(r)

        regime_weights: dict[str, dict] = {}
        min_regime_samples = 10
        for regime, bucket in regime_buckets.items():
            if len(bucket) >= min_regime_samples:
                rc = _compute_correlations(bucket)
                regime_weights[regime] = {
                    "proposed": _correlations_to_weights(rc),
                    "correlations": {k: round(v, 4) for k, v in rc.items()},
                    "n_decisions": len(bucket),
                    "win_rate": round(sum(1 for r in bucket if (r[7] or 0) > 0) / len(bucket), 4),
                }

        logger.info(
            "SHADOW_WEIGHTS global: tech=%.3f sent=%.3f macro=%.3f "
            "(from %d decisions, %d regimes tracked)",
            proposed["technical"], proposed["sentiment"], proposed["macro"],
            len(rows), len(regime_weights),
        )
        for regime, rw in regime_weights.items():
            logger.info(
                "  regime=%s: tech=%.3f sent=%.3f macro=%.3f (n=%d win=%.1f%%)",
                regime, rw["proposed"]["technical"], rw["proposed"]["sentiment"],
                rw["proposed"]["macro"], rw["n_decisions"], rw["win_rate"] * 100,
            )

        try:
            import json
            conn2 = get_connection(self.db_path)
            conn2.execute(
                "INSERT INTO decision_log (ts_ms, symbol, final_action, explanation) "
                "VALUES (?, ?, ?, ?)",
                (
                    int(time.time() * 1000),
                    symbol or "ALL",
                    "shadow_weight_update",
                    json.dumps({
                        "proposed_global": proposed,
                        "correlations_global": {k: round(v, 4) for k, v in global_corr.items()},
                        "by_regime": regime_weights,
                        "n_decisions": len(rows),
                    }),
                ),
            )
            conn2.commit()
            conn2.close()
        except Exception:
            pass

        return proposed

    def predict_proba_sgd(self, features: list[float]) -> float | None:
        """Return the SGD model's probability of up-move for *features*."""
        if self._sgd is None:
            return None
        try:
            x = np.array(features, dtype=np.float32).reshape(1, -1)
            proba = self._sgd.predict_proba(x)[0, 1]
            return float(proba)
        except Exception as exc:
            logger.debug("SGD predict failed: %s", exc)
            return None

    def predict_proba_histgb(self, features: list[float]) -> float | None:
        """Return the HistGB model's probability of up-move for *features*."""
        if self._histgb is None:
            return None
        try:
            x = np.array(features, dtype=np.float32).reshape(1, -1)
            proba = self._histgb.predict_proba(x)[0, 1]
            return float(proba)
        except Exception as exc:
            logger.debug("HistGB predict failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _partial_fit_sgd(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler

            if self._sgd is None:
                self._sgd = SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=1,
                    tol=None,
                    warm_start=True,
                    random_state=42,
                )
            self._sgd.partial_fit(X, y, classes=[0, 1])
            with open(self.sgd_path, "wb") as f:
                pickle.dump(self._sgd, f)
            return True
        except Exception as exc:
            logger.warning("SGD partial_fit failed: %s", exc)
            return False

    def _warm_fit_histgb(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            if self._histgb is None:
                self._histgb = HistGradientBoostingClassifier(
                    max_iter=50,
                    max_depth=4,
                    learning_rate=0.1,
                    warm_start=True,
                    random_state=42,
                )
            else:
                self._histgb.max_iter += 10  # add 10 more trees on warm restart
            self._histgb.fit(X, y)
            with open(self.histgb_path, "wb") as f:
                pickle.dump(self._histgb, f)
            return True
        except Exception as exc:
            logger.warning("HistGB warm fit failed: %s", exc)
            return False

    def _check_drift(self, X_live: np.ndarray) -> list[int]:
        """Return indices of features with KL divergence above threshold.

        Uses a simple histogram-based KL divergence estimate.
        """
        if self._train_stats is None:
            return []

        train_mean = self._train_stats.get("mean")
        train_std = self._train_stats.get("std")
        if train_mean is None or train_std is None:
            return []

        live_mean = X_live.mean(axis=0)
        live_std = X_live.std(axis=0) + 1e-8

        # Approximate KL(live || train) for Gaussian distributions:
        # KL = log(σ_t/σ_l) + (σ_l² + (μ_l - μ_t)²) / (2σ_t²) - 0.5
        sigma_t = np.array(train_std) + 1e-8
        sigma_l = live_std
        mu_diff = live_mean - np.array(train_mean)
        kl = (np.log(sigma_t / sigma_l)
              + (sigma_l**2 + mu_diff**2) / (2 * sigma_t**2)
              - 0.5)

        drifted = [int(i) for i, v in enumerate(kl) if v > self.drift_threshold]
        if drifted:
            logger.warning(
                "Feature drift detected: %d features above KL threshold %.3f: %s",
                len(drifted), self.drift_threshold, drifted[:10],
            )
            try:
                from hogan_bot.metrics import FEATURE_DRIFT
                # Use "unknown" as symbol if we can't determine it
                FEATURE_DRIFT.labels(symbol="unknown").inc(len(drifted))
            except Exception:
                pass
        return drifted

    def _update_train_stats(self, X: np.ndarray) -> None:
        """Save running mean/std of training features for future drift checks."""
        if self._train_stats is None:
            self._train_stats = {}

        existing_n = self._train_stats.get("n", 0)
        existing_mean = np.array(self._train_stats.get("mean", X.mean(axis=0)))
        new_n = existing_n + len(X)
        # Incremental mean update
        new_mean = (existing_mean * existing_n + X.sum(axis=0)) / new_n
        self._train_stats = {
            "n": new_n,
            "mean": new_mean.tolist(),
            "std": X.std(axis=0).tolist(),
            "updated_at": time.time(),
        }
        with open(self.train_stats_path, "wb") as f:
            pickle.dump(self._train_stats, f)

    def _load_sgd(self):
        if self.sgd_path.exists():
            try:
                with open(self.sgd_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def _load_histgb(self):
        if self.histgb_path.exists():
            try:
                with open(self.histgb_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def _load_train_stats(self) -> dict | None:
        if self.train_stats_path.exists():
            try:
                with open(self.train_stats_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# CLI daemon — call update() every N seconds
# ---------------------------------------------------------------------------
def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Online learner daemon")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--symbol", default=None, help="Filter to one symbol (default: all)")
    p.add_argument("--interval", type=float, default=3600.0,
                   help="Update interval in seconds (default 3600 = 1h)")
    p.add_argument("--once", action="store_true", help="Run once and exit")
    p.add_argument("--min-batch", type=int, default=_MIN_BATCH_SIZE)
    args = p.parse_args()

    learner = OnlineLearner(db_path=args.db, min_batch=args.min_batch)

    while True:
        result = learner.update(symbol=args.symbol)
        logger.info("Update result: %s", result)
        if args.once:
            break
        logger.info("Sleeping %.0fs until next update.", args.interval)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    _main()
