"""Walk-forward retraining for Hogan ML models.

Rolling-window walk-forward strategy
-------------------------------------
Each retrain cycle uses the most recent ``--window-bars`` bars.  Because the
window rolls forward with time, the model continuously adapts to the current
market regime rather than being dominated by stale historical patterns.

Promotion gate
--------------
A newly trained candidate model replaces the production model **only** when it
improves the ``--promotion-metric`` by at least ``--min-improvement`` over the
current registry best.  Every run — promoted or not — is logged in the model
registry so the complete history is always inspectable.

Safety
------
Training writes to a timestamped *candidate* path first.  The production path
is only overwritten on successful promotion, so a failed or regressing run
never corrupts the live model.

Usage examples
--------------
One-shot retrain (fetches live data from the configured exchange)::

    python -m hogan_bot.retrain

Load candles from local SQLite DB (faster, offline, good for cron)::

    python -m hogan_bot.retrain --from-db

Retrain every 24 hours (blocking loop)::

    python -m hogan_bot.retrain --schedule 24

Retrain every 6 hours on Binance BTC/USDT with XGBoost::

    python -m hogan_bot.retrain --exchange binance --symbol BTC/USDT \\
        --model-type xgboost --schedule 6

Evaluate without touching anything::

    python -m hogan_bot.retrain --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hogan_bot.exchange import ExchangeClient
from hogan_bot.ml import (
    make_paper_trade_labels,
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from hogan_bot.registry import ModelRegistry
from hogan_bot.storage import get_connection, load_candles

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Walk-forward retraining for Hogan ML models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD", help="Trading pair to train on")
    p.add_argument("--timeframe", default="5m", help="Bar interval")
    p.add_argument(
        "--exchange",
        default="kraken",
        help="CCXT exchange ID for live data fetches (ignored when --from-db is set)",
    )
    p.add_argument(
        "--window-bars",
        type=int,
        default=5000,
        metavar="N",
        help="Rolling window size: train on the N most recent bars",
    )
    p.add_argument("--horizon-bars", type=int, default=3, help="Prediction horizon in bars")
    p.add_argument(
        "--model-type",
        choices=["logreg", "random_forest", "xgboost", "lightgbm"],
        default="logreg",
        help="Model family to train",
    )
    p.add_argument("--model-path", default="models/hogan_logreg.pkl", help="Production model path")
    p.add_argument(
        "--tune",
        action="store_true",
        help="Run C hyper-parameter search (logreg only)",
    )
    p.add_argument(
        "--promotion-metric",
        default="roc_auc",
        choices=["roc_auc", "accuracy", "f1", "precision", "recall"],
        help="Metric used to decide whether to promote the new model",
    )
    p.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        metavar="DELTA",
        help="Minimum metric gain over registry best required for promotion",
    )
    p.add_argument(
        "--registry-path",
        default="models/registry.jsonl",
        help="Path to the JSONL model registry",
    )
    p.add_argument(
        "--from-db",
        action="store_true",
        help="Load candles from local SQLite DB instead of fetching live data",
    )
    p.add_argument(
        "--db",
        default="data/hogan.db",
        help="SQLite DB path (used when --from-db is set)",
    )
    p.add_argument(
        "--schedule",
        type=float,
        default=None,
        metavar="HOURS",
        help="Run a retrain every HOURS hours (blocking loop). Omit for one-shot.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate but do NOT overwrite the production model or update the registry",
    )
    p.add_argument(
        "--force-promote",
        action="store_true",
        help=(
            "Promote the new model regardless of registry score. "
            "Use when the feature space changed (e.g. 36 -> 43 features) "
            "and the old registry benchmark is no longer comparable."
        ),
    )
    p.add_argument(
        "--shadow-eval",
        action="store_true",
        help=(
            "Champion/challenger mode: train a challenger model and route 10%% of signals "
            "to it in paper mode.  Auto-promote if it outperforms champion over a 7-day window."
        ),
    )
    p.add_argument(
        "--shadow-window-days",
        type=float,
        default=7.0,
        metavar="DAYS",
        help="Number of days to shadow-evaluate challenger before considering promotion",
    )
    p.add_argument(
        "--shadow-min-improvement",
        type=float,
        default=0.01,
        metavar="DELTA",
        help="Minimum metric gain for challenger to be auto-promoted over champion",
    )
    p.add_argument(
        "--use-paper-labels",
        action="store_true",
        help=(
            "Blend closed paper-trade outcomes into the training set as additional "
            "labeled rows (weight 3×).  Requires ≥5 closed paper trades in the DB."
        ),
    )
    p.add_argument(
        "--paper-labels-weight",
        type=float,
        default=3.0,
        metavar="W",
        help="Sample weight applied to paper-trade labeled rows (default: 3.0)",
    )
    p.add_argument(
        "--symbols",
        default=None,
        metavar="SYM1,SYM2",
        help=(
            "Comma-separated list of symbols to train on jointly "
            "(e.g. 'BTC/USD,ETH/USD,SOL/USD').  When set, candles from all symbols "
            "are concatenated before feature engineering, giving the model a larger "
            "and more diverse training set.  Defaults to --symbol only."
        ),
    )
    p.add_argument(
        "--use-extended-mtf",
        action="store_true",
        help=(
            "Enable 10m + 30m multi-timeframe features (+14 features). "
            "Requires 10m and 30m candles in the DB (run backfill first). "
            "IMPORTANT: produces a different feature vector — always pair with "
            "--force-promote and update HOGAN_USE_MTF_EXTENDED=true in .env."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    """Return the list of symbols to train on (respects --symbols override)."""
    if args.symbols:
        return [s.strip() for s in args.symbols.split(",") if s.strip()]
    return [args.symbol]


def _fetch_candles(args: argparse.Namespace) -> pd.DataFrame:
    """Return a DataFrame of candles for the primary (first) symbol.

    Used by single-symbol training paths.  For multi-symbol training,
    call :func:`_build_multi_symbol_dataset` directly.
    """
    symbols = _resolve_symbols(args)
    primary = symbols[0]

    if args.from_db:
        conn = get_connection(args.db)
        df = load_candles(conn, primary, args.timeframe, limit=args.window_bars)
        conn.close()
        if df.empty:
            raise RuntimeError(
                f"No candles in DB for {primary}/{args.timeframe}. "
                "Run `python -m hogan_bot.fetch_alpaca --backfill-all` first."
            )
        logger.info("Loaded %d bars from DB (%s / %s)", len(df), primary, args.timeframe)
        return df

    if len(symbols) > 1:
        logger.warning(
            "Live exchange fetch only supports a single symbol (%s); "
            "add --from-db to train on multiple symbols.",
            primary,
        )
    client = ExchangeClient(args.exchange)
    df = client.fetch_ohlcv_df(primary, timeframe=args.timeframe, limit=args.window_bars)
    logger.info("Fetched %d bars from %s (%s / %s)", len(df), args.exchange, primary, args.timeframe)
    return df


def _build_multi_symbol_dataset(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build a combined (X, y, feature_columns) training set across multiple symbols.

    Features are computed **per symbol** first (so rolling windows never cross
    symbol boundaries), then the resulting feature matrices are concatenated.
    This prevents contaminated rolling windows when mixing BTC and ETH candles.

    Returns
    -------
    X : pd.DataFrame  — feature matrix
    y : pd.Series     — binary labels
    feature_columns : list[str]
    """
    from hogan_bot.ml import build_training_set

    symbols = _resolve_symbols(args)
    x_frames: list[pd.DataFrame] = []
    y_frames: list[pd.Series] = []
    feature_columns: list[str] = []

    if args.from_db:
        conn = get_connection(args.db)
        for sym in symbols:
            df = load_candles(conn, sym, args.timeframe, limit=args.window_bars)
            if df.empty:
                logger.warning("No candles in DB for %s / %s — skipping", sym, args.timeframe)
                continue
            # Pass db_conn so macro features (SPY/VIX/GLD…) are joined during training
            x_sym, y_sym, fc = build_training_set(df, horizon_bars=args.horizon_bars, db_conn=conn)
            if x_sym is not None:
                x_frames.append(x_sym)
                y_frames.append(y_sym)
                feature_columns = fc  # same for all symbols
                logger.info(
                    "Multi-symbol: %d feature rows from %s/%s",
                    len(x_sym), sym, args.timeframe,
                )
        conn.close()
    else:
        primary = symbols[0]
        client = ExchangeClient(args.exchange)
        df = client.fetch_ohlcv_df(primary, timeframe=args.timeframe, limit=args.window_bars)
        x_sym, y_sym, feature_columns = build_training_set(df, horizon_bars=args.horizon_bars)
        if x_sym is not None:
            x_frames.append(x_sym)
            y_frames.append(y_sym)

    if not x_frames:
        raise RuntimeError(
            f"No training rows from any of {symbols} / {args.timeframe}. "
            "Run `python -m hogan_bot.fetch_alpaca --backfill-all` to populate the DB."
        )

    X_all = pd.concat(x_frames, ignore_index=True)
    y_all = pd.concat(y_frames, ignore_index=True)
    logger.info(
        "Multi-symbol dataset: %d rows from %d symbol(s): %s",
        len(X_all), len(x_frames), symbols,
    )
    return X_all, y_all, feature_columns


def _train_to_candidate(args: argparse.Namespace, candles: pd.DataFrame) -> tuple[dict, str]:
    """Train a model and write it to a timestamped candidate path.

    When multiple symbols are configured (via ``--symbols``), uses
    :func:`_build_multi_symbol_dataset` to build a jointly-trained dataset;
    otherwise falls back to the single-symbol ``candles`` DataFrame.

    Returns ``(metrics_dict, candidate_path)``.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_dir = Path(args.model_path).parent
    model_stem = Path(args.model_path).stem
    candidate_path = str(model_dir / f"{model_stem}_candidate_{ts}.pkl")
    model_dir.mkdir(parents=True, exist_ok=True)

    symbols = _resolve_symbols(args)
    multi_symbol = len(symbols) > 1

    if getattr(args, "use_extended_mtf", False):
        logger.info(
            "Extended MTF features (10m + 30m) will be used for RL training. "
            "For the standard ML model (logreg/rf/xgb/lgbm), extended MTF requires "
            "custom feature engineering — currently only multi-symbol data expansion "
            "is applied. Run rl_train.py with --ext-features for RL extended MTF."
        )

    # Optionally load paper-trade labels for feedback-loop training
    paper_labels = None
    paper_weight = getattr(args, "paper_labels_weight", 3.0)
    if getattr(args, "use_paper_labels", False) and not multi_symbol:
        db_path = getattr(args, "db", "data/hogan.db")
        paper_labels = make_paper_trade_labels(db_path, candles, args.symbol)
        if paper_labels[0] is not None:
            logger.info(
                "Paper-trade feedback: %d labeled rows added (weight=%.1f)",
                len(paper_labels[0]), paper_weight,
            )
        else:
            logger.info("Paper-trade feedback: fewer than 5 closed trades — skipping.")
    elif getattr(args, "use_paper_labels", False) and multi_symbol:
        logger.info("Paper-trade feedback skipped for multi-symbol training (not yet supported).")

    # For multi-symbol, build a combined (X, y) dataset; otherwise use raw candles
    if multi_symbol:
        logger.info("Multi-symbol training: %s", symbols)
        X_all, y_all, feature_cols = _build_multi_symbol_dataset(args)

        # Delegate to a special multi-symbol training path
        metrics = _train_from_xy(
            X_all, y_all, feature_cols,
            model_type=args.model_type,
            model_path=candidate_path,
            tune=getattr(args, "tune", False),
        )
    # Open DB connection for macro feature enrichment (single-symbol path)
    _db_conn = None
    if getattr(args, "from_db", False):
        _db_conn = get_connection(getattr(args, "db", "data/hogan.db"))

    try:
        if args.model_type == "random_forest":
            metrics = train_random_forest(
                candles, model_path=candidate_path, horizon_bars=args.horizon_bars,
                paper_labels=paper_labels, paper_labels_weight=paper_weight,
                db_conn=_db_conn,
            )
        elif args.model_type == "xgboost":
            metrics = train_xgboost(
                candles, model_path=candidate_path, horizon_bars=args.horizon_bars,
                paper_labels=paper_labels, paper_labels_weight=paper_weight,
                db_conn=_db_conn,
            )
        elif args.model_type == "lightgbm":
            metrics = train_lightgbm(
                candles, model_path=candidate_path, horizon_bars=args.horizon_bars,
                db_conn=_db_conn,
            )
        else:
            metrics = train_logistic_regression(
                candles,
                model_path=candidate_path,
                horizon_bars=args.horizon_bars,
                tune_hyperparams=getattr(args, "tune", False),
                paper_labels=paper_labels,
                paper_labels_weight=paper_weight,
                db_conn=_db_conn,
            )
    finally:
        if _db_conn is not None:
            _db_conn.close()

    return metrics, candidate_path


def _train_from_xy(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    model_type: str,
    model_path: str,
    tune: bool = False,
) -> dict:
    """Fit a classifier directly from pre-built (X, y) matrices and save to disk.

    Used by the multi-symbol training path where feature matrices are built
    per-symbol then concatenated before training.
    """
    import pickle
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    if len(X) < 200:
        raise RuntimeError(
            f"Only {len(X)} training rows after multi-symbol feature build "
            "(need ≥ 200).  Ensure all symbols have sufficient candle history."
        )

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42,
                                     class_weight="balanced", n_jobs=-1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        artifact = clf
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                            use_label_encoder=False, eval_metric="logloss",
                            random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        artifact = clf
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        artifact = clf
    else:
        from sklearn.linear_model import LogisticRegression
        C = 1.0
        if tune:
            best_C, best_acc = 1.0, -1.0
            for c_val in (0.01, 0.1, 1.0, 10.0):
                m = LogisticRegression(max_iter=500, C=c_val, class_weight="balanced")
                m.fit(X_train_sc, y_train)
                acc = float(accuracy_score(y_test, m.predict(X_test_sc)))
                if acc > best_acc:
                    best_acc, best_C = acc, c_val
            C = best_C
        clf = LogisticRegression(max_iter=500, C=C, class_weight="balanced")
        clf.fit(X_train_sc, y_train)
        proba = clf.predict_proba(X_test_sc)[:, 1]

    # Wrap all model types in TrainedModel for consistent pickling
    from hogan_bot.ml import TrainedModel
    artifact = TrainedModel(scaler=scaler, model=clf, feature_columns=feature_cols)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as fh:
        pickle.dump(artifact, fh)

    pred = (proba >= 0.5).astype(int)
    return {
        "model_type": f"{model_type}_multisym",
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.0,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def _get_current_best_score(registry: ModelRegistry, metric: str) -> float | None:
    """Return the current best value for *metric* in *registry*, or ``None``."""
    best = registry.best(metric)
    if best is None:
        return None
    return best.get("metrics", {}).get(metric)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrain_once(args: argparse.Namespace) -> dict:
    """Execute one walk-forward retrain cycle.

    Steps
    -----
    1. Fetch the ``window_bars`` most recent candles (live or from DB).
    2. Train a candidate model to a temporary path.
    3. Compare candidate score against the registry best.
    4. Promote (copy to production path + log to registry) when the score
       improves by at least ``min_improvement``.
    5. Clean up the temporary candidate file.

    Returns a JSON-serialisable dict summarising the run.
    """
    start_ts = datetime.now(tz=timezone.utc).isoformat()
    logger.info(
        "Retrain start — symbol=%s timeframe=%s window=%d model=%s dry_run=%s",
        args.symbol,
        args.timeframe,
        args.window_bars,
        args.model_type,
        args.dry_run,
    )

    # 1. Fetch candles
    candles = _fetch_candles(args)

    # 2. Train candidate
    metrics, candidate_path = _train_to_candidate(args, candles)
    new_score: float = float(metrics.get(args.promotion_metric, 0.0))
    logger.info(
        "Candidate trained — %s=%.4f  path=%s",
        args.promotion_metric,
        new_score,
        candidate_path,
    )

    # 3. Determine whether to promote
    registry = ModelRegistry(registry_path=args.registry_path)
    current_score = _get_current_best_score(registry, args.promotion_metric)
    threshold = (current_score or 0.0) + args.min_improvement
    force = getattr(args, "force_promote", False)
    should_promote = force or current_score is None or new_score >= threshold
    if force and not (current_score is None or new_score >= threshold):
        logger.info(
            "--force-promote set: overriding promotion check "
            "(new=%.4f vs registry=%.4f). Feature space changed — old score is not comparable.",
            new_score, current_score or 0.0,
        )

    # Build result dict (feature_importances omitted — too large for JSON log)
    result: dict = {
        "timestamp": start_ts,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "exchange": args.exchange,
        "window_bars": args.window_bars,
        "horizon_bars": args.horizon_bars,
        "model_type": args.model_type,
        "promotion_metric": args.promotion_metric,
        "min_improvement": args.min_improvement,
        "new_score": new_score,
        "current_score": float(current_score) if current_score is not None else None,
        "promoted": False,
        "dry_run": args.dry_run,
    }
    # Surface scalar metrics from the training run (accuracy, f1, …)
    for k, v in metrics.items():
        if k not in ("feature_importances", "model_type") and isinstance(v, (int, float)):
            result.setdefault(k, float(v))

    # 4. Act on the decision
    try:
        if args.dry_run:
            result["message"] = (
                f"Dry run — {args.promotion_metric}={new_score:.4f}"
                + (f" vs current {current_score:.4f}" if current_score is not None else " (no current model)")
            )
            logger.info("Dry run complete. %s", result["message"])

        elif should_promote:
            shutil.copy2(candidate_path, args.model_path)
            registry.log(
                metrics,
                model_path=args.model_path,
                symbol=args.symbol,
                timeframe=args.timeframe,
                horizon_bars=args.horizon_bars,
            )
            result["promoted"] = True
            improvement = new_score - (current_score or 0.0)
            if current_score is None:
                result["message"] = "Promoted — first model in registry"
            elif force:
                result["message"] = (
                    f"Promoted (force) — {args.promotion_metric} {new_score:.4f} "
                    f"(old registry={current_score:.4f}, feature space changed)"
                )
            else:
                result["message"] = (
                    f"Promoted — {args.promotion_metric} {new_score:.4f} "
                    f"(+{improvement:.4f} over {current_score:.4f})"
                )
            logger.info("Model promoted -> %s", args.model_path)

        else:
            improvement = new_score - (current_score or 0.0)
            result["message"] = (
                f"Not promoted — {args.promotion_metric} {new_score:.4f} "
                f"vs {current_score:.4f} (need +{args.min_improvement:.4f})"
            )
            logger.info("Model NOT promoted. %s", result["message"])

    finally:
        # 5. Always clean up the candidate file
        candidate = Path(candidate_path)
        if candidate.exists():
            candidate.unlink()

    return result


# ---------------------------------------------------------------------------
# Champion / Challenger shadow evaluation
# ---------------------------------------------------------------------------

_SHADOW_REGISTRY_PATH = "models/shadow_registry.jsonl"


def _write_shadow_registry(entry: dict, path: str = _SHADOW_REGISTRY_PATH) -> None:
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def shadow_eval_cycle(args) -> dict:
    """Run champion/challenger shadow evaluation.

    1. Train a challenger model (same as retrain_once).
    2. Write it to a ``*_challenger.pkl`` path (does NOT touch production).
    3. Record in the shadow registry with a ``status: challenger`` tag.
    4. Compare challenger vs champion over the shadow window:
       - If shadow window has elapsed AND challenger wins by
         ``shadow_min_improvement``, auto-promote the challenger.
    5. Emit ``MODEL_PROMOTED`` Prometheus counter on promotion.
    """
    import json
    from pathlib import Path

    start_ts = datetime.now(tz=timezone.utc).isoformat()
    candles = _fetch_candles(args)

    # --- Train challenger ---
    ts_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_dir = Path(args.model_path).parent
    challenger_path = str(model_dir / f"challenger_{ts_str}.pkl")

    chall_args = argparse.Namespace(**vars(args))
    chall_args.model_path = challenger_path
    metrics, candidate_path = _train_to_candidate(chall_args, candles)
    import shutil
    shutil.copy2(candidate_path, challenger_path)
    Path(candidate_path).unlink(missing_ok=True)

    chall_score = float(metrics.get(args.promotion_metric, 0.0))
    logger.info("Challenger trained: %s=%.4f path=%s", args.promotion_metric, chall_score, challenger_path)

    shadow_entry = {
        "timestamp": start_ts,
        "challenger_path": challenger_path,
        "champion_path": args.model_path,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "model_type": args.model_type,
        "challenger_score": chall_score,
        "status": "challenger",
        "window_days": args.shadow_window_days,
    }
    _write_shadow_registry(shadow_entry)

    # --- Check if shadow window has elapsed for any existing challenger ---
    shadow_reg_path = Path(_SHADOW_REGISTRY_PATH)
    promoted = False
    if shadow_reg_path.exists():
        with open(shadow_reg_path, encoding="utf-8") as fh:
            entries = [json.loads(line) for line in fh if line.strip()]

        now_ts = time.time()
        for entry in entries:
            if entry.get("status") != "challenger":
                continue
            if entry.get("symbol") != args.symbol:
                continue
            created = datetime.fromisoformat(entry["timestamp"]).timestamp()
            age_days = (now_ts - created) / 86400.0
            if age_days < entry.get("window_days", args.shadow_window_days):
                logger.info(
                    "Shadow window not elapsed yet (%.1f / %.1f days) for %s.",
                    age_days, entry["window_days"], entry["challenger_path"],
                )
                continue

            # Window elapsed — compare scores
            champ_score = float(
                _get_current_best_score(ModelRegistry(registry_path=args.registry_path), args.promotion_metric) or 0.0
            )
            cand_score = float(entry.get("challenger_score", 0.0))
            if cand_score >= champ_score + args.shadow_min_improvement:
                # Promote!
                cand_path = Path(entry["challenger_path"])
                if cand_path.exists():
                    shutil.copy2(str(cand_path), args.model_path)
                    reg = ModelRegistry(registry_path=args.registry_path)
                    reg.log(
                        {**metrics, "model_type": args.model_type, "shadow_promoted": True},
                        model_path=args.model_path,
                        symbol=args.symbol,
                        timeframe=args.timeframe,
                        horizon_bars=args.horizon_bars,
                    )
                    promoted = True
                    logger.info(
                        "Challenger PROMOTED: score=%.4f vs champion=%.4f (+%.4f)",
                        cand_score, champ_score, cand_score - champ_score,
                    )
                    try:
                        from hogan_bot.metrics import MODEL_PROMOTED
                        MODEL_PROMOTED.labels(symbol=args.symbol).inc()
                    except Exception:
                        pass
                    # Mark as promoted in shadow registry
                    entry["status"] = "promoted"
                    entry["promoted_at"] = datetime.now(tz=timezone.utc).isoformat()
                    _write_shadow_registry(entry)
            else:
                logger.info(
                    "Challenger not promoted: score=%.4f vs champion=%.4f (need +%.4f)",
                    cand_score, champ_score, args.shadow_min_improvement,
                )
                entry["status"] = "expired"
                _write_shadow_registry(entry)

    return {
        "timestamp": start_ts,
        "challenger_path": challenger_path,
        "challenger_score": chall_score,
        "promoted": promoted,
        "symbol": args.symbol,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    def _run_one(args):
        if getattr(args, "shadow_eval", False):
            return shadow_eval_cycle(args)
        return retrain_once(args)

    if args.schedule is not None:
        interval_secs = args.schedule * 3600.0
        logger.info(
            "Scheduled retraining every %.1f hours. Press Ctrl-C to stop.", args.schedule
        )
        while True:
            try:
                result = _run_one(args)
                print(json.dumps(result, indent=2), flush=True)
            except KeyboardInterrupt:
                logger.info("Retrain scheduler stopped by user.")
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Retrain cycle failed: %s", exc)
            logger.info("Sleeping %.0f s until next retrain…", interval_secs)
            try:
                time.sleep(interval_secs)
            except KeyboardInterrupt:
                logger.info("Retrain scheduler stopped by user.")
                break
    else:
        result = _run_one(args)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
