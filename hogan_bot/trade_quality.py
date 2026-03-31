"""Trade Quality Classifier — replaces directional ML with setup quality prediction.

Instead of predicting "will price go up?" (AUC ~0.56, useless), this model
predicts "will this trade setup exit profitably via signal, or bleed out
via max_hold_time?" — a fundamentally different and more separable target.

Training data comes from backtest closed trades with entry_context features.
The label is: 1 = good exit (signal, take_profit, buy_signal), 0 = bad exit
(max_hold_time, trailing_stop, breakeven_stop).

Usage::

    # Generate training data and train
    python -m hogan_bot.trade_quality --db data/hogan.db --train

    # Walk-forward validate
    python -m hogan_bot.trade_quality --db data/hogan.db --validate

    # Inspect model
    python -m hogan_bot.trade_quality --db data/hogan.db --inspect
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GOOD_EXIT_REASONS = frozenset({
    "signal", "take_profit", "short_take_profit", "buy_signal",
    "proactive_momentum_exhaustion", "proactive_give_back",
})

FEATURE_COLUMNS = [
    "tech_confidence",
    "final_confidence",
    "up_prob",
    "atr_pct",
    "regime_confidence",
    "vol_ratio",
    "quality_scale",
    "ranging_scale",
    "pullback_scale",
    "momentum_scale",
    "conf_scale",
    "whipsaw_count",
    "local_range_position",
    "run_up_before_entry",
    "spread_est",
    "regime_trending_up",
    "regime_trending_down",
    "regime_ranging",
    "regime_volatile",
    "side_short",
]

DEFAULT_MODEL_PATH = "models/trade_quality.pkl"


@dataclass
class TradeQualityModel:
    model: object
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    threshold: float = 0.40
    scaler: object | None = None


def _build_feature_row_from_context(entry_context: dict, trade: dict) -> dict:
    """Build a flat feature dict from entry_context + trade-level enrichment."""
    ctx = entry_context
    regime = ctx.get("regime") or trade.get("entry_regime") or "unknown"
    return {
        "tech_confidence": ctx.get("tech_confidence") or 0.0,
        "final_confidence": ctx.get("final_confidence") or 0.0,
        "up_prob": ctx.get("up_prob") if ctx.get("up_prob") is not None else 0.5,
        "atr_pct": ctx.get("atr_pct") or 0.0,
        "regime_confidence": ctx.get("regime_confidence") or trade.get("regime_confidence") or 0.0,
        "vol_ratio": ctx.get("vol_ratio") or 0.0,
        "quality_scale": ctx.get("quality_scale", 1.0),
        "ranging_scale": ctx.get("ranging_scale", 1.0),
        "pullback_scale": ctx.get("pullback_scale", 1.0),
        "momentum_scale": ctx.get("momentum_scale", 1.0),
        "conf_scale": ctx.get("conf_scale", 1.0),
        "whipsaw_count": ctx.get("whipsaw_count", 0),
        "local_range_position": trade.get("local_range_position", 0.5),
        "run_up_before_entry": trade.get("run_up_before_entry", 0.0),
        "spread_est": ctx.get("spread_est") or trade.get("spread_est", 0.0),
        "regime_trending_up": 1.0 if regime == "trending_up" else 0.0,
        "regime_trending_down": 1.0 if regime == "trending_down" else 0.0,
        "regime_ranging": 1.0 if regime == "ranging" else 0.0,
        "regime_volatile": 1.0 if regime == "volatile" else 0.0,
        "side_short": 1.0 if (ctx.get("side") or trade.get("side")) == "short" else 0.0,
    }


def build_feature_row_from_live(
    *,
    tech_confidence: float = 0.0,
    final_confidence: float = 0.0,
    up_prob: float | None = None,
    atr_pct: float = 0.0,
    regime: str = "unknown",
    regime_confidence: float = 0.0,
    vol_ratio: float = 0.0,
    quality_scale: float = 1.0,
    ranging_scale: float = 1.0,
    pullback_scale: float = 1.0,
    momentum_scale: float = 1.0,
    conf_scale: float = 1.0,
    whipsaw_count: int = 0,
    local_range_position: float = 0.5,
    run_up_before_entry: float = 0.0,
    spread_est: float = 0.0,
    side: str = "long",
) -> list[float]:
    """Build feature vector for live inference. Returns values in FEATURE_COLUMNS order."""
    return [
        tech_confidence,
        final_confidence,
        up_prob if up_prob is not None else 0.5,
        atr_pct,
        regime_confidence,
        vol_ratio,
        quality_scale,
        ranging_scale,
        pullback_scale,
        momentum_scale,
        conf_scale,
        float(whipsaw_count),
        local_range_position,
        run_up_before_entry,
        spread_est,
        1.0 if regime == "trending_up" else 0.0,
        1.0 if regime == "trending_down" else 0.0,
        1.0 if regime == "ranging" else 0.0,
        1.0 if regime == "volatile" else 0.0,
        1.0 if side == "short" else 0.0,
    ]


def predict_trade_quality(
    features: list[float],
    model: TradeQualityModel,
) -> float:
    """Return probability that this trade setup will exit profitably."""
    x = np.array(features, dtype=np.float32).reshape(1, -1)
    try:
        proba = model.model.predict_proba(x)[0, 1]
        return float(min(max(proba, 0.0), 1.0))
    except Exception as exc:
        logger.warning("trade_quality predict failed: %s", exc)
        return 0.5


def generate_training_data(
    db_path: str = "data/hogan.db",
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    limit: int = 50000,
) -> tuple[pd.DataFrame, pd.Series] | None:
    """Run a backtest and extract labeled training data from closed trades."""
    from hogan_bot.backtest import run_backtest_on_candles
    from hogan_bot.champion import apply_champion_mode
    from hogan_bot.config import load_config
    from hogan_bot.ml import load_model
    from hogan_bot.profiles import CANONICAL_PROFILE, apply_profile
    from hogan_bot.storage import get_connection, load_candles

    conn = get_connection(db_path)
    candles = load_candles(conn, symbol, timeframe, limit=limit)
    conn.close()

    if candles.empty or len(candles) < 500:
        logger.warning("Insufficient candles for training: %d", len(candles))
        return None

    config = load_config()
    config, cli_ov = apply_profile(config, CANONICAL_PROFILE)
    config = apply_champion_mode(config)

    ml_model = None
    try:
        ml_model = load_model(config.ml_model_path)
    except Exception:
        pass

    _bt_kwargs = dict(
        symbol=symbol,
        starting_balance_usd=config.starting_balance_usd,
        aggressive_allocation=config.aggressive_allocation,
        max_risk_per_trade=config.max_risk_per_trade,
        max_drawdown=config.max_drawdown,
        short_ma_window=config.short_ma_window,
        long_ma_window=config.long_ma_window,
        volume_window=config.volume_window,
        volume_threshold=config.volume_threshold,
        fee_rate=config.fee_rate,
        timeframe=timeframe,
        ml_model=ml_model,
        ml_buy_threshold=config.ml_buy_threshold,
        ml_sell_threshold=config.ml_sell_threshold,
        trailing_stop_pct=config.trailing_stop_pct,
        take_profit_pct=config.take_profit_pct,
        max_hold_hours=config.max_hold_hours,
        enable_shorts=cli_ov.get("enable_shorts", getattr(config, "enable_shorts", True)),
        short_max_hold_hours=config.short_max_hold_hours,
        slippage_bps=5.0,
        execution_mode="next_open",
        use_ml_as_sizer=config.use_ml_as_sizer,
        db_path=db_path,
        trail_activation_pct=config.trail_activation_pct,
        breakeven_stop_pct=getattr(config, "breakeven_stop_pct", 0.015),
        enable_pullback_gate=cli_ov.get("enable_pullback_gate", False),
        use_policy_core=True,
    )
    result = run_backtest_on_candles(candles, **_bt_kwargs)

    from hogan_bot.backtest import enrich_trades_with_entry_context
    enrich_trades_with_entry_context(candles, result.closed_trades)

    return _trades_to_dataset(result.closed_trades)


def _trades_to_dataset(
    closed_trades: list[dict],
) -> tuple[pd.DataFrame, pd.Series] | None:
    """Convert closed trades with entry_context into (X, y) training data."""
    rows = []
    labels = []
    for trade in closed_trades:
        ctx = trade.get("entry_context", {})
        if not ctx:
            continue
        close_reason = trade.get("close_reason", "unknown")
        pnl_pct = trade.get("pnl_pct", 0.0)
        is_good_reason = close_reason in GOOD_EXIT_REASONS
        is_proactive_profitable = (
            close_reason.startswith("proactive_") and pnl_pct > 0.2
        )
        label = 1 if (is_good_reason or is_proactive_profitable) else 0

        features = _build_feature_row_from_context(ctx, trade)
        rows.append(features)
        labels.append(label)

    if len(rows) < 20:
        logger.warning("Only %d trades with entry_context, need 20+", len(rows))
        return None

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = pd.Series(labels, name="good_exit")
    X = X.fillna(0.0)

    logger.info(
        "Training data: %d trades, %d good exits (%.1f%%), %d bad exits",
        len(y), y.sum(), y.mean() * 100, len(y) - y.sum(),
    )
    return X, y


def train_trade_quality_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_path: str = DEFAULT_MODEL_PATH,
    embargo_bars: int = 24,
) -> dict:
    """Train a LightGBM trade quality classifier and save."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    n = len(X)
    test_n = max(1, int(n * 0.2))
    test_start = max(1, n - test_n)
    train_end = max(1, test_start - max(0, embargo_bars))
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]
    if len(X_train) < 50 or len(X_test) < 10:
        raise ValueError(
            f"Insufficient temporal split after embargo: "
            f"train={len(X_train)} test={len(X_test)} embargo={embargo_bars}"
        )

    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.05, random_state=42,
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = 0.5

    metrics = {
        "model_type": type(model).__name__,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": len(FEATURE_COLUMNS),
        "embargo_bars": int(embargo_bars),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(auc, 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "pos_rate_train": round(float(y_train.mean()), 4),
        "pos_rate_test": round(float(y_test.mean()), 4),
    }

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        metrics["feature_importances"] = {
            col: int(imp[i]) for i, col in enumerate(FEATURE_COLUMNS)
        }

    artifact = TradeQualityModel(model=model, feature_columns=list(FEATURE_COLUMNS))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    logger.info(
        "Trade quality model saved to %s: AUC=%.3f acc=%.3f prec=%.3f rec=%.3f",
        model_path, auc, metrics["accuracy"], metrics["precision"], metrics["recall"],
    )
    return metrics


class _TradeQualityUnpickler(pickle.Unpickler):
    """Handle models pickled when trade_quality.py was run as __main__."""

    def find_class(self, module: str, name: str):
        if name == "TradeQualityModel":
            return TradeQualityModel
        return super().find_class(module, name)


def load_trade_quality_model(model_path: str = DEFAULT_MODEL_PATH) -> TradeQualityModel:
    """Load a saved trade quality model."""
    with open(model_path, "rb") as f:
        model = _TradeQualityUnpickler(f).load()
    if not isinstance(model, TradeQualityModel):
        raise RuntimeError(f"Expected TradeQualityModel, got {type(model)}")
    return model


def walk_forward_validate(
    db_path: str = "data/hogan.db",
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    n_splits: int = 5,
    limit: int = 50000,
) -> dict:
    """Walk-forward validation of the trade quality model.

    For each window: train on past trades, predict on test trades,
    measure how well the model separates good from bad setups.
    """
    from sklearn.metrics import roc_auc_score

    from hogan_bot.backtest import run_backtest_on_candles
    from hogan_bot.champion import apply_champion_mode
    from hogan_bot.config import load_config
    from hogan_bot.ml import load_model
    from hogan_bot.profiles import CANONICAL_PROFILE, apply_profile
    from hogan_bot.storage import get_connection, load_candles

    conn = get_connection(db_path)
    candles = load_candles(conn, symbol, timeframe, limit=limit)
    conn.close()

    if len(candles) < 1000:
        return {"error": "insufficient data"}

    config = load_config()
    config, cli_ov = apply_profile(config, CANONICAL_PROFILE)
    config = apply_champion_mode(config)

    _enable_shorts = cli_ov.get("enable_shorts", getattr(config, "enable_shorts", True))
    _enable_pb = cli_ov.get("enable_pullback_gate", False)

    ml_model = None
    try:
        ml_model = load_model(config.ml_model_path)
    except Exception:
        pass

    n_bars = len(candles)
    window_size = n_bars // (n_splits + 1)
    results = []

    for wi in range(n_splits):
        train_end = window_size * (wi + 1)
        test_end = min(train_end + window_size, n_bars)
        if test_end <= train_end:
            break

        train_candles = candles.iloc[:train_end].copy()
        test_candles = candles.iloc[:test_end].copy()

        logger.info("  [W%d] Train on %d bars, test on %d bars", wi, train_end, test_end - train_end)

        _bt_kwargs = dict(
            symbol=symbol,
            starting_balance_usd=config.starting_balance_usd,
            aggressive_allocation=config.aggressive_allocation,
            max_risk_per_trade=config.max_risk_per_trade,
            max_drawdown=config.max_drawdown,
            short_ma_window=config.short_ma_window,
            long_ma_window=config.long_ma_window,
            volume_window=config.volume_window,
            volume_threshold=config.volume_threshold,
            fee_rate=config.fee_rate, timeframe=timeframe,
            ml_model=ml_model,
            ml_buy_threshold=config.ml_buy_threshold,
            ml_sell_threshold=config.ml_sell_threshold,
            trailing_stop_pct=config.trailing_stop_pct,
            take_profit_pct=config.take_profit_pct,
            max_hold_hours=config.max_hold_hours,
            enable_shorts=_enable_shorts,
            short_max_hold_hours=config.short_max_hold_hours,
            slippage_bps=5.0, execution_mode="next_open",
            use_ml_as_sizer=config.use_ml_as_sizer,
            db_path=db_path,
            trail_activation_pct=config.trail_activation_pct,
            breakeven_stop_pct=getattr(config, "breakeven_stop_pct", 0.015),
            enable_pullback_gate=_enable_pb,
            use_policy_core=True,
        )
        train_result = run_backtest_on_candles(train_candles, **_bt_kwargs)
        from hogan_bot.backtest import enrich_trades_with_entry_context
        enrich_trades_with_entry_context(train_candles, train_result.closed_trades)

        train_data = _trades_to_dataset(train_result.closed_trades)
        if train_data is None:
            logger.warning("  [W%d] Insufficient training data", wi)
            continue

        X_train, y_train = train_data

        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                num_leaves=31, min_child_samples=10,
                class_weight="balanced", random_state=42, verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(
                max_iter=150, max_depth=5, learning_rate=0.05, random_state=42,
            )

        model.fit(X_train, y_train)

        test_result = run_backtest_on_candles(test_candles, **_bt_kwargs)
        enrich_trades_with_entry_context(test_candles, test_result.closed_trades)

        test_trades = [t for t in test_result.closed_trades
                       if t.get("entry_bar_idx", 0) >= train_end]
        test_data = _trades_to_dataset(test_trades)
        if test_data is None:
            logger.warning("  [W%d] Insufficient test data", wi)
            continue

        X_test, y_test = test_data
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5

        high_q = y_proba >= 0.50
        low_q = y_proba < 0.50
        high_q_trades = [t for t, hq in zip(test_trades, high_q) if hq and t.get("entry_context")]
        low_q_trades = [t for t, lq in zip(test_trades, low_q) if lq and t.get("entry_context")]

        high_q_pnl = np.mean([t["pnl_pct"] for t in high_q_trades]) if high_q_trades else 0
        low_q_pnl = np.mean([t["pnl_pct"] for t in low_q_trades]) if low_q_trades else 0

        w_result = {
            "window": wi,
            "train_trades": len(X_train),
            "test_trades": len(X_test),
            "test_good_rate": round(float(y_test.mean()), 4),
            "auc": round(auc, 4),
            "high_quality_trades": int(high_q.sum()),
            "high_quality_avg_pnl": round(float(high_q_pnl), 4),
            "low_quality_trades": int(low_q.sum()),
            "low_quality_avg_pnl": round(float(low_q_pnl), 4),
            "separation": round(float(high_q_pnl - low_q_pnl), 4),
        }
        results.append(w_result)
        logger.info(
            "  [W%d] AUC=%.3f | high_q: %d trades, avg_pnl=%.2f%% | low_q: %d trades, avg_pnl=%.2f%% | sep=%.2f%%",
            wi, auc, w_result["high_quality_trades"], high_q_pnl,
            w_result["low_quality_trades"], low_q_pnl, w_result["separation"],
        )

    if not results:
        return {"error": "no windows completed"}

    return {
        "n_windows": len(results),
        "mean_auc": round(np.mean([r["auc"] for r in results]), 4),
        "mean_separation": round(np.mean([r["separation"] for r in results]), 4),
        "mean_high_q_pnl": round(np.mean([r["high_quality_avg_pnl"] for r in results]), 4),
        "mean_low_q_pnl": round(np.mean([r["low_quality_avg_pnl"] for r in results]), 4),
        "windows": results,
    }


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Trade Quality Classifier")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--limit", type=int, default=50000)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--train", action="store_true", help="Generate data and train")
    p.add_argument("--validate", action="store_true", help="Walk-forward validate")
    p.add_argument("--inspect", action="store_true", help="Inspect saved model")
    args = p.parse_args()

    if args.train:
        print("Generating training data from backtest...")
        data = generate_training_data(
            db_path=args.db, symbol=args.symbol,
            timeframe=args.timeframe, limit=args.limit,
        )
        if data is None:
            print("Failed to generate training data")
            return
        X, y = data
        print(f"\nTraining on {len(X)} trades ({y.sum()} good, {len(y) - y.sum()} bad)...")
        metrics = train_trade_quality_model(X, y, model_path=args.model_path)
        print(json.dumps(metrics, indent=2))

    if args.validate:
        print("Running walk-forward validation...")
        report = walk_forward_validate(
            db_path=args.db, symbol=args.symbol,
            timeframe=args.timeframe, n_splits=5, limit=args.limit,
        )
        print(json.dumps(report, indent=2))

    if args.inspect:
        model = load_trade_quality_model(args.model_path)
        print(f"Model type: {type(model.model).__name__}")
        print(f"Features ({len(model.feature_columns)}): {model.feature_columns}")
        print(f"Threshold: {model.threshold}")
        if hasattr(model.model, "feature_importances_"):
            imp = model.model.feature_importances_
            pairs = sorted(zip(model.feature_columns, imp), key=lambda x: -x[1])
            print("\nFeature importances:")
            for col, val in pairs:
                print(f"  {col:30s} {val:6.0f}")


if __name__ == "__main__":
    _main()
