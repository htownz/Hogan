"""Rolling walk-forward validation harness for Hogan.

Splits a candle dataset into N consecutive train/test windows and runs
a full retrain + backtest on each.  Reports per-window and aggregate
metrics, and applies a configurable promotion gate.

This is the single most important validation tool: if the strategy
cannot survive N consecutive out-of-sample windows after fees and
slippage, it is not ready for live trading.

Usage (CLI)::

    python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5

Usage (programmatic)::

    from hogan_bot.walk_forward import walk_forward_validate, WFConfig

    report = walk_forward_validate(candles, WFConfig(n_splits=5))
    print(report.passes_gate)
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WFConfig:
    """Walk-forward configuration."""
    n_splits: int = 5
    train_ratio: float = 0.70
    min_train_bars: int = 2000
    min_test_bars: int = 200

    symbol: str = "BTC/USD"
    timeframe: str = "1h"
    starting_balance: float = 10_000.0
    fee_rate: float = 0.0026
    slippage_bps: float = 5.0

    enable_shorts: bool = True
    enable_pullback_gate: bool = True
    short_max_hold_hours: float = 12.0
    max_hold_hours: float = 24.0
    loss_cooldown_hours: float = 2.0

    ml_buy_threshold: float = 0.51
    ml_sell_threshold: float = 0.49

    min_sharpe: float = 0.5
    max_drawdown_pct: float = 15.0
    min_trades_per_window: int = 5
    min_windows_positive: int = 3


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""
    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_bars: int
    test_bars: int

    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float | None = None
    sortino: float | None = None
    trades: int = 0
    win_rate: float = 0.0
    net_positive: bool = False

    train_auc: float | None = None
    error: str | None = None

    def summary_line(self) -> str:
        s = self.sharpe if self.sharpe is not None else 0.0
        return (
            f"W{self.window_idx}: "
            f"ret={self.total_return_pct:+.2f}% "
            f"dd={self.max_drawdown_pct:.1f}% "
            f"sharpe={s:.2f} "
            f"trades={self.trades} "
            f"win={self.win_rate:.0%} "
            f"{'PASS' if self.net_positive else 'FAIL'}"
        )


@dataclass
class WalkForwardReport:
    """Aggregate walk-forward validation report."""
    config: WFConfig
    windows: list[WindowResult] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def n_positive(self) -> int:
        return sum(1 for w in self.windows if w.net_positive)

    @property
    def n_failed(self) -> int:
        return sum(1 for w in self.windows if w.error is not None)

    @property
    def total_trades(self) -> int:
        return sum(w.trades for w in self.windows)

    @property
    def mean_return(self) -> float:
        rets = [w.total_return_pct for w in self.windows if w.error is None]
        return float(np.mean(rets)) if rets else 0.0

    @property
    def mean_sharpe(self) -> float:
        sharpes = [w.sharpe for w in self.windows if w.sharpe is not None]
        return float(np.mean(sharpes)) if sharpes else 0.0

    @property
    def mean_drawdown(self) -> float:
        dds = [w.max_drawdown_pct for w in self.windows if w.error is None]
        return float(np.mean(dds)) if dds else 0.0

    @property
    def worst_drawdown(self) -> float:
        dds = [w.max_drawdown_pct for w in self.windows if w.error is None]
        return max(dds) if dds else 0.0

    @property
    def passes_gate(self) -> bool:
        if self.n_failed > 0:
            return False
        if self.n_positive < self.config.min_windows_positive:
            return False
        if self.mean_sharpe < self.config.min_sharpe:
            return False
        if self.worst_drawdown > self.config.max_drawdown_pct:
            return False
        return True

    def summary(self) -> dict:
        return {
            "n_windows": self.n_windows,
            "n_positive": self.n_positive,
            "n_failed": self.n_failed,
            "total_trades": self.total_trades,
            "mean_return_pct": round(self.mean_return, 4),
            "mean_sharpe": round(self.mean_sharpe, 4),
            "mean_drawdown_pct": round(self.mean_drawdown, 2),
            "worst_drawdown_pct": round(self.worst_drawdown, 2),
            "passes_gate": self.passes_gate,
            "gate_config": {
                "min_sharpe": self.config.min_sharpe,
                "max_drawdown_pct": self.config.max_drawdown_pct,
                "min_windows_positive": self.config.min_windows_positive,
            },
        }


def _compute_windows(
    total_bars: int, cfg: WFConfig
) -> list[tuple[int, int, int, int]]:
    """Compute (train_start, train_end, test_start, test_end) for each fold.

    Uses an expanding-window scheme: each fold trains on everything up to
    the test window, ensuring later folds see more data (like reality).
    """
    test_total = total_bars - cfg.min_train_bars
    if test_total < cfg.min_test_bars:
        raise ValueError(
            f"Not enough bars ({total_bars}) for walk-forward with "
            f"min_train={cfg.min_train_bars}, min_test={cfg.min_test_bars}"
        )

    test_size = test_total // cfg.n_splits
    if test_size < cfg.min_test_bars:
        test_size = cfg.min_test_bars

    windows = []
    for i in range(cfg.n_splits):
        test_start = cfg.min_train_bars + i * test_size
        test_end = min(test_start + test_size, total_bars)
        if test_end > total_bars:
            break
        if test_end - test_start < cfg.min_test_bars and i > 0:
            break
        train_start = 0
        train_end = test_start
        windows.append((train_start, train_end, test_start, test_end))

    return windows


def _train_and_evaluate_window(
    candles: pd.DataFrame,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    window_idx: int,
    cfg: WFConfig,
) -> WindowResult:
    """Train on [train_start:train_end], evaluate on [test_start:test_end]."""
    import sys
    import time as _time

    from hogan_bot.backtest import run_backtest_on_candles
    from hogan_bot.config import BotConfig, load_config
    from hogan_bot.champion import apply_champion_mode, is_champion_mode
    from hogan_bot.ml import TrainedModel, build_training_set

    result = WindowResult(
        window_idx=window_idx,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        train_bars=train_end - train_start,
        test_bars=test_end - test_start,
    )

    try:
        train_candles = candles.iloc[train_start:train_end].copy()
        test_candles = candles.iloc[test_start:test_end].copy()

        t0 = _time.perf_counter()
        logger.info("    [W%d] Building training set (%d bars)...", window_idx, len(train_candles))
        sys.stdout.flush()

        X, y, feature_cols, _ = build_training_set(
            train_candles,
            horizon_bars=6,
            fee_rate=cfg.fee_rate,
            use_champion_features=True,
        )

        if len(X) < 100:
            result.error = f"insufficient_training_samples ({len(X)})"
            return result

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(
            max_iter=1000, C=0.1, class_weight="balanced", solver="lbfgs"
        )
        model.fit(X_scaled, y)

        t_train = _time.perf_counter() - t0
        logger.info("    [W%d] Model trained in %.1fs (samples=%d, features=%d)",
                     window_idx, t_train, len(X), X.shape[1])
        sys.stdout.flush()

        try:
            y_prob = model.predict_proba(X_scaled)[:, 1]
            result.train_auc = float(roc_auc_score(y, y_prob))
            logger.info("    [W%d] Train AUC: %.4f", window_idx, result.train_auc)
        except Exception:
            pass

        trained = TrainedModel(
            model=model,
            feature_columns=feature_cols,
            scaler=scaler,
        )

        bot_cfg = load_config()
        if is_champion_mode():
            bot_cfg = apply_champion_mode(bot_cfg)

        logger.info("    [W%d] Running backtest on %d test bars...", window_idx, len(test_candles))
        sys.stdout.flush()
        t1 = _time.perf_counter()

        bt = run_backtest_on_candles(
            test_candles,
            symbol=cfg.symbol,
            starting_balance_usd=cfg.starting_balance,
            aggressive_allocation=0.50,
            max_risk_per_trade=bot_cfg.max_risk_per_trade,
            max_drawdown=0.20,
            short_ma_window=bot_cfg.short_ma_window,
            long_ma_window=bot_cfg.long_ma_window,
            volume_window=bot_cfg.volume_window,
            volume_threshold=bot_cfg.volume_threshold,
            fee_rate=cfg.fee_rate,
            timeframe=cfg.timeframe,
            ml_model=trained,
            ml_buy_threshold=cfg.ml_buy_threshold,
            ml_sell_threshold=cfg.ml_sell_threshold,
            trailing_stop_pct=bot_cfg.trailing_stop_pct,
            take_profit_pct=bot_cfg.take_profit_pct,
            ml_confidence_sizing=True,
            max_hold_hours=cfg.max_hold_hours,
            loss_cooldown_hours=cfg.loss_cooldown_hours,
            slippage_bps=cfg.slippage_bps,
            enable_shorts=cfg.enable_shorts,
            enable_pullback_gate=cfg.enable_pullback_gate,
            short_max_hold_hours=cfg.short_max_hold_hours,
            min_edge_multiple=bot_cfg.min_edge_multiple,
            min_final_confidence=bot_cfg.min_final_confidence,
            min_tech_confidence=bot_cfg.min_tech_confidence,
            min_regime_confidence=bot_cfg.min_regime_confidence,
            max_whipsaws=bot_cfg.max_whipsaws,
            reversal_confidence_mult=bot_cfg.reversal_confidence_multiplier,
        )

        t_bt = _time.perf_counter() - t1
        t_total = _time.perf_counter() - t0
        logger.info("    [W%d] Backtest done in %.1fs (total %.1fs)",
                     window_idx, t_bt, t_total)

        result.total_return_pct = bt.total_return_pct
        result.max_drawdown_pct = bt.max_drawdown_pct
        result.sharpe = bt.sharpe_ratio
        result.sortino = bt.sortino_ratio
        result.trades = bt.trades
        result.win_rate = bt.win_rate
        result.net_positive = bt.total_return_pct > 0.0

    except Exception as exc:
        result.error = str(exc)
        logger.exception("Walk-forward window %d failed", window_idx)

    return result


def walk_forward_validate(
    candles: pd.DataFrame,
    cfg: WFConfig | None = None,
) -> WalkForwardReport:
    """Run rolling walk-forward validation across the full candle dataset.

    Each window trains a fresh LogReg model on the training portion and
    evaluates on the unseen test portion, using the canonical profile
    parameters.  Returns a structured report with per-window results
    and an aggregate promotion gate.
    """
    if cfg is None:
        cfg = WFConfig()

    windows = _compute_windows(len(candles), cfg)
    report = WalkForwardReport(config=cfg)

    import sys
    import time as _time

    total_test_bars = sum(ve - vs for _, _, vs, ve in windows)
    logger.info(
        "Walk-forward: %d windows over %d bars (%d total test bars)",
        len(windows), len(candles), total_test_bars,
    )
    sys.stdout.flush()

    wf_start = _time.perf_counter()
    for i, (ts, te, vs, ve) in enumerate(windows):
        logger.info(
            "  Window %d/%d: train[%d:%d] (%d bars) -> test[%d:%d] (%d bars)",
            i + 1, len(windows), ts, te, te - ts, vs, ve, ve - vs,
        )
        sys.stdout.flush()
        w = _train_and_evaluate_window(candles, ts, te, vs, ve, i, cfg)
        report.windows.append(w)
        logger.info("    %s", w.summary_line())
        sys.stdout.flush()

    elapsed = _time.perf_counter() - wf_start
    logger.info("Walk-forward complete in %.1fs: %s",
                elapsed, json.dumps(report.summary(), indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sqlite3

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(description="Hogan walk-forward validation")
    p.add_argument("--db", default="data/hogan.db", help="SQLite DB path")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2000)
    p.add_argument("--min-test", type=int, default=200)
    p.add_argument("--min-sharpe", type=float, default=0.5)
    p.add_argument("--max-dd", type=float, default=15.0)
    p.add_argument("--output", default="diagnostics/walk_forward_report.json")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    query = """
        SELECT ts_ms, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND timeframe = ?
        ORDER BY ts_ms
    """
    df = pd.read_sql_query(query, conn, params=(args.symbol, args.timeframe))
    conn.close()

    if df.empty:
        logger.error("No candles found for %s %s in %s", args.symbol, args.timeframe, args.db)
        return

    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    logger.info("Loaded %d candles for %s %s", len(df), args.symbol, args.timeframe)

    cfg = WFConfig(
        n_splits=args.n_splits,
        min_train_bars=args.min_train,
        min_test_bars=args.min_test,
        symbol=args.symbol,
        timeframe=args.timeframe,
        min_sharpe=args.min_sharpe,
        max_drawdown_pct=args.max_dd,
    )

    report = walk_forward_validate(df, cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "summary": report.summary(),
        "windows": [
            {
                "window_idx": w.window_idx,
                "train_bars": w.train_bars,
                "test_bars": w.test_bars,
                "total_return_pct": round(w.total_return_pct, 4),
                "max_drawdown_pct": round(w.max_drawdown_pct, 2),
                "sharpe": round(w.sharpe, 4) if w.sharpe is not None else None,
                "sortino": round(w.sortino, 4) if w.sortino is not None else None,
                "trades": w.trades,
                "win_rate": round(w.win_rate, 4),
                "net_positive": w.net_positive,
                "train_auc": round(w.train_auc, 4) if w.train_auc is not None else None,
                "error": w.error,
            }
            for w in report.windows
        ],
    }

    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Report saved to %s", out_path)

    gate = "PASS" if report.passes_gate else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"WALK-FORWARD GATE: {gate}")
    print(f"  Windows: {report.n_positive}/{report.n_windows} positive")
    print(f"  Mean Sharpe: {report.mean_sharpe:.2f}")
    print(f"  Mean Return: {report.mean_return:+.2f}%")
    print(f"  Worst Drawdown: {report.worst_drawdown:.1f}%")
    print(f"  Total Trades: {report.total_trades}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
