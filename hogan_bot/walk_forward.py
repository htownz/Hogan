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

    use_ml_filter: bool = True
    use_ml_as_sizer: bool = False
    use_macro_sitout: bool = False
    use_funding_overlay: bool = False
    ml_buy_threshold: float = 0.51
    ml_sell_threshold: float = 0.49
    model_type: str = "logreg"
    label_mode: str = "fee_threshold"

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

    signal_funnel: dict = field(default_factory=dict)
    regime_distribution: dict = field(default_factory=dict)
    closed_trades: list[dict] = field(default_factory=list)
    test_start_date: str | None = None
    test_end_date: str | None = None

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
    macro_sitout=None,
    funding_overlay=None,
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
        trained = None

        if cfg.use_ml_filter or cfg.use_ml_as_sizer:
            logger.info("    [W%d] Building training set (%d bars, model=%s, labels=%s)...",
                         window_idx, len(train_candles), cfg.model_type, cfg.label_mode)
            sys.stdout.flush()

            X, y, feature_cols, meta_quality = build_training_set(
                train_candles,
                horizon_bars=6,
                fee_rate=cfg.fee_rate,
                use_champion_features=True,
                label_mode=cfg.label_mode,
            )

            if X is None or y is None or len(X) < 100:
                result.error = f"insufficient_training_samples ({len(X) if X is not None else 0})"
                return result

            from sklearn.metrics import roc_auc_score

            scaler = None
            X_arr = X

            if cfg.model_type == "logreg":
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_arr = scaler.fit_transform(X)
                model = LogisticRegression(
                    max_iter=1000, C=0.1, class_weight="balanced", solver="lbfgs"
                )
                model.fit(X_arr, y)
            elif cfg.model_type in ("histgb", "hist_gb"):
                from sklearn.ensemble import HistGradientBoostingClassifier
                model = HistGradientBoostingClassifier(
                    max_depth=3, learning_rate=0.02, max_iter=150,
                    min_samples_leaf=50, l2_regularization=0.1,
                    random_state=42,
                )
                class_counts = y.value_counts()
                n_samples, n_classes = len(y), len(class_counts)
                class_weight = y.map(
                    {c: n_samples / (n_classes * cnt) for c, cnt in class_counts.items()}
                ).values
                if meta_quality is not None:
                    mq = np.clip(meta_quality, 0.1, 1.0)
                    sample_weight = class_weight * mq
                else:
                    sample_weight = class_weight
                model.fit(X_arr, y, sample_weight=sample_weight)
            else:
                from hogan_bot.ml import _make_cv_model
                model = _make_cv_model(cfg.model_type)
                model.fit(X_arr, y)

            t_train = _time.perf_counter() - t0
            logger.info("    [W%d] %s trained in %.1fs (samples=%d, features=%d)",
                         window_idx, cfg.model_type, t_train, len(X), X.shape[1])
            sys.stdout.flush()

            try:
                X_score = X_arr if not hasattr(X_arr, 'values') else X_arr
                y_prob = model.predict_proba(X_score)[:, 1]
                result.train_auc = float(roc_auc_score(y, y_prob))
                logger.info("    [W%d] Train AUC: %.4f", window_idx, result.train_auc)
            except Exception:
                pass

            trained = TrainedModel(
                model=model,
                feature_columns=feature_cols,
                scaler=scaler,
            )
        else:
            logger.info("    [W%d] ML filter disabled — technical pipeline only", window_idx)
            sys.stdout.flush()

        bot_cfg = load_config()
        if is_champion_mode():
            bot_cfg = apply_champion_mode(bot_cfg)

        logger.info("    [W%d] Running backtest on %d test bars%s...",
                     window_idx, len(test_candles),
                     "" if trained else " (no ML)")
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
            trail_activation_pct=bot_cfg.trail_activation_pct,
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
            macro_sitout=macro_sitout,
            use_ml_as_sizer=cfg.use_ml_as_sizer,
            funding_overlay=funding_overlay,
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
        result.signal_funnel = dict(bt.signal_funnel)
        result.regime_distribution = bt.signal_funnel.get("regime_distribution", {})
        result.closed_trades = list(bt.closed_trades)

        if "timestamp" in test_candles.columns:
            result.test_start_date = str(test_candles["timestamp"].iloc[0])
            result.test_end_date = str(test_candles["timestamp"].iloc[-1])

    except Exception as exc:
        result.error = str(exc)
        logger.exception("Walk-forward window %d failed", window_idx)

    return result


def walk_forward_validate(
    candles: pd.DataFrame,
    cfg: WFConfig | None = None,
    macro_sitout=None,
    funding_overlay=None,
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
    _parts = []
    if cfg.use_ml_as_sizer:
        _parts.append("ML-sizer")
    elif cfg.use_ml_filter:
        _parts.append("ML")
    if cfg.use_macro_sitout:
        _parts.append("macro")
    if cfg.use_funding_overlay:
        _parts.append("funding")
    ml_label = "+".join(_parts) if _parts else "no filters"
    logger.info(
        "Walk-forward: %d windows over %d bars (%d total test bars) [%s]",
        len(windows), len(candles), total_test_bars, ml_label,
    )
    sys.stdout.flush()

    wf_start = _time.perf_counter()
    for i, (ts, te, vs, ve) in enumerate(windows):
        logger.info(
            "  Window %d/%d: train[%d:%d] (%d bars) -> test[%d:%d] (%d bars)",
            i + 1, len(windows), ts, te, te - ts, vs, ve, ve - vs,
        )
        sys.stdout.flush()
        w = _train_and_evaluate_window(
            candles, ts, te, vs, ve, i, cfg,
            macro_sitout=macro_sitout,
            funding_overlay=funding_overlay,
        )
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

def _print_funnel_comparison(report: WalkForwardReport) -> None:
    """Print a per-window funnel comparison table to stdout."""

    print(f"\n{'=' * 90}")
    print("PER-WINDOW FUNNEL ANALYSIS")
    print(f"{'=' * 90}")

    for w in report.windows:
        f = w.signal_funnel
        label = "PASS" if w.net_positive else "FAIL"
        print(f"\n--- W{w.window_idx} ({label}) | {w.test_start_date or '?'} to {w.test_end_date or '?'} ---")
        print(f"  Return: {w.total_return_pct:+.2f}%  Sharpe: {w.sharpe or 0:.2f}  "
              f"Trades: {w.trades}  Win: {w.win_rate:.0%}")

        bars = f.get("bars_evaluated", 0)
        pipe_buy = f.get("pipeline_buy", 0)
        pipe_sell = f.get("pipeline_sell", 0)
        post_ml_buy = f.get("post_ml_buy", 0)
        post_ml_sell = f.get("post_ml_sell", 0)
        post_edge_buy = f.get("post_edge_buy", 0)
        post_edge_sell = f.get("post_edge_sell", 0)
        post_qual_buy = f.get("post_quality_buy", 0)
        post_qual_sell = f.get("post_quality_sell", 0)
        post_rang_buy = f.get("post_ranging_buy", 0)
        post_rang_sell = f.get("post_ranging_sell", 0)
        exec_buy = f.get("executed_buy", 0)
        exec_short = f.get("executed_short_entry", 0)
        blk_regime_long = f.get("blocked_regime_no_longs", 0)
        blk_regime_short = f.get("blocked_regime_no_shorts", 0)
        blk_cooldown = f.get("blocked_cooldown", 0)
        blk_already = f.get("blocked_already_long", 0)

        print(f"  Bars evaluated:     {bars}")
        print(f"  Pipeline signals:   buy={pipe_buy}  sell={pipe_sell}")

        _ml_active = "post_ml_buy" in f or "post_ml_sell" in f
        if _ml_active:
            ml_kill_buy = pipe_buy - post_ml_buy if pipe_buy else 0
            ml_kill_pct = (ml_kill_buy / pipe_buy * 100) if pipe_buy else 0
            print(f"  Post-ML filter:     buy={post_ml_buy}  sell={post_ml_sell}  "
                  f"(ML killed {ml_kill_pct:.0f}% of buys)")
        else:
            print(f"  Post-ML filter:     (ML disabled)")

        _pre_edge_buy = post_ml_buy if _ml_active else pipe_buy
        edge_kill = _pre_edge_buy - post_edge_buy
        print(f"  Post-edge gate:     buy={post_edge_buy}  sell={post_edge_sell}  "
              f"(edge killed {edge_kill})")

        qual_kill = post_edge_buy - post_qual_buy
        print(f"  Post-quality gate:  buy={post_qual_buy}  sell={post_qual_sell}  "
              f"(quality killed {qual_kill})")

        rang_kill = post_qual_buy - post_rang_buy
        print(f"  Post-ranging gate:  buy={post_rang_buy}  sell={post_rang_sell}  "
              f"(ranging killed {rang_kill})")

        macro_sitout_cnt = f.get("macro_sitout", 0)
        macro_scaled_cnt = f.get("macro_scaled", 0)

        print(f"  Executed:           longs={exec_buy}  shorts={exec_short}")
        if macro_sitout_cnt or macro_scaled_cnt:
            print(f"  Macro sitout:       blocked={macro_sitout_cnt}  scaled={macro_scaled_cnt}")
        print(f"  Blocked:            regime_long={blk_regime_long}  "
              f"regime_short={blk_regime_short}  cooldown={blk_cooldown}  "
              f"already_long={blk_already}")

        regime = w.regime_distribution
        if regime:
            total_regime_bars = sum(regime.values())
            print(f"  Regime distribution:")
            for r, cnt in sorted(regime.items(), key=lambda x: -x[1]):
                pct = cnt / total_regime_bars * 100 if total_regime_bars else 0
                print(f"    {r:<16} {cnt:>5} bars ({pct:.1f}%)")

        if w.closed_trades:
            wins = [t for t in w.closed_trades if t.get("pnl_pct", 0) > 0]
            losses = [t for t in w.closed_trades if t.get("pnl_pct", 0) <= 0]
            avg_win = sum(t.get("pnl_pct", 0) for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t.get("pnl_pct", 0) for t in losses) / len(losses) if losses else 0
            sides = {}
            for t in w.closed_trades:
                s = t.get("side", "unknown")
                sides[s] = sides.get(s, 0) + 1
            side_str = "  ".join(f"{s}={c}" for s, c in sorted(sides.items()))
            print(f"  Trade breakdown:    {side_str}")
            print(f"  Avg win: {avg_win:+.3f}%  Avg loss: {avg_loss:+.3f}%  "
                  f"Payoff ratio: {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0
                  else f"  Avg win: {avg_win:+.3f}%  No losses")

    print(f"\n{'=' * 90}\n")


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
    p.add_argument("--no-ml", action="store_true", help="Disable ML filter (technical pipeline only)")
    p.add_argument("--ml-sizer", action="store_true", help="Use ML probability as continuous position sizer instead of binary filter")
    p.add_argument("--macro-sitout", action="store_true", help="Enable macro event sit-out filter")
    p.add_argument("--funding", action="store_true", help="Enable BTC funding rate overlay")
    p.add_argument("--model-type", default="histgb",
                   choices=["logreg", "histgb", "xgboost", "lightgbm", "random_forest"],
                   help="ML model type for walk-forward training (default: histgb)")
    p.add_argument("--label-mode", default="enhanced_triple_barrier",
                   choices=["fee_threshold", "triple_barrier", "enhanced_triple_barrier"],
                   help="Label construction method (default: enhanced_triple_barrier)")
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
        use_ml_filter=not args.no_ml and not args.ml_sizer,
        use_ml_as_sizer=args.ml_sizer,
        use_macro_sitout=args.macro_sitout,
        use_funding_overlay=args.funding,
        model_type=args.model_type,
        label_mode=args.label_mode,
    )

    sitout = None
    if args.macro_sitout:
        macro_conn = sqlite3.connect(args.db)
        from hogan_bot.macro_sitout import MacroSitout
        sitout = MacroSitout.from_db(macro_conn)
        macro_conn.close()
        logger.info("Macro sitout filter enabled")

    funding = None
    if args.funding:
        fund_conn = sqlite3.connect(args.db)
        from hogan_bot.funding_overlay import FundingOverlay
        funding = FundingOverlay.from_db(fund_conn)
        fund_conn.close()
        logger.info("Funding rate overlay enabled")

    report = walk_forward_validate(df, cfg, macro_sitout=sitout, funding_overlay=funding)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "summary": report.summary(),
        "windows": [
            {
                "window_idx": w.window_idx,
                "train_bars": w.train_bars,
                "test_bars": w.test_bars,
                "test_start_date": w.test_start_date,
                "test_end_date": w.test_end_date,
                "total_return_pct": round(w.total_return_pct, 4),
                "max_drawdown_pct": round(w.max_drawdown_pct, 2),
                "sharpe": round(w.sharpe, 4) if w.sharpe is not None else None,
                "sortino": round(w.sortino, 4) if w.sortino is not None else None,
                "trades": w.trades,
                "win_rate": round(w.win_rate, 4),
                "net_positive": w.net_positive,
                "train_auc": round(w.train_auc, 4) if w.train_auc is not None else None,
                "error": w.error,
                "signal_funnel": w.signal_funnel,
                "regime_distribution": w.regime_distribution,
                "closed_trades": [
                    {
                        "side": t.get("side", "?"),
                        "pnl_pct": round(t.get("pnl_pct", 0), 4),
                        "regime": t.get("entry_regime", "?"),
                        "exit_reason": t.get("close_reason", "?"),
                        "bars_held": t.get("bars_held", 0),
                        "entry_bar": t.get("entry_bar_idx"),
                    }
                    for t in w.closed_trades
                ],
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
    print(f"{'=' * 60}")

    _print_funnel_comparison(report)


if __name__ == "__main__":
    main()
