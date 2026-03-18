"""Out-of-sample evaluation harness for Hogan models.

Provides walk-forward OOS evaluation that runs a candidate model
on unseen data and returns structured metrics for the registry.

This module is used by retrain.py to log OOS performance alongside
training metrics.  Promotion gating (enforcing minimum OOS Sharpe,
max drawdown, and trade count) is scaffolded here but only activated
when the operator sets ``enforce=True``.

Usage::

    from hogan_bot.oos_eval import oos_evaluate

    result = oos_evaluate(
        candidate_path="models/candidate_20260309.pkl",
        candles=full_candle_df,
        symbol="BTC/USD",
        train_end_idx=split_idx,
        db_path="data/hogan.db",
    )
    # result.sharpe, result.max_drawdown_pct, result.trade_count, ...
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OOSResult:
    """Structured output from an OOS evaluation run."""
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0

    train_window: str = ""
    val_window: str = ""
    oos_window: str = ""
    oos_bars: int = 0

    cost_model: str = "fee_rate + slippage_bps"
    fee_rate: float = 0.0026
    slippage_bps: float = 5.0

    label_mode: str = "fee_threshold"
    freshness_note: str = ""

    # Forecast calibration metrics (populated when forecast models are used)
    forecast_brier_4h: float | None = None
    forecast_brier_12h: float | None = None
    forecast_brier_24h: float | None = None
    forecast_ece: float | None = None
    forecast_roc_auc: float | None = None

    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {f"oos_{k}": v for k, v in asdict(self).items()}

    def passes_gate(
        self,
        min_sharpe: float = 0.5,
        max_drawdown: float = 15.0,
        min_trades: int = 10,
    ) -> bool:
        """Check whether this OOS result passes promotion requirements."""
        if self.trade_count < min_trades:
            return False
        if self.sharpe < min_sharpe:
            return False
        if self.max_drawdown_pct > max_drawdown:
            return False
        return True


def oos_evaluate(
    candidate_path: str,
    candles: pd.DataFrame,
    symbol: str = "BTC/USD",
    train_end_idx: int | None = None,
    timeframe: str = "1h",
    db_path: str | None = None,
    fee_rate: float = 0.0026,
    slippage_bps: float = 5.0,
    starting_balance: float = 10_000.0,
    short_ma_window: int = 12,
    long_ma_window: int = 48,
    volume_window: int = 20,
    volume_threshold: float = 1.0,
    aggressive_allocation: float = 0.5,
    max_risk_per_trade: float = 0.02,
    max_drawdown: float = 0.20,
    max_hold_bars: int = 24,
    loss_cooldown_bars: int = 2,
    label_mode: str = "fee_threshold",
) -> OOSResult:
    """Run an OOS backtest on the tail of *candles* using the candidate model.

    Parameters
    ----------
    candidate_path
        Path to the pickled candidate model.
    candles
        Full candle DataFrame.  The OOS window is everything after
        ``train_end_idx``.
    train_end_idx
        Index (row number) where the training data ends.  The OOS window
        is ``candles.iloc[train_end_idx:]``.  If ``None``, defaults to
        80% of the data.
    db_path
        When provided, backtest uses historical sentiment/macro data
        with point-in-time (as-of) semantics.
    """
    from hogan_bot.backtest import run_backtest_on_candles
    from hogan_bot.ml import load_model

    if train_end_idx is None:
        train_end_idx = int(len(candles) * 0.80)

    oos_candles = candles.iloc[train_end_idx:].copy()
    if len(oos_candles) < 50:
        logger.warning("OOS window too small (%d bars), skipping evaluation", len(oos_candles))
        return OOSResult(freshness_note="oos_window_too_small")

    try:
        ml_model = load_model(candidate_path)
    except Exception as exc:
        logger.warning("Failed to load candidate model for OOS eval: %s", exc)
        return OOSResult(freshness_note=f"model_load_failed: {exc}")

    def _ts_label(df: pd.DataFrame) -> str:
        if "timestamp" in df.columns and len(df) > 0:
            first = df["timestamp"].iloc[0]
            last = df["timestamp"].iloc[-1]
            return f"{first} -> {last}"
        return f"idx {df.index[0]} -> {df.index[-1]}" if len(df) > 0 else "empty"

    train_window = _ts_label(candles.iloc[:train_end_idx])
    oos_window = _ts_label(oos_candles)

    try:
        bt = run_backtest_on_candles(
            oos_candles,
            symbol=symbol,
            starting_balance_usd=starting_balance,
            aggressive_allocation=aggressive_allocation,
            max_risk_per_trade=max_risk_per_trade,
            max_drawdown=max_drawdown,
            short_ma_window=short_ma_window,
            long_ma_window=long_ma_window,
            volume_window=volume_window,
            volume_threshold=volume_threshold,
            fee_rate=fee_rate,
            timeframe=timeframe,
            ml_model=ml_model,
            ml_confidence_sizing=False,
            use_ml_as_sizer=True,
            max_hold_bars=max_hold_bars,
            loss_cooldown_bars=loss_cooldown_bars,
            slippage_bps=slippage_bps,
            db_path=db_path,
        )
    except Exception as exc:
        logger.warning("OOS backtest failed: %s", exc)
        return OOSResult(freshness_note=f"backtest_failed: {exc}")

    return OOSResult(
        sharpe=bt.sharpe_ratio,
        sortino=bt.sortino_ratio,
        calmar=bt.calmar_ratio,
        total_return_pct=bt.total_return_pct,
        max_drawdown_pct=bt.max_drawdown_pct,
        trade_count=bt.trades,
        win_rate=bt.win_rate,
        train_window=train_window,
        val_window="",
        oos_window=oos_window,
        oos_bars=len(oos_candles),
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        label_mode=label_mode,
    )
