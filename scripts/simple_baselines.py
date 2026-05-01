#!/usr/bin/env python3
"""Run simple baseline strategies in Hogan's walk-forward geometry.

Provides reference baselines (buy-and-hold, MA trend-follow, RSI mean-reversion,
volatility breakout) so we can measure whether the live Hogan pipeline actually
beats trivial rules on the same OHLCV data.

Output JSON is intentionally compatible with hogan_bot.walk_forward summaries so
the strategy comparison runner can ingest baselines alongside Hogan scenarios.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BaselineConfig:
    """Walk-forward and execution config for the simple baselines."""

    n_splits: int = 2
    min_train_bars: int = 16000
    min_test_bars: int = 1000

    starting_balance: float = 10_000.0
    fee_rate: float = 0.0026
    slippage_bps: float = 5.0

    ma_short: int = 20
    ma_long: int = 50

    rsi_period: int = 14
    rsi_buy: float = 30.0
    rsi_exit: float = 50.0

    breakout_lookback: int = 20

    bars_per_year: int = 24 * 365

    min_sharpe: float = 0.5
    min_calmar: float = 0.0
    max_drawdown_pct: float = 25.0
    min_trades_per_window: int = 1
    min_total_trades: int = 1
    min_windows_positive: int = 1


@dataclass
class BaselineWindowResult:
    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    test_start_date: str | None = None
    test_end_date: str | None = None
    trades: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float | None = None
    calmar: float | None = None
    net_positive: bool = False
    closed_trades: list[dict] = field(default_factory=list)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _compute_windows(total_bars: int, cfg: BaselineConfig) -> list[tuple[int, int, int, int]]:
    """Mirror hogan_bot.walk_forward._compute_windows so geometry matches Hogan runs."""
    test_total = total_bars - cfg.min_train_bars
    if test_total < cfg.min_test_bars:
        raise ValueError(
            f"Not enough bars ({total_bars}) for baselines with "
            f"min_train={cfg.min_train_bars}, min_test={cfg.min_test_bars}"
        )
    if cfg.n_splits <= 0:
        raise ValueError("n_splits must be >= 1")
    test_size = max(test_total // cfg.n_splits, cfg.min_test_bars)
    windows: list[tuple[int, int, int, int]] = []
    for i in range(cfg.n_splits):
        ts = cfg.min_train_bars + i * test_size
        te = min(ts + test_size, total_bars)
        if te > total_bars:
            break
        if te - ts < cfg.min_test_bars and i > 0:
            break
        windows.append((0, ts, ts, te))
    return windows


def _signals_buy_hold(closes: np.ndarray) -> np.ndarray:
    return np.ones_like(closes, dtype=float)


def _signals_ma_trend(closes: np.ndarray, short: int, long: int) -> np.ndarray:
    s = pd.Series(closes)
    short_ma = s.rolling(short).mean().to_numpy()
    long_ma = s.rolling(long).mean().to_numpy()
    pos = np.where(short_ma > long_ma, 1.0, 0.0)
    pos[: max(short, long)] = 0.0
    return pos


def _signals_rsi_mean_revert(
    closes: np.ndarray,
    period: int,
    buy: float,
    exit_lvl: float,
) -> np.ndarray:
    s = pd.Series(closes)
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50.0).to_numpy()

    pos = np.zeros_like(closes, dtype=float)
    holding = False
    for i in range(len(closes)):
        if not holding:
            if rsi[i] < buy:
                holding = True
                pos[i] = 1.0
        else:
            if rsi[i] >= exit_lvl:
                holding = False
            else:
                pos[i] = 1.0
    return pos


def _signals_breakout(closes: np.ndarray, lookback: int) -> np.ndarray:
    s = pd.Series(closes)
    upper = s.rolling(lookback).max().shift(1).to_numpy()
    lower = s.rolling(lookback).min().shift(1).to_numpy()
    pos = np.zeros_like(closes, dtype=float)
    holding = False
    for i in range(len(closes)):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if not holding:
            if closes[i] > upper[i]:
                holding = True
                pos[i] = 1.0
        else:
            if closes[i] < lower[i]:
                holding = False
            else:
                pos[i] = 1.0
    return pos


SignalFn = Callable[[np.ndarray, "BaselineConfig"], np.ndarray]


STRATEGIES: dict[str, SignalFn] = {
    "buy_hold": lambda c, cfg: _signals_buy_hold(c),
    "ma_trend": lambda c, cfg: _signals_ma_trend(c, cfg.ma_short, cfg.ma_long),
    "rsi_mean_revert": lambda c, cfg: _signals_rsi_mean_revert(
        c, cfg.rsi_period, cfg.rsi_buy, cfg.rsi_exit
    ),
    "breakout": lambda c, cfg: _signals_breakout(c, cfg.breakout_lookback),
}


def _simulate(
    closes: np.ndarray,
    position: np.ndarray,
    *,
    fee_rate: float,
    slippage_bps: float,
    starting_balance: float,
) -> tuple[list[dict], np.ndarray]:
    """Simulate long-only positions, applying fees and slippage on each side change."""
    n = len(closes)
    equity = np.full(n, starting_balance, dtype=float)
    trades: list[dict] = []
    capital = starting_balance
    units = 0.0
    entry_price: float | None = None
    entry_bar: int | None = None
    slip = slippage_bps / 1e4

    for i in range(n):
        prev_pos = position[i - 1] if i > 0 else 0.0
        cur_pos = position[i]

        if prev_pos == 0 and cur_pos > 0 and units == 0.0:
            fill_price = closes[i] * (1 + slip)
            units = (capital * (1 - fee_rate)) / fill_price
            entry_price = fill_price
            entry_bar = i
            capital = 0.0
        elif prev_pos > 0 and cur_pos == 0 and units > 0:
            fill_price = closes[i] * (1 - slip)
            proceeds = units * fill_price * (1 - fee_rate)
            ret = (fill_price - entry_price) / entry_price if entry_price else 0.0
            trades.append({
                "side": "long",
                "entry_bar_idx": int(entry_bar) if entry_bar is not None else None,
                "exit_bar_idx": int(i),
                "entry_price": float(entry_price) if entry_price is not None else None,
                "exit_price": float(fill_price),
                "pnl_pct": float(ret * 100.0),
                "bars_held": int(i - entry_bar) if entry_bar is not None else 0,
                "close_reason": "signal_exit",
                "entry_regime": "?",
                "exit_regime": "?",
                "max_adverse_pct": 0.0,
                "max_favorable_pct": 0.0,
            })
            capital = proceeds
            units = 0.0
            entry_price = None
            entry_bar = None

        equity[i] = units * closes[i] if units > 0 else capital

    if units > 0 and entry_price is not None and entry_bar is not None:
        i = n - 1
        fill_price = closes[i] * (1 - slip)
        proceeds = units * fill_price * (1 - fee_rate)
        ret = (fill_price - entry_price) / entry_price
        trades.append({
            "side": "long",
            "entry_bar_idx": int(entry_bar),
            "exit_bar_idx": int(i),
            "entry_price": float(entry_price),
            "exit_price": float(fill_price),
            "pnl_pct": float(ret * 100.0),
            "bars_held": int(i - entry_bar),
            "close_reason": "end_of_window",
            "entry_regime": "?",
            "exit_regime": "?",
            "max_adverse_pct": 0.0,
            "max_favorable_pct": 0.0,
        })
        equity[i] = proceeds

    return trades, equity


def _metrics(equity: np.ndarray, starting_balance: float, bars_per_year: int) -> dict[str, Any]:
    if len(equity) < 2:
        return {"return_pct": 0.0, "drawdown_pct": 0.0, "sharpe": None, "calmar": None}
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]
    total_return_pct = float((equity[-1] / starting_balance - 1) * 100.0)
    cummax = np.maximum.accumulate(equity)
    dd = (cummax - equity) / np.where(cummax > 0, cummax, 1.0)
    max_dd_pct = float(dd.max() * 100.0) if len(dd) else 0.0
    sharpe: float | None = None
    if rets.size > 1 and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
    calmar: float | None = None
    if max_dd_pct > 0:
        calmar = float(total_return_pct / max_dd_pct)
    return {
        "return_pct": total_return_pct,
        "drawdown_pct": max_dd_pct,
        "sharpe": sharpe,
        "calmar": calmar,
    }


def run_baseline(name: str, candles: pd.DataFrame, cfg: BaselineConfig) -> dict[str, Any]:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown baseline: {name}. Valid: {sorted(STRATEGIES)}")
    closes = candles["close"].to_numpy(dtype=float)
    timestamps = candles["timestamp"] if "timestamp" in candles.columns else None

    windows = _compute_windows(len(closes), cfg)
    window_results: list[BaselineWindowResult] = []

    for idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
        test_closes = closes[test_start:test_end]
        positions = STRATEGIES[name](test_closes, cfg)
        trades, equity = _simulate(
            test_closes,
            positions,
            fee_rate=cfg.fee_rate,
            slippage_bps=cfg.slippage_bps,
            starting_balance=cfg.starting_balance,
        )
        m = _metrics(equity, cfg.starting_balance, cfg.bars_per_year)
        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        win_rate = wins / len(trades) if trades else 0.0

        ts_start = ts_end = None
        if timestamps is not None and len(timestamps) >= test_end:
            try:
                ts_start = pd.Timestamp(timestamps.iloc[test_start]).strftime("%Y-%m-%d")
                ts_end = pd.Timestamp(timestamps.iloc[test_end - 1]).strftime("%Y-%m-%d")
            except (ValueError, TypeError, KeyError):
                pass

        window_results.append(
            BaselineWindowResult(
                window_idx=idx + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                test_start_date=ts_start,
                test_end_date=ts_end,
                trades=len(trades),
                win_rate=win_rate,
                total_return_pct=m["return_pct"],
                max_drawdown_pct=m["drawdown_pct"],
                sharpe=m["sharpe"],
                calmar=m["calmar"],
                net_positive=m["return_pct"] > 0,
                closed_trades=trades,
            )
        )

    return _build_report(name, cfg, window_results)


def _build_report(
    name: str,
    cfg: BaselineConfig,
    windows: list[BaselineWindowResult],
) -> dict[str, Any]:
    if not windows:
        return {"summary": {"baseline": name, "n_windows": 0}, "windows": []}

    rets = [w.total_return_pct for w in windows]
    dds = [w.max_drawdown_pct for w in windows]
    sharpes = [w.sharpe for w in windows if w.sharpe is not None]
    calmars = [w.calmar for w in windows if w.calmar is not None]
    n_pos = sum(1 for w in windows if w.net_positive)
    total_trades = sum(w.trades for w in windows)

    mean_calmar = float(np.mean(calmars)) if calmars else 0.0
    worst_calmar = float(min(calmars)) if calmars else 0.0
    mean_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
    worst_dd = float(max(dds)) if dds else 0.0

    passes = (
        n_pos >= cfg.min_windows_positive
        and total_trades >= cfg.min_total_trades
        and all(w.trades >= cfg.min_trades_per_window for w in windows)
        and mean_sharpe >= cfg.min_sharpe
        and worst_dd <= cfg.max_drawdown_pct
        and mean_calmar >= cfg.min_calmar
    )

    return {
        "summary": {
            "baseline": name,
            "n_windows": len(windows),
            "n_positive": n_pos,
            "n_failed": 0,
            "total_trades": total_trades,
            "mean_return_pct": round(float(np.mean(rets)), 4) if rets else 0.0,
            "mean_sharpe": round(mean_sharpe, 4),
            "mean_calmar": round(mean_calmar, 4),
            "worst_calmar": round(worst_calmar, 4),
            "mean_drawdown_pct": round(float(np.mean(dds)), 2) if dds else 0.0,
            "worst_drawdown_pct": round(worst_dd, 2),
            "passes_gate": passes,
            "gate_config": {
                "min_sharpe": cfg.min_sharpe,
                "min_calmar": cfg.min_calmar,
                "max_drawdown_pct": cfg.max_drawdown_pct,
                "min_total_trades": cfg.min_total_trades,
                "min_trades_per_window": cfg.min_trades_per_window,
                "min_windows_positive": cfg.min_windows_positive,
            },
        },
        "windows": [
            {
                "window_idx": w.window_idx,
                "train_bars": w.train_end - w.train_start,
                "test_bars": w.test_end - w.test_start,
                "test_start_date": w.test_start_date,
                "test_end_date": w.test_end_date,
                "total_return_pct": round(w.total_return_pct, 4),
                "max_drawdown_pct": round(w.max_drawdown_pct, 2),
                "sharpe": round(w.sharpe, 4) if w.sharpe is not None else None,
                "calmar": round(w.calmar, 4) if w.calmar is not None else None,
                "trades": w.trades,
                "win_rate": round(w.win_rate, 4),
                "net_positive": w.net_positive,
                "closed_trades": w.closed_trades,
            }
            for w in windows
        ],
        "config": {
            "starting_balance": cfg.starting_balance,
            "fee_rate": cfg.fee_rate,
            "slippage_bps": cfg.slippage_bps,
            "min_train_bars": cfg.min_train_bars,
            "min_test_bars": cfg.min_test_bars,
            "n_splits": cfg.n_splits,
        },
    }


def _load_candles(db: str, symbol: str, timeframe: str) -> pd.DataFrame:
    import sys as _sys

    if str(PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(PROJECT_ROOT))
    from hogan_bot.storage import get_connection

    conn = get_connection(db)
    df = pd.read_sql_query(
        "SELECT ts_ms, open, high, low, close, volume FROM candles "
        "WHERE symbol = ? AND timeframe = ? ORDER BY ts_ms",
        conn,
        params=(symbol, timeframe),
    )
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run simple baselines in walk-forward windows")
    parser.add_argument("--db", default="data/hogan.db")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--min-train", type=int, default=16000)
    parser.add_argument("--min-test", type=int, default=1000)
    parser.add_argument(
        "--baseline",
        action="append",
        choices=list(STRATEGIES),
        help="Baseline to run. May be repeated. Defaults to all baselines.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/validation/baselines",
        help="Where to write per-baseline reports and the leaderboard",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = _load_candles(args.db, args.symbol, args.timeframe)
    if df.empty:
        logger.error("No candles for %s %s in %s", args.symbol, args.timeframe, args.db)
        return 1

    cfg = BaselineConfig(
        n_splits=args.n_splits,
        min_train_bars=args.min_train,
        min_test_bars=args.min_test,
    )

    targets = args.baseline or list(STRATEGIES)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()

    summaries: list[dict[str, Any]] = []
    for name in targets:
        report = run_baseline(name, df, cfg)
        out_path = out_dir / f"baseline_{name}_{stamp}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary = report["summary"]
        summaries.append({
            "name": name,
            "report": display_path(out_path),
            **summary,
        })
        logger.info(
            "%-18s ret=%+.2f%% sharpe=%s calmar=%s worst_dd=%.2f%% trades=%d passes=%s",
            name,
            summary.get("mean_return_pct", 0),
            f"{summary.get('mean_sharpe', 0):.2f}",
            f"{summary.get('mean_calmar', 0):.2f}",
            summary.get("worst_drawdown_pct", 0),
            summary.get("total_trades", 0),
            summary.get("passes_gate"),
        )

    leaderboard_path = out_dir / f"baselines_leaderboard_{stamp}.json"
    leaderboard_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "stamp": stamp,
                "db": args.db,
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "n_splits": args.n_splits,
                "min_train": args.min_train,
                "min_test": args.min_test,
                "results": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Baselines leaderboard: {display_path(leaderboard_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
