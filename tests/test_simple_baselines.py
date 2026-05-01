from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import simple_baselines as sb


def _synthetic_candles(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.005, n)
    closes = 30_000.0 * np.cumprod(1 + rets)
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ts_ms": (ts.astype("int64") // 10**6).to_numpy(),
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": np.full(n, 10.0),
            "timestamp": ts,
        }
    )


def test_buy_hold_signal_is_constant_long():
    closes = np.array([1.0, 1.1, 1.2, 0.9, 1.0])
    sig = sb._signals_buy_hold(closes)
    assert (sig == 1.0).all()


def test_ma_trend_warmup_is_flat():
    closes = np.linspace(1, 10, 60)
    sig = sb._signals_ma_trend(closes, short=10, long=30)
    assert (sig[:30] == 0.0).all()


def test_rsi_signal_enters_on_drop():
    closes = np.concatenate([np.linspace(100, 90, 30), np.linspace(90, 110, 30)])
    sig = sb._signals_rsi_mean_revert(closes, period=14, buy=30.0, exit_lvl=50.0)
    assert sig.max() == 1.0
    assert sig[0] == 0.0


def test_breakout_signal_triggers_on_new_high():
    closes = np.concatenate([np.full(30, 100.0), np.linspace(100.0, 130.0, 30)])
    sig = sb._signals_breakout(closes, lookback=20)
    assert sig.max() == 1.0


def test_compute_windows_count_matches():
    cfg = sb.BaselineConfig(n_splits=2, min_train_bars=20, min_test_bars=5)
    windows = sb._compute_windows(50, cfg)
    assert len(windows) == 2
    for ws in windows:
        train_start, train_end, test_start, test_end = ws
        assert train_start == 0
        assert test_start >= cfg.min_train_bars
        assert test_end - test_start >= cfg.min_test_bars


def test_compute_windows_rejects_short_data():
    cfg = sb.BaselineConfig(n_splits=2, min_train_bars=20, min_test_bars=5)
    with pytest.raises(ValueError):
        sb._compute_windows(20, cfg)


def test_simulate_long_only_records_pnl_and_fees():
    closes = np.array([100.0, 110.0, 120.0])
    positions = np.array([0.0, 1.0, 0.0])
    trades, equity = sb._simulate(
        closes,
        positions,
        fee_rate=0.0,
        slippage_bps=0.0,
        starting_balance=1000.0,
    )
    assert len(trades) == 1
    trade = trades[0]
    assert trade["side"] == "long"
    assert trade["pnl_pct"] == pytest.approx((120.0 - 110.0) / 110.0 * 100.0)
    assert equity[-1] == pytest.approx(1000.0 * (120.0 / 110.0))


def test_simulate_force_closes_open_position_at_end():
    closes = np.array([100.0, 110.0, 120.0])
    positions = np.array([0.0, 1.0, 1.0])
    trades, _ = sb._simulate(
        closes,
        positions,
        fee_rate=0.0,
        slippage_bps=0.0,
        starting_balance=1000.0,
    )
    assert len(trades) == 1
    assert trades[0]["close_reason"] == "end_of_window"


def test_run_baseline_buy_hold_returns_compatible_report():
    candles = _synthetic_candles(60)
    cfg = sb.BaselineConfig(n_splits=2, min_train_bars=20, min_test_bars=10)
    report = sb.run_baseline("buy_hold", candles, cfg)

    summary = report["summary"]
    assert summary["baseline"] == "buy_hold"
    assert summary["n_windows"] == 2
    assert summary["total_trades"] >= 1
    expected_keys = {
        "mean_return_pct",
        "mean_sharpe",
        "mean_calmar",
        "worst_calmar",
        "worst_drawdown_pct",
        "passes_gate",
    }
    assert expected_keys.issubset(summary.keys())


def test_run_baseline_unknown_raises():
    candles = _synthetic_candles(40)
    cfg = sb.BaselineConfig(n_splits=2, min_train_bars=20, min_test_bars=5)
    with pytest.raises(ValueError):
        sb.run_baseline("unknown_strategy", candles, cfg)


def test_main_writes_reports_and_leaderboard(tmp_path, monkeypatch):
    candles = _synthetic_candles(60)
    monkeypatch.setattr(sb, "_load_candles", lambda *args, **kwargs: candles)

    out_dir = tmp_path / "baselines_out"
    exit_code = sb.main(
        [
            "--db",
            "data/dummy.db",
            "--n-splits",
            "2",
            "--min-train",
            "20",
            "--min-test",
            "10",
            "--baseline",
            "buy_hold",
            "--baseline",
            "ma_trend",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    leaderboards = list(out_dir.glob("baselines_leaderboard_*.json"))
    assert len(leaderboards) == 1
    payload = json.loads(leaderboards[0].read_text(encoding="utf-8"))
    names = {row["name"] for row in payload["results"]}
    assert names == {"buy_hold", "ma_trend"}
    for row in payload["results"]:
        assert "report" in row
        assert "mean_return_pct" in row
