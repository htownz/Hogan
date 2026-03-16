"""Tests for walk-forward validation harness."""
from __future__ import annotations

import pytest
from hogan_bot.walk_forward import WFConfig, WalkForwardReport, WindowResult, _compute_windows


class TestComputeWindows:
    def test_basic_splits(self):
        windows = _compute_windows(total_bars=5000, cfg=WFConfig(n_splits=5, min_train_bars=2000, min_test_bars=200))
        assert len(windows) == 5
        for ts, te, vs, ve in windows:
            assert ts == 0
            assert te == vs
            assert ve > vs
            assert ve <= 5000

    def test_expanding_train(self):
        windows = _compute_windows(total_bars=5000, cfg=WFConfig(n_splits=3, min_train_bars=2000, min_test_bars=200))
        train_sizes = [te - ts for ts, te, _, _ in windows]
        assert train_sizes == sorted(train_sizes)

    def test_no_overlap(self):
        windows = _compute_windows(total_bars=5000, cfg=WFConfig(n_splits=4, min_train_bars=2000, min_test_bars=200))
        for i in range(len(windows) - 1):
            _, _, _, ve_prev = windows[i]
            _, _, vs_next, _ = windows[i + 1]
            assert vs_next >= ve_prev or vs_next == ve_prev

    def test_insufficient_bars_raises(self):
        with pytest.raises(ValueError, match="Not enough bars"):
            _compute_windows(total_bars=500, cfg=WFConfig(min_train_bars=2000))


class TestWalkForwardReport:
    def _make_report(self, returns: list[float], sharpes: list[float]) -> WalkForwardReport:
        cfg = WFConfig(min_windows_positive=3, min_sharpe=0.5, max_drawdown_pct=15.0)
        report = WalkForwardReport(config=cfg)
        for i, (ret, sh) in enumerate(zip(returns, sharpes)):
            report.windows.append(WindowResult(
                window_idx=i, train_start=0, train_end=1000,
                test_start=1000, test_end=1500,
                train_bars=1000, test_bars=500,
                total_return_pct=ret, max_drawdown_pct=5.0,
                sharpe=sh, trades=10, win_rate=0.5,
                net_positive=ret > 0,
            ))
        return report

    def test_passes_gate_all_positive(self):
        report = self._make_report(
            returns=[2.0, 1.5, 3.0, 0.5, 1.0],
            sharpes=[1.2, 0.8, 1.5, 0.6, 0.9],
        )
        assert report.passes_gate is True
        assert report.n_positive == 5

    def test_fails_gate_too_few_positive(self):
        report = self._make_report(
            returns=[2.0, -1.5, -3.0, -0.5, 1.0],
            sharpes=[1.2, -0.5, -1.0, -0.2, 0.9],
        )
        assert report.passes_gate is False
        assert report.n_positive == 2

    def test_fails_gate_low_sharpe(self):
        report = self._make_report(
            returns=[0.1, 0.1, 0.1, 0.1, 0.1],
            sharpes=[0.1, 0.1, 0.1, 0.1, 0.1],
        )
        assert report.passes_gate is False

    def test_summary_structure(self):
        report = self._make_report(
            returns=[2.0, 1.5, 3.0],
            sharpes=[1.2, 0.8, 1.5],
        )
        s = report.summary()
        assert "n_windows" in s
        assert "passes_gate" in s
        assert "mean_sharpe" in s
        assert s["n_windows"] == 3
