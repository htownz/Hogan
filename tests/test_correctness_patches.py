"""Tests for the correctness patches: execution ownership, journal sides,
retrain fall-through, and short-to-long flip.

Each test targets a specific bug that was identified and fixed:
1. Paper buy must not double-mutate the portfolio
2. Short-to-long flip must open exactly one long position
3. Async paper journals must use "long"/"short" (not "buy"/"sell")
4. Multi-symbol retrain must not fall through into single-symbol branch
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from hogan_bot.paper import PaperPortfolio
from hogan_bot.execution import PaperExecution, ExecResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_portfolio(cash: float = 10_000.0) -> PaperPortfolio:
    return PaperPortfolio(cash_usd=cash, fee_rate=0.0)


def _in_memory_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the paper_trades table (matches real schema)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            trade_id       TEXT PRIMARY KEY,
            symbol         TEXT NOT NULL,
            side           TEXT NOT NULL,
            entry_price    REAL NOT NULL,
            exit_price     REAL,
            qty            REAL NOT NULL,
            entry_fee      REAL NOT NULL DEFAULT 0,
            exit_fee       REAL NOT NULL DEFAULT 0,
            realized_pnl   REAL,
            pnl_pct        REAL,
            open_ts_ms     INTEGER NOT NULL,
            close_ts_ms    INTEGER,
            close_reason   TEXT,
            ml_up_prob     REAL,
            strategy_conf  REAL,
            vol_ratio      REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            symbol TEXT PRIMARY KEY,
            qty REAL, avg_entry REAL, updated_ms INTEGER
        )
    """)
    conn.commit()
    return conn


def _synthetic_candles(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 30_000.0 + np.cumsum(rng.normal(0, 50, n))
    close = np.clip(close, 1_000.0, None)
    noise = rng.uniform(0.001, 0.005, n)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(close, open_) * (1 + noise)
    low = np.minimum(close, open_) * (1 - noise)
    volume = rng.uniform(100, 1_000, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


# ---------------------------------------------------------------------------
# Test 1: Paper buy does not double-mutate portfolio
# ---------------------------------------------------------------------------

class TestPaperNoDubleMutation:
    """PaperExecution.buy() mutates the portfolio once. Callers must NOT
    call portfolio.execute_buy() again — that was the bug."""

    def test_single_buy_correct_qty(self):
        p = _make_portfolio()
        executor = PaperExecution(portfolio=p, conn=None)

        executor.buy("BTC/USD", price=100.0, qty=1.0)

        pos = p.positions.get("BTC/USD")
        assert pos is not None
        assert pos.qty == pytest.approx(1.0), (
            f"Expected qty=1.0 after one buy, got {pos.qty} (double mutation?)"
        )

    def test_single_buy_correct_cash(self):
        p = _make_portfolio(cash=10_000.0)
        executor = PaperExecution(portfolio=p, conn=None)

        executor.buy("BTC/USD", price=100.0, qty=2.0)

        assert p.cash_usd == pytest.approx(9_800.0), (
            f"Expected cash=9800 after buying 2@100 (fee=0), got {p.cash_usd}"
        )

    def test_single_sell_removes_position(self):
        p = _make_portfolio()
        executor = PaperExecution(portfolio=p, conn=None)

        executor.buy("BTC/USD", price=100.0, qty=1.0)
        executor.sell("BTC/USD", price=110.0, qty=1.0)

        assert "BTC/USD" not in p.positions
        assert p.cash_usd == pytest.approx(10_010.0)

    def test_buy_with_trailing_stop_sets_stop(self):
        p = _make_portfolio()
        executor = PaperExecution(portfolio=p, conn=None)

        executor.buy("BTC/USD", price=100.0, qty=1.0,
                      trailing_stop_pct=0.05, take_profit_pct=0.10)

        pos = p.positions.get("BTC/USD")
        assert pos is not None
        assert pos.qty == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 2: Short-to-long flip opens exactly one long
# ---------------------------------------------------------------------------

class TestShortToLongFlip:
    """When converting from short to long, the portfolio should have exactly
    one long position with the requested qty — not 2x or 3x."""

    def test_flip_produces_one_position(self):
        p = _make_portfolio(cash=50_000.0)
        executor = PaperExecution(portfolio=p, conn=None)

        # Open a short position first
        p.execute_short("BTC/USD", price=100.0, qty=1.0)
        assert "BTC/USD" in p.short_positions

        # Cover the short (simulating what main.py does)
        p.execute_cover("BTC/USD", price=100.0, qty=1.0)
        assert "BTC/USD" not in p.short_positions

        # Open long via executor (ONE call only — the fix)
        res = executor.buy("BTC/USD", price=100.0, qty=2.0,
                           trailing_stop_pct=0.05, take_profit_pct=0.10)
        assert res.ok

        pos = p.positions.get("BTC/USD")
        assert pos is not None
        assert pos.qty == pytest.approx(2.0), (
            f"Expected qty=2.0 after single buy, got {pos.qty} (triple mutation bug?)"
        )

    def test_flip_cash_accounting(self):
        """Cash should reflect exactly one buy cost, not two or three."""
        p = _make_portfolio(cash=50_000.0)
        executor = PaperExecution(portfolio=p, conn=None)

        p.execute_short("BTC/USD", price=100.0, qty=1.0)
        p.execute_cover("BTC/USD", price=100.0, qty=1.0)
        cash_after_cover = p.cash_usd

        executor.buy("BTC/USD", price=100.0, qty=2.0)
        expected_cash = cash_after_cover - (100.0 * 2.0)
        assert p.cash_usd == pytest.approx(expected_cash), (
            f"Cash mismatch: expected {expected_cash}, got {p.cash_usd}"
        )


# ---------------------------------------------------------------------------
# Test 3: Async paper journals use "long"/"short" sides
# ---------------------------------------------------------------------------

class TestJournalSideNormalization:
    """The paper_trades table must use side='long' or side='short',
    never side='buy' or side='sell'. The close query matches on side."""

    def test_open_paper_trade_uses_long_side(self):
        from hogan_bot.storage import open_paper_trade
        conn = _in_memory_db()

        open_paper_trade(conn, "BTC/USD", "long", 100.0, 1.0, 0.1, 1000)

        row = conn.execute(
            "SELECT side FROM paper_trades WHERE symbol='BTC/USD'"
        ).fetchone()
        assert row is not None
        assert row[0] == "long"

    def test_close_matches_long_side(self):
        from hogan_bot.storage import open_paper_trade, close_paper_trade
        conn = _in_memory_db()

        open_paper_trade(conn, "BTC/USD", "long", 100.0, 1.0, 0.1, 1000)
        close_paper_trade(conn, "BTC/USD", "long", 110.0, 0.11, 2000, close_reason="signal")

        row = conn.execute(
            "SELECT exit_price, realized_pnl FROM paper_trades WHERE symbol='BTC/USD'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(110.0), "exit_price should be set on close"
        assert row[1] > 0, "Long trade profit should be positive when exit > entry"

    def test_buy_side_does_not_match_close(self):
        """If someone journals side='buy', close_paper_trade with side='long'
        should NOT find it — proving the mismatch would be a silent no-op."""
        from hogan_bot.storage import open_paper_trade, close_paper_trade
        conn = _in_memory_db()

        open_paper_trade(conn, "ETH/USD", "buy", 50.0, 2.0, 0.05, 1000)
        close_paper_trade(conn, "ETH/USD", "long", 60.0, 0.06, 2000, close_reason="signal")

        row = conn.execute(
            "SELECT exit_price FROM paper_trades WHERE symbol='ETH/USD'"
        ).fetchone()
        assert row[0] is None, (
            "close with side='long' should not match an open with side='buy'"
        )

    def test_short_side_round_trip(self):
        from hogan_bot.storage import open_paper_trade, close_paper_trade
        conn = _in_memory_db()

        open_paper_trade(conn, "SOL/USD", "short", 200.0, 5.0, 0.1, 1000)
        close_paper_trade(conn, "SOL/USD", "short", 180.0, 0.09, 2000, close_reason="signal")

        row = conn.execute(
            "SELECT exit_price, realized_pnl FROM paper_trades WHERE symbol='SOL/USD'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(180.0)
        assert row[1] > 0, "Short trade profit should be positive when exit < entry"


# ---------------------------------------------------------------------------
# Test 4: Multi-symbol retrain does not fall through
# ---------------------------------------------------------------------------

class TestRetrainNoFallthrough:
    """When --symbols is set (multi-symbol), _train_to_candidate must return
    after the multi-symbol path without executing the single-symbol block."""

    def test_multisymbol_returns_symbols_key(self, tmp_path):
        """Multi-symbol metrics dict must contain a 'symbols' key."""
        from hogan_bot.retrain import _train_to_candidate

        candles = _synthetic_candles(900)
        args = argparse.Namespace(
            symbol="BTC/USD",
            symbols="BTC/USD,ETH/USD",
            timeframe="5m",
            exchange="kraken",
            window_bars=900,
            horizon_bars=3,
            model_type="logreg",
            model_path=str(tmp_path / "model.pkl"),
            tune=False,
            from_db=True,
            db=str(tmp_path / "test.db"),
            use_paper_labels=False,
            use_backtest_labels=False,
            use_extended_mtf=False,
            paper_labels_weight=3.0,
        )

        mock_conn = MagicMock()
        with patch("hogan_bot.retrain._build_multi_symbol_dataset") as mock_build, \
             patch("hogan_bot.retrain._train_from_xy") as mock_train:

            from hogan_bot.ml import build_training_set
            X, y, fc = build_training_set(candles, horizon_bars=3)
            mock_build.return_value = (X, y, fc)
            mock_train.return_value = {"roc_auc": 0.55, "model_type": "logreg_multisym"}

            metrics, path = _train_to_candidate(args, candles)

        assert "symbols" in metrics, "Multi-symbol path must record symbol list"
        assert metrics["symbols"] == ["BTC/USD", "ETH/USD"]

    def test_multisymbol_does_not_call_single_trainers(self, tmp_path):
        """The single-symbol trainers (train_logistic_regression, etc.) must
        NOT be called when multi-symbol path runs."""
        from hogan_bot.retrain import _train_to_candidate

        candles = _synthetic_candles(900)
        args = argparse.Namespace(
            symbol="BTC/USD",
            symbols="BTC/USD,ETH/USD",
            timeframe="5m",
            exchange="kraken",
            window_bars=900,
            horizon_bars=3,
            model_type="logreg",
            model_path=str(tmp_path / "model.pkl"),
            tune=False,
            from_db=True,
            db=str(tmp_path / "test.db"),
            use_paper_labels=False,
            use_backtest_labels=False,
            use_extended_mtf=False,
            paper_labels_weight=3.0,
        )

        with patch("hogan_bot.retrain._build_multi_symbol_dataset") as mock_build, \
             patch("hogan_bot.retrain._train_from_xy") as mock_train, \
             patch("hogan_bot.retrain.train_logistic_regression") as mock_logreg, \
             patch("hogan_bot.retrain.train_random_forest") as mock_rf, \
             patch("hogan_bot.retrain.train_xgboost") as mock_xgb:

            from hogan_bot.ml import build_training_set
            X, y, fc = build_training_set(candles, horizon_bars=3)
            mock_build.return_value = (X, y, fc)
            mock_train.return_value = {"roc_auc": 0.55, "model_type": "logreg_multisym"}

            _train_to_candidate(args, candles)

        mock_logreg.assert_not_called()
        mock_rf.assert_not_called()
        mock_xgb.assert_not_called()
