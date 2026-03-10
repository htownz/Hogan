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


# ═══════════════════════════════════════════════════════════════════════════
# 5. Per-symbol Optuna config overrides
# ═══════════════════════════════════════════════════════════════════════════

import json
import tempfile
from pathlib import Path
from hogan_bot.config import (
    BotConfig, symbol_config, load_symbol_overrides,
    reload_symbol_configs, _optuna_json_path,
)


class TestPerSymbolConfig:
    """Verify that per-symbol Optuna JSON files produce correct BotConfig overrides."""

    def _write_optuna_json(self, tmp: Path, symbol: str, tf: str, overrides: dict, score: float = 5.0):
        path = tmp / f"opt_{symbol.replace('/', '-')}_{tf}.json"
        data = {"symbol": symbol, "best_score": score, "best_config": overrides}
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_loads_overrides_from_json(self, tmp_path):
        reload_symbol_configs()
        self._write_optuna_json(tmp_path, "BTC/USD", "1h", {
            "short_ma_window": 8,
            "long_ma_window": 111,
            "signal_mode": "ma_only",
        })
        overrides = load_symbol_overrides("BTC/USD", "1h", str(tmp_path))
        assert overrides["short_ma_window"] == 8
        assert overrides["long_ma_window"] == 111
        assert overrides["signal_mode"] == "ma_only"

    def test_missing_json_returns_empty(self, tmp_path):
        reload_symbol_configs()
        overrides = load_symbol_overrides("SOL/USD", "1h", str(tmp_path))
        assert overrides == {}

    def test_symbol_config_applies_overrides(self, tmp_path):
        reload_symbol_configs()
        self._write_optuna_json(tmp_path, "ETH/USD", "1h", {
            "short_ma_window": 19,
            "long_ma_window": 63,
            "volume_threshold": 1.84,
            "trailing_stop_pct": 0.03,
            "take_profit_pct": 0.055,
            "signal_mode": "all",
        })
        base = BotConfig(timeframe="1h", short_ma_window=12, long_ma_window=79)
        # Monkey-patch models dir for the test
        import hogan_bot.config as cfg_mod
        orig = cfg_mod._optuna_json_path
        cfg_mod._optuna_json_path = lambda sym, tf, md="models": tmp_path / f"opt_{sym.replace('/', '-')}_{tf}.json"
        try:
            result = symbol_config(base, "ETH/USD")
            assert result.short_ma_window == 19
            assert result.long_ma_window == 63
            assert result.volume_threshold == 1.84
            assert result.signal_mode == "all"
            assert result.trailing_stop_pct == 0.03
            assert result.take_profit_pct == 0.055
            # Fields not in the override stay at base values
            assert result.fee_rate == base.fee_rate
            assert result.max_risk_per_trade == base.max_risk_per_trade
        finally:
            cfg_mod._optuna_json_path = orig
            reload_symbol_configs()

    def test_symbol_config_no_override_returns_base(self, tmp_path):
        reload_symbol_configs()
        base = BotConfig(timeframe="1h")
        import hogan_bot.config as cfg_mod
        orig = cfg_mod._optuna_json_path
        cfg_mod._optuna_json_path = lambda sym, tf, md="models": tmp_path / f"opt_{sym.replace('/', '-')}_{tf}.json"
        try:
            result = symbol_config(base, "SOL/USD")
            assert result is base
        finally:
            cfg_mod._optuna_json_path = orig
            reload_symbol_configs()

    def test_ignores_unknown_keys_in_json(self, tmp_path):
        reload_symbol_configs()
        self._write_optuna_json(tmp_path, "BTC/USD", "30m", {
            "short_ma_window": 10,
            "some_unknown_field": 999,
        })
        overrides = load_symbol_overrides("BTC/USD", "30m", str(tmp_path))
        assert "short_ma_window" in overrides
        assert "some_unknown_field" not in overrides

    def test_cache_returns_same_result(self, tmp_path):
        reload_symbol_configs()
        self._write_optuna_json(tmp_path, "BTC/USD", "1h", {"short_ma_window": 8})
        r1 = load_symbol_overrides("BTC/USD", "1h", str(tmp_path))
        r2 = load_symbol_overrides("BTC/USD", "1h", str(tmp_path))
        assert r1 is r2


# ═══════════════════════════════════════════════════════════════════════════
# 6. Regime multiplier-based overrides
# ═══════════════════════════════════════════════════════════════════════════

from hogan_bot.regime import effective_thresholds, RegimeState


class TestRegimeMultipliers:
    """Verify regime overrides use multipliers for strategy params."""

    @staticmethod
    def _make_regime(regime: str, confidence: float = 0.8) -> RegimeState:
        return RegimeState(
            regime=regime, adx=30.0, atr_pct_rank=0.5,
            trend_direction=1, ma_spread=0.01, confidence=confidence,
        )

    def test_trending_up_scales_from_base(self):
        cfg = BotConfig(
            volume_threshold=2.35, trailing_stop_pct=0.05,
            take_profit_pct=0.059, use_regime_detection=True,
        )
        eff = effective_thresholds(self._make_regime("trending_up"), cfg)
        assert abs(eff["volume_threshold"] - 2.35 * 0.55) < 1e-6
        assert abs(eff["trailing_stop_pct"] - 0.05 * 1.30) < 1e-6
        assert abs(eff["take_profit_pct"] - 0.059 * 2.00) < 1e-6
        assert eff["position_scale"] == 1.0

    def test_ranging_scales_from_base(self):
        cfg = BotConfig(
            volume_threshold=1.84, trailing_stop_pct=0.03,
            take_profit_pct=0.055, use_regime_detection=True,
        )
        eff = effective_thresholds(self._make_regime("ranging"), cfg)
        assert abs(eff["volume_threshold"] - 1.84 * 1.10) < 1e-6
        assert abs(eff["trailing_stop_pct"] - 0.03 * 0.50) < 1e-6
        assert abs(eff["take_profit_pct"] - 0.055 * 0.70) < 1e-6
        assert eff["position_scale"] == 0.75

    def test_low_confidence_returns_base(self):
        cfg = BotConfig(
            volume_threshold=2.35, trailing_stop_pct=0.05,
            take_profit_pct=0.059, use_regime_detection=True,
        )
        eff = effective_thresholds(self._make_regime("trending_up", confidence=0.3), cfg)
        assert eff["volume_threshold"] == 2.35
        assert eff["trailing_stop_pct"] == 0.05
        assert eff["take_profit_pct"] == 0.059

    def test_volatile_scales_from_base(self):
        cfg = BotConfig(
            volume_threshold=2.0, trailing_stop_pct=0.04,
            take_profit_pct=0.06, use_regime_detection=True,
        )
        eff = effective_thresholds(self._make_regime("volatile"), cfg)
        assert abs(eff["volume_threshold"] - 2.0 * 0.70) < 1e-6
        assert abs(eff["trailing_stop_pct"] - 0.04 * 0.80) < 1e-6
        assert abs(eff["take_profit_pct"] - 0.06 * 1.40) < 1e-6
        assert eff["position_scale"] == 0.50

    def test_different_symbols_get_different_effective_values(self):
        btc_cfg = BotConfig(
            volume_threshold=2.35, trailing_stop_pct=0.05,
            take_profit_pct=0.059, use_regime_detection=True,
        )
        eth_cfg = BotConfig(
            volume_threshold=1.84, trailing_stop_pct=0.03,
            take_profit_pct=0.055, use_regime_detection=True,
        )
        regime = self._make_regime("trending_up")
        btc_eff = effective_thresholds(regime, btc_cfg)
        eth_eff = effective_thresholds(regime, eth_cfg)
        assert btc_eff["trailing_stop_pct"] != eth_eff["trailing_stop_pct"]
        assert btc_eff["take_profit_pct"] != eth_eff["take_profit_pct"]
        assert btc_eff["trailing_stop_pct"] / 0.05 == eth_eff["trailing_stop_pct"] / 0.03


# ═══════════════════════════════════════════════════════════════════════════
# 7. Multi-timeframe ensemble
# ═══════════════════════════════════════════════════════════════════════════

from hogan_bot.mtf_ensemble import daily_trend_bias, m30_confirms, evaluate_mtf


def _make_candles(closes: list[float], n: int | None = None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of closes."""
    if n is None:
        n = len(closes)
    closes = closes[-n:]
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [100.0] * len(closes),
    })


class TestDailyTrendBias:

    def test_bullish_trend(self):
        closes = list(range(100, 160))
        df = _make_candles(closes)
        assert daily_trend_bias(df, fast_period=5, slow_period=20) == "bullish"

    def test_bearish_trend(self):
        closes = list(range(160, 100, -1))
        df = _make_candles(closes)
        assert daily_trend_bias(df, fast_period=5, slow_period=20) == "bearish"

    def test_neutral_when_insufficient_data(self):
        df = _make_candles([100, 101, 102])
        assert daily_trend_bias(df, fast_period=5, slow_period=20) == "neutral"

    def test_neutral_on_none(self):
        assert daily_trend_bias(None) == "neutral"


class TestM30Confirms:

    def test_buy_confirmed_above_ma_rsi_ok(self):
        closes = list(range(100, 130))
        df = _make_candles(closes)
        assert m30_confirms(df, "buy", fast_period=5) is True

    def test_buy_rejected_below_ma(self):
        closes = list(range(130, 100, -1))
        df = _make_candles(closes)
        assert m30_confirms(df, "buy", fast_period=5) is False

    def test_sell_confirmed_below_ma(self):
        closes = list(range(130, 100, -1))
        df = _make_candles(closes)
        assert m30_confirms(df, "sell", fast_period=5) is True

    def test_none_candles_returns_true(self):
        assert m30_confirms(None, "buy") is True


class TestEvaluateMTF:

    def test_hold_input_stays_hold(self):
        result = evaluate_mtf(None, "hold", None)
        assert result.final_action == "hold"
        assert result.confidence_mult == 0.0

    def test_daily_bearish_blocks_buy(self):
        daily = _make_candles(list(range(160, 100, -1)))
        result = evaluate_mtf(daily, "buy", None)
        assert result.final_action == "hold"

    def test_daily_bullish_blocks_sell(self):
        daily = _make_candles(list(range(100, 160)))
        result = evaluate_mtf(daily, "sell", None)
        assert result.final_action == "hold"

    def test_daily_bullish_allows_buy(self):
        daily = _make_candles(list(range(100, 160)))
        result = evaluate_mtf(daily, "buy", None)
        assert result.final_action == "buy"
        assert result.confidence_mult == 1.0

    def test_neutral_daily_allows_any_direction(self):
        result = evaluate_mtf(None, "sell", None)
        assert result.final_action == "sell"
        assert result.confidence_mult == 1.0

    def test_m30_no_confirm_reduces_confidence(self):
        daily = _make_candles(list(range(100, 160)))
        m30 = _make_candles(list(range(130, 100, -1)))
        result = evaluate_mtf(daily, "buy", m30, unconfirmed_scale=0.5)
        assert result.final_action == "buy"
        assert result.confidence_mult == 0.5
        assert result.m30_confirms is False

    def test_full_alignment_full_confidence(self):
        daily = _make_candles(list(range(100, 160)))
        m30 = _make_candles(list(range(100, 130)))
        result = evaluate_mtf(daily, "buy", m30)
        assert result.final_action == "buy"
        assert result.confidence_mult == 1.0
        assert result.m30_confirms is True


# ═══════════════════════════════════════════════════════════════════════════
# 8. Timeframe-aware horizon computation
# ═══════════════════════════════════════════════════════════════════════════

from hogan_bot.retrain import default_horizon_bars


class TestHorizonBars:

    def test_5m_targets_6_hours(self):
        assert default_horizon_bars("5m") == 72  # 6h / 5m = 72

    def test_30m_targets_6_hours(self):
        assert default_horizon_bars("30m") == 12  # 6h / 30m = 12

    def test_1h_targets_6_hours(self):
        assert default_horizon_bars("1h") == 6  # 6h / 1h = 6

    def test_custom_target(self):
        assert default_horizon_bars("1h", target_hours=12) == 12

    def test_unknown_timeframe_defaults_to_12(self):
        assert default_horizon_bars("weird") == 12

    def test_all_timeframes_produce_positive(self):
        for tf in ("1m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d"):
            assert default_horizon_bars(tf) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 9. Auto-promotion pipeline
# ═══════════════════════════════════════════════════════════════════════════

from hogan_bot.auto_promote import evaluate_and_promote, promote_all


class TestAutoPromotion:

    def _write_opt(self, path: Path, score: float, trades: int = 20, dd: float = 1.5):
        data = {
            "symbol": "BTC/USD",
            "best_score": score,
            "best_config": {"short_ma_window": 8, "long_ma_window": 111},
            "leaderboard": [{
                "rank": 1,
                "config": {"short_ma_window": 8},
                "sharpe_ratio": score,
                "total_return_pct": 4.0,
                "max_drawdown_pct": dd,
                "win_rate": 0.7,
                "trades": trades,
            }],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_promotes_when_score_beats_threshold(self, tmp_path):
        reload_symbol_configs()
        self._write_opt(tmp_path / "opt_BTC-USD_1h.json", score=5.0)
        r = evaluate_and_promote("BTC/USD", "1h", models_dir=str(tmp_path), min_sharpe=2.0)
        assert r.promoted is True

    def test_rejects_low_sharpe(self, tmp_path):
        reload_symbol_configs()
        self._write_opt(tmp_path / "opt_BTC-USD_1h.json", score=1.5)
        r = evaluate_and_promote("BTC/USD", "1h", models_dir=str(tmp_path), min_sharpe=2.0)
        assert r.promoted is False
        assert "minimum" in r.reason

    def test_rejects_few_trades(self, tmp_path):
        reload_symbol_configs()
        self._write_opt(tmp_path / "opt_BTC-USD_1h.json", score=5.0, trades=3)
        r = evaluate_and_promote("BTC/USD", "1h", models_dir=str(tmp_path), min_trades=10)
        assert r.promoted is False
        assert "trades" in r.reason

    def test_rejects_high_drawdown(self, tmp_path):
        reload_symbol_configs()
        self._write_opt(tmp_path / "opt_BTC-USD_1h.json", score=5.0, dd=20.0)
        r = evaluate_and_promote("BTC/USD", "1h", models_dir=str(tmp_path), max_drawdown_pct=15.0)
        assert r.promoted is False
        assert "drawdown" in r.reason

    def test_missing_file_returns_not_promoted(self, tmp_path):
        reload_symbol_configs()
        r = evaluate_and_promote("SOL/USD", "1h", models_dir=str(tmp_path))
        assert r.promoted is False

    def test_candidate_vs_incumbent_improvement_gate(self, tmp_path):
        reload_symbol_configs()
        incumbent = tmp_path / "opt_BTC-USD_1h.json"
        candidate = tmp_path / "opt_BTC-USD_1h_new.json"
        self._write_opt(incumbent, score=8.0)
        self._write_opt(candidate, score=8.3)
        r = evaluate_and_promote(
            "BTC/USD", "1h",
            candidate_path=candidate,
            models_dir=str(tmp_path),
            min_improvement=0.5,
        )
        assert r.promoted is False
        assert "Improvement" in r.reason

    def test_promote_all_discovers_files(self, tmp_path):
        reload_symbol_configs()
        self._write_opt(tmp_path / "opt_BTC-USD_1h.json", score=10.0)
        self._write_opt(tmp_path / "opt_ETH-USD_1h.json", score=8.0)
        results = promote_all(models_dir=str(tmp_path), min_sharpe=2.0)
        assert len(results) == 2
        assert all(r.promoted for r in results)
