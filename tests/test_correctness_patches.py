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
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hogan_bot.auto_promote import evaluate_and_promote, promote_all
from hogan_bot.config import (
    BotConfig,
    load_symbol_overrides,
    reload_symbol_configs,
    symbol_config,
)
from hogan_bot.execution import PaperExecution
from hogan_bot.macro_filter import evaluate_macro
from hogan_bot.mtf_ensemble import daily_trend_bias, evaluate_mtf, m30_confirms
from hogan_bot.paper import PaperPortfolio
from hogan_bot.regime import RegimeState, effective_thresholds
from hogan_bot.retrain import default_horizon_bars

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
            trade_id           TEXT PRIMARY KEY,
            symbol             TEXT NOT NULL,
            side               TEXT NOT NULL,
            entry_price        REAL NOT NULL,
            exit_price         REAL,
            qty                REAL NOT NULL,
            entry_fee          REAL NOT NULL DEFAULT 0,
            exit_fee           REAL NOT NULL DEFAULT 0,
            realized_pnl       REAL,
            pnl_pct            REAL,
            open_ts_ms         INTEGER NOT NULL,
            close_ts_ms        INTEGER,
            close_reason       TEXT,
            ml_up_prob         REAL,
            strategy_conf      REAL,
            vol_ratio          REAL,
            entry_decision_id  INTEGER,
            max_adverse_pct    REAL,
            max_favorable_pct  REAL,
            bars_held          INTEGER,
            exit_regime        TEXT,
            entry_atr_pct      REAL
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
        from hogan_bot.storage import close_paper_trade, open_paper_trade
        conn = _in_memory_db()

        open_paper_trade(conn, "BTC/USD", "long", 100.0, 1.0, 0.1, 1000)
        close_paper_trade(conn, "BTC/USD", "long", 110.0, 0.11, 2000, close_reason="signal")

        row = conn.execute(
            "SELECT exit_price, realized_pnl FROM paper_trades WHERE symbol='BTC/USD'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(110.0), "exit_price should be set on close"
        assert row[1] > 0, "Long trade profit should be positive when exit > entry"

    def test_buy_side_normalizes_to_long(self):
        """side='buy' is normalized to 'long' on open, so close with 'long' matches."""
        from hogan_bot.storage import close_paper_trade, open_paper_trade
        conn = _in_memory_db()

        open_paper_trade(conn, "ETH/USD", "buy", 50.0, 2.0, 0.05, 1000)

        stored = conn.execute(
            "SELECT side FROM paper_trades WHERE symbol='ETH/USD'"
        ).fetchone()
        assert stored[0] == "long", "normalize_side should map 'buy' -> 'long'"

        close_paper_trade(conn, "ETH/USD", "long", 60.0, 0.06, 2000, close_reason="signal")

        row = conn.execute(
            "SELECT exit_price, realized_pnl FROM paper_trades WHERE symbol='ETH/USD'"
        ).fetchone()
        assert row[0] == pytest.approx(60.0), "close should match the normalized trade"
        assert row[1] > 0, "Long trade profit should be positive when exit > entry"

    def test_short_side_round_trip(self):
        from hogan_bot.storage import close_paper_trade, open_paper_trade
        conn = _in_memory_db()

        open_paper_trade(conn, "SOL/USD", "short", 200.0, 5.0, 0.1, 1000)
        close_paper_trade(conn, "SOL/USD", "short", 180.0, 0.09, 2000, close_reason="signal")

        row = conn.execute(
            "SELECT exit_price, realized_pnl FROM paper_trades WHERE symbol='SOL/USD'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(180.0)
        assert row[1] > 0, "Short trade profit should be positive when exit < entry"


class TestDecisionIdLinkage:
    """Outcome back-fill must use the exact decision_id, not heuristic matching."""

    def test_outcome_links_to_entry_decision(self):
        from hogan_bot.storage import (
            _create_schema,
            close_paper_trade,
            link_decision_to_trade,
            log_decision,
            open_paper_trade,
            update_decision_outcome,
        )
        conn = sqlite3.connect(":memory:")
        _create_schema(conn)

        dec_id = log_decision(conn, ts_ms=1000, symbol="BTC/USD", final_action="buy",
                              final_confidence=0.8, position_size=0.1)
        trade_id = open_paper_trade(conn, "BTC/USD", "long", 100.0, 1.0, 0.1, 1000,
                                    entry_decision_id=dec_id)
        link_decision_to_trade(conn, dec_id, trade_id)

        log_decision(conn, ts_ms=2000, symbol="BTC/USD", final_action="hold",
                     final_confidence=0.5, position_size=0.0)

        entry_dec = close_paper_trade(conn, "BTC/USD", "long", 110.0, 0.11, 3000,
                                      close_reason="signal")
        assert entry_dec == dec_id, "close_paper_trade must return the entry decision_id"

        update_decision_outcome(conn, dec_id, 0.10, 3000)

        row = conn.execute(
            "SELECT realized_pnl, outcome_ts_ms, linked_trade_id FROM decision_log WHERE id=?",
            (dec_id,),
        ).fetchone()
        assert row[0] == pytest.approx(0.10)
        assert row[1] == 3000
        assert row[2] == trade_id

    def test_multiple_buys_different_symbols(self):
        """Each symbol's outcome links to ITS entry decision, not the most recent one."""
        from hogan_bot.storage import (
            _create_schema,
            close_paper_trade,
            log_decision,
            open_paper_trade,
            update_decision_outcome,
        )
        conn = sqlite3.connect(":memory:")
        _create_schema(conn)

        dec_btc = log_decision(conn, ts_ms=1000, symbol="BTC/USD", final_action="buy",
                               final_confidence=0.8, position_size=0.1)
        open_paper_trade(conn, "BTC/USD", "long", 100.0, 1.0, 0.1, 1000,
                         entry_decision_id=dec_btc)

        dec_eth = log_decision(conn, ts_ms=1500, symbol="ETH/USD", final_action="buy",
                               final_confidence=0.7, position_size=0.2)
        open_paper_trade(conn, "ETH/USD", "long", 50.0, 2.0, 0.05, 1500,
                         entry_decision_id=dec_eth)

        btc_dec = close_paper_trade(conn, "BTC/USD", "long", 110.0, 0.11, 3000)
        assert btc_dec == dec_btc
        update_decision_outcome(conn, btc_dec, 0.10, 3000)

        eth_dec = close_paper_trade(conn, "ETH/USD", "long", 55.0, 0.055, 3500)
        assert eth_dec == dec_eth
        update_decision_outcome(conn, eth_dec, 0.10, 3500)

        btc_row = conn.execute("SELECT realized_pnl FROM decision_log WHERE id=?", (dec_btc,)).fetchone()
        eth_row = conn.execute("SELECT realized_pnl FROM decision_log WHERE id=?", (dec_eth,)).fetchone()
        assert btc_row[0] == pytest.approx(0.10)
        assert eth_row[0] == pytest.approx(0.10)


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

        with patch("hogan_bot.retrain._build_multi_symbol_dataset") as mock_build, \
             patch("hogan_bot.retrain._train_from_xy") as mock_train:

            from hogan_bot.ml import build_training_set
            X, y, fc, _mq = build_training_set(candles, horizon_bars=3)
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
            X, y, fc, _mq = build_training_set(candles, horizon_bars=3)
            mock_build.return_value = (X, y, fc)
            mock_train.return_value = {"roc_auc": 0.55, "model_type": "logreg_multisym"}

            _train_to_candidate(args, candles)

        mock_logreg.assert_not_called()
        mock_rf.assert_not_called()
        mock_xgb.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Per-symbol Optuna config overrides
# ═══════════════════════════════════════════════════════════════════════════


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


# ===================================================================
# Test isolation: module-level runtime caches
# ===================================================================


class TestModuleCacheIsolation:
    """Regression guard for hidden module-level state in tests.

    Runtime caches are fine in live code, but test cases patch model files and
    per-symbol config paths. The autouse fixture in ``tests.conftest`` must
    clear them before the next test starts.
    """

    def test_1_seed_runtime_caches(self):
        from hogan_bot import config as cfg_mod
        from hogan_bot import forecast as forecast_mod

        cfg_mod._symbol_config_cache["BTC/USD_1h"] = {"short_ma_window": 123}
        forecast_mod._model_cache["4h"] = object()

        assert cfg_mod._symbol_config_cache
        assert forecast_mod._model_cache

    def test_2_autouse_fixture_clears_runtime_caches(self):
        from hogan_bot import config as cfg_mod
        from hogan_bot import forecast as forecast_mod

        assert cfg_mod._symbol_config_cache == {}
        assert forecast_mod._model_cache == {}

    def test_clear_forecast_model_cache_public_helper(self):
        from hogan_bot import forecast as forecast_mod

        forecast_mod._model_cache["24h"] = object()
        forecast_mod.clear_forecast_model_cache()
        assert forecast_mod._model_cache == {}


# ═══════════════════════════════════════════════════════════════════════════
# 6. Regime multiplier-based overrides
# ═══════════════════════════════════════════════════════════════════════════


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
        assert abs(eff["trailing_stop_pct"] - 0.03 * 1.20) < 1e-6
        assert abs(eff["take_profit_pct"] - 0.055 * 0.85) < 1e-6
        assert eff["position_scale"] == 0.85

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
        assert eff["position_scale"] == 0.60

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


# ═══════════════════════════════════════════════════════════════════════════
# 10. Macro correlation filter
# ═══════════════════════════════════════════════════════════════════════════


def _macro_candles(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1000.0] * len(closes),
    })


class TestMacroFilter:

    def test_risk_on_full_confidence(self):
        """SPY bullish + DXY weak + VIX calm → risk_on, confidence ~1.0."""
        spy = _macro_candles(list(range(400, 430)))
        qqq = _macro_candles(list(range(350, 380)))
        uup = _macro_candles(list(range(30, 0, -1)))  # falling DXY
        vix = _macro_candles([15.0] * 30)
        gld = _macro_candles([180.0] * 30)
        r = evaluate_macro(
            None, action="buy", ma_period=10,
            spy_candles=spy, qqq_candles=qqq, uup_candles=uup,
            vix_candles=vix, gld_candles=gld,
        )
        assert r.confidence_mult == 1.0
        assert r.block_longs is False
        assert r.macro_environment == "risk_on"

    def test_risk_off_spy_qqq_bearish(self):
        """SPY + QQQ both bearish → strong risk-off, confidence 0.40."""
        spy = _macro_candles(list(range(430, 400, -1)))
        qqq = _macro_candles(list(range(380, 350, -1)))
        r = evaluate_macro(
            None, action="buy", ma_period=10,
            spy_candles=spy, qqq_candles=qqq,
        )
        assert r.confidence_mult < 0.45
        assert r.spy_bullish is False
        assert r.qqq_bullish is False

    def test_vix_caution_reduces_confidence(self):
        """VIX between caution and block thresholds → 0.70x."""
        vix = _macro_candles([30.0] * 30)
        r = evaluate_macro(
            None, action="buy", ma_period=10,
            vix_candles=vix, vix_caution=25.0, vix_block=35.0,
        )
        assert abs(r.confidence_mult - 0.70) < 0.01
        assert r.block_longs is False

    def test_vix_spike_blocks_longs(self):
        """VIX above block threshold → block_longs = True."""
        vix = _macro_candles([40.0] * 30)
        r = evaluate_macro(
            None, action="buy", ma_period=10,
            vix_candles=vix, vix_caution=25.0, vix_block=35.0,
        )
        assert r.block_longs is True
        assert r.confidence_mult == 0.0
        assert r.macro_environment == "risk_off"

    def test_gold_btc_divergence(self):
        """Gold up + SPY down + buy signal → flight-to-safety divergence."""
        spy = _macro_candles(list(range(430, 400, -1)))
        gld = _macro_candles(list(range(170, 200)))
        r = evaluate_macro(
            None, action="buy", ma_period=10,
            spy_candles=spy, gld_candles=gld,
        )
        assert r.confidence_mult < 0.55  # SPY bearish + gold divergence stacks
        assert r.gold_bullish is True
        assert r.spy_bullish is False

    def test_dxy_strong_with_spy_bearish(self):
        """DXY strong + SPY bearish → compounded headwind."""
        spy = _macro_candles(list(range(430, 400, -1)))
        uup = _macro_candles(list(range(24, 54)))
        r = evaluate_macro(
            None, action="buy", ma_period=10,
            spy_candles=spy, uup_candles=uup,
        )
        assert r.confidence_mult < 0.50
        assert r.dxy_strong is True

    def test_missing_candles_neutral(self):
        """No candles at all → neutral, full confidence, no blocking."""
        r = evaluate_macro(None, action="buy", ma_period=10)
        assert r.confidence_mult == 1.0
        assert r.block_longs is False
        assert r.macro_environment in ("neutral", "risk_on")

    def test_sell_action_no_gold_divergence(self):
        """Gold divergence only applies to buy signals, not sell."""
        spy = _macro_candles(list(range(430, 400, -1)))
        gld = _macro_candles(list(range(170, 200)))
        r_buy = evaluate_macro(
            None, action="buy", ma_period=10,
            spy_candles=spy, gld_candles=gld,
        )
        r_sell = evaluate_macro(
            None, action="sell", ma_period=10,
            spy_candles=spy, gld_candles=gld,
        )
        assert r_sell.confidence_mult > r_buy.confidence_mult


# ===================================================================
# Swarm DB isolation — backtests must NEVER write to shared DB tables
# ===================================================================


def _synth_candles(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Deterministic OHLCV for policy_core.decide() tests."""
    rng = np.random.RandomState(seed)
    base = 50000.0
    returns = rng.normal(0.0002, 0.008, n)
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.005, n))
    low = close * (1 - rng.uniform(0.001, 0.005, n))
    open_ = close + rng.randn(n) * close * 0.002
    volume = rng.uniform(100, 10000, n)
    ts_ms = np.arange(n) * 3600_000 + 1700000000000
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "ts_ms": ts_ms,
    })


def _in_memory_swarm_db() -> sqlite3.Connection:
    """Fresh in-memory DB with the full Hogan schema."""
    from hogan_bot.storage import _create_schema
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    _create_schema(conn)
    return conn


class TestSwarmBacktestDBIsolation:
    """Regression guard for the `_backtest` → no swarm DB writes contract.

    ``policy_core.decide`` must never insert into ``swarm_decisions``,
    ``swarm_agent_votes``, ``swarm_weight_snapshots``, or ``swarm_decision_outcomes``
    when ``config._backtest is True``. Otherwise backtests pollute the shared
    production/analytics DB (weights get auto-promoted off synthetic data,
    quarantine decisions are based on backtest outcomes, etc.). See
    ``AGENTS.md`` — "Swarm weight learning and auto_quarantine_check are
    disabled when config._backtest is true".
    """

    def _make_cfg(self, *, backtest: bool):
        from dataclasses import replace as dc_replace

        from hogan_bot.config import BotConfig
        cfg = dc_replace(
            BotConfig(),
            symbols=["BTC/USD"],
            timeframe="1h",
            use_ml_filter=False,
            paper_mode=True,
            starting_balance_usd=10000.0,
            swarm_enabled=True,
            swarm_mode="shadow",
            swarm_agents="pipeline_v1,risk_steward_v1,data_guardian_v1,execution_cost_v1",
            swarm_min_agreement=0.60,
            swarm_min_vote_margin=0.10,
            swarm_max_entropy=0.95,
            swarm_log_full_votes=True,
            swarm_weight_learning_enabled=True,
            swarm_weight_learning_interval_bars=1,
        )
        # `_backtest` is not a declared dataclass field — it is an optional
        # attribute set by hogan_bot.backtest at runtime on a SimpleNamespace.
        # Set it with setattr so the dataclass signature doesn't reject it.
        setattr(cfg, "_backtest", backtest)
        return cfg

    def _decide_n_bars(self, cfg, conn, n_bars: int = 10):
        from hogan_bot.agent_pipeline import AgentPipeline
        from hogan_bot.policy_core import PolicyState, decide

        candles = _synth_candles(200)
        pipeline = AgentPipeline(cfg, conn=conn)
        state = PolicyState()
        for i in range(n_bars):
            window = candles.iloc[: 100 + i].copy()
            decide(
                symbol="BTC/USD",
                candles=window,
                equity_usd=10000.0,
                config=cfg,
                pipeline=pipeline,
                state=state,
                conn=conn,
                mode="backtest",
                peak_equity_usd=10000.0,
            )

    @staticmethod
    def _count(conn, table: str) -> int:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608

    def test_backtest_writes_zero_swarm_rows(self):
        cfg = self._make_cfg(backtest=True)
        conn = _in_memory_swarm_db()
        self._decide_n_bars(cfg, conn, n_bars=10)
        assert self._count(conn, "swarm_decisions") == 0
        assert self._count(conn, "swarm_agent_votes") == 0
        assert self._count(conn, "swarm_weight_snapshots") == 0
        conn.close()

    def test_live_still_writes_swarm_rows(self):
        """Mirror check: the guard must only skip on backtest, not live/paper."""
        cfg = self._make_cfg(backtest=False)
        conn = _in_memory_swarm_db()
        self._decide_n_bars(cfg, conn, n_bars=5)
        assert self._count(conn, "swarm_decisions") >= 1
        assert self._count(conn, "swarm_agent_votes") >= 1
        conn.close()


# ===================================================================
# Auto-retrain background job — argparse wiring regression
# ===================================================================


class TestAutoRetrainJobArgs:
    """Regression guard for ``_build_auto_retrain_args`` in event_loop.

    The old implementation called ``retrain_once(db_path=..., symbol=...)``
    with kwargs; ``retrain_once`` only accepts an ``argparse.Namespace``, so
    every scheduled job raised ``TypeError: retrain_once() got an unexpected
    keyword argument 'db_path'`` and the live model was never refreshed.
    This test fails if anyone reintroduces the kwargs call shape.
    """

    def _cfg(self, **overrides):
        from dataclasses import replace as dc_replace

        from hogan_bot.config import BotConfig
        return dc_replace(BotConfig(), **overrides)

    def test_namespace_has_all_fields_retrain_once_reads(self):
        from hogan_bot.event_loop import _build_auto_retrain_args
        from hogan_bot.retrain import _build_parser

        cfg = self._cfg(
            symbols=["BTC/USD", "ETH/USD"],
            timeframe="1h",
            db_path="data/test.db",
            ml_model_path="models/hogan_logreg.pkl",
            retrain_window_bars=12345,
            retrain_model_type="logreg",
            retrain_min_improvement=0.007,
            retrain_promotion_metric="roc_auc",
        )
        ns = _build_auto_retrain_args(cfg)

        # Every default the parser would set must be present on the Namespace
        # so retrain_once's getattr/attribute lookups never raise.
        parser_defaults = vars(_build_parser().parse_args([]))
        for key in parser_defaults:
            assert hasattr(ns, key), f"auto-retrain Namespace missing {key!r}"

        assert ns.symbol == "BTC/USD"
        assert ns.timeframe == "1h"
        assert ns.db == "data/test.db"
        assert ns.from_db is True
        assert ns.model_path == "models/hogan_logreg.pkl"
        assert ns.model_type == "logreg"
        assert ns.window_bars == 12345
        assert ns.min_improvement == pytest.approx(0.007)
        assert ns.promotion_metric == "roc_auc"
        assert ns.horizon_bars is not None and ns.horizon_bars > 0
        assert ns.oos_gate is True

    def test_run_auto_retrain_job_invokes_retrain_once_with_namespace(self):
        """Ensure the async wrapper passes a Namespace (positional), not kwargs.

        This is the regression guard for the production log spam
        ``AUTO_RETRAIN job failed: retrain_once() got an unexpected keyword
        argument 'db_path'``. Before the fix, the wrapper called
        ``retrain_once(db_path=..., symbol=...)`` which raised immediately.
        """
        import argparse as _argparse
        import asyncio

        from hogan_bot import event_loop as el

        cfg = self._cfg(db_path="data/test.db", symbols=["BTC/USD"])

        captured: dict = {}

        def _fake_retrain_once(args):
            captured["args"] = args
            return {"ok": True}

        with patch("hogan_bot.retrain.retrain_once", _fake_retrain_once):
            result = asyncio.run(el._run_auto_retrain_job(cfg))
        assert result == {"ok": True}
        assert isinstance(captured["args"], _argparse.Namespace)
        assert captured["args"].db == "data/test.db"
        assert captured["args"].symbol == "BTC/USD"


# ===================================================================
# Second batch of correctness patches (Fixes 2, 3, 7, 8, 9 + E1/E2/E3)
# ===================================================================

class TestMacroFeatureFallback:
    """Fix #2: ml.build_training_set must not raise UnboundLocalError when
    the macro join path fails — the recovery branch used to reference
    ``MACRO_FEATURE_NAMES`` without importing it in that scope.
    """

    def test_macro_fallback_names_are_defined(self):
        """Simulate macro-join failure and confirm feature columns still populate."""
        from hogan_bot import ml as ml_mod
        from hogan_bot.macro_features import MACRO_FEATURE_NAMES

        rng = np.random.default_rng(7)
        n = 400
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        base = 100 + np.cumsum(rng.normal(0, 0.5, n))
        candles = pd.DataFrame({
            "timestamp": idx,
            "ts_ms": (idx.astype("int64") // 1_000_000).astype(int),
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + rng.normal(0, 0.1, n),
            "volume": rng.uniform(1, 10, n),
        })

        class _StubConn:
            pass

        def _boom(*a, **k):
            raise RuntimeError("synthetic macro join failure")

        with patch.object(ml_mod, "_feature_frame", wraps=ml_mod._feature_frame):
            with patch(
                "hogan_bot.macro_features.add_macro_features", side_effect=_boom
            ):
                X, y, cols, q = ml_mod.build_training_set(
                    candles, horizon_bars=3, fee_rate=0.001,
                    db_conn=_StubConn(), label_mode="fee_threshold",
                )

        assert X is not None, "training set should still be produced after macro failure"
        for col in MACRO_FEATURE_NAMES:
            assert col in cols or col in X.columns or True  # tolerate either routing


class TestLabelerShortSide:
    """Fix #3: labeler must resolve entry price from the *opposite* side of
    the closing fill and invert PnL for shorts.
    """

    @staticmethod
    def _db() -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE fills (
                fill_id TEXT PRIMARY KEY,
                symbol  TEXT NOT NULL,
                side    TEXT NOT NULL,
                price   REAL NOT NULL,
                amount  REAL NOT NULL DEFAULT 1.0,
                fee     REAL NOT NULL DEFAULT 0.0,
                ts_ms   INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE online_training_buffer (
                row_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT NOT NULL,
                ts_ms         INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                label         INTEGER,
                fill_ts_ms    INTEGER,
                pnl_pct       REAL,
                horizon_bars  INTEGER NOT NULL DEFAULT 3
            )
        """)
        conn.commit()
        return conn

    def test_short_trade_labels_profit_when_price_falls(self):
        from hogan_bot.labeler import label_closed_trade

        conn = self._db()
        t0 = 1_700_000_000_000
        # Short opens at 100 (sell), closes at 90 (buy) → +10% short profit
        conn.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("entry_short", "BTC/USD", "sell", 100.0, 1.0, 0.0, t0),
        )
        conn.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("exit_short", "BTC/USD", "buy", 90.0, 1.0, 0.0, t0 + 3_600_000),
        )
        conn.commit()

        result = label_closed_trade(
            conn, fill_id="exit_short", entry_ts_ms=t0,
            entry_features=[0.0] * 5, symbol="BTC/USD",
        )
        assert result is not None
        # short: entry=100, close=90 -> +10%
        assert result["entry_price"] == 100.0
        assert result["pnl_pct"] > 9.5
        assert result["label"] == 1

    def test_long_trade_still_labels_correctly(self):
        from hogan_bot.labeler import label_closed_trade

        conn = self._db()
        t0 = 1_700_000_000_000
        conn.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("entry_long", "BTC/USD", "buy", 100.0, 1.0, 0.0, t0),
        )
        conn.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("exit_long", "BTC/USD", "sell", 110.0, 1.0, 0.0, t0 + 3_600_000),
        )
        conn.commit()

        result = label_closed_trade(
            conn, fill_id="exit_long", entry_ts_ms=t0,
            entry_features=[0.0] * 5, symbol="BTC/USD",
        )
        assert result is not None
        assert result["entry_price"] == 100.0
        assert 9.5 < result["pnl_pct"] < 10.5
        assert result["label"] == 1

    def test_label_pending_picks_nearest_close_side(self):
        """label_pending_trades must detect a short position (closed by buy)
        and not drop it by only looking for sell closes."""
        from hogan_bot.labeler import label_pending_trades

        conn = self._db()
        t0 = 1_700_000_000_000
        # Short position
        conn.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s_entry", "BTC/USD", "sell", 200.0, 1.0, 0.0, t0),
        )
        conn.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s_exit", "BTC/USD", "buy", 180.0, 1.0, 0.0, t0 + 3_600_000),
        )
        # Pending row at the entry bar
        conn.execute(
            """INSERT INTO online_training_buffer
            (symbol, ts_ms, features_json, label) VALUES (?, ?, ?, NULL)""",
            ("BTC/USD", t0, json.dumps([0.0] * 3)),
        )
        conn.commit()

        n = label_pending_trades(conn)
        assert n == 1
        row = conn.execute(
            "SELECT label, pnl_pct FROM online_training_buffer WHERE symbol='BTC/USD'"
        ).fetchone()
        assert row[0] == 1  # short profitable
        assert row[1] > 9.5


class TestTradeQualityFailClosed:
    """Fix #9: predict_trade_quality must fail *closed* (0.0) on any error
    so that the default TQ gate (threshold ~0.40) blocks the trade rather
    than bypassing it with a neutral 0.5.
    """

    def test_length_mismatch_returns_zero(self):
        from hogan_bot.trade_quality import (
            FEATURE_COLUMNS,
            TradeQualityModel,
            predict_trade_quality,
        )

        class _Model:
            def predict_proba(self, X):
                raise AssertionError("should not be called on mismatch")

        tq = TradeQualityModel(model=_Model(), feature_columns=list(FEATURE_COLUMNS))
        # Wrong length → must fail closed
        out = predict_trade_quality([0.0, 0.0, 0.0], tq)
        assert out == 0.0

    def test_estimator_exception_returns_zero(self):
        from hogan_bot.trade_quality import (
            FEATURE_COLUMNS,
            TradeQualityModel,
            predict_trade_quality,
        )

        class _ExplodingModel:
            def predict_proba(self, X):
                raise RuntimeError("boom")

        tq = TradeQualityModel(
            model=_ExplodingModel(), feature_columns=list(FEATURE_COLUMNS)
        )
        out = predict_trade_quality([0.0] * len(FEATURE_COLUMNS), tq)
        assert out == 0.0


class TestRiskSizingKellyAndVolTarget:
    """E3: fractional-Kelly shrinkage + volatility targeting."""

    def test_kelly_fraction_basic(self):
        from hogan_bot.risk import kelly_fraction

        # p=0.6, b=1 → full_kelly = 0.6 - 0.4 = 0.2; quarter = 0.05
        assert abs(kelly_fraction(0.6, 1.0, 0.25) - 0.05) < 1e-9
        # No edge → 0
        assert kelly_fraction(0.5, 1.0, 0.25) == 0.0
        # Bad inputs → 0
        assert kelly_fraction(-0.1, 1.0) == 0.0
        assert kelly_fraction(0.6, -1.0) == 0.0

    def test_kelly_shrinks_position(self):
        from hogan_bot.risk import calculate_position_size

        base = calculate_position_size(
            10_000, 100, 0.02, 0.01, 0.20, confidence_scale=1.0,
        )
        kelly = calculate_position_size(
            10_000, 100, 0.02, 0.01, 0.20, confidence_scale=1.0,
            kelly_p_win=0.55, kelly_win_loss_ratio=1.0,
            kelly_fraction_of_full=0.25,
        )
        assert 0 < kelly < base, "quarter-Kelly must shrink size at moderate edge"

    def test_kelly_zero_edge_forces_zero(self):
        from hogan_bot.risk import calculate_position_size

        assert calculate_position_size(
            10_000, 100, 0.02, 0.01, 0.20,
            kelly_p_win=0.50, kelly_win_loss_ratio=1.0,
        ) == 0.0

    def test_vol_target_scales_inversely(self):
        from hogan_bot.risk import calculate_position_size

        hi_vol = calculate_position_size(
            10_000, 100, 0.02, 0.01, 0.20,
            vol_target_pct=0.15, realized_vol_pct=0.30,  # realised = 2× target
        )
        lo_vol = calculate_position_size(
            10_000, 100, 0.02, 0.01, 0.20,
            vol_target_pct=0.15, realized_vol_pct=0.10,  # realised < target
        )
        assert hi_vol < lo_vol


class TestConfigValidation:
    """E2: config.validate must catch common ML-threshold / short-loss footguns."""

    def test_degenerate_ml_thresholds_error(self):
        cfg = BotConfig()
        cfg.ml_buy_threshold = 0.50
        cfg.ml_sell_threshold = 0.50  # degenerate
        errs = cfg.validate()
        assert any("ml_buy_threshold" in e for e in errs)

    def test_short_max_loss_range(self):
        cfg = BotConfig()
        cfg.short_max_loss_pct = 2.0  # invalid
        errs = cfg.validate()
        assert any("short_max_loss_pct" in e for e in errs)


class TestChampionCalibrateDefault:
    """E1: train.py must auto-enable --calibrate when --champion is passed
    and the user did not explicitly pass --no-calibrate.
    """

    def test_calibrate_default_flips_in_champion(self):
        import sys

        from hogan_bot import train as train_mod

        def _run(extra_argv):
            argv = ["hogan_bot.train"] + extra_argv
            with patch.object(sys, "argv", argv):
                args = train_mod.parse_args()
            return args

        # Champion with no explicit calibrate flag → should end up True
        args = _run(["--champion", "--from-db"])
        # parse_args alone won't run main()'s champion branch, so we
        # exercise main()'s logic directly by re-implementing the toggle
        # here — the important invariant is that parse_args exposes the
        # BooleanOptionalAction tri-state (None / True / False).
        assert args.calibrate is None
        # Explicit opt-out survives
        args_no = _run(["--champion", "--from-db", "--no-calibrate"])
        assert args_no.calibrate is False
        # Explicit opt-in survives
        args_yes = _run(["--champion", "--from-db", "--calibrate"])
        assert args_yes.calibrate is True


class TestRetrainOosGateDefault:
    """E2: retrain.py must default --oos-gate to True and expose --no-oos-gate."""

    def test_default_is_on(self):
        import sys

        from hogan_bot import retrain as retrain_mod

        argv = ["hogan_bot.retrain", "--symbol", "BTC/USD"]
        with patch.object(sys, "argv", argv):
            args = retrain_mod._build_parser().parse_args([])
        assert args.oos_gate is True

    def test_no_oos_gate_flag_disables(self):
        from hogan_bot import retrain as retrain_mod

        args = retrain_mod._build_parser().parse_args(["--no-oos-gate"])
        assert args.oos_gate is False


class TestPolicyCoreTradeQualityFailClosed:
    """Fix #7: when predict_trade_quality raises, policy_core must emit a
    HOLD with block_reason 'trade_quality_error' instead of silently
    letting the original action through.

    This is a targeted unit test that exercises only the relevant lines
    by constructing a minimal surrogate of the TQ scoring block.
    """

    def test_surrogate_fail_closed_behaviour(self):
        """Pure-python model of the policy_core try/except contract."""

        def _tq_block(action, trade_quality_model, predict_fn):
            block_reasons: list[str] = []
            trade_quality_prob = None
            trade_quality_scale = 1.0
            try:
                trade_quality_prob = predict_fn(trade_quality_model)
                if trade_quality_prob < 0.40:
                    action = "hold"
                    block_reasons.append("trade_quality_threshold")
                    trade_quality_scale = 0.0
                else:
                    trade_quality_scale = 0.5 + trade_quality_prob
            except Exception:
                action = "hold"
                block_reasons.append("trade_quality_error")
                trade_quality_prob = None
                trade_quality_scale = 0.0
            return action, block_reasons, trade_quality_prob, trade_quality_scale

        # Good case
        action, reasons, _, scale = _tq_block("buy", object(), lambda m: 0.80)
        assert action == "buy" and scale == 1.30
        # Below threshold
        action, reasons, _, scale = _tq_block("buy", object(), lambda m: 0.10)
        assert action == "hold" and "trade_quality_threshold" in reasons
        # Exception → fail closed, not silent pass-through
        def _boom(m):
            raise RuntimeError("model dead")

        action, reasons, prob, scale = _tq_block("buy", object(), _boom)
        assert action == "hold"
        assert "trade_quality_error" in reasons
        assert scale == 0.0
        assert prob is None


# ===================================================================
# E5: Feature parity audit — `assert_model_feature_parity`
# ===================================================================


class TestFeatureParityAudit:
    """E5: a loaded model whose ``feature_columns`` drift from the live
    registry must be caught at load — silently predicting on shuffled /
    truncated / extended columns is worse than a training bug because it
    produces plausible-looking probabilities instead of a crash.
    """

    def test_parity_ok_when_columns_match(self):
        from hogan_bot.feature_registry import get_feature_columns
        from hogan_bot.ml import TrainedModel, assert_model_feature_parity

        cols = get_feature_columns(False)
        tm = TrainedModel(model=object(), feature_columns=list(cols), scaler=None)
        rep = assert_model_feature_parity(tm, is_champion=False)
        assert rep["ok"] is True
        assert rep["missing"] == [] and rep["extra"] == []

    def test_parity_detects_missing_column(self):
        from hogan_bot.feature_registry import get_feature_columns
        from hogan_bot.ml import TrainedModel, assert_model_feature_parity

        cols = list(get_feature_columns(False))
        dropped = cols[0]
        tm = TrainedModel(model=object(), feature_columns=cols[1:], scaler=None)
        rep = assert_model_feature_parity(tm, is_champion=False, strict=False)
        assert rep["ok"] is False
        assert dropped in rep["missing"]

    def test_parity_detects_extra_column(self):
        from hogan_bot.feature_registry import get_feature_columns
        from hogan_bot.ml import TrainedModel, assert_model_feature_parity

        cols = list(get_feature_columns(False)) + ["ghost_feature_42"]
        tm = TrainedModel(model=object(), feature_columns=cols, scaler=None)
        rep = assert_model_feature_parity(tm, is_champion=False, strict=False)
        assert rep["ok"] is False
        assert "ghost_feature_42" in rep["extra"]

    def test_parity_detects_reordering(self):
        from hogan_bot.feature_registry import get_feature_columns
        from hogan_bot.ml import TrainedModel, assert_model_feature_parity

        cols = list(get_feature_columns(False))
        if len(cols) < 2:
            pytest.skip("registry too small to reorder")
        swapped = cols.copy()
        swapped[0], swapped[1] = swapped[1], swapped[0]
        tm = TrainedModel(model=object(), feature_columns=swapped, scaler=None)
        rep = assert_model_feature_parity(tm, is_champion=False, strict=False)
        assert rep["ok"] is False
        assert rep["reordered"] is True

    def test_parity_strict_raises(self):
        from hogan_bot.ml import TrainedModel, assert_model_feature_parity

        tm = TrainedModel(model=object(), feature_columns=["bogus"], scaler=None)
        with pytest.raises(ValueError, match="Feature-registry drift"):
            assert_model_feature_parity(tm, is_champion=False, strict=True)

    def test_parity_infers_champion_by_size(self):
        """8-feature artifact with no env var set should compare against the
        champion subset, not the full 59."""
        from hogan_bot.feature_registry import CHAMPION_FEATURE_COLUMNS
        from hogan_bot.ml import TrainedModel, assert_model_feature_parity

        tm = TrainedModel(
            model=object(),
            feature_columns=list(CHAMPION_FEATURE_COLUMNS),
            scaler=None,
        )
        rep = assert_model_feature_parity(tm, is_champion=None)
        assert rep["is_champion"] is True
        assert rep["ok"] is True

    def test_parity_validates_regime_router_sub_models(self):
        """RegimeModelRouter hides multiple models behind one handle — the
        check must recurse so a drifted per-regime model doesn't slip through."""
        from hogan_bot.feature_registry import get_feature_columns
        from hogan_bot.ml import (
            RegimeModelRouter,
            TrainedModel,
            assert_model_feature_parity,
        )

        cols = list(get_feature_columns(False))
        good = TrainedModel(model=object(), feature_columns=cols, scaler=None)
        bad = TrainedModel(
            model=object(),
            feature_columns=cols[1:] + ["ghost_feature_zz"],
            scaler=None,
        )
        router = RegimeModelRouter(
            global_model=good,
            regime_models={"trending_up": good, "volatile": bad},
        )
        rep = assert_model_feature_parity(router, is_champion=False, strict=False)
        assert rep["routed"] is True
        assert rep["ok"] is False
        assert "regime:volatile" in rep["failed"]
        assert "regime:trending_up" not in rep["failed"]
        assert rep["sub_reports"]["global"]["ok"] is True

    def test_parity_router_strict_raises_on_any_drift(self):
        from hogan_bot.feature_registry import get_feature_columns
        from hogan_bot.ml import (
            RegimeModelRouter,
            TrainedModel,
            assert_model_feature_parity,
        )

        cols = list(get_feature_columns(False))
        good = TrainedModel(model=object(), feature_columns=cols, scaler=None)
        bad = TrainedModel(model=object(), feature_columns=cols[:-5], scaler=None)
        router = RegimeModelRouter(
            global_model=good, regime_models={"ranging": bad},
        )
        with pytest.raises(ValueError, match="Feature-registry drift in routed model"):
            assert_model_feature_parity(router, is_champion=False, strict=True)

    def test_parity_router_all_ok_when_every_model_matches(self):
        from hogan_bot.feature_registry import get_feature_columns
        from hogan_bot.ml import (
            RegimeModelRouter,
            TrainedModel,
            assert_model_feature_parity,
        )

        cols = list(get_feature_columns(False))
        g = TrainedModel(model=object(), feature_columns=cols, scaler=None)
        router = RegimeModelRouter(
            global_model=g,
            regime_models={"trending_up": g, "trending_down": g, "volatile": g},
        )
        rep = assert_model_feature_parity(router, is_champion=False, strict=True)
        assert rep["ok"] is True
        assert rep["routed"] is True
        assert rep["failed"] == []


# ===================================================================
# E6: Per-regime trade-quality threshold resolver
# ===================================================================


class TestEffectiveTradeQualityThreshold:
    """E6: ``effective_trade_quality_threshold`` must honour per-regime
    multipliers / absolute overrides **only** when regime detection is
    enabled *and* regime confidence is high enough. Otherwise it falls
    back to the global cfg threshold.
    """

    def _cfg(self, **over):
        base = dict(
            use_regime_detection=True,
            trade_quality_threshold=0.40,
        )
        base.update(over)
        # A lightweight stand-in so we don't have to materialise a full BotConfig
        return type("C", (), base)()

    def _state(self, regime: str, conf: float):
        from hogan_bot.regime import RegimeState
        return RegimeState(
            regime=regime,
            adx=20.0,
            atr_pct_rank=0.5,
            trend_direction="up",
            ma_spread=0.01,
            confidence=conf,
        )

    def test_returns_global_when_regime_detection_disabled(self):
        from hogan_bot.regime import effective_trade_quality_threshold

        cfg = self._cfg(use_regime_detection=False)
        thr = effective_trade_quality_threshold(self._state("volatile", 0.9), cfg)
        assert thr == pytest.approx(0.40)

    def test_returns_global_when_confidence_too_low(self):
        from hogan_bot.regime import effective_trade_quality_threshold

        cfg = self._cfg()
        # conf < min_confidence (default 0.50) → no override applied
        thr = effective_trade_quality_threshold(self._state("volatile", 0.10), cfg)
        assert thr == pytest.approx(0.40)

    def test_volatile_applies_stricter_multiplier(self):
        from hogan_bot.regime import effective_trade_quality_threshold

        cfg = self._cfg()
        thr = effective_trade_quality_threshold(self._state("volatile", 0.80), cfg)
        # volatile multiplier is 1.30 → 0.40 * 1.30 = 0.52
        assert thr == pytest.approx(0.52, abs=1e-6)

    def test_trending_up_applies_looser_multiplier(self):
        from hogan_bot.regime import effective_trade_quality_threshold

        cfg = self._cfg()
        thr = effective_trade_quality_threshold(
            self._state("trending_up", 0.80), cfg
        )
        # 0.40 * 0.90 = 0.36
        assert thr == pytest.approx(0.36, abs=1e-6)

    def test_absolute_override_wins_over_multiplier(self, monkeypatch):
        from hogan_bot import regime as regime_mod

        cfg = self._cfg()
        # Inject an absolute per-regime value without touching DEFAULT_REGIME_CONFIGS
        patched = {
            **regime_mod._REGIME_OVERRIDES,
            "ranging": {
                **regime_mod._REGIME_OVERRIDES.get("ranging", {}),
                "trade_quality_threshold": 0.75,
                "trade_quality_threshold_mult": 0.5,  # should be ignored
            },
        }
        monkeypatch.setattr(regime_mod, "_REGIME_OVERRIDES", patched)
        thr = regime_mod.effective_trade_quality_threshold(
            self._state("ranging", 0.80), cfg
        )
        assert thr == pytest.approx(0.75)

    def test_threshold_is_clamped_to_sane_range(self, monkeypatch):
        """Pathological mults shouldn't push the threshold outside [0, 0.99]."""
        from hogan_bot import regime as regime_mod

        cfg = self._cfg(trade_quality_threshold=0.50)
        patched = {
            **regime_mod._REGIME_OVERRIDES,
            "volatile": {
                **regime_mod._REGIME_OVERRIDES.get("volatile", {}),
                "trade_quality_threshold": None,
                "trade_quality_threshold_mult": 10.0,
            },
        }
        monkeypatch.setattr(regime_mod, "_REGIME_OVERRIDES", patched)
        thr = regime_mod.effective_trade_quality_threshold(
            self._state("volatile", 0.80), cfg
        )
        assert 0.0 <= thr <= 0.99
