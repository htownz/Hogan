"""Tests for the Champion Path — canonical production settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import patch

from hogan_bot.champion import (
    CHAMPION_LOCKS,
    ChampionLocks,
    apply_champion_mode,
    get_champion_summary,
    is_champion_mode,
)


class TestIsChampionMode:
    def test_default_is_off(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HOGAN_CHAMPION_MODE", None)
            assert is_champion_mode() is False

    def test_env_var_enables(self):
        with patch.dict(os.environ, {"HOGAN_CHAMPION_MODE": "1"}):
            assert is_champion_mode() is True


class TestChampionLocks:
    def test_locks_are_frozen_defaults(self):
        assert isinstance(CHAMPION_LOCKS, ChampionLocks)
        assert CHAMPION_LOCKS.use_ema_clouds is False
        assert CHAMPION_LOCKS.use_fvg is False
        assert CHAMPION_LOCKS.use_rl_agent is False


class TestApplyChampionMode:
    def test_no_op_when_mode_off(self):
        from types import SimpleNamespace
        config = SimpleNamespace(
            use_ema_clouds=True,
            use_fvg=True,
            use_rl_agent=True,
            use_ict=True,
        )
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HOGAN_CHAMPION_MODE", None)
            result = apply_champion_mode(config)
        assert result.use_ema_clouds is True
        assert result is config

    def test_warns_on_experimental_features(self):
        from types import SimpleNamespace
        config = SimpleNamespace(
            use_ict=True,
            use_rl_agent=True,
        )
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HOGAN_CHAMPION_MODE", None)
            with patch("hogan_bot.champion.logger") as mock_logger:
                apply_champion_mode(config)
                mock_logger.warning.assert_called_once()

    def test_champion_mode_uses_replace_on_dataclass(self):
        """Champion mode applies overrides via dataclasses.replace()."""
        @dataclass
        class FakeConfig:
            use_ema_clouds: bool = True
            use_fvg: bool = True
            use_rl_agent: bool = True
            use_ict: bool = True
            use_mtf_ensemble: bool = False
            use_macro_filter: bool = False
            use_online_learning: bool = False
            use_mtf_extended: bool = True
            signal_mode: str = "any"
            signal_min_vote_margin: int = 2
            use_regime_detection: bool = True
            ml_confidence_sizing: bool = True
            use_strategy_router: bool = True
            volatile_policy: str = "breakout"
            min_hold_bars: int = 3
            exit_confirmation_bars: int = 2
            min_edge_multiple: float = 1.5
            min_final_confidence: float = 0.25
            min_tech_confidence: float = 0.15
            min_regime_confidence: float = 0.30
            max_whipsaws: int = 3
            reversal_confidence_multiplier: float = 1.3
            max_hold_hours: float = 24.0
            loss_cooldown_hours: float = 2.0
            ml_model_path: str = "models/hogan_logreg.pkl"
            champion_ml_model_path: str = "models/hogan_champion.pkl"

        config = FakeConfig()
        with patch.dict(os.environ, {"HOGAN_CHAMPION_MODE": "1"}):
            result = apply_champion_mode(config)
        assert result.use_ema_clouds is False
        assert result.use_fvg is False
        assert result.use_rl_agent is False
        assert result.use_ict is False
        assert result.ml_model_path == "models/hogan_champion.pkl"


class TestChampionFeatureSubset:
    """Tests for champion feature subset enforcement in feature_registry and ml."""

    def test_get_feature_columns_true_returns_champion_subset(self):
        from hogan_bot.feature_registry import (
            CHAMPION_FEATURE_COLUMNS,
            get_feature_columns,
        )
        cols = get_feature_columns(True)
        assert cols == list(CHAMPION_FEATURE_COLUMNS)
        assert len(cols) == len(CHAMPION_FEATURE_COLUMNS)

    def test_get_feature_columns_false_returns_59(self):
        from hogan_bot.feature_registry import (
            _FULL_FEATURE_COLUMNS,
            get_feature_columns,
        )
        cols = get_feature_columns(False)
        assert cols == list(_FULL_FEATURE_COLUMNS)
        assert len(cols) == 59

    def test_get_feature_columns_none_respects_env(self):
        from hogan_bot.feature_registry import (
            CHAMPION_FEATURE_COLUMNS,
            get_feature_columns,
        )
        with patch.dict(os.environ, {"HOGAN_CHAMPION_MODE": "1"}):
            cols = get_feature_columns(None)
        assert cols == list(CHAMPION_FEATURE_COLUMNS)
        assert len(cols) == len(CHAMPION_FEATURE_COLUMNS)

    def test_get_feature_columns_none_env_off_returns_full(self):
        from hogan_bot.feature_registry import get_feature_columns
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HOGAN_CHAMPION_MODE", None)
            cols = get_feature_columns(None)
        assert len(cols) == 59

    def test_build_training_set_champion_returns_champion_columns(self):
        import numpy as np
        import pandas as pd

        from hogan_bot.ml import build_training_set

        # Minimal candles with enough rows for training
        n = 300
        np.random.seed(42)
        candles = pd.DataFrame({
            "open": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "high": 101 + np.cumsum(np.random.randn(n) * 0.5),
            "low": 99 + np.cumsum(np.random.randn(n) * 0.5),
            "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "volume": np.abs(np.random.randn(n) * 1e6).astype(int),
        })
        candles["high"] = candles[["open", "high", "close"]].max(axis=1)
        candles["low"] = candles[["open", "low", "close"]].min(axis=1)

        from hogan_bot.feature_registry import CHAMPION_FEATURE_COLUMNS

        x, y, feature_cols, _mq = build_training_set(
            candles, horizon_bars=12, db_conn=None, use_champion_features=True
        )
        assert x is not None and y is not None
        assert feature_cols == list(CHAMPION_FEATURE_COLUMNS)
        assert x.shape[1] == len(CHAMPION_FEATURE_COLUMNS)


class TestGetChampionSummary:
    def test_returns_dict(self):
        summary = get_champion_summary()
        assert isinstance(summary, dict)
        assert "champion_mode" in summary
        assert "locked_experiments_off" in summary
        assert "min_hold_bars" in summary
