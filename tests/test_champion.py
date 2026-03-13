"""Tests for the Champion Path — canonical production settings."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

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
        import logging
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HOGAN_CHAMPION_MODE", None)
            with patch("hogan_bot.champion.logger") as mock_logger:
                apply_champion_mode(config)
                mock_logger.warning.assert_called_once()

    def test_champion_mode_uses_replace_on_dataclass(self):
        """Champion mode applies overrides via dataclasses.replace()."""
        from dataclasses import dataclass

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
            min_hold_bars: int = 3
            exit_confirmation_bars: int = 2
            min_edge_multiple: float = 1.5
            max_hold_hours: float = 24.0
            loss_cooldown_hours: float = 2.0

        config = FakeConfig()
        with patch.dict(os.environ, {"HOGAN_CHAMPION_MODE": "1"}):
            result = apply_champion_mode(config)
        assert result.use_ema_clouds is False
        assert result.use_fvg is False
        assert result.use_rl_agent is False
        assert result.use_ict is False


class TestGetChampionSummary:
    def test_returns_dict(self):
        summary = get_champion_summary()
        assert isinstance(summary, dict)
        assert "champion_mode" in summary
        assert "locked_experiments_off" in summary
        assert "min_hold_bars" in summary
