"""Tests for hogan_bot.retrain walk-forward retraining."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hogan_bot.ml import TrainedModel
from hogan_bot.retrain import _get_current_best_score, retrain_once


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _synthetic_candles(n: int = 400) -> pd.DataFrame:
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


def _make_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    """Return a minimal Namespace for retrain_once."""
    defaults = {
        "symbol": "BTC/USD",
        "timeframe": "5m",
        "exchange": "kraken",
        "window_bars": 400,
        "horizon_bars": 3,
        "model_type": "logreg",
        "model_path": str(tmp_path / "model.pkl"),
        "tune": False,
        "promotion_metric": "roc_auc",
        "min_improvement": 0.005,
        "registry_path": str(tmp_path / "registry.jsonl"),
        "from_db": False,
        "db": str(tmp_path / "hogan.db"),
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# retrain_once: first-run behaviour
# ---------------------------------------------------------------------------


class TestFirstRun:
    def test_promotes_on_first_run(self, tmp_path):
        args = _make_args(tmp_path)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            result = retrain_once(args)

        assert result["promoted"] is True
        assert result["current_score"] is None
        assert Path(args.model_path).exists(), "Production model must exist after promotion"

    def test_result_has_required_keys(self, tmp_path):
        args = _make_args(tmp_path)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            result = retrain_once(args)

        required = {
            "timestamp", "symbol", "timeframe", "exchange", "window_bars",
            "horizon_bars", "model_type", "promotion_metric", "min_improvement",
            "new_score", "current_score", "promoted", "dry_run", "message",
        }
        assert required <= result.keys()

    def test_result_is_json_serialisable(self, tmp_path):
        import json
        args = _make_args(tmp_path)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            result = retrain_once(args)

        json.dumps(result)  # must not raise

    def test_registry_has_one_entry_after_first_run(self, tmp_path):
        args = _make_args(tmp_path)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            retrain_once(args)

        from hogan_bot.registry import ModelRegistry
        entries = ModelRegistry(registry_path=args.registry_path).load_all()
        assert len(entries) == 1
        assert entries[0]["symbol"] == "BTC/USD"

    def test_first_run_message_mentions_registry(self, tmp_path):
        args = _make_args(tmp_path)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            result = retrain_once(args)

        assert "first" in result["message"].lower() or "promoted" in result["message"].lower()


# ---------------------------------------------------------------------------
# retrain_once: promotion gate
# ---------------------------------------------------------------------------


class TestPromotionGate:
    def _run_twice(self, tmp_path, min_improvement):
        """Helper: run two retrain cycles and return both results."""
        args = _make_args(tmp_path, min_improvement=min_improvement)
        candles = _synthetic_candles()
        with patch("hogan_bot.retrain._fetch_candles", return_value=candles):
            first = retrain_once(args)
        with patch("hogan_bot.retrain._fetch_candles", return_value=candles):
            second = retrain_once(args)
        return first, second

    def test_does_not_promote_when_below_threshold(self, tmp_path):
        # An impossibly high threshold ensures the second run never promotes.
        _, second = self._run_twice(tmp_path, min_improvement=0.9999)
        assert second["promoted"] is False

    def test_current_score_populated_on_second_run(self, tmp_path):
        first, second = self._run_twice(tmp_path, min_improvement=0.9999)
        assert second["current_score"] == pytest.approx(first["new_score"])

    def test_not_promoted_message_explains_gap(self, tmp_path):
        _, second = self._run_twice(tmp_path, min_improvement=0.9999)
        assert "not promoted" in second["message"].lower()

    def test_zero_improvement_threshold_always_promotes(self, tmp_path):
        """With min_improvement=0, a model that matches the current score still promotes."""
        _, second = self._run_twice(tmp_path, min_improvement=0.0)
        assert second["promoted"] is True

    def test_registry_grows_on_each_promotion(self, tmp_path):
        from hogan_bot.registry import ModelRegistry
        args = _make_args(tmp_path, min_improvement=0.0)  # always promote
        candles = _synthetic_candles()
        for _ in range(3):
            with patch("hogan_bot.retrain._fetch_candles", return_value=candles):
                retrain_once(args)

        entries = ModelRegistry(registry_path=args.registry_path).load_all()
        assert len(entries) == 3


# ---------------------------------------------------------------------------
# retrain_once: dry-run mode
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_does_not_create_model_file(self, tmp_path):
        args = _make_args(tmp_path, dry_run=True)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            retrain_once(args)

        assert not Path(args.model_path).exists()

    def test_dry_run_does_not_write_registry(self, tmp_path):
        args = _make_args(tmp_path, dry_run=True)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            retrain_once(args)

        assert not Path(args.registry_path).exists()

    def test_dry_run_result_promoted_is_false(self, tmp_path):
        args = _make_args(tmp_path, dry_run=True)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            result = retrain_once(args)

        assert result["promoted"] is False
        assert result["dry_run"] is True

    def test_dry_run_does_not_overwrite_existing_model(self, tmp_path):
        """When a production model already exists, --dry-run must not touch it."""
        args = _make_args(tmp_path)
        candles = _synthetic_candles()

        # Seed the production model via a real run
        with patch("hogan_bot.retrain._fetch_candles", return_value=candles):
            retrain_once(args)

        original_mtime = Path(args.model_path).stat().st_mtime

        # Dry run must not change it
        dry_args = _make_args(tmp_path, dry_run=True)
        with patch("hogan_bot.retrain._fetch_candles", return_value=candles):
            retrain_once(dry_args)

        assert Path(args.model_path).stat().st_mtime == original_mtime

    def test_dry_run_reports_new_score(self, tmp_path):
        args = _make_args(tmp_path, dry_run=True)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            result = retrain_once(args)

        assert result["new_score"] is not None
        assert 0.0 <= result["new_score"] <= 1.0


# ---------------------------------------------------------------------------
# retrain_once: candidate file lifecycle
# ---------------------------------------------------------------------------


class TestCandidateCleanup:
    def test_no_candidate_files_remain_on_success(self, tmp_path):
        args = _make_args(tmp_path)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            retrain_once(args)

        leftover = list(tmp_path.glob("*_candidate_*.pkl"))
        assert leftover == [], f"Candidate files not cleaned up: {leftover}"

    def test_no_candidate_files_remain_after_dry_run(self, tmp_path):
        args = _make_args(tmp_path, dry_run=True)
        with patch("hogan_bot.retrain._fetch_candles", return_value=_synthetic_candles()):
            retrain_once(args)

        leftover = list(tmp_path.glob("*_candidate_*.pkl"))
        assert leftover == []

    def test_no_candidate_files_remain_when_not_promoted(self, tmp_path):
        args = _make_args(tmp_path, min_improvement=0.9999)
        candles = _synthetic_candles()
        with patch("hogan_bot.retrain._fetch_candles", return_value=candles):
            retrain_once(args)  # seed registry
        with patch("hogan_bot.retrain._fetch_candles", return_value=candles):
            retrain_once(args)  # second run — not promoted

        leftover = list(tmp_path.glob("*_candidate_*.pkl"))
        assert leftover == []


# ---------------------------------------------------------------------------
# retrain_once: --from-db flag
# ---------------------------------------------------------------------------


class TestFromDb:
    def test_from_db_calls_load_candles_not_exchange(self, tmp_path):
        args = _make_args(tmp_path, from_db=True)
        candles = _synthetic_candles()

        with patch("hogan_bot.retrain.load_candles", return_value=candles) as mock_load, \
             patch("hogan_bot.retrain.get_connection", return_value=MagicMock()):
            retrain_once(args)

        mock_load.assert_called_once()

    def test_from_db_raises_on_empty_db(self, tmp_path):
        args = _make_args(tmp_path, from_db=True)

        with patch("hogan_bot.retrain.load_candles", return_value=pd.DataFrame()), \
             patch("hogan_bot.retrain.get_connection", return_value=MagicMock()):
            with pytest.raises(RuntimeError, match="No candles found"):
                retrain_once(args)


# ---------------------------------------------------------------------------
# _get_current_best_score
# ---------------------------------------------------------------------------


class TestGetCurrentBestScore:
    def test_returns_none_for_empty_registry(self, tmp_path):
        from hogan_bot.registry import ModelRegistry
        reg = ModelRegistry(registry_path=str(tmp_path / "empty.jsonl"))
        assert _get_current_best_score(reg, "roc_auc") is None

    def _seed_model_file(self, path: str) -> None:
        """Write a minimal picklable TrainedModel artifact."""
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression()
        m.fit([[0], [1]], [0, 1])
        artifact = TrainedModel(model=m, feature_columns=[], scaler=None)
        with open(path, "wb") as fh:
            pickle.dump(artifact, fh)

    def test_returns_correct_score(self, tmp_path):
        from hogan_bot.registry import ModelRegistry

        model_path = str(tmp_path / "model.pkl")
        self._seed_model_file(model_path)

        reg = ModelRegistry(registry_path=str(tmp_path / "reg.jsonl"))
        reg.log({"roc_auc": 0.72, "model_type": "logreg", "features": 10}, model_path=model_path)

        score = _get_current_best_score(reg, "roc_auc")
        assert score == pytest.approx(0.72)

    def test_returns_max_when_multiple_entries(self, tmp_path):
        from hogan_bot.registry import ModelRegistry

        model_path = str(tmp_path / "model.pkl")
        self._seed_model_file(model_path)

        reg = ModelRegistry(registry_path=str(tmp_path / "reg.jsonl"))
        reg.log({"roc_auc": 0.65, "model_type": "logreg", "features": 10}, model_path=model_path)
        reg.log({"roc_auc": 0.72, "model_type": "logreg", "features": 10}, model_path=model_path)
        reg.log({"roc_auc": 0.68, "model_type": "logreg", "features": 10}, model_path=model_path)

        assert _get_current_best_score(reg, "roc_auc") == pytest.approx(0.72)
