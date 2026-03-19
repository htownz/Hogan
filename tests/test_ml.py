"""Tests for the enhanced ML pipeline in hogan_bot.ml."""
import unittest

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None
    np = None

try:
    import lightgbm  # noqa: F401
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

from hogan_bot.feature_registry import get_feature_columns
from hogan_bot.ml import (
    _FEATURE_COLUMNS,
    _feature_frame,
    build_feature_frame,
    build_training_set,
    predict_up_probability,
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
    walk_forward_cv,
)


def _synthetic_candles(n: int = 400, seed: int = 42) -> "pd.DataFrame":
    """Generate a realistic-looking random-walk OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = 30_000.0 + np.cumsum(rng.normal(0, 50, n))
    close = np.clip(close, 1_000.0, None)
    noise = rng.uniform(0.001, 0.005, n)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(close, open_) * (1 + noise)
    low = np.minimum(close, open_) * (1 - noise)
    volume = rng.uniform(500, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@unittest.skipUnless(pd is not None, "pandas not installed")
class FeatureFrameTests(unittest.TestCase):
    def setUp(self):
        self.df = _synthetic_candles(n=200)

    def test_all_feature_columns_present(self):
        frame = build_feature_frame(self.df)
        for col in _FEATURE_COLUMNS:
            self.assertIn(col, frame.columns, f"Missing feature column: {col}")

    def test_new_indicator_features_present(self):
        frame = _feature_frame(self.df)
        for col in ("atr_pct", "macd_hist_pct", "bb_pct_b", "vol_regime"):
            self.assertIn(col, frame.columns, f"Missing new feature: {col}")

    def test_feature_column_count(self):
        self.assertEqual(len(_FEATURE_COLUMNS), 59)

    def test_rsi_bounded(self):
        frame = _feature_frame(self.df)
        rsi_scaled = frame["rsi_14"].dropna()
        self.assertTrue((rsi_scaled >= 0.0).all(), "RSI below 0")
        self.assertTrue((rsi_scaled <= 1.0).all(), "RSI above 1")

    def test_no_inf_values(self):
        frame = build_feature_frame(self.df)
        numeric = frame[_FEATURE_COLUMNS].select_dtypes(include="number")
        self.assertFalse(numeric.isin([float("inf"), float("-inf")]).any().any())

    def test_ret_12_is_12bar_return(self):
        frame = _feature_frame(self.df)
        close = self.df["close"].astype(float)
        expected = (close.iloc[12] / close.iloc[0]) - 1.0
        self.assertAlmostEqual(frame["ret_12"].iloc[12], expected, places=8)

    def test_wick_features_non_negative(self):
        frame = _feature_frame(self.df)
        self.assertTrue((frame["upper_wick_pct"].dropna() >= 0).all())
        self.assertTrue((frame["lower_wick_pct"].dropna() >= 0).all())


@unittest.skipUnless(pd is not None, "pandas not installed")
class BuildTrainingSetTests(unittest.TestCase):
    def test_returns_correct_shapes(self):
        df = _synthetic_candles(n=300)
        x, y, cols, _mq = build_training_set(df, horizon_bars=3, fee_rate=0.0)
        self.assertIsNotNone(x)
        self.assertEqual(len(x), len(y))
        self.assertEqual(list(x.columns), cols)

    def test_feature_columns_match_constant(self):
        df = _synthetic_candles(n=300)
        _, _, cols, _mq = build_training_set(df, horizon_bars=3, fee_rate=0.0)
        self.assertEqual(cols, get_feature_columns())

    def test_no_nan_in_output(self):
        df = _synthetic_candles(n=300)
        x, y, _, _mq = build_training_set(df, horizon_bars=3, fee_rate=0.0)
        self.assertFalse(x.isnull().any().any())
        self.assertFalse(y.isnull().any())


@unittest.skipUnless(pd is not None, "pandas not installed")
class LogRegTrainingTests(unittest.TestCase):
    def setUp(self):
        self.df = _synthetic_candles(n=800)
        import tempfile
        self.tmp = tempfile.mktemp(suffix=".pkl")

    def tearDown(self):
        import os
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_metrics_keys_present(self):
        metrics = train_logistic_regression(self.df, model_path=self.tmp)
        for key in ("accuracy", "roc_auc", "precision", "recall", "f1", "train_rows", "test_rows", "features"):
            self.assertIn(key, metrics)

    def test_model_type_reported(self):
        metrics = train_logistic_regression(self.df, model_path=self.tmp)
        self.assertEqual(metrics["model_type"], "logistic_regression")

    def test_scaler_saved_in_artifact(self):
        train_logistic_regression(self.df, model_path=self.tmp)
        import pickle
        with open(self.tmp, "rb") as f:
            artifact = pickle.load(f)
        self.assertIsNotNone(artifact.scaler)

    def test_predict_returns_float_in_range(self):
        train_logistic_regression(self.df, model_path=self.tmp)
        import pickle
        with open(self.tmp, "rb") as f:
            artifact = pickle.load(f)
        prob = predict_up_probability(self.df, artifact)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_scaler_applied_during_prediction(self):
        """Ensure old artifacts without scaler still work via getattr fallback."""
        train_logistic_regression(self.df, model_path=self.tmp)
        import pickle
        with open(self.tmp, "rb") as f:
            artifact = pickle.load(f)
        # Simulate old artifact without scaler attribute
        del artifact.scaler
        prob = predict_up_probability(self.df, artifact)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_feature_count_in_metrics(self):
        metrics = train_logistic_regression(self.df, model_path=self.tmp)
        self.assertEqual(metrics["features"], len(get_feature_columns()))


@unittest.skipUnless(pd is not None, "pandas not installed")
class RandomForestTrainingTests(unittest.TestCase):
    def setUp(self):
        self.df = _synthetic_candles(n=800)
        import tempfile
        self.tmp = tempfile.mktemp(suffix=".pkl")

    def tearDown(self):
        import os
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_metrics_keys_present(self):
        metrics = train_random_forest(self.df, model_path=self.tmp)
        for key in ("accuracy", "roc_auc", "precision", "recall", "f1", "feature_importances"):
            self.assertIn(key, metrics)

    def test_model_type_reported(self):
        metrics = train_random_forest(self.df, model_path=self.tmp)
        self.assertEqual(metrics["model_type"], "random_forest")

    def test_feature_importances_sum_to_one(self):
        metrics = train_random_forest(self.df, model_path=self.tmp)
        total = sum(metrics["feature_importances"].values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_scaler_is_none_for_rf(self):
        train_random_forest(self.df, model_path=self.tmp)
        import pickle
        with open(self.tmp, "rb") as f:
            artifact = pickle.load(f)
        self.assertIsNone(artifact.scaler)

    def test_rf_predict_returns_valid_probability(self):
        train_random_forest(self.df, model_path=self.tmp)
        import pickle
        with open(self.tmp, "rb") as f:
            artifact = pickle.load(f)
        prob = predict_up_probability(self.df, artifact)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


@unittest.skipUnless(pd is not None, "pandas not installed")
class WalkForwardCVTests(unittest.TestCase):
    def test_returns_correct_number_of_folds(self):
        df = _synthetic_candles(n=800)
        results = walk_forward_cv(df, horizon_bars=3, n_splits=4, fee_rate=0.0)
        self.assertEqual(len(results), 4)

    def test_fold_structure(self):
        df = _synthetic_candles(n=600)
        results = walk_forward_cv(df, horizon_bars=3, n_splits=3, fee_rate=0.0)
        for fold in results:
            for key in ("fold", "train_rows", "test_rows", "accuracy", "roc_auc"):
                self.assertIn(key, fold)

    def test_train_rows_increase_across_folds(self):
        df = _synthetic_candles(n=800)
        results = walk_forward_cv(df, horizon_bars=3, n_splits=4, fee_rate=0.0)
        train_sizes = [f["train_rows"] for f in results]
        self.assertEqual(train_sizes, sorted(train_sizes))

    def test_accuracy_is_valid_probability(self):
        df = _synthetic_candles(n=600)
        results = walk_forward_cv(df, horizon_bars=3, n_splits=3, fee_rate=0.0)
        for fold in results:
            self.assertGreaterEqual(fold["accuracy"], 0.0)
            self.assertLessEqual(fold["accuracy"], 1.0)

    def test_roc_auc_bounded(self):
        df = _synthetic_candles(n=600)
        results = walk_forward_cv(df, horizon_bars=3, n_splits=3, fee_rate=0.0)
        for fold in results:
            self.assertGreaterEqual(fold["roc_auc"], 0.0)
            self.assertLessEqual(fold["roc_auc"], 1.0)

    def test_insufficient_data_raises(self):
        df = _synthetic_candles(n=30)
        with self.assertRaises(RuntimeError):
            walk_forward_cv(df, horizon_bars=3, n_splits=5, fee_rate=0.0)


@unittest.skipUnless(_HAS_LIGHTGBM, "lightgbm not installed")
class TestTrainLightGBM(unittest.TestCase):
    def test_returns_expected_keys(self):
        df = _synthetic_candles(n=2000)
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lgbm.pkl")
            metrics = train_lightgbm(df, model_path=path, horizon_bars=3)
        for key in ("model_type", "accuracy", "roc_auc", "features", "feature_importances"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["model_type"], "lightgbm")

    def test_model_type_label(self):
        df = _synthetic_candles(n=2000)
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lgbm.pkl")
            metrics = train_lightgbm(df, model_path=path)
        self.assertEqual(metrics["model_type"], "lightgbm")

    def test_feature_importances_all_features_present(self):
        df = _synthetic_candles(n=2000)
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lgbm.pkl")
            metrics = train_lightgbm(df, model_path=path)
        imps = metrics["feature_importances"]
        self.assertEqual(set(imps.keys()), set(get_feature_columns()))


class TestTrainLightGBMMissingPackage(unittest.TestCase):
    def test_raises_runtime_error_when_not_installed(self):
        """Verify the helpful error is raised when lightgbm is absent."""
        if _HAS_LIGHTGBM:
            self.skipTest("lightgbm is installed; skip missing-package test")
        df = _synthetic_candles(n=300)
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                train_lightgbm(df, model_path=os.path.join(tmp, "m.pkl"))


class TestRegimeModelRouter(unittest.TestCase):
    """Tests for RegimeModelRouter routing logic."""

    def test_delegates_to_global_by_default(self):
        from hogan_bot.ml import RegimeModelRouter
        global_model = TrainedModel(model="global", feature_columns=["a", "b"], scaler=None)
        router = RegimeModelRouter(global_model)
        assert router.model == "global"
        assert router.feature_columns == ["a", "b"]
        assert router.scaler is None
        assert not router.has_regime_models

    def test_routes_to_regime_model(self):
        from hogan_bot.ml import RegimeModelRouter
        global_model = TrainedModel(model="global", feature_columns=["a", "b"], scaler=None)
        regime_model = TrainedModel(model="trending_up_model", feature_columns=["a", "b"], scaler="s1")
        router = RegimeModelRouter(global_model, {"trending_up": regime_model})
        assert router.has_regime_models
        assert router.regime_names == ["trending_up"]

        router.set_regime("trending_up")
        assert router.model == "trending_up_model"
        assert router.scaler == "s1"

        router.set_regime("ranging")
        assert router.model == "global"

        router.set_regime(None)
        assert router.model == "global"


if __name__ == "__main__":
    unittest.main()
