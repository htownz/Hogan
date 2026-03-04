"""Tests for hogan_bot.registry.ModelRegistry."""
import json
import os
import tempfile
import unittest

from hogan_bot.registry import ModelRegistry

_SAMPLE_METRICS = {
    "model_type": "logistic_regression",
    "accuracy": 0.52,
    "roc_auc": 0.55,
    "precision": 0.51,
    "recall": 0.60,
    "f1": 0.55,
    "train_rows": 400.0,
    "test_rows": 100.0,
    "features": 24,
}


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.tmp_dir, "registry.jsonl")
        self.model_path = os.path.join(self.tmp_dir, "model.pkl")
        # create a dummy model file so the hash can be computed
        with open(self.model_path, "wb") as f:
            f.write(b"dummy")

    def _make_registry(self):
        return ModelRegistry(registry_path=self.registry_path)

    def test_log_creates_file(self):
        reg = self._make_registry()
        reg.log(_SAMPLE_METRICS, model_path=self.model_path)
        self.assertTrue(os.path.exists(self.registry_path))

    def test_log_writes_valid_json(self):
        reg = self._make_registry()
        reg.log(_SAMPLE_METRICS, model_path=self.model_path)
        with open(self.registry_path) as f:
            line = f.readline().strip()
        entry = json.loads(line)
        self.assertIn("timestamp", entry)
        self.assertIn("metrics", entry)

    def test_log_returns_entry(self):
        reg = self._make_registry()
        entry = reg.log(_SAMPLE_METRICS, model_path=self.model_path)
        self.assertEqual(entry["model_type"], "logistic_regression")
        self.assertIn("model_hash", entry)

    def test_load_all_empty_when_no_file(self):
        reg = ModelRegistry(registry_path=os.path.join(self.tmp_dir, "nonexistent.jsonl"))
        self.assertEqual(reg.load_all(), [])

    def test_load_all_returns_all_entries(self):
        reg = self._make_registry()
        reg.log(_SAMPLE_METRICS, model_path=self.model_path, symbol="BTC/USD")
        reg.log({**_SAMPLE_METRICS, "roc_auc": 0.60}, model_path=self.model_path, symbol="ETH/USD")
        entries = reg.load_all()
        self.assertEqual(len(entries), 2)

    def test_best_returns_highest_metric(self):
        reg = self._make_registry()
        reg.log({**_SAMPLE_METRICS, "roc_auc": 0.53}, model_path=self.model_path)
        reg.log({**_SAMPLE_METRICS, "roc_auc": 0.61}, model_path=self.model_path)
        reg.log({**_SAMPLE_METRICS, "roc_auc": 0.57}, model_path=self.model_path)
        best = reg.best(metric="roc_auc")
        self.assertIsNotNone(best)
        self.assertAlmostEqual(best["metrics"]["roc_auc"], 0.61)

    def test_best_returns_none_on_empty(self):
        reg = self._make_registry()
        self.assertIsNone(reg.best())

    def test_summary_flat_structure(self):
        reg = self._make_registry()
        reg.log(_SAMPLE_METRICS, model_path=self.model_path)
        rows = reg.summary()
        self.assertEqual(len(rows), 1)
        for key in ("timestamp", "model_type", "symbol", "accuracy", "roc_auc"):
            self.assertIn(key, rows[0])

    def test_model_hash_changes_when_file_changes(self):
        reg = self._make_registry()
        e1 = reg.log(_SAMPLE_METRICS, model_path=self.model_path)
        with open(self.model_path, "wb") as f:
            f.write(b"different content")
        e2 = reg.log(_SAMPLE_METRICS, model_path=self.model_path)
        self.assertNotEqual(e1["model_hash"], e2["model_hash"])


if __name__ == "__main__":
    unittest.main()
