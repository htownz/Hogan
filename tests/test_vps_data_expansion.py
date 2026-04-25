from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def _candles(n: int = 3) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "volume": [10.0 + i for i in range(n)],
    })


class TestDockerVpsStack:
    def test_deployment_files_exist_and_name_core_services(self):
        root = Path(__file__).resolve().parent.parent
        assert (root / "Dockerfile").exists()
        assert (root / "docker-compose.yml").exists()
        assert (root / ".dockerignore").exists()
        compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
        for service in ("hogan-bot", "timescaledb", "prometheus", "grafana"):
            assert f"{service}:" in compose
        assert "deployment/timescale/init" in compose


class TestCandleStoreAbstraction:
    def test_sqlite_candle_store_round_trip(self):
        from hogan_bot.candle_store import SQLiteCandleStore
        from hogan_bot.storage import _create_schema

        conn = sqlite3.connect(":memory:")
        _create_schema(conn)
        store = SQLiteCandleStore(conn)
        written = store.upsert_candles("BTC/USD", "1h", _candles(4))
        assert written == 4
        assert store.candle_count("BTC/USD", "1h") == 4
        loaded = store.load_candles("BTC/USD", "1h", limit=2)
        assert list(loaded["close"]) == [102.5, 103.5]
        assert store.available_symbols() == [("BTC/USD", "1h", 4)]
        assert store.oldest_ts_ms("BTC/USD", "1h") is not None


class TestSubMinuteIngestion:
    def test_seconds_timeframes_parse_and_infer(self):
        from hogan_bot.timeframe_utils import (
            bars_per_day,
            default_horizon_bars,
            infer_timeframe_from_candles,
            parse_timeframe_to_seconds,
        )

        assert parse_timeframe_to_seconds("10s") == 10
        assert parse_timeframe_to_seconds("1m") == 60
        assert bars_per_day("30s") == 2880
        assert bars_per_day("1m") == 1440
        assert default_horizon_bars("10s", target_hours=1.0) == 360
        df = pd.DataFrame({"ts_ms": [0, 10_000, 20_000, 30_000]})
        assert infer_timeframe_from_candles(df) == "10s"

    def test_trade_candle_aggregator_closes_previous_bucket(self):
        from hogan_bot.data_engine import TradeCandleAggregator

        agg = TradeCandleAggregator("10s")
        assert agg.add_trade("BTC/USD", 1_000, 100.0, 0.5) is None
        assert agg.add_trade("BTC/USD", 2_000, 101.0, 0.25) is None
        closed = agg.add_trade("BTC/USD", 10_000, 99.0, 0.1)
        assert closed == {
            "ts_ms": 0,
            "open": 100.0,
            "high": 101.0,
            "low": 100.0,
            "close": 101.0,
            "volume": 0.75,
        }


class TestExperimentalFeatureExpansion:
    def test_experimental_features_are_challenger_only(self):
        from hogan_bot.feature_registry import (
            EXPERIMENTAL_FEATURE_COLUMNS,
            get_feature_columns,
        )

        full = get_feature_columns(False, include_experimental=True)
        champion = get_feature_columns(True, include_experimental=True)
        for col in EXPERIMENTAL_FEATURE_COLUMNS:
            assert col in full
            assert col not in champion

    def test_social_whale_features_are_point_in_time(self):
        from hogan_bot.social_whale_features import add_social_whale_features
        from hogan_bot.storage import _create_schema

        conn = sqlite3.connect(":memory:")
        _create_schema(conn)
        conn.executemany(
            """
            INSERT INTO onchain_metrics (symbol, date, metric, value)
            VALUES ('BTC/USD', ?, ?, ?)
            """,
            [
                ("2024-01-01", "social_nlp_sentiment_score", 0.25),
                ("2024-01-03", "social_nlp_sentiment_score", 0.75),
                ("2024-01-01", "whale_exchange_flow_norm", -0.2),
            ],
        )
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(
                ["2024-01-02T00:00:00Z", "2024-01-04T00:00:00Z"],
                utc=True,
            ),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        })
        out = add_social_whale_features(frame, conn)
        assert list(out["social_nlp_sentiment_score"]) == [0.25, 0.75]
        assert list(out["whale_exchange_flow_norm"]) == [-0.2, -0.2]

    def test_predict_up_probability_builds_experimental_columns(self):
        from hogan_bot.feature_registry import (
            EXPERIMENTAL_FEATURE_COLUMNS,
            get_feature_columns,
        )
        from hogan_bot.ml import TrainedModel, predict_up_probability
        from hogan_bot.storage import _create_schema

        class _Model:
            def predict_proba(self, x):
                assert x.shape[1] == len(get_feature_columns(False, include_experimental=True))
                return np.array([[0.30, 0.70]])

        conn = sqlite3.connect(":memory:")
        _create_schema(conn)
        conn.executemany(
            """
            INSERT INTO onchain_metrics (symbol, date, metric, value)
            VALUES ('BTC/USD', ?, ?, ?)
            """,
            [
                ("2024-01-01", "social_nlp_sentiment_score", 0.25),
                ("2024-01-01", "social_volume_anomaly", 0.10),
                ("2024-01-01", "whale_exchange_flow_norm", -0.20),
                ("2024-01-01", "whale_large_tx_count_norm", 0.30),
            ],
        )
        candles = _candles(80)
        artifact = TrainedModel(
            model=_Model(),
            feature_columns=get_feature_columns(False, include_experimental=True),
            scaler=None,
        )
        assert all(col in artifact.feature_columns for col in EXPERIMENTAL_FEATURE_COLUMNS)
        assert predict_up_probability(candles, artifact, db_conn=conn) == 0.70

    def test_feature_row_checked_keeps_experimental_names_aligned(self):
        from hogan_bot.feature_registry import get_feature_columns
        from hogan_bot.ml import build_feature_row_checked
        from hogan_bot.storage import _create_schema

        conn = sqlite3.connect(":memory:")
        _create_schema(conn)
        result = build_feature_row_checked(
            _candles(80),
            db_conn=conn,
            use_champion_features=False,
            include_experimental_features=True,
        )
        assert result is not None
        assert len(result.values) == len(get_feature_columns(False, include_experimental=True))


class TestTimescaleSchema:
    def test_timescale_python_schema_uses_ts_ms_index(self):
        from hogan_bot.candle_store import TimescaleCandleStore

        class _Cursor:
            def __init__(self):
                self.sql: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def execute(self, sql, *_args):
                self.sql.append(str(sql))

        class _Conn:
            def __init__(self):
                self.cursor_obj = _Cursor()

            def cursor(self):
                return self.cursor_obj

            def commit(self):
                pass

        store = TimescaleCandleStore.__new__(TimescaleCandleStore)
        store.conn = _Conn()
        store.ensure_schema()
        sql = "\n".join(store.conn.cursor_obj.sql)
        assert "ts_ms DESC" in sql
        assert "timeframe, ts DESC" not in sql


class TestNeuralNetChallenger:
    def test_cv_factory_returns_predict_proba_model(self):
        from hogan_bot.ml import _make_cv_model

        model = _make_cv_model("neural_net")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")
