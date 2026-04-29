from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _candles(start: str, periods: int, base: float, step: float) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    closes = [base + idx * step for idx in range(periods)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": [close * 1.01 for close in closes],
        "low": [close * 0.99 for close in closes],
        "close": closes,
        "volume": [100.0] * periods,
    })


def test_polymarket_correlation_feature_panel_and_reports(tmp_path):
    from hogan_bot.polymarket_correlation import (
        build_feature_panel,
        lead_lag_correlations,
        run_correlation_research,
    )
    from hogan_bot.storage import (
        get_connection,
        insert_polymarket_market_snapshots,
        upsert_candles,
        upsert_onchain,
    )

    db_path = tmp_path / "hogan.db"
    conn = get_connection(str(db_path))
    upsert_candles(conn, "BTC/USD", "1h", _candles("2026-01-01", 120, 100.0, 1.0))
    upsert_candles(conn, "SPY/USD", "1h", _candles("2026-01-01", 120, 500.0, 0.5))
    upsert_candles(conn, "QQQ/USD", "1h", _candles("2026-01-01", 120, 400.0, 0.6))
    upsert_candles(conn, "UUP/USD", "1h", _candles("2026-01-01", 120, 30.0, -0.01))
    upsert_candles(conn, "VIX/USD", "1h", _candles("2026-01-01", 120, 15.0, 0.01))
    upsert_candles(conn, "GLD/USD", "1h", _candles("2026-01-01", 120, 190.0, 0.1))
    upsert_onchain(
        conn,
        "BTC/USD",
        [
            ("2026-01-01", "news_sentiment_score", 0.4),
            ("2026-01-01", "news_volume_norm", 1.8),
            ("2026-01-01", "fear_greed_value", 20.0),
            ("2026-01-02", "news_sentiment_score", 0.2),
            ("2026-01-02", "news_volume_norm", 1.0),
            ("2026-01-02", "fear_greed_value", 30.0),
        ],
    )
    for idx, ts in enumerate(pd.date_range("2026-01-01", periods=5, freq="12h", tz="UTC")):
        insert_polymarket_market_snapshots(
            conn,
            "BTC/USD",
            int(ts.timestamp() * 1000),
            [
                {
                    "market_id": f"m{idx}",
                    "slug": "btc-150k",
                    "question": "Will Bitcoin hit $150k by December 31, 2026?",
                    "category": "btc",
                    "category_id": "crypto_price_target",
                    "market_type": "price_target",
                    "horizon": "long_term",
                    "yes_probability": 0.05 + idx * 0.01,
                    "probability_source": "gamma_outcome_price",
                    "spread": 0.01,
                    "liquidity": 100000.0,
                    "volume": 5000.0,
                    "data_quality_score": 0.8,
                    "eligibility": "shadow_candidate",
                    "required_evidence_source": "crypto_price_history",
                    "shadow_policy": "fair_value_required",
                }
            ],
        )

    panel = build_feature_panel(conn, horizons=("1h", "4h"), limit=120)
    correlations = lead_lag_correlations(panel, horizons=("1h", "4h"), min_samples=10)
    conn.close()

    assert "fwd_btc_ret_1h" in panel.columns
    assert "spy_ret_1h" in panel.columns
    assert "news_sentiment_score" in panel.columns
    assert "poly_crypto_price_target_prob" in panel.columns
    assert correlations

    result = run_correlation_research(
        db_path=str(db_path),
        horizons=("1h", "4h"),
        min_samples=10,
        report_dir=str(tmp_path / "reports"),
    )
    assert result["rows"] == len(panel)
    assert result["correlations"]
    assert result["markdown_path"].endswith(".md")
    payload = json.loads(Path(result["json_path"]).read_text())
    assert "intelligence_hooks" in payload


def test_polymarket_correlation_hypotheses_are_research_only():
    from hogan_bot.polymarket_correlation import evaluate_strategy_hypotheses

    panel = pd.DataFrame({
        "spy_ret_1h": [0.01, 0.02, -0.01, 0.03, 0.02],
        "qqq_ret_1h": [0.02, 0.01, -0.01, 0.04, 0.01],
        "uup_ret_1h": [-0.01, -0.02, 0.01, -0.01, -0.02],
        "vix_ret_1h": [0.0, -0.01, 0.02, 0.0, -0.01],
        "poly_crypto_price_target_prob_change": [0.01, 0.02, 0.01, 0.03, 0.02],
        "news_volume_norm": [1.0, 1.2, 2.0, 1.1, 1.0],
        "news_sentiment_score": [0.1, 0.2, 0.4, 0.1, 0.0],
        "fear_greed_value": [20, 22, 80, 30, 18],
        "btc_ret_24h": [0.0, 0.02, 0.04, 0.01, 0.0],
        "fwd_btc_ret_1h": [0.01, 0.02, -0.01, 0.03, 0.02],
    })

    results = evaluate_strategy_hypotheses(panel, horizons=("1h",), min_samples=1)

    assert results
    assert all("Research-only" in result.notes for result in results)
