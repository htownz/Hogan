from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd


def test_parse_polymarket_deadline_month_day_year():
    from hogan_bot.polymarket_long_horizon import parse_polymarket_deadline

    deadline = parse_polymarket_deadline(
        "Will Bitcoin hit $150k by June 30, 2026?",
        as_of=datetime(2026, 4, 28, tzinfo=UTC),
    )

    assert deadline == datetime(2026, 6, 30, 23, 59, tzinfo=UTC)


def test_estimate_btc_long_horizon_probability_from_local_candles(tmp_path):
    from hogan_bot.polymarket_long_horizon import estimate_btc_long_horizon_probability
    from hogan_bot.storage import get_connection, upsert_candles

    db_path = tmp_path / "hogan.db"
    conn = get_connection(str(db_path))
    timestamps = pd.date_range("2025-01-01", periods=220, freq="1D", tz="UTC")
    closes = [90_000 + idx * 90 for idx in range(len(timestamps))]
    candles = pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": [close * 1.01 for close in closes],
        "low": [close * 0.99 for close in closes],
        "close": closes,
        "volume": [100.0] * len(closes),
    })
    upsert_candles(conn, "BTC/USD", "1d", candles)

    estimate = estimate_btc_long_horizon_probability(
        conn,
        target_price=125_000,
        question="Will Bitcoin hit $125k by December 31, 2026?",
        as_of=datetime(2025, 8, 9, tzinfo=UTC),
    )
    conn.close()

    assert estimate is not None
    assert 0.0 < estimate.probability < 1.0
    assert estimate.source == "btc_long_horizon_lognormal_v1"
    assert estimate.sample_size >= 90


def test_estimate_btc_long_horizon_probability_requires_deadline(tmp_path):
    from hogan_bot.polymarket_long_horizon import estimate_btc_long_horizon_probability
    from hogan_bot.storage import get_connection

    conn = get_connection(str(tmp_path / "hogan.db"))
    estimate = estimate_btc_long_horizon_probability(
        conn,
        target_price=150_000,
        question="Will bitcoin hit $1m before GTA VI?",
        as_of=datetime(2026, 4, 28, tzinfo=UTC),
    )
    conn.close()

    assert estimate is None
