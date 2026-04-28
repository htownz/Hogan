from __future__ import annotations

import sqlite3
from types import SimpleNamespace


def test_extract_polymarket_metrics_directional_and_risk():
    from hogan_bot.fetch_polymarket import extract_polymarket_metrics

    markets = [
        {
            "question": "Will Bitcoin reach $120,000 this year?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.62", "0.38"]',
            "liquidity": "1000",
            "volume24hr": "500",
        },
        {
            "question": "Will Ethereum fall below $2,000 this month?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.30", "0.70"]',
            "liquidity": "1000",
            "volume24hr": "0",
        },
        {
            "question": "Will the Fed hike rates at the next meeting?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.40", "0.60"]',
            "liquidity": "100",
        },
    ]

    metrics = extract_polymarket_metrics(markets)

    assert metrics["poly_btc_bull_prob"] == 0.62
    assert metrics["poly_eth_bull_prob"] == 0.70
    assert metrics["poly_macro_risk_prob"] == 0.40
    assert metrics["poly_market_count"] == 3.0
    assert metrics["poly_signal_dispersion"] > 0


def test_extract_polymarket_metrics_uses_clob_midpoint_and_spread():
    from hogan_bot.fetch_polymarket import extract_polymarket_metrics

    metrics = extract_polymarket_metrics([
        {
            "question": "Will Bitcoin reach $120,000 this year?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.20", "0.80"]',
            "poly_clob_midpoint": 0.66,
            "poly_clob_spread": 0.02,
            "liquidity": "1000",
        }
    ])

    assert metrics["poly_btc_bull_prob"] == 0.66
    assert metrics["poly_orderbook_midpoint_avg"] == 0.66
    assert metrics["poly_orderbook_spread_avg"] == 0.02


def test_enrich_clob_snapshots_attaches_public_orderbook_metrics(monkeypatch):
    from hogan_bot.fetch_polymarket import enrich_clob_snapshots

    def _snapshot(token_id):
        assert token_id == "yes-token"
        return {"midpoint": 0.61, "spread": 0.03}

    monkeypatch.setattr("hogan_bot.fetch_polymarket.fetch_clob_token_snapshot", _snapshot)
    markets = [
        {
            "question": "Will Bitcoin reach $100,000?",
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["yes-token", "no-token"]',
        }
    ]

    out = enrich_clob_snapshots(markets, max_markets=1)

    assert out[0]["poly_clob_midpoint"] == 0.61
    assert out[0]["poly_clob_spread"] == 0.03


def test_normalize_market_snapshot_scores_data_quality():
    from hogan_bot.fetch_polymarket import normalize_market_snapshot

    snapshot = normalize_market_snapshot({
        "id": "m1",
        "slug": "btc-100k",
        "eventSlug": "btc",
        "question": "Will Bitcoin reach $100,000 this month?",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.42", "0.58"]',
        "poly_clob_midpoint": 0.43,
        "poly_clob_spread": 0.01,
        "liquidity": "100000",
        "volume24hr": "25000",
    })

    assert snapshot.market_id == "m1"
    assert snapshot.market_type == "price_target"
    assert snapshot.horizon == "short_term"
    assert snapshot.probability_source == "clob_midpoint"
    assert snapshot.data_quality_score > 0.8
    assert snapshot.eligibility == "shadow_candidate"
    assert snapshot.to_dict()["target_price"] == 100_000


def test_score_polymarket_opportunities_uses_hogan_disagreement():
    from hogan_bot.fetch_polymarket import score_polymarket_opportunities

    opportunities = score_polymarket_opportunities(
        [
            {
                "id": "m1",
                "slug": "btc-100k",
                "question": "Will Bitcoin reach $100,000 this month?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.42", "0.58"]',
                "poly_clob_spread": 0.01,
                "liquidity": "100000",
                "volume24hr": "25000",
            },
            {
                "id": "m2",
                "slug": "fed-hike",
                "question": "Will the Fed hike rates this month?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.55", "0.45"]',
                "poly_clob_spread": 0.08,
                "liquidity": "1000",
            },
        ],
        hogan_btc_bull_prob=0.72,
    )

    assert opportunities[0].market_id == "m1"
    assert opportunities[0].candidate_side == "buy_yes"
    assert opportunities[0].edge_score > 0.9
    assert opportunities[0].total_score > opportunities[-1].total_score


def test_score_polymarket_opportunities_keeps_long_targets_research_without_long_fair_value():
    from hogan_bot.fetch_polymarket import score_polymarket_opportunities

    opportunities = score_polymarket_opportunities(
        [
            {
                "id": "m1",
                "slug": "btc-150k-2026",
                "question": "Will Bitcoin hit $150k by December 31, 2026?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.05", "0.95"]',
                "poly_clob_spread": 0.01,
                "liquidity": "100000",
                "volume24hr": "25000",
            }
        ],
        hogan_btc_bull_prob=0.72,
    )

    assert opportunities[0].candidate_side == "research"
    assert opportunities[0].hogan_prob is None
    assert opportunities[0].market_type == "price_target"
    assert opportunities[0].horizon == "long_term"
    assert opportunities[0].target_price == 150_000
    assert opportunities[0].safety_note == "long_horizon_price_target_requires_calibrated_fair_value"


def test_score_polymarket_opportunities_allows_long_target_with_calibrated_fair_value():
    from hogan_bot.fetch_polymarket import score_polymarket_opportunities

    opportunities = score_polymarket_opportunities(
        [
            {
                "id": "m1",
                "slug": "btc-150k-2026",
                "question": "Will Bitcoin hit $150k by December 31, 2026?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.05", "0.95"]',
                "poly_clob_spread": 0.01,
                "liquidity": "100000",
                "volume24hr": "25000",
            }
        ],
        hogan_btc_bull_prob=0.72,
        hogan_btc_long_horizon_prob=0.18,
    )

    assert opportunities[0].candidate_side == "buy_yes"
    assert opportunities[0].hogan_prob == 0.18
    assert opportunities[0].safety_note is None


def test_score_polymarket_opportunities_handles_bearish_yes_mapping():
    from hogan_bot.fetch_polymarket import score_polymarket_opportunities

    opportunities = score_polymarket_opportunities(
        [
            {
                "id": "m1",
                "question": "Will Ethereum fall below $2,000 this week?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.60", "0.40"]',
                "poly_clob_spread": 0.02,
                "liquidity": "50000",
            }
        ],
        hogan_eth_bull_prob=0.70,
    )

    assert opportunities[0].category == "eth"
    assert opportunities[0].candidate_side == "buy_no"


def test_score_polymarket_opportunities_does_not_match_eth_inside_names():
    from hogan_bot.fetch_polymarket import score_polymarket_opportunities

    opportunities = score_polymarket_opportunities(
        [
            {
                "id": "m1",
                "question": "Will Beth Van Duyne win the 2026 Texas Republican Primary?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.60", "0.40"]',
                "liquidity": "100000",
            }
        ],
        hogan_eth_bull_prob=0.70,
    )

    assert opportunities == []


def test_fetch_and_store_writes_public_metrics(monkeypatch, tmp_path):
    from hogan_bot.fetch_polymarket import fetch_and_store
    from hogan_bot.storage import _create_schema

    db_path = tmp_path / "hogan.db"
    conn = sqlite3.connect(db_path)
    _create_schema(conn)
    conn.close()

    def _markets(limit):
        return [
            {
                "question": "Will Bitcoin be above $100,000 in 2026?",
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.55", "0.45"],
                "liquidity": 100.0,
            }
        ]

    monkeypatch.setattr("hogan_bot.fetch_polymarket.fetch_active_markets", _markets)
    monkeypatch.setattr("hogan_bot.fetch_polymarket.enrich_clob_snapshots", lambda markets, max_markets: markets)

    written = fetch_and_store(db_path=str(db_path), limit=1, hogan_btc_bull_prob=0.75)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT metric, value FROM onchain_metrics WHERE symbol='BTC/USD'"
    ).fetchall()
    conn.close()
    assert written >= 1
    assert ("poly_btc_bull_prob", 0.55) in rows
    assert any(metric == "poly_top_edge_score" for metric, _value in rows)


def test_agent_pipeline_reads_polymarket_metrics_point_in_time():
    from hogan_bot.agent_pipeline import MacroAgent, SentimentAgent
    from hogan_bot.storage import _create_schema

    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    conn.executemany(
        "INSERT INTO onchain_metrics (symbol, metric, date, value) VALUES (?, ?, ?, ?)",
        [
            ("BTC/USD", "poly_btc_bull_prob", "2024-01-01", 0.70),
            ("BTC/USD", "poly_btc_bull_prob", "2024-01-03", 0.20),
            ("BTC/USD", "poly_crypto_risk_prob", "2024-01-01", 0.25),
            ("BTC/USD", "poly_macro_risk_prob", "2024-01-01", 0.90),
        ],
    )

    as_of_ms = 1_704_153_600_000  # 2024-01-02T00:00:00Z
    sentiment = SentimentAgent(conn=conn, symbol="BTC/USD").analyze(as_of_ms=as_of_ms)
    macro = MacroAgent(conn=conn, symbol="BTC/USD").analyze(as_of_ms=as_of_ms)

    assert sentiment.details["polymarket_bull"] == 0.70
    assert sentiment.details["polymarket_crypto_risk"] == 0.75
    assert macro.details["poly_macro_risk_prob"] == 0.90
    assert not bool(macro.risk_on)


def test_polymarket_swarm_agent_votes_and_vetoes_on_conflict():
    from hogan_bot.swarm_decision.agents.polymarket import PolymarketAgent

    signal = SimpleNamespace(
        action="sell",
        sentiment=SimpleNamespace(details={
            "polymarket_bull": 0.90,
            "polymarket_crypto_risk": 0.80,
        }),
        macro=SimpleNamespace(details={"poly_macro_risk_prob": 0.20}),
    )
    vote = PolymarketAgent(veto_threshold=0.65).vote(
        symbol="BTC/USD",
        candles=None,
        as_of_ms=None,
        shared_context={"pipeline_signal": signal},
    )

    assert vote.agent_id == "polymarket_v1"
    assert vote.action == "buy"
    assert vote.veto is True
    assert vote.size_scale == 0.0
    assert "polymarket_strong_conflict" in vote.block_reasons


def test_polymarket_swarm_agent_holds_without_signal():
    from hogan_bot.swarm_decision.agents.polymarket import PolymarketAgent

    vote = PolymarketAgent().vote(
        symbol="BTC/USD",
        candles=None,
        as_of_ms=None,
        shared_context={"pipeline_signal": SimpleNamespace(action="hold")},
    )

    assert vote.action == "hold"
    assert vote.confidence == 0.0
    assert "polymarket_no_signal" in vote.block_reasons
