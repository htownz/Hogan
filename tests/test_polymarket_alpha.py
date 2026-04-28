from __future__ import annotations

import sqlite3


def test_calibration_metrics_and_bias_diagnostics():
    from hogan_bot.polymarket_calibration import (
        brier_score,
        calibration_bins,
        favorite_longshot_bias,
        log_loss,
    )

    probs = [0.10, 0.20, 0.80, 0.90]
    outcomes = [0, 0, 1, 1]

    assert brier_score(probs, outcomes) < 0.05
    assert log_loss(probs, outcomes) < 0.25
    bins = calibration_bins(probs, outcomes, n_bins=5)
    assert sum(bin_.count for bin_ in bins) == 4
    bias = favorite_longshot_bias(probs, outcomes)
    assert bias["longshot_overpricing"] > 0
    assert bias["favorite_underpricing"] > 0


def test_edge_engine_scores_shadow_trade_and_rejects_bad_costs():
    from hogan_bot.fetch_polymarket import PolymarketOpportunity
    from hogan_bot.polymarket_edge import assess_opportunity_edge

    good = PolymarketOpportunity(
        market_id="m1",
        slug="btc-100k",
        question="Will Bitcoin reach $100,000?",
        category="btc",
        candidate_side="buy_yes",
        crowd_prob=0.42,
        hogan_prob=0.72,
        edge_score=1.0,
        liquidity_score=0.80,
        spread_score=0.95,
        catalyst_score=0.80,
        total_score=0.90,
        rationale="test",
    )
    assessment = assess_opportunity_edge(good, fee_rate=0.0, slippage_bps=5)
    assert assessment.after_cost_ev > 0
    assert assessment.decision == "shadow_trade"

    bad = PolymarketOpportunity(
        market_id="m2",
        slug="illiquid",
        question="Will Bitcoin reach $1?",
        category="btc",
        candidate_side="buy_yes",
        crowd_prob=0.50,
        hogan_prob=0.51,
        edge_score=0.05,
        liquidity_score=0.05,
        spread_score=0.10,
        catalyst_score=0.10,
        total_score=0.10,
        rationale="test",
    )
    bad_assessment = assess_opportunity_edge(bad)
    assert bad_assessment.decision == "reject"
    assert "low_liquidity" in bad_assessment.reject_reasons


def test_arbitrage_alerts_detect_ladder_and_group_overpricing():
    from hogan_bot.polymarket_arbitrage import (
        detect_crypto_ladder_violations,
        detect_mutually_exclusive_overpricing,
    )

    ladder_alerts = detect_crypto_ladder_violations([
        {
            "id": "low",
            "question": "Will Bitcoin reach $100,000?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.40", "0.60"]',
        },
        {
            "id": "high",
            "question": "Will Bitcoin reach $120,000?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.55", "0.45"]',
        },
    ])
    assert ladder_alerts
    assert ladder_alerts[0].kind == "crypto_ladder_monotonicity"

    group_alerts = detect_mutually_exclusive_overpricing([
        {
            "id": "a",
            "eventSlug": "winner",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.70", "0.30"]',
        },
        {
            "id": "b",
            "eventSlug": "winner",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.50", "0.50"]',
        },
    ])
    assert group_alerts
    assert group_alerts[0].kind == "mutually_exclusive_overpricing"


def test_opportunity_storage_and_shadow_ledger_round_trip():
    from hogan_bot.fetch_polymarket import PolymarketOpportunity
    from hogan_bot.polymarket_shadow import (
        close_shadow,
        open_shadow_from_opportunity,
        promotion_snapshot,
    )
    from hogan_bot.storage import (
        _create_schema,
        insert_polymarket_opportunities,
        load_polymarket_opportunities,
    )

    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    opp = PolymarketOpportunity(
        market_id="m1",
        slug="btc-100k",
        question="Will Bitcoin reach $100,000?",
        category="btc",
        candidate_side="buy_yes",
        crowd_prob=0.42,
        hogan_prob=0.72,
        edge_score=1.0,
        liquidity_score=0.80,
        spread_score=0.95,
        catalyst_score=0.80,
        total_score=0.90,
        rationale="test",
    )

    assert insert_polymarket_opportunities(conn, "BTC/USD", 1_700_000_000_000, [opp.to_dict()]) == 1
    loaded = load_polymarket_opportunities(conn)
    assert loaded.iloc[0]["market_id"] == "m1"

    trade_id = open_shadow_from_opportunity(conn, opp, size_usd=10.0)
    pnl = close_shadow(conn, trade_id, exit_prob=0.52)
    assert pnl > 0
    snapshot = promotion_snapshot(conn)
    assert snapshot["trades"] == 1.0
    assert snapshot["win_rate"] == 1.0


def test_polymarket_promotion_gate_requires_shadow_evidence():
    from hogan_bot.polymarket_promotion import evaluate_polymarket_promotion

    rejected = evaluate_polymarket_promotion(
        {"trades": 3.0, "total_pnl": 12.0, "avg_pnl": 4.0, "win_rate": 0.8},
        min_trades=10,
    )
    assert rejected.approved is False
    assert any("insufficient_shadow_trades" in reason for reason in rejected.reasons)

    approved = evaluate_polymarket_promotion(
        {"trades": 60.0, "total_pnl": 120.0, "avg_pnl": 2.0, "win_rate": 0.62},
    )
    assert approved.approved is True
    assert approved.reasons == []
