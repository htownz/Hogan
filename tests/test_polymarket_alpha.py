from __future__ import annotations

import sqlite3
from pathlib import Path


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


def test_intelligence_assessment_gates_shadow_eligibility():
    from hogan_bot.fetch_polymarket import (
        normalize_market_snapshot,
        score_polymarket_opportunities,
    )
    from hogan_bot.polymarket_edge import assess_opportunity_edge
    from hogan_bot.polymarket_intelligence import assess_intelligence

    market = {
        "id": "m1",
        "slug": "btc-100k",
        "question": "Will Bitcoin reach $100,000 this month?",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.42", "0.58"]',
        "poly_clob_midpoint": 0.42,
        "poly_clob_spread": 0.01,
        "liquidity": "100000",
        "volume24hr": "25000",
    }
    opp = score_polymarket_opportunities([market], hogan_btc_bull_prob=0.72)[0]
    edge = assess_opportunity_edge(opp)
    assessment = assess_intelligence(opp, edge, normalize_market_snapshot(market))

    assert assessment.recommendation == "shadow_candidate"
    assert assessment.fair_value_source == "hogan_short_term_ml"
    assert assessment.shadow_eligible is True
    assert assessment.evidence_score > 0.5
    assert assessment.confidence > 0.0
    assert assessment.recommended_size_usd > 0.0


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
        cancel_shadow,
        close_shadow,
        load_shadow_ledger,
        open_shadow_from_opportunity,
        print_shadow_ledger,
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
    assert snapshot["max_drawdown"] == 0.0
    assert snapshot["worst_loss_streak"] == 0.0
    assert "data_quality_weighted_pnl" in snapshot
    ledger = load_shadow_ledger(conn, status="closed", limit=5)
    assert ledger[0]["market_id"] == "m1"
    assert ledger[0]["status"] == "closed"
    print_shadow_ledger(conn, status="closed", limit=5)

    cancel_id = open_shadow_from_opportunity(conn, opp, size_usd=10.0)
    assert cancel_shadow(conn, cancel_id, reason="test_cancel") == 0.0
    cancelled = load_shadow_ledger(conn, status="cancelled", limit=5)
    assert cancelled[0]["id"] == cancel_id
    assert cancelled[0]["realized_pnl"] == 0.0
    assert "test_cancel" in cancelled[0]["rationale"]
    try:
        cancel_shadow(conn, cancel_id, reason="already_cancelled")
    except ValueError as exc:
        assert "open Polymarket shadow trade not found" in str(exc)
    else:
        raise AssertionError("expected cancelling a non-open shadow trade to fail")

    snapshot_after_cancel = promotion_snapshot(conn)
    assert snapshot_after_cancel["trades"] == 1.0


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

    rejected_risk = evaluate_polymarket_promotion(
        {
            "trades": 60.0,
            "total_pnl": 120.0,
            "avg_pnl": 2.0,
            "win_rate": 0.62,
            "max_drawdown": 30.0,
            "worst_loss_streak": 6.0,
            "market_type_coverage": 0.0,
            "data_quality_weighted_pnl": -0.1,
        },
    )
    assert rejected_risk.approved is False
    assert any("drawdown_above_gate" in reason for reason in rejected_risk.reasons)
    assert any("loss_streak_above_gate" in reason for reason in rejected_risk.reasons)


def test_alpha_lab_runner_persists_report_and_opens_shadow_once(monkeypatch, tmp_path):
    from hogan_bot.polymarket_alpha import run_alpha_lab

    markets = [
        {
            "id": "m1",
            "slug": "btc-100k",
            "question": "Will Bitcoin reach $100,000 this month?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.42", "0.58"]',
            "poly_clob_spread": 0.01,
            "liquidity": "100000",
            "volume24hr": "25000",
        }
    ]
    monkeypatch.setattr("hogan_bot.polymarket_alpha.fetch_active_markets", lambda limit: markets)
    monkeypatch.setattr("hogan_bot.polymarket_alpha.enrich_clob_snapshots", lambda data, max_markets: data)
    monkeypatch.setattr("hogan_bot.polymarket_alpha.detect_arbitrage_alerts", lambda data: [])

    db_path = tmp_path / "hogan.db"
    report_dir = tmp_path / "reports"
    first = run_alpha_lab(
        db_path=str(db_path),
        limit=1,
        include_clob=False,
        btc_prob=0.72,
        report_dir=str(report_dir),
    )
    second = run_alpha_lab(
        db_path=str(db_path),
        limit=1,
        include_clob=False,
        btc_prob=0.72,
        report_dir=str(report_dir),
    )

    assert first.shadow_opened == 1
    assert first.authority_mode == "shadow"
    assert second.shadow_opened == 0
    report = Path(first.report_path).read_text()
    assert "Polymarket Alpha Lab Report" in report
    assert "## Data Quality" in report
    assert "## Machine Recommendations" in report
    assert "## Shadow Positions" in report
    assert "## Next Action" in report

    conn = sqlite3.connect(db_path)
    open_count = conn.execute(
        "SELECT COUNT(*) FROM polymarket_shadow_trades WHERE status='open'"
    ).fetchone()[0]
    opp_count = conn.execute("SELECT COUNT(*) FROM polymarket_opportunities").fetchone()[0]
    conn.close()
    assert open_count == 1
    assert opp_count >= 1


def test_alpha_lab_authority_modes_and_shadow_budget(monkeypatch, tmp_path):
    from hogan_bot.polymarket_alpha import run_alpha_lab

    markets = [
        {
            "id": "m1",
            "slug": "btc-100k",
            "question": "Will Bitcoin reach $100,000 this month?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.42", "0.58"]',
            "poly_clob_midpoint": 0.42,
            "poly_clob_spread": 0.01,
            "liquidity": "100000",
            "volume24hr": "25000",
        }
    ]
    monkeypatch.setattr("hogan_bot.polymarket_alpha.fetch_active_markets", lambda limit: markets)
    monkeypatch.setattr("hogan_bot.polymarket_alpha.enrich_clob_snapshots", lambda data, max_markets: data)
    monkeypatch.setattr("hogan_bot.polymarket_alpha.detect_arbitrage_alerts", lambda data: [])

    research = run_alpha_lab(
        db_path=str(tmp_path / "research.db"),
        limit=1,
        include_clob=False,
        btc_prob=0.72,
        authority_mode="research",
        report_dir=str(tmp_path / "research_reports"),
    )
    capped = run_alpha_lab(
        db_path=str(tmp_path / "capped.db"),
        limit=1,
        include_clob=False,
        btc_prob=0.72,
        max_open_shadow_trades=0,
        report_dir=str(tmp_path / "capped_reports"),
    )

    assert research.shadow_opened == 0
    assert research.authority_mode == "research"
    assert research.candidates[0].assessment.shadow_eligible is True
    assert capped.shadow_opened == 0
    assert "Authority is `research`" in Path(research.report_path).read_text()


def test_alpha_lab_uses_latest_decision_ml_probability(monkeypatch, tmp_path):
    from hogan_bot.polymarket_alpha import run_alpha_lab
    from hogan_bot.storage import get_connection, log_decision

    markets = [
        {
            "id": "m1",
            "slug": "btc-100k",
            "question": "Will Bitcoin reach $100,000 this month?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.42", "0.58"]',
            "poly_clob_spread": 0.01,
            "liquidity": "100000",
            "volume24hr": "25000",
        }
    ]
    monkeypatch.setattr("hogan_bot.polymarket_alpha.fetch_active_markets", lambda limit: markets)
    monkeypatch.setattr("hogan_bot.polymarket_alpha.enrich_clob_snapshots", lambda data, max_markets: data)
    monkeypatch.setattr("hogan_bot.polymarket_alpha.detect_arbitrage_alerts", lambda data: [])

    db_path = tmp_path / "hogan.db"
    conn = get_connection(str(db_path))
    log_decision(
        conn,
        ts_ms=1_700_000_000_000,
        symbol="BTC/USD",
        final_action="buy",
        ml_up_prob=0.72,
    )
    conn.close()

    result = run_alpha_lab(
        db_path=str(db_path),
        limit=1,
        include_clob=False,
        report_dir=str(tmp_path / "reports"),
    )

    assert result.btc_prob == 0.72
    assert result.opportunities[0].hogan_prob == 0.72
    assert "Hogan BTC probability: `0.7200`" in Path(result.report_path).read_text()


def test_alpha_lab_keeps_long_target_research_and_reports_arbitrage_alert(monkeypatch, tmp_path):
    from hogan_bot.polymarket_alpha import run_alpha_lab
    from hogan_bot.polymarket_arbitrage import ArbitrageAlert

    markets = [
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
    ]
    monkeypatch.setattr("hogan_bot.polymarket_alpha.fetch_active_markets", lambda limit: markets)
    monkeypatch.setattr("hogan_bot.polymarket_alpha.enrich_clob_snapshots", lambda data, max_markets: data)
    monkeypatch.setattr(
        "hogan_bot.polymarket_alpha.detect_arbitrage_alerts",
        lambda data: [
            ArbitrageAlert(
                kind="crypto_ladder_monotonicity",
                market_ids=["low", "high"],
                severity=0.25,
                message="test alert",
            )
        ],
    )

    result = run_alpha_lab(
        db_path=str(tmp_path / "hogan.db"),
        limit=1,
        include_clob=False,
        btc_prob=0.72,
        report_dir=str(tmp_path / "reports"),
    )

    assert result.shadow_opened == 0
    assert result.candidates[0].opportunity.candidate_side == "research"
    assert result.candidates[0].edge.decision == "research"
    report = Path(result.report_path).read_text()
    assert "long_horizon_price_target_requires_calibrated_fair_value" in report
    assert "## Arbitrage Alerts" in report
    assert "test alert" in report


def test_alpha_lab_marks_and_closes_shadow_trade_on_price_move(monkeypatch, tmp_path):
    from hogan_bot.polymarket_alpha import run_alpha_lab

    market = {
        "id": "m1",
        "slug": "btc-100k",
        "question": "Will Bitcoin reach $100,000 this month?",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.42", "0.58"]',
        "poly_clob_spread": 0.01,
        "liquidity": "100000",
        "volume24hr": "25000",
    }
    scans = [
        [dict(market)],
        [dict(market, outcomePrices='["0.56", "0.44"]')],
    ]

    def _markets(limit):
        return scans.pop(0)

    monkeypatch.setattr("hogan_bot.polymarket_alpha.fetch_active_markets", _markets)
    monkeypatch.setattr("hogan_bot.polymarket_alpha.enrich_clob_snapshots", lambda data, max_markets: data)
    monkeypatch.setattr("hogan_bot.polymarket_alpha.detect_arbitrage_alerts", lambda data: [])

    db_path = tmp_path / "hogan.db"
    report_dir = tmp_path / "reports"
    first = run_alpha_lab(
        db_path=str(db_path),
        limit=1,
        include_clob=False,
        btc_prob=0.72,
        report_dir=str(report_dir),
    )
    second = run_alpha_lab(
        db_path=str(db_path),
        limit=1,
        include_clob=False,
        btc_prob=0.72,
        report_dir=str(report_dir),
    )

    assert first.shadow_opened == 1
    assert second.shadow_opened == 0
    assert second.shadow_ledger.marked_open == 1
    assert second.shadow_ledger.closed == 1
    assert "Shadow closed: `1`" in Path(second.report_path).read_text()

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT status, realized_pnl FROM polymarket_shadow_trades WHERE market_id='m1'"
    ).fetchone()
    conn.close()
    assert row[0] == "closed"
    assert row[1] > 0
