from __future__ import annotations

import inspect
from types import SimpleNamespace


def _fake_alpha_result():
    return SimpleNamespace(
        report_path="reports/polymarket/test.md",
        opportunities=[],
        shadow_opened=0,
        shadow_ledger=SimpleNamespace(closed=0, unrealized_pnl=0.0),
        btc_prob=0.42,
        eth_prob=None,
        arbitrage_alerts=0,
        promotion_approved=False,
        authority_mode="research",
        promotion_reasons=[],
    )


def _fake_recommendation_result():
    return SimpleNamespace(
        opportunities=[],
        candidates=[],
        arbitrage_alerts=0,
        btc_prob=0.42,
        eth_prob=None,
    )


def test_polymarket_service_scan_calls_alpha_with_research_default(monkeypatch):
    from hogan_bot import polymarket_service

    captured = {}

    def _run_alpha_lab(**kwargs):
        captured.update(kwargs)
        return _fake_alpha_result()

    monkeypatch.setattr(polymarket_service, "run_alpha_lab", _run_alpha_lab)

    args = polymarket_service.parse_args(["--mode", "scan", "--db", "test.db"])
    rc = polymarket_service.run_service(args)

    assert rc == 0
    assert captured["db_path"] == "test.db"
    assert captured["authority_mode"] == "research"
    assert captured["auto_shadow"] is True
    assert captured["use_long_horizon_model"] is True


def test_polymarket_service_recommendations_only_uses_no_write_runner(monkeypatch):
    from hogan_bot import polymarket_service

    captured = {}
    printed = {}

    def _run_recommendations_only(**kwargs):
        captured.update(kwargs)
        return _fake_recommendation_result()

    monkeypatch.setattr(polymarket_service, "run_recommendations_only", _run_recommendations_only)
    monkeypatch.setattr(polymarket_service, "print_recommendations", lambda result, limit: printed.update({"limit": limit}))

    args = polymarket_service.parse_args([
        "--mode",
        "recommendations-only",
        "--recommendation-limit",
        "3",
        "--no-long-horizon-model",
    ])
    rc = polymarket_service.run_service(args)

    assert rc == 0
    assert captured["use_long_horizon_model"] is False
    assert printed["limit"] == 3


def test_polymarket_service_daemon_honors_iteration_limit(monkeypatch):
    from hogan_bot import polymarket_service

    calls = []
    sleeps = []
    monkeypatch.setattr(polymarket_service, "run_service_once", lambda args: calls.append(args.mode))
    monkeypatch.setattr(polymarket_service.time, "sleep", lambda seconds: sleeps.append(seconds))

    args = polymarket_service.parse_args([
        "--mode",
        "daemon",
        "--iterations",
        "2",
        "--interval-minutes",
        "0.01",
    ])
    rc = polymarket_service.run_service(args)

    assert rc == 0
    assert calls == ["daemon", "daemon"]
    assert sleeps == [0.6]


def test_polymarket_service_has_no_real_trading_surface():
    from hogan_bot import polymarket_service

    source = inspect.getsource(polymarket_service).lower()

    assert "private_key" not in source
    assert "seed_phrase" not in source
    assert "place_order" not in source
