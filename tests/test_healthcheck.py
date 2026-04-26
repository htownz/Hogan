from __future__ import annotations

from pathlib import Path

from hogan_bot.healthcheck import LIVE_ACK, run_healthcheck


def _base_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOGAN_DB_PATH", str(tmp_path / "hogan.db"))
    monkeypatch.setenv("HOGAN_PAPER_MODE", "true")
    monkeypatch.setenv("HOGAN_LIVE_MODE", "false")
    monkeypatch.setenv("HOGAN_USE_ML_FILTER", "false")
    monkeypatch.setenv("HOGAN_ML_AS_SIZER", "false")
    monkeypatch.setenv("HOGAN_USE_TRADE_QUALITY", "false")
    monkeypatch.setenv("HOGAN_USE_RL_AGENT", "false")


def test_healthcheck_passes_without_metrics(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)

    errors = run_healthcheck(check_metrics=False)

    assert errors == []


def test_healthcheck_requires_live_ack(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HOGAN_LIVE_MODE", "true")
    monkeypatch.delenv("HOGAN_LIVE_ACK", raising=False)

    errors = run_healthcheck(check_metrics=False)

    assert any(f"HOGAN_LIVE_ACK={LIVE_ACK}" in error for error in errors)


def test_healthcheck_strict_models_reports_missing_model(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HOGAN_ML_AS_SIZER", "true")
    monkeypatch.setenv("HOGAN_ML_MODEL_PATH", str(tmp_path / "missing.pkl"))

    errors = run_healthcheck(check_metrics=False, strict_models=True)

    assert any("required model file(s) missing" in error for error in errors)
