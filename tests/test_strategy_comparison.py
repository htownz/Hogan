from __future__ import annotations

import json

from scripts import strategy_comparison as sc


def test_rank_results_prefers_gate_then_calmar():
    results = [
        {"scenario": "a", "passes_gate": False, "mean_calmar": 1.0, "mean_return_pct": 3.0},
        {"scenario": "b", "passes_gate": True, "mean_calmar": 0.1, "mean_return_pct": 0.1},
        {"scenario": "c", "passes_gate": False, "mean_calmar": 2.0, "mean_return_pct": 1.0},
    ]

    ranked = sc.rank_results(results)

    assert [row["scenario"] for row in ranked] == ["b", "c", "a"]


def test_build_walk_forward_command_includes_fixed_geometry(tmp_path):
    scenario = sc.StrategyScenario("ml_sizer", "ML sizer", ("--ml-sizer",))
    output_path = sc.PROJECT_ROOT / "reports" / "validation" / "wf_test.json"

    cmd = sc.build_walk_forward_command(
        scenario,
        python_exe="python",
        db="data/hogan.db",
        symbol="BTC/USD",
        timeframe="1h",
        n_splits=2,
        min_train=16000,
        min_test=1000,
        min_calmar=0.0,
        output_path=output_path,
    )

    assert cmd[:3] == ["python", "-m", "hogan_bot.walk_forward"]
    assert "--db" in cmd
    assert "data/hogan.db" in cmd
    assert "--min-train" in cmd
    assert "16000" in cmd
    assert "--min-test" in cmd
    assert "1000" in cmd
    assert "--ml-sizer" in cmd
    assert str(output_path.relative_to(sc.PROJECT_ROOT)) in cmd


def test_summarize_result_reads_report(tmp_path):
    report_path = tmp_path / "wf.json"
    log_path = tmp_path / "wf.log"
    report_path.write_text(
        json.dumps({
            "summary": {
                "passes_gate": True,
                "mean_return_pct": 1.2,
                "mean_calmar": 0.5,
                "worst_drawdown_pct": 3.4,
            }
        }),
        encoding="utf-8",
    )

    result = sc.summarize_result(
        sc.DEFAULT_SCENARIOS[0],
        command=["python"],
        exit_code=0,
        report_path=report_path,
        log_path=log_path,
    )

    assert result["passes_gate"] is True
    assert result["mean_return_pct"] == 1.2
    assert result["mean_calmar"] == 0.5
    assert result["worst_drawdown_pct"] == 3.4


def test_dry_run_main_writes_manifest(tmp_path):
    out_dir = tmp_path / "comparison"

    exit_code = sc.main([
        "--dry-run",
        "--output-dir",
        str(out_dir),
        "--scenario",
        "ml_sizer",
    ])

    manifests = list(out_dir.glob("strategy_comparison_*.json"))
    assert exit_code == 0
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["best_scenario"] == "ml_sizer"
    assert payload["ranked_results"][0]["scenario"] == "ml_sizer"
    assert payload["ranked_results"][0]["exit_code"] is None


def test_scenario_by_name_includes_baselines_when_requested():
    default = sc.scenario_by_name(include_baselines=False)
    expanded = sc.scenario_by_name(include_baselines=True)
    assert len(expanded) > len(default)
    expanded_names = {scenario.name for scenario in expanded}
    assert {"baseline_buy_hold", "baseline_ma_trend", "baseline_rsi_mean_revert", "baseline_breakout"} <= expanded_names


def test_scenario_by_name_resolves_baseline_explicitly():
    selected = sc.scenario_by_name(["baseline_buy_hold"])
    assert len(selected) == 1
    assert selected[0].kind == "baseline"
    assert selected[0].baseline_name == "buy_hold"


def test_run_baseline_scenario_writes_compatible_report(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    from scripts import simple_baselines as sb

    rng = np.random.default_rng(7)
    rets = rng.normal(0.0001, 0.005, 60)
    closes = 30_000.0 * np.cumprod(1 + rets)
    ts = pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC")
    candles = pd.DataFrame(
        {
            "ts_ms": (ts.astype("int64") // 10**6).to_numpy(),
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": np.full(60, 10.0),
            "timestamp": ts,
        }
    )
    monkeypatch.setattr(sb, "_load_candles", lambda *args, **kwargs: candles)

    scenario = sc.scenario_by_name(["baseline_buy_hold"])[0]
    report_path = tmp_path / "wf_baseline_buy_hold.json"
    log_path = tmp_path / "wf_baseline_buy_hold.log"

    rc = sc.run_baseline_scenario(
        scenario,
        db="data/dummy.db",
        symbol="BTC/USD",
        timeframe="1h",
        n_splits=2,
        min_train=20,
        min_test=10,
        report_path=report_path,
        log_path=log_path,
    )

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["baseline"] == "buy_hold"
    assert payload["summary"]["n_windows"] == 2
