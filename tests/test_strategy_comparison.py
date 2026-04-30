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
