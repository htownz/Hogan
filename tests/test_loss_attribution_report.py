from __future__ import annotations

import json

from scripts.loss_attribution_report import build_report


def test_loss_attribution_groups_dominant_buckets(tmp_path):
    report_path = tmp_path / "wf.json"
    report_path.write_text(
        json.dumps({
            "windows": [
                {
                    "window_idx": 0,
                    "signal_funnel": {
                        "quality_gate_final_conf": 3,
                        "blocked_already_long": 2,
                    },
                    "closed_trades": [
                        {
                            "side": "long",
                            "pnl_pct": -1.2,
                            "entry_regime": "ranging",
                            "exit_regime": "trending_down",
                            "exit_reason": "proactive_trend_reversal",
                            "bars_held": 7,
                            "entry_bar": 10,
                            "exit_bar": 17,
                            "max_adverse_pct": -1.5,
                            "max_favorable_pct": 0.4,
                        },
                        {
                            "side": "short",
                            "pnl_pct": 0.8,
                            "entry_regime": "volatile",
                            "exit_regime": "volatile",
                            "exit_reason": "short_max_hold_time",
                        },
                    ],
                }
            ],
        }),
        encoding="utf-8",
    )

    report = build_report([report_path])

    assert report["summary"]["trades"] == 2
    assert report["summary"]["loss_rate"] == 0.5
    assert report["by_regime"] == report["by_entry_regime"]
    assert report["by_entry_regime"]["ranging"]["loss_drag_pct"] == -1.2
    assert report["by_exit_regime"]["trending_down"]["loss_drag_pct"] == -1.2
    assert report["by_exit_reason"]["proactive_trend_reversal"]["loss_drag_pct"] == -1.2
    assert report["top_loss_buckets"][0]["bucket"] == "ranging|long|proactive_trend_reversal"
    assert report["top_exit_loss_buckets"][0]["bucket"] == "trending_down|long|proactive_trend_reversal"
    assert report["worst_trades_by_entry_bucket"][0]["bucket"] == "ranging|long|proactive_trend_reversal"
    assert report["worst_trades_by_entry_bucket"][0]["worst_trades"][0]["bars_held"] == 7
    assert report["worst_trades_by_exit_bucket"][0]["bucket"] == "trending_down|long|proactive_trend_reversal"
    assert report["worst_trades_by_exit_bucket"][0]["worst_trades"][0]["max_adverse_pct"] == -1.5
    assert report["gate_and_block_counts"]["quality_gate_final_conf"] == 3
    assert report["gate_and_block_counts"]["blocked_already_long"] == 2
