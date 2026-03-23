"""Tests for swarm policy observability helpers and validation battery CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hogan_bot.metrics import record_swarm_policy_events
from hogan_bot.swarm_decision.types import SwarmDecision
from hogan_bot.swarm_metrics import aggregate_swarm_policy_block_reasons

ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_swarm_policy_block_reasons_counts_tags() -> None:
    rows = [
        json.dumps(["edge_block", "swarm_direction_clash"]),
        json.dumps(["swarm_blocked_unsigned_signal"]),
        "not-json",
        "",
        None,
        json.dumps({"bad": 1}),
    ]
    out = aggregate_swarm_policy_block_reasons(rows)
    assert out["rows_with_swarm_policy_tags"] == 2
    assert out["total_swarm_policy_tag_events"] == 2
    assert out["counts"]["swarm_direction_clash"] == 1
    assert out["counts"]["swarm_blocked_unsigned_signal"] == 1


def test_record_swarm_policy_events_no_raise() -> None:
    sd = SwarmDecision(
        final_action="hold",
        final_confidence=0.5,
        final_size_scale=0.0,
        agreement=0.5,
        entropy=0.5,
        weights_used={},
        votes=[],
        vetoed=True,
        dominant_veto_agent="risk_steward_v1",
    )
    record_swarm_policy_events(
        swarm_mode="active",
        swarm_decision=sd,
        block_reasons=["swarm_direction_clash"],
    )
    record_swarm_policy_events(
        swarm_mode="shadow",
        swarm_decision=None,
        block_reasons=[],
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["--skip-walk-forward", "--skip-certification", "--output-dir", "reports/validation"],
    ],
)
def test_run_validation_battery_cli(args: list[str]) -> None:
    script = ROOT / "scripts" / "run_validation_battery.py"
    r = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0


def test_promotion_checklist_doc_exists() -> None:
    assert (ROOT / "docs" / "PROMOTION_CHECKLIST.md").is_file()
    assert (ROOT / "docs" / "SWARM_CONDITIONAL_TUNING.md").is_file()
    assert (ROOT / "docs" / "STRATEGY_CHANGE_GATE.md").is_file()
