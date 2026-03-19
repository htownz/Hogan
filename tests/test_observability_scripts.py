"""Smoke tests for shadow_report.py and promotion_check.py.

Verifies that both scripts import cleanly, produce valid reports on an
empty DB and on a DB seeded with synthetic swarm data, and that the
CLI entrypoints exit without crashing.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from hogan_bot.storage import _create_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    _create_schema(conn)
    return conn


def _seed_shadow_data(conn: sqlite3.Connection, n: int = 20) -> None:
    """Insert n synthetic shadow decisions + 4 votes each + baseline decisions."""
    for i in range(n):
        ts_ms = 1700000000000 + i * 3600_000
        vetoed = 1 if i % 5 == 0 else 0
        action = "hold" if vetoed else ("buy" if i % 3 == 0 else "sell")
        regime = ["trending", "ranging", "volatile", "risk_off"][i % 4]
        decision_json = json.dumps({"regime": regime})
        block_reasons = json.dumps(["stale_data"] if vetoed else [])
        weights = json.dumps({"pipeline_v1": 0.4, "risk_steward_v1": 0.25,
                              "data_guardian_v1": 0.2, "execution_cost_v1": 0.15})

        cur = conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms, symbol, timeframe, as_of_ms, mode, final_action,
                final_conf, final_scale, agreement, entropy, vetoed,
                block_reasons_json, weights_json, decision_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts_ms, "BTC/USD", "1h", ts_ms, "shadow", action,
             0.65, 1.0, 0.75, 0.3, vetoed,
             block_reasons, weights, decision_json),
        )
        dec_id = cur.lastrowid

        for agent in ["pipeline_v1", "risk_steward_v1", "data_guardian_v1", "execution_cost_v1"]:
            a_veto = 1 if (vetoed and agent == "data_guardian_v1") else 0
            a_reasons = json.dumps(["stale_data"] if a_veto else [])
            vote_json = json.dumps({"agent_id": agent, "action": action})
            conn.execute(
                """INSERT INTO swarm_agent_votes
                   (ts_ms, symbol, timeframe, as_of_ms, agent_id, action,
                    confidence, expected_edge_bps, size_scale, veto,
                    block_reasons_json, vote_json, decision_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts_ms, "BTC/USD", "1h", ts_ms, agent, action,
                 0.7, 5.0, 1.0, a_veto, a_reasons, vote_json, dec_id),
            )

        # Matching baseline decision_log entry
        realized_pnl = -0.5 if vetoed else 1.2
        conn.execute(
            """INSERT INTO decision_log
               (ts_ms, symbol, final_action, final_confidence, position_size, realized_pnl)
               VALUES (?,?,?,?,?,?)""",
            (ts_ms, "BTC/USD", action, 0.6, 100.0, realized_pnl),
        )

    conn.commit()


# ===================================================================
# shadow_report.py
# ===================================================================

class TestShadowReport:
    def test_import(self):
        from scripts.shadow_report import build_shadow_report, ShadowReport  # noqa: F401

    def test_empty_db(self):
        from scripts.shadow_report import build_shadow_report
        conn = _in_memory_db()
        rpt = build_shadow_report(conn)
        assert rpt.total_shadow_decisions == 0
        assert rpt.recommendation in ("collecting", "hold")
        conn.close()

    def test_seeded_db(self):
        from scripts.shadow_report import build_shadow_report
        conn = _in_memory_db()
        _seed_shadow_data(conn, n=20)
        rpt = build_shadow_report(conn, symbol="BTC/USD")
        assert rpt.total_shadow_decisions == 20
        assert rpt.would_trade > 0
        assert rpt.veto_count > 0
        assert len(rpt.agent_leaderboard) == 4
        assert len(rpt.gates) >= 5
        conn.close()

    def test_gates_fail_on_small_sample(self):
        from scripts.shadow_report import build_shadow_report
        conn = _in_memory_db()
        _seed_shadow_data(conn, n=5)
        rpt = build_shadow_report(conn, symbol="BTC/USD")
        assert rpt.recommendation != "advance"
        assert any(not g["passed"] for g in rpt.gates)
        conn.close()

    def test_json_output(self):
        from scripts.shadow_report import build_shadow_report
        conn = _in_memory_db()
        _seed_shadow_data(conn, n=10)
        rpt = build_shadow_report(conn)
        d = rpt.to_dict()
        assert isinstance(d, dict)
        assert "gates" in d
        assert "recommendation" in d
        serialized = json.dumps(d, default=str)
        assert len(serialized) > 50
        conn.close()

    def test_cli_no_crash(self, tmp_path):
        """CLI doesn't crash on a seeded DB file."""
        from scripts.shadow_report import main
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))
        _create_schema(conn)
        _seed_shadow_data(conn, n=10)
        conn.close()
        exit_code = main(["--db", str(db_file), "--symbol", "BTC/USD"])
        assert exit_code in (0, 1)

    def test_cli_json_flag(self, tmp_path, capsys):
        from scripts.shadow_report import main
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))
        _create_schema(conn)
        _seed_shadow_data(conn, n=10)
        conn.close()
        exit_code = main(["--db", str(db_file), "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "gates" in parsed
        assert exit_code in (0, 1)

    def test_cli_missing_db(self):
        from scripts.shadow_report import main
        exit_code = main(["--db", "nonexistent.db"])
        assert exit_code == 1


# ===================================================================
# promotion_check.py
# ===================================================================

class TestPromotionCheck:
    def test_import(self):
        from scripts.promotion_check import build_promotion_report, detect_phase  # noqa: F401

    def test_empty_db_phase0(self):
        from scripts.promotion_check import build_promotion_report
        conn = _in_memory_db()
        rpt = build_promotion_report(conn)
        assert rpt.current_phase == "Phase0_Certification"
        conn.close()

    def test_seeded_db_phase1(self):
        from scripts.promotion_check import build_promotion_report
        conn = _in_memory_db()
        _seed_shadow_data(conn, n=20)
        rpt = build_promotion_report(conn, symbol="BTC/USD")
        assert rpt.current_phase == "Phase1_Shadow"
        assert rpt.next_phase == "Phase2_VetoOnly"
        assert len(rpt.gates) >= 4
        conn.close()

    def test_evidence_populated(self):
        from scripts.promotion_check import build_promotion_report
        conn = _in_memory_db()
        _seed_shadow_data(conn, n=20)
        rpt = build_promotion_report(conn, symbol="BTC/USD")
        assert rpt.evidence["shadow_decisions"] == 20
        assert rpt.evidence["would_trade"] > 0
        assert rpt.evidence["veto_count"] > 0
        conn.close()

    def test_json_serializable(self):
        from scripts.promotion_check import build_promotion_report
        conn = _in_memory_db()
        _seed_shadow_data(conn, n=10)
        rpt = build_promotion_report(conn)
        d = rpt.to_dict()
        serialized = json.dumps(d, default=str)
        assert len(serialized) > 50
        conn.close()

    def test_cli_no_crash(self, tmp_path):
        from scripts.promotion_check import main
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))
        _create_schema(conn)
        _seed_shadow_data(conn, n=10)
        conn.close()
        exit_code = main(["--db", str(db_file), "--symbol", "BTC/USD"])
        assert exit_code in (0, 1)

    def test_cli_json_flag(self, tmp_path, capsys):
        from scripts.promotion_check import main
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))
        _create_schema(conn)
        _seed_shadow_data(conn, n=10)
        conn.close()
        exit_code = main(["--db", str(db_file), "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "current_phase" in parsed
        assert "gates" in parsed
        assert exit_code in (0, 1)

    def test_cli_missing_db(self):
        from scripts.promotion_check import main
        exit_code = main(["--db", "nonexistent.db"])
        assert exit_code == 1

    def test_detect_phase_transitions(self):
        from scripts.promotion_check import detect_phase
        assert detect_phase({"shadow_decisions": 0, "latest_mode": None,
                            "active_vetoes": 0, "closed_paper_trades": 0,
                            "weight_proposals": 0}) == "Phase0_Certification"
        assert detect_phase({"shadow_decisions": 100, "latest_mode": "shadow",
                            "active_vetoes": 0, "closed_paper_trades": 0,
                            "weight_proposals": 0}) == "Phase1_Shadow"
        assert detect_phase({"shadow_decisions": 300, "latest_mode": "active",
                            "active_vetoes": 30, "closed_paper_trades": 40,
                            "weight_proposals": 0}) == "Phase2_VetoOnly"
        assert detect_phase({"shadow_decisions": 300, "latest_mode": "active",
                            "active_vetoes": 50, "closed_paper_trades": 80,
                            "weight_proposals": 0}) == "Phase3_SizeEntry"

    def test_detect_phase_swarm_phase_pin(self):
        """Operator-pinned swarm_phase overrides DB inference."""
        from scripts.promotion_check import detect_phase
        ev = {"shadow_decisions": 500, "latest_mode": "shadow",
              "active_vetoes": 0, "closed_paper_trades": 0,
              "weight_proposals": 0, "swarm_phase": "paper_veto"}
        assert detect_phase(ev) == "Phase2_VetoOnly"

    def test_detect_phase_micro_live(self):
        from scripts.promotion_check import detect_phase
        ev = {"shadow_decisions": 500, "latest_mode": "active",
              "active_vetoes": 100, "closed_paper_trades": 300,
              "weight_proposals": 5, "swarm_phase": "micro_live"}
        assert detect_phase(ev) == "Phase6_MicroLive"

    def test_phase6_checker(self):
        from scripts.promotion_check import _check_phase6
        ev = {"closed_paper_trades": 35, "total_paper_pnl": 150.0,
              "paper_win_rate": 0.55}
        gates, blockers = _check_phase6(ev)
        assert len(gates) == 3
        assert all(g.passed for g in gates)
        assert len(blockers) == 0

    def test_phase6_checker_fails(self):
        from scripts.promotion_check import _check_phase6
        ev = {"closed_paper_trades": 10, "total_paper_pnl": -50.0,
              "paper_win_rate": 0.30}
        gates, blockers = _check_phase6(ev)
        assert not all(g.passed for g in gates)
        assert len(blockers) >= 2

    def test_phase0_checks_policy_core(self):
        from scripts.promotion_check import _check_phase0
        ev_on = {"use_policy_core": True}
        gates, blockers = _check_phase0(ev_on)
        pc_gate = [g for g in gates if g.name == "use_policy_core_enabled"][0]
        assert pc_gate.passed

        ev_off = {"use_policy_core": False}
        gates2, blockers2 = _check_phase0(ev_off)
        pc_gate2 = [g for g in gates2 if g.name == "use_policy_core_enabled"][0]
        assert not pc_gate2.passed
        assert len(blockers2) > 0


# ===================================================================
# Config defaults
# ===================================================================

class TestConfigDefaults:
    def test_use_policy_core_default_true(self):
        from hogan_bot.config import BotConfig
        cfg = BotConfig()
        assert cfg.use_policy_core is True

    def test_swarm_phase_default(self):
        from hogan_bot.config import BotConfig
        cfg = BotConfig()
        assert cfg.swarm_phase == "certification"

    def test_load_config_policy_core_true(self):
        from hogan_bot.config import load_config
        cfg = load_config()
        assert cfg.use_policy_core is True
