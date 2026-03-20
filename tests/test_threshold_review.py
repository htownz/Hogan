"""Tests for threshold_review — per-agent review and recommendation."""
from __future__ import annotations

import json
import sqlite3

from hogan_bot.threshold_review import (
    render_review_json,
    render_review_md,
    review_agent,
)


def _make_conn():
    conn = sqlite3.connect(":memory:")
    from hogan_bot.storage import _create_schema
    _create_schema(conn)
    return conn


def _seed_stalled(conn: sqlite3.Connection, n: int = 100) -> None:
    base_ts = 1_742_169_600_000
    for i in range(n):
        conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms,symbol,timeframe,mode,final_action,final_conf,final_scale,
                agreement,entropy,vetoed,block_reasons_json,weights_json,decision_json,
                pre_veto_action,pre_veto_confidence,pre_veto_agreement,pre_veto_entropy,
                dominant_veto_agent,veto_count,veto_agents_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (base_ts + i * 60_000, "BTC/USD", "1h", "shadow", "hold",
             0.0, 0.0, 1.0, 0.0, 1, json.dumps(["risk_steward_v1:vol_spike"]),
             json.dumps({}), json.dumps({}),
             "buy", 0.6, 0.7, 0.3, "risk_steward_v1", 1, json.dumps(["risk_steward_v1"])),
        )
        conn.execute(
            """INSERT INTO swarm_agent_votes
               (ts_ms,symbol,timeframe,agent_id,action,confidence,size_scale,
                veto,block_reasons_json,vote_json,decision_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (base_ts + i * 60_000, "BTC/USD", "1h", "risk_steward_v1",
             "hold", 0.0, 0.0, 1, json.dumps(["vol_spike"]),
             json.dumps({}), i + 1),
        )
    conn.commit()


def _seed_healthy(conn: sqlite3.Connection, n: int = 100) -> None:
    base_ts = 1_742_169_600_000
    for i in range(n):
        vetoed = 1 if i < 20 else 0
        action = "hold" if vetoed else ("buy" if i % 2 == 0 else "sell")
        conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms,symbol,timeframe,mode,final_action,final_conf,final_scale,
                agreement,entropy,vetoed,block_reasons_json,weights_json,decision_json,
                pre_veto_action,pre_veto_confidence,pre_veto_agreement,pre_veto_entropy)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (base_ts + i * 60_000, "BTC/USD", "1h", "shadow", action,
             0.0 if vetoed else 0.7, 0.0 if vetoed else 0.8,
             1.0 if vetoed else 0.65, 0.0 if vetoed else 0.5, vetoed,
             json.dumps([]), json.dumps({}), json.dumps({}),
             action, 0.7, 0.65, 0.5),
        )
    conn.commit()


class TestReviewAgent:
    def test_stalled_recommends_disable_veto_or_worse(self):
        conn = _make_conn()
        _seed_stalled(conn)
        result = review_agent(conn, "risk_steward_v1", window_hours=72,
                              end_ts_ms=1_742_169_600_000 + 200 * 60_000)
        assert result.recommendation in ("disable_veto_only", "quarantine", "relax_thresholds")
        assert result.would_trade_count == 0
        assert result.veto_ratio > 0.5
        conn.close()

    def test_healthy_recommends_hold(self):
        conn = _make_conn()
        _seed_healthy(conn)
        result = review_agent(conn, "risk_steward_v1", window_hours=72,
                              end_ts_ms=1_742_169_600_000 + 200 * 60_000)
        assert result.recommendation == "hold"
        assert result.would_trade_count > 0
        conn.close()

    def test_empty_db(self):
        conn = _make_conn()
        result = review_agent(conn, "risk_steward_v1")
        assert result.decision_count == 0
        assert result.recommendation == "hold"
        conn.close()

    def test_pre_veto_metrics_populated(self):
        conn = _make_conn()
        _seed_stalled(conn)
        result = review_agent(conn, "risk_steward_v1", window_hours=72,
                              end_ts_ms=1_742_169_600_000 + 200 * 60_000)
        assert result.pre_veto_would_trade_count > 0
        assert result.pre_veto_agreement_mean is not None
        assert result.pre_veto_confidence_mean is not None
        conn.close()

    def test_stall_alerts_populated(self):
        conn = _make_conn()
        _seed_stalled(conn)
        result = review_agent(conn, "risk_steward_v1", window_hours=72,
                              end_ts_ms=1_742_169_600_000 + 200 * 60_000)
        codes = {a.code for a in result.stall_alerts}
        assert "CRITICAL_STALL" in codes
        conn.close()


class TestReviewRendering:
    def test_json_output(self):
        conn = _make_conn()
        _seed_stalled(conn)
        result = review_agent(conn, "risk_steward_v1", window_hours=72,
                              end_ts_ms=1_742_169_600_000 + 200 * 60_000)
        j = render_review_json(result)
        parsed = json.loads(j)
        assert "recommendation" in parsed
        assert "stall_alerts" in parsed
        conn.close()

    def test_md_output(self):
        conn = _make_conn()
        _seed_stalled(conn)
        result = review_agent(conn, "risk_steward_v1", window_hours=72,
                              end_ts_ms=1_742_169_600_000 + 200 * 60_000)
        md = render_review_md(result)
        assert "Threshold Review" in md
        assert "Pre-Veto" in md
        assert "Stall Alerts" in md
        conn.close()
