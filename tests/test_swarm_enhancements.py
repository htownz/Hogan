"""Tests for swarm enhancements: outcome writer, weight learner, configurable weights."""
from __future__ import annotations

import json
import sqlite3
import time

import pandas as pd
import pytest

from hogan_bot.storage import _create_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    _create_schema(conn)
    return conn


def _seed_decisions_and_candles(conn: sqlite3.Connection, n: int = 20) -> list[int]:
    """Insert n decisions with matching candle data and decision_log entries.

    Returns list of decision ids.
    """
    ids = []
    base_ts = 1700000000000
    for i in range(n):
        ts_ms = base_ts + i * 3600_000
        action = "buy" if i % 3 == 0 else ("sell" if i % 3 == 1 else "hold")
        vetoed = 1 if i % 7 == 0 else 0
        if vetoed:
            action = "hold"

        cur = conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms, symbol, timeframe, as_of_ms, mode, final_action,
                final_conf, final_scale, agreement, entropy, vetoed,
                block_reasons_json, weights_json, decision_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts_ms, "BTC/USD", "1h", ts_ms, "shadow", action,
             0.65, 1.0, 0.75, 0.3, vetoed,
             json.dumps(["stale_data"] if vetoed else []),
             json.dumps({"pipeline_v1": 0.25, "risk_steward_v1": 0.25,
                         "data_guardian_v1": 0.25, "execution_cost_v1": 0.25}),
             json.dumps({"regime": "trending"})),
        )
        dec_id = cur.lastrowid
        ids.append(dec_id)

        for agent in ["pipeline_v1", "risk_steward_v1", "data_guardian_v1", "execution_cost_v1"]:
            a_veto = 1 if (vetoed and agent == "data_guardian_v1") else 0
            conn.execute(
                """INSERT INTO swarm_agent_votes
                   (ts_ms, symbol, timeframe, as_of_ms, agent_id, action,
                    confidence, expected_edge_bps, size_scale, veto,
                    block_reasons_json, vote_json, decision_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts_ms, "BTC/USD", "1h", ts_ms, agent, action,
                 0.7, 5.0, 1.0, a_veto,
                 json.dumps(["stale_data"] if a_veto else []),
                 json.dumps({"agent_id": agent}), dec_id),
            )

        # Baseline decision_log entry
        conn.execute(
            """INSERT INTO decision_log
               (ts_ms, symbol, final_action, final_confidence, position_size)
               VALUES (?,?,?,?,?)""",
            (ts_ms, "BTC/USD", "buy" if i % 2 == 0 else "sell", 0.6, 100.0),
        )

    # Insert candle data covering the full range + forward
    for i in range(n + 70):
        ts_ms = base_ts + i * 3600_000
        close = 40000 + i * 10 + (i % 5) * 3
        conn.execute(
            """INSERT OR IGNORE INTO candles (ts_ms, symbol, timeframe, open, high, low, close, volume)
               VALUES (?,?,?,?,?,?,?,?)""",
            (ts_ms, "BTC/USD", "1h", close - 5, close + 10, close - 15, close, 1000.0),
        )

    conn.commit()
    return ids


# ===================================================================
# Outcome writer tests
# ===================================================================

class TestOutcomeWriter:
    def test_import(self):
        from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
        assert callable(backfill_outcomes)

    def test_backfill_empty_db(self):
        from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
        conn = _db()
        count = backfill_outcomes(conn, lookback_hours=0)
        assert count == 0
        conn.close()

    def test_backfill_writes_outcomes(self):
        from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
        conn = _db()
        ids = _seed_decisions_and_candles(conn, n=10)
        count = backfill_outcomes(conn, symbol="BTC/USD", lookback_hours=0)
        assert count > 0

        outcomes = pd.read_sql_query("SELECT * FROM swarm_outcomes", conn)
        assert len(outcomes) > 0
        assert "forward_60m_bps" in outcomes.columns
        conn.close()

    def test_backfill_idempotent(self):
        from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
        conn = _db()
        _seed_decisions_and_candles(conn, n=10)
        c1 = backfill_outcomes(conn, lookback_hours=0)
        c2 = backfill_outcomes(conn, lookback_hours=0)
        assert c1 > 0
        assert c2 == 0  # already written
        conn.close()

    def test_outcome_has_veto_correctness(self):
        from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
        conn = _db()
        _seed_decisions_and_candles(conn, n=20)
        backfill_outcomes(conn, lookback_hours=0)

        vetoed_outcomes = pd.read_sql_query(
            """SELECT so.* FROM swarm_outcomes so
               JOIN swarm_decisions sd ON so.decision_id = sd.id
               WHERE sd.vetoed = 1""",
            conn,
        )
        if not vetoed_outcomes.empty:
            assert "was_veto_correct" in vetoed_outcomes.columns
        conn.close()

    def test_outcome_label_values(self):
        from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
        conn = _db()
        _seed_decisions_and_candles(conn, n=10)
        backfill_outcomes(conn, lookback_hours=0)

        labels = pd.read_sql_query(
            "SELECT outcome_label FROM swarm_outcomes", conn
        )["outcome_label"].unique()
        for label in labels:
            assert label in ("win", "loss", "scratch", "pending", "no_trade")
        conn.close()

    def test_tf_to_minutes(self):
        from hogan_bot.swarm_decision.outcome_writer import _tf_to_minutes
        assert _tf_to_minutes("1h") == 60
        assert _tf_to_minutes("30m") == 30
        assert _tf_to_minutes("1d") == 1440
        assert _tf_to_minutes("5m") == 5


# ===================================================================
# Weight learner tests
# ===================================================================

class TestWeightLearner:
    def test_import(self):
        from hogan_bot.swarm_decision.weight_learner import propose_weights
        assert callable(propose_weights)

    def test_propose_weights_no_data(self):
        from hogan_bot.swarm_decision.weight_learner import propose_weights
        conn = _db()
        current = {"a": 0.5, "b": 0.5}
        proposal = propose_weights(conn, current, min_trades=10)
        assert not proposal.min_trades_met
        assert proposal.proposed_weights == current
        conn.close()

    def test_propose_weights_with_data(self):
        from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
        from hogan_bot.swarm_decision.weight_learner import propose_weights

        conn = _db()
        _seed_decisions_and_candles(conn, n=20)
        backfill_outcomes(conn, lookback_hours=0)

        current = {
            "pipeline_v1": 0.25, "risk_steward_v1": 0.25,
            "data_guardian_v1": 0.25, "execution_cost_v1": 0.25,
        }
        proposal = propose_weights(
            conn, current, min_trades=5, days=36500,
        )

        assert isinstance(proposal.proposed_weights, dict)
        assert len(proposal.proposed_weights) == 4
        total = sum(proposal.proposed_weights.values())
        assert abs(total - 1.0) < 0.01

        # Deltas should be bounded
        for d in proposal.deltas.values():
            assert abs(d) <= 0.05 + 0.001
        conn.close()

    def test_propose_weights_bounded_shift(self):
        from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
        from hogan_bot.swarm_decision.weight_learner import propose_weights

        conn = _db()
        _seed_decisions_and_candles(conn, n=20)
        backfill_outcomes(conn, lookback_hours=0)

        current = {"pipeline_v1": 0.9, "risk_steward_v1": 0.03,
                    "data_guardian_v1": 0.04, "execution_cost_v1": 0.03}
        proposal = propose_weights(
            conn, current, min_trades=5, max_daily_shift=0.02, days=36500,
        )

        for k, d in proposal.deltas.items():
            assert abs(d) <= 0.10  # renormalization can amplify shifts slightly
        conn.close()

    def test_log_weight_proposal(self):
        from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
        from hogan_bot.swarm_decision.weight_learner import propose_weights, log_weight_proposal

        conn = _db()
        _seed_decisions_and_candles(conn, n=20)
        backfill_outcomes(conn, lookback_hours=0)

        current = {"pipeline_v1": 0.25, "risk_steward_v1": 0.25,
                    "data_guardian_v1": 0.25, "execution_cost_v1": 0.25}
        proposal = propose_weights(conn, current, min_trades=5, days=36500)
        row_id = log_weight_proposal(conn, "BTC/USD", "1h", proposal)
        assert row_id > 0

        snap = conn.execute(
            "SELECT source FROM swarm_weight_snapshots WHERE id = ?",
            (row_id,),
        ).fetchone()
        assert snap[0] == "shadow_update"
        conn.close()

    def test_compute_agent_accuracy(self):
        from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
        from hogan_bot.swarm_decision.weight_learner import compute_agent_accuracy

        conn = _db()
        _seed_decisions_and_candles(conn, n=20)
        backfill_outcomes(conn, lookback_hours=0)

        acc = compute_agent_accuracy(conn, days=36500)
        assert not acc.empty
        assert "accuracy" in acc.columns
        assert "agent_id" in acc.columns
        conn.close()


# ===================================================================
# Configurable weights tests
# ===================================================================

class TestConfigurableWeights:
    def test_parse_weight_string(self):
        from hogan_bot.policy_core import _parse_weight_string
        result = _parse_weight_string("a:0.4,b:0.3,c:0.3")
        assert result == {"a": 0.4, "b": 0.3, "c": 0.3}

    def test_parse_weight_string_empty(self):
        from hogan_bot.policy_core import _parse_weight_string
        assert _parse_weight_string("") == {}

    def test_parse_weight_string_bad(self):
        from hogan_bot.policy_core import _parse_weight_string
        assert _parse_weight_string("not_valid") == {}

    def test_resolve_swarm_weights_from_config(self):
        from hogan_bot.policy_core import _resolve_swarm_weights

        class MockConfig:
            swarm_weights = "a:0.6,b:0.4"
            swarm_use_regime_weights = False

        result = _resolve_swarm_weights(MockConfig(), None, "BTC/USD", "trending")
        assert result == {"a": 0.6, "b": 0.4}

    def test_resolve_swarm_weights_empty_config(self):
        from hogan_bot.policy_core import _resolve_swarm_weights

        class MockConfig:
            swarm_weights = ""
            swarm_use_regime_weights = False

        result = _resolve_swarm_weights(MockConfig(), None, "BTC/USD", None)
        assert result is None  # let controller default to uniform

    def test_resolve_swarm_weights_regime_lookup(self):
        from hogan_bot.policy_core import _resolve_swarm_weights

        conn = _db()
        conn.execute(
            """INSERT INTO swarm_weight_snapshots
               (ts_ms, symbol, timeframe, regime, weights_json, source, notes)
               VALUES (?,?,?,?,?,?,?)""",
            (int(time.time() * 1000), "BTC/USD", "1h", "trending",
             json.dumps({"a": 0.7, "b": 0.3}), "promoted", "test"),
        )
        conn.commit()

        class MockConfig:
            swarm_weights = ""
            swarm_use_regime_weights = True

        result = _resolve_swarm_weights(MockConfig(), conn, "BTC/USD", "trending")
        assert result == {"a": 0.7, "b": 0.3}
        conn.close()


# ===================================================================
# Schema / FK link tests
# ===================================================================

class TestSchemaEnhancements:
    def test_decision_log_has_swarm_decision_id(self):
        conn = _db()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(decision_log)").fetchall()]
        assert "swarm_decision_id" in cols
        conn.close()

    def test_swarm_outcomes_table_exists(self):
        conn = _db()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "swarm_outcomes" in tables
        conn.close()

    def test_log_decision_with_swarm_id(self):
        from hogan_bot.storage import log_decision
        conn = _db()

        # Insert a swarm decision first
        cur = conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms, symbol, timeframe, mode, final_action, final_conf,
                final_scale, agreement, entropy, vetoed,
                block_reasons_json, weights_json, decision_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (1700000000000, "BTC/USD", "1h", "shadow", "buy", 0.7,
             1.0, 0.8, 0.2, 0, "[]", "{}", "{}"),
        )
        conn.commit()
        sw_id = cur.lastrowid

        dec_id = log_decision(
            conn, ts_ms=1700000000000, symbol="BTC/USD",
            final_action="buy", swarm_decision_id=sw_id,
        )

        row = conn.execute(
            "SELECT swarm_decision_id FROM decision_log WHERE id = ?",
            (dec_id,),
        ).fetchone()
        assert row[0] == sw_id
        conn.close()

    def test_config_swarm_weights_field(self):
        from hogan_bot.config import BotConfig
        cfg = BotConfig()
        assert hasattr(cfg, "swarm_weights")
        assert isinstance(cfg.swarm_weights, str)
        assert hasattr(cfg, "swarm_use_regime_weights")
        assert cfg.swarm_use_regime_weights is False
