"""Tests for swarm attribution, replay queries, and similar events."""
from __future__ import annotations

import json
import sqlite3

from hogan_bot.storage import _create_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    _create_schema(conn)
    return conn


def _seed_full(conn: sqlite3.Connection, n: int = 20) -> list[int]:
    """Insert decisions, votes, outcomes, baseline entries, and candles."""
    ids = []
    base_ts = 1700000000000
    for i in range(n):
        ts_ms = base_ts + i * 3600_000
        action = "buy" if i % 3 == 0 else ("sell" if i % 3 == 1 else "hold")
        vetoed = 1 if i % 7 == 0 else 0
        if vetoed:
            action = "hold"
        regime = ["trending", "ranging", "volatile", "risk_off"][i % 4]

        cur = conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms, symbol, timeframe, as_of_ms, mode, final_action,
                final_conf, final_scale, agreement, entropy, vetoed,
                block_reasons_json, weights_json, decision_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts_ms, "BTC/USD", "1h", ts_ms, "shadow", action,
             0.55 + i * 0.02, 0.8 + (i % 3) * 0.1, 0.65 + (i % 10) * 0.03,
             0.2 + (i % 5) * 0.1, vetoed,
             json.dumps(["stale_data"] if vetoed else []),
             json.dumps({"p": 0.25, "r": 0.25, "d": 0.25, "e": 0.25}),
             json.dumps({"regime": regime, "atr_pct": 0.02})),
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

        # Baseline
        bl_action = "buy" if i % 2 == 0 else "sell"
        conn.execute(
            """INSERT INTO decision_log
               (ts_ms, symbol, final_action, final_confidence, position_size)
               VALUES (?,?,?,?,?)""",
            (ts_ms, "BTC/USD", bl_action, 0.6, 100.0),
        )

        # Outcome (some positive, some negative)
        fwd = 30.0 if i % 2 == 0 else -25.0
        was_veto_correct = None
        if vetoed:
            was_veto_correct = 1 if fwd < 0 else 0
        conn.execute(
            """INSERT INTO swarm_outcomes
               (decision_id, forward_5m_bps, forward_15m_bps, forward_30m_bps,
                forward_60m_bps, mae_bps, mfe_bps,
                was_trade_taken, baseline_would_trade, swarm_would_trade,
                was_veto_correct, was_skip_correct, outcome_label, updated_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (dec_id, fwd * 0.2, fwd * 0.5, fwd * 0.8, fwd,
             abs(fwd) * 0.3, abs(fwd) * 1.2,
             1 if action in ("buy", "sell") and not vetoed else 0,
             1, 1 if action in ("buy", "sell") else 0,
             was_veto_correct, None,
             "win" if fwd > 0 else "loss",
             ts_ms + 3600_000),
        )

    # Candles
    for i in range(n + 70):
        ts_ms = base_ts + i * 3600_000
        close = 40000 + i * 10
        conn.execute(
            """INSERT OR IGNORE INTO candles (ts_ms, symbol, timeframe, open, high, low, close, volume)
               VALUES (?,?,?,?,?,?,?,?)""",
            (ts_ms, "BTC/USD", "1h", close - 5, close + 10, close - 15, close, 1000.0),
        )

    conn.commit()
    return ids


# ===================================================================
# Attribution tests
# ===================================================================

class TestClassifyOutcome:
    def test_winner(self):
        from hogan_bot.swarm_attribution import classify_outcome
        dec = {"final_action": "buy", "vetoed": False}
        out = {"forward_60m_bps": 25.0}
        assert classify_outcome(dec, out) == "Winner"

    def test_loser(self):
        from hogan_bot.swarm_attribution import classify_outcome
        dec = {"final_action": "buy", "vetoed": False}
        out = {"forward_60m_bps": -30.0}
        assert classify_outcome(dec, out) == "Loser"

    def test_sell_winner(self):
        from hogan_bot.swarm_attribution import classify_outcome
        dec = {"final_action": "sell", "vetoed": False}
        out = {"forward_60m_bps": -25.0}
        assert classify_outcome(dec, out) == "Winner"

    def test_saved_by_veto(self):
        from hogan_bot.swarm_attribution import classify_outcome
        dec = {"final_action": "hold", "vetoed": True}
        out = {"forward_60m_bps": -30.0}
        bl = {"final_action": "buy"}
        assert classify_outcome(dec, out, bl) == "Saved by Veto"

    def test_false_veto(self):
        from hogan_bot.swarm_attribution import classify_outcome
        dec = {"final_action": "hold", "vetoed": True}
        out = {"forward_60m_bps": 50.0}
        bl = {"final_action": "buy"}
        assert classify_outcome(dec, out, bl) == "False Veto"

    def test_correct_skip(self):
        from hogan_bot.swarm_attribution import classify_outcome
        dec = {"final_action": "hold", "vetoed": False}
        out = {"forward_60m_bps": -20.0}
        bl = {"final_action": "buy"}
        assert classify_outcome(dec, out, bl) == "Correct Skip"

    def test_missed_winner(self):
        from hogan_bot.swarm_attribution import classify_outcome
        dec = {"final_action": "hold", "vetoed": False}
        out = {"forward_60m_bps": 50.0}
        bl = {"final_action": "buy"}
        assert classify_outcome(dec, out, bl) == "Missed Winner"

    def test_pending_no_outcome(self):
        from hogan_bot.swarm_attribution import classify_outcome
        dec = {"final_action": "buy", "vetoed": False}
        out = {}
        assert classify_outcome(dec, out) == "Pending"

    def test_scratch(self):
        from hogan_bot.swarm_attribution import classify_outcome
        dec = {"final_action": "buy", "vetoed": False}
        out = {"forward_60m_bps": 3.0}
        assert classify_outcome(dec, out) == "Scratch"


class TestAttributionScores:
    def test_direction_positive(self):
        from hogan_bot.swarm_attribution import compute_direction_attribution
        assert compute_direction_attribution(
            {"final_action": "buy"}, {"forward_60m_bps": 25}
        ) == 1.0

    def test_direction_negative(self):
        from hogan_bot.swarm_attribution import compute_direction_attribution
        assert compute_direction_attribution(
            {"final_action": "buy"}, {"forward_60m_bps": -25}
        ) == -1.0

    def test_direction_hold(self):
        from hogan_bot.swarm_attribution import compute_direction_attribution
        assert compute_direction_attribution(
            {"final_action": "hold"}, {"forward_60m_bps": 25}
        ) == 0.0

    def test_veto_saved(self):
        from hogan_bot.swarm_attribution import compute_veto_attribution
        assert compute_veto_attribution(
            {"vetoed": True}, {"forward_60m_bps": -30, "was_veto_correct": 1}
        ) == 1.0

    def test_veto_false(self):
        from hogan_bot.swarm_attribution import compute_veto_attribution
        assert compute_veto_attribution(
            {"vetoed": True}, {"forward_60m_bps": 50, "was_veto_correct": 0},
            {"final_action": "buy"},
        ) == -1.0

    def test_entry_good_timing(self):
        from hogan_bot.swarm_attribution import compute_entry_attribution
        assert compute_entry_attribution(
            {}, {"mae_bps": 3, "mfe_bps": 30}
        ) > 0.5

    def test_entry_bad_timing(self):
        from hogan_bot.swarm_attribution import compute_entry_attribution
        assert compute_entry_attribution(
            {}, {"mae_bps": 50, "mfe_bps": 20}
        ) < -0.3

    def test_full_attribution(self):
        from hogan_bot.swarm_attribution import compute_full_attribution
        dec = {"final_action": "buy", "vetoed": False, "agreement": 0.8, "final_scale": 0.9}
        out = {"forward_60m_bps": 25, "mae_bps": 5, "mfe_bps": 30}
        attr = compute_full_attribution(dec, out)
        assert attr["outcome_label"] == "Winner"
        assert attr["direction_attr"] == 1.0
        assert "posture_attr" in attr


class TestLearningNote:
    def test_winner_note(self):
        from hogan_bot.swarm_attribution import build_learning_note
        dec = {"final_action": "buy", "agreement": 0.8, "regime": "trending", "final_conf": 0.75}
        out = {"forward_60m_bps": 25}
        attr = {"outcome_label": "Winner", "direction_attr": 1.0, "entry_attr": 0.8,
                "veto_attr": 0, "posture_attr": 0.5, "cost_attr": 0, "disagreement_attr": 0}
        note = build_learning_note(dec, [], out, attr)
        assert "Winner" in note
        assert "buy" in note

    def test_veto_note(self):
        from hogan_bot.swarm_attribution import build_learning_note
        dec = {"final_action": "hold", "agreement": 0.6, "regime": "ranging", "vetoed": True}
        votes = [{"agent_id": "data_guardian_v1", "veto": True}]
        out = {"forward_60m_bps": -30}
        attr = {"outcome_label": "Saved by Veto", "direction_attr": 0, "entry_attr": 0,
                "veto_attr": 1.0, "posture_attr": 0, "cost_attr": 0, "disagreement_attr": 0}
        note = build_learning_note(dec, votes, out, attr)
        assert "Saved by Veto" in note
        assert "data_guardian_v1" in note


# ===================================================================
# Replay queries tests
# ===================================================================

class TestReplayQueries:
    def test_list_empty(self):
        from hogan_bot.swarm_replay_queries import ReplayFilter, list_replay_decisions
        conn = _db()
        results = list_replay_decisions(conn, ReplayFilter())
        assert results == []
        conn.close()

    def test_list_with_data(self):
        from hogan_bot.swarm_replay_queries import ReplayFilter, list_replay_decisions
        conn = _db()
        _seed_full(conn, n=10)
        results = list_replay_decisions(conn, ReplayFilter(symbol="BTC/USD"))
        assert len(results) > 0
        assert "swarm_action" in results[0]
        assert "outcome_label" in results[0]
        conn.close()

    def test_list_filter_vetoed(self):
        from hogan_bot.swarm_replay_queries import ReplayFilter, list_replay_decisions
        conn = _db()
        _seed_full(conn, n=20)
        results = list_replay_decisions(conn, ReplayFilter(source="vetoed"))
        for r in results:
            assert r["vetoed"] == 1
        conn.close()

    def test_list_filter_traded(self):
        from hogan_bot.swarm_replay_queries import ReplayFilter, list_replay_decisions
        conn = _db()
        _seed_full(conn, n=20)
        results = list_replay_decisions(conn, ReplayFilter(source="traded"))
        for r in results:
            assert r["swarm_action"] in ("buy", "sell")
            assert r["vetoed"] == 0
        conn.close()

    def test_list_sort_biggest_winner(self):
        from hogan_bot.swarm_replay_queries import ReplayFilter, list_replay_decisions
        conn = _db()
        _seed_full(conn, n=10)
        results = list_replay_decisions(conn, ReplayFilter(sort_by="biggest_winner"))
        if len(results) >= 2:
            fwd_first = results[0].get("forward_60m_bps") or -99999
            fwd_second = results[1].get("forward_60m_bps") or -99999
            assert fwd_first >= fwd_second
        conn.close()

    def test_get_replay_decision(self):
        from hogan_bot.swarm_replay_queries import get_replay_decision
        conn = _db()
        ids = _seed_full(conn, n=5)
        replay = get_replay_decision(conn, ids[0])
        assert replay is not None
        assert "decision" in replay
        assert "votes" in replay
        assert "outcome" in replay
        assert "candles" in replay
        assert "similar_events" in replay
        assert len(replay["votes"]) == 4
        conn.close()

    def test_get_replay_decision_missing(self):
        from hogan_bot.swarm_replay_queries import get_replay_decision
        conn = _db()
        assert get_replay_decision(conn, 9999) is None
        conn.close()

    def test_get_replay_candles(self):
        from hogan_bot.swarm_replay_queries import get_replay_candles
        conn = _db()
        _seed_full(conn, n=10)
        candles = get_replay_candles(conn, "BTC/USD", "1h", 1700000000000 + 5 * 3600_000)
        assert not candles.empty
        assert "close" in candles.columns
        conn.close()

    def test_get_replay_baseline(self):
        from hogan_bot.swarm_replay_queries import get_replay_baseline_compare
        conn = _db()
        ids = _seed_full(conn, n=5)
        bl = get_replay_baseline_compare(conn, ids[0])
        assert bl is not None
        assert "final_action" in bl
        conn.close()


# ===================================================================
# Similar events tests
# ===================================================================

class TestSimilarEvents:
    def test_similar_events_returns_list(self):
        from hogan_bot.swarm_replay_queries import get_replay_similar_events
        conn = _db()
        ids = _seed_full(conn, n=20)
        similar = get_replay_similar_events(conn, ids[5], limit=5)
        assert isinstance(similar, list)
        assert len(similar) <= 5
        conn.close()

    def test_similar_events_excludes_self(self):
        from hogan_bot.swarm_replay_queries import get_replay_similar_events
        conn = _db()
        ids = _seed_full(conn, n=20)
        similar = get_replay_similar_events(conn, ids[5])
        similar_ids = [s.get("id") for s in similar]
        assert ids[5] not in similar_ids
        conn.close()

    def test_similar_events_empty_db(self):
        from hogan_bot.swarm_replay_queries import get_replay_similar_events
        conn = _db()
        similar = get_replay_similar_events(conn, 1)
        assert similar == []
        conn.close()


# ===================================================================
# Schema test
# ===================================================================

class TestReplaySchema:
    def test_attribution_table_exists(self):
        conn = _db()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "swarm_attribution" in tables
        conn.close()

    def test_config_replay_fields(self):
        from hogan_bot.config import BotConfig
        cfg = BotConfig()
        assert cfg.swarm_replay_forward_window_bars == 12
        assert cfg.swarm_replay_bars_before == 60
        assert cfg.swarm_replay_positive_bps == 10.0
        assert cfg.swarm_replay_enable_similar_events is True
