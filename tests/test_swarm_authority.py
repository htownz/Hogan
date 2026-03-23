"""Tests for hogan_bot.swarm_authority — startup guards and drift detection."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from hogan_bot.storage import _create_schema
from hogan_bot.swarm_authority import (
    DriftReport,
    compute_shadow_active_drift,
    validate_swarm_config,
)


def _in_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    _create_schema(conn)
    return conn


def _cfg(**overrides) -> SimpleNamespace:
    defaults = {
        "swarm_mode": "conditional_active",
        "swarm_active_allow_new_signals": False,
        "swarm_use_regime_weights": False,
        "swarm_conditional_min_agreement": 0.70,
        "swarm_conditional_min_confidence": 0.60,
        "_backtest": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestValidateSwarmConfig:
    def test_safe_defaults_no_warnings(self) -> None:
        w = validate_swarm_config(_cfg())
        assert w == []

    def test_allow_new_signals_live_error(self) -> None:
        w = validate_swarm_config(_cfg(
            swarm_active_allow_new_signals=True,
            swarm_mode="active",
        ))
        assert any(x.code == "allow_new_signals_live" for x in w)
        assert any(x.severity == "error" for x in w)

    def test_allow_new_signals_backtest_ok(self) -> None:
        w = validate_swarm_config(_cfg(
            swarm_active_allow_new_signals=True,
            swarm_mode="active",
            _backtest=True,
        ))
        assert not any(x.code == "allow_new_signals_live" for x in w)

    def test_regime_weights_no_evidence(self) -> None:
        conn = _in_memory_db()
        w = validate_swarm_config(
            _cfg(swarm_use_regime_weights=True),
            conn=conn,
        )
        assert any(x.code == "regime_weights_no_evidence" for x in w)
        conn.close()

    def test_active_without_shadow_evidence(self) -> None:
        conn = _in_memory_db()
        w = validate_swarm_config(
            _cfg(swarm_mode="active"),
            conn=conn,
        )
        assert any(x.code == "active_without_shadow_evidence" for x in w)
        conn.close()

    def test_loose_conditional_thresholds(self) -> None:
        w = validate_swarm_config(_cfg(
            swarm_conditional_min_agreement=0.40,
            swarm_conditional_min_confidence=0.30,
        ))
        assert any(x.code == "loose_conditional_thresholds" for x in w)


def _seed_decisions(conn: sqlite3.Connection, mode: str, n: int, vetoed_frac: float = 0.1) -> None:
    import json
    for i in range(n):
        ts_ms = 1700000000000 + i * 3600_000
        vetoed = 1 if i < int(n * vetoed_frac) else 0
        action = "hold" if vetoed else ("buy" if i % 2 == 0 else "sell")
        conn.execute(
            """INSERT INTO swarm_decisions
               (ts_ms, symbol, timeframe, as_of_ms, mode, final_action,
                final_conf, final_scale, agreement, entropy, vetoed,
                block_reasons_json, weights_json, decision_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts_ms, "BTC/USD", "1h", ts_ms, mode, action,
             0.65, 1.0, 0.75, 0.3, vetoed,
             json.dumps([]), json.dumps({}), json.dumps({})),
        )
    conn.commit()


class TestDriftDetection:
    def test_empty_db(self) -> None:
        conn = _in_memory_db()
        d = compute_shadow_active_drift(conn)
        assert d.drift_acceptable is True
        assert d.shadow_trade_count == 0
        conn.close()

    def test_low_drift_acceptable(self) -> None:
        conn = _in_memory_db()
        _seed_decisions(conn, "shadow", 100, vetoed_frac=0.10)
        _seed_decisions(conn, "active", 95, vetoed_frac=0.12)
        d = compute_shadow_active_drift(conn)
        assert d.drift_acceptable is True
        assert d.shadow_trade_count > 0
        assert d.active_trade_count > 0
        conn.close()

    def test_high_trade_drift_flagged(self) -> None:
        conn = _in_memory_db()
        _seed_decisions(conn, "shadow", 100, vetoed_frac=0.10)
        _seed_decisions(conn, "active", 20, vetoed_frac=0.50)
        d = compute_shadow_active_drift(conn)
        assert d.drift_acceptable is False
        assert any("Trade count drift" in w for w in d.warnings)
        conn.close()

    def test_to_dict(self) -> None:
        d = DriftReport()
        out = d.to_dict()
        assert "shadow_trade_count" in out
        assert "drift_acceptable" in out
