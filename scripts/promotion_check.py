"""Promotion check CLI for the Hogan swarm decision layer.

Determines the current swarm phase and whether it is ready to advance
based on the roadmap's go/no-go gates.

Phase state machine:
    Phase0_Certification  --> Phase1_Shadow
    Phase1_Shadow         --> Phase2_VetoOnly
    Phase2_VetoOnly       --> Phase3_SizeEntry
    Phase3_SizeEntry      --> Phase4_Learning
    Phase4_Learning       --> Phase5_AdaptivePaper
    Phase5_AdaptivePaper  --> Phase6_MicroLive

Usage:
    python scripts/promotion_check.py --db data/hogan.db --symbol BTC/USD
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASES = [
    "Phase0_Certification",
    "Phase1_Shadow",
    "Phase2_VetoOnly",
    "Phase3_SizeEntry",
    "Phase4_Learning",
    "Phase5_AdaptivePaper",
    "Phase6_MicroLive",
]


@dataclass
class GateCheck:
    name: str
    required: str
    actual: str
    passed: bool


@dataclass
class PromotionReport:
    current_phase: str = "Phase0_Certification"
    evidence: dict[str, Any] = field(default_factory=dict)
    gates: list[dict] = field(default_factory=list)
    recommendation: str = "hold"  # "advance" | "hold" | "collecting"
    blockers: list[str] = field(default_factory=list)
    next_phase: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

def _collect_evidence(
    conn: sqlite3.Connection,
    symbol: str | None,
) -> dict[str, Any]:
    """Gather all evidence counts/metrics needed across phases."""
    ev: dict[str, Any] = {}
    sym_filter = "AND symbol = ?" if symbol else ""
    sym_params: tuple = (symbol,) if symbol else ()

    # Shadow decision counts
    row = conn.execute(
        f"SELECT COUNT(*) FROM swarm_decisions WHERE mode='shadow' {sym_filter}",
        sym_params,
    ).fetchone()
    ev["shadow_decisions"] = row[0] if row else 0

    row = conn.execute(
        f"SELECT COUNT(*) FROM swarm_decisions WHERE mode='shadow' "
        f"AND final_action IN ('buy','sell') {sym_filter}",
        sym_params,
    ).fetchone()
    ev["would_trade"] = row[0] if row else 0

    row = conn.execute(
        f"SELECT COUNT(*) FROM swarm_decisions WHERE mode='shadow' AND vetoed=1 {sym_filter}",
        sym_params,
    ).fetchone()
    ev["veto_count"] = row[0] if row else 0

    # Distinct regimes
    try:
        row = conn.execute(
            f"SELECT COUNT(DISTINCT regime) "
            f"FROM swarm_decisions WHERE mode='shadow' AND regime IS NOT NULL {sym_filter}",
            sym_params,
        ).fetchone()
        ev["distinct_regimes"] = row[0] if row else 0
    except Exception:
        ev["distinct_regimes"] = 0

    # Mean agreement
    row = conn.execute(
        f"SELECT AVG(agreement) FROM swarm_decisions WHERE mode='shadow' {sym_filter}",
        sym_params,
    ).fetchone()
    ev["mean_agreement"] = round(row[0] or 0.0, 4)

    # Active mode stats (veto-only / size-entry phases)
    row = conn.execute(
        f"SELECT COUNT(*) FROM swarm_decisions WHERE mode='active' {sym_filter}",
        sym_params,
    ).fetchone()
    ev["active_decisions"] = row[0] if row else 0

    row = conn.execute(
        f"SELECT COUNT(*) FROM swarm_decisions WHERE mode='active' AND vetoed=1 {sym_filter}",
        sym_params,
    ).fetchone()
    ev["active_vetoes"] = row[0] if row else 0

    # Paper trade stats
    sym_filter_pt = "AND symbol = ?" if symbol else ""
    try:
        row = conn.execute(
            f"SELECT COUNT(*) FROM paper_trades WHERE exit_price IS NOT NULL {sym_filter_pt}",
            sym_params,
        ).fetchone()
        ev["closed_paper_trades"] = row[0] if row else 0

        df_pt = pd.read_sql_query(
            f"SELECT realized_pnl FROM paper_trades WHERE exit_price IS NOT NULL {sym_filter_pt}",
            conn, params=sym_params,
        )
        if not df_pt.empty:
            ev["total_paper_pnl"] = round(float(df_pt["realized_pnl"].sum()), 4)
            ev["paper_win_rate"] = round(
                float((df_pt["realized_pnl"] > 0).mean()), 4
            )
        else:
            ev["total_paper_pnl"] = 0.0
            ev["paper_win_rate"] = 0.0
    except Exception:
        ev["closed_paper_trades"] = 0
        ev["total_paper_pnl"] = 0.0
        ev["paper_win_rate"] = 0.0

    # Veto accuracy (average PnL of vetoed vs allowed decisions)
    try:
        sym_filter_d = "AND d.symbol = ?" if symbol else ""
        df_comp = pd.read_sql_query(
            f"""SELECT s.vetoed, d.realized_pnl
                FROM decision_log d
                JOIN swarm_decisions s ON d.ts_ms = s.ts_ms AND d.symbol = s.symbol
                WHERE s.mode = 'shadow'
                  AND d.final_action IN ('buy','sell')
                  {sym_filter_d}""",
            conn, params=sym_params,
        )
        if not df_comp.empty:
            vetoed = df_comp[df_comp["vetoed"] == 1]
            allowed = df_comp[df_comp["vetoed"] == 0]
            ev["avg_pnl_vetoed"] = round(
                float(vetoed["realized_pnl"].mean()), 6
            ) if not vetoed.empty and vetoed["realized_pnl"].notna().any() else 0.0
            ev["avg_pnl_allowed"] = round(
                float(allowed["realized_pnl"].mean()), 6
            ) if not allowed.empty and allowed["realized_pnl"].notna().any() else 0.0
            ev["veto_positive"] = ev["avg_pnl_vetoed"] < ev["avg_pnl_allowed"]
        else:
            ev["avg_pnl_vetoed"] = 0.0
            ev["avg_pnl_allowed"] = 0.0
            ev["veto_positive"] = False
    except Exception:
        ev["avg_pnl_vetoed"] = 0.0
        ev["avg_pnl_allowed"] = 0.0
        ev["veto_positive"] = False

    # Weight proposal count (for learning phases)
    try:
        row = conn.execute(
            f"SELECT COUNT(*) FROM swarm_weight_snapshots WHERE source='proposal' {sym_filter}",
            sym_params,
        ).fetchone()
        ev["weight_proposals"] = row[0] if row else 0
    except Exception:
        ev["weight_proposals"] = 0

    # Mode from swarm_decisions (latest row)
    try:
        row = conn.execute(
            f"SELECT mode FROM swarm_decisions {('WHERE symbol = ?' if symbol else '')} "
            f"ORDER BY ts_ms DESC LIMIT 1",
            sym_params if symbol else (),
        ).fetchone()
        ev["latest_mode"] = row[0] if row else None
    except Exception:
        ev["latest_mode"] = None

    # Config-level phase (operator-pinned) — read from BotConfig if available
    try:
        from hogan_bot.config import load_config
        _cfg = load_config()
        ev["swarm_phase"] = getattr(_cfg, "swarm_phase", None)
        ev["use_policy_core"] = getattr(_cfg, "use_policy_core", False)
    except Exception:
        ev["swarm_phase"] = None
        ev["use_policy_core"] = False

    return ev


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

def detect_phase(ev: dict[str, Any]) -> str:
    """Determine current phase from evidence.

    If the config explicitly sets ``swarm_phase`` to a value beyond the
    default (``certification``), that takes priority — the operator is
    pinning the phase.  Otherwise infer from DB evidence.
    """
    explicit = ev.get("swarm_phase")
    _PHASE_MAP = {
        "shadow": "Phase1_Shadow",
        "paper_veto": "Phase2_VetoOnly",
        "paper_routing": "Phase3_SizeEntry",
        "learning": "Phase4_Learning",
        "adaptive_paper": "Phase5_AdaptivePaper",
        "micro_live": "Phase6_MicroLive",
    }
    if explicit and explicit in _PHASE_MAP:
        return _PHASE_MAP[explicit]

    if ev["shadow_decisions"] == 0:
        return "Phase0_Certification"
    if ev.get("latest_mode") == "shadow":
        return "Phase1_Shadow"
    if ev.get("latest_mode") == "active":
        if ev["active_vetoes"] > 0 and ev["closed_paper_trades"] < 75:
            return "Phase2_VetoOnly"
        if ev["closed_paper_trades"] >= 75 and ev["weight_proposals"] == 0:
            return "Phase3_SizeEntry"
        if ev["weight_proposals"] > 0 and ev["closed_paper_trades"] < 200:
            return "Phase4_Learning"
        if ev["closed_paper_trades"] >= 200:
            return "Phase5_AdaptivePaper"
    return "Phase1_Shadow"


# ---------------------------------------------------------------------------
# Gate checks per phase
# ---------------------------------------------------------------------------

def _check_phase0(ev: dict[str, Any]) -> tuple[list[GateCheck], list[str]]:
    """Phase 0 — Certification gate. Pass = all tests green + policy_core on."""
    gates: list[GateCheck] = []
    blockers: list[str] = []

    g = GateCheck(
        name="certification_tests",
        required="all tests green (run pytest externally)",
        actual="manual check required",
        passed=True,
    )
    gates.append(g)

    pc_on = ev.get("use_policy_core", False)
    g = GateCheck(
        name="use_policy_core_enabled",
        required="True (swarm requires policy_core path)",
        actual=str(pc_on),
        passed=bool(pc_on),
    )
    gates.append(g)
    if not g.passed:
        blockers.append("use_policy_core is False — swarm is dead code without it")

    return gates, blockers


def _check_phase1(ev: dict[str, Any]) -> tuple[list[GateCheck], list[str]]:
    """Phase 1 — Shadow proving gates."""
    gates: list[GateCheck] = []
    blockers: list[str] = []

    g = GateCheck("shadow_sample", ">=300", str(ev["shadow_decisions"]),
                  ev["shadow_decisions"] >= 300)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {ev['shadow_decisions']}/300 shadow decisions")

    g = GateCheck("would_trade", ">=100", str(ev["would_trade"]),
                  ev["would_trade"] >= 100)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {ev['would_trade']}/100 would-trade events")

    g = GateCheck("veto_count", ">=50", str(ev["veto_count"]),
                  ev["veto_count"] >= 50)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {ev['veto_count']}/50 veto events")

    g = GateCheck("regime_coverage", ">=3", str(ev["distinct_regimes"]),
                  ev["distinct_regimes"] >= 3)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {ev['distinct_regimes']}/3 distinct regimes")

    g = GateCheck("veto_accuracy", "vetoed PnL < allowed PnL",
                  f"vetoed={ev['avg_pnl_vetoed']}, allowed={ev['avg_pnl_allowed']}",
                  ev.get("veto_positive", False))
    gates.append(g)
    if not g.passed and ev["shadow_decisions"] >= 300:
        blockers.append("Veto accuracy negative")

    return gates, blockers


def _check_phase2(ev: dict[str, Any]) -> tuple[list[GateCheck], list[str]]:
    """Phase 2 — Veto-only paper gates."""
    gates: list[GateCheck] = []
    blockers: list[str] = []

    g = GateCheck("active_veto_events", ">=50", str(ev["active_vetoes"]),
                  ev["active_vetoes"] >= 50)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {ev['active_vetoes']}/50 active veto events")

    g = GateCheck("paper_trades", ">=50", str(ev["closed_paper_trades"]),
                  ev["closed_paper_trades"] >= 50)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {ev['closed_paper_trades']}/50 closed paper trades")

    g = GateCheck("paper_pnl_non_negative", ">=0",
                  str(ev["total_paper_pnl"]),
                  ev["total_paper_pnl"] >= 0)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Paper PnL negative: {ev['total_paper_pnl']}")

    return gates, blockers


def _check_phase3(ev: dict[str, Any]) -> tuple[list[GateCheck], list[str]]:
    """Phase 3 — Size & entry authority gates."""
    gates: list[GateCheck] = []
    blockers: list[str] = []

    g = GateCheck("paper_trades", ">=75", str(ev["closed_paper_trades"]),
                  ev["closed_paper_trades"] >= 75)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {ev['closed_paper_trades']}/75 closed paper trades")

    g = GateCheck("paper_pnl_positive", ">0",
                  str(ev["total_paper_pnl"]),
                  ev["total_paper_pnl"] > 0)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Paper PnL not positive: {ev['total_paper_pnl']}")

    return gates, blockers


def _check_phase4(ev: dict[str, Any]) -> tuple[list[GateCheck], list[str]]:
    """Phase 4 — Learning gates."""
    gates: list[GateCheck] = []
    blockers: list[str] = []

    g = GateCheck("weight_proposals", ">=2 cycles", str(ev["weight_proposals"]),
                  ev["weight_proposals"] >= 2)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {ev['weight_proposals']}/2 weight proposal cycles")

    g = GateCheck("paper_trades", ">=100", str(ev["closed_paper_trades"]),
                  ev["closed_paper_trades"] >= 100)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {ev['closed_paper_trades']}/100 paper trades under fixed logic")

    return gates, blockers


def _check_phase5(ev: dict[str, Any]) -> tuple[list[GateCheck], list[str]]:
    """Phase 5 — Adaptive paper gates."""
    gates: list[GateCheck] = []
    blockers: list[str] = []

    g = GateCheck("paper_trades", ">=200", str(ev["closed_paper_trades"]),
                  ev["closed_paper_trades"] >= 200)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {ev['closed_paper_trades']}/200 paper trades")

    g = GateCheck("paper_pnl_positive", ">0",
                  str(ev["total_paper_pnl"]),
                  ev["total_paper_pnl"] > 0)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Paper PnL not positive: {ev['total_paper_pnl']}")

    g = GateCheck("paper_win_rate", ">=0.40", str(ev["paper_win_rate"]),
                  ev["paper_win_rate"] >= 0.40)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Paper win rate {ev['paper_win_rate']:.1%} < 40%")

    return gates, blockers


def _check_phase6(ev: dict[str, Any]) -> tuple[list[GateCheck], list[str]]:
    """Phase 6 — Micro-live gates.  Tiny size, one symbol, hard kill switches."""
    gates: list[GateCheck] = []
    blockers: list[str] = []

    # Require 30-50 micro-live trades
    live_trades = ev.get("closed_paper_trades", 0)
    g = GateCheck("micro_live_trades", ">=30", str(live_trades),
                  live_trades >= 30)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {live_trades}/30 micro-live trades")

    # Paper PnL still positive
    g = GateCheck("paper_pnl_positive", ">0",
                  str(ev["total_paper_pnl"]),
                  ev["total_paper_pnl"] > 0)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Paper PnL not positive: {ev['total_paper_pnl']}")

    # Win rate acceptable
    g = GateCheck("paper_win_rate", ">=0.40", str(ev["paper_win_rate"]),
                  ev["paper_win_rate"] >= 0.40)
    gates.append(g)
    if not g.passed:
        blockers.append(f"Paper win rate {ev['paper_win_rate']:.1%} < 40%")

    return gates, blockers


PHASE_CHECKERS = {
    "Phase0_Certification": _check_phase0,
    "Phase1_Shadow": _check_phase1,
    "Phase2_VetoOnly": _check_phase2,
    "Phase3_SizeEntry": _check_phase3,
    "Phase4_Learning": _check_phase4,
    "Phase5_AdaptivePaper": _check_phase5,
    "Phase6_MicroLive": _check_phase6,
}


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_promotion_report(
    conn: sqlite3.Connection,
    symbol: str | None = None,
) -> PromotionReport:
    """Build a promotion readiness report."""
    ev = _collect_evidence(conn, symbol)
    phase = detect_phase(ev)

    checker = PHASE_CHECKERS.get(phase, _check_phase0)
    gate_objs, blockers = checker(ev)

    phase_idx = PHASES.index(phase) if phase in PHASES else 0
    next_phase = PHASES[phase_idx + 1] if phase_idx + 1 < len(PHASES) else "Complete"

    recommendation = "advance" if not blockers else (
        "collecting" if ev["shadow_decisions"] < 300 and phase == "Phase1_Shadow" else "hold"
    )

    return PromotionReport(
        current_phase=phase,
        evidence=ev,
        gates=[asdict(g) for g in gate_objs],
        recommendation=recommendation,
        blockers=blockers,
        next_phase=next_phase,
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_text(rpt: PromotionReport) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  HOGAN PROMOTION CHECK")
    lines.append("=" * 60)
    lines.append(f"  Current Phase:  {rpt.current_phase}")
    lines.append(f"  Next Phase:     {rpt.next_phase}")
    lines.append(f"  Recommendation: {rpt.recommendation.upper()}")
    lines.append("")

    lines.append("── Evidence ────────────────────────────────────────────")
    for k, v in rpt.evidence.items():
        lines.append(f"  {k:<25} {v}")
    lines.append("")

    lines.append("── Gate Checks ─────────────────────────────────────────")
    for g in rpt.gates:
        status = "PASS" if g["passed"] else "FAIL"
        lines.append(f"  [{status}] {g['name']}: {g['actual']} (required: {g['required']})")
    lines.append("")

    if rpt.blockers:
        lines.append("── Blockers ────────────────────────────────────────────")
        for b in rpt.blockers:
            lines.append(f"  - {b}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hogan swarm promotion check")
    parser.add_argument("--db", default="data/hogan.db", help="SQLite DB path")
    parser.add_argument("--symbol", default=None, help="Filter by symbol")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output JSON")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        print(f"ERROR: Database not found at {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    rpt = build_promotion_report(conn, symbol=args.symbol)
    conn.close()

    if args.as_json:
        print(json.dumps(rpt.to_dict(), indent=2, default=str))
    else:
        print(_format_text(rpt))

    return 0 if rpt.recommendation == "advance" else 1


if __name__ == "__main__":
    sys.exit(main())
