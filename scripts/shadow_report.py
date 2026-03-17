"""Shadow evaluation report for the Hogan swarm decision layer.

Reads the SQLite DB and produces a structured shadow evaluation report
comparing baseline decisions against swarm decisions, measuring veto
accuracy, no-trade rate, agent leaderboard, and go/no-go summary.

Usage:
    python scripts/shadow_report.py --db data/hogan.db --symbol BTC/USD
    python scripts/shadow_report.py --db data/hogan.db --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    name: str
    required: str
    actual: str
    passed: bool


@dataclass
class ShadowReport:
    symbol: str = ""
    total_shadow_decisions: int = 0
    would_trade: int = 0
    veto_count: int = 0
    regime_breakdown: dict[str, int] = field(default_factory=dict)
    agreement_rate: float = 0.0
    baseline_match_count: int = 0
    baseline_mismatch_count: int = 0
    veto_saved_money: bool = False
    avg_pnl_vetoed: float = 0.0
    avg_pnl_allowed: float = 0.0
    no_trade_rate: float = 0.0
    baseline_trade_count: int = 0
    skipped_by_swarm: int = 0
    agent_leaderboard: list[dict] = field(default_factory=list)
    gates: list[dict] = field(default_factory=list)
    recommendation: str = "collecting"
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core report logic
# ---------------------------------------------------------------------------

def build_shadow_report(
    conn: sqlite3.Connection,
    symbol: str | None = None,
) -> ShadowReport:
    """Build a complete shadow evaluation report from the DB."""
    rpt = ShadowReport(symbol=symbol or "ALL")

    sym_filter = "AND s.symbol = ?" if symbol else ""
    sym_params: tuple = (symbol,) if symbol else ()

    # 1. Sample size ----------------------------------------------------------
    row = conn.execute(
        f"SELECT COUNT(*) FROM swarm_decisions s "
        f"WHERE s.mode = 'shadow' {sym_filter}",
        sym_params,
    ).fetchone()
    rpt.total_shadow_decisions = row[0] if row else 0

    row = conn.execute(
        f"SELECT COUNT(*) FROM swarm_decisions s "
        f"WHERE s.mode = 'shadow' AND s.final_action IN ('buy','sell') {sym_filter}",
        sym_params,
    ).fetchone()
    rpt.would_trade = row[0] if row else 0

    row = conn.execute(
        f"SELECT COUNT(*) FROM swarm_decisions s "
        f"WHERE s.mode = 'shadow' AND s.vetoed = 1 {sym_filter}",
        sym_params,
    ).fetchone()
    rpt.veto_count = row[0] if row else 0

    # Regime breakdown via json_extract on decision_json
    try:
        regimes = conn.execute(
            f"SELECT json_extract(s.decision_json, '$.regime') AS r, COUNT(*) "
            f"FROM swarm_decisions s WHERE s.mode = 'shadow' {sym_filter} "
            f"GROUP BY r",
            sym_params,
        ).fetchall()
        rpt.regime_breakdown = {str(r): c for r, c in regimes if r is not None}
    except Exception:
        pass

    # 2. Baseline vs swarm comparison -----------------------------------------
    sym_filter_d = "AND d.symbol = ?" if symbol else ""
    try:
        df_comp = pd.read_sql_query(
            f"""SELECT d.ts_ms, d.final_action AS baseline, s.final_action AS swarm,
                       s.agreement, s.vetoed, d.realized_pnl
                FROM decision_log d
                LEFT JOIN swarm_decisions s
                  ON d.ts_ms = s.ts_ms AND d.symbol = s.symbol
                WHERE s.mode = 'shadow'
                  AND d.final_action NOT IN ('hold', 'shadow_weight_update')
                  {sym_filter_d}""",
            conn,
            params=sym_params,
        )
        if not df_comp.empty:
            match = (df_comp["baseline"] == df_comp["swarm"]).sum()
            rpt.baseline_match_count = int(match)
            rpt.baseline_mismatch_count = len(df_comp) - int(match)
            rpt.agreement_rate = round(match / len(df_comp), 4) if len(df_comp) else 0.0
    except Exception:
        df_comp = pd.DataFrame()

    # 3. Veto accuracy --------------------------------------------------------
    try:
        if not df_comp.empty:
            vetoed = df_comp[df_comp["vetoed"] == 1]
            allowed = df_comp[df_comp["vetoed"] == 0]
            rpt.avg_pnl_vetoed = round(float(vetoed["realized_pnl"].mean()), 6) if not vetoed.empty and vetoed["realized_pnl"].notna().any() else 0.0
            rpt.avg_pnl_allowed = round(float(allowed["realized_pnl"].mean()), 6) if not allowed.empty and allowed["realized_pnl"].notna().any() else 0.0
            rpt.veto_saved_money = rpt.avg_pnl_vetoed < rpt.avg_pnl_allowed
    except Exception:
        pass

    # 4. No-trade rate --------------------------------------------------------
    try:
        sym_filter_bl = "AND d.symbol = ?" if symbol else ""
        row_bl = conn.execute(
            f"SELECT COUNT(*) FROM decision_log d "
            f"WHERE d.final_action IN ('buy','sell') {sym_filter_bl}",
            sym_params,
        ).fetchone()
        rpt.baseline_trade_count = row_bl[0] if row_bl else 0

        if not df_comp.empty and rpt.baseline_trade_count > 0:
            baseline_trades = df_comp[df_comp["baseline"].isin(["buy", "sell"])]
            swarm_skipped = baseline_trades[
                ~baseline_trades["swarm"].isin(["buy", "sell"])
            ]
            rpt.skipped_by_swarm = len(swarm_skipped)
            rpt.no_trade_rate = round(rpt.skipped_by_swarm / len(baseline_trades), 4) if len(baseline_trades) > 0 else 0.0
    except Exception:
        pass

    # 5. Agent leaderboard ----------------------------------------------------
    try:
        df_agents = pd.read_sql_query(
            f"""SELECT sav.agent_id,
                       COUNT(*) AS votes,
                       SUM(sav.veto) AS vetoes,
                       AVG(sav.confidence) AS mean_confidence,
                       SUM(CASE WHEN sav.action='buy' THEN 1 ELSE 0 END) AS buys,
                       SUM(CASE WHEN sav.action='sell' THEN 1 ELSE 0 END) AS sells,
                       SUM(CASE WHEN sav.action='hold' THEN 1 ELSE 0 END) AS holds
                FROM swarm_agent_votes sav
                JOIN swarm_decisions s ON sav.decision_id = s.id
                WHERE s.mode = 'shadow' {sym_filter}
                GROUP BY sav.agent_id
                ORDER BY vetoes DESC""",
            conn,
            params=sym_params,
        )
        if not df_agents.empty:
            rpt.agent_leaderboard = df_agents.round(4).to_dict("records")
    except Exception:
        pass

    # 6. Go/no-go gates -------------------------------------------------------
    rpt.gates, rpt.blockers = _evaluate_gates(rpt)
    rpt.recommendation = "advance" if not rpt.blockers else (
        "collecting" if rpt.total_shadow_decisions < 300 else "hold"
    )

    return rpt


def _evaluate_gates(rpt: ShadowReport) -> tuple[list[dict], list[str]]:
    """Evaluate Phase 1 shadow go/no-go gates from the roadmap."""
    gates: list[GateResult] = []
    blockers: list[str] = []

    # Gate 1: Sample size
    g = GateResult(
        name="shadow_sample_size",
        required=">=300 scored opportunities",
        actual=str(rpt.total_shadow_decisions),
        passed=rpt.total_shadow_decisions >= 300,
    )
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {rpt.total_shadow_decisions}/300 shadow decisions")

    # Gate 2: Would-trade count
    g = GateResult(
        name="would_trade_count",
        required=">=100 would-trade events",
        actual=str(rpt.would_trade),
        passed=rpt.would_trade >= 100,
    )
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {rpt.would_trade}/100 would-trade events")

    # Gate 3: Veto count
    g = GateResult(
        name="veto_count",
        required=">=50 veto events",
        actual=str(rpt.veto_count),
        passed=rpt.veto_count >= 50,
    )
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {rpt.veto_count}/50 veto events")

    # Gate 4: Regime coverage
    n_regimes = len(rpt.regime_breakdown)
    g = GateResult(
        name="regime_coverage",
        required=">=3 distinct regimes",
        actual=str(n_regimes),
        passed=n_regimes >= 3,
    )
    gates.append(g)
    if not g.passed:
        blockers.append(f"Only {n_regimes}/3 distinct regimes")

    # Gate 5: Veto accuracy (vetoed trades underperform allowed)
    g = GateResult(
        name="veto_accuracy",
        required="avg_pnl_vetoed < avg_pnl_allowed",
        actual=f"vetoed={rpt.avg_pnl_vetoed:.6f}, allowed={rpt.avg_pnl_allowed:.6f}",
        passed=rpt.veto_saved_money,
    )
    gates.append(g)
    if not g.passed and rpt.total_shadow_decisions >= 300:
        blockers.append("Vetoes not saving money (avg PnL of vetoed >= allowed)")

    # Gate 6: No-trade rate controlled (<= 30%)
    g = GateResult(
        name="no_trade_rate",
        required="<=30% of baseline trades skipped",
        actual=f"{rpt.no_trade_rate:.1%}",
        passed=rpt.no_trade_rate <= 0.30,
    )
    gates.append(g)
    if not g.passed:
        blockers.append(f"No-trade rate {rpt.no_trade_rate:.1%} exceeds 30% threshold")

    return [asdict(g) for g in gates], blockers


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_text(rpt: ShadowReport) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  HOGAN SHADOW EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"  Symbol: {rpt.symbol}")
    lines.append("")

    lines.append("── Sample Size ─────────────────────────────────────────")
    lines.append(f"  Shadow decisions:   {rpt.total_shadow_decisions}")
    lines.append(f"  Would-trade:        {rpt.would_trade}")
    lines.append(f"  Vetoes:             {rpt.veto_count}")
    if rpt.regime_breakdown:
        lines.append(f"  Regimes:            {rpt.regime_breakdown}")
    lines.append("")

    lines.append("── Baseline vs Swarm ───────────────────────────────────")
    lines.append(f"  Agreement rate:     {rpt.agreement_rate:.1%}")
    lines.append(f"  Matches:            {rpt.baseline_match_count}")
    lines.append(f"  Mismatches:         {rpt.baseline_mismatch_count}")
    lines.append("")

    lines.append("── Veto Accuracy ───────────────────────────────────────")
    lines.append(f"  Avg PnL (vetoed):   {rpt.avg_pnl_vetoed:.6f}")
    lines.append(f"  Avg PnL (allowed):  {rpt.avg_pnl_allowed:.6f}")
    lines.append(f"  Veto saved money:   {rpt.veto_saved_money}")
    lines.append("")

    lines.append("── No-Trade Rate ───────────────────────────────────────")
    lines.append(f"  Baseline trades:    {rpt.baseline_trade_count}")
    lines.append(f"  Skipped by swarm:   {rpt.skipped_by_swarm}")
    lines.append(f"  No-trade rate:      {rpt.no_trade_rate:.1%}")
    lines.append("")

    if rpt.agent_leaderboard:
        lines.append("── Agent Leaderboard ───────────────────────────────────")
        header = f"  {'Agent':<20} {'Votes':>6} {'Vetoes':>7} {'Conf':>6} {'Buys':>5} {'Sells':>6} {'Holds':>6}"
        lines.append(header)
        for a in rpt.agent_leaderboard:
            lines.append(
                f"  {a['agent_id']:<20} {a['votes']:>6} {a['vetoes']:>7} "
                f"{a['mean_confidence']:>6.3f} {a['buys']:>5} {a['sells']:>6} {a['holds']:>6}"
            )
        lines.append("")

    lines.append("── Go / No-Go Gates ────────────────────────────────────")
    for g in rpt.gates:
        status = "PASS" if g["passed"] else "FAIL"
        lines.append(f"  [{status}] {g['name']}: {g['actual']} (required: {g['required']})")
    lines.append("")

    lines.append("── Recommendation ──────────────────────────────────────")
    lines.append(f"  {rpt.recommendation.upper()}")
    if rpt.blockers:
        lines.append("  Blockers:")
        for b in rpt.blockers:
            lines.append(f"    - {b}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hogan shadow evaluation report")
    parser.add_argument("--db", default="data/hogan.db", help="SQLite DB path")
    parser.add_argument("--symbol", default=None, help="Filter by symbol")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output JSON")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        print(f"ERROR: Database not found at {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    rpt = build_shadow_report(conn, symbol=args.symbol)
    conn.close()

    if args.as_json:
        print(json.dumps(rpt.to_dict(), indent=2, default=str))
    else:
        print(_format_text(rpt))

    all_passed = all(g["passed"] for g in rpt.gates)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
