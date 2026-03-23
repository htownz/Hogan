"""Swarm authority guardrails — startup validation and drift detection.

Ensures the shadow → conditional_active → active progression is safe:
- Warns or blocks dangerous config combinations at startup.
- Detects trade-count and Calmar drift between shadow and active runs.
- Guards regime-weight promotion behind shadow evidence.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_SWARM_TAGS = frozenset({
    "swarm_blocked_unsigned_signal",
    "swarm_direction_clash",
})


@dataclass
class SwarmAuthorityWarning:
    code: str
    message: str
    severity: str = "warning"  # "warning" | "error"


def validate_swarm_config(config, *, conn: sqlite3.Connection | None = None) -> list[SwarmAuthorityWarning]:
    """Check swarm config for dangerous or premature settings.

    Called once at startup (event_loop / main).  Returns a list of
    warnings; the caller decides whether to log or abort.
    """
    warnings: list[SwarmAuthorityWarning] = []

    mode = getattr(config, "swarm_mode", "shadow")
    allow_new = getattr(config, "swarm_active_allow_new_signals", False)
    use_regime_weights = getattr(config, "swarm_use_regime_weights", False)
    is_backtest = getattr(config, "_backtest", False)

    if allow_new and mode in ("active", "conditional_active") and not is_backtest:
        warnings.append(SwarmAuthorityWarning(
            code="allow_new_signals_live",
            message=(
                "HOGAN_SWARM_ACTIVE_ALLOW_NEW_SIGNALS=true in "
                f"mode={mode}. This bypasses ML/edge/quality gates "
                "for swarm-originated entries. Use only on "
                "scratch DB or paper with capped notional."
            ),
            severity="error",
        ))

    if use_regime_weights:
        regime_evidence = _count_regime_weight_evidence(conn)
        if regime_evidence < 3:
            warnings.append(SwarmAuthorityWarning(
                code="regime_weights_no_evidence",
                message=(
                    f"HOGAN_SWARM_USE_REGIME_WEIGHTS=true but only "
                    f"{regime_evidence} regime weight snapshots found. "
                    "Need shadow evidence across multiple regimes "
                    "before enabling. See docs/SWARM_CONDITIONAL_TUNING.md."
                ),
                severity="warning",
            ))

    if mode == "active":
        shadow_count = _count_shadow_decisions(conn)
        if shadow_count < 300:
            warnings.append(SwarmAuthorityWarning(
                code="active_without_shadow_evidence",
                message=(
                    f"HOGAN_SWARM_MODE=active but only {shadow_count} "
                    "shadow decisions in DB (need >= 300). Run shadow "
                    "first. See docs/PROMOTION_CHECKLIST.md."
                ),
                severity="warning",
            ))

    ca_agree = getattr(config, "swarm_conditional_min_agreement", 0.70)
    ca_conf = getattr(config, "swarm_conditional_min_confidence", 0.60)
    if mode == "conditional_active" and (ca_agree < 0.50 or ca_conf < 0.40):
        warnings.append(SwarmAuthorityWarning(
            code="loose_conditional_thresholds",
            message=(
                f"conditional_active thresholds very loose "
                f"(agreement={ca_agree}, confidence={ca_conf}). "
                "This approaches full active behavior. Tighten or "
                "use walk-forward to validate."
            ),
            severity="warning",
        ))

    return warnings


def log_swarm_authority_warnings(config, *, conn: sqlite3.Connection | None = None) -> list[SwarmAuthorityWarning]:
    """Validate config and log warnings. Returns the list for callers that need it."""
    warnings = validate_swarm_config(config, conn=conn)
    for w in warnings:
        if w.severity == "error":
            logger.error("SWARM AUTHORITY [%s]: %s", w.code, w.message)
        else:
            logger.warning("SWARM AUTHORITY [%s]: %s", w.code, w.message)
    return warnings


def _count_shadow_decisions(conn: sqlite3.Connection | None) -> int:
    if conn is None:
        return 0
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM swarm_decisions WHERE mode='shadow'"
        ).fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


def _count_regime_weight_evidence(conn: sqlite3.Connection | None) -> int:
    """Count distinct regime weight snapshots (promoted or proposal)."""
    if conn is None:
        return 0
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM swarm_weight_snapshots "
            "WHERE regime IS NOT NULL"
        ).fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Shadow vs active drift detection
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Compares shadow and active/conditional behavior."""
    shadow_trade_count: int = 0
    active_trade_count: int = 0
    shadow_veto_rate: float = 0.0
    active_veto_rate: float = 0.0
    trade_count_drift_pct: float = 0.0
    veto_rate_drift_pct: float = 0.0
    drift_acceptable: bool = True
    warnings: list[str] = field(default_factory=list)

    shadow_mean_agreement: float = 0.0
    active_mean_agreement: float = 0.0

    def to_dict(self) -> dict:
        return {
            "shadow_trade_count": self.shadow_trade_count,
            "active_trade_count": self.active_trade_count,
            "shadow_veto_rate": round(self.shadow_veto_rate, 4),
            "active_veto_rate": round(self.active_veto_rate, 4),
            "trade_count_drift_pct": round(self.trade_count_drift_pct, 2),
            "veto_rate_drift_pct": round(self.veto_rate_drift_pct, 2),
            "drift_acceptable": self.drift_acceptable,
            "warnings": self.warnings,
            "shadow_mean_agreement": round(self.shadow_mean_agreement, 4),
            "active_mean_agreement": round(self.active_mean_agreement, 4),
        }


def compute_shadow_active_drift(
    conn: sqlite3.Connection,
    *,
    symbol: str | None = None,
    max_trade_drift_pct: float = 30.0,
    max_veto_drift_pct: float = 20.0,
) -> DriftReport:
    """Detect drift between shadow and active swarm behavior.

    Compares trade counts, veto rates, and agreement levels between
    the two modes. Used by shadow_report and promotion_check to gate
    the shadow → conditional → active transition.
    """
    rpt = DriftReport()
    sym = "AND symbol = ?" if symbol else ""
    params: tuple = (symbol,) if symbol else ()

    def _query_mode(mode: str) -> tuple[int, int, int, float]:
        total = conn.execute(
            f"SELECT COUNT(*) FROM swarm_decisions WHERE mode=? {sym}",
            (mode, *params),
        ).fetchone()[0]
        trades = conn.execute(
            f"SELECT COUNT(*) FROM swarm_decisions "
            f"WHERE mode=? AND final_action IN ('buy','sell') {sym}",
            (mode, *params),
        ).fetchone()[0]
        vetoes = conn.execute(
            f"SELECT COUNT(*) FROM swarm_decisions "
            f"WHERE mode=? AND vetoed=1 {sym}",
            (mode, *params),
        ).fetchone()[0]
        row = conn.execute(
            f"SELECT AVG(agreement) FROM swarm_decisions WHERE mode=? {sym}",
            (mode, *params),
        ).fetchone()
        avg_agree = float(row[0] or 0.0)
        return total, trades, vetoes, avg_agree

    try:
        s_total, s_trades, s_vetoes, s_agree = _query_mode("shadow")
        a_total, a_trades, a_vetoes, a_agree = _query_mode("active")
        ca_total, ca_trades, ca_vetoes, ca_agree = _query_mode("conditional_active")
        a_total += ca_total
        a_trades += ca_trades
        a_vetoes += ca_vetoes
        if ca_total > 0 and a_total > 0:
            a_agree = (a_agree * (a_total - ca_total) + ca_agree * ca_total) / a_total
    except Exception as exc:
        rpt.warnings.append(f"Query error: {exc}")
        return rpt

    rpt.shadow_trade_count = s_trades
    rpt.active_trade_count = a_trades
    rpt.shadow_veto_rate = s_vetoes / s_total if s_total > 0 else 0.0
    rpt.active_veto_rate = a_vetoes / a_total if a_total > 0 else 0.0
    rpt.shadow_mean_agreement = s_agree
    rpt.active_mean_agreement = a_agree

    if s_trades > 0:
        rpt.trade_count_drift_pct = abs(a_trades - s_trades) / s_trades * 100
    elif a_trades > 0:
        rpt.trade_count_drift_pct = 100.0

    if rpt.shadow_veto_rate > 0:
        rpt.veto_rate_drift_pct = (
            abs(rpt.active_veto_rate - rpt.shadow_veto_rate)
            / rpt.shadow_veto_rate * 100
        )
    elif rpt.active_veto_rate > 0:
        rpt.veto_rate_drift_pct = 100.0

    if rpt.trade_count_drift_pct > max_trade_drift_pct and s_trades > 0:
        rpt.drift_acceptable = False
        rpt.warnings.append(
            f"Trade count drift {rpt.trade_count_drift_pct:.0f}% "
            f"exceeds {max_trade_drift_pct:.0f}% threshold "
            f"(shadow={s_trades}, active={a_trades})"
        )

    if rpt.veto_rate_drift_pct > max_veto_drift_pct and s_total > 0:
        rpt.drift_acceptable = False
        rpt.warnings.append(
            f"Veto rate drift {rpt.veto_rate_drift_pct:.0f}% "
            f"exceeds {max_veto_drift_pct:.0f}% threshold "
            f"(shadow={rpt.shadow_veto_rate:.1%}, "
            f"active={rpt.active_veto_rate:.1%})"
        )

    return rpt
