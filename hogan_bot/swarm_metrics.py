"""Computed metrics for the Hogan swarm decision layer.

Pure functions that take DataFrames (from swarm_observability queries)
and return structured metric dicts.  No DB access — the caller provides
the data.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def compute_veto_precision(
    decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> dict[str, Any]:
    """How often were vetoes correct (vetoed trades would have lost)?

    Returns:
        dict with total_vetoes, correct_vetoes, precision, avg_pnl_vetoed,
        avg_pnl_allowed.
    """
    if decisions.empty:
        return {"total_vetoes": 0, "correct_vetoes": 0, "precision": 0.0,
                "avg_pnl_vetoed": 0.0, "avg_pnl_allowed": 0.0}

    merged = decisions.merge(outcomes, left_on="id", right_on="decision_id", how="left")
    vetoed = merged[merged["vetoed"] == 1]
    allowed = merged[merged["vetoed"] == 0]

    total = len(vetoed)
    correct = int((vetoed.get("was_veto_correct", pd.Series(dtype="int")) == 1).sum())
    precision = correct / total if total > 0 else 0.0

    fwd_col = "forward_60m_bps"
    avg_vetoed = float(vetoed[fwd_col].mean()) if not vetoed.empty and fwd_col in vetoed.columns and vetoed[fwd_col].notna().any() else 0.0
    avg_allowed = float(allowed[fwd_col].mean()) if not allowed.empty and fwd_col in allowed.columns and allowed[fwd_col].notna().any() else 0.0

    return {
        "total_vetoes": total,
        "correct_vetoes": correct,
        "precision": round(precision, 4),
        "avg_pnl_vetoed": round(avg_vetoed, 2),
        "avg_pnl_allowed": round(avg_allowed, 2),
    }


def compute_no_trade_rate(
    decisions: pd.DataFrame,
    baseline_decisions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """What fraction of opportunities does the swarm skip?

    Returns:
        dict with total_decisions, would_trade, would_hold, no_trade_rate,
        baseline_would_trade (if baseline provided).
    """
    if decisions.empty:
        return {"total_decisions": 0, "would_trade": 0, "would_hold": 0,
                "no_trade_rate": 0.0}

    would_trade = len(decisions[decisions["final_action"].isin(["buy", "sell"])])
    total = len(decisions)
    no_trade_rate = 1.0 - (would_trade / total) if total > 0 else 0.0

    result: dict[str, Any] = {
        "total_decisions": total,
        "would_trade": would_trade,
        "would_hold": total - would_trade,
        "no_trade_rate": round(no_trade_rate, 4),
    }

    if baseline_decisions is not None and not baseline_decisions.empty:
        bl_trades = len(baseline_decisions[
            baseline_decisions["final_action"].isin(["buy", "sell"])
        ])
        result["baseline_would_trade"] = bl_trades
        skipped = bl_trades - would_trade if bl_trades > would_trade else 0
        result["skipped_vs_baseline"] = skipped
        result["skip_rate_vs_baseline"] = round(
            skipped / bl_trades, 4
        ) if bl_trades > 0 else 0.0

    return result


def compute_trade_density(
    decisions: pd.DataFrame,
    bucket_hours: int = 24,
) -> pd.DataFrame:
    """Trade count per time bucket.

    Returns a DataFrame with columns: bucket_start, trades, holds, total.
    """
    if decisions.empty:
        return pd.DataFrame(columns=["bucket_start", "trades", "holds", "total"])

    df = decisions.copy()
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["bucket_start"] = df["ts"].dt.floor(f"{bucket_hours}h")
    df["is_trade"] = df["final_action"].isin(["buy", "sell"]).astype(int)

    grouped = df.groupby("bucket_start").agg(
        trades=("is_trade", "sum"),
        total=("is_trade", "count"),
    ).reset_index()
    grouped["holds"] = grouped["total"] - grouped["trades"]
    return grouped


def compute_agent_leaderboard(
    votes: pd.DataFrame,
    outcomes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-agent summary: vote count, veto count, mean confidence, action dist.

    If outcomes are provided, also computes hit rate (how often the agent's
    vote direction matched the forward return direction).
    """
    if votes.empty:
        return pd.DataFrame()

    agg = votes.groupby("agent_id").agg(
        vote_count=("action", "count"),
        veto_count=("veto", "sum"),
        mean_confidence=("confidence", "mean"),
        buys=("action", lambda s: (s == "buy").sum()),
        sells=("action", lambda s: (s == "sell").sum()),
        holds=("action", lambda s: (s == "hold").sum()),
    ).reset_index()

    agg["veto_rate"] = (agg["veto_count"] / agg["vote_count"]).round(4)
    agg["mean_confidence"] = agg["mean_confidence"].round(4)

    if (outcomes is not None and not outcomes.empty
            and "decision_id" in votes.columns
            and "forward_60m_bps" in outcomes.columns):
        merged = votes.merge(
            outcomes[["decision_id", "forward_60m_bps"]],
            on="decision_id", how="left",
        )
        merged["direction_correct"] = (
            ((merged["action"] == "buy") & (merged["forward_60m_bps"] > 0)) |
            ((merged["action"] == "sell") & (merged["forward_60m_bps"] < 0))
        )
        hit = merged.groupby("agent_id")["direction_correct"].mean().reset_index()
        hit.columns = ["agent_id", "hit_rate"]
        agg = agg.merge(hit, on="agent_id", how="left")
        agg["hit_rate"] = agg["hit_rate"].round(4)

    return agg.sort_values("veto_count", ascending=False)


def compute_opportunity_monotonicity(
    calibration_df: pd.DataFrame,
    score_col: str = "final_conf",
    return_col: str = "forward_60m_bps",
    n_bins: int = 5,
) -> dict[str, Any]:
    """Check if higher opportunity scores lead to better forward returns.

    Returns dict with bins (list of dicts), monotonic (bool), and
    correlation.
    """
    if calibration_df.empty or return_col not in calibration_df.columns:
        return {"bins": [], "monotonic": False, "correlation": 0.0}

    df = calibration_df.dropna(subset=[score_col, return_col]).copy()
    if len(df) < n_bins:
        return {"bins": [], "monotonic": False, "correlation": 0.0}

    df["bin"] = pd.qcut(df[score_col], n_bins, labels=False, duplicates="drop")
    bins = df.groupby("bin").agg(
        mean_score=(score_col, "mean"),
        mean_return=(return_col, "mean"),
        count=(return_col, "count"),
    ).reset_index()

    returns_series = bins["mean_return"].tolist()
    monotonic = all(returns_series[i] <= returns_series[i + 1]
                    for i in range(len(returns_series) - 1))

    corr = float(df[score_col].corr(df[return_col]))

    return {
        "bins": bins.round(4).to_dict("records"),
        "monotonic": monotonic,
        "correlation": round(corr, 4) if pd.notna(corr) else 0.0,
    }


def compute_disagreement_stats(
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    """Summary statistics on agreement/disagreement across decisions."""
    if decisions.empty or "agreement" not in decisions.columns:
        return {"mean_agreement": 0.0, "mean_entropy": 0.0,
                "high_disagreement_pct": 0.0, "count": 0}

    return {
        "mean_agreement": round(float(decisions["agreement"].mean()), 4),
        "mean_entropy": round(float(decisions["entropy"].mean()), 4),
        "high_disagreement_pct": round(
            float((decisions["agreement"] < 0.5).mean()), 4
        ),
        "count": len(decisions),
    }
