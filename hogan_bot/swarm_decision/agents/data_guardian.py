"""Data guardian agent — veto on stale/corrupt data.

Checks candle freshness, gap detection, and monotonic timestamps.
Forces hold when data quality is too low to make reliable decisions.

When data quality is acceptable, endorses the pipeline's direction
instead of defaulting to hold.
"""
from __future__ import annotations

import pandas as pd

from hogan_bot.swarm_decision.agents._utils import get_baseline_action
from hogan_bot.swarm_decision.types import AgentVote, FreshnessInfo


class DataGuardianAgent:
    """Vetoes when data quality is below acceptable thresholds."""

    agent_id: str = "data_guardian_v1"

    def __init__(
        self,
        max_gap_bars: int = 3,
        max_stale_hours: float = 2.0,
        min_bars_required: int = 50,
    ) -> None:
        self._max_gap = max_gap_bars
        self._max_stale_h = max_stale_hours
        self._min_bars = min_bars_required

    def vote(
        self,
        *,
        symbol: str,
        candles: pd.DataFrame,
        as_of_ms: int | None,
        shared_context: dict,
    ) -> AgentVote:
        reasons: list[str] = []
        veto = False
        size_scale = 1.0
        freshness: FreshnessInfo | None = None

        if len(candles) < self._min_bars:
            return AgentVote(
                agent_id=self.agent_id,
                action="hold",
                confidence=0.0,
                size_scale=0.0,
                veto=True,
                block_reasons=[f"insufficient_bars:{len(candles)}"],
            )

        if "ts_ms" in candles.columns:
            ts = candles["ts_ms"]
            diffs = ts.diff().dropna()
            if len(diffs) > 0:
                median_gap = diffs.median()
                if median_gap > 0:
                    max_gap = diffs.max()
                    gap_ratio = max_gap / median_gap
                    if gap_ratio > self._max_gap * 3:
                        # Catastrophic gap (9x+): hard veto
                        veto = True
                        reasons.append(f"candle_gap_severe:{gap_ratio:.1f}x")
                    elif gap_ratio > self._max_gap * 2:
                        # Major gap (6-9x): heavy size reduction
                        size_scale *= 0.25
                        reasons.append(f"candle_gap_major:{gap_ratio:.1f}x")
                    elif gap_ratio > self._max_gap:
                        # Minor gap (3-6x): moderate size reduction
                        size_scale *= 0.50
                        reasons.append(f"candle_gap_minor:{gap_ratio:.1f}x")

                    non_mono = (diffs <= 0).sum()
                    if non_mono > 0:
                        veto = True
                        reasons.append(f"non_monotonic_ts:{non_mono}")

                if as_of_ms is not None:
                    latest_ts = int(ts.iloc[-1])
                    age_s = (as_of_ms - latest_ts) / 1000.0
                    age_h = age_s / 3600.0
                    is_stale = age_h > self._max_stale_h
                    freshness = FreshnessInfo(
                        as_of_ms=as_of_ms,
                        latest_source_ts_ms=latest_ts,
                        age_seconds=max(0.0, age_s),
                        is_stale=is_stale,
                    )
                    if is_stale:
                        veto = True
                        reasons.append(f"stale_candles:{age_h:.1f}h")

        dups = candles.duplicated(subset=["ts_ms"] if "ts_ms" in candles.columns else None)
        if dups.any():
            n_dups = int(dups.sum())
            reasons.append(f"duplicate_candles:{n_dups}")
            size_scale *= 0.5

        if veto:
            return AgentVote(
                agent_id=self.agent_id,
                action="hold",
                confidence=0.0,
                size_scale=0.0,
                veto=True,
                block_reasons=reasons,
                freshness=freshness,
            )

        data_ages = shared_context.get("data_ages", {})
        for source, age_h in data_ages.items():
            if age_h > 48.0:
                size_scale *= 0.50
                reasons.append(f"stale_{source}:{age_h:.0f}h")
            elif age_h > 24.0:
                size_scale *= 0.75
                reasons.append(f"aging_{source}:{age_h:.0f}h")

        baseline = get_baseline_action(shared_context)
        confidence = 0.5 + 0.5 * size_scale
        return AgentVote(
            agent_id=self.agent_id,
            action=baseline,
            confidence=max(0.0, min(1.0, confidence)),
            size_scale=max(0.0, min(1.0, size_scale)),
            veto=False,
            block_reasons=reasons,
            freshness=freshness,
        )
