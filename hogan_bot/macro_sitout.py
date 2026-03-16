"""Macro event sit-out filter for Hogan.

Combines three independent signals to decide whether technical signals are
reliable at a given point in time:

1. **Event calendar** — known macro events (FOMC, CPI, NFP) where technicals
   historically break down.  Blackout windows around announcement times.
2. **Crypto Fear & Greed Index** — extreme fear (<20) or extreme greed (>80)
   indicate event-driven regimes where patterns are unreliable.
3. **VIX level** — elevated VIX = macro-driven volatility overriding technicals.

Usage
-----
    sitout = MacroSitout.from_db(conn)
    result = sitout.check(bar_timestamp)
    if result.should_sitout:
        action = "hold"
    else:
        size *= result.size_scale
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FOMC / CPI / NFP event calendar (2024-06 through 2026-06)
# ---------------------------------------------------------------------------

_FOMC_DATES: list[str] = [
    # 2024 (statement day = 2nd day of meeting, 14:00 ET)
    "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]

# CPI: released ~10-14th of each month, 08:30 ET
_CPI_DATES: list[str] = [
    # 2024
    "2024-06-12", "2024-07-11", "2024-08-14", "2024-09-11",
    "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-11", "2025-10-14", "2025-11-13", "2025-12-18",
    # 2026
    "2026-01-13", "2026-02-13", "2026-03-11",
]

# NFP (Non-Farm Payrolls): first Friday of each month, 08:30 ET
_NFP_DATES: list[str] = [
    # 2024
    "2024-06-07", "2024-07-05", "2024-08-02", "2024-09-06",
    "2024-10-04", "2024-11-01", "2024-12-06",
    # 2025
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    # 2026
    "2026-01-09", "2026-02-06", "2026-03-06",
]


@dataclass
class SitoutResult:
    """Output of the macro sitout check."""
    should_sitout: bool = False
    size_scale: float = 1.0
    reasons: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.reasons:
            return "clear"
        return "; ".join(self.reasons)


@dataclass
class MacroSitout:
    """Pre-loaded macro data for fast per-bar lookups."""

    event_blackout_hours: float = 4.0
    fng_extreme_fear: int = 20
    fng_extreme_greed: int = 80
    # NOTE: Fear/Greed scaling is intentionally asymmetric. A fear scale of 1.0
    # means extreme fear does NOT reduce position size; this field is retained
    # for config/API compatibility and is effectively a no-op at the default.
    fng_fear_scale: float = 1.0
    fng_greed_scale: float = 0.30
    vix_caution: float = 25.0
    vix_block: float = 35.0
    vix_caution_scale: float = 0.50

    _event_dates: set[str] = field(default_factory=set, repr=False)
    _fng_by_date: dict[str, int] = field(default_factory=dict, repr=False)
    _vix_by_hour: dict[str, float] = field(default_factory=dict, repr=False)

    @classmethod
    def from_db(
        cls,
        conn,
        *,
        event_blackout_hours: float = 4.0,
        fng_extreme_fear: int = 20,
        fng_extreme_greed: int = 80,
        vix_caution: float = 25.0,
        vix_block: float = 35.0,
    ) -> MacroSitout:
        """Build a MacroSitout instance pre-loaded with DB data."""
        fng_by_date: dict[str, int] = {}
        try:
            rows = conn.execute(
                "SELECT date, value FROM onchain_metrics "
                "WHERE metric = 'fear_greed_value' ORDER BY date"
            ).fetchall()
            for date_str, value in rows:
                fng_by_date[date_str] = int(value)
            logger.info("MacroSitout: loaded %d Fear & Greed records", len(fng_by_date))
        except Exception as exc:
            logger.warning("MacroSitout: no Fear & Greed data: %s", exc)

        vix_by_hour: dict[str, float] = {}
        try:
            rows = conn.execute(
                "SELECT ts_ms, close FROM candles "
                "WHERE symbol = 'VIX/USD' ORDER BY ts_ms"
            ).fetchall()
            for ts_ms, close_val in rows:
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                key = dt.strftime("%Y-%m-%d-%H")
                vix_by_hour[key] = float(close_val)
            logger.info("MacroSitout: loaded %d VIX hourly records", len(vix_by_hour))
        except Exception as exc:
            logger.warning("MacroSitout: no VIX data: %s", exc)

        all_events = set(_FOMC_DATES) | set(_CPI_DATES) | set(_NFP_DATES)

        return cls(
            event_blackout_hours=event_blackout_hours,
            fng_extreme_fear=fng_extreme_fear,
            fng_extreme_greed=fng_extreme_greed,
            vix_caution=vix_caution,
            vix_block=vix_block,
            _event_dates=all_events,
            _fng_by_date=fng_by_date,
            _vix_by_hour=vix_by_hour,
        )

    def check(self, ts: pd.Timestamp | datetime | None) -> SitoutResult:
        """Check whether the given timestamp falls in a sitout zone.

        Returns a SitoutResult with should_sitout, size_scale, and reasons.
        """
        result = SitoutResult()

        if ts is None:
            return result

        if isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        elif isinstance(ts, datetime):
            dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        else:
            return result

        self._check_event_calendar(dt, result)
        self._check_fear_greed(dt, result)
        self._check_vix(dt, result)

        return result

    def _check_event_calendar(self, dt: datetime, result: SitoutResult) -> None:
        """Check if we're within blackout hours of a known macro event."""
        date_str = dt.strftime("%Y-%m-%d")
        prev_date = (dt - timedelta(days=1)).strftime("%Y-%m-%d")

        if date_str in self._event_dates:
            result.should_sitout = True
            events = []
            if date_str in _FOMC_DATES:
                events.append("FOMC")
            if date_str in _CPI_DATES:
                events.append("CPI")
            if date_str in _NFP_DATES:
                events.append("NFP")
            result.reasons.append(f"event_day({'+'.join(events)})")
            return

        if prev_date in self._event_dates and dt.hour < self.event_blackout_hours:
            result.size_scale = min(result.size_scale, 0.50)
            result.reasons.append("event_aftermath")

    def _check_fear_greed(self, dt: datetime, result: SitoutResult) -> None:
        """Check Fear & Greed extremes."""
        date_str = dt.strftime("%Y-%m-%d")
        prev_date = (dt - timedelta(days=1)).strftime("%Y-%m-%d")

        fng = self._fng_by_date.get(date_str) or self._fng_by_date.get(prev_date)
        if fng is None:
            return

        if fng <= self.fng_extreme_fear:
            result.size_scale = min(result.size_scale, self.fng_fear_scale)
            result.reasons.append(f"extreme_fear(FnG={fng})")
        elif fng >= self.fng_extreme_greed:
            result.size_scale = min(result.size_scale, self.fng_greed_scale)
            result.reasons.append(f"extreme_greed(FnG={fng})")

    def _check_vix(self, dt: datetime, result: SitoutResult) -> None:
        """Check VIX level."""
        hour_key = dt.strftime("%Y-%m-%d-%H")
        prev_key = (dt - timedelta(hours=1)).strftime("%Y-%m-%d-%H")

        vix = self._vix_by_hour.get(hour_key) or self._vix_by_hour.get(prev_key)
        if vix is None:
            return

        if vix >= self.vix_block:
            result.should_sitout = True
            result.reasons.append(f"vix_extreme({vix:.1f}>={self.vix_block})")
        elif vix >= self.vix_caution:
            result.size_scale = min(result.size_scale, self.vix_caution_scale)
            result.reasons.append(f"vix_elevated({vix:.1f}>={self.vix_caution})")


def count_sitout_bars(
    sitout: MacroSitout,
    candles: pd.DataFrame,
) -> dict:
    """Count how many bars would be blocked/scaled by each filter.

    Useful for diagnostics.
    """
    stats = {
        "total_bars": len(candles),
        "sitout_bars": 0,
        "scaled_bars": 0,
        "clear_bars": 0,
        "event_sitouts": 0,
        "fng_scales": 0,
        "vix_sitouts": 0,
        "vix_scales": 0,
    }
    for _, row in candles.iterrows():
        ts = row.get("timestamp")
        r = sitout.check(ts)
        if r.should_sitout:
            stats["sitout_bars"] += 1
            for reason in r.reasons:
                if "event" in reason:
                    stats["event_sitouts"] += 1
                if "vix_extreme" in reason:
                    stats["vix_sitouts"] += 1
        elif r.size_scale < 1.0:
            stats["scaled_bars"] += 1
            for reason in r.reasons:
                if "fear" in reason or "greed" in reason:
                    stats["fng_scales"] += 1
                if "vix_elevated" in reason:
                    stats["vix_scales"] += 1
        else:
            stats["clear_bars"] += 1

    return stats
