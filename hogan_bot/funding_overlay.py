"""BTC funding rate overlay for position sizing.

Uses perpetual futures funding rate as a crowded-positioning indicator:
- High positive funding → longs are crowded → reduce long size, maintain short size
- High negative funding → shorts are crowded → reduce short size, maintain long size
- Neutral funding → no adjustment

The overlay is directional: it only penalizes trades that go WITH the crowd.
Walk-forward validated: funding rate data available March 2025+.

Usage
-----
    from hogan_bot.funding_overlay import FundingOverlay
    overlay = FundingOverlay.from_db(conn)
    scale = overlay.position_scale("buy", timestamp)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundingOverlay:
    """Position sizing overlay based on BTC perpetual funding rate."""

    _funding: dict = field(default_factory=dict, repr=False)

    high_funding_threshold: float = 2.0
    extreme_funding_threshold: float = 5.0
    crowd_penalty: float = 0.70
    extreme_penalty: float = 0.40
    contrarian_boost: float = 1.0
    extreme_boost: float = 1.0

    @classmethod
    def from_db(cls, conn, symbol: str = "BTC/USD") -> FundingOverlay:
        """Load funding rate history from the derivatives_metrics table."""
        import pandas as pd
        try:
            df = pd.read_sql_query(
                "SELECT ts_ms, value FROM derivatives_metrics "
                "WHERE symbol = ? AND metric = 'funding_rate' ORDER BY ts_ms",
                conn,
                params=(symbol,),
            )
        except Exception as exc:
            logger.error("FundingOverlay: DB query failed (not just empty data): %s", exc)
            df = pd.DataFrame()

        funding_dict: dict = {}
        if not df.empty:
            df["hour_key"] = (df["ts_ms"] // 3_600_000).astype(int)
            grouped = df.groupby("hour_key")["value"].mean()
            for hk, val in grouped.items():
                funding_dict[int(hk)] = float(val)
            logger.info("FundingOverlay: loaded %d hourly records", len(funding_dict))
        else:
            logger.warning("FundingOverlay: no funding rate data in derivatives_metrics table")

        return cls(_funding=funding_dict)

    def _get_funding(self, timestamp) -> Optional[float]:
        """Look up the funding rate for the given timestamp."""
        if not self._funding:
            return None
        if timestamp is None:
            return None
        if isinstance(timestamp, datetime):
            ts_ms = int(timestamp.timestamp() * 1000)
        elif isinstance(timestamp, (int, float)):
            ts_ms = int(timestamp) if timestamp > 1e12 else int(timestamp * 1000)
        else:
            return None

        hour_key = ts_ms // 3_600_000
        if hour_key in self._funding:
            return self._funding[hour_key]
        for offset in range(1, 9):
            if hour_key - offset in self._funding:
                return self._funding[hour_key - offset]
        return None

    def position_scale(self, action: str, timestamp) -> float:
        """Return a position scale factor based on funding rate crowding.

        Parameters
        ----------
        action : "buy" or "sell"
        timestamp : bar timestamp (datetime or ms epoch)

        Returns 1.0 when no data or neutral funding.
        Returns < 1.0 when trading WITH the crowd (penalize crowded side).
        Returns > 1.0 when trading AGAINST the crowd (contrarian boost).
        """
        rate = self._get_funding(timestamp)
        if rate is None:
            return 1.0

        abs_rate = abs(rate)
        if abs_rate < self.high_funding_threshold:
            return 1.0

        if abs_rate >= self.extreme_funding_threshold:
            penalty = self.extreme_penalty
            boost = self.extreme_boost
        else:
            t = (abs_rate - self.high_funding_threshold) / (
                self.extreme_funding_threshold - self.high_funding_threshold
            )
            penalty = self.crowd_penalty - t * (self.crowd_penalty - self.extreme_penalty)
            boost = self.contrarian_boost + t * (self.extreme_boost - self.contrarian_boost)

        # Positive funding = longs crowded
        if rate > 0:
            if action == "buy":
                return penalty
            elif action == "sell":
                return boost
        # Negative funding = shorts crowded
        elif rate < 0:
            if action == "sell":
                return penalty
            elif action == "buy":
                return boost

        return 1.0
