"""Exit packs for the strategy matrix tournament.

Three standardized exit configurations. A good entry should show life
across more than one exit pack — fragile entries only work with one.

T1. Trend       -- wide stop, trailing, no TP, long hold
T2. MeanRevert  -- tight stop, fixed TP, short hold
T3. Balanced    -- moderate stop, moderate TP, medium hold
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExitPack:
    """Immutable exit configuration for the tournament matrix."""
    name: str
    stop_atr_mult: float
    take_profit_atr_mult: float   # 0.0 = no fixed TP
    trailing_stop_atr_mult: float  # 0.0 = no trail
    max_hold_hours: float

    @property
    def has_take_profit(self) -> bool:
        return self.take_profit_atr_mult > 0.0

    @property
    def has_trailing_stop(self) -> bool:
        return self.trailing_stop_atr_mult > 0.0


T1_TREND = ExitPack(
    name="T1_trend",
    stop_atr_mult=2.0,
    take_profit_atr_mult=0.0,
    trailing_stop_atr_mult=2.5,
    max_hold_hours=120.0,
)

T2_MEAN_REVERT = ExitPack(
    name="T2_mean_revert",
    stop_atr_mult=1.25,
    take_profit_atr_mult=2.0,
    trailing_stop_atr_mult=0.0,
    max_hold_hours=36.0,
)

T3_BALANCED = ExitPack(
    name="T3_balanced",
    stop_atr_mult=1.5,
    take_profit_atr_mult=3.0,
    trailing_stop_atr_mult=0.0,
    max_hold_hours=72.0,
)

EXIT_PACKS: dict[str, ExitPack] = {
    "T1_trend": T1_TREND,
    "T2_mean_revert": T2_MEAN_REVERT,
    "T3_balanced": T3_BALANCED,
}

ALL_EXIT_PACKS: list[ExitPack] = [T1_TREND, T2_MEAN_REVERT, T3_BALANCED]
