"""Regime-aware strategy routing.

Maps the current market regime to the appropriate StrategyFamily and
delegates signal generation.  When regime confidence is too low or the
regime is unrecognised, the router returns a hold signal.

Tournament winner (D_bb_squeeze x T1_trend) can be activated with
``use_tournament_winner=True``. This replaces the regime-routed families
with the Bollinger Squeeze Breakout entry, which showed positive edge
across BTC, ETH, and SOL in zero-cost walk-forward testing.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from hogan_bot.strategy import (
    BreakoutFamily,
    MeanRevertFamily,
    StrategySignal,
    TrendFollowFamily,
)

if TYPE_CHECKING:
    from hogan_bot.strategy import StrategyFamily

logger = logging.getLogger(__name__)


class StrategyRouter:
    """Select a StrategyFamily based on the detected market regime."""

    def __init__(self, config=None, use_tournament_winner: bool | None = None):
        if use_tournament_winner is None:
            use_tournament_winner = os.getenv("HOGAN_TOURNAMENT_WINNER", "").lower() in ("1", "true")

        if use_tournament_winner:
            from hogan_bot.strategy_candidates import BollingerSqueezeBreakout
            winner = BollingerSqueezeBreakout()
            self.families: dict[str, StrategyFamily] = {
                "trending_up": winner,
                "trending_down": winner,
                "ranging": winner,
                "volatile": winner,
            }
            self._tournament_mode = True
            logger.info("StrategyRouter: tournament winner (bb_squeeze) active for all regimes")
        else:
            self.families = {
                "trending_up": TrendFollowFamily(),
                "trending_down": TrendFollowFamily(),
                "ranging": MeanRevertFamily(),
                "volatile": BreakoutFamily(),
            }
            self._tournament_mode = False
        self._config = config

    def route(
        self,
        candles,
        config,
        regime_state=None,
    ) -> StrategySignal:
        """Dispatch to the appropriate strategy family.

        Parameters
        ----------
        candles : pd.DataFrame
            OHLCV candle data.
        config : BotConfig
            Bot configuration.
        regime_state : RegimeState | None
            Output of ``detect_regime()``.  When ``None`` or confidence is
            below the configured threshold, defaults to trend-follow.

        Returns
        -------
        StrategySignal
        """
        if regime_state is None:
            logger.debug("No regime state — defaulting to trend_follow")
            return self.families["trending_up"].generate_signal(candles, config, regime_state)

        regime = regime_state.regime
        confidence = getattr(regime_state, "confidence", 0.0)
        min_conf = getattr(config, "min_regime_confidence", 0.3)

        if confidence < min_conf:
            logger.debug(
                "Regime %s confidence %.2f below threshold %.2f — using default family",
                regime, confidence, min_conf,
            )
            family = self.families.get(regime, self.families.get("trending_up"))
            sig = family.generate_signal(candles, config, regime_state)
            sig = StrategySignal(sig.action, sig.stop_distance_pct,
                                sig.confidence * 0.7, sig.volume_ratio)
            return sig

        family = self.families.get(regime)
        if family is None:
            logger.debug("Unknown regime '%s' — hold", regime)
            return StrategySignal("hold", 0.01, 0.0, 0.0)

        logger.debug("Routing to %s for regime=%s (conf=%.2f)", family.name, regime, confidence)
        sig = family.generate_signal(candles, config, regime_state)

        # Cross-family fallback: if primary family returns hold, try others
        if sig.action == "hold":
            _fallback_order = [f for r, f in self.families.items() if r != regime]
            for alt_family in _fallback_order:
                alt_sig = alt_family.generate_signal(candles, config, regime_state)
                if alt_sig.action != "hold" and alt_sig.confidence > 0.15:
                    logger.debug(
                        "Cross-family fallback: %s produced %s (conf=%.2f)",
                        alt_family.name, alt_sig.action, alt_sig.confidence,
                    )
                    return StrategySignal(
                        alt_sig.action, alt_sig.stop_distance_pct,
                        alt_sig.confidence * 0.6, alt_sig.volume_ratio,
                    )

        return sig

    @property
    def family_names(self) -> dict[str, str]:
        """Mapping of regime -> strategy family name for logging."""
        return {r: f.name for r, f in self.families.items()}
