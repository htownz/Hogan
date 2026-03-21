"""Champion Path — the single canonical production configuration for Hogan.

When ``HOGAN_CHAMPION_MODE=true``, all experimental features are disabled and
the bot runs only the validated, promoted configuration. This prevents the
codebase from becoming "five half-married systems in a trench coat."

Usage::

    from hogan_bot.champion import apply_champion_mode

    config = load_config()
    config = apply_champion_mode(config)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChampionLocks:
    """Defines which features are locked in champion mode.

    Experimental features are explicitly disabled. Strategy parameters are
    frozen to the validated production values. Any feature wanting promotion
    must beat the champion on net expectancy, drawdown, stability, trade count,
    and cost-adjusted performance.
    """
    use_ict: bool = False
    use_rl_agent: bool = False
    use_mtf_ensemble: bool = False
    use_macro_filter: bool = False
    use_online_learning: bool = False
    use_mtf_extended: bool = True
    use_ema_clouds: bool = False
    use_fvg: bool = False

    signal_mode: str = "any"
    # With all extra voters disabled (EMA clouds, FVG, ICT, RL), only MA
    # crossover/trend remain — a single voter.  Margin=2 is mathematically
    # impossible with 1 voter; margin=1 lets the tech signal through and
    # relies on the ML filter + edge gate + quality gate stack for protection.
    signal_min_vote_margin: int = 1

    use_regime_detection: bool = True
    ml_confidence_sizing: bool = False
    use_ml_as_sizer: bool = True

    # Regime-routed strategy families (trend/mean-revert/breakout)
    use_strategy_router: bool = True
    volatile_policy: str = "breakout"

    min_hold_bars: int = 3
    exit_confirmation_bars: int = 2
    min_edge_multiple: float = 1.5

    # Entry quality gate thresholds — lowered to allow mixed-signal trades
    # through. Sentiment often opposes tech, tanking combined confidence;
    # the quality gate and sizing already scale positions down for weak signals.
    min_final_confidence: float = 0.04
    min_tech_confidence: float = 0.10
    min_regime_confidence: float = 0.20
    max_whipsaws: int = 3
    reversal_confidence_multiplier: float = 1.3

    max_hold_hours: float = 24.0
    loss_cooldown_hours: float = 2.0


CHAMPION_LOCKS = ChampionLocks()

_EXPERIMENTAL_FLAGS = (
    "use_ict", "use_rl_agent", "use_mtf_ensemble",
    "use_macro_filter", "use_online_learning",
)


def is_champion_mode() -> bool:
    """Check whether champion mode is enabled via environment."""
    return os.getenv("HOGAN_CHAMPION_MODE", "false").lower() in ("true", "1", "yes")


def apply_champion_mode(config):
    """Apply champion locks to a BotConfig, returning a new config.

    If HOGAN_CHAMPION_MODE is not set, returns the config unchanged but logs
    warnings for any active experimental features.
    """
    if not is_champion_mode():
        active_experiments = [
            flag for flag in _EXPERIMENTAL_FLAGS
            if getattr(config, flag, False)
        ]
        if active_experiments:
            logger.warning(
                "Experimental features active outside champion mode: %s. "
                "Set HOGAN_CHAMPION_MODE=true to lock down production config.",
                ", ".join(active_experiments),
            )
        return config

    overrides = {}
    for field_name in ChampionLocks.__dataclass_fields__:
        if not hasattr(config, field_name):
            logger.debug("CHAMPION_MODE: skipping %s (not in BotConfig)", field_name)
            continue
        champion_val = getattr(CHAMPION_LOCKS, field_name)
        current_val = getattr(config, field_name)
        if current_val != champion_val:
            logger.info(
                "CHAMPION_MODE: overriding %s = %r -> %r",
                field_name, current_val, champion_val,
            )
            overrides[field_name] = champion_val

    # Champion mode also switches to champion model (trained on 8-feature subset)
    champion_path = getattr(config, "champion_ml_model_path", "models/hogan_champion.pkl")
    if config.ml_model_path != champion_path:
        overrides["ml_model_path"] = champion_path
        logger.info("CHAMPION_MODE: ml_model_path -> %s (8-feature model)", champion_path)

    if overrides:
        config = replace(config, **overrides)
        logger.info("Champion mode applied — %d overrides", len(overrides))
    else:
        logger.info("Champion mode active — all values already match")

    return config


def get_champion_summary() -> dict:
    """Return a dict describing the champion configuration for logging."""
    from hogan_bot.feature_registry import CHAMPION_FEATURE_COLUMNS
    return {
        "champion_mode": is_champion_mode(),
        "champion_feature_count": len(CHAMPION_FEATURE_COLUMNS),
        "locked_experiments_off": list(_EXPERIMENTAL_FLAGS),
        "signal_mode": CHAMPION_LOCKS.signal_mode,
        "min_vote_margin": CHAMPION_LOCKS.signal_min_vote_margin,
        "use_strategy_router": CHAMPION_LOCKS.use_strategy_router,
        "volatile_policy": CHAMPION_LOCKS.volatile_policy,
        "min_hold_bars": CHAMPION_LOCKS.min_hold_bars,
        "exit_confirmation_bars": CHAMPION_LOCKS.exit_confirmation_bars,
        "min_edge_multiple": CHAMPION_LOCKS.min_edge_multiple,
        "min_final_confidence": CHAMPION_LOCKS.min_final_confidence,
        "min_tech_confidence": CHAMPION_LOCKS.min_tech_confidence,
        "min_regime_confidence": CHAMPION_LOCKS.min_regime_confidence,
        "max_whipsaws": CHAMPION_LOCKS.max_whipsaws,
        "reversal_confidence_multiplier": CHAMPION_LOCKS.reversal_confidence_multiplier,
        "max_hold_hours": CHAMPION_LOCKS.max_hold_hours,
        "loss_cooldown_hours": CHAMPION_LOCKS.loss_cooldown_hours,
    }
