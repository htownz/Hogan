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
    signal_min_vote_margin: int = 2

    use_regime_detection: bool = True
    ml_confidence_sizing: bool = True

    min_hold_bars: int = 3
    exit_confirmation_bars: int = 2
    min_edge_multiple: float = 1.5

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
        champion_val = getattr(CHAMPION_LOCKS, field_name)
        current_val = getattr(config, field_name, None)
        if current_val != champion_val:
            logger.info(
                "CHAMPION_MODE: overriding %s = %r -> %r",
                field_name, current_val, champion_val,
            )
            overrides[field_name] = champion_val

    if overrides:
        config = replace(config, **overrides)
        logger.info("Champion mode applied — %d overrides", len(overrides))
    else:
        logger.info("Champion mode active — all values already match")

    return config


def get_champion_summary() -> dict:
    """Return a dict describing the champion configuration for logging."""
    return {
        "champion_mode": is_champion_mode(),
        "locked_experiments_off": list(_EXPERIMENTAL_FLAGS),
        "signal_mode": CHAMPION_LOCKS.signal_mode,
        "min_vote_margin": CHAMPION_LOCKS.signal_min_vote_margin,
        "min_hold_bars": CHAMPION_LOCKS.min_hold_bars,
        "exit_confirmation_bars": CHAMPION_LOCKS.exit_confirmation_bars,
        "min_edge_multiple": CHAMPION_LOCKS.min_edge_multiple,
        "max_hold_hours": CHAMPION_LOCKS.max_hold_hours,
        "loss_cooldown_hours": CHAMPION_LOCKS.loss_cooldown_hours,
    }
