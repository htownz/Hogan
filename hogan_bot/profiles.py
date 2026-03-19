"""Named strategy profiles for reproducible backtests.

Each profile is a dict of keyword overrides applied on top of
:func:`~hogan_bot.config.load_config`.  The ``CANONICAL_PROFILE`` is
Hogan's blessed production baseline — every comparison should start here.
"""
from __future__ import annotations

CANONICAL_PROFILE: dict[str, object] = {
    # ── Instrument & timeframe ────────────────────────────────────────
    "symbol": "BTC/USD",
    "timeframe": "1h",

    # ── ML filter ─────────────────────────────────────────────────────
    "use_ml_filter": True,
    "ml_buy_threshold": 0.51,
    "ml_sell_threshold": 0.49,
    "ml_confidence_sizing": False,
    "use_ml_as_sizer": True,

    # ── Signal quality gates ──────────────────────────────────────────
    "min_final_confidence": 0.20,
    "min_tech_confidence": 0.15,
    "min_regime_confidence": 0.30,
    "min_edge_multiple": 1.5,
    "max_whipsaws": 3,

    # ── Risk parameters ───────────────────────────────────────────────
    "trailing_stop_pct": 0.030,
    "take_profit_pct": 0.0572,
    "fee_rate": 0.0026,

    # ── Hold limits ───────────────────────────────────────────────────
    "max_hold_hours": 24.0,
    "short_max_hold_hours": 12.0,
    "loss_cooldown_hours": 2.0,

    # ── Position sides ────────────────────────────────────────────────
    "enable_shorts": True,
    "enable_close_and_reverse": False,
    "enable_pullback_gate": True,

    # ── Regime detection ──────────────────────────────────────────────
    "use_regime_detection": True,
}

PROFILES: dict[str, dict[str, object]] = {
    "canonical": CANONICAL_PROFILE,
    "long_only": {**CANONICAL_PROFILE, "enable_shorts": False},
    "no_ml": {**CANONICAL_PROFILE, "use_ml_filter": False},
}


def get_profile(name: str) -> dict[str, object]:
    """Return a named profile dict, raising ``KeyError`` on unknown names."""
    return PROFILES[name]


def apply_profile(cfg, profile: dict[str, object]):
    """Overlay *profile* values onto a :class:`~hogan_bot.config.BotConfig`.

    Only keys that exist as BotConfig attributes are applied.  Keys that
    are backtest-CLI-only (``enable_shorts``, ``enable_pullback_gate``,
    ``enable_close_and_reverse``) are returned separately so the caller
    can forward them.

    Returns ``(cfg, cli_overrides)`` where *cli_overrides* is a dict of
    keys that should be passed to the backtest runner, not set on cfg.
    """
    cli_keys = {"enable_shorts", "enable_pullback_gate", "enable_close_and_reverse",
                "symbol", "timeframe"}
    cli_overrides: dict[str, object] = {}
    for key, value in profile.items():
        if key in cli_keys:
            cli_overrides[key] = value
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg, cli_overrides
