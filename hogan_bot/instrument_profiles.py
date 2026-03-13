"""Instrument profiles for Hogan.

Maps asset classes to appropriate risk parameters, cost models, and
stop/target logic.  This prevents crypto-style percentage stops from
being blindly applied to FX pairs (and vice versa).

Usage::

    from hogan_bot.instrument_profiles import get_profile, classify_symbol

    profile = get_profile("EUR/USD")
    # profile.asset_class == "fx_major"
    # profile.default_stop_pips == 30
    # profile.typical_spread_bps == 1.0
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentProfile:
    """Risk and cost parameters for an instrument class."""
    asset_class: str
    # Stop/target in pips (FX) or percent (crypto/equity)
    use_pip_based_risk: bool = False
    default_stop_pips: float = 0.0
    default_tp_pips: float = 0.0
    default_stop_pct: float = 0.02
    default_tp_pct: float = 0.054
    # Cost model
    typical_spread_bps: float = 5.0
    typical_fee_bps: float = 26.0     # round-trip
    # Position limits
    max_leverage: float = 1.0
    min_trade_size: float = 0.0


_PROFILES: dict[str, InstrumentProfile] = {
    "crypto_spot": InstrumentProfile(
        asset_class="crypto_spot",
        use_pip_based_risk=False,
        default_stop_pct=0.02,
        default_tp_pct=0.054,
        typical_spread_bps=5.0,
        typical_fee_bps=26.0,
        max_leverage=1.0,
    ),
    "fx_major": InstrumentProfile(
        asset_class="fx_major",
        use_pip_based_risk=True,
        default_stop_pips=30.0,
        default_tp_pips=60.0,
        default_stop_pct=0.003,   # ~30 pips on EUR/USD ≈ 0.3%
        default_tp_pct=0.006,     # ~60 pips ≈ 0.6%
        typical_spread_bps=1.0,   # ~1 pip = ~1 bps for EUR/USD
        typical_fee_bps=0.0,      # no commission on Oanda standard
        max_leverage=1.0,
        min_trade_size=100.0,
    ),
    "fx_cross": InstrumentProfile(
        asset_class="fx_cross",
        use_pip_based_risk=True,
        default_stop_pips=40.0,
        default_tp_pips=80.0,
        default_stop_pct=0.004,
        default_tp_pct=0.008,
        typical_spread_bps=2.0,
        typical_fee_bps=0.0,
        max_leverage=1.0,
        min_trade_size=100.0,
    ),
    "equity": InstrumentProfile(
        asset_class="equity",
        use_pip_based_risk=False,
        default_stop_pct=0.015,
        default_tp_pct=0.03,
        typical_spread_bps=2.0,
        typical_fee_bps=10.0,
        max_leverage=1.0,
    ),
}

_FX_MAJORS = {"EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD", "USD/CAD", "USD/CHF", "USD/JPY"}
_FX_CROSSES = {"EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/CHF"}
_CRYPTO_MARKERS = {"BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "AVAX", "DOT", "LINK", "MATIC"}


def classify_symbol(symbol: str) -> str:
    """Classify a symbol into an asset class."""
    sym = symbol.upper()
    if sym in _FX_MAJORS:
        return "fx_major"
    if sym in _FX_CROSSES:
        return "fx_cross"
    if any(c in sym for c in _CRYPTO_MARKERS):
        return "crypto_spot"
    if "/" in sym:
        base, quote = sym.split("/", 1)
        if base in _FX_MAJORS or quote in ("JPY", "CHF", "GBP", "EUR", "CAD", "AUD", "NZD"):
            return "fx_cross"
    return "crypto_spot"


def get_profile(symbol: str) -> InstrumentProfile:
    """Get the instrument profile for a symbol."""
    return _PROFILES[classify_symbol(symbol)]


def spread_cost_bps(symbol: str) -> float:
    """Return the typical spread cost in basis points for the symbol.

    For FX this is based on the Oanda typical spread (not a fake
    flat percentage).  For crypto, uses the configured spread estimate.
    """
    return get_profile(symbol).typical_spread_bps


def total_friction_bps(symbol: str) -> float:
    """Round-trip friction in bps (spread + fees)."""
    p = get_profile(symbol)
    return p.typical_spread_bps * 2 + p.typical_fee_bps
