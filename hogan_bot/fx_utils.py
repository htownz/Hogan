"""FX-specific utilities for Hogan.

Session-aware trading, pip-based risk management, and FX candle ingestion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, time as dt_time
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session definitions (UTC times)
# ---------------------------------------------------------------------------

SessionName = Literal["asia", "london", "new_york", "overlap_london_ny", "off_hours"]


@dataclass(frozen=True)
class TradingSession:
    name: SessionName
    start: dt_time
    end: dt_time
    description: str


SESSIONS = {
    "asia": TradingSession(
        name="asia",
        start=dt_time(0, 0),
        end=dt_time(8, 0),
        description="Tokyo/Sydney session (00:00-08:00 UTC)",
    ),
    "london": TradingSession(
        name="london",
        start=dt_time(7, 0),
        end=dt_time(16, 0),
        description="London session (07:00-16:00 UTC)",
    ),
    "new_york": TradingSession(
        name="new_york",
        start=dt_time(12, 0),
        end=dt_time(21, 0),
        description="New York session (12:00-21:00 UTC)",
    ),
    "overlap_london_ny": TradingSession(
        name="overlap_london_ny",
        start=dt_time(12, 0),
        end=dt_time(16, 0),
        description="London/NY overlap — highest liquidity (12:00-16:00 UTC)",
    ),
}


def current_session(utc_now: datetime | None = None) -> SessionName:
    """Identify the active trading session based on UTC time."""
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)
    t = utc_now.time()

    if dt_time(12, 0) <= t < dt_time(16, 0):
        return "overlap_london_ny"
    if dt_time(7, 0) <= t < dt_time(16, 0):
        return "london"
    if dt_time(12, 0) <= t < dt_time(21, 0):
        return "new_york"
    if dt_time(0, 0) <= t < dt_time(8, 0):
        return "asia"
    return "off_hours"


def is_weekend(utc_now: datetime | None = None) -> bool:
    """Check if FX markets are closed (Saturday/Sunday)."""
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)
    wd = utc_now.weekday()
    if wd == 4 and utc_now.time() >= dt_time(21, 0):
        return True
    if wd == 5:
        return True
    if wd == 6 and utc_now.time() < dt_time(21, 0):
        return True
    return False


# ---------------------------------------------------------------------------
# Session-based trading filters
# ---------------------------------------------------------------------------

@dataclass
class SessionFilter:
    """Controls which sessions Hogan is allowed to trade in."""
    allowed_sessions: list[SessionName] = None
    block_weekends: bool = True
    block_off_hours: bool = True
    reduce_size_asia: float = 0.5

    def __post_init__(self):
        if self.allowed_sessions is None:
            self.allowed_sessions = ["london", "new_york", "overlap_london_ny"]

    def should_trade(self, utc_now: datetime | None = None) -> tuple[bool, float, str]:
        """Check if trading is allowed now.

        Returns
        -------
        (allowed, size_scale, reason)
            allowed: whether to trade
            size_scale: 0.0 to 1.0 multiplier for position size
            reason: human-readable explanation
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        if self.block_weekends and is_weekend(utc_now):
            return False, 0.0, "FX markets closed (weekend)"

        session = current_session(utc_now)

        if session == "off_hours" and self.block_off_hours:
            return False, 0.0, f"Off-hours session ({utc_now.time().isoformat()[:5]} UTC)"

        if session == "asia":
            if "asia" not in self.allowed_sessions:
                return False, 0.0, "Asia session not in allowed sessions"
            return True, self.reduce_size_asia, f"Asia session (reduced size {self.reduce_size_asia:.0%})"

        if session not in self.allowed_sessions:
            return False, 0.0, f"Session '{session}' not in allowed sessions"

        scale = 1.2 if session == "overlap_london_ny" else 1.0
        return True, scale, f"Active session: {session}"


# ---------------------------------------------------------------------------
# Pip-based risk management
# ---------------------------------------------------------------------------

_PIP_SIZES: dict[str, float] = {
    "EUR/USD": 0.0001, "GBP/USD": 0.0001, "AUD/USD": 0.0001,
    "NZD/USD": 0.0001, "USD/CAD": 0.0001, "USD/CHF": 0.0001,
    "EUR/GBP": 0.0001, "EUR/JPY": 0.01,
    "USD/JPY": 0.01, "GBP/JPY": 0.01, "AUD/JPY": 0.01,
    "XAU/USD": 0.01,
}


def pip_size(symbol: str) -> float:
    return _PIP_SIZES.get(symbol, 0.0001)


def pips_to_price(symbol: str, pips: float) -> float:
    """Convert N pips to a price difference for the given symbol."""
    return pips * pip_size(symbol)


def price_to_pips(symbol: str, price_diff: float) -> float:
    """Convert a price difference to pips."""
    return price_diff / pip_size(symbol)


def pip_stop_loss(symbol: str, entry_price: float, side: str, stop_pips: float) -> float:
    """Calculate stop-loss price from entry and stop distance in pips."""
    offset = pips_to_price(symbol, stop_pips)
    if side == "long":
        return entry_price - offset
    return entry_price + offset


def pip_take_profit(symbol: str, entry_price: float, side: str, tp_pips: float) -> float:
    """Calculate take-profit price from entry and target in pips."""
    offset = pips_to_price(symbol, tp_pips)
    if side == "long":
        return entry_price + offset
    return entry_price - offset


def fx_position_size(
    account_balance: float,
    risk_pct: float,
    stop_pips: float,
    symbol: str,
    price: float,
) -> float:
    """Calculate FX position size in units based on pip-denominated risk.

    Parameters
    ----------
    account_balance : float
        Account balance in the account currency (USD).
    risk_pct : float
        Max risk per trade as a fraction (e.g. 0.01 = 1%).
    stop_pips : float
        Stop-loss distance in pips.
    symbol : str
        Trading pair (e.g. "EUR/USD").
    price : float
        Current price of the instrument.

    Returns
    -------
    float
        Number of units (e.g. 10000 = 0.1 lot).
    """
    if stop_pips <= 0 or price <= 0 or account_balance <= 0:
        return 0.0

    risk_amount = account_balance * risk_pct
    pip_val = pip_size(symbol)

    # For XXX/USD pairs, 1 pip per unit = pip_size
    # For USD/XXX pairs, 1 pip per unit = pip_size / price
    if symbol.endswith("/USD") or symbol.endswith("/USDT"):
        pip_value_per_unit = pip_val
    else:
        pip_value_per_unit = pip_val / price

    units = risk_amount / (stop_pips * pip_value_per_unit)
    return round(units, 0)


# ---------------------------------------------------------------------------
# FX candle ingestion (wrapper for OandaClient)
# ---------------------------------------------------------------------------

def fetch_fx_candles(
    symbol: str,
    timeframe: str = "15m",
    count: int = 500,
    context_timeframe: str = "1h",
    context_count: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch primary + context timeframe candles from Oanda.

    Returns
    -------
    (primary_candles, context_candles)
        Both as DataFrames with standard OHLCV columns.
    """
    from hogan_bot.oanda_client import OandaClient

    client = OandaClient()
    primary = client.fetch_candles(symbol, timeframe=timeframe, count=count)
    context = client.fetch_candles(symbol, timeframe=context_timeframe, count=context_count)

    logger.info(
        "FX candles: %s %s=%d bars, %s=%d bars",
        symbol, timeframe, len(primary), context_timeframe, len(context),
    )
    return primary, context
