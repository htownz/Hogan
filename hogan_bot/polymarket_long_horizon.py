"""Conservative long-horizon fair value estimates for Polymarket crypto targets."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from hogan_bot.storage import load_candles

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_DAY_YEAR_RE = re.compile(
    r"\b("
    + "|".join(_MONTHS)
    + r")\s+([0-9]{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)(20[0-9]{2})\b",
    re.I,
)
_YEAR_RE = re.compile(r"\b(?:by|before|in)\s+(20[0-9]{2})\b", re.I)


@dataclass(frozen=True)
class LongHorizonProbability:
    probability: float
    source: str
    current_price: float
    target_price: float
    days_to_deadline: float
    annualized_drift: float
    annualized_volatility: float
    sample_size: int

    def to_dict(self) -> dict:
        return {
            "probability": round(self.probability, 6),
            "source": self.source,
            "current_price": round(self.current_price, 2),
            "target_price": round(self.target_price, 2),
            "days_to_deadline": round(self.days_to_deadline, 2),
            "annualized_drift": round(self.annualized_drift, 6),
            "annualized_volatility": round(self.annualized_volatility, 6),
            "sample_size": self.sample_size,
        }


def parse_polymarket_deadline(question: str, *, as_of: datetime | None = None) -> datetime | None:
    """Extract a conservative expiry date from a Polymarket question."""
    now = as_of or datetime.now(UTC)
    match = _MONTH_DAY_YEAR_RE.search(question)
    if match:
        month_name, day, year = match.groups()
        try:
            return datetime(int(year), _MONTHS[month_name.lower()], int(day), 23, 59, tzinfo=UTC)
        except ValueError:
            return None
    year_match = _YEAR_RE.search(question)
    if not year_match:
        return None
    year = int(year_match.group(1))
    if "before" in question.lower():
        return datetime(year - 1, 12, 31, 23, 59, tzinfo=UTC)
    if year < now.year:
        return None
    return datetime(year, 12, 31, 23, 59, tzinfo=UTC)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _load_price_history(conn, symbol: str) -> tuple[pd.DataFrame, float]:
    daily = load_candles(conn, symbol, "1d", limit=1500)
    if len(daily) >= 90:
        return daily, 365.0
    hourly = load_candles(conn, symbol, "1h", limit=24 * 1500)
    return hourly, 365.0 * 24.0


def estimate_btc_long_horizon_probability(
    conn,
    *,
    target_price: float,
    question: str,
    symbol: str = "BTC/USD",
    as_of: datetime | None = None,
    min_samples: int = 90,
) -> LongHorizonProbability | None:
    """Estimate P(BTC closes above target by deadline) from local candle history.

    This is intentionally conservative for shadow gating: it uses terminal
    lognormal probability, shrinks positive historical drift, and requires a
    parseable market deadline plus enough local BTC price history.
    """
    deadline = parse_polymarket_deadline(question, as_of=as_of)
    now = as_of or datetime.now(UTC)
    if deadline is None or deadline <= now or target_price <= 0:
        return None

    candles, periods_per_year = _load_price_history(conn, symbol)
    if len(candles) < min_samples or "close" not in candles:
        return None
    closes = candles["close"].astype(float)
    closes = closes[closes > 0]
    if len(closes) < min_samples:
        return None

    log_returns = (closes / closes.shift(1)).map(math.log).dropna()
    if len(log_returns) < min_samples - 1:
        return None
    current_price = float(closes.iloc[-1])
    if target_price <= current_price:
        return LongHorizonProbability(
            probability=0.95,
            source="btc_long_horizon_lognormal_v1",
            current_price=current_price,
            target_price=float(target_price),
            days_to_deadline=max(0.0, (deadline - now).total_seconds() / 86_400.0),
            annualized_drift=0.0,
            annualized_volatility=0.35,
            sample_size=len(log_returns),
        )

    mean_return = float(log_returns.mean())
    std_return = float(log_returns.std(ddof=1))
    if not math.isfinite(std_return) or std_return <= 0:
        return None
    raw_drift = mean_return * periods_per_year
    drift = raw_drift * 0.25 if raw_drift > 0 else raw_drift
    drift = max(-0.50, min(0.25, drift))
    volatility = max(0.35, min(1.50, std_return * math.sqrt(periods_per_year)))
    years = max((deadline - now).total_seconds() / (365.0 * 86_400.0), 1.0 / 365.0)
    threshold = math.log(float(target_price) / current_price)
    denominator = volatility * math.sqrt(years)
    z = (threshold - (drift - 0.5 * volatility * volatility) * years) / denominator
    probability = max(0.0, min(1.0, 1.0 - _normal_cdf(z)))
    return LongHorizonProbability(
        probability=probability,
        source="btc_long_horizon_lognormal_v1",
        current_price=current_price,
        target_price=float(target_price),
        days_to_deadline=years * 365.0,
        annualized_drift=drift,
        annualized_volatility=volatility,
        sample_size=len(log_returns),
    )
