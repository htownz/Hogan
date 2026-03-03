from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass
class BotConfig:
    """Runtime configuration for Hogan."""

    starting_balance_usd: float = 1800.0
    aggressive_allocation: float = 0.75
    max_risk_per_trade: float = 0.03
    max_drawdown: float = 0.15
    symbols: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])

    timeframe: str = "5m"
    ohlcv_limit: int = 500
    short_ma_window: int = 20
    long_ma_window: int = 50
    volume_window: int = 20
    volume_threshold: float = 1.2

    fee_rate: float = 0.0026
    sleep_seconds: int = 30
    trade_weekends: bool = False
    paper_mode: bool = True

    use_ml_filter: bool = False
    ml_model_path: str = "models/hogan_logreg.pkl"
    ml_buy_threshold: float = 0.55
    ml_sell_threshold: float = 0.45

    kraken_api_key: str | None = None
    kraken_api_secret: str | None = None


def _split_symbols(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def load_config() -> BotConfig:
    """Load bot configuration from environment variables."""
    load_dotenv()
    return BotConfig(
        starting_balance_usd=float(os.getenv("HOGAN_STARTING_BALANCE", "1800")),
        aggressive_allocation=float(os.getenv("HOGAN_AGGRESSIVE_ALLOCATION", "0.75")),
        max_risk_per_trade=float(os.getenv("HOGAN_MAX_RISK_PER_TRADE", "0.03")),
        max_drawdown=float(os.getenv("HOGAN_MAX_DRAWDOWN", "0.15")),
        symbols=_split_symbols(os.getenv("HOGAN_SYMBOLS", "BTC/USD,ETH/USD")),
        timeframe=os.getenv("HOGAN_TIMEFRAME", "5m"),
        ohlcv_limit=int(os.getenv("HOGAN_OHLCV_LIMIT", "500")),
        short_ma_window=int(os.getenv("HOGAN_SHORT_MA", "20")),
        long_ma_window=int(os.getenv("HOGAN_LONG_MA", "50")),
        volume_window=int(os.getenv("HOGAN_VOLUME_WINDOW", "20")),
        volume_threshold=float(os.getenv("HOGAN_VOLUME_THRESHOLD", "1.2")),
        fee_rate=float(os.getenv("HOGAN_FEE_RATE", "0.0026")),
        sleep_seconds=int(os.getenv("HOGAN_SLEEP_SECONDS", "30")),
        trade_weekends=os.getenv("HOGAN_TRADE_WEEKENDS", "false").lower() == "true",
        paper_mode=os.getenv("HOGAN_PAPER_MODE", "true").lower() == "true",
        use_ml_filter=os.getenv("HOGAN_USE_ML_FILTER", "false").lower() == "true",
        ml_model_path=os.getenv("HOGAN_ML_MODEL_PATH", "models/hogan_logreg.pkl"),
        ml_buy_threshold=float(os.getenv("HOGAN_ML_BUY_THRESHOLD", "0.55")),
        ml_sell_threshold=float(os.getenv("HOGAN_ML_SELL_THRESHOLD", "0.45")),
        kraken_api_key=os.getenv("KRAKEN_API_KEY"),
        kraken_api_secret=os.getenv("KRAKEN_API_SECRET"),
    )
