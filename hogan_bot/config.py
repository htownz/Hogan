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

    # Ripster EMA cloud settings
    use_ema_clouds: bool = False
    ema_fast_short: int = 8
    ema_fast_long: int = 9
    ema_slow_short: int = 34
    ema_slow_long: int = 50

    # ICT Fair-Value Gap settings
    use_fvg: bool = False
    fvg_min_gap_pct: float = 0.001

    # Signal combinator: "ma_only" | "any" | "all"
    signal_mode: str = "any"

    # Exit management (0 = disabled)
    trailing_stop_pct: float = 0.0
    take_profit_pct: float = 0.0

    # ML confidence-based position sizing: scales size by |prob−0.5|×2
    ml_confidence_sizing: bool = False

    # Webhook URL for trade/drawdown notifications (empty string = disabled)
    webhook_url: str = ""

    # CCXT exchange ID — any of the 110+ exchanges in the library
    exchange_id: str = "kraken"

    # Walk-forward retraining defaults (used by hogan_bot.retrain)
    retrain_window_bars: int = 5000
    retrain_model_type: str = "logreg"
    retrain_min_improvement: float = 0.005
    retrain_promotion_metric: str = "roc_auc"
    retrain_schedule_hours: float = 24.0

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
        use_ema_clouds=os.getenv("HOGAN_USE_EMA_CLOUDS", "false").lower() == "true",
        ema_fast_short=int(os.getenv("HOGAN_EMA_FAST_SHORT", "8")),
        ema_fast_long=int(os.getenv("HOGAN_EMA_FAST_LONG", "9")),
        ema_slow_short=int(os.getenv("HOGAN_EMA_SLOW_SHORT", "34")),
        ema_slow_long=int(os.getenv("HOGAN_EMA_SLOW_LONG", "50")),
        use_fvg=os.getenv("HOGAN_USE_FVG", "false").lower() == "true",
        fvg_min_gap_pct=float(os.getenv("HOGAN_FVG_MIN_GAP_PCT", "0.001")),
        signal_mode=os.getenv("HOGAN_SIGNAL_MODE", "any"),
        trailing_stop_pct=float(os.getenv("HOGAN_TRAILING_STOP_PCT", "0.0")),
        take_profit_pct=float(os.getenv("HOGAN_TAKE_PROFIT_PCT", "0.0")),
        ml_confidence_sizing=os.getenv("HOGAN_ML_CONFIDENCE_SIZING", "false").lower() == "true",
        webhook_url=os.getenv("HOGAN_WEBHOOK_URL", ""),
        exchange_id=os.getenv("HOGAN_EXCHANGE", "kraken"),
        retrain_window_bars=int(os.getenv("HOGAN_RETRAIN_WINDOW_BARS", "5000")),
        retrain_model_type=os.getenv("HOGAN_RETRAIN_MODEL_TYPE", "logreg"),
        retrain_min_improvement=float(os.getenv("HOGAN_RETRAIN_MIN_IMPROVEMENT", "0.005")),
        retrain_promotion_metric=os.getenv("HOGAN_RETRAIN_PROMOTION_METRIC", "roc_auc"),
        retrain_schedule_hours=float(os.getenv("HOGAN_RETRAIN_SCHEDULE_HOURS", "24.0")),
        kraken_api_key=os.getenv("KRAKEN_API_KEY"),
        kraken_api_secret=os.getenv("KRAKEN_API_SECRET"),
    )
