from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


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
    short_ma_window: int = 12
    long_ma_window: int = 79
    volume_window: int = 20
    volume_threshold: float = 1.8

    fee_rate: float = 0.0026
    sleep_seconds: int = 30
    trade_weekends: bool = False
    paper_mode: bool = True

    # Persistence
    db_path: str = "data/hogan.db"

    # Live trading safety latch (must be true AND HOGAN_LIVE_ACK set)
    live_mode: bool = False

    use_ml_filter: bool = False
    ml_model_path: str = "models/hogan_logreg.pkl"
    ml_buy_threshold: float = 0.60
    ml_sell_threshold: float = 0.40

    # Ripster EMA cloud settings (disabled: empirically hurts win rate as
    # cloud-based confirmation filters out good crossover trades)
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
    # Minimum directional vote edge required in "any" mode (buy_votes - sell_votes).
    # Default 1 = any majority. Set to 2 to require two more votes before trading.
    signal_min_vote_margin: int = 1

    # Exit management (0 = disabled)
    trailing_stop_pct: float = 0.02
    take_profit_pct: float = 0.054

    # ATR stop-distance multiplier (strategy.py line: ATR × multiplier)
    atr_stop_multiplier: float = 2.5

    # Maximum bars to hold a position before force-closing (0 = disabled).
    # 144 bars on 5m = 12 hours.
    max_hold_bars: int = 144

    # Cooldown bars after a losing trade before the next entry (0 = disabled).
    # 12 bars on 5m = 1 hour.
    loss_cooldown_bars: int = 12

    # ICT (Inner Circle Trader) signal pillars
    use_ict: bool = False
    ict_model: str = "silver_bullet"          # "silver_bullet" | "killzone"
    ict_swing_left: int = 2
    ict_swing_right: int = 2
    ict_eq_tolerance_pct: float = 0.0008
    ict_min_displacement_pct: float = 0.003
    ict_require_time_window: bool = True
    ict_time_windows: str = "03:00-04:00,10:00-11:00,14:00-15:00"
    ict_require_pd: bool = True
    ict_ote_enabled: bool = False
    ict_ote_low: float = 0.62
    ict_ote_high: float = 0.79

    # ML confidence-based position sizing: scales size by |prob−0.5|×2
    ml_confidence_sizing: bool = True

    # Short selling in paper mode: open a synthetic short when a SELL signal fires
    # with no existing long position.  Flip from short to long and back on signal change.
    allow_shorts: bool = False

    # ── Regime detection ─────────────────────────────────────────────────────
    # When enabled, the bot classifies the current market as trending_up,
    # trending_down, ranging, or volatile each iteration and dynamically
    # adjusts volume_threshold, ML thresholds, stop-loss, and position scale.
    use_regime_detection: bool = True
    regime_adx_trending: float = 25.0    # ADX ≥ this → trending
    regime_adx_ranging: float = 20.0     # ADX < this → ranging
    regime_atr_volatile_pct: float = 0.80  # ATR percentile ≥ this → volatile

    # Webhook URL for trade/drawdown notifications (empty string = disabled)
    webhook_url: str = ""

    # CCXT exchange ID — any of the 110+ exchanges in the library
    exchange_id: str = "kraken"

    # Walk-forward retraining defaults (used by hogan_bot.retrain)
    retrain_window_bars: int = 50000
    retrain_model_type: str = "logreg"
    retrain_min_improvement: float = 0.005
    retrain_promotion_metric: str = "roc_auc"
    retrain_schedule_hours: float = 24.0

    # Multi-symbol training: comma-separated symbols for joint model training.
    # When set, candles from all symbols are used to build a larger training set.
    # Example: "BTC/USD,ETH/USD,SOL/USD"
    training_symbols: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD", "SOL/USD"])

    # Extended MTF features: when True, includes 10m + 30m timeframe features
    # in build_feature_row_extended (+14 features vs standard 1h/15m only).
    # REQUIRES retraining with --force-promote before enabling in production.
    # Set HOGAN_USE_MTF_EXTENDED=true in .env after retraining.
    use_mtf_extended: bool = True

    # Online learning
    use_online_learning: bool = False
    online_learning_interval: int = 50

    # Reinforcement Learning agent
    use_rl_agent: bool = False
    rl_model_path: str = "models/hogan_rl_policy.zip"
    rl_reward_type: str = "risk_adjusted"
    rl_timesteps: int = 200_000

    # Account valuation currency for spot equity (USD, USDT, USDC, ...)
    quote_currency: str = "USD"

    # Monitoring
    metrics_port: int = 8000
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: str = ""

    kraken_api_key: str | None = None
    kraken_api_secret: str | None = None


def _split_symbols(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Per-symbol Optuna config overrides
# ---------------------------------------------------------------------------

_OPTUNA_OVERRIDE_FIELDS = frozenset({
    "short_ma_window", "long_ma_window", "volume_threshold",
    "atr_stop_multiplier", "use_ema_clouds", "signal_mode",
    "trailing_stop_pct", "take_profit_pct",
    "use_ict", "ict_swing_left", "ict_swing_right",
    "ict_eq_tolerance_pct", "ict_min_displacement_pct",
    "ict_require_time_window", "ict_require_pd", "ict_ote_enabled",
})

_symbol_config_cache: dict[str, dict] = {}


def _optuna_json_path(symbol: str, timeframe: str, models_dir: str = "models") -> Path:
    slug = symbol.replace("/", "-")
    return Path(models_dir) / f"opt_{slug}_{timeframe}.json"


def load_symbol_overrides(
    symbol: str,
    timeframe: str,
    models_dir: str = "models",
) -> dict:
    """Load the Optuna best_config for a symbol/timeframe, or empty dict if absent."""
    cache_key = f"{symbol}_{timeframe}"
    if cache_key in _symbol_config_cache:
        return _symbol_config_cache[cache_key]

    path = _optuna_json_path(symbol, timeframe, models_dir)
    if not path.exists():
        _symbol_config_cache[cache_key] = {}
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        best = data.get("best_config", {})
        overrides = {k: v for k, v in best.items() if k in _OPTUNA_OVERRIDE_FIELDS}
        _symbol_config_cache[cache_key] = overrides
        logger.info(
            "Loaded per-symbol config for %s/%s (%d overrides, sharpe=%.2f)",
            symbol, timeframe, len(overrides), data.get("best_score", 0),
        )
        return overrides
    except Exception as exc:
        logger.warning("Failed to load Optuna config from %s: %s", path, exc)
        _symbol_config_cache[cache_key] = {}
        return {}


def symbol_config(base: BotConfig, symbol: str) -> BotConfig:
    """Return a BotConfig with per-symbol Optuna overrides applied.

    Reads ``models/opt_{SYMBOL}_{TIMEFRAME}.json`` and overlays the
    ``best_config`` values onto a copy of *base*.  If no Optuna file
    exists for the symbol, returns *base* unchanged (no copy needed).
    """
    overrides = load_symbol_overrides(symbol, base.timeframe)
    if not overrides:
        return base
    return replace(base, **overrides)


def reload_symbol_configs() -> None:
    """Clear the override cache so the next call re-reads from disk."""
    _symbol_config_cache.clear()


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
        short_ma_window=int(os.getenv("HOGAN_SHORT_MA", "12")),
        long_ma_window=int(os.getenv("HOGAN_LONG_MA", "79")),
        volume_window=int(os.getenv("HOGAN_VOLUME_WINDOW", "20")),
        volume_threshold=float(os.getenv("HOGAN_VOLUME_THRESHOLD", "1.8")),
        fee_rate=float(os.getenv("HOGAN_FEE_RATE", "0.0026")),
        sleep_seconds=int(os.getenv("HOGAN_SLEEP_SECONDS", "30")),
        trade_weekends=os.getenv("HOGAN_TRADE_WEEKENDS", "false").lower() == "true",
        paper_mode=os.getenv("HOGAN_PAPER_MODE", "true").lower() == "true",
        db_path=os.getenv("HOGAN_DB_PATH", "data/hogan.db"),
        live_mode=os.getenv("HOGAN_LIVE_MODE", "false").lower() == "true",
        use_ml_filter=os.getenv("HOGAN_USE_ML_FILTER", "false").lower() == "true",
        ml_model_path=os.getenv("HOGAN_ML_MODEL_PATH", "models/hogan_logreg.pkl"),
        ml_buy_threshold=float(os.getenv("HOGAN_ML_BUY_THRESHOLD", "0.60")),
        ml_sell_threshold=float(os.getenv("HOGAN_ML_SELL_THRESHOLD", "0.40")),
        use_ema_clouds=os.getenv("HOGAN_USE_EMA_CLOUDS", "false").lower() == "true",
        ema_fast_short=int(os.getenv("HOGAN_EMA_FAST_SHORT", "8")),
        ema_fast_long=int(os.getenv("HOGAN_EMA_FAST_LONG", "9")),
        ema_slow_short=int(os.getenv("HOGAN_EMA_SLOW_SHORT", "34")),
        ema_slow_long=int(os.getenv("HOGAN_EMA_SLOW_LONG", "50")),
        use_fvg=os.getenv("HOGAN_USE_FVG", "false").lower() == "true",
        fvg_min_gap_pct=float(os.getenv("HOGAN_FVG_MIN_GAP_PCT", "0.001")),
        signal_mode=os.getenv("HOGAN_SIGNAL_MODE", "any"),
        signal_min_vote_margin=max(1, int(os.getenv("HOGAN_SIGNAL_MIN_VOTE_MARGIN", "1"))),
        trailing_stop_pct=float(os.getenv("HOGAN_TRAILING_STOP_PCT", "0.02")),
        take_profit_pct=float(os.getenv("HOGAN_TAKE_PROFIT_PCT", "0.054")),
        atr_stop_multiplier=float(os.getenv("HOGAN_ATR_STOP_MULTIPLIER", "2.5")),
        max_hold_bars=int(os.getenv("HOGAN_MAX_HOLD_BARS", "144")),
        loss_cooldown_bars=int(os.getenv("HOGAN_LOSS_COOLDOWN_BARS", "12")),
        use_ict=os.getenv("HOGAN_USE_ICT", "false").lower() == "true",
        ict_model=os.getenv("HOGAN_ICT_MODEL", "silver_bullet"),
        ict_swing_left=int(os.getenv("HOGAN_ICT_SWING_LEFT", "2")),
        ict_swing_right=int(os.getenv("HOGAN_ICT_SWING_RIGHT", "2")),
        ict_eq_tolerance_pct=float(os.getenv("HOGAN_ICT_EQ_TOLERANCE_PCT", "0.0008")),
        ict_min_displacement_pct=float(os.getenv("HOGAN_ICT_MIN_DISPLACEMENT_PCT", "0.003")),
        ict_require_time_window=os.getenv("HOGAN_ICT_REQUIRE_TIME_WINDOW", "true").lower() == "true",
        ict_time_windows=os.getenv("HOGAN_ICT_TIME_WINDOWS", "03:00-04:00,10:00-11:00,14:00-15:00"),
        ict_require_pd=os.getenv("HOGAN_ICT_REQUIRE_PD", "true").lower() == "true",
        ict_ote_enabled=os.getenv("HOGAN_ICT_OTE_ENABLED", "false").lower() == "true",
        ict_ote_low=float(os.getenv("HOGAN_ICT_OTE_LOW", "0.62")),
        ict_ote_high=float(os.getenv("HOGAN_ICT_OTE_HIGH", "0.79")),
        ml_confidence_sizing=os.getenv("HOGAN_ML_CONFIDENCE_SIZING", "true").lower() == "true",
        allow_shorts=os.getenv("HOGAN_ALLOW_SHORTS", "false").lower() == "true",
        use_regime_detection=os.getenv("HOGAN_USE_REGIME_DETECTION", "true").lower() == "true",
        regime_adx_trending=float(os.getenv("HOGAN_REGIME_ADX_TRENDING", "25.0")),
        regime_adx_ranging=float(os.getenv("HOGAN_REGIME_ADX_RANGING", "20.0")),
        regime_atr_volatile_pct=float(os.getenv("HOGAN_REGIME_ATR_VOLATILE_PCT", "0.80")),
        webhook_url=os.getenv("HOGAN_DISCORD_WEBHOOK_URL") or os.getenv("HOGAN_WEBHOOK_URL", ""),
        exchange_id=os.getenv("HOGAN_EXCHANGE", "kraken"),
        quote_currency=os.getenv("HOGAN_QUOTE_CCY", "USD"),
        metrics_port=int(os.getenv("HOGAN_METRICS_PORT", "8000")),
        email_smtp_host=os.getenv("HOGAN_EMAIL_SMTP_HOST", ""),
        email_smtp_port=int(os.getenv("HOGAN_EMAIL_SMTP_PORT", "587")),
        email_username=os.getenv("HOGAN_EMAIL_USERNAME", ""),
        email_password=os.getenv("HOGAN_EMAIL_PASSWORD", ""),
        email_from=os.getenv("HOGAN_EMAIL_FROM", ""),
        email_to=os.getenv("HOGAN_EMAIL_TO", ""),
        retrain_window_bars=int(os.getenv("HOGAN_RETRAIN_WINDOW_BARS", "50000")),
        retrain_model_type=os.getenv("HOGAN_RETRAIN_MODEL_TYPE", "logreg"),
        retrain_min_improvement=float(os.getenv("HOGAN_RETRAIN_MIN_IMPROVEMENT", "0.005")),
        retrain_promotion_metric=os.getenv("HOGAN_RETRAIN_PROMOTION_METRIC", "roc_auc"),
        retrain_schedule_hours=float(os.getenv("HOGAN_RETRAIN_SCHEDULE_HOURS", "24.0")),
        training_symbols=_split_symbols(
            os.getenv("HOGAN_TRAINING_SYMBOLS", "BTC/USD,ETH/USD,SOL/USD")
        ),
        use_mtf_extended=os.getenv("HOGAN_USE_MTF_EXTENDED", "true").lower() == "true",
        use_online_learning=os.getenv("HOGAN_USE_ONLINE_LEARNING", "false").lower() == "true",
        online_learning_interval=int(os.getenv("HOGAN_ONLINE_LEARNING_INTERVAL", "50")),
        use_rl_agent=os.getenv("HOGAN_USE_RL_AGENT", "false").lower() == "true",
        rl_model_path=os.getenv("HOGAN_RL_MODEL_PATH", "models/hogan_rl_policy.zip"),
        rl_reward_type=os.getenv("HOGAN_RL_REWARD_TYPE", "risk_adjusted"),
        rl_timesteps=int(os.getenv("HOGAN_RL_TIMESTEPS", "200000")),
        kraken_api_key=os.getenv("KRAKEN_API_KEY"),
        kraken_api_secret=os.getenv("KRAKEN_API_SECRET"),
    )
