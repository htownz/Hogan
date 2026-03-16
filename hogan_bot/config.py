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

    timeframe: str = "1h"
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
    champion_ml_model_path: str = "models/hogan_champion.pkl"
    ml_buy_threshold: float = 0.55
    ml_sell_threshold: float = 0.45

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
    # Set to 1: with most extra voters disabled (EMA clouds, FVG, ICT, RL),
    # only 1 voter (MA) remains; margin > 1 blocks all signals.
    signal_min_vote_margin: int = 1

    # Exit management (0 = disabled)
    trailing_stop_pct: float = 0.025
    take_profit_pct: float = 0.035

    # ATR stop-distance multiplier (strategy.py line: ATR × multiplier)
    atr_stop_multiplier: float = 2.5

    # Maximum bars to hold a position before force-closing (0 = disabled).
    max_hold_bars: int = 24          # 24 bars on 1h = 24 hours

    # Cooldown bars after a losing trade before the next entry (0 = disabled).
    loss_cooldown_bars: int = 2      # 2 bars on 1h = 2 hours

    # Hour-based overrides (preferred): convert to bars using timeframe at runtime.
    # Ensures parity between backtest and live/paper across different timeframes.
    max_hold_hours: float = 24.0     # 24h max hold (canonical)
    short_max_hold_hours: float = 12.0  # 12h from short-hold sweep (best Sharpe/return)
    loss_cooldown_hours: float = 2.0 # 2h cooldown (canonical)

    # Exit model thresholds (ExitEvaluator)
    exit_drawdown_pct: float = 0.03       # unrealized loss % triggering panic exit
    exit_time_decay: float = 0.75         # hold_ratio above which stale positions exit
    exit_vol_expansion: float = 2.0       # ATR ratio triggering vol-expansion exit
    exit_stagnation_bars: int = 12        # bars of near-zero PnL before stagnation exit

    # Conviction persistence: minimum bars to hold before signal exits are allowed.
    # Trailing stop / take profit / max_hold exits are unaffected.
    min_hold_bars: int = 3           # 3 bars on 1h = 3 hours

    # Exit confirmation: require N consecutive sell signals before a signal exit.
    exit_confirmation_bars: int = 2

    # Fee-aware entry gate: minimum multiple of round-trip fees (2 * fee_rate)
    # that the expected move (ATR or take_profit) must exceed before entry.
    min_edge_multiple: float = 1.5

    # Entry quality gate thresholds (hard pre-trade filter)
    min_final_confidence: float = 0.08
    min_tech_confidence: float = 0.15
    min_regime_confidence: float = 0.30
    max_whipsaws: int = 3

    # Signal-exit reversal asymmetry: require this multiple of entry confidence
    # to reverse (e.g., 1.3 = 30% stronger evidence needed to exit than to enter).
    reversal_confidence_multiplier: float = 1.3

    # Execution timeframe — used by the 15m execution model for entry/exit timing
    execution_timeframe: str = "15m"

    # ── EXPERIMENTAL: ICT (Inner Circle Trader) signal pillars ────────────
    # Quarantined from the champion path. Set HOGAN_USE_ICT=true to opt in.
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
    # ML probability sizer: use ML probability as continuous position scale
    # instead of binary filter. Replaces both ml_filter and ml_confidence_sizing.
    use_ml_as_sizer: bool = False

    # Short selling in paper mode: open a synthetic short when a SELL signal fires
    # with no existing long position.  Flip from short to long and back on signal change.
    allow_shorts: bool = False

    # ── MetaWeigher (agent pipeline) ───────────────────────────────────────
    meta_weight_technical: float = 0.55
    meta_weight_sentiment: float = 0.25
    meta_weight_macro: float = 0.20
    meta_buy_threshold: float = 0.25    # combined score ≥ this → buy
    meta_sell_threshold: float = -0.25  # combined score ≤ this → sell

    # ── Regime detection ─────────────────────────────────────────────────────
    # When enabled, the bot classifies the current market as trending_up,
    # trending_down, ranging, or volatile each iteration and dynamically
    # adjusts volume_threshold, ML thresholds, stop-loss, and position scale.
    use_regime_detection: bool = True
    regime_adx_trending: float = 25.0    # ADX ≥ this → trending
    regime_adx_ranging: float = 20.0     # ADX < this → ranging
    regime_atr_volatile_pct: float = 0.80  # ATR percentile ≥ this → volatile

    # Strategy router: when True, TechnicalAgent uses StrategyRouter to
    # dispatch to regime-specific strategy families instead of always using
    # generate_signal().  Requires use_regime_detection=True for full effect.
    use_strategy_router: bool = True

    # Policy for volatile regime: "breakout" to trade vol breakouts,
    # "hold" to sit out volatile markets entirely.
    volatile_policy: str = "breakout"

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
    use_learned_weights: bool = False

    # Multi-timeframe ensemble: daily bias + primary signal + 30m confirmation
    use_mtf_ensemble: bool = False
    mtf_timeframes: list[str] | None = None  # sub-hourly context frames e.g. ["15m", "30m"]
    mtf_use_daily_filter: bool = False   # enable after daily is Optuna-optimised
    mtf_daily_timeframe: str = "1d"
    mtf_m30_timeframe: str = "30m"
    mtf_daily_fast_ma: int = 10
    mtf_daily_slow_ma: int = 30
    mtf_unconfirmed_scale: float = 0.60

    # Macro correlation filter: SPY/DXY/VIX/Gold risk gates for BTC trades
    use_macro_filter: bool = False
    macro_vix_caution: float = 25.0      # VIX above this → reduce confidence
    macro_vix_block: float = 35.0        # VIX above this → block new longs
    macro_equity_ma_period: int = 20     # MA period for SPY/QQQ/GLD/UUP trend

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


@dataclass
class RegimeConfig:
    """Per-regime parameter overrides.

    Multiplier fields (``*_mult``) scale from the base BotConfig value.
    Absolute fields override directly.

    Responsibility boundaries:
    - MetaWeigher: direction and vote-level regime adaptation (meta_*_delta, meta_*_threshold)
    - entry_quality_gate: minimum setup cleanliness (quality_final_mult, quality_tech_mult)
    - ranging_gate: chop-specific suppression only
    - effective_thresholds: execution economics (ML gates, TP/SL, position_scale)
    """
    volume_threshold_mult: float = 1.0
    ml_buy_threshold: float = 0.55
    ml_sell_threshold: float = 0.45
    trailing_stop_mult: float = 1.0
    take_profit_mult: float = 1.0
    position_scale: float = 1.0
    strategy_family: str = "trend_follow"
    min_confidence_to_trade: float = 0.30
    meta_tech_delta: float = 0.0
    meta_sent_delta: float = 0.0
    meta_macro_delta: float = 0.0
    meta_buy_threshold: float | None = None
    meta_sell_threshold: float | None = None
    # Quality gate: multipliers applied to min_final_confidence / min_tech_confidence
    quality_final_mult: float = 1.0
    quality_tech_mult: float = 1.0

    # Side-specific participation controls
    allow_longs: bool = True
    allow_shorts: bool = True
    long_size_scale: float = 1.0
    short_size_scale: float = 1.0


DEFAULT_REGIME_CONFIGS: dict[str, RegimeConfig] = {
    "trending_up": RegimeConfig(
        volume_threshold_mult=0.55,
        ml_buy_threshold=0.53,
        ml_sell_threshold=0.47,
        trailing_stop_mult=1.30,
        take_profit_mult=2.00,
        position_scale=1.00,
        strategy_family="trend_follow",
        meta_tech_delta=+0.10,
        meta_sent_delta=-0.05,
        meta_macro_delta=-0.05,
        meta_buy_threshold=0.12,
        meta_sell_threshold=-0.12,
        quality_final_mult=0.80,
        quality_tech_mult=1.00,
        allow_longs=True,
        allow_shorts=False,
        long_size_scale=0.25,
        short_size_scale=0.0,
    ),
    "trending_down": RegimeConfig(
        volume_threshold_mult=0.55,
        ml_buy_threshold=0.57,
        ml_sell_threshold=0.43,
        trailing_stop_mult=1.50,
        take_profit_mult=1.70,
        position_scale=1.00,
        strategy_family="trend_follow",
        meta_tech_delta=+0.10,
        meta_sent_delta=-0.05,
        meta_macro_delta=-0.05,
        meta_buy_threshold=0.12,
        meta_sell_threshold=-0.12,
        quality_final_mult=0.50,
        quality_tech_mult=1.00,
        allow_longs=True,
        allow_shorts=True,
        long_size_scale=0.5,
        short_size_scale=1.0,
    ),
    "ranging": RegimeConfig(
        volume_threshold_mult=1.10,
        ml_buy_threshold=0.58,
        ml_sell_threshold=0.42,
        trailing_stop_mult=1.20,
        take_profit_mult=0.85,
        position_scale=0.75,
        strategy_family="mean_revert",
        meta_tech_delta=-0.05,
        meta_sent_delta=+0.00,
        meta_macro_delta=+0.05,
        meta_buy_threshold=0.15,
        meta_sell_threshold=-0.15,
        quality_final_mult=1.00,
        quality_tech_mult=1.25,
        allow_longs=True,
        allow_shorts=False,
        long_size_scale=0.25,
        short_size_scale=0.0,
    ),
    "volatile": RegimeConfig(
        volume_threshold_mult=0.70,
        ml_buy_threshold=0.57,
        ml_sell_threshold=0.43,
        trailing_stop_mult=0.80,
        take_profit_mult=1.40,
        position_scale=0.50,
        strategy_family="breakout",
        meta_tech_delta=-0.05,
        meta_sent_delta=+0.00,
        meta_macro_delta=+0.05,
        meta_buy_threshold=0.18,
        meta_sell_threshold=-0.18,
        quality_final_mult=1.20,
        quality_tech_mult=1.10,
        allow_longs=True,
        allow_shorts=True,
        long_size_scale=0.25,
        short_size_scale=0.50,
    ),
}


def _split_symbols(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def effective_hold_cooldown_bars(config: BotConfig, timeframe: str) -> tuple[int, int]:
    """Return (max_hold_bars, loss_cooldown_bars) for the given timeframe.

    When max_hold_hours or loss_cooldown_hours are set, converts hours to bars
    for parity between backtest and live/paper across timeframes.
    """
    from hogan_bot.timeframe_utils import hours_to_bars
    if config.max_hold_hours > 0:
        max_hold = hours_to_bars(config.max_hold_hours, timeframe)
    else:
        max_hold = config.max_hold_bars
    if config.loss_cooldown_hours > 0:
        cooldown = hours_to_bars(config.loss_cooldown_hours, timeframe)
    else:
        cooldown = config.loss_cooldown_bars
    return max_hold, cooldown


def effective_short_max_hold_bars(config: BotConfig, timeframe: str) -> int:
    """Return short_max_hold_bars for the given timeframe.

    Uses ``short_max_hold_hours`` when > 0, otherwise falls back to
    the long ``max_hold_bars`` (no separate short hold).
    """
    from hogan_bot.timeframe_utils import hours_to_bars
    if config.short_max_hold_hours > 0:
        return hours_to_bars(config.short_max_hold_hours, timeframe)
    max_hold, _ = effective_hold_cooldown_bars(config, timeframe)
    return max_hold


# ---------------------------------------------------------------------------
# Per-symbol Optuna config overrides
# ---------------------------------------------------------------------------

_OPTUNA_OVERRIDE_FIELDS = frozenset({
    "short_ma_window", "long_ma_window", "volume_threshold",
    "atr_stop_multiplier", "use_ema_clouds", "signal_mode",
    "trailing_stop_pct", "take_profit_pct",
})

# ICT overrides only applied when use_ict=True (experimental)
_OPTUNA_EXPERIMENTAL_FIELDS = frozenset({
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
        allowed = _OPTUNA_OVERRIDE_FIELDS | _OPTUNA_EXPERIMENTAL_FIELDS
        overrides = {k: v for k, v in best.items() if k in allowed}
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
        timeframe=os.getenv("HOGAN_TIMEFRAME", "1h"),
        execution_timeframe=os.getenv("HOGAN_EXECUTION_TIMEFRAME", "15m"),
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
        champion_ml_model_path=os.getenv("HOGAN_CHAMPION_ML_MODEL_PATH", "models/hogan_champion.pkl"),
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
        signal_min_vote_margin=max(1, int(os.getenv("HOGAN_SIGNAL_MIN_VOTE_MARGIN", "1"))),
        trailing_stop_pct=float(os.getenv("HOGAN_TRAILING_STOP_PCT", "0.02")),
        take_profit_pct=float(os.getenv("HOGAN_TAKE_PROFIT_PCT", "0.054")),
        atr_stop_multiplier=float(os.getenv("HOGAN_ATR_STOP_MULTIPLIER", "2.5")),
        exit_drawdown_pct=float(os.getenv("HOGAN_EXIT_DRAWDOWN_PCT", "0.03")),
        exit_time_decay=float(os.getenv("HOGAN_EXIT_TIME_DECAY", "0.75")),
        exit_vol_expansion=float(os.getenv("HOGAN_EXIT_VOL_EXPANSION", "2.0")),
        exit_stagnation_bars=int(os.getenv("HOGAN_EXIT_STAGNATION_BARS", "12")),
        max_hold_bars=int(os.getenv("HOGAN_MAX_HOLD_BARS", "24")),
        loss_cooldown_bars=int(os.getenv("HOGAN_LOSS_COOLDOWN_BARS", "2")),
        max_hold_hours=float(os.getenv("HOGAN_MAX_HOLD_HOURS", "0")),
        short_max_hold_hours=float(os.getenv("HOGAN_SHORT_MAX_HOLD_HOURS", "12")),
        loss_cooldown_hours=float(os.getenv("HOGAN_LOSS_COOLDOWN_HOURS", "2")),
        min_hold_bars=int(os.getenv("HOGAN_MIN_HOLD_BARS", "3")),
        exit_confirmation_bars=int(os.getenv("HOGAN_EXIT_CONFIRM_BARS", "2")),
        min_edge_multiple=float(os.getenv("HOGAN_MIN_EDGE_MULTIPLE", "1.5")),
        min_final_confidence=float(os.getenv("HOGAN_MIN_FINAL_CONFIDENCE", "0.25")),
        min_tech_confidence=float(os.getenv("HOGAN_MIN_TECH_CONFIDENCE", "0.15")),
        min_regime_confidence=float(os.getenv("HOGAN_MIN_REGIME_CONFIDENCE", "0.30")),
        max_whipsaws=int(os.getenv("HOGAN_MAX_WHIPSAWS", "3")),
        reversal_confidence_multiplier=float(os.getenv("HOGAN_REVERSAL_CONFIDENCE_MULT", "1.3")),
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
        use_ml_as_sizer=os.getenv("HOGAN_ML_AS_SIZER", "false").lower() == "true",
        allow_shorts=os.getenv("HOGAN_ALLOW_SHORTS", "false").lower() == "true",
        meta_weight_technical=float(os.getenv("HOGAN_META_WEIGHT_TECH", "0.55")),
        meta_weight_sentiment=float(os.getenv("HOGAN_META_WEIGHT_SENT", "0.25")),
        meta_weight_macro=float(os.getenv("HOGAN_META_WEIGHT_MACRO", "0.20")),
        meta_buy_threshold=float(os.getenv("HOGAN_META_BUY_THRESHOLD", "0.25")),
        meta_sell_threshold=float(os.getenv("HOGAN_META_SELL_THRESHOLD", "-0.25")),
        use_regime_detection=os.getenv("HOGAN_USE_REGIME_DETECTION", "true").lower() == "true",
        regime_adx_trending=float(os.getenv("HOGAN_REGIME_ADX_TRENDING", "25.0")),
        regime_adx_ranging=float(os.getenv("HOGAN_REGIME_ADX_RANGING", "20.0")),
        regime_atr_volatile_pct=float(os.getenv("HOGAN_REGIME_ATR_VOLATILE_PCT", "0.80")),
        use_strategy_router=os.getenv("HOGAN_USE_STRATEGY_ROUTER", "true").lower() == "true",
        volatile_policy=os.getenv("HOGAN_VOLATILE_POLICY", "breakout"),
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
        use_mtf_ensemble=os.getenv("HOGAN_USE_MTF_ENSEMBLE", "false").lower() == "true",
        mtf_timeframes=os.getenv("HOGAN_MTF_TIMEFRAMES", "").split(",") if os.getenv("HOGAN_MTF_TIMEFRAMES") else None,
        mtf_use_daily_filter=os.getenv("HOGAN_MTF_USE_DAILY_FILTER", "false").lower() == "true",
        mtf_daily_timeframe=os.getenv("HOGAN_MTF_DAILY_TF", "1d"),
        mtf_m30_timeframe=os.getenv("HOGAN_MTF_M30_TF", "30m"),
        mtf_daily_fast_ma=int(os.getenv("HOGAN_MTF_DAILY_FAST_MA", "10")),
        mtf_daily_slow_ma=int(os.getenv("HOGAN_MTF_DAILY_SLOW_MA", "30")),
        mtf_unconfirmed_scale=float(os.getenv("HOGAN_MTF_UNCONFIRMED_SCALE", "0.60")),
        use_online_learning=os.getenv("HOGAN_USE_ONLINE_LEARNING", "false").lower() == "true",
        use_learned_weights=os.getenv("HOGAN_USE_LEARNED_WEIGHTS", "false").lower() == "true",
        online_learning_interval=int(os.getenv("HOGAN_ONLINE_LEARNING_INTERVAL", "50")),
        use_macro_filter=os.getenv("HOGAN_USE_MACRO_FILTER", "false").lower() == "true",
        macro_vix_caution=float(os.getenv("HOGAN_MACRO_VIX_CAUTION", "25.0")),
        macro_vix_block=float(os.getenv("HOGAN_MACRO_VIX_BLOCK", "35.0")),
        macro_equity_ma_period=int(os.getenv("HOGAN_MACRO_EQUITY_MA", "20")),
        use_rl_agent=os.getenv("HOGAN_USE_RL_AGENT", "false").lower() == "true",
        rl_model_path=os.getenv("HOGAN_RL_MODEL_PATH", "models/hogan_rl_policy.zip"),
        rl_reward_type=os.getenv("HOGAN_RL_REWARD_TYPE", "risk_adjusted"),
        rl_timesteps=int(os.getenv("HOGAN_RL_TIMESTEPS", "200000")),
        kraken_api_key=os.getenv("KRAKEN_API_KEY"),
        kraken_api_secret=os.getenv("KRAKEN_API_SECRET"),
    )
