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

    # Paper mode gate relaxation: when True (and paper_mode=True), lower entry
    # thresholds by 20% to generate more trades for learning.  NEVER affects
    # live trading — the check is ``paper_mode and paper_relaxed_gates``.
    paper_relaxed_gates: bool = True

    # Persistence
    db_path: str = "data/hogan.db"

    # Live trading safety latch (must be true AND HOGAN_LIVE_ACK set)
    live_mode: bool = False

    # Short max-loss guardrail %: emergency exit if unrealized short loss
    # exceeds this fraction of entry value.  Default 10%.
    short_max_loss_pct: float = 0.10

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
    trailing_stop_pct: float = 0.030
    take_profit_pct: float = 0.054
    # Trailing stop activation: only start trailing after MFE reaches this %.
    # Prevents noise-triggered stops in the first bars after entry.
    trail_activation_pct: float = 0.005
    # Break-even stop: once MFE reaches this %, stop cannot fall below entry.
    # Protects winning trades from reversing into losses. 0 = disabled.
    breakeven_stop_pct: float = 0.015

    # ATR stop-distance multiplier (strategy.py line: ATR × multiplier)
    atr_stop_multiplier: float = 2.5

    # Maximum bars to hold a position before force-closing (0 = disabled).
    max_hold_bars: int = 24          # 24 bars on 1h = 24 hours

    # Cooldown bars after a losing trade before the next entry (0 = disabled).
    loss_cooldown_bars: int = 2      # 2 bars on 1h = 2 hours

    # Hour-based overrides (preferred): convert to bars using timeframe at runtime.
    # Ensures parity between backtest and live/paper across different timeframes.
    # Default 0 = disabled (falls back to max_hold_bars); canonical profile sets 24.0.
    max_hold_hours: float = 0
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
    # ATR-friction multiples for edge gate: ATR must exceed friction * multiple.
    # Buys use a lower multiple because longs benefit from mean-reversion in
    # low-vol periods.  Lowered from 0.5 after observing 100% buy-signal block
    # rate at 26 bps fee tier.
    buy_atr_friction_multiple: float = 0.25
    sell_atr_friction_multiple: float = 0.8

    # Entry quality gate thresholds (hard pre-trade filter)
    min_final_confidence: float = 0.25
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
    ml_confidence_sizing: bool = False
    # ML probability sizer: use ML probability as continuous position scale
    # instead of binary filter. Replaces both ml_filter and ml_confidence_sizing.
    use_ml_as_sizer: bool = True

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

    # Portfolio correlation: when holding a position in one symbol and entering
    # another highly correlated symbol, scale down the new position to avoid
    # doubling effective exposure (BTC/ETH ~0.85 correlation).
    portfolio_correlation_scale: float = 0.60

    # Auto-train forecast models on startup if pkl files are missing.
    # Safe to enable: only trains once if models don't exist, then persists.
    auto_train_forecast: bool = True

    # Phase 5A: Auto-apply weight proposals from PerformanceTracker.
    # After every 50 trades, if propose_weight_update() returns a proposal
    # with sufficient evidence, apply it to the MetaWeigher.  Bounded to
    # ±0.05 per agent per cycle as a safety guardrail.
    auto_apply_weights: bool = True
    auto_apply_min_trades: int = 30  # min trades in proposal evidence

    # Phase 5D: Forecast-driven position sizing.
    # Scale position size by forecast conviction relative to take profit target.
    # expected_return > 2× TP → 1.2× size; < 0.5× TP → 0.7× size.
    # Direction conflict → 0.5× size.
    forecast_driven_sizing: bool = True

    # Phase 5E: Walk-forward auto-retrain on schedule.
    # When True, checks model staleness every 4h of runtime and retrains
    # if model is older than retrain_schedule_hours and enough new candles.
    auto_retrain: bool = True
    auto_retrain_min_candles: int = 1000  # new candles since last retrain

    # Regime ensemble: blend per-regime ML models with standard prediction.
    # Requires a trained AdvancedEnsembleArtifact (see ml_advanced.py).
    use_regime_ensemble: bool = False
    regime_ensemble_blend: float = 0.30  # weight of regime-ensemble prediction
    regime_ensemble_path: str = "models/advanced_ensemble.pkl"

    # Online learning
    use_online_learning: bool = False
    online_learning_interval: int = 50
    use_learned_weights: bool = False

    # Multi-timeframe ensemble: daily bias + primary signal + 30m confirmation
    use_mtf_ensemble: bool = False
    mtf_timeframes: list[str] | None = field(default_factory=lambda: ["15m", "30m", "3h"])  # sub-hourly context frames
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

    # Unified decision core (policy_core.decide) — the canonical decision
    # path for both live and backtest.  The swarm decision layer only runs
    # through this path; setting False reverts to the legacy pipeline path
    # where the swarm is inert.
    use_policy_core: bool = True

    # Swarm Decision Layer
    swarm_enabled: bool = True
    swarm_mode: str = "conditional_active"
    swarm_phase: str = "certification"
    swarm_agents: str = "pipeline_v1,risk_steward_v1,data_guardian_v1,execution_cost_v1"
    swarm_min_agreement: float = 0.60
    swarm_min_vote_margin: float = 0.10
    swarm_max_entropy: float = 0.95
    swarm_weights: str = ""
    swarm_weight_update_mode: str = "shadow"
    swarm_weight_min_trades: int = 50
    swarm_weight_max_daily_shift: float = 0.05
    swarm_log_full_votes: bool = True
    swarm_use_regime_weights: bool = False
    swarm_weight_learning_enabled: bool = True   # Phase 5B: auto-learn swarm weights
    swarm_weight_learning_interval_bars: int = 24
    swarm_weight_auto_promote: bool = True       # Phase 5B: auto-promote when evidence sufficient
    swarm_conditional_min_agreement: float = 0.70
    swarm_conditional_min_confidence: float = 0.60

    # Swarm agent thresholds (configurable via env)
    swarm_risk_max_drawdown_pct: float = 0.10
    swarm_risk_vol_scale_threshold: float = 2.5
    swarm_risk_vol_veto_threshold: float = 4.0
    swarm_data_min_bars: int = 50
    swarm_data_max_gap_bars: int = 3
    swarm_data_max_stale_hours: float = 2.0
    swarm_exec_fee_rate: float = 0.0026
    swarm_exec_min_edge_ratio: float = 1.5

    swarm_replay_forward_window_bars: int = 12
    swarm_replay_bars_before: int = 60
    swarm_replay_bars_after: int = 60
    swarm_replay_positive_bps: float = 10.0
    swarm_replay_negative_bps: float = -10.0
    swarm_replay_strong_positive_bps: float = 20.0
    swarm_replay_strong_negative_bps: float = -20.0
    swarm_replay_enable_similar_events: bool = True

    # Swarm Daily Digest
    swarm_daily_digest_enabled: bool = True
    swarm_daily_digest_report_dir: str = "reports/digests"
    swarm_daily_digest_default_window_hours: int = 24
    swarm_daily_digest_notify: bool = False
    swarm_daily_digest_max_replay_candidates: int = 12
    swarm_daily_digest_stall_decision_min: int = 50
    swarm_daily_digest_stall_would_trade_max: int = 0
    swarm_daily_digest_critical_veto_ratio: float = 0.80
    swarm_daily_digest_warning_veto_ratio: float = 0.60
    swarm_daily_digest_min_regime_coverage: int = 3
    swarm_daily_digest_max_baseline_miss_ratio: float = 0.10
    swarm_daily_digest_max_import_error_count: int = 0

    # Swarm Weekly Review
    swarm_weekly_review_enabled: bool = True
    swarm_weekly_review_report_dir: str = "reports/weekly"
    swarm_weekly_review_default_window_days: int = 7
    swarm_weekly_review_notify: bool = False
    swarm_weekly_review_max_replay_candidates: int = 20
    swarm_weekly_review_min_decisions: int = 300
    swarm_weekly_review_min_would_trade: int = 100
    swarm_weekly_review_min_veto_events: int = 50
    swarm_weekly_review_min_regime_coverage: int = 3
    swarm_weekly_review_critical_veto_ratio: float = 0.80
    swarm_weekly_review_warning_veto_ratio: float = 0.60
    swarm_weekly_review_stall_zero_trade_decision_min: int = 50
    swarm_weekly_review_agent_dominance_ratio_warn: float = 0.70
    swarm_weekly_review_baseline_miss_ratio_warn: float = 0.10
    swarm_weekly_review_recommend_tuning_on_no_trade_ratio: float = 0.30

    # Swarm Threshold Tuning & Agent Quarantine
    swarm_stall_decision_min: int = 50
    swarm_stall_zero_trade_decision_min: int = 50
    swarm_stall_low_trade_decision_min: int = 100
    swarm_stall_low_trade_ratio: float = 0.05
    swarm_over_veto_ratio_warn: float = 0.70
    swarm_single_agent_veto_share_warn: float = 0.60
    swarm_threshold_require_manual_ack: bool = True
    swarm_threshold_ack_phrase: str = "I_APPROVE_THRESHOLD_CHANGE"
    swarm_default_operator: str = "local_operator"

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

    # ── Startup validation ───────────────────────────────────────────────
    def validate(self) -> list[str]:
        """Validate config values and return a list of error messages (empty = OK)."""
        errors: list[str] = []
        if self.starting_balance_usd <= 0:
            errors.append(f"starting_balance_usd must be > 0, got {self.starting_balance_usd}")
        if not self.symbols:
            errors.append("symbols list is empty — at least one trading pair required")
        for sym in self.symbols:
            if "/" not in sym:
                errors.append(f"symbol '{sym}' missing '/' separator (expected format: BTC/USD)")
        if self.max_drawdown <= 0 or self.max_drawdown > 1.0:
            errors.append(f"max_drawdown must be in (0, 1.0], got {self.max_drawdown}")
        if self.max_risk_per_trade <= 0 or self.max_risk_per_trade > 1.0:
            errors.append(f"max_risk_per_trade must be in (0, 1.0], got {self.max_risk_per_trade}")
        if self.trailing_stop_pct <= 0 or self.trailing_stop_pct > 0.5:
            errors.append(f"trailing_stop_pct must be in (0, 0.5], got {self.trailing_stop_pct}")
        if self.take_profit_pct <= 0 or self.take_profit_pct > 1.0:
            errors.append(f"take_profit_pct must be in (0, 1.0], got {self.take_profit_pct}")
        if self.fee_rate < 0:
            errors.append(f"fee_rate must be >= 0, got {self.fee_rate}")
        if self.ohlcv_limit < 50:
            errors.append(f"ohlcv_limit must be >= 50, got {self.ohlcv_limit}")
        if self.aggressive_allocation <= 0 or self.aggressive_allocation > 1.0:
            errors.append(f"aggressive_allocation must be in (0, 1.0], got {self.aggressive_allocation}")
        return errors


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
        long_size_scale=1.00,
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
        quality_final_mult=0.70,
        quality_tech_mult=1.00,
        allow_longs=True,
        allow_shorts=True,
        long_size_scale=0.40,
        short_size_scale=1.00,
    ),
    "ranging": RegimeConfig(
        volume_threshold_mult=1.10,
        ml_buy_threshold=0.58,
        ml_sell_threshold=0.42,
        trailing_stop_mult=0.90,
        take_profit_mult=0.85,
        position_scale=0.85,
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
        long_size_scale=0.50,
        short_size_scale=0.0,
    ),
    "volatile": RegimeConfig(
        volume_threshold_mult=0.70,
        ml_buy_threshold=0.57,
        ml_sell_threshold=0.43,
        trailing_stop_mult=0.80,
        take_profit_mult=1.40,
        position_scale=0.60,
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
        long_size_scale=0.50,
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

    Handles both real ``BotConfig`` dataclasses and
    ``types.SimpleNamespace`` configs used by the backtest engine.
    """
    overrides = load_symbol_overrides(symbol, base.timeframe)
    if not overrides:
        return base
    import dataclasses as _dc
    if _dc.is_dataclass(base):
        return replace(base, **overrides)
    import copy
    ns = copy.copy(base)
    for k, v in overrides.items():
        if hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def reload_symbol_configs() -> None:
    """Clear the override cache so the next call re-reads from disk."""
    _symbol_config_cache.clear()


def _env_float(var: str, default: str) -> float:
    raw = os.getenv(var, default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.error("Invalid env %s=%r (expected float, using default %s)", var, raw, default)
        return float(default)


def _env_int(var: str, default: str) -> int:
    raw = os.getenv(var, default)
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.error("Invalid env %s=%r (expected int, using default %s)", var, raw, default)
        return int(default)


def apply_sweep_results(config: "BotConfig", sweep_path: str = "diagnostics/exit_sweep_results.json") -> None:
    """Apply best exit parameters from a sweep results file (if present).

    The sweep file is a JSON list of dicts with keys like trailing_stop_pct,
    take_profit_pct, and a ranking metric (mean_return_pct).  This picks the
    row with the highest mean_return_pct and applies its exit parameters.
    """
    from pathlib import Path
    sweep_file = Path(sweep_path)
    if not sweep_file.exists():
        return
    try:
        import json
        results = json.loads(sweep_file.read_text(encoding="utf-8"))
        if not results:
            return
        best = max(results, key=lambda r: r.get("mean_return_pct", -999))
        _ts = best.get("trailing_stop_pct")
        _tp = best.get("take_profit_pct")
        if _ts is not None and _ts > 0:
            config.trailing_stop_pct = float(_ts)
        if _tp is not None and _tp > 0:
            config.take_profit_pct = float(_tp)
        logger.info(
            "Sweep results applied: trailing_stop=%.3f take_profit=%.3f (from %s)",
            config.trailing_stop_pct, config.take_profit_pct, sweep_path,
        )
    except Exception as exc:
        logger.debug("Failed to apply sweep results: %s", exc)


def load_config() -> BotConfig:
    """Load bot configuration from environment variables."""
    load_dotenv()
    return BotConfig(
        starting_balance_usd=_env_float("HOGAN_STARTING_BALANCE", "1800"),
        aggressive_allocation=_env_float("HOGAN_AGGRESSIVE_ALLOCATION", "0.75"),
        max_risk_per_trade=_env_float("HOGAN_MAX_RISK_PER_TRADE", "0.03"),
        max_drawdown=_env_float("HOGAN_MAX_DRAWDOWN", "0.15"),
        symbols=_split_symbols(os.getenv("HOGAN_SYMBOLS", "BTC/USD,ETH/USD")),
        timeframe=os.getenv("HOGAN_TIMEFRAME", "1h"),
        execution_timeframe=os.getenv("HOGAN_EXECUTION_TIMEFRAME", "15m"),
        ohlcv_limit=_env_int("HOGAN_OHLCV_LIMIT", "500"),
        short_ma_window=_env_int("HOGAN_SHORT_MA", "12"),
        long_ma_window=_env_int("HOGAN_LONG_MA", "79"),
        volume_window=_env_int("HOGAN_VOLUME_WINDOW", "20"),
        volume_threshold=_env_float("HOGAN_VOLUME_THRESHOLD", "1.8"),
        fee_rate=_env_float("HOGAN_FEE_RATE", "0.0026"),
        sleep_seconds=_env_int("HOGAN_SLEEP_SECONDS", "30"),
        trade_weekends=os.getenv("HOGAN_TRADE_WEEKENDS", "false").lower() == "true",
        paper_mode=os.getenv("HOGAN_PAPER_MODE", "true").lower() == "true",
        db_path=os.getenv("HOGAN_DB_PATH", "data/hogan.db"),
        live_mode=os.getenv("HOGAN_LIVE_MODE", "false").lower() == "true",
        short_max_loss_pct=_env_float("HOGAN_SHORT_MAX_LOSS_PCT", "0.10"),
        use_ml_filter=os.getenv("HOGAN_USE_ML_FILTER", "false").lower() == "true",
        ml_model_path=os.getenv("HOGAN_ML_MODEL_PATH", "models/hogan_logreg.pkl"),
        champion_ml_model_path=os.getenv("HOGAN_CHAMPION_ML_MODEL_PATH", "models/hogan_champion.pkl"),
        ml_buy_threshold=_env_float("HOGAN_ML_BUY_THRESHOLD", "0.55"),
        ml_sell_threshold=_env_float("HOGAN_ML_SELL_THRESHOLD", "0.45"),
        use_ema_clouds=os.getenv("HOGAN_USE_EMA_CLOUDS", "false").lower() == "true",
        ema_fast_short=_env_int("HOGAN_EMA_FAST_SHORT", "8"),
        ema_fast_long=_env_int("HOGAN_EMA_FAST_LONG", "9"),
        ema_slow_short=_env_int("HOGAN_EMA_SLOW_SHORT", "34"),
        ema_slow_long=_env_int("HOGAN_EMA_SLOW_LONG", "50"),
        use_fvg=os.getenv("HOGAN_USE_FVG", "false").lower() == "true",
        fvg_min_gap_pct=_env_float("HOGAN_FVG_MIN_GAP_PCT", "0.001"),
        signal_mode=os.getenv("HOGAN_SIGNAL_MODE", "any"),
        signal_min_vote_margin=max(1, _env_int("HOGAN_SIGNAL_MIN_VOTE_MARGIN", "1")),
        trailing_stop_pct=_env_float("HOGAN_TRAILING_STOP_PCT", "0.030"),
        take_profit_pct=_env_float("HOGAN_TAKE_PROFIT_PCT", "0.054"),
        trail_activation_pct=_env_float("HOGAN_TRAIL_ACTIVATION_PCT", "0.005"),
        breakeven_stop_pct=_env_float("HOGAN_BREAKEVEN_STOP_PCT", "0.015"),
        atr_stop_multiplier=_env_float("HOGAN_ATR_STOP_MULTIPLIER", "2.5"),
        exit_drawdown_pct=_env_float("HOGAN_EXIT_DRAWDOWN_PCT", "0.03"),
        exit_time_decay=_env_float("HOGAN_EXIT_TIME_DECAY", "0.75"),
        exit_vol_expansion=_env_float("HOGAN_EXIT_VOL_EXPANSION", "2.0"),
        exit_stagnation_bars=_env_int("HOGAN_EXIT_STAGNATION_BARS", "12"),
        max_hold_bars=_env_int("HOGAN_MAX_HOLD_BARS", "24"),
        loss_cooldown_bars=_env_int("HOGAN_LOSS_COOLDOWN_BARS", "2"),
        max_hold_hours=_env_float("HOGAN_MAX_HOLD_HOURS", "0"),
        short_max_hold_hours=_env_float("HOGAN_SHORT_MAX_HOLD_HOURS", "12"),
        loss_cooldown_hours=_env_float("HOGAN_LOSS_COOLDOWN_HOURS", "2"),
        min_hold_bars=_env_int("HOGAN_MIN_HOLD_BARS", "3"),
        exit_confirmation_bars=_env_int("HOGAN_EXIT_CONFIRM_BARS", "2"),
        min_edge_multiple=_env_float("HOGAN_MIN_EDGE_MULTIPLE", "1.5"),
        buy_atr_friction_multiple=_env_float("HOGAN_BUY_ATR_FRICTION_MULT", "0.25"),
        sell_atr_friction_multiple=_env_float("HOGAN_SELL_ATR_FRICTION_MULT", "0.8"),
        min_final_confidence=_env_float("HOGAN_MIN_FINAL_CONFIDENCE", "0.25"),
        min_tech_confidence=_env_float("HOGAN_MIN_TECH_CONFIDENCE", "0.15"),
        min_regime_confidence=_env_float("HOGAN_MIN_REGIME_CONFIDENCE", "0.30"),
        max_whipsaws=_env_int("HOGAN_MAX_WHIPSAWS", "3"),
        reversal_confidence_multiplier=_env_float("HOGAN_REVERSAL_CONFIDENCE_MULT", "1.3"),
        use_ict=os.getenv("HOGAN_USE_ICT", "false").lower() == "true",
        ict_model=os.getenv("HOGAN_ICT_MODEL", "silver_bullet"),
        ict_swing_left=_env_int("HOGAN_ICT_SWING_LEFT", "2"),
        ict_swing_right=_env_int("HOGAN_ICT_SWING_RIGHT", "2"),
        ict_eq_tolerance_pct=_env_float("HOGAN_ICT_EQ_TOLERANCE_PCT", "0.0008"),
        ict_min_displacement_pct=_env_float("HOGAN_ICT_MIN_DISPLACEMENT_PCT", "0.003"),
        ict_require_time_window=os.getenv("HOGAN_ICT_REQUIRE_TIME_WINDOW", "true").lower() == "true",
        ict_time_windows=os.getenv("HOGAN_ICT_TIME_WINDOWS", "03:00-04:00,10:00-11:00,14:00-15:00"),
        ict_require_pd=os.getenv("HOGAN_ICT_REQUIRE_PD", "true").lower() == "true",
        ict_ote_enabled=os.getenv("HOGAN_ICT_OTE_ENABLED", "false").lower() == "true",
        ict_ote_low=_env_float("HOGAN_ICT_OTE_LOW", "0.62"),
        ict_ote_high=_env_float("HOGAN_ICT_OTE_HIGH", "0.79"),
        ml_confidence_sizing=os.getenv("HOGAN_ML_CONFIDENCE_SIZING", "false").lower() == "true",
        use_ml_as_sizer=os.getenv("HOGAN_ML_AS_SIZER", "true").lower() == "true",
        allow_shorts=os.getenv("HOGAN_ALLOW_SHORTS", "false").lower() == "true",
        meta_weight_technical=_env_float("HOGAN_META_WEIGHT_TECH", "0.55"),
        meta_weight_sentiment=_env_float("HOGAN_META_WEIGHT_SENT", "0.25"),
        meta_weight_macro=_env_float("HOGAN_META_WEIGHT_MACRO", "0.20"),
        meta_buy_threshold=_env_float("HOGAN_META_BUY_THRESHOLD", "0.25"),
        meta_sell_threshold=_env_float("HOGAN_META_SELL_THRESHOLD", "-0.25"),
        use_regime_detection=os.getenv("HOGAN_USE_REGIME_DETECTION", "true").lower() == "true",
        regime_adx_trending=_env_float("HOGAN_REGIME_ADX_TRENDING", "25.0"),
        regime_adx_ranging=_env_float("HOGAN_REGIME_ADX_RANGING", "20.0"),
        regime_atr_volatile_pct=_env_float("HOGAN_REGIME_ATR_VOLATILE_PCT", "0.80"),
        use_strategy_router=os.getenv("HOGAN_USE_STRATEGY_ROUTER", "true").lower() == "true",
        volatile_policy=os.getenv("HOGAN_VOLATILE_POLICY", "breakout"),
        webhook_url=os.getenv("HOGAN_DISCORD_WEBHOOK_URL") or os.getenv("HOGAN_WEBHOOK_URL", ""),
        exchange_id=os.getenv("HOGAN_EXCHANGE", "kraken"),
        quote_currency=os.getenv("HOGAN_QUOTE_CCY", "USD"),
        metrics_port=_env_int("HOGAN_METRICS_PORT", "8000"),
        email_smtp_host=os.getenv("HOGAN_EMAIL_SMTP_HOST", ""),
        email_smtp_port=_env_int("HOGAN_EMAIL_SMTP_PORT", "587"),
        email_username=os.getenv("HOGAN_EMAIL_USERNAME", ""),
        email_password=os.getenv("HOGAN_EMAIL_PASSWORD", ""),
        email_from=os.getenv("HOGAN_EMAIL_FROM", ""),
        email_to=os.getenv("HOGAN_EMAIL_TO", ""),
        retrain_window_bars=_env_int("HOGAN_RETRAIN_WINDOW_BARS", "50000"),
        retrain_model_type=os.getenv("HOGAN_RETRAIN_MODEL_TYPE", "logreg"),
        retrain_min_improvement=_env_float("HOGAN_RETRAIN_MIN_IMPROVEMENT", "0.005"),
        retrain_promotion_metric=os.getenv("HOGAN_RETRAIN_PROMOTION_METRIC", "roc_auc"),
        retrain_schedule_hours=_env_float("HOGAN_RETRAIN_SCHEDULE_HOURS", "24.0"),
        auto_apply_weights=os.getenv("HOGAN_AUTO_APPLY_WEIGHTS", "true").lower() == "true",
        auto_apply_min_trades=_env_int("HOGAN_AUTO_APPLY_MIN_TRADES", "30"),
        forecast_driven_sizing=os.getenv("HOGAN_FORECAST_DRIVEN_SIZING", "true").lower() == "true",
        auto_retrain=os.getenv("HOGAN_AUTO_RETRAIN", "true").lower() == "true",
        auto_retrain_min_candles=_env_int("HOGAN_AUTO_RETRAIN_MIN_CANDLES", "1000"),
        training_symbols=_split_symbols(
            os.getenv("HOGAN_TRAINING_SYMBOLS", "BTC/USD,ETH/USD,SOL/USD")
        ),
        use_mtf_extended=os.getenv("HOGAN_USE_MTF_EXTENDED", "true").lower() == "true",
        use_mtf_ensemble=os.getenv("HOGAN_USE_MTF_ENSEMBLE", "false").lower() == "true",
        mtf_timeframes=os.getenv("HOGAN_MTF_TIMEFRAMES", "15m,30m,3h").split(","),
        mtf_use_daily_filter=os.getenv("HOGAN_MTF_USE_DAILY_FILTER", "false").lower() == "true",
        mtf_daily_timeframe=os.getenv("HOGAN_MTF_DAILY_TF", "1d"),
        mtf_m30_timeframe=os.getenv("HOGAN_MTF_M30_TF", "30m"),
        mtf_daily_fast_ma=_env_int("HOGAN_MTF_DAILY_FAST_MA", "10"),
        mtf_daily_slow_ma=_env_int("HOGAN_MTF_DAILY_SLOW_MA", "30"),
        mtf_unconfirmed_scale=_env_float("HOGAN_MTF_UNCONFIRMED_SCALE", "0.60"),
        use_online_learning=os.getenv("HOGAN_USE_ONLINE_LEARNING", "false").lower() == "true",
        use_learned_weights=os.getenv("HOGAN_USE_LEARNED_WEIGHTS", "false").lower() == "true",
        online_learning_interval=_env_int("HOGAN_ONLINE_LEARNING_INTERVAL", "50"),
        use_macro_filter=os.getenv("HOGAN_USE_MACRO_FILTER", "false").lower() == "true",
        macro_vix_caution=_env_float("HOGAN_MACRO_VIX_CAUTION", "25.0"),
        macro_vix_block=_env_float("HOGAN_MACRO_VIX_BLOCK", "35.0"),
        macro_equity_ma_period=_env_int("HOGAN_MACRO_EQUITY_MA", "20"),
        use_rl_agent=os.getenv("HOGAN_USE_RL_AGENT", "false").lower() == "true",
        use_policy_core=os.getenv("HOGAN_USE_POLICY_CORE", "true").lower() == "true",
        swarm_enabled=os.getenv("HOGAN_SWARM_ENABLED", "true").lower() == "true",
        swarm_mode=os.getenv("HOGAN_SWARM_MODE", "conditional_active"),
        swarm_phase=os.getenv("HOGAN_SWARM_PHASE", "certification"),
        swarm_agents=os.getenv("HOGAN_SWARM_AGENTS", "pipeline_v1,risk_steward_v1,data_guardian_v1,execution_cost_v1"),
        swarm_min_agreement=_env_float("HOGAN_SWARM_MIN_AGREEMENT", "0.60"),
        swarm_min_vote_margin=_env_float("HOGAN_SWARM_MIN_VOTE_MARGIN", "0.10"),
        swarm_max_entropy=_env_float("HOGAN_SWARM_MAX_ENTROPY", "0.95"),
        swarm_weights=os.getenv("HOGAN_SWARM_WEIGHTS", ""),
        swarm_weight_update_mode=os.getenv("HOGAN_SWARM_WEIGHT_UPDATE_MODE", "shadow"),
        swarm_weight_min_trades=_env_int("HOGAN_SWARM_WEIGHT_MIN_TRADES", "50"),
        swarm_weight_max_daily_shift=_env_float("HOGAN_SWARM_WEIGHT_MAX_DAILY_SHIFT", "0.05"),
        swarm_log_full_votes=os.getenv("HOGAN_SWARM_LOG_FULL_VOTES", "true").lower() == "true",
        swarm_use_regime_weights=os.getenv("HOGAN_SWARM_USE_REGIME_WEIGHTS", "false").lower() == "true",
        swarm_weight_learning_enabled=os.getenv("HOGAN_SWARM_WEIGHT_LEARNING", "true").lower() == "true",
        swarm_weight_learning_interval_bars=_env_int("HOGAN_SWARM_WEIGHT_LEARNING_INTERVAL", "24"),
        swarm_weight_auto_promote=os.getenv("HOGAN_SWARM_WEIGHT_AUTO_PROMOTE", "true").lower() == "true",
        swarm_conditional_min_agreement=_env_float("HOGAN_SWARM_CONDITIONAL_MIN_AGREEMENT", "0.70"),
        swarm_conditional_min_confidence=_env_float("HOGAN_SWARM_CONDITIONAL_MIN_CONFIDENCE", "0.60"),
        rl_model_path=os.getenv("HOGAN_RL_MODEL_PATH", "models/hogan_rl_policy.zip"),
        rl_reward_type=os.getenv("HOGAN_RL_REWARD_TYPE", "risk_adjusted"),
        rl_timesteps=_env_int("HOGAN_RL_TIMESTEPS", "200000"),
        kraken_api_key=os.getenv("KRAKEN_API_KEY"),
        kraken_api_secret=os.getenv("KRAKEN_API_SECRET"),
    )
