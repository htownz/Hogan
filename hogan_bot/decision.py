from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality Components — structured setup-quality decomposition
# ---------------------------------------------------------------------------

@dataclass
class QualityComponents:
    """Multi-dimensional setup quality score for audit trail."""
    final_conf: float = 0.0
    tech_conf: float = 0.0
    regime_conf: float = 0.0
    ml_separation: float = 1.0
    spread_penalty: float = 1.0
    whipsaw_penalty: float = 1.0
    freshness_penalty: float = 1.0
    overall: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def compute_quality_components(
    *,
    final_confidence: float | None = None,
    tech_confidence: float | None = None,
    regime_confidence: float | None = None,
    up_prob: float | None = None,
    estimated_spread: float | None = None,
    atr_pct: float | None = None,
    recent_whipsaw_count: int = 0,
    freshness_summary: dict | None = None,
    ranging_scale: float = 1.0,
    pullback_scale: float = 1.0,
    quality_gate_scale: float = 1.0,
) -> QualityComponents:
    """Compute a structured quality score from signal evaluation inputs."""
    final_conf = min(1.0, max(0.0, (final_confidence or 0.0) / 0.6))
    tech_conf = min(1.0, max(0.0, (tech_confidence or 0.0) / 0.7))
    regime_conf = min(1.0, max(0.0, (regime_confidence or 0.0) / 0.75))

    if up_prob is not None:
        ml_separation = min(1.0, abs(up_prob - 0.5) / 0.25)
    else:
        ml_separation = 1.0

    if estimated_spread is not None and atr_pct is not None and atr_pct > 1e-9:
        spread_ratio = estimated_spread / atr_pct
        spread_penalty = max(0.0, 1.0 - min(1.0, spread_ratio / 0.25))
    else:
        spread_penalty = 1.0

    whipsaw_penalty = max(0.0, 1.0 - recent_whipsaw_count * 0.2)

    freshness_penalty = 1.0
    if freshness_summary:
        stale_count = freshness_summary.get("stale_count", 0)
        critical_stale = freshness_summary.get("critical_stale_count", 0)
        freshness_penalty -= min(0.5, stale_count * 0.05 + critical_stale * 0.15)
        freshness_penalty = max(0.0, freshness_penalty)

    overall = (
        0.20 * final_conf
        + 0.15 * tech_conf
        + 0.15 * regime_conf
        + 0.15 * ml_separation
        + 0.15 * spread_penalty
        + 0.10 * whipsaw_penalty
        + 0.10 * freshness_penalty
    ) * ranging_scale * pullback_scale * quality_gate_scale

    return QualityComponents(
        final_conf=round(final_conf, 4),
        tech_conf=round(tech_conf, 4),
        regime_conf=round(regime_conf, 4),
        ml_separation=round(ml_separation, 4),
        spread_penalty=round(spread_penalty, 4),
        whipsaw_penalty=round(whipsaw_penalty, 4),
        freshness_penalty=round(freshness_penalty, 4),
        overall=round(overall, 4),
    )


# ---------------------------------------------------------------------------
# Gate Decision — structured gate output with "why blocked" attribution
# ---------------------------------------------------------------------------

@dataclass
class GateDecision:
    """Structured output from any signal gate."""
    action: str
    size_scale: float = 1.0
    blocked_by: str | None = None
    detail: dict | None = None


# ---------------------------------------------------------------------------
# Spread estimation
# ---------------------------------------------------------------------------

def estimate_spread_from_candles(candles: pd.DataFrame, window: int = 20) -> float:
    """Estimate the bid-ask spread as a fraction of price from OHLCV data.

    Uses the Corwin-Schultz (2012) high-low spread estimator: the ratio of
    high-to-low range in a single bar versus the range over two consecutive
    bars reveals the non-information component of the spread.

    Falls back to a simpler (high-low)/close proxy when there are not enough
    bars.

    Returns
    -------
    float
        Estimated half-spread as a fraction of mid-price (e.g. 0.001 = 10 bps
        round-trip).  Multiply by 2 for the full round-trip spread cost.
    """
    if candles is None or len(candles) < 3:
        logger.debug("SPREAD_EST: insufficient data, using default 5bps")
        return 0.0005

    h = candles["high"].values[-window:]
    lo = candles["low"].values[-window:]
    c = candles["close"].values[-window:]

    if len(h) < 3:
        avg_range = np.mean((h - lo) / np.maximum(c, 1e-9))
        return max(0.0001, float(avg_range * 0.25))

    # Simple range-based proxy: spread ≈ small fraction of avg bar range
    avg_range_pct = float(np.mean((h - lo) / np.maximum(c, 1e-9)))
    simple_est = avg_range_pct * 0.15

    # Corwin-Schultz high-low estimator
    hl_ratio = np.log(h / np.maximum(lo, 1e-9))
    gamma = hl_ratio[:-1] ** 2 + hl_ratio[1:] ** 2

    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(lo[:-1], lo[1:])
    beta = np.log(h2 / np.maximum(l2, 1e-9)) ** 2

    valid = beta > gamma
    if not np.any(valid):
        return max(0.0001, min(simple_est, 0.001))

    alpha = (np.sqrt(2 * beta[valid]) - np.sqrt(gamma[valid])) / (
        3 - 2 * np.sqrt(2)
    )
    alpha = np.clip(alpha, 0, None)
    spread = 2.0 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    cs_est = float(np.median(spread))

    # Blend: take the smaller of the two estimates (CS can overestimate on
    # synthetic or noisy data where high/low are not real bid-ask artifacts).
    # Cap at 10 bps: hourly OHLCV conflates volatility with spread, and major
    # crypto pairs on Kraken/Coinbase typically have ~3-5 bps half-spread.
    est = min(cs_est, simple_est * 2)
    return max(0.0001, min(est, 0.001))


def estimate_spread_from_order_book(
    bids: list[list[float]], asks: list[list[float]]
) -> float:
    """Compute the quoted spread from top-of-book bid/ask.

    Parameters
    ----------
    bids, asks
        Lists of [price, qty] pairs from the order book.

    Returns
    -------
    float
        Half-spread as a fraction of mid-price.
    """
    if not bids or not asks:
        return 0.001
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return 0.001
    return max(0.0, (best_ask - best_bid) / (2.0 * mid))


def apply_ml_filter(signal_action: str, up_prob: float, buy_threshold: float, sell_threshold: float) -> GateDecision:
    if up_prob is None or np.isnan(up_prob):
        return GateDecision(action=signal_action)
    if signal_action == "buy" and up_prob < buy_threshold:
        return GateDecision(
            action="hold", blocked_by="ml_filter_buy",
            detail={"up_prob": float(up_prob), "threshold": buy_threshold},
        )
    if signal_action == "sell" and up_prob > sell_threshold:
        return GateDecision(
            action="hold", blocked_by="ml_filter_sell",
            detail={"up_prob": float(up_prob), "threshold": sell_threshold},
        )
    return GateDecision(action=signal_action)


def edge_gate(
    action: str,
    atr_pct: float,
    take_profit_pct: float,
    fee_rate: float,
    min_edge_multiple: float = 1.5,
    forecast_expected_return: float | None = None,
    estimated_spread: float = 0.0,
    atr_friction_multiple: float = 0.8,
    buy_atr_friction_multiple: float = 0.25,
) -> GateDecision:
    """Block entries where the expected move is insufficient relative to friction.

    Total friction = round-trip fees + round-trip spread.

    Asymmetric: buys use a lower ATR friction multiple (0.5x) than sells (0.8x)
    because the long side benefits from mean-reversion in low-vol periods, while
    the short side does not.

    Checks four conditions (any failure -> hold):
    1. ATR must exceed atr_friction_multiple * total friction (enough volatility)
    2. Take-profit target must exceed min_edge_multiple * total friction
    3. If forecast available, expected return must exceed total friction
    4. If spread alone exceeds ATR/3, conditions are too illiquid
    """
    if action == "hold":
        return GateDecision(action=action)

    round_trip_fees = 2.0 * fee_rate
    round_trip_spread = 2.0 * estimated_spread
    total_friction = round_trip_fees + round_trip_spread

    _eff_atr_mult = buy_atr_friction_multiple if action == "buy" else atr_friction_multiple
    if atr_pct < total_friction * _eff_atr_mult:
        logger.debug(
            "EDGE_GATE: ATR %.4f < %.4f (%.1fx friction: fees=%.4f spread=%.4f) -> hold",
            atr_pct, total_friction * _eff_atr_mult,
            _eff_atr_mult, round_trip_fees, round_trip_spread,
        )
        return GateDecision(
            action="hold", blocked_by="edge_gate_atr_low",
            detail={"atr_pct": atr_pct, "required": total_friction * _eff_atr_mult},
        )

    if take_profit_pct > 0 and take_profit_pct < total_friction * min_edge_multiple:
        logger.debug(
            "EDGE_GATE: TP %.4f < %.4f (%.1fx friction) -> hold",
            take_profit_pct, total_friction * min_edge_multiple, min_edge_multiple,
        )
        return GateDecision(
            action="hold", blocked_by="edge_gate_tp_low",
            detail={"tp": take_profit_pct, "required": total_friction * min_edge_multiple},
        )

    if forecast_expected_return is not None and abs(forecast_expected_return) < total_friction:
        logger.debug(
            "EDGE_GATE: forecast_ret %.4f < %.4f (friction) -> hold",
            abs(forecast_expected_return), total_friction,
        )
        return GateDecision(
            action="hold", blocked_by="edge_gate_forecast_low",
            detail={"forecast": abs(forecast_expected_return), "friction": total_friction},
        )

    if estimated_spread > 0 and atr_pct > 0 and estimated_spread > atr_pct / 3:
        logger.debug(
            "EDGE_GATE: spread %.4f > ATR/3 %.4f (illiquid) -> hold",
            estimated_spread, atr_pct / 3,
        )
        return GateDecision(
            action="hold", blocked_by="edge_gate_illiquid",
            detail={"spread": estimated_spread, "atr_third": atr_pct / 3},
        )

    return GateDecision(action=action)


_REGIME_QUALITY_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "trending_up":   {"final_mult": 0.80, "tech_mult": 1.00},
    "trending_down": {"final_mult": 0.80, "tech_mult": 1.00},
    "volatile":      {"final_mult": 1.20, "tech_mult": 1.10},
    "ranging":       {"final_mult": 1.00, "tech_mult": 1.25},
}


def get_regime_quality_adjustments(regime: str | None) -> dict[str, float]:
    """Get quality gate multipliers for a regime.

    Prefers values from RegimeConfig (if available), falling back to
    the static dict above. This bridges the transition from hardcoded
    to config-driven adjustments.
    """
    if not regime:
        return {}
    try:
        from hogan_bot.config import DEFAULT_REGIME_CONFIGS
        rc = DEFAULT_REGIME_CONFIGS.get(regime)
        if rc is not None:
            return {"final_mult": rc.quality_final_mult, "tech_mult": rc.quality_tech_mult}
    except Exception as exc:
        logger.warning("Regime quality config lookup failed for %s (using hardcoded fallback): %s", regime, exc)
    return _REGIME_QUALITY_ADJUSTMENTS.get(regime, {})


def entry_quality_gate(
    action: str,
    *,
    final_confidence: float | None = None,
    tech_confidence: float | None = None,
    regime: str | None = None,
    regime_confidence: float | None = None,
    recent_whipsaw_count: int = 0,
    min_final_confidence: float = 0.3,
    min_tech_confidence: float = 0.4,
    min_regime_confidence: float = 0.5,
    max_whipsaws: int = 3,
) -> GateDecision:
    """Block entries that don't meet quality thresholds.

    Thresholds are adjusted by regime:
    - Trending: relax final_confidence by 20% (trend-following needs less confirmation)
    - Volatile: tighten final_confidence by 20% (require stronger conviction)
    - Ranging: tighten tech_confidence by 25% (mean-revert needs stronger tech signal)
    """
    if action == "hold":
        return GateDecision(action=action)

    adj = get_regime_quality_adjustments(regime)
    eff_min_final = min_final_confidence * adj.get("final_mult", 1.0)
    eff_min_tech = min_tech_confidence * adj.get("tech_mult", 1.0)

    if final_confidence is not None and final_confidence < eff_min_final:
        logger.debug(
            "QUALITY_GATE: final_conf %.3f < %.3f (regime=%s) -> hold",
            final_confidence, eff_min_final, regime or "none",
        )
        return GateDecision(
            action="hold", blocked_by="quality_gate_final_conf",
            detail={"value": final_confidence, "required": eff_min_final, "regime": regime},
        )

    if tech_confidence is not None and tech_confidence < eff_min_tech:
        logger.debug(
            "QUALITY_GATE: tech_conf %.3f < %.3f (regime=%s) -> hold",
            tech_confidence, eff_min_tech, regime or "none",
        )
        return GateDecision(
            action="hold", blocked_by="quality_gate_tech_conf",
            detail={"value": tech_confidence, "required": eff_min_tech, "regime": regime},
        )

    if regime_confidence is not None and regime_confidence < min_regime_confidence:
        logger.debug(
            "QUALITY_GATE: regime_conf %.3f < %.3f -> hold",
            regime_confidence, min_regime_confidence,
        )
        return GateDecision(
            action="hold", blocked_by="quality_gate_regime_conf",
            detail={"value": regime_confidence, "required": min_regime_confidence},
        )

    if recent_whipsaw_count >= max_whipsaws:
        scale = max(0.25, 1.0 - (recent_whipsaw_count - max_whipsaws + 1) * 0.25)
        logger.debug(
            "QUALITY_GATE: whipsaws %d >= %d -> scale %.2f",
            recent_whipsaw_count, max_whipsaws, scale,
        )
        if scale <= 0.25:
            return GateDecision(
                action="hold", blocked_by="quality_gate_whipsaw",
                detail={"whipsaws": recent_whipsaw_count, "max": max_whipsaws},
            )
        return GateDecision(action=action, size_scale=scale)

    return GateDecision(action=action)


def ranging_gate(
    action: str,
    *,
    regime: str | None = None,
    tech_action: str | None = None,
    up_prob: float | None = None,
    recent_whipsaw_count: int = 0,
    ml_separation_min: float = 0.06,
    whipsaw_block_threshold: int = 2,
    soft_mode: bool = True,
    buy_ml_separation_min: float = 0.045,
    buy_whipsaw_block_threshold: int = 3,
) -> GateDecision:
    """Extra protections for ranging markets.

    Only active when regime == "ranging". When ``soft_mode=True`` (default),
    near-miss checks shrink size to 0.5x instead of always blocking.

    Asymmetric: buys get relaxed thresholds (lower ML separation, higher
    whipsaw threshold, always soft) because the long side has historically
    been over-filtered in ranging regimes. Sells keep strict thresholds.

    Checks:
    1. Tech action must agree with final action (sentiment/macro alone cannot carry)
    2. ML probability must be far enough from 0.5 (strong model conviction)
    3. Recent whipsaws above threshold block new entries entirely
    """
    if action == "hold" or regime != "ranging":
        return GateDecision(action=action)

    _is_buy = action == "buy"
    _eff_ml_sep = buy_ml_separation_min if _is_buy else ml_separation_min
    _eff_whip = buy_whipsaw_block_threshold if _is_buy else whipsaw_block_threshold
    _eff_soft = True if _is_buy else soft_mode

    if tech_action is not None and tech_action != action:
        if _eff_soft:
            _scale = 0.70 if _is_buy else 0.50
            logger.debug("RANGING_GATE: tech=%s != action=%s -> size %.1fx", tech_action, action, _scale)
            return GateDecision(
                action=action, size_scale=_scale,
                blocked_by=None,
                detail={"tech_action": tech_action, "final_action": action, "soft": True},
            )
        logger.debug("RANGING_GATE: tech=%s != action=%s -> hold", tech_action, action)
        return GateDecision(
            action="hold", blocked_by="ranging_gate_tech_disagree",
            detail={"tech_action": tech_action, "final_action": action},
        )

    if up_prob is not None:
        separation = abs(up_prob - 0.5)
        if separation < _eff_ml_sep:
            if _eff_soft and separation >= _eff_ml_sep * 0.6:
                _scale = 0.70 if _is_buy else 0.50
                logger.debug("RANGING_GATE: ML separation %.3f marginal -> size %.1fx", separation, _scale)
                return GateDecision(
                    action=action, size_scale=_scale,
                    detail={"separation": separation, "required": _eff_ml_sep, "soft": True},
                )
            logger.debug(
                "RANGING_GATE: ML separation %.3f < %.3f -> hold",
                separation, _eff_ml_sep,
            )
            return GateDecision(
                action="hold", blocked_by="ranging_gate_ml_indifference",
                detail={"separation": separation, "required": _eff_ml_sep},
            )

    if recent_whipsaw_count >= _eff_whip:
        logger.debug(
            "RANGING_GATE: whipsaws %d >= %d in ranging -> hold",
            recent_whipsaw_count, _eff_whip,
        )
        return GateDecision(
            action="hold", blocked_by="ranging_gate_whipsaw",
            detail={"whipsaws": recent_whipsaw_count, "threshold": _eff_whip},
        )

    return GateDecision(action=action)


def pullback_gate(
    action: str,
    candles: pd.DataFrame,
    *,
    lookback: int = 12,
    max_range_position: float = 0.45,
    max_run_up_pct: float = 2.0,
    regime: str | None = None,
) -> GateDecision:
    """Block buy entries that chase recent price run-ups.

    Regime-aware: in ``ranging`` markets, entering near the top of the
    range is buying at resistance — block even without a run-up.
    In other regimes, require both near-top AND run-up to block.

    Checks (buy signals only):
    1. Where the close sits in the [low, high] range of the last N bars.
    2. How much the close has risen from N bars ago.

    In ranging: near_top alone blocks (buying at resistance).
    Otherwise: both near_top AND chasing required to block.
    """
    if action != "buy" or len(candles) < lookback + 1:
        return GateDecision(action=action)

    close = float(candles["close"].iloc[-1])
    highs = candles["high"].iloc[-(lookback + 1):]
    lows = candles["low"].iloc[-(lookback + 1):]
    local_high = float(highs.max())
    local_low = float(lows.min())

    local_range = local_high - local_low
    if local_range <= 0:
        return GateDecision(action=action)
    range_pos = (close - local_low) / local_range

    close_lookback = float(candles["close"].iloc[-(lookback + 1)])
    run_up_pct = (close - close_lookback) / max(close_lookback, 1e-9) * 100

    _strict_regimes = ("ranging", "trending_up")
    _strict_thresh = 0.35 if regime in _strict_regimes else max_range_position
    near_top = range_pos > _strict_thresh
    chasing = run_up_pct > max_run_up_pct

    if regime in _strict_regimes and near_top:
        return GateDecision(
            action="hold",
            blocked_by=f"pullback_gate_{regime}_resistance",
            detail={
                "range_position": round(range_pos, 3),
                "threshold": _strict_thresh,
                "regime": regime,
            },
        )

    if near_top and chasing:
        return GateDecision(
            action="hold",
            blocked_by="pullback_gate_chasing",
            detail={
                "range_position": round(range_pos, 3),
                "max_range_position": max_range_position,
                "run_up_pct": round(run_up_pct, 2),
                "max_run_up_pct": max_run_up_pct,
            },
        )

    if near_top:
        return GateDecision(
            action=action,
            size_scale=0.5,
            detail={
                "range_position": round(range_pos, 3),
                "run_up_pct": round(run_up_pct, 2),
                "note": "near top but no excessive run-up; half-size",
            },
        )

    return GateDecision(action=action)


def sell_pullback_gate(
    action: str,
    candles: pd.DataFrame,
    *,
    lookback: int = 12,
    min_range_position: float = 0.45,
    max_drop_pct: float = 2.0,
    regime: str | None = None,
) -> GateDecision:
    """Block sell/short entries that chase recent price drops.

    Mirror of ``pullback_gate`` for the sell side:
    - In ``ranging`` or ``trending_down``, shorting near the range bottom
      is selling at support — block even without a large drop.
    - Otherwise, require both near-bottom AND chasing drop to block.
    """
    if action != "sell" or len(candles) < lookback + 1:
        return GateDecision(action=action)

    close = float(candles["close"].iloc[-1])
    highs = candles["high"].iloc[-(lookback + 1):]
    lows = candles["low"].iloc[-(lookback + 1):]
    local_high = float(highs.max())
    local_low = float(lows.min())

    local_range = local_high - local_low
    if local_range <= 0:
        return GateDecision(action=action)
    range_pos = (close - local_low) / local_range

    close_lookback = float(candles["close"].iloc[-(lookback + 1)])
    drop_pct = (close_lookback - close) / max(close_lookback, 1e-9) * 100

    _strict_regimes = ("ranging", "trending_down")
    _strict_thresh = 0.60 if regime in _strict_regimes else min_range_position
    near_bottom = range_pos < _strict_thresh
    chasing_drop = drop_pct > max_drop_pct

    if regime in _strict_regimes and near_bottom:
        return GateDecision(
            action="hold",
            blocked_by=f"sell_pullback_gate_{regime}_support",
            detail={
                "range_position": round(range_pos, 3),
                "threshold": _strict_thresh,
                "regime": regime,
            },
        )

    if near_bottom and chasing_drop:
        return GateDecision(
            action="hold",
            blocked_by="sell_pullback_gate_chasing",
            detail={
                "range_position": round(range_pos, 3),
                "min_range_position": min_range_position,
                "drop_pct": round(drop_pct, 2),
                "max_drop_pct": max_drop_pct,
            },
        )

    if near_bottom:
        return GateDecision(
            action=action,
            size_scale=0.5,
            detail={
                "range_position": round(range_pos, 3),
                "drop_pct": round(drop_pct, 2),
                "note": "near bottom but no excessive drop; half-size",
            },
        )

    return GateDecision(action=action)


def ml_confidence(up_prob: float) -> float:
    """Return a position-size scaling factor in [0, 1] based on how far the
    predicted probability is from the indifferent 0.5 mark.

    A probability of 0.5 means the model has no opinion → scale = 0.
    A probability of 0.0 or 1.0 means maximum confidence → scale = 1.
    The mapping is linear:  scale = |up_prob − 0.5| × 2
    """
    if up_prob is None or np.isnan(up_prob):
        return 0.0
    up_prob = max(0.0, min(1.0, up_prob))
    return min(1.0, abs(up_prob - 0.5) * 2.0)


def ml_probability_sizer(
    action: str,
    up_prob: float,
    sensitivity: float = 4.0,
    floor: float = 0.40,
    ceiling: float = 1.50,
) -> float:
    """Map ML probability to a continuous position-size scale.

    Unlike the binary ml_filter (which blocks trades), this function
    never blocks — it scales position size proportionally to how much
    the model agrees with the technical signal.

    For buy:  scale = 1.0 + (up_prob - 0.50) * sensitivity
    For sell: scale = 1.0 + (0.50 - up_prob) * sensitivity
    Clamped to [floor, ceiling].

    With sensitivity=4.0:
      buy  @ prob=0.40 → 0.60x     buy  @ prob=0.55 → 1.20x
      sell @ prob=0.60 → 0.60x     sell @ prob=0.45 → 1.20x
    """
    if up_prob is None or np.isnan(up_prob):
        return 1.0
    up_prob = max(0.0, min(1.0, up_prob))
    if action == "buy":
        scale = 1.0 + (up_prob - 0.50) * sensitivity
    elif action == "sell":
        scale = 1.0 + (0.50 - up_prob) * sensitivity
    else:
        return 1.0
    return max(floor, min(ceiling, scale))


def ml_blind_scale(
    recent_probs: list[float] | np.ndarray,
    *,
    window: int = 24,
    std_threshold: float = 0.015,
    floor_scale: float = 0.50,
) -> float:
    """Detect when the ML model is 'blind' and scale down accordingly.

    When the model's recent output probabilities cluster tightly around 0.50
    (low standard deviation), it has no conviction.  In that state we reduce
    position sizing rather than trusting a coinflip.

    Returns a scale factor in [floor_scale, 1.0].  Below *std_threshold*
    the scale drops linearly from 1.0 toward *floor_scale*.
    """
    if recent_probs is None or len(recent_probs) < max(4, window // 4):
        return 1.0
    tail = np.asarray(recent_probs[-window:], dtype=np.float64)
    tail = tail[~np.isnan(tail)]
    if len(tail) < 4:
        return 1.0
    std = float(np.std(tail))
    if std >= std_threshold:
        return 1.0
    ratio = std / std_threshold
    return floor_scale + (1.0 - floor_scale) * ratio


def ml_blind_blocks_shorts(
    recent_probs: list[float] | np.ndarray,
    *,
    window: int = 24,
    block_threshold: float = 0.010,
) -> bool:
    """Return True when the ML model is so indecisive that shorts should be blocked.

    Shorts with zero model conviction are pure noise — they never hit TP and
    bleed via stop-out or max-hold timeout.  This gate blocks short entries
    when the rolling probability std drops below *block_threshold* (stricter
    than the sizing scale's 0.02).
    """
    if recent_probs is None or len(recent_probs) < max(4, window // 4):
        return False
    tail = np.asarray(recent_probs[-window:], dtype=np.float64)
    tail = tail[~np.isnan(tail)]
    if len(tail) < 4:
        return False
    return float(np.std(tail)) < block_threshold


def loss_streak_scale(
    recent_outcomes: list[bool],
    *,
    streak_threshold: int = 3,
    dampened_scale: float = 0.50,
) -> float:
    """Reduce sizing after consecutive losses.

    Counts consecutive losses from the end of *recent_outcomes* (True=win,
    False=loss).  When the streak reaches *streak_threshold*, returns
    *dampened_scale*.  Otherwise returns 1.0.
    """
    if not recent_outcomes:
        return 1.0
    streak = 0
    for outcome in reversed(recent_outcomes):
        if not outcome:
            streak += 1
        else:
            break
    if streak >= streak_threshold:
        return dampened_scale
    return 1.0


class AdaptiveConfidence:
    """Compute position-size scaling that adapts to recent model accuracy,
    forecast agreement, and calibration quality.

    Instead of using a simple linear function of ``up_prob``, this class
    tracks recent prediction outcomes and adjusts confidence accordingly:

    - **Recency-weighted accuracy**: Recent correct predictions boost
      confidence; recent misses dampen it.
    - **Forecast agreement**: When the ML model and forecast head agree
      on direction, confidence is boosted.
    - **Calibration tracking**: Monitors predicted vs actual win rates
      in probability bins to detect miscalibration.

    Usage::

        ac = AdaptiveConfidence()

        # On each trade signal
        scale = ac.compute(up_prob=0.72, forecast_bias=0.15)

        # After trade closes, record outcome
        ac.record_outcome(predicted_prob=0.72, actual_label=1)
    """

    def __init__(
        self,
        max_history: int = 200,
        recency_halflife: int = 30,
        n_bins: int = 5,
    ) -> None:
        self._max_history = max_history
        self._recency_halflife = recency_halflife
        self._n_bins = n_bins

        self._predictions: list[float] = []
        self._outcomes: list[int] = []
        self._timestamps: list[float] = []

        self._bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        self._bin_correct: list[int] = [0] * n_bins
        self._bin_total: list[int] = [0] * n_bins

    def compute(
        self,
        up_prob: float,
        forecast_bias: float | None = None,
        regime: str | None = None,
    ) -> float:
        """Compute adaptive confidence scaling factor in [0, 1].

        Parameters
        ----------
        up_prob
            ML model predicted probability of up-move.
        forecast_bias
            Signed bias from the ForecastHead: positive = bullish,
            negative = bearish. Used to boost/dampen when ML and
            forecast agree/disagree.
        regime
            Current market regime for regime-specific adjustments.
        """
        base = ml_confidence(up_prob)

        recency_factor = self._recency_accuracy_factor()

        agreement_factor = 1.0
        if forecast_bias is not None:
            ml_direction = 1.0 if up_prob > 0.5 else -1.0
            fc_direction = 1.0 if forecast_bias > 0.02 else (-1.0 if forecast_bias < -0.02 else 0.0)
            if ml_direction * fc_direction > 0:
                agreement_factor = 1.0 + min(0.2, abs(forecast_bias) * 2.0)
            elif ml_direction * fc_direction < 0:
                disagreement_strength = min(0.3, abs(forecast_bias) * 2.0)
                agreement_factor = 1.0 - disagreement_strength

        calibration_factor = self._calibration_factor(up_prob)

        regime_factor = 1.0
        if regime == "volatile":
            regime_factor = 0.8
        elif regime == "ranging":
            regime_factor = 0.85

        final = base * recency_factor * agreement_factor * calibration_factor * regime_factor
        return max(0.0, min(1.0, final))

    def record_outcome(self, predicted_prob: float, actual_label: int) -> None:
        """Record a trade outcome for learning.

        Parameters
        ----------
        predicted_prob
            The ML probability that was used when the signal was generated.
        actual_label
            1 if the trade was profitable, 0 otherwise.
        """
        self._predictions.append(predicted_prob)
        self._outcomes.append(actual_label)
        self._timestamps.append(time.time())

        if len(self._predictions) > self._max_history:
            self._predictions = self._predictions[-self._max_history:]
            self._outcomes = self._outcomes[-self._max_history:]
            self._timestamps = self._timestamps[-self._max_history:]

        bin_idx = min(self._n_bins - 1, int(predicted_prob * self._n_bins))
        self._bin_total[bin_idx] += 1
        if actual_label == 1:
            self._bin_correct[bin_idx] += 1

    def _recency_accuracy_factor(self) -> float:
        """Compute a recency-weighted accuracy factor in [0.5, 1.3].

        Recent predictions that were correct boost confidence;
        recent incorrect predictions dampen it.
        """
        if len(self._outcomes) < 5:
            return 1.0

        n = len(self._outcomes)
        weights = np.array([
            2.0 ** (-(n - 1 - i) / self._recency_halflife) for i in range(n)
        ])
        weights /= weights.sum()

        preds = np.array(self._predictions)
        outcomes = np.array(self._outcomes)

        predicted_direction = (preds > 0.5).astype(int)
        correct = (predicted_direction == outcomes).astype(float)
        weighted_accuracy = float(np.dot(weights, correct))

        return 0.5 + weighted_accuracy * 0.8

    def _calibration_factor(self, up_prob: float) -> float:
        """Adjust confidence based on how well-calibrated the model is
        in the probability bin that ``up_prob`` falls into.

        If the model says 70% and historically 70% of such predictions
        were correct, calibration is good → factor ~1.0.
        If the model consistently over-predicts, factor < 1.0.
        """
        bin_idx = min(self._n_bins - 1, int(up_prob * self._n_bins))
        total = self._bin_total[bin_idx]
        if total < 10:
            return 1.0

        actual_rate = self._bin_correct[bin_idx] / total
        expected_rate = (self._bin_edges[bin_idx] + self._bin_edges[bin_idx + 1]) / 2.0
        calibration_error = abs(actual_rate - expected_rate)

        if calibration_error < 0.05:
            return 1.05
        elif calibration_error < 0.15:
            return 1.0
        else:
            return max(0.7, 1.0 - calibration_error)

    def get_diagnostics(self) -> dict:
        """Return diagnostic info for logging/monitoring."""
        n = len(self._outcomes)
        if n == 0:
            return {"n_outcomes": 0, "recency_factor": 1.0}

        outcomes = np.array(self._outcomes)
        bin_accuracy = {}
        for i in range(self._n_bins):
            if self._bin_total[i] > 0:
                lo = self._bin_edges[i]
                hi = self._bin_edges[i + 1]
                bin_accuracy[f"{lo:.1f}-{hi:.1f}"] = {
                    "accuracy": round(self._bin_correct[i] / self._bin_total[i], 3),
                    "count": self._bin_total[i],
                }

        return {
            "n_outcomes": n,
            "overall_accuracy": round(float(outcomes.mean()), 3),
            "recent_20_accuracy": round(float(outcomes[-20:].mean()), 3) if n >= 20 else None,
            "recency_factor": round(self._recency_accuracy_factor(), 3),
            "bin_accuracy": bin_accuracy,
        }


# ---------------------------------------------------------------------------
# Walk-forward failure response
# ---------------------------------------------------------------------------

def wf_failure_scale(
    report_path: str = "diagnostics/walk_forward_report.json",
    *,
    min_scale: float = 0.50,
    max_age_hours: float = 168.0,
) -> float:
    """Read the latest walk-forward report and return a sizing penalty.

    - PASS gate → 1.0 (no penalty)
    - FAIL gate → linear penalty based on how many windows failed, floored
      at *min_scale*
    - Missing or stale report → 1.0 (no penalty, cannot punish missing data)

    Designed to be called once at startup or periodically (e.g. every 24h).
    """
    from pathlib import Path
    rpt = Path(report_path)
    if not rpt.exists():
        return 1.0
    try:
        import os
        age_hours = (time.time() - os.path.getmtime(rpt)) / 3600.0
        if age_hours > max_age_hours:
            return 1.0
        data = json.loads(rpt.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        gate = summary.get("passes_gate", True)
        if gate:
            return 1.0
        n_windows = summary.get("n_windows", 1) or 1
        n_positive = summary.get("n_positive", 0)
        failure_frac = 1.0 - (n_positive / n_windows)
        scale = max(min_scale, 1.0 - failure_frac * 0.5)
        logger.info(
            "WF failure response: %d/%d windows positive → scale=%.2f",
            n_positive, n_windows, scale,
        )
        return scale
    except Exception as exc:
        logger.debug("wf_failure_scale error: %s", exc)
        return 1.0
