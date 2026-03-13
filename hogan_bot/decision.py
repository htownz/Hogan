from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    l = candles["low"].values[-window:]
    c = candles["close"].values[-window:]

    if len(h) < 3:
        avg_range = np.mean((h - l) / np.maximum(c, 1e-9))
        return max(0.0001, float(avg_range * 0.25))

    # Simple range-based proxy: spread ≈ small fraction of avg bar range
    avg_range_pct = float(np.mean((h - l) / np.maximum(c, 1e-9)))
    simple_est = avg_range_pct * 0.15

    # Corwin-Schultz high-low estimator
    hl_ratio = np.log(h / np.maximum(l, 1e-9))
    gamma = hl_ratio[:-1] ** 2 + hl_ratio[1:] ** 2

    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(l[:-1], l[1:])
    beta = np.log(h2 / np.maximum(l2, 1e-9)) ** 2

    valid = beta > gamma
    if not np.any(valid):
        return max(0.0001, min(simple_est, 0.005))

    alpha = (np.sqrt(2 * beta[valid]) - np.sqrt(gamma[valid])) / (
        3 - 2 * np.sqrt(2)
    )
    alpha = np.clip(alpha, 0, None)
    spread = 2.0 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    cs_est = float(np.median(spread))

    # Blend: take the smaller of the two estimates (CS can overestimate on
    # synthetic or noisy data where high/low are not real bid-ask artifacts)
    est = min(cs_est, simple_est * 2)
    return max(0.0001, min(est, 0.005))


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


def apply_ml_filter(signal_action: str, up_prob: float, buy_threshold: float, sell_threshold: float) -> str:
    if up_prob is None or np.isnan(up_prob):
        return signal_action
    if signal_action == "buy" and up_prob < buy_threshold:
        return "hold"
    if signal_action == "sell" and up_prob > sell_threshold:
        return "hold"
    return signal_action


def edge_gate(
    action: str,
    atr_pct: float,
    take_profit_pct: float,
    fee_rate: float,
    min_edge_multiple: float = 1.5,
    forecast_expected_return: float | None = None,
    estimated_spread: float = 0.0,
) -> str:
    """Block entries where the expected move is insufficient relative to friction.

    Total friction = round-trip fees + round-trip spread.

    Checks four conditions (any failure -> hold):
    1. ATR must exceed 1.5x total friction (enough volatility to move)
    2. Take-profit target must exceed min_edge_multiple * total friction
    3. If forecast available, expected return must exceed total friction
    4. If spread alone exceeds ATR/3, conditions are too illiquid
    """
    if action == "hold":
        return action

    round_trip_fees = 2.0 * fee_rate
    round_trip_spread = 2.0 * estimated_spread
    total_friction = round_trip_fees + round_trip_spread

    if atr_pct < total_friction * 1.5:
        logger.debug(
            "EDGE_GATE: ATR %.4f < %.4f (1.5x friction: fees=%.4f spread=%.4f) -> hold",
            atr_pct, total_friction * 1.5, round_trip_fees, round_trip_spread,
        )
        return "hold"

    if take_profit_pct > 0 and take_profit_pct < total_friction * min_edge_multiple:
        logger.debug(
            "EDGE_GATE: TP %.4f < %.4f (%.1fx friction) -> hold",
            take_profit_pct, total_friction * min_edge_multiple, min_edge_multiple,
        )
        return "hold"

    if forecast_expected_return is not None and abs(forecast_expected_return) < total_friction:
        logger.debug(
            "EDGE_GATE: forecast_ret %.4f < %.4f (friction) -> hold",
            abs(forecast_expected_return), total_friction,
        )
        return "hold"

    if estimated_spread > 0 and atr_pct > 0 and estimated_spread > atr_pct / 3:
        logger.debug(
            "EDGE_GATE: spread %.4f > ATR/3 %.4f (illiquid) -> hold",
            estimated_spread, atr_pct / 3,
        )
        return "hold"

    return action


_REGIME_QUALITY_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "trending_up":   {"final_mult": 0.80, "tech_mult": 1.00},
    "trending_down": {"final_mult": 0.80, "tech_mult": 1.00},
    "volatile":      {"final_mult": 1.20, "tech_mult": 1.10},
    "ranging":       {"final_mult": 1.00, "tech_mult": 1.25},
}


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
) -> tuple[str, float]:
    """Block entries that don't meet quality thresholds. Returns (action, size_scale).

    Thresholds are adjusted by regime:
    - Trending: relax final_confidence by 20% (trend-following needs less confirmation)
    - Volatile: tighten final_confidence by 20% (require stronger conviction)
    - Ranging: tighten tech_confidence by 25% (mean-revert needs stronger tech signal)

    Checks (any failure blocks the trade):
    1. Final confidence must exceed (regime-adjusted) minimum
    2. Technical confidence must exceed (regime-adjusted) minimum
    3. Regime confidence must be sufficient (blocks ambiguous regimes)
    4. Recent whipsaw count must be below threshold (reduces size or blocks)
    """
    if action == "hold":
        return action, 1.0

    adj = _REGIME_QUALITY_ADJUSTMENTS.get(regime or "", {})
    eff_min_final = min_final_confidence * adj.get("final_mult", 1.0)
    eff_min_tech = min_tech_confidence * adj.get("tech_mult", 1.0)

    size_scale = 1.0

    if final_confidence is not None and final_confidence < eff_min_final:
        logger.debug(
            "QUALITY_GATE: final_conf %.3f < %.3f (regime=%s) -> hold",
            final_confidence, eff_min_final, regime or "none",
        )
        return "hold", 1.0

    if tech_confidence is not None and tech_confidence < eff_min_tech:
        logger.debug(
            "QUALITY_GATE: tech_conf %.3f < %.3f (regime=%s) -> hold",
            tech_confidence, eff_min_tech, regime or "none",
        )
        return "hold", 1.0

    if regime_confidence is not None and regime_confidence < min_regime_confidence:
        logger.debug(
            "QUALITY_GATE: regime_conf %.3f < %.3f -> hold",
            regime_confidence, min_regime_confidence,
        )
        return "hold", 1.0

    if recent_whipsaw_count >= max_whipsaws:
        scale = max(0.25, 1.0 - (recent_whipsaw_count - max_whipsaws + 1) * 0.25)
        logger.debug(
            "QUALITY_GATE: whipsaws %d >= %d -> scale %.2f",
            recent_whipsaw_count, max_whipsaws, scale,
        )
        if scale <= 0.25:
            return "hold", 1.0
        size_scale = scale

    return action, size_scale


def ranging_gate(
    action: str,
    *,
    regime: str | None = None,
    tech_action: str | None = None,
    up_prob: float | None = None,
    recent_whipsaw_count: int = 0,
    ml_separation_min: float = 0.12,
    whipsaw_block_threshold: int = 2,
) -> tuple[str, float]:
    """Extra protections for ranging markets. Returns (action, size_scale).

    Only active when regime == "ranging". Checks:
    1. Tech action must agree with final action (sentiment/macro alone cannot carry)
    2. ML probability must be far enough from 0.5 (strong model conviction)
    3. Recent whipsaws above threshold block new entries entirely
    """
    if action == "hold" or regime != "ranging":
        return action, 1.0

    # 1. Require technical agent agrees with the final direction
    if tech_action is not None and tech_action != action:
        logger.debug(
            "RANGING_GATE: tech=%s != action=%s -> hold", tech_action, action,
        )
        return "hold", 1.0

    # 2. Require ML probability separated from indifference zone
    if up_prob is not None:
        separation = abs(up_prob - 0.5)
        if separation < ml_separation_min:
            logger.debug(
                "RANGING_GATE: ML separation %.3f < %.3f -> hold",
                separation, ml_separation_min,
            )
            return "hold", 1.0

    # 3. Whipsaw cooldown: ranging + recent flips = suppress
    if recent_whipsaw_count >= whipsaw_block_threshold:
        logger.debug(
            "RANGING_GATE: whipsaws %d >= %d in ranging -> hold",
            recent_whipsaw_count, whipsaw_block_threshold,
        )
        return "hold", 1.0

    return action, 1.0


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
