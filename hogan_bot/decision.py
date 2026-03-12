from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def apply_ml_filter(signal_action: str, up_prob: float, buy_threshold: float, sell_threshold: float) -> str:
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
) -> str:
    """Block entries where the expected move is insufficient relative to fees.

    Checks three conditions (any failure -> hold):
    1. ATR must exceed 1.5x round-trip fees (enough volatility to move)
    2. Take-profit target must exceed min_edge_multiple * round-trip fees
    3. If forecast available, expected return must exceed round-trip fees
    """
    if action == "hold":
        return action

    round_trip = 2.0 * fee_rate

    if atr_pct < round_trip * 1.5:
        logger.debug("EDGE_GATE: ATR %.4f < %.4f (1.5x fees) -> hold", atr_pct, round_trip * 1.5)
        return "hold"

    if take_profit_pct < round_trip * min_edge_multiple:
        logger.debug(
            "EDGE_GATE: TP %.4f < %.4f (%.1fx fees) -> hold",
            take_profit_pct, round_trip * min_edge_multiple, min_edge_multiple,
        )
        return "hold"

    if forecast_expected_return is not None and abs(forecast_expected_return) < round_trip:
        logger.debug(
            "EDGE_GATE: forecast_ret %.4f < %.4f (fees) -> hold",
            abs(forecast_expected_return), round_trip,
        )
        return "hold"

    return action


def ml_confidence(up_prob: float) -> float:
    """Return a position-size scaling factor in [0, 1] based on how far the
    predicted probability is from the indifferent 0.5 mark.

    A probability of 0.5 means the model has no opinion → scale = 0.
    A probability of 0.0 or 1.0 means maximum confidence → scale = 1.
    The mapping is linear:  scale = |up_prob − 0.5| × 2
    """
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
