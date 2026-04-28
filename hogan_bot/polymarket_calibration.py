"""Calibration utilities for Polymarket probability research.

These functions are pure math helpers used by the analysis and shadow-trading
layers. They do not fetch data or place trades.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    avg_probability: float
    observed_rate: float
    brier: float


def _clip_prob(prob: float, eps: float = 1e-6) -> float:
    return max(eps, min(1.0 - eps, float(prob)))


def brier_score(probabilities: list[float], outcomes: list[int]) -> float:
    """Return mean squared probability error."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    if not probabilities:
        return 0.0
    return sum((float(p) - int(y)) ** 2 for p, y in zip(probabilities, outcomes)) / len(probabilities)


def log_loss(probabilities: list[float], outcomes: list[int]) -> float:
    """Return binary log loss with probability clipping."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    if not probabilities:
        return 0.0
    total = 0.0
    for prob, outcome in zip(probabilities, outcomes):
        p = _clip_prob(prob)
        total += -(int(outcome) * math.log(p) + (1 - int(outcome)) * math.log(1.0 - p))
    return total / len(probabilities)


def calibration_bins(
    probabilities: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Group predictions into probability bins and compare with outcomes."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for prob, outcome in zip(probabilities, outcomes):
        p = max(0.0, min(1.0, float(prob)))
        idx = min(n_bins - 1, int(p * n_bins))
        buckets[idx].append((p, int(outcome)))

    results: list[CalibrationBin] = []
    for idx, rows in enumerate(buckets):
        lower = idx / n_bins
        upper = (idx + 1) / n_bins
        if not rows:
            results.append(CalibrationBin(lower, upper, 0, 0.0, 0.0, 0.0))
            continue
        probs = [p for p, _outcome in rows]
        outs = [outcome for _p, outcome in rows]
        results.append(CalibrationBin(
            lower=lower,
            upper=upper,
            count=len(rows),
            avg_probability=sum(probs) / len(probs),
            observed_rate=sum(outs) / len(outs),
            brier=brier_score(probs, outs),
        ))
    return results


def favorite_longshot_bias(probabilities: list[float], outcomes: list[int]) -> dict[str, float]:
    """Estimate underpricing of favorites and overpricing of longshots."""
    bins = calibration_bins(probabilities, outcomes, n_bins=5)
    longshot = bins[0]
    favorite = bins[-1]
    longshot_bias = longshot.avg_probability - longshot.observed_rate if longshot.count else 0.0
    favorite_bias = favorite.observed_rate - favorite.avg_probability if favorite.count else 0.0
    return {
        "longshot_overpricing": float(longshot_bias),
        "favorite_underpricing": float(favorite_bias),
        "brier": brier_score(probabilities, outcomes),
        "log_loss": log_loss(probabilities, outcomes),
    }


def calibrate_probability(
    probability: float,
    *,
    favorite_underpricing: float = 0.0,
    longshot_overpricing: float = 0.0,
) -> float:
    """Apply simple favorite-longshot calibration adjustments."""
    p = max(0.0, min(1.0, float(probability)))
    if p >= 0.80:
        p += max(0.0, favorite_underpricing)
    elif p <= 0.20:
        p -= max(0.0, longshot_overpricing)
    return max(0.0, min(1.0, p))
