from __future__ import annotations


def apply_ml_filter(signal_action: str, up_prob: float, buy_threshold: float, sell_threshold: float) -> str:
    if signal_action == "buy" and up_prob < buy_threshold:
        return "hold"
    if signal_action == "sell" and up_prob > sell_threshold:
        return "hold"
    return signal_action


def ml_confidence(up_prob: float) -> float:
    """Return a position-size scaling factor in [0, 1] based on how far the
    predicted probability is from the indifferent 0.5 mark.

    A probability of 0.5 means the model has no opinion → scale = 0.
    A probability of 0.0 or 1.0 means maximum confidence → scale = 1.
    The mapping is linear:  scale = |up_prob − 0.5| × 2
    """
    return min(1.0, abs(up_prob - 0.5) * 2.0)
