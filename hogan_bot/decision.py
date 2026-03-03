from __future__ import annotations


def apply_ml_filter(signal_action: str, up_prob: float, buy_threshold: float, sell_threshold: float) -> str:
    if signal_action == "buy" and up_prob < buy_threshold:
        return "hold"
    if signal_action == "sell" and up_prob > sell_threshold:
        return "hold"
    return signal_action
