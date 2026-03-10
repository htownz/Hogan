"""Macro correlation filter for Hogan.

Uses SPY/QQQ, UUP (DXY proxy), VIX, and GLD hourly candles from the local
SQLite DB to classify the macro environment as risk-on / risk-off / neutral
and gate or scale BTC trades accordingly.

Architecture
------------
Four sub-signals are evaluated each iteration:

1. **Equity trend** -- SPY (+ QQQ) hourly close vs MA.
2. **Dollar strength** -- UUP hourly close vs MA (inverse to BTC).
3. **VIX level** -- absolute level of the CBOE fear gauge.
4. **Gold-BTC divergence** -- GLD rallying while equities fall = flight-to-safety.

The combined result produces a ``MacroFilterResult`` with a confidence
multiplier (0.0--1.0) and an optional hard block on new longs.

Usage
-----
    from hogan_bot.macro_filter import evaluate_macro
    result = evaluate_macro(conn, action="buy", vix_caution=25, vix_block=35)
    if result.block_longs and action == "buy":
        action = "hold"
    conf_scale *= result.confidence_mult
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MacroFilterResult:
    """Output of the macro correlation filter."""
    macro_environment: str      # "risk_on" | "risk_off" | "neutral"
    confidence_mult: float      # 0.0--1.0 multiplier on position size
    block_longs: bool           # True → do not open new long positions
    spy_bullish: bool | None = None
    qqq_bullish: bool | None = None
    dxy_strong: bool | None = None
    vix_level: float | None = None
    gold_bullish: bool | None = None
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sub-signal helpers
# ---------------------------------------------------------------------------

def _is_bullish(candles: pd.DataFrame | None, ma_period: int = 20) -> bool | None:
    """True if latest close is above the MA, False if below, None if no data."""
    if candles is None or len(candles) < ma_period + 1:
        return None
    close = candles["close"].astype(float)
    ma = float(close.rolling(ma_period).mean().iloc[-1])
    return float(close.iloc[-1]) > ma


def _latest_close(candles: pd.DataFrame | None) -> float | None:
    if candles is None or candles.empty:
        return None
    return float(candles["close"].iloc[-1])


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_macro(
    conn,
    action: str = "buy",
    *,
    ma_period: int = 20,
    vix_caution: float = 25.0,
    vix_block: float = 35.0,
    spy_candles: pd.DataFrame | None = None,
    qqq_candles: pd.DataFrame | None = None,
    uup_candles: pd.DataFrame | None = None,
    vix_candles: pd.DataFrame | None = None,
    gld_candles: pd.DataFrame | None = None,
) -> MacroFilterResult:
    """Evaluate the macro environment and return a filter result.

    Parameters
    ----------
    conn : sqlite3.Connection or None
        DB connection for loading candles.  If candle DataFrames are passed
        directly (e.g. in tests), *conn* can be ``None``.
    action : str
        The proposed trade action ("buy" / "sell" / "hold").
    ma_period : int
        Moving-average lookback for trend determination.
    vix_caution / vix_block : float
        VIX thresholds for caution (reduce size) and block (no new longs).
    spy_candles, qqq_candles, uup_candles, vix_candles, gld_candles :
        Optional pre-loaded DataFrames.  When ``None``, loaded from *conn*.
    """
    # Load candles from DB if not passed directly
    if conn is not None:
        from hogan_bot.storage import load_candles
        if spy_candles is None:
            try:
                spy_candles = load_candles(conn, "SPY/USD", "1h", limit=50)
            except Exception:
                pass
        if qqq_candles is None:
            try:
                qqq_candles = load_candles(conn, "QQQ/USD", "1h", limit=50)
            except Exception:
                pass
        if uup_candles is None:
            try:
                uup_candles = load_candles(conn, "UUP/USD", "1h", limit=50)
            except Exception:
                pass
        if vix_candles is None:
            try:
                vix_candles = load_candles(conn, "VIX/USD", "1h", limit=50)
            except Exception:
                pass
        if gld_candles is None:
            try:
                gld_candles = load_candles(conn, "GLD/USD", "1h", limit=50)
            except Exception:
                pass

    # Sub-signals
    spy_bull = _is_bullish(spy_candles, ma_period)
    qqq_bull = _is_bullish(qqq_candles, ma_period)
    dxy_strong = _is_bullish(uup_candles, ma_period)
    gold_bull = _is_bullish(gld_candles, ma_period)
    vix_val = _latest_close(vix_candles)

    confidence = 1.0
    block_longs = False
    reasons: list[str] = []

    # ── VIX gate (highest priority) ────────────────────────────────────────
    if vix_val is not None:
        if vix_val >= vix_block:
            block_longs = True
            confidence *= 0.0
            reasons.append(f"VIX={vix_val:.1f} >= {vix_block} → block longs")
        elif vix_val >= vix_caution:
            confidence *= 0.70
            reasons.append(f"VIX={vix_val:.1f} >= {vix_caution} → caution 0.70x")

    # ── Equity trend ───────────────────────────────────────────────────────
    if spy_bull is False and qqq_bull is False:
        confidence *= 0.40
        reasons.append("SPY+QQQ both bearish → strong risk-off 0.40x")
    elif spy_bull is False:
        confidence *= 0.65
        reasons.append("SPY bearish → risk-off 0.65x")

    # ── Dollar strength (inverse to BTC) ───────────────────────────────────
    if dxy_strong is True:
        if spy_bull is False:
            confidence *= 0.70
            reasons.append("DXY strong + SPY bearish → headwind 0.70x")
        else:
            confidence *= 0.85
            reasons.append("DXY strong (alone) → mild headwind 0.85x")

    # ── Gold-BTC divergence ────────────────────────────────────────────────
    if gold_bull is True and spy_bull is False and action == "buy":
        confidence *= 0.80
        reasons.append("Gold up + SPY down → flight-to-safety divergence 0.80x")

    # ── Classify environment ───────────────────────────────────────────────
    if block_longs:
        environment = "risk_off"
    elif confidence >= 0.85:
        risk_on_signals = sum([
            spy_bull is True,
            dxy_strong is False and dxy_strong is not None,
            vix_val is not None and vix_val < 20,
        ])
        environment = "risk_on" if risk_on_signals >= 2 else "neutral"
    elif confidence < 0.50:
        environment = "risk_off"
    else:
        environment = "neutral"

    result = MacroFilterResult(
        macro_environment=environment,
        confidence_mult=max(0.0, min(1.0, confidence)),
        block_longs=block_longs,
        spy_bullish=spy_bull,
        qqq_bullish=qqq_bull,
        dxy_strong=dxy_strong,
        vix_level=vix_val,
        gold_bullish=gold_bull,
        details={"reasons": reasons},
    )

    if reasons:
        logger.info(
            "MACRO %s: env=%s conf=%.2f block_longs=%s | %s",
            action, environment, result.confidence_mult,
            block_longs, "; ".join(reasons),
        )
    else:
        logger.info(
            "MACRO %s: env=%s conf=%.2f (no adjustments)",
            action, environment, result.confidence_mult,
        )

    return result
