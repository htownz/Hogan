from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

_MAX_CONFIDENCE_SCALE = 1.50


def kelly_fraction(
    p_win: float,
    win_loss_ratio: float,
    kelly_fraction_of_full: float = 0.25,
    clamp: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Fractional-Kelly position multiplier.

    Parameters
    ----------
    p_win:
        Estimated probability of a winning trade (0-1). Typically the
        calibrated ML up-probability for longs, or ``1 - p_up`` for shorts.
    win_loss_ratio:
        Average win size divided by average loss size (``b`` in the classic
        Kelly formula :math:`f^* = p - (1-p)/b`). Must be > 0.
    kelly_fraction_of_full:
        Shrinkage factor. Full Kelly is wildly aggressive and unstable when
        ``p_win`` is mis-estimated; classic practice is to take 1/4 to 1/2
        of full Kelly to keep the equity curve tolerable. Default 0.25.
    clamp:
        ``(lo, hi)`` range for the returned multiplier.

    Returns
    -------
    float in ``clamp`` — 0 means "skip", 1 means "use the full base size".
    """
    if p_win <= 0 or p_win >= 1 or win_loss_ratio <= 0:
        return 0.0
    full_kelly = p_win - (1.0 - p_win) / win_loss_ratio
    if full_kelly <= 0:
        return 0.0
    scaled = full_kelly * max(0.0, kelly_fraction_of_full)
    lo, hi = clamp
    return float(min(max(scaled, lo), hi))


def calculate_position_size(
    equity_usd: float,
    price: float,
    stop_distance_pct: float,
    max_risk_per_trade: float,
    max_allocation_pct: float,
    confidence_scale: float = 1.0,
    fee_rate: float = 0.0,
    atr_pct: float = 0.0,
    avg_atr_pct: float = 0.0,
    *,
    vol_target_pct: float | None = None,
    realized_vol_pct: float | None = None,
    kelly_p_win: float | None = None,
    kelly_win_loss_ratio: float | None = None,
    kelly_fraction_of_full: float = 0.25,
) -> float:
    """Return coin amount based on risk and allocation constraints.

    *confidence_scale* (0.0–``_MAX_CONFIDENCE_SCALE``) multiplies the raw
    position size to allow ML-confidence-based dynamic sizing.  Values > 1.0
    increase size for high-conviction signals (e.g. ``ml_probability_sizer``
    returns up to 1.50).  Default ``1.0`` preserves the original behaviour.

    *fee_rate* — when provided, reduces size proportionally when the stop
    distance is tight relative to fees. When stop_distance_pct < 3 * fee_rate,
    fees eat a large share of the expected move so we scale down to limit damage.

    *atr_pct* / *avg_atr_pct* — when both are positive, apply volatility-
    adjusted sizing.  In high-vol regimes the position shrinks (inverse
    scaling) to keep dollar-risk constant; in low-vol periods it can grow
    slightly (up to 1.30x).  This keeps the *dollar volatility* of each
    position roughly constant regardless of market conditions.

    *vol_target_pct* / *realized_vol_pct* — explicit **volatility targeting**.
    When both are provided, we scale the position by ``vol_target /
    realized_vol`` (clamped to ``[0.25, 2.0]``) so that the *portfolio*
    realised vol tracks a target (e.g. 15% annualised). This is applied on
    top of the ATR-ratio heuristic above and is the preferred knob for
    running multiple strategies at comparable risk.

    *kelly_p_win* / *kelly_win_loss_ratio* / *kelly_fraction_of_full* —
    opt-in **fractional-Kelly** shrinkage. When ``kelly_p_win`` and
    ``kelly_win_loss_ratio`` are provided, the size is further multiplied by
    ``kelly_fraction(p_win, b, kelly_fraction_of_full)`` so that edgier
    setups get bigger size and uncertain setups get much smaller size.
    Disabled by default; pass ``kelly_p_win=None`` to skip.
    """
    if any(math.isnan(v) or math.isinf(v) for v in
           (equity_usd, price, stop_distance_pct, max_risk_per_trade,
            max_allocation_pct, confidence_scale, fee_rate, atr_pct, avg_atr_pct)):
        logger.error(
            "NaN/Inf in position sizing inputs: equity=%.2f price=%.2f "
            "stop=%.4f risk=%.4f alloc=%.4f conf=%.4f",
            equity_usd, price, stop_distance_pct,
            max_risk_per_trade, max_allocation_pct, confidence_scale,
        )
        return 0.0

    if equity_usd <= 0 or price <= 0:
        return 0.0
    if stop_distance_pct <= 0 or max_risk_per_trade <= 0 or max_allocation_pct <= 0:
        return 0.0

    risk_budget_usd = equity_usd * max_risk_per_trade
    size_from_risk = risk_budget_usd / (price * stop_distance_pct)

    allocation_budget_usd = equity_usd * max_allocation_pct
    size_from_allocation = allocation_budget_usd / price

    raw = max(0.0, min(size_from_risk, size_from_allocation))

    if fee_rate > 0:
        fee_floor = 3.0 * fee_rate
        if stop_distance_pct < fee_floor:
            fee_scale = stop_distance_pct / fee_floor
            raw *= max(0.1, fee_scale)

    # Volatility-adjusted sizing: inverse-scale by current vs average ATR.
    # When volatility spikes (atr_pct >> avg_atr_pct), shrink position to
    # maintain roughly constant dollar-risk.  When volatility is low, allow
    # a modest increase (capped at 1.30x) to capture more upside in calm
    # markets without excessive leverage.
    if atr_pct > 0 and avg_atr_pct > 0:
        vol_ratio = atr_pct / avg_atr_pct
        # Inverse square-root scaling: smooths out spikes
        vol_scale = max(0.40, min(1.30, vol_ratio ** -0.5))
        raw *= vol_scale
        if vol_scale < 0.80 or vol_scale > 1.10:
            logger.debug(
                "VOL_SIZE_ADJUST: atr=%.4f avg=%.4f ratio=%.2f scale=%.2f",
                atr_pct, avg_atr_pct, vol_ratio, vol_scale,
            )

    # Explicit volatility targeting — independent of the ATR-ratio heuristic.
    # target / realised, clamped to a sane band so a flash-quiet tape doesn't
    # balloon size and a flash-vol event doesn't zero it out entirely.
    if (
        vol_target_pct is not None
        and realized_vol_pct is not None
        and vol_target_pct > 0
        and realized_vol_pct > 0
    ):
        _vt_scale = max(0.25, min(2.0, vol_target_pct / realized_vol_pct))
        raw *= _vt_scale
        if _vt_scale < 0.70 or _vt_scale > 1.40:
            logger.debug(
                "VOL_TARGET: target=%.4f realized=%.4f scale=%.2f",
                vol_target_pct, realized_vol_pct, _vt_scale,
            )

    # Fractional-Kelly shrinkage — opt-in.
    if kelly_p_win is not None and kelly_win_loss_ratio is not None:
        _k = kelly_fraction(
            p_win=kelly_p_win,
            win_loss_ratio=kelly_win_loss_ratio,
            kelly_fraction_of_full=kelly_fraction_of_full,
        )
        raw *= _k
        if _k < 1.0:
            logger.debug(
                "KELLY: p=%.3f b=%.2f frac=%.2f -> scale=%.3f",
                kelly_p_win, kelly_win_loss_ratio, kelly_fraction_of_full, _k,
            )

    return raw * max(0.0, min(_MAX_CONFIDENCE_SCALE, confidence_scale))


class DrawdownGuard:
    """Tracks equity and stops trading if max drawdown is breached."""

    def __init__(self, starting_equity: float, max_drawdown: float) -> None:
        self.peak_equity = starting_equity
        self.max_drawdown = max_drawdown

    def update_and_check(self, current_equity: float) -> bool:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity <= 0:
            logger.error("DrawdownGuard: peak_equity=%.2f is non-positive — halting trading", self.peak_equity)
            return False

        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            logger.warning(
                "DrawdownGuard BREACH: drawdown=%.2f%% exceeds max=%.2f%% (equity=%.2f peak=%.2f)",
                drawdown * 100, self.max_drawdown * 100, current_equity, self.peak_equity,
            )
            return False
        return True


def compute_portfolio_exposure_scale(
    *,
    portfolio,
    symbol: str,
    candidate_qty: float,
    candidate_price: float,
    equity_usd: float,
    max_gross_exposure_pct: float = 1.50,
    max_symbol_exposure_pct: float = 0.50,
    loss_streak: int = 0,
    loss_regime_streak: int = 3,
    loss_regime_scale: float = 0.70,
) -> tuple[float, dict]:
    """Return an exposure-aware portfolio scale and attribution details."""
    if equity_usd <= 0 or candidate_price <= 0 or candidate_qty <= 0:
        return 0.0, {"reason": "invalid_inputs"}

    _candidate_notional = float(candidate_qty * candidate_price)
    _existing_longs = sum(
        float(getattr(p, "qty", 0.0)) * float(getattr(p, "avg_entry", 0.0))
        for p in getattr(portfolio, "positions", {}).values()
    )
    _existing_shorts = sum(
        float(getattr(p, "qty", 0.0)) * float(getattr(p, "avg_entry", 0.0))
        for p in getattr(portfolio, "short_positions", {}).values()
    )
    _gross_after = _existing_longs + _existing_shorts + _candidate_notional
    _gross_ratio = _gross_after / max(equity_usd, 1e-9)

    _symbol_notional = 0.0
    _lp = getattr(portfolio, "positions", {}).get(symbol)
    if _lp is not None:
        _symbol_notional += float(getattr(_lp, "qty", 0.0)) * float(getattr(_lp, "avg_entry", 0.0))
    _sp = getattr(portfolio, "short_positions", {}).get(symbol)
    if _sp is not None:
        _symbol_notional += float(getattr(_sp, "qty", 0.0)) * float(getattr(_sp, "avg_entry", 0.0))
    _symbol_after = _symbol_notional + _candidate_notional
    _symbol_ratio = _symbol_after / max(equity_usd, 1e-9)

    scale = 1.0
    reasons: list[str] = []
    if max_gross_exposure_pct > 0 and _gross_ratio > max_gross_exposure_pct:
        scale = min(scale, max_gross_exposure_pct / _gross_ratio)
        reasons.append("gross_exposure_cap")
    if max_symbol_exposure_pct > 0 and _symbol_ratio > max_symbol_exposure_pct:
        scale = min(scale, max_symbol_exposure_pct / _symbol_ratio)
        reasons.append("symbol_exposure_cap")
    if loss_streak >= max(1, loss_regime_streak):
        scale = min(scale, max(0.0, min(1.0, loss_regime_scale)))
        reasons.append("loss_regime_dampener")

    return max(0.0, min(1.0, scale)), {
        "gross_ratio": round(_gross_ratio, 4),
        "symbol_ratio": round(_symbol_ratio, 4),
        "loss_streak": int(loss_streak),
        "reasons": reasons,
    }
