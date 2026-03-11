from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hogan_bot.indicators import (
    cloud_signal,
    compute_atr,
    detect_fvgs,
    fvg_entry_signal,
    ripster_ema_clouds,
)


@dataclass
class StrategySignal:
    action: str
    stop_distance_pct: float
    confidence: float
    volume_ratio: float


@dataclass
class SignalVote:
    """A single signal provider's opinion on the current bar."""
    action: str           # "buy" | "sell" | "hold"
    confidence: float     # 0.0 – 1.0
    source: str           # provider name for logging


@runtime_checkable
class SignalProvider(Protocol):
    """Interface for independent signal providers (Phase A)."""
    def evaluate(self, candles, **kwargs) -> SignalVote: ...


def generate_signal(
    candles,
    short_window: int,
    long_window: int,
    volume_window: int,
    volume_threshold: float,
    *,
    use_ema_clouds: bool = False,
    ema_fast_short: int = 8,
    ema_fast_long: int = 9,
    ema_slow_short: int = 34,
    ema_slow_long: int = 50,
    use_fvg: bool = False,
    fvg_min_gap_pct: float = 0.001,
    signal_mode: str = "any",
    atr_stop_multiplier: float = 1.5,
    # ICT pillars (experimental — quarantined from champion path)
    use_ict: bool = False,
    ict_swing_left: int = 2,
    ict_swing_right: int = 2,
    ict_eq_tolerance_pct: float = 0.0008,
    ict_min_displacement_pct: float = 0.003,
    ict_require_time_window: bool = True,
    ict_time_windows: str = "03:00-04:00,10:00-11:00,14:00-15:00",
    ict_require_pd: bool = True,
    ict_ote_enabled: bool = False,
    ict_ote_low: float = 0.62,
    ict_ote_high: float = 0.79,
    # RL agent
    use_rl_agent: bool = False,
    rl_policy=None,
    rl_in_position: bool = False,
    rl_unrealized_pnl: float = 0.0,
    rl_bars_in_trade: int = 0,
    rl_use_ext_features: bool = False,
    rl_candles_1h=None,
    rl_candles_15m=None,
    rl_db_conn=None,
    rl_symbol: str = "BTC/USD",
    min_vote_margin: int = 1,
) -> StrategySignal:
    """Independent vote-based signal generation.

    All enabled signal providers vote independently.  No single provider is a
    mandatory gatekeeper — every provider's vote carries equal weight in the
    aggregation layer.

    Signal providers
    ----------------
    - MA crossover (always on)
    - EMA clouds (``use_ema_clouds``)
    - FVG (``use_fvg``, skipped when ICT is on)
    - ICT (``use_ict`` — **experimental**, quarantined from champion path)
    - RL agent (``use_rl_agent``)

    ``signal_mode``
    ---------------
    ``"ma_only"``  Only MA crossover (legacy mode).
    ``"any"``      Buy/sell fires when majority agrees (default).
    ``"all"``      Buy/sell fires only when every signal agrees.
    """
    min_len = max(long_window, volume_window) + 2
    if len(candles) < min_len:
        return StrategySignal("hold", 0.01, 0.0, 0.0)

    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    volume = candles["volume"].astype(float)

    short_ma = close.rolling(short_window).mean()
    long_ma = close.rolling(long_window).mean()
    avg_volume = volume.rolling(volume_window).mean()

    cur_above = short_ma.iloc[-1] > long_ma.iloc[-1]
    prev_above = short_ma.iloc[-2] > long_ma.iloc[-2]
    bullish_cross = cur_above and not prev_above
    bearish_cross = (not cur_above) and prev_above

    volume_ratio = float(volume.iloc[-1] / max(avg_volume.iloc[-1], 1e-9))
    volume_confirmed = volume_ratio >= volume_threshold

    confidence = min(1.0, max(0.0, (volume_ratio - 1.0) / 1.5)) if volume_confirmed else 0.0

    atr_series = compute_atr(candles, window=14)
    atr_stop = float(atr_series.iloc[-1] / max(close.iloc[-1], 1e-9)) * atr_stop_multiplier
    stop_distance_pct = max(0.004, min(0.03, atr_stop))

    if bullish_cross and volume_confirmed:
        ma_action = "buy"
    elif bearish_cross and volume_confirmed:
        ma_action = "sell"
    else:
        ma_action = "hold"

    # ── Trend continuation: pullback entries during strong, established trends ──
    # Only fires when:
    #   1) MA spread > 1% (confirmed strong trend, not a fresh/weak crossover)
    #   2) Price spent 2+ bars on the wrong side of the short MA (real pullback)
    #   3) Current bar reclaims the short MA with volume
    trend_action = "hold"
    if ma_action == "hold" and len(candles) >= long_window + 5:
        in_uptrend = cur_above
        price_now = float(close.iloc[-1])
        sma_now = float(short_ma.iloc[-1])
        lma_now = float(long_ma.iloc[-1])

        ma_spread_pct = abs(sma_now - lma_now) / max(lma_now, 1e-9)
        trend_strong = ma_spread_pct > 0.01

        if trend_strong and in_uptrend:
            bars_below = sum(
                1 for j in range(-4, -1)
                if float(close.iloc[j]) < float(short_ma.iloc[j])
            )
            reclaimed = price_now > sma_now
            if bars_below >= 2 and reclaimed and volume_confirmed:
                trend_action = "buy"
                confidence = max(confidence, 0.8)
        elif trend_strong and not in_uptrend:
            bars_above = sum(
                1 for j in range(-4, -1)
                if float(close.iloc[j]) > float(short_ma.iloc[j])
            )
            reclaimed = price_now < sma_now
            if bars_above >= 2 and reclaimed and volume_confirmed:
                trend_action = "sell"
                confidence = max(confidence, 0.8)

    # Fast path: MA-only mode or no extras enabled
    extra_enabled = use_ema_clouds or (use_fvg and not use_ict) or use_ict or use_rl_agent
    if signal_mode == "ma_only":
        if ma_action != "hold":
            return StrategySignal(ma_action, stop_distance_pct, confidence, volume_ratio)
        return StrategySignal(trend_action, stop_distance_pct, confidence, volume_ratio)

    # ── Collect votes from all providers (no MA gatekeeper) ────────────────
    effective_ma = ma_action if ma_action != "hold" else trend_action
    votes: list[str] = [effective_ma]

    # ── EMA cloud vote ────────────────────────────────────────────────────────
    if use_ema_clouds:
        enriched = ripster_ema_clouds(
            candles,
            fast_short=ema_fast_short,
            fast_long=ema_fast_long,
            slow_short=ema_slow_short,
            slow_long=ema_slow_long,
        )
        latest_cloud = cloud_signal(enriched).iloc[-1]
        if latest_cloud == "bullish":
            votes.append("buy")
            confidence = (confidence + 0.5) / 2.0 if confidence > 0 else 0.5
        elif latest_cloud == "bearish":
            votes.append("sell")
            confidence = (confidence + 0.5) / 2.0 if confidence > 0 else 0.5
        else:
            votes.append("hold")

    # ── Standalone FVG vote (skipped when ICT is enabled) ────────────────────
    if use_fvg and not use_ict:
        fvgs = detect_fvgs(candles, min_gap_pct=fvg_min_gap_pct)
        last_close = float(close.iloc[-1])
        votes.append(fvg_entry_signal(fvgs, last_close))

    # ── ICT vote (EXPERIMENTAL — only when explicitly enabled) ───────────────
    if use_ict:
        from hogan_bot.ict import ict_setup_signal, parse_time_windows
        tw_list = parse_time_windows(ict_time_windows)
        ict_action, ict_conf, _debug = ict_setup_signal(
            candles,
            swing_left=ict_swing_left,
            swing_right=ict_swing_right,
            eq_tolerance_pct=ict_eq_tolerance_pct,
            min_displacement_pct=ict_min_displacement_pct,
            require_time_window=ict_require_time_window,
            time_windows=tw_list if tw_list else None,
            require_pd=ict_require_pd,
            ote_enabled=ict_ote_enabled,
            ote_low=ict_ote_low,
            ote_high=ict_ote_high,
        )
        votes.append(ict_action)
        if ict_action != "hold":
            confidence = (confidence + ict_conf) / 2.0

    # ── RL agent vote ─────────────────────────────────────────────────────
    if use_rl_agent:
        try:
            from hogan_bot.rl_agent import predict_rl_action
            rl_action = predict_rl_action(
                candles,
                rl_policy,
                in_position=rl_in_position,
                unrealized_pnl_pct=rl_unrealized_pnl,
                bars_in_trade=rl_bars_in_trade,
                use_ext_features=rl_use_ext_features,
                candles_1h=rl_candles_1h,
                candles_15m=rl_candles_15m,
                db_conn=rl_db_conn,
                symbol=rl_symbol,
            )
            votes.append(rl_action)
            if rl_action != "hold":
                confidence = (confidence + 0.8) / 2.0
        except Exception:  # noqa: BLE001
            votes.append("hold")

    # ── Vote resolution ───────────────────────────────────────────────────────
    if signal_mode == "all":
        if all(v == "buy" for v in votes):
            action = "buy"
        elif all(v == "sell" for v in votes):
            action = "sell"
        else:
            action = "hold"
    else:  # "any" with conflict guard
        buy_votes = sum(1 for v in votes if v == "buy")
        sell_votes = sum(1 for v in votes if v == "sell")
        vote_edge = abs(buy_votes - sell_votes)

        if buy_votes > sell_votes and vote_edge >= min_vote_margin:
            action = "buy"
        elif sell_votes > buy_votes and vote_edge >= min_vote_margin:
            action = "sell"
        else:
            action = "hold"

        directional_votes = buy_votes + sell_votes
        if directional_votes > 0:
            consensus_ratio = max(buy_votes, sell_votes) / directional_votes
            confidence *= consensus_ratio

    # ── ATR minimum-move guard ────────────────────────────────────────────────
    min_atr_pct = float(os.getenv("HOGAN_ATR_MIN_PCT", "0.0015"))
    if min_atr_pct > 0 and action != "hold":
        atr_pct = float(atr_series.iloc[-1] / max(close.iloc[-1], 1e-9))
        if atr_pct < min_atr_pct:
            action = "hold"

    return StrategySignal(action, stop_distance_pct, confidence, volume_ratio)
