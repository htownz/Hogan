from __future__ import annotations

import os
from dataclasses import dataclass

from hogan_bot.ict import ict_setup_signal, parse_time_windows
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
    # ICT pillars
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
    # Vote-margin guard for "any" mode: buy/sell only fires when the directional
    # vote lead is >= this value.  Default 1 = any majority.  Set to 2 to require
    # at least two more buy than sell votes (or vice-versa) before trading.
    min_vote_margin: int = 1,
) -> StrategySignal:
    """MA crossover + optional EMA clouds / FVG / ICT pillars, combined via votes.

    ICT note
    --------
    When ``use_ict=True`` the ICT signal is an additional vote alongside the MA
    crossover.  The standalone FVG vote (``use_fvg``) is automatically skipped
    when ICT is enabled because ICT already incorporates FVG internally —
    enabling both would double-count the same information.

    ``signal_mode``
    ---------------
    ``"ma_only"``  Original MA crossover only; all extra signals ignored.
    ``"any"``      Buy/sell fires when *any* enabled signal agrees (default).
    ``"all"``      Buy/sell fires only when *every* enabled signal agrees.

    RL agent note
    -------------
    When ``use_rl_agent=True`` the PPO policy is added as an additional vote.
    Pass ``rl_policy`` as a loaded :class:`~hogan_bot.rl_agent.RLPolicy` object.
    Position-state context (``rl_in_position``, ``rl_unrealized_pnl``,
    ``rl_bars_in_trade``) should reflect the live paper portfolio so the agent
    sees its own state during inference.

    When ``rl_use_ext_features=True`` the policy expects the 73-dim extended obs;
    pass ``rl_candles_1h``, ``rl_candles_15m``, ``rl_db_conn``, and ``rl_symbol``
    to supply the MTF and external-data inputs.
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

    # MA crossover detection: signal only on the bar where the MAs actually cross,
    # not on every bar where the trend holds.  This prevents entering at random
    # points within an established trend.
    cur_above = short_ma.iloc[-1] > long_ma.iloc[-1]
    prev_above = short_ma.iloc[-2] > long_ma.iloc[-2]
    bullish_cross = cur_above and not prev_above
    bearish_cross = (not cur_above) and prev_above

    volume_ratio = float(volume.iloc[-1] / max(avg_volume.iloc[-1], 1e-9))
    volume_confirmed = volume_ratio >= volume_threshold

    confidence = min(1.0, max(0.0, (volume_ratio - 1.0) / 1.5)) if volume_confirmed else 0.0

    # ATR-based stop distance — multiplier is now configurable
    atr_series = compute_atr(candles, window=14)
    atr_stop = float(atr_series.iloc[-1] / max(close.iloc[-1], 1e-9)) * atr_stop_multiplier
    stop_distance_pct = max(0.004, min(0.03, atr_stop))

    if bullish_cross and volume_confirmed:
        ma_action = "buy"
    elif bearish_cross and volume_confirmed:
        ma_action = "sell"
    else:
        ma_action = "hold"

    # Fast path: MA-only mode or no extras enabled
    extra_enabled = use_ema_clouds or (use_fvg and not use_ict) or use_ict or use_rl_agent
    if signal_mode == "ma_only" or not extra_enabled:
        return StrategySignal(ma_action, stop_distance_pct, confidence, volume_ratio)

    votes: list[str] = [ma_action]

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
            # EMA cloud is a persistent trend signal; blend 0.5 baseline confidence
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

    # ── ICT vote (primary signal; incorporates FVG internally) ───────────────
    if use_ict:
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
        # Blend ICT confidence into the overall confidence metric
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

        # Penalize confidence when signals disagree (mixed-signal candle)
        directional_votes = buy_votes + sell_votes
        if directional_votes > 0:
            consensus_ratio = max(buy_votes, sell_votes) / directional_votes
            confidence *= consensus_ratio

    # ── ATR minimum-move guard (optional) ───────────────────────────────────────
    # Skip trade only when volatility is extremely low (dead candles).
    # 0.15% = typical 5m BTC ATR floor; set HOGAN_ATR_MIN_PCT=0 to disable.
    min_atr_pct = float(os.getenv("HOGAN_ATR_MIN_PCT", "0.0015"))  # 0.15%
    if min_atr_pct > 0 and action != "hold":
        atr_pct = float(atr_series.iloc[-1] / max(close.iloc[-1], 1e-9))
        if atr_pct < min_atr_pct:
            action = "hold"

    return StrategySignal(action, stop_distance_pct, confidence, volume_ratio)
