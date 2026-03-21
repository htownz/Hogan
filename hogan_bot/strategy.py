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


# ---------------------------------------------------------------------------
# Strategy Family protocol — regime-routed strategy selection
# ---------------------------------------------------------------------------

class StrategyFamily(Protocol):
    """Interface for regime-specific strategy families.

    Each family implements its own signal generation logic optimized for
    a specific market regime.  The StrategyRouter selects which family
    to invoke based on the current regime.
    """
    name: str

    def generate_signal(
        self,
        candles,
        config,
        regime_state=None,
    ) -> StrategySignal: ...


# ---------------------------------------------------------------------------
# TrendFollowFamily — for trending_up / trending_down regimes
# ---------------------------------------------------------------------------

class TrendFollowFamily:
    """MA crossover + trend continuation strategy.

    Active during trending regimes.  Uses the existing generate_signal()
    logic with MA crossover and pullback entries.
    """
    name: str = "trend_follow"

    def generate_signal(self, candles, config, regime_state=None) -> StrategySignal:
        return generate_signal(
            candles,
            short_window=config.short_ma_window,
            long_window=config.long_ma_window,
            volume_window=config.volume_window,
            volume_threshold=config.volume_threshold,
            use_ema_clouds=config.use_ema_clouds,
            ema_fast_short=config.ema_fast_short,
            ema_fast_long=config.ema_fast_long,
            ema_slow_short=config.ema_slow_short,
            ema_slow_long=config.ema_slow_long,
            use_fvg=config.use_fvg,
            fvg_min_gap_pct=config.fvg_min_gap_pct,
            signal_mode=config.signal_mode,
            atr_stop_multiplier=config.atr_stop_multiplier,
            use_ict=config.use_ict,
            min_vote_margin=getattr(config, "signal_min_vote_margin", 1),
        )


# ---------------------------------------------------------------------------
# MeanRevertFamily — for ranging regime
# ---------------------------------------------------------------------------

class MeanRevertFamily:
    """RSI + Bollinger Band mean-reversion for ranging markets.

    Tiered signal generation:
    - Strong: RSI < 30 / > 70 with BB confirmation → high confidence
    - Standard: RSI < 40 / > 60 with BB confirmation → moderate confidence
    - Stochastic RSI crossover as secondary signal source
    """
    name: str = "mean_revert"

    def generate_signal(self, candles, config, regime_state=None) -> StrategySignal:
        import numpy as np

        min_bars = max(getattr(config, "long_ma_window", 50), 30)
        if len(candles) < min_bars:
            return StrategySignal("hold", 0.01, 0.0, 0.0)

        close = candles["close"].astype(float)
        high = candles["high"].astype(float)
        low = candles["low"].astype(float)
        volume = candles["volume"].astype(float)

        atr_series = compute_atr(candles, window=14)
        atr_val = float(atr_series.iloc[-1])
        px = float(close.iloc[-1])
        atr_pct = atr_val / max(px, 1e-9)
        stop_distance_pct = max(0.003, min(0.015, atr_pct * 1.5))

        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio = float(volume.iloc[-1] / max(avg_vol, 1e-9))

        rsi_period = 14
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=rsi_period - 1, min_periods=rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=rsi_period - 1, min_periods=rsi_period, adjust=False).mean()
        rs = gain / loss.clip(lower=1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_val = float(rsi.iloc[-1])
        if np.isnan(rsi_val):
            rsi_val = 50.0

        bb_period = 20
        bb_ma = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        bb_range = bb_upper - bb_lower
        bb_pct_b = (close - bb_lower) / bb_range.replace(0, np.nan)
        bb_val = float(bb_pct_b.iloc[-1]) if not np.isnan(float(bb_pct_b.iloc[-1])) else 0.5

        adx_check = True
        if regime_state is not None and hasattr(regime_state, "adx"):
            adx_check = regime_state.adx < 30

        action = "hold"
        confidence = 0.0

        if adx_check:
            # Tier 1 — strong extremes (high confidence)
            if rsi_val < 30 and bb_val < 0.20:
                action = "buy"
                confidence = min(1.0, 0.50 + (30 - rsi_val) / 30.0 + (0.20 - bb_val))
            elif rsi_val > 70 and bb_val > 0.80:
                action = "sell"
                confidence = min(1.0, 0.50 + (rsi_val - 70) / 30.0 + (bb_val - 0.80))
            # Tier 2 — moderate oversold/overbought (moderate confidence)
            elif rsi_val < 40 and bb_val < 0.30:
                action = "buy"
                confidence = min(0.60, 0.20 + (40 - rsi_val) / 40.0 + (0.30 - bb_val) * 0.5)
            elif rsi_val > 60 and bb_val > 0.70:
                action = "sell"
                confidence = min(0.60, 0.20 + (rsi_val - 60) / 40.0 + (bb_val - 0.70) * 0.5)

        # Tier 3 — Keltner Channel mean-reversion (wider bands, catches more setups)
        if action == "hold" and adx_check:
            kc_ema = close.ewm(span=20, adjust=False).mean()
            kc_upper = kc_ema + 2.0 * atr_series
            kc_lower = kc_ema - 2.0 * atr_series
            kc_up = float(kc_upper.iloc[-1])
            kc_lo = float(kc_lower.iloc[-1])

            if px <= kc_lo and rsi_val < 45:
                action = "buy"
                deviation = (kc_lo - px) / max(atr_val, 1e-9)
                confidence = min(0.45, 0.25 + deviation * 0.10)
            elif px >= kc_up and rsi_val > 55:
                action = "sell"
                deviation = (px - kc_up) / max(atr_val, 1e-9)
                confidence = min(0.45, 0.25 + deviation * 0.10)

        # Tier 4 — VWAP reversion (fires in normal conditions)
        if action == "hold" and adx_check and len(candles) >= 20:
            typical = (high + low + close) / 3.0
            cum_vol = volume.rolling(20).sum()
            cum_tp_vol = (typical * volume).rolling(20).sum()
            vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
            vwap_val = float(vwap.iloc[-1]) if not np.isnan(float(vwap.iloc[-1])) else px

            vwap_dev = (px - vwap_val) / max(atr_val, 1e-9)

            if vwap_dev < -0.6 and rsi_val < 50:
                action = "buy"
                confidence = min(0.35, 0.15 + abs(vwap_dev) * 0.08)
            elif vwap_dev > 0.6 and rsi_val > 50:
                action = "sell"
                confidence = min(0.35, 0.15 + abs(vwap_dev) * 0.08)

        # Tier 5 — ROC momentum crossover (catches early reversals)
        if action == "hold" and adx_check and len(candles) >= 10:
            roc_5 = close.pct_change(5)
            roc_now = float(roc_5.iloc[-1]) if not np.isnan(float(roc_5.iloc[-1])) else 0.0
            roc_prev = float(roc_5.iloc[-2]) if not np.isnan(float(roc_5.iloc[-2])) else 0.0

            if roc_prev < 0 and roc_now > 0 and rsi_val < 55:
                action = "buy"
                confidence = 0.20
            elif roc_prev > 0 and roc_now < 0 and rsi_val > 45:
                action = "sell"
                confidence = 0.20

        # Tier 6 — Stochastic RSI crossover (lowest confidence fallback)
        if action == "hold" and adx_check and len(candles) >= 20:
            stoch_k = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min()).replace(0, np.nan)
            stoch_d = stoch_k.rolling(3).mean()
            k_val = float(stoch_k.iloc[-1]) if not np.isnan(float(stoch_k.iloc[-1])) else 0.5
            d_val = float(stoch_d.iloc[-1]) if not np.isnan(float(stoch_d.iloc[-1])) else 0.5
            k_prev = float(stoch_k.iloc[-2]) if not np.isnan(float(stoch_k.iloc[-2])) else 0.5

            if k_val < 0.25 and k_val > k_prev and k_val > d_val:
                action = "buy"
                confidence = 0.28
            elif k_val > 0.75 and k_val < k_prev and k_val < d_val:
                action = "sell"
                confidence = 0.28

        return StrategySignal(action, stop_distance_pct, confidence, vol_ratio)


# ---------------------------------------------------------------------------
# BreakoutFamily — for volatile regime
# ---------------------------------------------------------------------------

class BreakoutFamily:
    """Volume breakout + Keltner channel for volatile markets.

    Trades only when volume is 2x+ average AND price breaks the
    Keltner channel.  Uses wider stops and larger targets.
    Falls back to hold when ``volatile_policy="hold"`` is configured.
    """
    name: str = "breakout"

    def generate_signal(self, candles, config, regime_state=None) -> StrategySignal:
        volatile_policy = getattr(config, "volatile_policy", "breakout")
        if volatile_policy == "hold":
            return StrategySignal("hold", 0.01, 0.0, 0.0)

        min_bars = max(getattr(config, "long_ma_window", 50), 30)
        if len(candles) < min_bars:
            return StrategySignal("hold", 0.01, 0.0, 0.0)

        close = candles["close"].astype(float)
        volume = candles["volume"].astype(float)

        atr_series = compute_atr(candles, window=14)
        atr_val = float(atr_series.iloc[-1])
        px = float(close.iloc[-1])
        atr_pct = atr_val / max(px, 1e-9)
        stop_distance_pct = max(0.006, min(0.04, atr_pct * 3.0))

        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio = float(volume.iloc[-1] / max(avg_vol, 1e-9))

        # Keltner channel
        kc_period = 20
        kc_ma = close.rolling(kc_period).mean()
        kc_upper = kc_ma + 2.0 * atr_series
        kc_lower = kc_ma - 2.0 * atr_series
        kc_upper_val = float(kc_upper.iloc[-1])
        kc_lower_val = float(kc_lower.iloc[-1])

        vol_breakout = vol_ratio >= 2.0
        action = "hold"
        confidence = 0.0

        if vol_breakout:
            if px > kc_upper_val:
                action = "buy"
                confidence = min(1.0, (vol_ratio - 2.0) / 2.0 + 0.5)
            elif px < kc_lower_val:
                action = "sell"
                confidence = min(1.0, (vol_ratio - 2.0) / 2.0 + 0.5)

        return StrategySignal(action, stop_distance_pct, confidence, vol_ratio)


# ---------------------------------------------------------------------------
# SqueezeFamily — for volatile / compression-expansion transitions
# ---------------------------------------------------------------------------

class SqueezeFamily:
    """Bollinger Band Squeeze strategy for compression-expansion transitions.

    Detects when Bollinger Bands narrow inside the Keltner Channel (classic
    TTM Squeeze condition), waits for the release (BB expansion), and enters
    in the direction of momentum.
    """
    name: str = "squeeze"

    def generate_signal(self, candles, config, regime_state=None) -> StrategySignal:
        import numpy as np

        min_bars = max(getattr(config, "long_ma_window", 50), 30)
        if len(candles) < min_bars:
            return StrategySignal("hold", 0.01, 0.0, 0.0)

        close = candles["close"].astype(float)
        volume = candles["volume"].astype(float)

        atr_series = compute_atr(candles, window=14)
        atr_val = float(atr_series.iloc[-1])
        px = float(close.iloc[-1])
        atr_pct = atr_val / max(px, 1e-9)

        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio = float(volume.iloc[-1] / max(avg_vol, 1e-9))

        bb_period = 20
        bb_ma = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_ma.replace(0, np.nan)
        bb_width = bb_width.replace([np.inf, -np.inf], np.nan)

        kc_period = 20
        kc_ma = close.rolling(kc_period).mean()
        kc_upper = kc_ma + 1.5 * atr_series
        kc_lower = kc_ma - 1.5 * atr_series
        kc_width = (kc_upper - kc_lower) / kc_ma.replace(0, np.nan)
        kc_width = kc_width.replace([np.inf, -np.inf], np.nan)

        if bb_width.isna().iloc[-1] or kc_width.isna().iloc[-1]:
            return StrategySignal("hold", 0.01, 0.0, vol_ratio)

        lookback = 20
        bb_w_recent = bb_width.iloc[-lookback:]
        if len(bb_w_recent) < lookback:
            return StrategySignal("hold", 0.01, 0.0, vol_ratio)

        bb_w_min = float(bb_w_recent.min())
        bb_w_now = float(bb_width.iloc[-1])
        kc_w_now = float(kc_width.iloc[-1])

        was_squeezed = False
        for offset in range(1, min(6, lookback)):
            _bb = float(bb_width.iloc[-1 - offset])
            _kc = float(kc_width.iloc[-1 - offset])
            if _bb < _kc:
                was_squeezed = True
                break

        released = bb_w_now > bb_w_min * 1.05

        ema20 = close.ewm(span=20, adjust=False).mean()
        momentum = float(close.iloc[-1] - ema20.iloc[-1])

        stop_distance_pct = max(0.004, min(0.025, atr_pct * 2.0))

        action = "hold"
        confidence = 0.0

        if was_squeezed and released and momentum > 0:
            action = "buy"
            squeeze_strength = max(0.0, 1.0 - bb_w_min / max(kc_w_now, 1e-9))
            confidence = min(1.0, 0.4 + squeeze_strength * 0.4 + min(vol_ratio, 2.0) * 0.1)

        return StrategySignal(action, stop_distance_pct, confidence, vol_ratio)


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
        # Adaptive threshold: 0.7× ATR stop distance (scales with volatility)
        _trend_spread_threshold = max(0.005, stop_distance_pct * 0.7)
        trend_strong = ma_spread_pct > _trend_spread_threshold

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

        # Penalize confidence when there is disagreement between enabled signals.
        directional_votes = buy_votes + sell_votes
        if directional_votes > 0:
            consensus_ratio = max(buy_votes, sell_votes) / directional_votes
            confidence *= consensus_ratio

    # ── ATR minimum-move guard ────────────────────────────────────────────────
    # Default raised to 0.008 (0.8%): must exceed 1.5x round-trip fees (~0.52%)
    min_atr_pct = float(os.getenv("HOGAN_ATR_MIN_PCT", "0.008"))
    if min_atr_pct > 0 and action != "hold":
        atr_pct = float(atr_series.iloc[-1] / max(close.iloc[-1], 1e-9))
        if atr_pct < min_atr_pct:
            action = "hold"

    return StrategySignal(action, stop_distance_pct, confidence, volume_ratio)
