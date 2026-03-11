"""ICT (Inner Circle Trader) technical analysis module for Hogan.

Implements the canonical ICT price-action sequence as pure, testable functions:

    Liquidity pools (equal H/L, swing H/L, prev-day H/L)
        → Liquidity sweep / raid detection
        → Market Structure Shift (MSS) confirmation
        → Order Block (OB) identification
        → FVG + OB overlap entry
        → Premium / Discount filter (+ OTE Fibonacci zone)
        → Time-window / "Silver Bullet" session filter

All functions operate on a pandas DataFrame with columns:
``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``.

Indices in returned dicts are integer positional indices into the DataFrame.

References
----------
* Liquidity, MSS, OB, FVG sequence: ICT "Silver Bullet" model.
  Silver Bullet windows: 03:00–04:00, 10:00–11:00, 14:00–15:00 NY time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Type aliases (plain dicts for zero-dependency portability)
# ---------------------------------------------------------------------------

SwingPoint = dict[str, Any]      # {index: int, price: float}
LiquidityPool = dict[str, Any]   # {price: float, count: int, indices: list[int]}
Sweep = dict[str, Any]           # {side: str, pool_level: float, sweep_index: int, sweep_strength: float}
MSS = dict[str, Any]             # {direction: str, break_index: int, broken_level: float}
OrderBlock = dict[str, Any]      # {direction: str, top: float, bottom: float, index: int}
DealingRange = dict[str, Any]    # {high: float, low: float, mid: float}

# Default Silver Bullet time windows (HH:MM pairs, America/New_York)
SILVER_BULLET_WINDOWS: list[tuple[str, str]] = [
    ("03:00", "04:00"),   # London open
    ("10:00", "11:00"),   # NY AM
    ("14:00", "15:00"),   # NY PM
]

# Confidence weights that sum to 1.0
_W_SWEEP = 0.25
_W_MSS = 0.25
_W_FVG = 0.20
_W_OB = 0.15
_W_PD = 0.10
_W_TIME = 0.05


# ---------------------------------------------------------------------------
# 1. Swing point detection
# ---------------------------------------------------------------------------


def find_swings(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """Detect swing highs and swing lows.

    A swing high at index *i* requires ``high[i] > high[j]`` for all *j* in
    ``[i-left, i)`` and ``high[i] >= high[j]`` for all *j* in ``(i, i+right]``.
    Lows use the symmetric rule.

    Returns
    -------
    (swing_highs, swing_lows)
        Each is a list of ``{index, price}`` dicts sorted by index.
    """
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    n = len(df)

    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for i in range(left, n - right):
        # Swing high
        if (
            all(high[i] > high[i - j] for j in range(1, left + 1))
            and all(high[i] >= high[i + j] for j in range(1, right + 1))
        ):
            swing_highs.append({"index": i, "price": float(high[i])})

        # Swing low
        if (
            all(low[i] < low[i - j] for j in range(1, left + 1))
            and all(low[i] <= low[i + j] for j in range(1, right + 1))
        ):
            swing_lows.append({"index": i, "price": float(low[i])})

    return swing_highs, swing_lows


# ---------------------------------------------------------------------------
# 2. Equal highs / lows — liquidity cluster detection
# ---------------------------------------------------------------------------


def detect_equal_highs_lows(
    swings: list[SwingPoint],
    tolerance_pct: float = 0.0008,
) -> list[LiquidityPool]:
    """Group swing points that are within *tolerance_pct* of each other.

    Two or more swing highs/lows at approximately the same price form a
    *liquidity pool* (equal highs = buy-side liquidity resting above price;
    equal lows = sell-side liquidity resting below price).

    Returns a list of pool dicts, each with ``price`` (average), ``count``,
    and ``indices``.
    """
    if not swings:
        return []

    used: set[int] = set()
    pools: list[LiquidityPool] = []

    for i, s in enumerate(swings):
        if i in used:
            continue
        cluster: list[SwingPoint] = [s]
        for j in range(i + 1, len(swings)):
            if j in used:
                continue
            t = swings[j]
            if s["price"] > 0 and abs(s["price"] - t["price"]) / s["price"] <= tolerance_pct:
                cluster.append(t)
                used.add(j)
        if len(cluster) >= 2:
            used.add(i)
            avg = sum(c["price"] for c in cluster) / len(cluster)
            pools.append(
                {
                    "price": avg,
                    "count": len(cluster),
                    "indices": [c["index"] for c in cluster],
                }
            )

    return pools


def liquidity_pools(
    df: pd.DataFrame,
    swing_left: int = 2,
    swing_right: int = 2,
    eq_tolerance_pct: float = 0.0008,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Aggregate all liquidity pool types into a single snapshot dict.

    Returns keys:
    ``swing_highs``, ``swing_lows``, ``equal_highs``, ``equal_lows``,
    ``recent_swing_high``, ``recent_swing_low``,
    ``prev_day_high``, ``prev_day_low``.

    *prev_day* values use the previous UTC calendar day when timestamps exist;
    otherwise fall back to bars_per_day(timeframe) for bar-based approximation.
    """
    n = len(df)
    sh, sl = find_swings(df, left=swing_left, right=swing_right)

    pdh = pdl = None
    # Previous UTC calendar day high/low
    if "ts_ms" in df.columns or "timestamp" in df.columns:
        ts_col = "ts_ms" if "ts_ms" in df.columns else "timestamp"
        if ts_col == "ts_ms":
            dt = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        else:
            dt = pd.to_datetime(df["timestamp"], utc=True)
        day_key = dt.dt.date
        last_date = day_key.iloc[-1]
        # Find previous calendar day
        unique_dates = sorted(day_key.unique())
        prev_date = None
        for d in reversed(unique_dates):
            if d < last_date:
                prev_date = d
                break
        if prev_date is not None:
            prev_mask = day_key == prev_date
            if prev_mask.any():
                prev_df = df.loc[prev_mask]
                pdh = float(prev_df["high"].max())
                pdl = float(prev_df["low"].min())
    else:
        # Fallback: bar-based (previous "day" = previous bpd bars when no timestamps)
        from hogan_bot.timeframe_utils import bars_per_day, infer_timeframe_from_candles
        tf = timeframe or infer_timeframe_from_candles(df) or "1h"
        bpd = bars_per_day(tf)
        if n >= 2 * bpd:
            prev = df.iloc[n - 2 * bpd : n - bpd]
            if len(prev) > 0:
                pdh = float(prev["high"].max())
                pdl = float(prev["low"].min())

    return {
        "swing_highs": sh,
        "swing_lows": sl,
        "equal_highs": detect_equal_highs_lows(sh, tolerance_pct=eq_tolerance_pct),
        "equal_lows": detect_equal_highs_lows(sl, tolerance_pct=eq_tolerance_pct),
        "recent_swing_high": sh[-1] if sh else None,
        "recent_swing_low": sl[-1] if sl else None,
        "prev_day_high": pdh,
        "prev_day_low": pdl,
    }


# ---------------------------------------------------------------------------
# 3. Liquidity sweep / raid detection
# ---------------------------------------------------------------------------


def detect_liquidity_sweep(
    df: pd.DataFrame,
    pools: dict[str, Any],
    lookback: int = 50,
    wick_only: bool = True,
) -> Sweep | None:
    """Scan the last *lookback* bars for a sweep of any known liquidity level.

    A **sweep** (or "raid") occurs when price wicks beyond a pool level and
    then closes back inside:

    * **Buy-side sweep** (BSL): wick above a swing high / equal-high level,
      close back below → market hunted stops above; bias shifts **bearish**.
    * **Sell-side sweep** (SSL): wick below a swing low / equal-low level,
      close back above → market hunted stops below; bias shifts **bullish**.

    When *wick_only* is ``False``, a candle that merely exceeds the level
    (even without closing back) is counted.

    Returns the **most recently swept** pool, or ``None``.

    ``side`` in the returned dict indicates the expected **trade direction**
    after the sweep (``"buy"`` = bullish setup after SSL sweep;
    ``"sell"`` = bearish setup after BSL sweep).
    """
    n = len(df)
    start = max(0, n - lookback)

    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()

    # Collect all levels tagged as buy-side (above price) or sell-side (below price)
    levels: list[tuple[str, float]] = []  # ("bsl"|"ssl", price)

    for eq in pools.get("equal_highs", []):
        levels.append(("bsl", eq["price"]))
    for eq in pools.get("equal_lows", []):
        levels.append(("ssl", eq["price"]))

    if pools.get("recent_swing_high"):
        levels.append(("bsl", pools["recent_swing_high"]["price"]))
    if pools.get("recent_swing_low"):
        levels.append(("ssl", pools["recent_swing_low"]["price"]))
    if pools.get("prev_day_high"):
        levels.append(("bsl", pools["prev_day_high"]))
    if pools.get("prev_day_low"):
        levels.append(("ssl", pools["prev_day_low"]))

    if not levels:
        return None

    best: Sweep | None = None
    best_strength = 0.0
    best_idx = -1

    for i in range(start, n):
        for ltype, level in levels:
            if level <= 0:
                continue
            if ltype == "bsl":
                wicked = high[i] > level
                closed_back = close[i] < level
                is_sweep = wicked and (closed_back if wick_only else True)
                if is_sweep:
                    strength = (high[i] - level) / level
                    if i > best_idx or (i == best_idx and strength > best_strength):
                        best_idx = i
                        best_strength = strength
                        best = {
                            "side": "sell",
                            "pool_level": level,
                            "sweep_index": i,
                            "sweep_strength": float(strength),
                        }
            else:  # ssl
                wicked = low[i] < level
                closed_back = close[i] > level
                is_sweep = wicked and (closed_back if wick_only else True)
                if is_sweep:
                    strength = (level - low[i]) / level
                    if i > best_idx or (i == best_idx and strength > best_strength):
                        best_idx = i
                        best_strength = strength
                        best = {
                            "side": "buy",
                            "pool_level": level,
                            "sweep_index": i,
                            "sweep_strength": float(strength),
                        }

    return best


# ---------------------------------------------------------------------------
# 4. Market Structure Shift (MSS)
# ---------------------------------------------------------------------------


def detect_mss(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    after_index: int,
    direction: str,
) -> MSS | None:
    """Confirm a Market Structure Shift after a liquidity sweep.

    After sweeping **buy-side liquidity** the market should break below the
    most recent swing low to confirm bearish MSS (``direction="bear"``).
    After sweeping **sell-side liquidity** the market should break above the
    most recent swing high to confirm bullish MSS (``direction="bull"``).

    Parameters
    ----------
    after_index:
        The bar index where the sweep occurred; the MSS must happen strictly
        after this point.
    direction:
        ``"bull"`` — looking for a close above the last swing high (SSL sweep
        setup).  ``"bear"`` — looking for a close below the last swing low
        (BSL sweep setup).

    Returns ``None`` when no MSS is found within the remaining bars.
    """
    close = df["close"].astype(float).to_numpy()
    n = len(df)

    if direction == "bull":
        # Find the most recent swing high at or before after_index
        candidates = [s for s in swing_highs if s["index"] <= after_index]
        if not candidates:
            return None
        target = candidates[-1]
        for i in range(after_index + 1, n):
            if close[i] > target["price"]:
                return {
                    "direction": "bull",
                    "break_index": i,
                    "broken_level": float(target["price"]),
                }

    elif direction == "bear":
        candidates = [s for s in swing_lows if s["index"] <= after_index]
        if not candidates:
            return None
        target = candidates[-1]
        for i in range(after_index + 1, n):
            if close[i] < target["price"]:
                return {
                    "direction": "bear",
                    "break_index": i,
                    "broken_level": float(target["price"]),
                }

    return None


# ---------------------------------------------------------------------------
# 5. Order Block detection
# ---------------------------------------------------------------------------


def detect_order_block(
    df: pd.DataFrame,
    mss: MSS | None,
    lookback: int = 20,
    body_only: bool = False,
) -> OrderBlock | None:
    """Identify the Order Block associated with an MSS displacement.

    * **Bearish MSS**: the last *bullish* candle before the displacement leg
      down — institutions were selling into retail buyers there.
    * **Bullish MSS**: the last *bearish* candle before the displacement leg
      up — institutions were buying into retail sellers there.

    *body_only* restricts the zone to ``[open, close]`` (candle body) rather
    than the full wick range ``[low, high]``.

    Returns ``None`` when no appropriate candle is found.
    """
    if mss is None:
        return None

    open_ = df["open"].astype(float).to_numpy()
    close_ = df["close"].astype(float).to_numpy()
    high_ = df["high"].astype(float).to_numpy()
    low_ = df["low"].astype(float).to_numpy()

    direction = mss["direction"]
    break_idx = mss["break_index"]
    start = max(0, break_idx - lookback)

    if direction == "bear":
        # Last bullish candle before the displacement
        for i in range(break_idx - 1, start - 1, -1):
            if close_[i] > open_[i]:
                return {
                    "direction": "bear",
                    "top": float(high_[i] if not body_only else max(open_[i], close_[i])),
                    "bottom": float(low_[i] if not body_only else min(open_[i], close_[i])),
                    "index": i,
                }

    elif direction == "bull":
        # Last bearish candle before the displacement
        for i in range(break_idx - 1, start - 1, -1):
            if close_[i] < open_[i]:
                return {
                    "direction": "bull",
                    "top": float(high_[i] if not body_only else max(open_[i], close_[i])),
                    "bottom": float(low_[i] if not body_only else min(open_[i], close_[i])),
                    "index": i,
                }

    return None


# ---------------------------------------------------------------------------
# 6. Premium / Discount + OTE
# ---------------------------------------------------------------------------


def dealing_range(
    df: pd.DataFrame,
    anchor_start: int,
    anchor_end: int,
) -> DealingRange:
    """Compute the high, low and equilibrium of a price range.

    ``anchor_start`` and ``anchor_end`` are inclusive positional indices.
    """
    window = df.iloc[max(0, anchor_start) : anchor_end + 1]
    h = float(window["high"].max())
    lo = float(window["low"].min())
    mid = (h + lo) / 2.0
    return {"high": h, "low": lo, "mid": mid}


def is_in_discount(price: float, range_low: float, range_high: float) -> bool:
    """Return ``True`` when *price* is in the discount (lower) half of the range.

    Discount = below equilibrium (midpoint).  Bullish entries are preferred
    in discount.
    """
    if range_high <= range_low:
        return False
    mid = (range_low + range_high) / 2.0
    return price < mid


def is_in_premium(price: float, range_low: float, range_high: float) -> bool:
    """Return ``True`` when *price* is in the premium (upper) half of the range.

    Premium = above equilibrium.  Bearish entries are preferred in premium.
    """
    if range_high <= range_low:
        return False
    mid = (range_low + range_high) / 2.0
    return price > mid


def ote_zone(
    range_low: float,
    range_high: float,
    direction: str,
    fib_low: float = 0.62,
    fib_high: float = 0.79,
) -> tuple[float, float]:
    """Compute the Optimal Trade Entry (OTE) Fibonacci retracement zone.

    For a **bullish** setup the OTE is where price retraces *down* from the
    range high into the discount (0.62–0.79 retracement from high to low)::

        zone_high = range_high - fib_low  × (range_high - range_low)
        zone_low  = range_high - fib_high × (range_high - range_low)

    For a **bearish** setup the OTE is where price retraces *up* from the
    range low into the premium::

        zone_low  = range_low + fib_low  × (range_high - range_low)
        zone_high = range_low + fib_high × (range_high - range_low)

    Returns ``(zone_low, zone_high)``.
    """
    span = range_high - range_low
    if direction == "bull":
        zone_high = range_high - fib_low * span
        zone_low = range_high - fib_high * span
    else:
        zone_low = range_low + fib_low * span
        zone_high = range_low + fib_high * span
    return float(zone_low), float(zone_high)


# ---------------------------------------------------------------------------
# 7. Time window / session filter
# ---------------------------------------------------------------------------


def _parse_windows(windows: list[tuple[str, str]]) -> list[tuple[datetime, datetime]]:
    """Parse ``("HH:MM", "HH:MM")`` string pairs into ``datetime.time`` objects."""
    parsed = []
    for start_s, end_s in windows:
        start = datetime.strptime(start_s, "%H:%M").time()
        end = datetime.strptime(end_s, "%H:%M").time()
        parsed.append((start, end))
    return parsed


def in_time_window(
    ts: Any,
    windows: list[tuple[str, str]],
    tz: str = "America/New_York",
) -> bool:
    """Return ``True`` when *ts* falls within any of the given time windows.

    Parameters
    ----------
    ts:
        A timezone-aware or naive ``datetime`` / ``pandas.Timestamp``.
    windows:
        List of ``("HH:MM", "HH:MM")`` string pairs (inclusive on both ends).
        The default Silver Bullet windows are
        ``[("03:00","04:00"), ("10:00","11:00"), ("14:00","15:00")]`` (NY time).
    tz:
        IANA timezone name for conversion.  Requires ``tzdata`` package on
        Windows (``pip install tzdata``).  Falls back to ``True`` (fail-open)
        when timezone data is unavailable.
    """
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+

        zone = ZoneInfo(tz)
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            local = ts.astimezone(zone)
        else:
            local = ts.replace(tzinfo=zone)
        local_time = local.time()
    except Exception:
        # Fail open: if timezone handling fails, don't block the signal
        return True

    parsed = _parse_windows(windows)
    return any(start <= local_time <= end for start, end in parsed)


def parse_time_windows(windows_csv: str) -> list[tuple[str, str]]:
    """Parse a CSV string like ``"03:00-04:00,10:00-11:00"`` into window pairs.

    Used to convert the ``HOGAN_ICT_TIME_WINDOWS`` env-var value.
    """
    result = []
    for part in windows_csv.split(","):
        part = part.strip()
        if "-" in part:
            start, _, end = part.partition("-")
            result.append((start.strip(), end.strip()))
    return result


# ---------------------------------------------------------------------------
# 8. ICT Setup Signal — the unified entry point for strategy.py
# ---------------------------------------------------------------------------


def ict_setup_signal(
    df: pd.DataFrame,
    *,
    swing_left: int = 2,
    swing_right: int = 2,
    eq_tolerance_pct: float = 0.0008,
    min_displacement_pct: float = 0.003,
    lookback_sweep: int = 100,
    require_time_window: bool = True,
    time_windows: list[tuple[str, str]] | None = None,
    require_pd: bool = True,
    ote_enabled: bool = False,
    ote_low: float = 0.62,
    ote_high: float = 0.79,
    tz: str = "America/New_York",
) -> tuple[str, float, dict]:
    """Generate an ICT-based directional signal for the current (last) bar.

    Implements the canonical ICT sequence:

    1. Identify liquidity pools (equal H/L, swing H/L, prev-day H/L).
    2. Detect a sweep of one side.
    3. Set **draw-on-liquidity bias** (SSL sweep → bullish; BSL sweep → bearish).
    4. Confirm MSS in that direction (break of opposing swing after the sweep).
    5. Check FVG entry (price inside a gap formed during the displacement).
    6. Check Order Block overlap.
    7. Filter by Premium/Discount (and optionally OTE Fibonacci zone).
    8. Filter by time window (Silver Bullet by default).

    Confidence is deterministic and additive:

    +0.25 sweep | +0.25 MSS | +0.20 FVG entry | +0.15 OB overlap
    +0.10 PD zone | +0.05 time window

    Returns
    -------
    (action, confidence, debug)
        ``action`` is ``"buy"``, ``"sell"``, or ``"hold"``.
        ``confidence`` is in ``[0.0, 1.0]``.
        ``debug`` is a plain dict for backtest introspection.
    """
    n = len(df)
    min_bars = max(swing_left + swing_right + 10, 30)
    if n < min_bars:
        return "hold", 0.0, {"reason": "insufficient_data", "bars": n}

    if time_windows is None:
        time_windows = SILVER_BULLET_WINDOWS

    close = df["close"].astype(float).to_numpy()
    current_price = float(close[-1])
    last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else None

    # ── 1 + 2. Liquidity pools + sweep ──────────────────────────────────────
    pools = liquidity_pools(df, swing_left=swing_left, swing_right=swing_right,
                            eq_tolerance_pct=eq_tolerance_pct)
    sw = detect_liquidity_sweep(df, pools, lookback=min(lookback_sweep, n - 5))

    if sw is None:
        return "hold", 0.0, {"reason": "no_sweep"}

    confidence = _W_SWEEP
    debug: dict = {
        "sweep": {
            "side": sw["side"],
            "level": sw["pool_level"],
            "index": sw["sweep_index"],
            "strength": sw["sweep_strength"],
        }
    }

    # ── 3. Bias ──────────────────────────────────────────────────────────────
    bias = "bull" if sw["side"] == "buy" else "bear"

    # ── 4. MSS ───────────────────────────────────────────────────────────────
    sh = pools["swing_highs"]
    sl = pools["swing_lows"]
    mss = detect_mss(df, sh, sl, sw["sweep_index"], bias)

    if mss is None:
        return "hold", confidence, {**debug, "reason": "no_mss"}

    # Minimum displacement check
    displacement = abs(close[mss["break_index"]] - mss["broken_level"]) / max(
        mss["broken_level"], 1e-9
    )
    if displacement < min_displacement_pct:
        return "hold", confidence, {
            **debug,
            "reason": "displacement_too_small",
            "displacement": float(displacement),
        }

    confidence += _W_MSS
    debug["mss"] = {
        "direction": mss["direction"],
        "level": mss["broken_level"],
        "index": mss["break_index"],
        "displacement": float(displacement),
    }

    # ── 5. FVG entry ─────────────────────────────────────────────────────────
    fvg_action = "hold"
    try:
        from hogan_bot.indicators import active_fvgs, detect_fvgs, fvg_entry_signal

        # Scan from just before the displacement to the current bar
        fvg_start = max(0, mss["break_index"] - 5)
        fvg_window = df.iloc[fvg_start:]
        if len(fvg_window) >= 3:
            all_fvgs = detect_fvgs(fvg_window, min_gap_pct=0.001)
            # Only count FVGs aligned with the bias
            aligned = [
                f for f in active_fvgs(all_fvgs)
                if f["direction"] == ("bull" if bias == "bull" else "bear")
            ]
            fvg_action = fvg_entry_signal(aligned, current_price)
    except Exception:
        pass

    if fvg_action != "hold":
        confidence += _W_FVG
        debug["fvg_entry"] = True
    else:
        debug["fvg_entry"] = False

    # ── 6. Order Block overlap ───────────────────────────────────────────────
    ob = detect_order_block(df, mss, lookback=20)
    in_ob = ob is not None and ob["bottom"] <= current_price <= ob["top"]
    if in_ob:
        confidence += _W_OB
        debug["ob"] = {"top": ob["top"], "bottom": ob["bottom"], "index": ob["index"]}
    debug["in_ob"] = in_ob

    # ── 7. Premium / Discount + OTE ─────────────────────────────────────────
    anchor_start = max(0, sw["sweep_index"] - 50)
    anchor_end = sw["sweep_index"]
    dr = dealing_range(df, anchor_start, anchor_end)
    debug["dealing_range"] = {"high": dr["high"], "low": dr["low"], "mid": dr["mid"]}

    if ote_enabled:
        zone_lo, zone_hi = ote_zone(dr["low"], dr["high"], bias, fib_low=ote_low, fib_high=ote_high)
        in_zone = zone_lo <= current_price <= zone_hi
        debug["ote_zone"] = {"low": zone_lo, "high": zone_hi, "in_zone": in_zone}
        if require_pd:
            pd_ok = in_zone
        else:
            pd_ok = True
    elif require_pd:
        if bias == "bull":
            pd_ok = is_in_discount(current_price, dr["low"], dr["high"])
        else:
            pd_ok = is_in_premium(current_price, dr["low"], dr["high"])
    else:
        pd_ok = True

    if pd_ok:
        confidence += _W_PD
    debug["in_pd_zone"] = pd_ok

    # ── 8. Time window ───────────────────────────────────────────────────────
    if last_ts is not None and require_time_window:
        in_window = in_time_window(last_ts, time_windows, tz=tz)
    else:
        in_window = True  # no timestamp or window not required → pass

    if in_window:
        confidence += _W_TIME
    debug["in_time_window"] = in_window

    # ── Entry condition ──────────────────────────────────────────────────────
    # Core trigger: price must be in the FVG and/or OB zone to enter
    if not fvg_action != "hold" and not in_ob:
        return "hold", min(1.0, confidence), {**debug, "reason": "not_in_entry_zone"}

    # Time-window gate (hard block only when explicitly required)
    if require_time_window and not in_window:
        return "hold", min(1.0, confidence), {**debug, "reason": "outside_time_window"}

    action = "buy" if bias == "bull" else "sell"
    return action, min(1.0, confidence), debug
