"""Recursive Formula Checker — Phase 7 (Freqtrade-inspired).

Detects features computed with rolling windows that inadvertently reference
themselves (recursive / circular feature definitions), which can silently
corrupt training data by introducing smooth-looking but nonsensical values.

How it works
------------
1. Computes indicators on a real candle sequence.
2. Replaces one candle's OHLCV values with random noise.
3. Recomputes indicators.
4. Checks whether the perturbation propagates for MORE than the expected
   number of bars (the window length of the indicator).

If the perturbation at bar X is still visible at bar X + window_length × 2,
the indicator is likely recursive/self-referential.

Also performs a NaN audit: a well-formed indicator should have exactly
``window_length - 1`` leading NaN values, not more (which would indicate
incorrect ``min_periods`` or unnecessary dependencies).

Usage::

    python -m hogan_bot.recursive_check --db data/hogan.db
    python -m hogan_bot.recursive_check --symbol BTC/USD --timeframe 5m
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected window sizes per indicator (max periods before stabilization)
_EXPECTED_WINDOWS = {
    "sma_10": 10, "sma_20": 20, "sma_50": 50,
    "ema_9": 9, "ema_21": 21,
    "rsi_14": 14,
    "atr_14": 14,
    "macd_hist": 35,    # 26 + 9
    "bb_upper": 20, "bb_lower": 20, "bb_pct_b": 20,
    "vol_ratio": 20,
}
_DEFAULT_PERTURB_IDX = 30    # perturbation bar index (must be > max window)
_DEFAULT_SETTLE_MULTIPLIER = 2  # indicator should settle within window × this


def _compute_indicators_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Simple indicator computation without FVG (faster, for recursive check)."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out.get("volume", pd.Series(1.0, index=out.index))

    out["sma_10"] = close.rolling(10).mean()
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["ema_9"] = close.ewm(span=9, adjust=False).mean()
    out["ema_21"] = close.ewm(span=21, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    hl = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low - close.shift(1)).abs()
    out["atr_14"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd_line - signal_line

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_pct_b"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)

    vol_ma = volume.rolling(20).mean()
    out["vol_ratio"] = volume / (vol_ma + 1e-9)

    return out


def check_recursive(
    candles: pd.DataFrame,
    perturb_idx: int = _DEFAULT_PERTURB_IDX,
    settle_multiplier: int = _DEFAULT_SETTLE_MULTIPLIER,
) -> dict[str, Any]:
    """Check for recursive / self-referential indicator formulas.

    Parameters
    ----------
    candles:
        Candle DataFrame (``open``, ``high``, ``low``, ``close``, ``volume``).
    perturb_idx:
        Bar index to perturb (must be > largest expected window size).
    settle_multiplier:
        How many window-lengths to wait for the effect to dissipate.

    Returns
    -------
    dict with ``ok``, ``recursive_columns``, ``clean_columns``, ``details``.
    """
    if len(candles) < perturb_idx + 100:
        return {
            "ok": True,
            "recursive_columns": [],
            "clean_columns": [],
            "warning": f"Too few candles (need > {perturb_idx + 100})",
        }

    # Baseline
    df_base = candles.copy().reset_index(drop=True)
    ind_base = _compute_indicators_simple(df_base)

    # Perturbed — inject large random noise at perturb_idx
    rng = np.random.default_rng(42)
    df_perturbed = df_base.copy()
    noise_scale = float(df_base["close"].median()) * 0.5
    df_perturbed.loc[perturb_idx, "open"] = rng.uniform(1, noise_scale)
    df_perturbed.loc[perturb_idx, "high"] = rng.uniform(1, noise_scale)
    df_perturbed.loc[perturb_idx, "low"] = rng.uniform(1, noise_scale)
    df_perturbed.loc[perturb_idx, "close"] = rng.uniform(1, noise_scale)
    df_perturbed.loc[perturb_idx, "volume"] = rng.uniform(1e3, 1e6)
    ind_pert = _compute_indicators_simple(df_perturbed)

    recursive: list[str] = []
    clean: list[str] = []
    details: dict[str, Any] = {}

    indicator_cols = [
        c for c in _EXPECTED_WINDOWS.keys()
        if c in ind_base.columns and c in ind_pert.columns
    ]

    for col in indicator_cols:
        window = _EXPECTED_WINDOWS.get(col, 20)
        settle_bar = perturb_idx + window * settle_multiplier

        if settle_bar >= len(ind_base):
            continue

        val_base = float(ind_base[col].iloc[settle_bar])
        val_pert = float(ind_pert[col].iloc[settle_bar])

        if np.isnan(val_base) or np.isnan(val_pert):
            continue

        diff = abs(val_base - val_pert)
        relative_diff = diff / (abs(val_base) + 1e-9)

        details[col] = {
            "window": window,
            "settle_bar": settle_bar,
            "val_base": round(val_base, 6),
            "val_pert": round(val_pert, 6),
            "relative_diff": round(relative_diff, 6),
        }

        if relative_diff > 1e-6:
            recursive.append(col)
            logger.warning(
                "RECURSIVE indicator: %s diff still %.4e at bar %d (window=%d)",
                col, relative_diff, settle_bar, window,
            )
        else:
            clean.append(col)

    # NaN audit
    nan_warnings: list[str] = []
    for col in indicator_cols:
        window = _EXPECTED_WINDOWS.get(col, 20)
        nan_count = int(ind_base[col].isna().sum())
        if nan_count > window + 5:  # allow small slack
            nan_warnings.append(f"{col}: {nan_count} NaN (expected ≤{window + 5})")

    result = {
        "ok": len(recursive) == 0,
        "recursive_columns": recursive,
        "clean_columns": clean,
        "nan_warnings": nan_warnings,
        "details": {k: v for k, v in details.items() if k in recursive},
        "total_checked": len(indicator_cols),
    }

    if recursive:
        logger.error(
            "Recursive formulas found in %d/%d columns: %s",
            len(recursive), len(indicator_cols), recursive,
        )
    else:
        logger.info(
            "Recursive check PASSED: %d columns clean%s.",
            len(clean),
            f", {len(nan_warnings)} NaN warnings" if nan_warnings else "",
        )

    return result


# ---------------------------------------------------------------------------
# Dry-run validation gate (used in trader_service.py before each session)
# ---------------------------------------------------------------------------
def dry_run_validate(
    candles: pd.DataFrame,
    n_loops: int = 10,
    min_non_hold_ratio: float = 0.1,
) -> dict[str, Any]:
    """Execute the full signal pipeline on the last N bars without writing trades.

    Confirms:
    1. No exceptions during signal generation.
    2. At least ``min_non_hold_ratio`` of signals are non-trivial (buy or sell).

    Returns ``{"ok": True/False, "non_hold_ratio": ..., "errors": [...]}``
    """
    from hogan_bot.config import load_config
    from hogan_bot.strategy import generate_signal

    cfg = load_config()
    errors: list[str] = []
    actions: list[str] = []

    for i in range(max(1, len(candles) - n_loops), len(candles)):
        window = candles.iloc[max(0, i - cfg.long_ma_window):i + 1]
        if len(window) < max(cfg.long_ma_window, 20):
            continue
        try:
            signal = generate_signal(
                window,
                short_window=cfg.short_ma_window,
                long_window=cfg.long_ma_window,
                volume_window=cfg.volume_window,
                volume_threshold=cfg.volume_threshold,
                use_ema_clouds=cfg.use_ema_clouds,
                ema_fast_short=cfg.ema_fast_short,
                ema_fast_long=cfg.ema_fast_long,
                ema_slow_short=cfg.ema_slow_short,
                ema_slow_long=cfg.ema_slow_long,
                use_fvg=cfg.use_fvg,
                fvg_min_gap_pct=cfg.fvg_min_gap_pct,
                signal_mode=cfg.signal_mode,
            )
            actions.append(signal.action)
        except Exception as exc:
            errors.append(f"bar {i}: {exc}")

    if not actions:
        return {"ok": False, "errors": ["No bars processed"], "non_hold_ratio": 0.0}

    non_hold = sum(1 for a in actions if a != "hold")
    non_hold_ratio = non_hold / len(actions)
    ok = len(errors) == 0 and non_hold_ratio >= min_non_hold_ratio

    return {
        "ok": ok,
        "actions": actions,
        "non_hold_ratio": round(non_hold_ratio, 4),
        "non_hold_count": non_hold,
        "total_bars": len(actions),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Recursive formula checker")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--dry-run-validate", action="store_true",
                   help="Also run the dry-run validation gate")
    args = p.parse_args()

    from hogan_bot.storage import get_connection, load_candles
    conn = get_connection(args.db)
    candles = load_candles(conn, args.symbol, args.timeframe, limit=args.limit)
    conn.close()

    if candles.empty:
        print(f"No candles for {args.symbol}/{args.timeframe}")
        return

    result = check_recursive(candles)
    print("Recursive check:")
    print(json.dumps(result, indent=2, default=str))

    if args.dry_run_validate:
        print("\nDry-run validation:")
        drv = dry_run_validate(candles)
        print(json.dumps(drv, indent=2, default=str))
        if not drv["ok"]:
            import sys
            sys.exit(1)

    if not result["ok"]:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    _main()
