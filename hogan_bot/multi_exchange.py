"""Multi-exchange data aggregation for Hogan.

This module provides utilities for fetching and combining market data from
several exchanges simultaneously.  All network calls are issued in parallel
using a thread pool so wall-clock latency is approximately that of the
slowest exchange.

Typical use-cases
-----------------
* **Data richness** — average OHLCV across Binance + Bybit + Kraken for a
  more representative price series.
* **Arbitrage monitoring** — spot when the same pair is trading at a premium
  on one venue vs another.
* **Funding-rate signals** — fetch perpetual funding rates (e.g. Binance
  BTCUSDT) as an extra feature for the ML model.
* **Volume cross-check** — compare reported volume across exchanges to
  filter out low-liquidity signals.

Examples
--------
>>> from hogan_bot.multi_exchange import fetch_multi_ohlcv, vwap_composite
>>> dfs = fetch_multi_ohlcv("BTC/USDT", ["binance", "bybit"], timeframe="1h")
>>> composite = vwap_composite(dfs)

>>> from hogan_bot.multi_exchange import fetch_funding_rates
>>> rates = fetch_funding_rates("BTC/USDT:USDT", ["binance", "bybit"])
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from hogan_bot.exchange import ExchangeClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OHLCV aggregation
# ---------------------------------------------------------------------------


def fetch_multi_ohlcv(
    symbol: str,
    exchange_ids: list[str],
    timeframe: str = "1h",
    limit: int = 500,
    workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV bars from multiple exchanges in parallel.

    Returns a dict mapping ``exchange_id → DataFrame``.  Exchanges that fail
    (e.g. symbol not listed) are silently excluded from the result.

    Parameters
    ----------
    symbol:
        Trading pair.  Spot pairs (``"BTC/USDT"``) work on most exchanges;
        perpetual pairs (``"BTC/USDT:USDT"``) work on derivative venues.
    exchange_ids:
        List of CCXT exchange IDs, e.g. ``["binance", "bybit", "kraken"]``.
    timeframe:
        OHLCV bar interval recognised by CCXT, e.g. ``"5m"``.
    limit:
        Number of bars to request from each exchange.
    workers:
        Thread pool size.  Defaults to ``len(exchange_ids)``.
    """
    results: dict[str, pd.DataFrame] = {}
    n = workers or len(exchange_ids)

    def _fetch(eid: str) -> tuple[str, pd.DataFrame]:
        client = ExchangeClient(eid)
        return eid, client.fetch_ohlcv_df(symbol, timeframe=timeframe, limit=limit)

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(_fetch, eid): eid for eid in exchange_ids}
        for future in as_completed(futures):
            eid = futures[future]
            try:
                key, df = future.result()
                if not df.empty:
                    results[key] = df
                else:
                    logger.debug("%s returned empty OHLCV for %s", eid, symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning("fetch_multi_ohlcv [%s / %s] failed: %s", eid, symbol, exc)

    return results


def vwap_composite(
    dfs: dict[str, pd.DataFrame],
    align_on: str = "timestamp",
) -> pd.DataFrame:
    """Compute a volume-weighted average OHLCV across multiple exchange frames.

    Only timestamps that appear in **all** supplied DataFrames are included
    (inner join on *align_on*).

    The composite ``close`` is the volume-weighted mean close price::

        composite_close[t] = Σ(close_i[t] * volume_i[t]) / Σ(volume_i[t])

    ``high`` / ``low`` are taken as the max / min across exchanges.
    ``open`` is the volume-weighted mean open.
    ``volume`` is the sum across exchanges (total market volume).

    Returns an empty DataFrame when fewer than two sources are provided or no
    overlapping timestamps exist.
    """
    if len(dfs) < 1:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    if len(dfs) == 1:
        return next(iter(dfs.values())).copy()

    merged: pd.DataFrame | None = None
    for eid, df in dfs.items():
        df2 = df.set_index(align_on)[["open", "high", "low", "close", "volume"]].copy()
        df2.columns = [f"{c}_{eid}" for c in df2.columns]
        merged = df2 if merged is None else merged.join(df2, how="inner")

    if merged is None or merged.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    eids = list(dfs.keys())
    vol_cols = [f"volume_{e}" for e in eids]
    close_cols = [f"close_{e}" for e in eids]
    open_cols = [f"open_{e}" for e in eids]
    high_cols = [f"high_{e}" for e in eids]
    low_cols = [f"low_{e}" for e in eids]

    total_vol = merged[vol_cols].sum(axis=1).clip(lower=1e-12)

    out = pd.DataFrame(index=merged.index)
    out["open"] = (merged[open_cols].multiply(merged[vol_cols].values).sum(axis=1) / total_vol)
    out["high"] = merged[high_cols].max(axis=1)
    out["low"] = merged[low_cols].min(axis=1)
    out["close"] = (merged[close_cols].multiply(merged[vol_cols].values).sum(axis=1) / total_vol)
    out["volume"] = merged[vol_cols].sum(axis=1)
    out = out.reset_index().rename(columns={"index": "timestamp"})
    return out


def price_spread(
    dfs: dict[str, pd.DataFrame],
    align_on: str = "timestamp",
) -> pd.DataFrame:
    """Compute the cross-exchange price spread on overlapping timestamps.

    Returns a DataFrame with columns:

    * ``timestamp``
    * ``max_close`` / ``min_close`` — highest/lowest close across venues
    * ``spread_pct`` — ``(max_close − min_close) / mean_close × 100``
    * one ``close_{exchange_id}`` column per exchange

    A rising ``spread_pct`` can signal arbitrage opportunities or
    deteriorating cross-venue liquidity.
    """
    if not dfs:
        return pd.DataFrame()

    merged: pd.DataFrame | None = None
    for eid, df in dfs.items():
        col = df.set_index(align_on)[["close"]].rename(columns={"close": f"close_{eid}"})
        merged = col if merged is None else merged.join(col, how="inner")

    if merged is None or merged.empty:
        return pd.DataFrame()

    close_cols = [c for c in merged.columns if c.startswith("close_")]
    merged["max_close"] = merged[close_cols].max(axis=1)
    merged["min_close"] = merged[close_cols].min(axis=1)
    mean_close = merged[close_cols].mean(axis=1).clip(lower=1e-12)
    merged["spread_pct"] = (merged["max_close"] - merged["min_close"]) / mean_close * 100
    return merged.reset_index().rename(columns={"index": "timestamp"})


# ---------------------------------------------------------------------------
# Funding rates / open interest
# ---------------------------------------------------------------------------


def fetch_funding_rates(
    symbol: str,
    exchange_ids: list[str],
    workers: int | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Fetch the current funding rate from multiple derivative exchanges.

    Returns a dict mapping ``exchange_id → funding_rate_dict`` (or ``None``
    when the exchange/symbol does not support funding rates).

    Typical usage::

        rates = fetch_funding_rates("BTC/USDT:USDT", ["binance", "bybit"])
        for eid, r in rates.items():
            if r:
                print(eid, r["fundingRate"])
    """
    results: dict[str, dict[str, Any] | None] = {}
    n = workers or len(exchange_ids)

    def _fetch(eid: str) -> tuple[str, dict | None]:
        return eid, ExchangeClient(eid).fetch_funding_rate(symbol)

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(_fetch, eid): eid for eid in exchange_ids}
        for future in as_completed(futures):
            eid = futures[future]
            try:
                key, rate = future.result()
                results[key] = rate
            except Exception as exc:  # noqa: BLE001
                logger.warning("fetch_funding_rates [%s / %s] failed: %s", eid, symbol, exc)
                results[eid] = None

    return results


def fetch_open_interests(
    symbol: str,
    exchange_ids: list[str],
    workers: int | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Fetch current open interest from multiple derivative exchanges.

    Returns a dict mapping ``exchange_id → open_interest_dict`` (or ``None``
    when unsupported).
    """
    results: dict[str, dict[str, Any] | None] = {}
    n = workers or len(exchange_ids)

    def _fetch(eid: str) -> tuple[str, dict | None]:
        return eid, ExchangeClient(eid).fetch_open_interest(symbol)

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(_fetch, eid): eid for eid in exchange_ids}
        for future in as_completed(futures):
            eid = futures[future]
            try:
                key, oi = future.result()
                results[key] = oi
            except Exception as exc:  # noqa: BLE001
                logger.warning("fetch_open_interests [%s / %s] failed: %s", eid, symbol, exc)
                results[eid] = None

    return results


# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------


def fetch_tickers(
    symbol: str,
    exchange_ids: list[str],
    workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch the latest ticker from multiple exchanges in parallel.

    Returns a dict mapping ``exchange_id → ticker_dict``.  Exchanges that
    fail are excluded from the result.
    """
    results: dict[str, dict[str, Any]] = {}
    n = workers or len(exchange_ids)

    def _fetch(eid: str) -> tuple[str, dict]:
        return eid, ExchangeClient(eid).fetch_ticker(symbol)

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(_fetch, eid): eid for eid in exchange_ids}
        for future in as_completed(futures):
            eid = futures[future]
            try:
                key, ticker = future.result()
                results[key] = ticker
            except Exception as exc:  # noqa: BLE001
                logger.warning("fetch_tickers [%s / %s] failed: %s", eid, symbol, exc)

    return results


def composite_last_price(tickers: dict[str, dict[str, Any]]) -> float | None:
    """Return the volume-weighted last price across ticker snapshots.

    Uses ``quoteVolume`` as weight when available, otherwise falls back to a
    simple mean.  Returns ``None`` when *tickers* is empty.
    """
    if not tickers:
        return None

    prices, weights = [], []
    for t in tickers.values():
        last = t.get("last")
        vol = t.get("quoteVolume") or t.get("baseVolume")
        if last is not None:
            prices.append(float(last))
            weights.append(float(vol) if vol else 1.0)

    if not prices:
        return None

    w = np.array(weights)
    p = np.array(prices)
    return float(np.dot(p, w) / w.sum())
