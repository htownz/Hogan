"""Experimental social/NLP and whale-flow feature joins.

The features in this module are challenger-only. They are intentionally not
part of ``CHAMPION_FEATURE_COLUMNS`` and are only appended to full feature
sets when callers opt in with ``include_experimental``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from hogan_bot.feature_registry import EXPERIMENTAL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

SOCIAL_WHALE_FEATURE_NAMES: list[str] = list(EXPERIMENTAL_FEATURE_COLUMNS)

_BTC_SUPPLY = 21_000_000.0


def _identity(value: float) -> float:
    return float(value)


def _clip_unit(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _volume_norm_to_anomaly(value: float) -> float:
    return _clip_unit(float(value) - 1.0)


def _btc_flow_to_supply_norm(value: float) -> float:
    return _clip_unit(float(value) / _BTC_SUPPLY)


_MetricSource = tuple[str, Callable[[float], float]]

_FEATURE_SOURCES: dict[str, list[_MetricSource]] = {
    # Explicit challenger feature names win when present; existing fetchers
    # provide the fallback aliases below.
    "social_nlp_sentiment_score": [
        ("social_nlp_sentiment_score", _clip_unit),
        ("news_sentiment_score", _clip_unit),
    ],
    "social_volume_anomaly": [
        ("social_volume_anomaly", _clip_unit),
        ("santiment_social_vol_chg", _clip_unit),
        ("news_volume_norm", _volume_norm_to_anomaly),
    ],
    "whale_exchange_flow_norm": [
        ("whale_exchange_flow_norm", _clip_unit),
        ("glassnode_exchange_netflow", _clip_unit),
        ("dune_btc_exchange_netflow", _btc_flow_to_supply_norm),
    ],
    "whale_large_tx_count_norm": [
        ("whale_large_tx_count_norm", _clip_unit),
        ("dune_btc_whale_pct", _clip_unit),
    ],
}


def _load_metric_table(conn) -> pd.DataFrame:
    feature_frames: dict[str, pd.DataFrame] = {}
    all_ts: list[pd.Series] = []
    for feature, sources in _FEATURE_SOURCES.items():
        candidates: list[pd.DataFrame] = []
        for priority, (metric, transform) in enumerate(sources):
            try:
                df = pd.read_sql_query(
                    """
                    SELECT date, value
                    FROM onchain_metrics
                    WHERE symbol = 'BTC/USD' AND metric = ?
                    ORDER BY date
                    """,
                    conn,
                    params=(metric,),
                )
            except Exception as exc:
                logger.debug("social/whale metric load failed for %s: %s", metric, exc)
                continue
            if df.empty:
                continue
            df["ts_ms"] = (
                pd.to_datetime(df["date"], utc=True).astype("int64") // 1_000_000
            )
            df[feature] = df["value"].astype(float).map(transform)
            df["_priority"] = priority
            candidates.append(df[["ts_ms", feature, "_priority"]])
        if not candidates:
            continue
        feature_df = pd.concat(candidates, ignore_index=True)
        feature_df = (
            feature_df.sort_values(["ts_ms", "_priority"])
            .drop_duplicates(subset=["ts_ms"], keep="first")
            .drop(columns=["_priority"])
            .sort_values("ts_ms")
        )
        feature_frames[feature] = feature_df
        all_ts.append(feature_df["ts_ms"])

    if not all_ts:
        return pd.DataFrame(columns=["ts_ms", *SOCIAL_WHALE_FEATURE_NAMES])

    ts = pd.concat(all_ts, ignore_index=True).drop_duplicates().sort_values()
    out = pd.DataFrame({"ts_ms": ts.astype("int64")})
    for feature in SOCIAL_WHALE_FEATURE_NAMES:
        feature_df = feature_frames.get(feature)
        if feature_df is None or feature_df.empty:
            out[feature] = 0.0
            continue
        out = pd.merge_asof(
            out.sort_values("ts_ms"),
            feature_df.sort_values("ts_ms"),
            on="ts_ms",
            direction="backward",
        )
        out[feature] = out[feature].ffill().fillna(0.0).astype(float)
    return out[["ts_ms", *SOCIAL_WHALE_FEATURE_NAMES]].sort_values("ts_ms")


def add_social_whale_features(frame: pd.DataFrame, conn) -> pd.DataFrame:
    """Point-in-time merge social/whale scalar features into a candle frame."""
    table = _load_metric_table(conn)
    out = frame.copy()
    if table.empty:
        for col in SOCIAL_WHALE_FEATURE_NAMES:
            out[col] = 0.0
        return out
    if "ts_ms" not in out.columns:
        if "timestamp" not in out.columns:
            for col in SOCIAL_WHALE_FEATURE_NAMES:
                out[col] = 0.0
            return out
        out["ts_ms"] = pd.to_datetime(out["timestamp"], utc=True).astype("int64") // 1_000_000
    merged = pd.merge_asof(
        out.sort_values("ts_ms"),
        table.sort_values("ts_ms"),
        on="ts_ms",
        direction="backward",
    )
    for col in SOCIAL_WHALE_FEATURE_NAMES:
        merged[col] = merged[col].ffill().fillna(0.0).astype(float)
    return merged


def get_social_whale_feature_row(conn, ts_ms: int | None = None) -> list[float]:
    """Return the latest point-in-time social/whale feature vector."""
    table = _load_metric_table(conn)
    if table.empty:
        return [0.0] * len(SOCIAL_WHALE_FEATURE_NAMES)
    if ts_ms is not None:
        table = table[table["ts_ms"] <= int(ts_ms)]
    if table.empty:
        return [0.0] * len(SOCIAL_WHALE_FEATURE_NAMES)
    row = table.iloc[-1]
    return [float(row.get(col, 0.0) or 0.0) for col in SOCIAL_WHALE_FEATURE_NAMES]
