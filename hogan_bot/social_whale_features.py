"""Experimental social/NLP and whale-flow feature joins.

The features in this module are challenger-only. They are intentionally not
part of ``CHAMPION_FEATURE_COLUMNS`` and are only appended to full feature
sets when callers opt in with ``include_experimental``.
"""
from __future__ import annotations

import logging

import pandas as pd

from hogan_bot.feature_registry import EXPERIMENTAL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

SOCIAL_WHALE_FEATURE_NAMES: list[str] = list(EXPERIMENTAL_FEATURE_COLUMNS)

_METRIC_TO_FEATURE = {
    "social_nlp_sentiment_score": "social_nlp_sentiment_score",
    "social_volume_anomaly": "social_volume_anomaly",
    "whale_exchange_flow_norm": "whale_exchange_flow_norm",
    "whale_large_tx_count_norm": "whale_large_tx_count_norm",
}


def _load_metric_table(conn) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metric, feature in _METRIC_TO_FEATURE.items():
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
        df = df[["ts_ms", "value"]].rename(columns={"value": feature})
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ts_ms", *SOCIAL_WHALE_FEATURE_NAMES])
    out = frames[0]
    for nxt in frames[1:]:
        out = pd.merge_asof(
            out.sort_values("ts_ms"),
            nxt.sort_values("ts_ms"),
            on="ts_ms",
            direction="backward",
        )
    for col in SOCIAL_WHALE_FEATURE_NAMES:
        if col not in out.columns:
            out[col] = 0.0
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
