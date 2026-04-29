"""Research Polymarket, macro, social/news, and BTC lead-lag relationships.

This module is research-only. It does not open shadow trades, change authority
modes, or place real orders. It builds a point-in-time feature panel and writes
correlation/hypothesis reports for operator review.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hogan_bot.storage import get_connection, load_candles

_MACRO_ASSETS = {
    "spy": "SPY/USD",
    "qqq": "QQQ/USD",
    "uup": "UUP/USD",
    "vix": "VIX/USD",
    "gld": "GLD/USD",
    "tnx": "TNX/USD",
}
_ONCHAIN_METRICS = (
    "news_sentiment_score",
    "news_volume_norm",
    "fear_greed_value",
    "santiment_social_vol_chg",
    "santiment_dev_activity_chg",
    "social_nlp_sentiment_score",
    "social_volume_anomaly",
    "whale_exchange_flow_norm",
)


@dataclass(frozen=True)
class CorrelationResult:
    feature: str
    horizon: str
    samples: int
    correlation: float
    directional_hit_rate: float
    top_bucket_avg_forward_return: float

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "horizon": self.horizon,
            "samples": self.samples,
            "correlation": round(self.correlation, 6),
            "directional_hit_rate": round(self.directional_hit_rate, 6),
            "top_bucket_avg_forward_return": round(self.top_bucket_avg_forward_return, 6),
        }


@dataclass(frozen=True)
class HypothesisResult:
    name: str
    horizon: str
    samples: int
    hit_rate: float
    avg_forward_return: float
    notes: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "horizon": self.horizon,
            "samples": self.samples,
            "hit_rate": round(self.hit_rate, 6),
            "avg_forward_return": round(self.avg_forward_return, 6),
            "notes": self.notes,
        }


def _horizon_to_bars(horizon: str) -> int:
    value = horizon.strip().lower()
    if value.endswith("h"):
        return max(1, int(value[:-1]))
    if value.endswith("d"):
        return max(1, int(value[:-1]) * 24)
    raise ValueError(f"unsupported horizon: {horizon}")


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or float(left.std(ddof=0)) == 0.0 or float(right.std(ddof=0)) == 0.0:
        return 0.0
    value = left.corr(right)
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return float(value)


def _load_macro_features(conn, timeframe: str, limit: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for prefix, symbol in _MACRO_ASSETS.items():
        try:
            df = load_candles(conn, symbol, timeframe, limit=limit)
        except Exception:
            continue
        if df.empty:
            continue
        out = df[["ts_ms", "close"]].copy()
        close = out["close"].astype(float)
        out[f"{prefix}_ret_1h"] = close.pct_change()
        out[f"{prefix}_ret_24h"] = close.pct_change(24)
        out[f"{prefix}_trend_24h"] = (close > close.rolling(24).mean()).astype(float)
        frames.append(out.drop(columns=["close"]))
    if not frames:
        return pd.DataFrame(columns=["ts_ms"])
    merged = frames[0].sort_values("ts_ms")
    for frame in frames[1:]:
        merged = pd.merge_asof(
            merged.sort_values("ts_ms"),
            frame.sort_values("ts_ms"),
            on="ts_ms",
            direction="backward",
        )
    return merged


def _load_onchain_features(conn, symbol: str) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in _ONCHAIN_METRICS)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT date, metric, value
            FROM onchain_metrics
            WHERE symbol = ? AND metric IN ({placeholders})
            ORDER BY date
            """,
            conn,
            params=(symbol, *_ONCHAIN_METRICS),
        )
    except Exception:
        return pd.DataFrame(columns=["ts_ms"])
    if df.empty:
        return pd.DataFrame(columns=["ts_ms"])
    df["ts_ms"] = pd.to_datetime(df["date"], utc=True).astype("int64") // 1_000_000
    pivot = (
        df.pivot_table(index="ts_ms", columns="metric", values="value", aggfunc="last")
        .reset_index()
        .sort_values("ts_ms")
    )
    return pivot


def _load_polymarket_features(conn, symbol: str) -> pd.DataFrame:
    try:
        df = pd.read_sql_query(
            """
            SELECT ts_ms, category_id, yes_probability, data_quality_score, eligibility
            FROM polymarket_market_snapshots
            WHERE symbol = ? AND yes_probability IS NOT NULL
            ORDER BY ts_ms
            """,
            conn,
            params=(symbol,),
        )
    except Exception:
        return pd.DataFrame(columns=["ts_ms"])
    if df.empty:
        return pd.DataFrame(columns=["ts_ms"])
    df["yes_probability"] = df["yes_probability"].astype(float)
    df["data_quality_score"] = df["data_quality_score"].astype(float)
    base = df.groupby("ts_ms").agg(
        poly_snapshot_count=("yes_probability", "count"),
        poly_avg_yes_probability=("yes_probability", "mean"),
        poly_avg_data_quality=("data_quality_score", "mean"),
    )
    for category_id, group in df.groupby("category_id"):
        safe = str(category_id or "unknown").replace("-", "_")
        category = group.groupby("ts_ms").agg(
            **{
                f"poly_{safe}_prob": ("yes_probability", "mean"),
                f"poly_{safe}_count": ("yes_probability", "count"),
            }
        )
        base = base.join(category, how="outer")
    out = base.reset_index().sort_values("ts_ms")
    prob_cols = [col for col in out.columns if col.endswith("_prob") or col == "poly_avg_yes_probability"]
    for col in prob_cols:
        out[f"{col}_change"] = out[col].astype(float).diff()
    return out


def build_feature_panel(
    conn,
    *,
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    limit: int = 5000,
    horizons: tuple[str, ...] = ("1h", "4h", "1d", "3d"),
) -> pd.DataFrame:
    """Build a point-in-time correlation panel with forward BTC returns."""
    btc = load_candles(conn, symbol, timeframe, limit=limit)
    if btc.empty:
        return pd.DataFrame()
    panel = btc[["ts_ms", "timestamp", "close", "volume"]].copy()
    close = panel["close"].astype(float)
    panel["btc_ret_1h"] = close.pct_change()
    panel["btc_ret_4h"] = close.pct_change(4)
    panel["btc_ret_24h"] = close.pct_change(24)
    panel["btc_volatility_24h"] = panel["btc_ret_1h"].rolling(24).std()
    for horizon in horizons:
        bars = _horizon_to_bars(horizon)
        panel[f"fwd_btc_ret_{horizon}"] = close.shift(-bars) / close - 1.0

    for feature_frame in (
        _load_macro_features(conn, timeframe, limit),
        _load_onchain_features(conn, symbol),
        _load_polymarket_features(conn, symbol),
    ):
        if feature_frame.empty or "ts_ms" not in feature_frame:
            continue
        panel = pd.merge_asof(
            panel.sort_values("ts_ms"),
            feature_frame.sort_values("ts_ms"),
            on="ts_ms",
            direction="backward",
        )
    return panel.sort_values("ts_ms").reset_index(drop=True)


def lead_lag_correlations(
    panel: pd.DataFrame,
    *,
    horizons: tuple[str, ...],
    min_samples: int = 20,
) -> list[CorrelationResult]:
    """Measure feature correlations against future BTC returns."""
    if panel.empty:
        return []
    excluded = {"ts_ms", "timestamp", "close", "volume"}
    target_cols = {f"fwd_btc_ret_{horizon}" for horizon in horizons}
    feature_cols = [
        col
        for col in panel.select_dtypes(include="number").columns
        if col not in excluded and col not in target_cols and not col.startswith("fwd_")
    ]
    results: list[CorrelationResult] = []
    for horizon in horizons:
        target_col = f"fwd_btc_ret_{horizon}"
        if target_col not in panel:
            continue
        for feature in feature_cols:
            subset = panel[[feature, target_col]].dropna()
            if len(subset) < min_samples:
                continue
            x = subset[feature].astype(float)
            y = subset[target_col].astype(float)
            corr = _safe_corr(x, y)
            signed = subset[(x != 0) & (y != 0)]
            hit_rate = 0.0
            if not signed.empty:
                hit_rate = float(((signed[feature] > 0) == (signed[target_col] > 0)).mean())
            threshold = float(x.quantile(0.75))
            top = subset[x >= threshold]
            top_avg = float(top[target_col].mean()) if not top.empty else 0.0
            results.append(CorrelationResult(
                feature=feature,
                horizon=horizon,
                samples=len(subset),
                correlation=corr,
                directional_hit_rate=hit_rate,
                top_bucket_avg_forward_return=top_avg,
            ))
    return sorted(results, key=lambda result: abs(result.correlation), reverse=True)


def evaluate_strategy_hypotheses(
    panel: pd.DataFrame,
    *,
    horizons: tuple[str, ...],
    min_samples: int = 5,
) -> list[HypothesisResult]:
    """Evaluate simple shadow-only strategy hypotheses."""
    if panel.empty:
        return []
    masks: dict[str, pd.Series] = {}
    masks["risk_on_confirmation"] = (
        (panel.get("spy_ret_1h", 0) > 0)
        & (panel.get("qqq_ret_1h", 0) > 0)
        & (panel.get("uup_ret_1h", 0) < 0)
        & (panel.get("poly_crypto_price_target_prob_change", 0) > 0)
    )
    masks["risk_off_veto"] = (
        (panel.get("vix_ret_1h", 0) > 0)
        & (panel.get("uup_ret_1h", 0) > 0)
        & (panel.get("poly_crypto_price_target_prob_change", 0) > 0)
    )
    masks["social_news_exhaustion"] = (
        (panel.get("news_volume_norm", 0) > 1.5)
        & (panel.get("news_sentiment_score", 0) > 0.25)
        & (panel.get("btc_ret_24h", 0) > 0.03)
    )
    masks["contrarian_fear_setup"] = (
        (panel.get("fear_greed_value", 50) < 25)
        & (panel.get("spy_ret_1h", 0) >= 0)
        & (panel.get("poly_crypto_price_target_prob_change", 0) >= 0)
    )
    masks["polymarket_mispricing_monitor"] = (
        (panel.get("poly_crypto_price_target_prob_change", 0) > 0)
        & (panel.get("spy_ret_1h", 0) > 0)
        & (panel.get("qqq_ret_1h", 0) > 0)
    )

    results: list[HypothesisResult] = []
    for horizon in horizons:
        target_col = f"fwd_btc_ret_{horizon}"
        if target_col not in panel:
            continue
        for name, mask in masks.items():
            subset = panel.loc[mask.fillna(False), [target_col]].dropna()
            if len(subset) < min_samples:
                continue
            returns = subset[target_col].astype(float)
            results.append(HypothesisResult(
                name=name,
                horizon=horizon,
                samples=len(returns),
                hit_rate=float((returns > 0).mean()),
                avg_forward_return=float(returns.mean()),
                notes="Research-only; do not trade without shadow/OOS validation.",
            ))
    return sorted(results, key=lambda result: result.avg_forward_return, reverse=True)


def _write_reports(
    *,
    report_dir: str,
    panel: pd.DataFrame,
    correlations: list[CorrelationResult],
    hypotheses: list[HypothesisResult],
    horizons: tuple[str, ...],
) -> tuple[str, str]:
    ts_ms = int(time.time() * 1000)
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"polymarket_correlation_{ts_ms}.json"
    md_path = out_dir / f"polymarket_correlation_{ts_ms}.md"
    payload = {
        "ts_ms": ts_ms,
        "rows": int(len(panel)),
        "horizons": list(horizons),
        "correlations": [result.to_dict() for result in correlations],
        "hypotheses": [result.to_dict() for result in hypotheses],
        "intelligence_hooks": {
            "macro_alignment_score": "candidate for future reporting after shadow validation",
            "social_confirmation_score": "candidate for future reporting after shadow validation",
            "news_risk_flag": "candidate for future reporting after shadow validation",
        },
        "caveats": [
            "Correlation is not causation.",
            "All features are point-in-time merged, but external feed delays still matter.",
            "No social/news feature should create trades alone.",
            "Use shadow/OOS evidence before promotion.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Polymarket Correlation Research",
        "",
        f"- Rows: `{len(panel)}`",
        f"- Horizons: `{', '.join(horizons)}`",
        "",
        "## Top Lead/Lag Correlations",
        "",
    ]
    if not correlations:
        lines.append("No correlations met the sample threshold.")
    for result in correlations[:20]:
        lines.append(
            f"- `{result.feature}` -> `{result.horizon}` "
            f"corr=`{result.correlation:.4f}` samples=`{result.samples}` "
            f"hit=`{result.directional_hit_rate:.2%}` "
            f"top_bucket_avg=`{result.top_bucket_avg_forward_return:.4%}`"
        )
    lines.extend(["", "## Strategy Hypotheses", ""])
    if not hypotheses:
        lines.append("No hypotheses met the sample threshold.")
    for result in hypotheses:
        lines.append(
            f"- `{result.name}` horizon=`{result.horizon}` samples=`{result.samples}` "
            f"hit=`{result.hit_rate:.2%}` avg_forward_return=`{result.avg_forward_return:.4%}`"
        )
    lines.extend([
        "",
        "## Future Intelligence Hooks",
        "",
        "- `macro_alignment_score`: use only after shadow/OOS validation.",
        "- `social_confirmation_score`: use only after shadow/OOS validation.",
        "- `news_risk_flag`: use only after shadow/OOS validation.",
        "",
        "## Caveats",
        "",
        "- Correlation is not causation.",
        "- External feed delays and publication timing can degrade apparent signal.",
        "- No social/news feature should create trades alone.",
        "- Keep all outputs research/shadow-only until promotion evidence exists.",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return str(md_path), str(json_path)


def run_correlation_research(
    *,
    db_path: str = "data/hogan.db",
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    limit: int = 5000,
    horizons: tuple[str, ...] = ("1h", "4h", "1d", "3d"),
    min_samples: int = 20,
    report_dir: str = "reports/polymarket/correlation",
) -> dict:
    conn = get_connection(db_path)
    try:
        panel = build_feature_panel(
            conn,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            horizons=horizons,
        )
    finally:
        conn.close()
    correlations = lead_lag_correlations(panel, horizons=horizons, min_samples=min_samples)
    hypotheses = evaluate_strategy_hypotheses(panel, horizons=horizons)
    md_path, json_path = _write_reports(
        report_dir=report_dir,
        panel=panel,
        correlations=correlations,
        hypotheses=hypotheses,
        horizons=horizons,
    )
    return {
        "rows": len(panel),
        "correlations": correlations,
        "hypotheses": hypotheses,
        "markdown_path": md_path,
        "json_path": json_path,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research Polymarket/macro/social lead-lag correlations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--horizons", nargs="+", default=["1h", "4h", "1d", "3d"])
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--report-dir", default="reports/polymarket/correlation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_correlation_research(
        db_path=args.db,
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        horizons=tuple(args.horizons),
        min_samples=args.min_samples,
        report_dir=args.report_dir,
    )
    print(f"Rows: {result['rows']}")
    print(f"Correlations: {len(result['correlations'])}")
    print(f"Hypotheses: {len(result['hypotheses'])}")
    print(f"Report: {result['markdown_path']}")
    print(f"JSON: {result['json_path']}")


if __name__ == "__main__":
    main()
