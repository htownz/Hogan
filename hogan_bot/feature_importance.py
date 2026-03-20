"""Feature importance audit for Hogan's champion model.

Runs permutation importance on the 8 champion features against the
full candle dataset.  Identifies which features carry real signal versus
which add noise and degrade generalization.

Usage::

    python -m hogan_bot.feature_importance --db data/hogan.db

Output: diagnostics/feature_importance_report.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureScore:
    name: str
    importance_mean: float
    importance_std: float
    rank: int
    decision_relevance: str
    recommendation: str


def run_feature_importance(
    candles: pd.DataFrame,
    n_repeats: int = 10,
    test_ratio: float = 0.30,
    fee_rate: float = 0.0026,
    horizon_bars: int = 6,
) -> tuple[list[FeatureScore], dict]:
    """Train champion model, run permutation importance, return ranked scores."""
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from hogan_bot.feature_registry import (
        CHAMPION_FEATURE_DECISIONS,
    )
    from hogan_bot.ml import build_training_set

    logger.info("Building training set from %d candles...", len(candles))
    X, y, feature_cols, _ = build_training_set(
        candles,
        horizon_bars=horizon_bars,
        fee_rate=fee_rate,
        use_champion_features=True,
    )

    logger.info("Training set: %d samples, %d features, %.1f%% positive",
                len(X), X.shape[1], y.mean() * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000, C=0.1, class_weight="balanced", solver="lbfgs"
    )
    model.fit(X_train_s, y_train)

    y_prob_train = model.predict_proba(X_train_s)[:, 1]
    y_prob_test = model.predict_proba(X_test_s)[:, 1]

    train_auc = roc_auc_score(y_train, y_prob_train)
    test_auc = roc_auc_score(y_test, y_prob_test)
    logger.info("Model AUC: train=%.4f, test=%.4f", train_auc, test_auc)

    logger.info("Running permutation importance (%d repeats)...", n_repeats)
    perm_result = permutation_importance(
        model, X_test_s, y_test,
        n_repeats=n_repeats,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )

    scores: list[FeatureScore] = []
    for i, col in enumerate(feature_cols):
        mean_imp = float(perm_result.importances_mean[i])
        std_imp = float(perm_result.importances_std[i])
        decision = CHAMPION_FEATURE_DECISIONS.get(col, "unknown")

        if mean_imp > 0.005:
            rec = "KEEP — strong signal"
        elif mean_imp > 0.001:
            rec = "KEEP — moderate signal"
        elif mean_imp > -0.001:
            rec = "REVIEW — marginal"
        else:
            rec = "DROP — noise or harmful"

        scores.append(FeatureScore(
            name=col,
            importance_mean=mean_imp,
            importance_std=std_imp,
            rank=0,
            decision_relevance=decision,
            recommendation=rec,
        ))

    scores.sort(key=lambda s: s.importance_mean, reverse=True)
    for rank, s in enumerate(scores, 1):
        s.rank = rank

    return scores, {
        "train_auc": round(train_auc, 4),
        "test_auc": round(test_auc, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "positive_rate": round(float(y.mean()), 4),
    }


def print_report(scores: list[FeatureScore], model_info: dict) -> None:
    print(f"\n{'=' * 72}")
    print("CHAMPION FEATURE IMPORTANCE AUDIT")
    print(f"{'=' * 72}")
    print(f"Train AUC: {model_info['train_auc']:.4f}   "
          f"Test AUC: {model_info['test_auc']:.4f}   "
          f"Samples: {model_info['train_samples']}+{model_info['test_samples']}")
    print(f"{'-' * 72}")
    print(f"{'Rank':<5} {'Feature':<20} {'Importance':>12} {'+-Std':>8} {'Decision':>15} {'Recommendation'}")
    print(f"{'-' * 72}")
    for s in scores:
        print(f"{s.rank:<5} {s.name:<20} {s.importance_mean:>12.6f} {s.importance_std:>8.6f} "
              f"{s.decision_relevance:>15} {s.recommendation}")
    print(f"{'-' * 72}")

    keep = sum(1 for s in scores if s.recommendation.startswith("KEEP"))
    review = sum(1 for s in scores if s.recommendation.startswith("REVIEW"))
    drop = sum(1 for s in scores if s.recommendation.startswith("DROP"))
    print(f"Summary: {keep} KEEP, {review} REVIEW, {drop} DROP")

    print(f"\n{'-' * 72}")
    print("LOGISTIC REGRESSION COEFFICIENT MAGNITUDES")
    print(f"{'-' * 72}")
    for s in scores:
        bar_len = max(0, int(abs(s.importance_mean) * 2000))
        bar = "#" * min(bar_len, 40)
        sign = "+" if s.importance_mean >= 0 else "-"
        print(f"  {s.name:<20} {sign}{abs(s.importance_mean):.6f} {bar}")
    print(f"{'=' * 72}\n")


ICT_CANDIDATE_FEATURES = [
    "fvg_bull_active", "fvg_bear_active", "in_bull_fvg", "in_bear_fvg",
]


def run_ict_audit(
    candles: pd.DataFrame,
    n_repeats: int = 10,
    test_ratio: float = 0.30,
    fee_rate: float = 0.0026,
    horizon_bars: int = 6,
) -> dict:
    """Compare champion-only vs champion+ICT to measure ICT marginal value."""
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from hogan_bot.feature_registry import CHAMPION_FEATURE_COLUMNS
    from hogan_bot.ml import build_training_set

    X_full, y, feature_cols, _ = build_training_set(
        candles, horizon_bars=horizon_bars, fee_rate=fee_rate,
        use_champion_features=False,
    )
    if X_full is None or y is None or len(X_full) < 100:
        return {"error": "insufficient data"}

    champion_idx = [i for i, c in enumerate(feature_cols) if c in CHAMPION_FEATURE_COLUMNS]
    ict_idx = [i for i, c in enumerate(feature_cols) if c in ICT_CANDIDATE_FEATURES]
    available_ict = [feature_cols[i] for i in ict_idx]
    if not ict_idx:
        return {"error": "no ICT features available in training set", "available_features": list(feature_cols)}

    combined_idx = champion_idx + ict_idx

    results = {}
    for label, col_idx in [("champion_only", champion_idx), ("champion_plus_ict", combined_idx)]:
        X_sub = X_full.iloc[:, col_idx] if hasattr(X_full, "iloc") else X_full[:, col_idx]
        sub_cols = [feature_cols[i] for i in col_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y, test_size=test_ratio, shuffle=False,
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced", solver="lbfgs")
        model.fit(X_train_s, y_train)

        y_prob_train = model.predict_proba(X_train_s)[:, 1]
        y_prob_test = model.predict_proba(X_test_s)[:, 1]
        train_auc = roc_auc_score(y_train, y_prob_train)
        test_auc = roc_auc_score(y_test, y_prob_test)

        perm = permutation_importance(
            model, X_test_s, y_test, n_repeats=n_repeats,
            scoring="roc_auc", random_state=42, n_jobs=-1,
        )
        feature_scores = {
            sub_cols[i]: {
                "importance": round(float(perm.importances_mean[i]), 6),
                "std": round(float(perm.importances_std[i]), 6),
            }
            for i in range(len(sub_cols))
        }

        results[label] = {
            "features": sub_cols,
            "n_features": len(sub_cols),
            "train_auc": round(train_auc, 4),
            "test_auc": round(test_auc, 4),
            "feature_importances": feature_scores,
        }

    delta_auc = results["champion_plus_ict"]["test_auc"] - results["champion_only"]["test_auc"]
    results["ict_delta_auc"] = round(delta_auc, 4)
    results["ict_features_tested"] = available_ict
    if delta_auc > 0.005:
        results["verdict"] = "ICT features ADD value — consider promoting to champion"
    elif delta_auc > -0.005:
        results["verdict"] = "ICT features are NEUTRAL — not harmful, not helpful"
    else:
        results["verdict"] = "ICT features HURT — do not add to champion"

    return results


def print_ict_report(results: dict) -> None:
    print(f"\n{'=' * 72}")
    print("ICT FEATURE IMPORTANCE AUDIT")
    print(f"{'=' * 72}")
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    print(f"ICT features tested: {results['ict_features_tested']}")
    print(f"\n  {'Model':<25} {'Features':>8} {'Train AUC':>10} {'Test AUC':>10}")
    print(f"  {'-' * 55}")
    for label in ("champion_only", "champion_plus_ict"):
        r = results[label]
        print(f"  {label:<25} {r['n_features']:>8} {r['train_auc']:>10.4f} {r['test_auc']:>10.4f}")
    print(f"\n  Delta AUC: {results['ict_delta_auc']:+.4f}")
    print(f"  Verdict: {results['verdict']}")

    ict_r = results.get("champion_plus_ict", {})
    ict_imps = ict_r.get("feature_importances", {})
    ict_feats = {k: v for k, v in ict_imps.items() if k in results.get("ict_features_tested", [])}
    if ict_feats:
        print("\n  ICT Feature Importances (in combined model):")
        for name, v in sorted(ict_feats.items(), key=lambda x: x[1]["importance"], reverse=True):
            print(f"    {name:<25} {v['importance']:>+.6f} +/-{v['std']:.6f}")
    print(f"{'=' * 72}\n")


def main() -> None:
    import argparse
    import sqlite3

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(description="Hogan champion feature importance audit")
    p.add_argument("--db", default="data/hogan.db", help="SQLite DB path")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--output", default="diagnostics/feature_importance_report.json")
    p.add_argument("--ict-audit", action="store_true", help="Run ICT feature value audit")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    query = """
        SELECT ts_ms, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND timeframe = ?
        ORDER BY ts_ms
    """
    df = pd.read_sql_query(query, conn, params=(args.symbol, args.timeframe))
    conn.close()

    if df.empty:
        logger.error("No candles found for %s %s", args.symbol, args.timeframe)
        return

    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    logger.info("Loaded %d candles", len(df))

    if args.ict_audit:
        ict_results = run_ict_audit(df, n_repeats=args.n_repeats)
        print_ict_report(ict_results)
        out_path = Path(args.output).parent / "ict_audit_report.json"
        out_path.write_text(json.dumps(ict_results, indent=2), encoding="utf-8")
        logger.info("ICT audit saved to %s", out_path)
        return

    scores, model_info = run_feature_importance(
        df, n_repeats=args.n_repeats,
    )

    print_report(scores, model_info)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model_info": model_info,
        "features": [
            {
                "rank": s.rank,
                "name": s.name,
                "importance_mean": round(s.importance_mean, 6),
                "importance_std": round(s.importance_std, 6),
                "decision_relevance": s.decision_relevance,
                "recommendation": s.recommendation,
            }
            for s in scores
        ],
    }
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
