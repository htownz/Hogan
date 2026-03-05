
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hogan_bot.config import load_config
from hogan_bot.ml import build_feature_frame
from hogan_bot.ml_advanced import train_advanced_ensemble, save_artifact, load_artifact, triple_barrier_labels
from hogan_bot.storage import get_connection, load_candles
from hogan_bot.notifier import make_notifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hogan.trainer")


def _eval_auc(artifact, candles: pd.DataFrame) -> float:
    feat = build_feature_frame(candles).dropna()
    if len(feat) < 1000:
        return 0.0
    y = triple_barrier_labels(candles.loc[feat.index, "close"], horizon=artifact.horizon, k=artifact.label_k)
    # predict in batch (per-bar) using regime-aware model selection
    from hogan_bot.ml_advanced import infer_regime
    regs = infer_regime(artifact.regime_model, candles.loc[feat.index])
    probs = []
    for i in range(len(feat)):
        r = int(regs.iloc[i])
        model = artifact.models_by_regime.get(r) or list(artifact.models_by_regime.values())[0]
        p = float(model.predict_proba(feat.values[i:i+1].astype(float))[:,1][0])
        probs.append(p)
    try:
        return float(roc_auc_score(y.values, np.array(probs)))
    except Exception:
        return 0.0


def train_once(from_db: bool = True) -> None:
    cfg = load_config()
    conn = get_connection(cfg.db_path)

    email_cfg = None
    if cfg.email_smtp_host and cfg.email_to and cfg.email_from:
        email_cfg = dict(
            smtp_host=cfg.email_smtp_host,
            smtp_port=cfg.email_smtp_port,
            username=cfg.email_username,
            password=cfg.email_password,
            from_addr=cfg.email_from,
            to_addr=cfg.email_to,
            use_tls=True,
        )
    notifier = make_notifier(
        webhook_url=cfg.webhook_url or None,
        telegram_token=cfg.telegram_token or None,
        telegram_chat_id=cfg.telegram_chat_id or None,
        email=email_cfg,
    )

    symbol = cfg.symbols[0]
    df = load_candles(conn, symbol, cfg.timeframe, limit=cfg.retrain_window_bars)
    if df.empty or len(df) < 2000:
        logger.warning("Not enough candles in DB to retrain (%s %s)", symbol, cfg.timeframe)
        return

    # Train candidate
    artifact, metrics = train_advanced_ensemble(df, horizon=48, label_k=2.0, n_regimes=3)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cand_path = f"models/candidate_advanced_{ts}.pkl"
    save_artifact(artifact, cand_path)

    # Evaluate candidate on tail holdout
    split = int(len(df) * 0.8)
    auc_cand = _eval_auc(artifact, df.iloc[split:].copy())

    # Evaluate current (if exists)
    cur_path = cfg.ml_model_path
    auc_cur = None
    promoted = False
    if cur_path and Path(cur_path).exists():
        try:
            cur = load_artifact(cur_path)
            auc_cur = _eval_auc(cur, df.iloc[split:].copy())
        except Exception:
            auc_cur = None

    improve = auc_cand - (auc_cur or 0.0)
    if auc_cur is None or improve >= cfg.retrain_min_improvement:
        Path(cur_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cand_path).replace(cur_path)
        promoted = True

    msg = {
        "symbol": symbol,
        "timeframe": cfg.timeframe,
        "candidate_auc": auc_cand,
        "current_auc": auc_cur,
        "improvement": improve,
        "promoted": promoted,
        "candidate_path": cand_path,
        "production_path": cur_path,
        "metrics": metrics,
    }
    logger.info("Retrain result: %s", msg)
    notifier.notify("retrain", msg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Scheduled retraining service (advanced ensemble).")
    ap.add_argument("--schedule-hours", type=float, default=None, help="If set, retrain every N hours in a loop.")
    args = ap.parse_args()

    if args.schedule_hours is None:
        train_once()
        return

    period = max(0.5, float(args.schedule_hours)) * 3600.0
    while True:
        try:
            train_once()
        except Exception as exc:
            logger.exception("Trainer exception: %s", exc)
        time.sleep(period)


if __name__ == "__main__":
    main()
