"""Hogan monitoring dashboard — run with: streamlit run dashboard.py

Requires optional dependencies:
    pip install streamlit plotly
"""
from __future__ import annotations

try:
    import streamlit as st
except ImportError:
    raise SystemExit(
        "Streamlit is not installed.\n"
        "Install it with:  pip install streamlit plotly\n"
        "Then run:         streamlit run dashboard.py"
    )

import json
import sqlite3
from pathlib import Path

import pandas as pd

st.set_page_config(page_title="Hogan Dashboard", layout="wide", page_icon="H")
st.title("Hogan Trading Bot — Dashboard")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
registry_path = st.sidebar.text_input("Registry file", value="models/registry.jsonl")
db_path = st.sidebar.text_input("SQLite DB", value="data/hogan.db")
model_path_input = st.sidebar.text_input("Classical model", value="models/hogan_logreg.pkl")
rl_model_path = st.sidebar.text_input("RL policy (.zip)", value="models/hogan_rl_policy.zip")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_rl, tab_classical, tab_data, tab_backtest = st.tabs(
    ["RL Agent", "Classical Model", "Data Coverage", "Backtest"]
)

# ===========================================================================
# TAB 1 — RL Agent
# ===========================================================================
with tab_rl:
    st.header("RL Agent (PPO)")

    rl_path = Path(rl_model_path)
    best_path = Path(rl_model_path).parent / "best_model.zip"
    hparams_path = Path(rl_model_path).parent / "best_hparams.json"

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Policy file", rl_path.name if rl_path.exists() else "NOT FOUND")
    col_b.metric(
        "Best model (EvalCallback)",
        "best_model.zip" if best_path.exists() else "none yet",
    )
    col_c.metric(
        "Tuned hyperparams",
        "best_hparams.json" if hparams_path.exists() else "defaults",
    )

    if hparams_path.exists():
        with st.expander("Optuna best hyperparameters", expanded=False):
            try:
                hparams = json.loads(hparams_path.read_text())
                st.json(hparams)
            except Exception as exc:
                st.error(f"Could not read hyperparams: {exc}")

    # Checkpoint browser
    ckpt_dir = Path(rl_model_path).parent / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            with st.expander(f"Checkpoints ({len(ckpts)} saved)", expanded=False):
                df_ckpts = pd.DataFrame(
                    [
                        {
                            "file": p.name,
                            "size_mb": round(p.stat().st_size / 1e6, 2),
                            "modified": pd.Timestamp(p.stat().st_mtime, unit="s").isoformat()[:19],
                        }
                        for p in ckpts[:20]
                    ]
                )
                st.dataframe(df_ckpts, use_container_width=True)

    # Training command helper
    st.subheader("Training commands")
    obs_mode = st.radio(
        "Observation mode",
        ["Base (27-dim)", "Extended (73-dim)"],
        horizontal=True,
    )
    ext_flag = "--ext-features --load-1h --load-15m" if "Extended" in obs_mode else ""
    reward = st.selectbox("Reward type", ["risk_adjusted", "sharpe", "sortino", "delta_equity"])
    steps = st.select_slider("Timesteps", options=[100_000, 250_000, 500_000, 1_000_000, 2_000_000], value=500_000)
    st.code(
        f"python -m hogan_bot.rl_train --from-db --symbol BTC/USD "
        f"--reward {reward} --timesteps {steps:,} "
        f"--eval-freq 10000 --checkpoint-freq 25000 {ext_flag}".strip(),
        language="bash",
    )

    st.subheader("Hyperparameter tuning")
    st.code(
        "python -m hogan_bot.rl_tune --from-db --symbol BTC/USD --n-trials 50",
        language="bash",
    )

# ===========================================================================
# TAB 2 — Classical Model
# ===========================================================================
with tab_classical:
    st.header("Classical ML Model Registry")

    if Path(registry_path).exists():
        try:
            from hogan_bot.registry import ModelRegistry

            reg = ModelRegistry(registry_path=registry_path)
            rows = reg.summary()
            if rows:
                df_reg = pd.DataFrame(rows)
                for col in ("accuracy", "roc_auc", "f1"):
                    if col in df_reg.columns:
                        df_reg[col] = df_reg[col].map(lambda v: f"{v:.4f}" if v is not None else "—")
                st.dataframe(df_reg, use_container_width=True)
                best = reg.best(metric="roc_auc")
                if best:
                    st.success(
                        f"Best by ROC-AUC -> `{best['model_path']}`  "
                        f"({best['model_type']}, ROC-AUC = {best['metrics'].get('roc_auc', '?')})"
                    )
            else:
                st.info("Registry empty. Run `python -m hogan_bot.train` first.")
        except Exception as exc:
            st.error(f"Could not load registry: {exc}")
    else:
        st.info(f"Registry not found at `{registry_path}`.")

    st.subheader("Feature Importances")
    if Path(model_path_input).exists():
        try:
            import pickle

            with open(model_path_input, "rb") as f:
                artifact = pickle.load(f)

            importances: dict | None = None
            if hasattr(artifact.model, "feature_importances_"):
                importances = dict(
                    zip(artifact.feature_columns, artifact.model.feature_importances_)
                )
            elif hasattr(artifact.model, "coef_"):
                import numpy as np
                coef = artifact.model.coef_[0]
                importances = dict(zip(artifact.feature_columns, np.abs(coef)))

            if importances:
                df_imp = (
                    pd.DataFrame(
                        {"feature": list(importances.keys()), "importance": list(importances.values())}
                    )
                    .sort_values("importance", ascending=True)
                    .tail(36)
                )
                try:
                    import plotly.express as px
                    fig = px.bar(
                        df_imp, x="importance", y="feature", orientation="h",
                        title="Feature Importance (top 36)", height=700,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(df_imp.set_index("feature")["importance"])
            else:
                st.info("Model type does not expose feature importances.")
        except Exception as exc:
            st.error(f"Could not load model: {exc}")
    else:
        st.info(f"Model not found at `{model_path_input}`.")

# ===========================================================================
# TAB 3 — Data Coverage
# ===========================================================================
with tab_data:
    st.header("Data Coverage")

    if not Path(db_path).exists():
        st.warning(f"Database not found at `{db_path}`.")
    else:
        try:
            from hogan_bot.storage import available_symbols, get_connection

            conn = get_connection(db_path)

            # Candle store
            st.subheader("OHLCV Candles")
            series = available_symbols(conn)
            if series:
                df_store = pd.DataFrame(series, columns=["symbol", "timeframe", "candles"])
                st.dataframe(df_store, use_container_width=True)
            else:
                st.info("No candles stored yet.")

            # External / on-chain metrics coverage
            st.subheader("External & On-Chain Metrics")
            _EXT_METRICS = [
                # on-chain / derivatives
                "mvrv_ratio", "sopr", "active_addresses",
                "funding_rate", "open_interest",
                # coingecko
                "btc_dominance_pct", "stablecoin_dominance_pct",
                "global_mcap_change_24h", "defi_dominance_pct",
                "btc_ath_pct", "coingecko_sentiment",
                # alternative data
                "fear_greed_value", "news_sentiment_score",
                "news_volume_norm", "gpr_index",
                # glassnode
                "glassnode_exchange_netflow", "glassnode_realized_dist",
                "glassnode_puell_multiple",
                # santiment
                "santiment_social_vol_chg", "santiment_dev_activity_chg",
            ]
            raw_conn = sqlite3.connect(db_path)
            metric_rows = []
            for metric in _EXT_METRICS:
                try:
                    cur = raw_conn.execute(
                        "SELECT COUNT(*), MIN(date), MAX(date) FROM onchain_metrics WHERE metric=?",
                        (metric,),
                    )
                    count, min_d, max_d = cur.fetchone()
                    metric_rows.append({
                        "metric": metric,
                        "rows": count or 0,
                        "earliest": min_d or "—",
                        "latest": max_d or "—",
                        "status": "OK" if count else "EMPTY",
                    })
                except Exception:
                    metric_rows.append({
                        "metric": metric, "rows": 0,
                        "earliest": "—", "latest": "—", "status": "ERROR",
                    })

            # derivatives_metrics table
            try:
                cur = raw_conn.execute(
                    "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM derivatives_metrics"
                )
                count, min_t, max_t = cur.fetchone()
                metric_rows.append({
                    "metric": "derivatives_metrics (table)",
                    "rows": count or 0,
                    "earliest": str(min_t or "—")[:10],
                    "latest": str(max_t or "—")[:10],
                    "status": "OK" if count else "EMPTY",
                })
            except Exception:
                pass
            raw_conn.close()

            df_ext = pd.DataFrame(metric_rows)
            # colour the status column
            def _colour_status(val):
                if val == "OK":
                    return "color: green"
                elif val == "EMPTY":
                    return "color: orange"
                return "color: red"

            st.dataframe(
                df_ext.style.applymap(_colour_status, subset=["status"]),
                use_container_width=True,
            )

            # Data refresh helper
            st.subheader("Refresh commands")
            st.code(
                "# Refresh all daily data sources (run once per day)\n"
                "python refresh_daily.py\n\n"
                "# Or individually:\n"
                "python -m hogan_bot.fetch_feargreed --days 7\n"
                "python -m hogan_bot.fetch_coingecko\n"
                "python -m hogan_bot.fetch_gpr\n"
                "python -m hogan_bot.fetch_news_sentiment --days 7\n"
                "python -m hogan_bot.fetch_derivatives\n",
                language="bash",
            )

            conn.close()
        except Exception as exc:
            st.error(f"Could not open database: {exc}")

# ===========================================================================
# TAB 4 — Backtest
# ===========================================================================
with tab_backtest:
    st.header("Quick Backtest")

    col1, col2, col3 = st.columns(3)
    bt_symbol = col1.text_input("Symbol", value="BTC/USD", key="bt_sym")
    bt_tf = col2.text_input("Timeframe", value="5m", key="bt_tf")
    bt_limit = col3.number_input("Max bars", value=1000, min_value=100, step=100, key="bt_lim")

    if st.button("Run Backtest"):
        if not Path(db_path).exists():
            st.error("No local database found.")
        else:
            try:
                from hogan_bot.backtest import run_backtest_on_candles
                from hogan_bot.config import load_config
                from hogan_bot.storage import get_connection, load_candles

                cfg = load_config()
                conn = get_connection(db_path)
                candles = load_candles(conn, bt_symbol, bt_tf, limit=int(bt_limit))
                conn.close()

                if candles.empty:
                    st.warning(f"No candles found for {bt_symbol} / {bt_tf}.")
                else:
                    result = run_backtest_on_candles(
                        candles=candles,
                        symbol=bt_symbol,
                        starting_balance_usd=cfg.starting_balance_usd,
                        aggressive_allocation=cfg.aggressive_allocation,
                        max_risk_per_trade=cfg.max_risk_per_trade,
                        max_drawdown=cfg.max_drawdown,
                        short_ma_window=cfg.short_ma_window,
                        long_ma_window=cfg.long_ma_window,
                        volume_window=cfg.volume_window,
                        volume_threshold=cfg.volume_threshold,
                        fee_rate=cfg.fee_rate,
                        use_ema_clouds=cfg.use_ema_clouds,
                        signal_mode=cfg.signal_mode,
                        trailing_stop_pct=cfg.trailing_stop_pct,
                        take_profit_pct=cfg.take_profit_pct,
                    )

                    summary = result.summary_dict()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Return", f"{summary.get('total_return_pct', 0):.2f}%")
                    m2.metric("Sharpe", f"{summary.get('sharpe_ratio', 0):.3f}")
                    m3.metric("Max Drawdown", f"{summary.get('max_drawdown_pct', 0):.2f}%")
                    m4.metric("Trades", summary.get("total_trades", 0))

                    with st.expander("Full metrics", expanded=False):
                        st.json(summary)

                    if result.equity_curve:
                        eq_df = pd.DataFrame(
                            {"bar": range(len(result.equity_curve)),
                             "equity_usd": result.equity_curve}
                        )
                        try:
                            import plotly.express as px
                            fig = px.line(eq_df, x="bar", y="equity_usd", title="Equity Curve")
                            st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            st.line_chart(eq_df.set_index("bar")["equity_usd"])

            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
