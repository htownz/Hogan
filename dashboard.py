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
from pathlib import Path

import pandas as pd

st.set_page_config(page_title="Hogan Dashboard", layout="wide")
st.title("Hogan Trading Bot — Dashboard")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
registry_path = st.sidebar.text_input("Registry file", value="models/registry.jsonl")
db_path = st.sidebar.text_input("SQLite DB", value="data/hogan.db")
model_path_input = st.sidebar.text_input("Model to inspect", value="models/hogan_logreg.pkl")

# ---------------------------------------------------------------------------
# Model registry leaderboard
# ---------------------------------------------------------------------------
st.header("Model Registry")

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
                    f"Best model by ROC-AUC → `{best['model_path']}`  "
                    f"({best['model_type']}, ROC-AUC = {best['metrics'].get('roc_auc', '?')})"
                )
        else:
            st.info("Registry is empty. Run `python -m hogan_bot.train` to populate it.")
    except Exception as exc:
        st.error(f"Could not load registry: {exc}")
else:
    st.info(f"Registry file not found at `{registry_path}`. Train a model first.")

# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------
st.header("Feature Importances")

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
                .tail(24)
            )
            try:
                import plotly.express as px

                fig = px.bar(
                    df_imp, x="importance", y="feature", orientation="h",
                    title="Feature Importance"
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

# ---------------------------------------------------------------------------
# SQLite candle store — available data
# ---------------------------------------------------------------------------
st.header("Local Candle Store")

if Path(db_path).exists():
    try:
        from hogan_bot.storage import available_symbols, get_connection

        conn = get_connection(db_path)
        series = available_symbols(conn)
        conn.close()
        if series:
            df_store = pd.DataFrame(series, columns=["symbol", "timeframe", "candles"])
            st.dataframe(df_store, use_container_width=True)
        else:
            st.info("No candles stored yet. Run the data ingestion step first.")
    except Exception as exc:
        st.error(f"Could not open database: {exc}")
else:
    st.info(f"Database not found at `{db_path}`.")

# ---------------------------------------------------------------------------
# Quick backtest + equity curve
# ---------------------------------------------------------------------------
st.header("Quick Backtest")

with st.expander("Run backtest from local DB", expanded=False):
    col1, col2, col3 = st.columns(3)
    bt_symbol = col1.text_input("Symbol", value="BTC/USD", key="bt_sym")
    bt_tf = col2.text_input("Timeframe", value="5m", key="bt_tf")
    bt_limit = col3.number_input("Max bars", value=500, min_value=100, step=100, key="bt_lim")

    if st.button("Run"):
        if not Path(db_path).exists():
            st.error("No local database found. Store candles first.")
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
                    st.json(result.summary_dict())

                    if result.equity_curve:
                        eq_df = pd.DataFrame(
                            {"bar": range(len(result.equity_curve)),
                             "equity_usd": result.equity_curve}
                        )
                        st.line_chart(eq_df.set_index("bar")["equity_usd"])

            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
