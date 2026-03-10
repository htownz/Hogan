"""Hogan Trading Bot — Live Dashboard

Launch:
    streamlit run dashboard.py

Auto-refreshes every 30 seconds. All data comes from the SQLite DB —
safe to run while the bot is running.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh  # type: ignore[import]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hogan Bot",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Auto-refresh (every 30 seconds)
# ---------------------------------------------------------------------------
try:
    st_autorefresh(interval=30_000, key="autorefresh")
except Exception:
    pass  # streamlit-autorefresh may not be installed

# ---------------------------------------------------------------------------
# Sidebar — config paths
# ---------------------------------------------------------------------------
st.sidebar.header("Config")
DB_PATH = st.sidebar.text_input("SQLite DB", value="data/hogan.db")
MODEL_PATH = st.sidebar.text_input("Model", value="models/hogan_logreg.pkl")
REGISTRY_PATH = st.sidebar.text_input("Registry", value="models/registry.jsonl")
AUTO_REFRESH = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
if st.sidebar.button("Refresh now"):
    st.rerun()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CACHE_TTL = 25  # seconds


@st.cache_data(ttl=_CACHE_TTL)
def _load_equity(db: str, limit: int = 5000) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(
            f"SELECT ts_ms, equity_usd, drawdown FROM equity_snapshots ORDER BY ts_ms DESC LIMIT {limit}",
            conn,
        )
        conn.close()
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        return df.sort_values("ts")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_CACHE_TTL)
def _load_paper_trades(db: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(
            """SELECT trade_id, symbol, side, entry_price, exit_price, qty,
                      realized_pnl, pnl_pct, entry_fee, exit_fee,
                      open_ts_ms, close_ts_ms, close_reason,
                      ml_up_prob, strategy_conf, vol_ratio
               FROM paper_trades ORDER BY open_ts_ms DESC""",
            conn,
        )
        # Join trade explanations (Mistral/Ollama) — fill_id = symbol_open_ts_ms
        try:
            expl = pd.read_sql_query(
                "SELECT fill_id, explanation, model_used FROM trade_explanations",
                conn,
            )
            if not expl.empty:
                df["fill_id"] = df["symbol"] + "_" + df["open_ts_ms"].astype(str)
                df = df.merge(expl[["fill_id", "explanation", "model_used"]], on="fill_id", how="left")
                df = df.drop(columns=["fill_id"], errors="ignore")
        except Exception:
            pass
        conn.close()
        df["opened"] = pd.to_datetime(df["open_ts_ms"], unit="ms", utc=True)
        df["closed"] = pd.to_datetime(df["close_ts_ms"], unit="ms", utc=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_CACHE_TTL)
def _load_candles(db: str, symbol: str, limit: int = 200) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(
            "SELECT ts_ms, open, high, low, close, volume FROM candles "
            "WHERE symbol=? ORDER BY ts_ms DESC LIMIT ?",
            conn, params=(symbol, limit),
        )
        conn.close()
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        return df.sort_values("ts")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_CACHE_TTL)
def _load_onchain_latest(db: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(
            """SELECT metric, value, date FROM onchain_metrics
               WHERE (metric, date) IN (
                   SELECT metric, MAX(date) FROM onchain_metrics GROUP BY metric
               ) ORDER BY metric""",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_model_info(model_path: str) -> dict:
    try:
        with open(model_path, "rb") as f:
            art = pickle.load(f)
        info = {"model_type": type(art.model).__name__, "features": len(art.feature_columns)}
        if hasattr(art.model, "feature_importances_"):
            import numpy as np
            info["importances"] = dict(zip(art.feature_columns, art.model.feature_importances_))
        elif hasattr(art.model, "coef_"):
            import numpy as np
            info["importances"] = dict(zip(art.feature_columns, np.abs(art.model.coef_[0])))
        return info
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def _load_registry(registry_path: str) -> list[dict]:
    try:
        rows = []
        with open(registry_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    except Exception:
        return []


def _pnl_color(val):
    if pd.isna(val):
        return ""
    return "color: #2ecc71" if val >= 0 else "color: #e74c3c"


# ---------------------------------------------------------------------------
# Main tab layout
# ---------------------------------------------------------------------------
tab_live, tab_signals, tab_model, tab_data, tab_backtest = st.tabs([
    "Live Trading", "Signals & Candles", "ML Models", "Data Coverage", "Backtest"
])

# ===========================================================================
# TAB 1 — LIVE TRADING
# ===========================================================================
with tab_live:
    st.header("Live Paper Trading")

    trades = _load_paper_trades(DB_PATH)
    equity = _load_equity(DB_PATH)

    # ── KPI row ──────────────────────────────────────────────────────────────
    closed = trades[trades["exit_price"].notna()] if not trades.empty else pd.DataFrame()
    open_trades = trades[trades["exit_price"].isna()] if not trades.empty else pd.DataFrame()

    cur_equity = equity["equity_usd"].iloc[-1] if not equity.empty else 10000.0
    start_equity = equity["equity_usd"].iloc[0] if not equity.empty else 10000.0
    total_pnl = cur_equity - start_equity
    total_pnl_pct = (total_pnl / start_equity * 100) if start_equity else 0.0

    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    if not closed.empty and "realized_pnl" in closed.columns:
        wins = closed[closed["realized_pnl"] > 0]
        losses = closed[closed["realized_pnl"] <= 0]
        win_rate = len(wins) / len(closed) * 100
        avg_win = wins["realized_pnl"].mean() if not wins.empty else 0.0
        avg_loss = losses["realized_pnl"].mean() if not losses.empty else 0.0

    max_dd = equity["drawdown"].max() * 100 if not equity.empty and "drawdown" in equity.columns else 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Equity", f"${cur_equity:,.2f}", delta=f"${total_pnl:+,.2f}")
    k2.metric("Total P&L", f"${total_pnl:+,.2f}", delta=f"{total_pnl_pct:+.2f}%")
    k3.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{len(closed)} closed trades")
    k4.metric("Avg Win / Loss", f"${avg_win:.2f} / ${avg_loss:.2f}")
    k5.metric("Max Drawdown", f"{max_dd:.2f}%")
    k6.metric("Open Positions", len(open_trades))

    st.divider()

    # ── Equity curve ─────────────────────────────────────────────────────────
    col_eq, col_dd = st.columns([3, 1])
    with col_eq:
        if not equity.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=equity["ts"], y=equity["equity_usd"],
                mode="lines", name="Equity",
                line=dict(color="#3498db", width=2),
                fill="tozeroy", fillcolor="rgba(52,152,219,0.1)",
            ))
            fig_eq.update_layout(
                title="Equity Curve (USD)", height=280,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title=None, yaxis_title="USD",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_eq, width='stretch')
        else:
            st.info("No equity snapshots yet — start the bot.")

    with col_dd:
        if not equity.empty and "drawdown" in equity.columns:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=equity["ts"], y=equity["drawdown"] * 100,
                mode="lines", name="Drawdown",
                line=dict(color="#e74c3c", width=1.5),
                fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
            ))
            fig_dd.update_layout(
                title="Drawdown %", height=280,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title=None, yaxis_title="%",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dd, width='stretch')

    st.divider()

    # ── Open positions ────────────────────────────────────────────────────────
    st.subheader(f"Open Positions ({len(open_trades)})")
    if not open_trades.empty:
        disp = open_trades[[
            "symbol", "side", "entry_price", "qty", "ml_up_prob", "strategy_conf", "opened"
        ]].copy()
        disp["notional_usd"] = (disp["qty"] * disp["entry_price"]).round(2)
        disp["opened"] = disp["opened"].dt.strftime("%m-%d %H:%M")
        st.dataframe(disp, width='stretch', hide_index=True)
    else:
        st.info("No open positions right now.")

    # ── Closed trades ─────────────────────────────────────────────────────────
    st.subheader(f"Trade Journal — {len(closed)} Closed Trades")
    if not closed.empty:
        # P&L over time area chart
        pnl_cum = closed.sort_values("close_ts_ms").copy()
        pnl_cum["cumulative_pnl"] = pnl_cum["realized_pnl"].cumsum()
        pnl_cum["closed_dt"] = pd.to_datetime(pnl_cum["close_ts_ms"], unit="ms", utc=True)

        if len(pnl_cum) >= 2:
            fig_pnl = go.Figure()
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in pnl_cum["cumulative_pnl"]]
            fig_pnl.add_trace(go.Scatter(
                x=pnl_cum["closed_dt"], y=pnl_cum["cumulative_pnl"],
                mode="lines+markers", name="Cumulative P&L",
                line=dict(color="#2ecc71", width=2),
                fill="tozeroy",
                fillcolor="rgba(46,204,113,0.12)",
            ))
            fig_pnl.update_layout(
                title="Cumulative Realized P&L", height=220,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pnl, width='stretch')

        # Per-symbol P&L bar
        sym_pnl = closed.groupby("symbol")["realized_pnl"].sum().reset_index()
        if len(sym_pnl) > 0:
            fig_sym = px.bar(
                sym_pnl, x="symbol", y="realized_pnl",
                color="realized_pnl", color_continuous_scale=["#e74c3c", "#2ecc71"],
                title="P&L by Symbol", height=180,
            )
            fig_sym.update_layout(
                margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sym, width='stretch')

        # Close reason breakdown
        reason_counts = closed.groupby("close_reason").size().reset_index(name="count")
        fig_reason = px.pie(
            reason_counts, names="close_reason", values="count",
            title="Close Reason Breakdown", height=200,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_reason.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_reason, width='stretch')

        # Trade table (include explanation if available)
        base_cols = [
            "symbol", "side", "entry_price", "exit_price", "qty",
            "realized_pnl", "pnl_pct", "close_reason",
            "ml_up_prob", "strategy_conf", "opened", "closed"
        ]
        if "explanation" in closed.columns:
            base_cols.insert(base_cols.index("close_reason") + 1, "explanation")
        disp2 = closed[[c for c in base_cols if c in closed.columns]].copy()
        disp2["opened"] = disp2["opened"].dt.strftime("%m-%d %H:%M")
        disp2["closed"] = disp2["closed"].dt.strftime("%m-%d %H:%M")
        disp2["pnl_pct"] = (disp2["pnl_pct"] * 100).round(3)
        disp2["realized_pnl"] = disp2["realized_pnl"].round(4)
        disp2 = disp2.sort_values("closed", ascending=False).head(100)
        st.dataframe(
            disp2.style.map(_pnl_color, subset=["realized_pnl"]),
            width='stretch', hide_index=True,
        )
    else:
        st.info("No closed trades yet. Positions will appear here once they exit.")

    # ── Win/Loss histogram ────────────────────────────────────────────────────
    if not closed.empty and len(closed) >= 5:
        st.subheader("P&L Distribution")
        fig_hist = px.histogram(
            closed, x="realized_pnl", nbins=30,
            color_discrete_sequence=["#3498db"],
            title="Trade P&L Distribution",
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_hist.update_layout(
            height=220, margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, width='stretch')


# ===========================================================================
# TAB 2 — SIGNALS & CANDLES
# ===========================================================================
with tab_signals:
    st.header("Signals & Candles")

    sig_sym = st.selectbox("Symbol", ["BTC/USD", "ETH/USD"], key="sig_sym")
    candles = _load_candles(DB_PATH, sig_sym, limit=288)  # last 24h at 5m

    if candles.empty:
        st.warning(f"No candles in DB for {sig_sym}")
    else:
        # Candlestick chart
        fig_candle = go.Figure(data=[go.Candlestick(
            x=candles["ts"],
            open=candles["open"], high=candles["high"],
            low=candles["low"], close=candles["close"],
            increasing_line_color="#2ecc71",
            decreasing_line_color="#e74c3c",
            name=sig_sym,
        )])

        # Overlay EMA cloud (fast 8/9, slow 34/50)
        import numpy as np
        for period, color, name in [(8, "#f39c12", "EMA8"), (9, "#e67e22", "EMA9"),
                                     (34, "#9b59b6", "EMA34"), (50, "#8e44ad", "EMA50")]:
            ema = candles["close"].ewm(span=period, adjust=False).mean()
            fig_candle.add_trace(go.Scatter(
                x=candles["ts"], y=ema, mode="lines",
                line=dict(width=1, color=color), name=name, opacity=0.6,
            ))

        fig_candle.update_layout(
            title=f"{sig_sym} — Last 24h (5m candles)",
            xaxis_rangeslider_visible=False,
            height=480,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_candle, width='stretch')

        # Volume bar chart
        fig_vol = go.Figure()
        vol_colors = ["#2ecc71" if c >= o else "#e74c3c"
                      for c, o in zip(candles["close"], candles["open"])]
        fig_vol.add_trace(go.Bar(x=candles["ts"], y=candles["volume"],
                                  marker_color=vol_colors, name="Volume"))
        avg_vol = candles["volume"].rolling(10).mean()
        fig_vol.add_trace(go.Scatter(x=candles["ts"], y=avg_vol,
                                      mode="lines", line=dict(color="#3498db", width=1.5),
                                      name="Avg(10)"))
        fig_vol.update_layout(
            title="Volume", height=180,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_vol, width='stretch')

        # Latest candle stats
        last = candles.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close", f"${last['close']:,.2f}")
        prev = candles.iloc[-2]["close"] if len(candles) > 1 else last["close"]
        c2.metric("Change", f"${last['close'] - prev:+.2f}", delta=f"{(last['close']-prev)/prev*100:+.3f}%")
        c3.metric("High / Low", f"${last['high']:,.2f} / ${last['low']:,.2f}")
        avg_v = candles["volume"].rolling(10).mean().iloc[-1]
        c4.metric("Vol Ratio", f"{last['volume'] / avg_v:.2f}x")


# ===========================================================================
# TAB 3 — ML MODELS
# ===========================================================================
with tab_model:
    st.header("ML Model & Registry")

    # Registry table
    st.subheader("Model Registry")
    reg_rows = _load_registry(REGISTRY_PATH)
    if reg_rows:
        reg_df = pd.json_normalize(reg_rows)
        for col in ("new_score", "current_score"):
            if col in reg_df.columns:
                reg_df[col] = reg_df[col].map(lambda v: f"{v:.4f}" if v is not None else "—")
        st.dataframe(reg_df, width='stretch', hide_index=True)
    else:
        st.info("No registry entries found.")

    # Feature importance
    st.subheader("Feature Importances")
    model_info = _load_model_info(MODEL_PATH)
    if "error" in model_info:
        st.error(model_info["error"])
    else:
        m1, m2 = st.columns(2)
        m1.metric("Model type", model_info.get("model_type", "unknown"))
        m2.metric("Feature count", model_info.get("features", "?"))

        if "importances" in model_info:
            imp_df = (
                pd.DataFrame({
                    "feature": list(model_info["importances"].keys()),
                    "importance": list(model_info["importances"].values()),
                })
                .sort_values("importance", ascending=True)
                .tail(43)
            )
            fig_imp = px.bar(
                imp_df, x="importance", y="feature", orientation="h",
                title="Feature Importance", height=700,
                color="importance", color_continuous_scale=["#3498db", "#e74c3c"],
            )
            fig_imp.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_imp, width='stretch')

    # Training commands
    st.subheader("Retrain commands")
    st.code(
        "# Standard retrain (last 25k bars, XGBoost)\n"
        "python -m hogan_bot.retrain --from-db --window-bars 25000 --model-type xgboost --horizon-bars 12\n\n"
        "# With paper trade feedback labels (after 50+ closed trades)\n"
        "python -m hogan_bot.retrain --from-db --window-bars 25000 --model-type xgboost --use-paper-labels\n\n"
        "# Force promote (override score check)\n"
        "python -m hogan_bot.retrain --from-db --window-bars 25000 --model-type xgboost --force-promote",
        language="bash",
    )


# ===========================================================================
# TAB 4 — DATA COVERAGE
# ===========================================================================
with tab_data:
    st.header("Data Coverage")

    onchain = _load_onchain_latest(DB_PATH)

    if onchain.empty:
        st.warning("No onchain_metrics data found.")
    else:
        # Group by metric category
        st.subheader(f"Latest External Metrics ({len(onchain)} metrics)")
        st.dataframe(onchain, width='stretch', hide_index=True)

    # Candle coverage
    st.subheader("OHLCV Candle Coverage")
    try:
        conn = sqlite3.connect(DB_PATH)
        candle_cov = pd.read_sql_query(
            """SELECT symbol, timeframe, COUNT(*) as candles,
                      date(MIN(ts_ms)/1000, 'unixepoch') as earliest,
                      date(MAX(ts_ms)/1000, 'unixepoch') as latest
               FROM candles GROUP BY symbol, timeframe ORDER BY symbol, timeframe""",
            conn,
        )
        conn.close()
        st.dataframe(candle_cov, width='stretch', hide_index=True)
    except Exception as e:
        st.error(f"Could not load candle coverage: {e}")

    # Refresh commands
    st.subheader("Refresh commands")
    st.code(
        "# Run daily data refresh\npython refresh_daily.py\n\n"
        "# Backfill 1yr of 5m candles via Alpaca\n"
        "python -m hogan_bot.fetch_alpaca --backfill-bars 100000 --symbol BTC/USD\n"
        "python -m hogan_bot.fetch_alpaca --backfill-bars 100000 --symbol ETH/USD",
        language="bash",
    )


# ===========================================================================
# TAB 5 — BACKTEST
# ===========================================================================
with tab_backtest:
    st.header("Quick Backtest")

    col1, col2, col3, col4 = st.columns(4)
    bt_symbol = col1.selectbox("Symbol", ["BTC/USD", "ETH/USD"], key="bt_sym")
    bt_tf = col2.selectbox("Timeframe", ["5m", "1h", "1d"], key="bt_tf")
    bt_limit = col3.number_input("Bars", value=2000, min_value=200, step=200, key="bt_lim")
    bt_use_ml = col4.checkbox("Use ML filter", value=True, key="bt_ml")

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                from hogan_bot.backtest import run_backtest_on_candles
                from hogan_bot.config import load_config
                from hogan_bot.storage import get_connection, load_candles

                cfg = load_config()
                conn_bt = get_connection(DB_PATH)
                bt_candles = load_candles(conn_bt, bt_symbol, bt_tf, limit=int(bt_limit))
                conn_bt.close()

                if bt_candles.empty:
                    st.warning(f"No candles found for {bt_symbol}/{bt_tf}.")
                else:
                    ml_model = None
                    if bt_use_ml and Path(MODEL_PATH).exists():
                        import pickle
                        with open(MODEL_PATH, "rb") as f:
                            ml_model = pickle.load(f)

                    result = run_backtest_on_candles(
                        candles=bt_candles,
                        symbol=bt_symbol,
                        timeframe=bt_tf,
                        starting_balance_usd=10_000.0,
                        aggressive_allocation=cfg.aggressive_allocation,
                        max_risk_per_trade=cfg.max_risk_per_trade,
                        max_drawdown=cfg.max_drawdown,
                        short_ma_window=cfg.short_ma_window,
                        long_ma_window=cfg.long_ma_window,
                        volume_window=cfg.volume_window,
                        volume_threshold=cfg.volume_threshold,
                        fee_rate=cfg.fee_rate,
                        ml_model=ml_model,
                        ml_buy_threshold=cfg.ml_buy_threshold,
                        ml_sell_threshold=cfg.ml_sell_threshold,
                        ml_confidence_sizing=cfg.ml_confidence_sizing,
                        use_ema_clouds=cfg.use_ema_clouds,
                        use_ict=cfg.use_ict,
                        signal_mode=cfg.signal_mode,
                        min_vote_margin=cfg.signal_min_vote_margin,
                        trailing_stop_pct=cfg.trailing_stop_pct,
                        take_profit_pct=cfg.take_profit_pct,
                    )

                    summary = result.summary_dict()
                    m1, m2, m3, m4, m5 = st.columns(5)
                    ret = summary.get("total_return_pct", 0)
                    m1.metric("Return", f"{ret:.2f}%", delta="profit" if ret > 0 else "loss")
                    m2.metric("Sharpe", f"{summary.get('sharpe_ratio', 0):.3f}")
                    m3.metric("Max DD", f"{summary.get('max_drawdown_pct', 0):.2f}%")
                    m4.metric("Trades", summary.get("total_trades", 0))
                    m5.metric("Win Rate", f"{summary.get('win_rate_pct', 0):.1f}%")

                    with st.expander("Full metrics"):
                        st.json(summary)

                    if result.equity_curve:
                        eq_df = pd.DataFrame({
                            "bar": range(len(result.equity_curve)),
                            "equity_usd": result.equity_curve,
                        })
                        fig_bt = px.line(eq_df, x="bar", y="equity_usd", title="Backtest Equity Curve")
                        fig_bt.update_layout(
                            height=300, margin=dict(l=0, r=0, t=30, b=0),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_bt, width='stretch')

            except Exception as exc:
                st.error(f"Backtest failed: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Hogan Trading Bot · Paper Mode · Last refresh: {pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M:%S UTC')} "
    f"· DB: {DB_PATH}"
)
