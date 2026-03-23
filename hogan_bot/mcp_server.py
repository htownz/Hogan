"""MCP Server for Hogan — Phase 5.

Exposes Hogan's internals as MCP (Model Context Protocol) tools so AI agents
(Cursor, Claude, AnythingLLM, etc.) can query and control the bot.

Tools exposed:
    hogan_get_portfolio        — current paper positions, cash, equity, drawdown
    hogan_get_signal           — generate signal for symbol/timeframe on demand
    hogan_run_backtest         — trigger backtest, return metrics JSON
    hogan_get_model_registry   — list all trained models with metrics
    hogan_get_data_coverage    — candle counts + external metric freshness
    hogan_trigger_retrain      — kick off walk-forward retrain cycle
    hogan_get_recent_trades    — last N fills from the fills table

Run::

    python -m hogan_bot.mcp_server           # port 8080
    python -m hogan_bot.mcp_server --port 9000

Register in .cursor/mcp.json::

    {
      "mcpServers": {
        "hogan": {
          "url": "http://localhost:8080/mcp",
          "type": "http"
        }
      }
    }

AnythingLLM compatibility: the /tools endpoint returns MCP-compatible JSON
schema so AnythingLLM can add Hogan as a custom MCP server.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)


def _get_db_path() -> str:
    import os
    return os.getenv("HOGAN_DB", "data/hogan.db")


def _get_registry_path() -> str:
    import os
    return os.getenv("HOGAN_REGISTRY", "models/registry.jsonl")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_get_portfolio(db_path: str | None = None) -> dict:
    """Return current paper portfolio state."""
    db = db_path or _get_db_path()
    try:
        from hogan_bot.storage import get_connection, load_equity, load_positions
        conn = get_connection(db)
        positions_df = load_positions(conn)
        equity_df = load_equity(conn, limit=1)
        conn.close()

        positions = []
        for _, row in positions_df.iterrows():
            positions.append({
                "symbol": row.get("symbol", ""),
                "qty": float(row.get("qty", 0)),
                "avg_entry": float(row.get("avg_entry", 0)),
            })

        equity_usd = 0.0
        cash_usd = 0.0
        drawdown = 0.0
        if not equity_df.empty:
            last = equity_df.iloc[0]
            equity_usd = float(last.get("equity_usd", 0))
            cash_usd = float(last.get("cash_usd", 0))
            drawdown = float(last.get("drawdown", 0))

        return {
            "ok": True,
            "positions": positions,
            "equity_usd": equity_usd,
            "cash_usd": cash_usd,
            "drawdown": drawdown,
            "position_count": len([p for p in positions if p["qty"] > 0]),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def tool_get_signal(
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    db_path: str | None = None,
) -> dict:
    """Generate a trading signal for the given symbol/timeframe."""
    db = db_path or _get_db_path()
    try:
        from hogan_bot.config import load_config
        from hogan_bot.storage import get_connection, load_candles
        from hogan_bot.strategy import generate_signal

        conn = get_connection(db)
        candles = load_candles(conn, symbol, timeframe, limit=500)
        conn.close()

        if candles.empty:
            return {"ok": False, "error": f"No candles for {symbol}/{timeframe}"}

        cfg = load_config()
        signal = generate_signal(
            candles,
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
            min_vote_margin=cfg.signal_min_vote_margin,
        )
        return {
            "ok": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "action": signal.action,
            "confidence": round(signal.confidence, 4),
            "volume_ratio": round(signal.volume_ratio, 4),
            "stop_distance_pct": round(signal.stop_distance_pct, 4),
            "current_price": float(candles["close"].iloc[-1]),
            "bars": len(candles),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def tool_run_backtest(
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    limit: int = 1000,
    db_path: str | None = None,
) -> dict:
    """Run a backtest and return metrics."""
    db = db_path or _get_db_path()
    try:
        from hogan_bot.backtest import run_backtest_on_candles
        from hogan_bot.config import load_config
        from hogan_bot.storage import get_connection, load_candles

        cfg = load_config()
        conn = get_connection(db)
        candles = load_candles(conn, symbol, timeframe, limit=limit)
        conn.close()

        if candles.empty:
            return {"ok": False, "error": f"No candles for {symbol}/{timeframe}"}

        result = run_backtest_on_candles(
            candles=candles,
            symbol=symbol,
            timeframe=timeframe,
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
            min_vote_margin=cfg.signal_min_vote_margin,
            trailing_stop_pct=cfg.trailing_stop_pct,
            take_profit_pct=cfg.take_profit_pct,
        )
        summary = result.summary_dict()
        summary["ok"] = True
        summary["symbol"] = symbol
        summary["timeframe"] = timeframe
        summary["bars"] = len(candles)
        return summary
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def tool_get_model_registry(registry_path: str | None = None) -> dict:
    """Return all models in the registry with their metrics."""
    reg_path = registry_path or _get_registry_path()
    try:
        from hogan_bot.registry import ModelRegistry
        reg = ModelRegistry(registry_path=reg_path)
        rows = reg.summary()
        best = reg.best(metric="roc_auc")
        return {
            "ok": True,
            "models": rows,
            "best_model": best,
            "total": len(rows),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def tool_get_data_coverage(db_path: str | None = None) -> dict:
    """Return candle counts and external metric freshness."""
    db = db_path or _get_db_path()
    try:
        import sqlite3

        from hogan_bot.storage import available_symbols, get_connection

        conn = get_connection(db)
        series = available_symbols(conn)
        conn.close()

        raw = sqlite3.connect(db)
        # Sample of external metrics
        ext_check = {}
        for metric in ["mvrv_zscore", "sopr", "fear_greed_value", "gpr_index",
                        "funding_rate", "btc_dominance_pct"]:
            try:
                cur = raw.execute(
                    "SELECT COUNT(*), MAX(date) FROM onchain_metrics WHERE metric=?",
                    (metric,),
                )
                cnt, latest = cur.fetchone()
                ext_check[metric] = {"rows": cnt or 0, "latest": latest or "—"}
            except Exception as exc:
                logger.debug("data_coverage check failed for %s: %s", metric, exc)
                ext_check[metric] = {"rows": 0, "latest": "—"}
        raw.close()

        return {
            "ok": True,
            "candle_series": [
                {"symbol": s, "timeframe": tf, "candles": n}
                for s, tf, n in series
            ],
            "external_metrics": ext_check,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


_ALLOWED_MODEL_TYPES = frozenset({"logreg", "random_forest", "xgboost", "lightgbm", "hist_gb"})


def tool_trigger_retrain(
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    model_type: str = "logreg",
    db_path: str | None = None,
) -> dict:
    """Kick off a walk-forward retrain cycle (runs in subprocess)."""
    if model_type not in _ALLOWED_MODEL_TYPES:
        return {"ok": False, "error": f"Invalid model_type '{model_type}'. Allowed: {sorted(_ALLOWED_MODEL_TYPES)}"}
    db = db_path or _get_db_path()
    try:
        cmd = [
            sys.executable, "-m", "hogan_bot.retrain",
            "--from-db", "--db", db,
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--model-type", model_type,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr[:500]}
        try:
            return {"ok": True, **json.loads(result.stdout)}
        except json.JSONDecodeError:
            return {"ok": True, "output": result.stdout[:500]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Retrain timed out after 300s"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def tool_get_recent_trades(
    limit: int = 50,
    symbol: str | None = None,
    db_path: str | None = None,
) -> dict:
    """Return recent fills with optional LLM explanations."""
    db = db_path or _get_db_path()
    try:
        import sqlite3

        from hogan_bot.storage import get_connection, load_fills

        conn = get_connection(db)
        df = load_fills(conn, limit=limit)
        conn.close()

        if df is None or df.empty:
            return {"ok": True, "trades": [], "total": 0}

        if symbol:
            df = df[df["symbol"] == symbol]

        # Try to join LLM explanations
        raw = sqlite3.connect(db)
        try:
            import pandas as pd
            expl_df = pd.read_sql_query(
                "SELECT fill_id, explanation FROM trade_explanations ORDER BY ts_ms DESC LIMIT 500",
                raw,
            )
            df = df.merge(expl_df, on="fill_id", how="left")
        except Exception as exc:
            logger.debug("trade_explanations join failed: %s", exc)
        finally:
            raw.close()

        trades = df.to_dict(orient="records")
        return {"ok": True, "trades": trades, "total": len(trades)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# MCP tool schema builder
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "hogan_get_portfolio",
        "description": "Get current paper trading portfolio: positions, equity, cash, drawdown.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "db_path": {"type": "string", "description": "Optional custom DB path"}
            }
        },
    },
    {
        "name": "hogan_get_signal",
        "description": "Generate a real-time trading signal for a given symbol and timeframe.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "default": "BTC/USD"},
                "timeframe": {"type": "string", "default": "1h"},
            }
        },
    },
    {
        "name": "hogan_run_backtest",
        "description": "Run a backtest on stored candles and return performance metrics (Sharpe, drawdown, return %).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "default": "BTC/USD"},
                "timeframe": {"type": "string", "default": "1h"},
                "limit": {"type": "integer", "default": 1000},
            }
        },
    },
    {
        "name": "hogan_get_model_registry",
        "description": "List all trained ML models with accuracy, ROC-AUC, F1 metrics.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hogan_get_data_coverage",
        "description": "Show candle counts per symbol/timeframe and external data freshness.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hogan_trigger_retrain",
        "description": "Trigger a walk-forward ML retrain cycle and return the result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "default": "BTC/USD"},
                "timeframe": {"type": "string", "default": "1h"},
                "model_type": {
                    "type": "string",
                    "enum": ["logreg", "random_forest", "xgboost", "lightgbm"],
                    "default": "logreg",
                },
            }
        },
    },
    {
        "name": "hogan_get_recent_trades",
        "description": "Return the most recent N fills with LLM explanations if available.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50},
                "symbol": {"type": "string"},
            }
        },
    },
]

_TOOL_DISPATCH: dict[str, Any] = {
    "hogan_get_portfolio": tool_get_portfolio,
    "hogan_get_signal": tool_get_signal,
    "hogan_run_backtest": tool_run_backtest,
    "hogan_get_model_registry": tool_get_model_registry,
    "hogan_get_data_coverage": tool_get_data_coverage,
    "hogan_trigger_retrain": tool_trigger_retrain,
    "hogan_get_recent_trades": tool_get_recent_trades,
}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def _filter_params(fn, body: dict) -> dict:
    """Filter request body to only include valid function parameters."""
    valid_params = set(inspect.signature(fn).parameters.keys())
    return {k: v for k, v in body.items() if k in valid_params}


_MCP_API_KEY = os.getenv("HOGAN_MCP_API_KEY", "")


def create_app(db_path: str | None = None):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError:
        raise SystemExit("fastapi not installed. Run: pip install fastapi uvicorn") from None

    app = FastAPI(
        title="Hogan MCP Server",
        description="MCP-compatible API for the Hogan crypto trading bot",
        version="1.0.0",
    )

    def _check_auth(request: Request) -> JSONResponse | None:
        """Return an error response if API key auth is configured but missing/wrong."""
        if not _MCP_API_KEY:
            return None
        token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if token != _MCP_API_KEY:
            return JSONResponse(status_code=401, content={"error": "Invalid or missing API key"})
        return None

    @app.get("/health")
    async def health():
        return {"status": "ok", "server": "hogan-mcp", "timestamp": time.time()}

    @app.get("/tools")
    async def list_tools(request: Request):
        """MCP tool discovery endpoint."""
        auth_err = _check_auth(request)
        if auth_err:
            return auth_err
        return {"tools": _TOOLS}

    @app.post("/tools/{tool_name}")
    async def call_tool(tool_name: str, request: Request):
        """MCP tool call endpoint."""
        auth_err = _check_auth(request)
        if auth_err:
            return auth_err
        fn = _TOOL_DISPATCH.get(tool_name)
        if fn is None:
            return JSONResponse(status_code=404, content={"error": f"Unknown tool: {tool_name}"})
        try:
            body = await request.json()
        except Exception:
            body = {}

        if db_path and "db_path" not in body:
            body["db_path"] = db_path

        try:
            result = fn(**_filter_params(fn, body))
            return result
        except Exception as exc:
            logger.exception("Tool %s failed", tool_name)
            return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        """Unified MCP JSON-RPC endpoint."""
        auth_err = _check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

        method = body.get("method", "")
        params = body.get("params", {})
        req_id = body.get("id", 1)

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _TOOLS}}

        if method == "tools/call":
            tool_name = params.get("name") or params.get("tool")
            arguments = params.get("arguments", params.get("input", {}))
            fn = _TOOL_DISPATCH.get(tool_name)
            if fn is None:
                return JSONResponse(status_code=404, content={
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                })
            try:
                result = fn(**_filter_params(fn, arguments))
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result)}]},
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(exc)},
                }

        return JSONResponse(status_code=400, content={
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32600, "message": f"Unknown method: {method}"}
        })

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hogan MCP Server")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--db", default=None, help="Override DB path")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    try:
        import uvicorn
    except ImportError:
        sys.exit("uvicorn not installed. Run: pip install fastapi uvicorn")

    app = create_app(db_path=args.db)
    logger.info("Starting Hogan MCP server at http://%s:%d", args.host, args.port)
    logger.info("Register in .cursor/mcp.json: {\"hogan\": {\"url\": \"http://localhost:%d/mcp\"}}", args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
