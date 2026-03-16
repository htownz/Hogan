"""One-time historical data backfill using Yahoo Finance (yfinance).

Yahoo Finance provides up to 60 days of free 5-minute OHLCV data for
BTC-USD — roughly 17,000 bars — which is more than enough to train the RL
agent meaningfully.  No API key required.

Usage
-----
    # Backfill 60 days of 5m BTC/USD (default)
    python -m hogan_bot.backfill

    # Backfill both BTC and ETH at 1-hour resolution
    python -m hogan_bot.backfill --symbol BTC/USD ETH/USD --timeframe 1h

    # Custom date range (1h / 1d only; yfinance 5m is limited to last 60 days)
    python -m hogan_bot.backfill --symbol BTC/USD --timeframe 1h --period 2y

    # Backfill SPY daily data for macro features (stored as SPY/USD 1d)
    python -m hogan_bot.backfill --macro

Supported periods : 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
Supported intervals: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo
  Note: intervals shorter than 1h are only available for the last 60 days.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# yfinance symbol conversion
# ---------------------------------------------------------------------------

def _to_yf_symbol(symbol: str) -> str:
    """Convert CCXT-style symbol (``BTC/USD``) to Yahoo Finance ticker (``BTC-USD``)."""
    return symbol.replace("/", "-")


def _to_ccxt_symbol(yf_symbol: str) -> str:
    return yf_symbol.replace("-", "/")


# ---------------------------------------------------------------------------
# Fetch + normalise
# ---------------------------------------------------------------------------

_TF_TO_YF: dict[str, str] = {
    "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "60m": "60m", "90m": "90m",
    "1d": "1d", "1w": "1wk", "1M": "1mo",
}

_TF_PERIOD_DEFAULT: dict[str, str] = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "1h": "2y", "60m": "2y", "90m": "60d",
    "1d": "10y", "1w": "10y", "1M": "10y",
}


def fetch_yfinance(
    symbol: str,
    timeframe: str = "1h",
    period: str | None = None,
) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance and return a normalised DataFrame.

    The returned DataFrame matches the schema expected by
    :func:`~hogan_bot.storage.upsert_candles`:
    ``timestamp`` (UTC datetime), ``open``, ``high``, ``low``, ``close``, ``volume``.
    """
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed.  Run: pip install yfinance")

    yf_interval = _TF_TO_YF.get(timeframe)
    if yf_interval is None:
        sys.exit(f"Unsupported timeframe '{timeframe}' for yfinance backfill.  "
                 f"Supported: {list(_TF_TO_YF)}")

    use_period = period or _TF_PERIOD_DEFAULT.get(timeframe, "60d")
    yf_sym = _to_yf_symbol(symbol)

    df = yf.download(
        yf_sym,
        period=use_period,
        interval=yf_interval,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {yf_sym} {yf_interval}")

    # Flatten MultiIndex columns that yfinance sometimes produces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df.index.name = "timestamp"
    df = df.reset_index()

    # Ensure UTC-aware timestamps
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def fetch_spy_daily(period: str = "10y") -> pd.DataFrame:
    """Download SPY daily OHLCV from Yahoo Finance.

    Returns a DataFrame with the standard ``timestamp, open, high, low,
    close, volume`` schema stored under symbol ``"SPY/USD"`` timeframe ``"1d"``.
    """
    return fetch_yfinance("SPY/USD", timeframe="1d", period=period)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill historical OHLCV data from Yahoo Finance into hogan.db",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", nargs="+", default=["BTC/USD"],
                   metavar="SYMBOL", help="Trading pairs, e.g. BTC/USD ETH/USD")
    p.add_argument("--timeframe", default="1h",
                   help="Bar interval: 1m 5m 15m 1h 1d etc.")
    p.add_argument("--period", default=None,
                   help="Lookback period: 7d 60d 1y 2y 5y max (default depends on timeframe)")
    p.add_argument("--db", default="data/hogan.db",
                   help="Path to the SQLite database file")
    p.add_argument(
        "--macro",
        action="store_true",
        help="Also backfill SPY daily data for macro features (stored as SPY/USD 1d)",
    )
    return p.parse_args()


def main() -> None:
    from hogan_bot.storage import candle_count, get_connection, upsert_candles

    args = parse_args()
    results = []

    for symbol in args.symbol:
        print(f"Fetching {symbol} {args.timeframe} from Yahoo Finance ...", flush=True)
        try:
            df = fetch_yfinance(symbol, timeframe=args.timeframe, period=args.period)
        except Exception as exc:
            msg = {"symbol": symbol, "error": str(exc)}
            print(json.dumps(msg))
            results.append(msg)
            continue

        conn = get_connection(args.db)
        before = candle_count(conn, symbol, args.timeframe)
        upsert_candles(conn, symbol, args.timeframe, df)
        after = candle_count(conn, symbol, args.timeframe)
        conn.close()

        summary = {
            "symbol": symbol,
            "timeframe": args.timeframe,
            "fetched": len(df),
            "new_rows": after - before,
            "total_stored": after,
            "oldest": str(df["timestamp"].iloc[0]),
            "newest": str(df["timestamp"].iloc[-1]),
        }
        print(json.dumps(summary))
        results.append(summary)

    if args.macro:
        print("Fetching SPY daily data from Yahoo Finance ...", flush=True)
        try:
            spy_period = args.period or "10y"
            df_spy = fetch_spy_daily(period=spy_period)
        except Exception as exc:
            msg = {"symbol": "SPY/USD", "error": str(exc)}
            print(json.dumps(msg))
            results.append(msg)
        else:
            conn = get_connection(args.db)
            before = candle_count(conn, "SPY/USD", "1d")
            upsert_candles(conn, "SPY/USD", "1d", df_spy)
            after = candle_count(conn, "SPY/USD", "1d")
            conn.close()
            summary = {
                "symbol": "SPY/USD",
                "timeframe": "1d",
                "fetched": len(df_spy),
                "new_rows": after - before,
                "total_stored": after,
                "oldest": str(df_spy["timestamp"].iloc[0]),
                "newest": str(df_spy["timestamp"].iloc[-1]),
            }
            print(json.dumps(summary))
            results.append(summary)


if __name__ == "__main__":
    main()
