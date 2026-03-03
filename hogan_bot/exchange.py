from __future__ import annotations

import ccxt
import pandas as pd


class KrakenClient:
    def __init__(self, api_key: str | None, api_secret: str | None) -> None:
        self.exchange = ccxt.kraken(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )

    def fetch_ohlcv_df(self, symbol: str, timeframe: str = "5m", limit: int = 500) -> pd.DataFrame:
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df
