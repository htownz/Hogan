"""Tests for hogan_bot.exchange and hogan_bot.multi_exchange.

All CCXT network calls are mocked so the test suite runs offline and without
exchange credentials.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hogan_bot.exchange import ExchangeClient, KrakenClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_rows(n: int = 5, base_ts: int = 1_700_000_000_000) -> list[list]:
    """Return *n* synthetic OHLCV rows as CCXT would deliver them."""
    rows = []
    price = 30_000.0
    for i in range(n):
        ts = base_ts + i * 300_000  # 5-minute steps in ms
        rows.append([ts, price + i, price + i + 10, price + i - 5, price + i + 5, 1.0 + i * 0.1])
    return rows


def _mock_exchange(
    ohlcv_rows: list | None = None,
    ticker: dict | None = None,
    order_book: dict | None = None,
    funding_rate: dict | None = None,
    open_interest: dict | None = None,
    has_overrides: dict | None = None,
    markets: dict | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics a ccxt exchange object."""
    mock = MagicMock()
    mock.fetch_ohlcv.return_value = _make_ohlcv_rows() if ohlcv_rows is None else ohlcv_rows
    mock.fetch_ticker.return_value = ticker or {"last": 30_000.0, "bid": 29_990.0, "ask": 30_010.0}
    mock.fetch_order_book.return_value = order_book or {
        "bids": [[29_990.0, 1.0]],
        "asks": [[30_010.0, 1.0]],
        "timestamp": 1_700_000_000_000,
    }
    mock.fetch_funding_rate.return_value = funding_rate or {"fundingRate": 0.0001}
    mock.fetch_open_interest.return_value = open_interest or {"openInterest": 10_000.0}
    default_has = {
        "fetchFundingRate": True,
        "fetchOpenInterest": True,
        "fetchOrderBook": True,
    }
    if has_overrides:
        default_has.update(has_overrides)
    mock.has = default_has
    mock.load_markets.return_value = markets or {
        "BTC/USD": {"active": True, "base": "BTC", "quote": "USD"},
        "ETH/USD": {"active": True, "base": "ETH", "quote": "USD"},
        "BTC/USDT": {"active": True, "base": "BTC", "quote": "USDT"},
    }
    mock.timeframes = {"1m": "1m", "5m": "5m", "1h": "1h", "1d": "1d"}
    return mock


# ---------------------------------------------------------------------------
# ExchangeClient construction
# ---------------------------------------------------------------------------


class TestExchangeClientConstruction:
    def test_unknown_exchange_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            ExchangeClient("not_a_real_exchange_xyz")

    def test_exchange_id_stored(self):
        with patch("ccxt.kraken") as mock_cls:
            mock_cls.return_value = _mock_exchange()
            client = ExchangeClient("kraken")
        assert client.exchange_id == "kraken"

    def test_exchange_id_normalised_to_lower(self):
        with patch("ccxt.kraken") as mock_cls:
            mock_cls.return_value = _mock_exchange()
            client = ExchangeClient("KRAKEN")
        assert client.exchange_id == "kraken"


# ---------------------------------------------------------------------------
# fetch_ohlcv_df
# ---------------------------------------------------------------------------


class TestFetchOhlcvDf:
    def _client(self, rows=None):
        with patch("ccxt.kraken") as mock_cls:
            mock_cls.return_value = _mock_exchange(ohlcv_rows=rows)
            return ExchangeClient("kraken")

    def test_returns_dataframe(self):
        client = self._client()
        df = client.fetch_ohlcv_df("BTC/USD")
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        client = self._client()
        df = client.fetch_ohlcv_df("BTC/USD")
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_row_count(self):
        rows = _make_ohlcv_rows(n=10)
        client = self._client(rows)
        df = client.fetch_ohlcv_df("BTC/USD")
        assert len(df) == 10

    def test_timestamp_is_datetime(self):
        client = self._client()
        df = client.fetch_ohlcv_df("BTC/USD")
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_empty_response_returns_empty_df(self):
        client = self._client(rows=[])
        df = client.fetch_ohlcv_df("BTC/USD")
        assert df.empty
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# fetch_ticker
# ---------------------------------------------------------------------------


class TestFetchTicker:
    def test_returns_dict_with_last(self):
        with patch("ccxt.kraken") as mock_cls:
            mock_cls.return_value = _mock_exchange(ticker={"last": 40_000.0, "bid": 39_990.0, "ask": 40_010.0})
            client = ExchangeClient("kraken")
        t = client.fetch_ticker("BTC/USD")
        assert t["last"] == 40_000.0


# ---------------------------------------------------------------------------
# fetch_order_book
# ---------------------------------------------------------------------------


class TestFetchOrderBook:
    def test_has_bids_and_asks(self):
        with patch("ccxt.kraken") as mock_cls:
            mock_cls.return_value = _mock_exchange()
            client = ExchangeClient("kraken")
        ob = client.fetch_order_book("BTC/USD", depth=10)
        assert "bids" in ob and "asks" in ob


# ---------------------------------------------------------------------------
# Optional endpoints (funding rate, open interest)
# ---------------------------------------------------------------------------


class TestOptionalEndpoints:
    def test_funding_rate_returns_dict_when_supported(self):
        with patch("ccxt.binance") as mock_cls:
            mock_cls.return_value = _mock_exchange(has_overrides={"fetchFundingRate": True})
            client = ExchangeClient("binance")
        result = client.fetch_funding_rate("BTC/USDT:USDT")
        assert result is not None
        assert "fundingRate" in result

    def test_funding_rate_returns_none_when_unsupported(self):
        with patch("ccxt.kraken") as mock_cls:
            mock_cls.return_value = _mock_exchange(has_overrides={"fetchFundingRate": False})
            client = ExchangeClient("kraken")
        result = client.fetch_funding_rate("BTC/USD")
        assert result is None

    def test_open_interest_returns_none_when_unsupported(self):
        with patch("ccxt.kraken") as mock_cls:
            mock_cls.return_value = _mock_exchange(has_overrides={"fetchOpenInterest": False})
            client = ExchangeClient("kraken")
        result = client.fetch_open_interest("BTC/USD")
        assert result is None

    def test_open_interest_returns_dict_when_supported(self):
        with patch("ccxt.binance") as mock_cls:
            mock_cls.return_value = _mock_exchange(has_overrides={"fetchOpenInterest": True})
            client = ExchangeClient("binance")
        result = client.fetch_open_interest("BTC/USDT:USDT")
        assert result is not None


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


class TestDiscovery:
    def _client(self):
        with patch("ccxt.kraken") as mock_cls:
            mock_cls.return_value = _mock_exchange()
            return ExchangeClient("kraken")

    def test_list_symbols_returns_sorted_list(self):
        client = self._client()
        syms = client.list_symbols()
        assert isinstance(syms, list)
        assert syms == sorted(syms)

    def test_list_symbols_filtered_by_quote(self):
        client = self._client()
        syms = client.list_symbols(quote="USD")
        assert all(s.endswith("/USD") for s in syms)

    def test_list_timeframes_returns_list(self):
        client = self._client()
        tfs = client.list_timeframes()
        assert "5m" in tfs

    def test_supports_known_method(self):
        client = self._client()
        assert client.supports("fetchFundingRate") is True

    def test_supports_unknown_method(self):
        client = self._client()
        assert client.supports("watchOHLCV") is False


# ---------------------------------------------------------------------------
# KrakenClient backward compatibility
# ---------------------------------------------------------------------------


class TestKrakenClientAlias:
    def test_is_subclass_of_exchange_client(self):
        assert issubclass(KrakenClient, ExchangeClient)

    def test_exchange_id_is_kraken(self):
        with patch("ccxt.kraken") as mock_cls:
            mock_cls.return_value = _mock_exchange()
            client = KrakenClient()
        assert client.exchange_id == "kraken"

    def test_fetch_ohlcv_df_works(self):
        with patch("ccxt.kraken") as mock_cls:
            mock_cls.return_value = _mock_exchange()
            client = KrakenClient()
        df = client.fetch_ohlcv_df("BTC/USD")
        assert not df.empty


# ---------------------------------------------------------------------------
# multi_exchange module
# ---------------------------------------------------------------------------


class TestVwapComposite:
    def _make_df(self, close: float, volume: float, n: int = 5) -> pd.DataFrame:
        ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": close,
                "high": close + 10,
                "low": close - 5,
                "close": close,
                "volume": volume,
            }
        )

    def test_single_exchange_passthrough(self):
        from hogan_bot.multi_exchange import vwap_composite

        df = self._make_df(30_000, 1.0)
        result = vwap_composite({"kraken": df})
        assert len(result) == len(df)
        assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_two_exchange_vwap(self):
        from hogan_bot.multi_exchange import vwap_composite

        df_a = self._make_df(30_000, 2.0)
        df_b = self._make_df(30_100, 1.0)
        result = vwap_composite({"a": df_a, "b": df_b})
        # VWAP close = (30000*2 + 30100*1) / 3 ≈ 30033.33
        assert not result.empty
        expected_close = (30_000 * 2 + 30_100 * 1) / 3
        assert abs(result["close"].iloc[0] - expected_close) < 1.0

    def test_volume_is_summed(self):
        from hogan_bot.multi_exchange import vwap_composite

        df_a = self._make_df(30_000, 2.0)
        df_b = self._make_df(30_000, 3.0)
        result = vwap_composite({"a": df_a, "b": df_b})
        assert result["volume"].iloc[0] == pytest.approx(5.0)

    def test_empty_dict_returns_empty(self):
        from hogan_bot.multi_exchange import vwap_composite

        result = vwap_composite({})
        assert result.empty

    def test_high_is_max_low_is_min(self):
        from hogan_bot.multi_exchange import vwap_composite

        df_a = self._make_df(30_000, 1.0)
        df_b = self._make_df(31_000, 1.0)
        result = vwap_composite({"a": df_a, "b": df_b})
        assert result["high"].iloc[0] >= result["low"].iloc[0]


class TestPriceSpread:
    def _make_df(self, close: float, n: int = 5) -> pd.DataFrame:
        ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "close": close, "open": close, "high": close + 5, "low": close - 5, "volume": 1.0})

    def test_zero_spread_when_same_price(self):
        from hogan_bot.multi_exchange import price_spread

        df = self._make_df(30_000)
        result = price_spread({"a": df, "b": df.copy()})
        assert result["spread_pct"].abs().max() < 1e-9

    def test_spread_positive_when_prices_differ(self):
        from hogan_bot.multi_exchange import price_spread

        df_a = self._make_df(30_000)
        df_b = self._make_df(30_300)
        result = price_spread({"a": df_a, "b": df_b})
        assert (result["spread_pct"] > 0).all()


class TestCompositLastPrice:
    def test_simple_mean_without_volume(self):
        from hogan_bot.multi_exchange import composite_last_price

        tickers = {
            "a": {"last": 30_000.0, "quoteVolume": None},
            "b": {"last": 30_100.0, "quoteVolume": None},
        }
        result = composite_last_price(tickers)
        assert result == pytest.approx(30_050.0)

    def test_volume_weighted(self):
        from hogan_bot.multi_exchange import composite_last_price

        tickers = {
            "a": {"last": 30_000.0, "quoteVolume": 2.0},
            "b": {"last": 31_000.0, "quoteVolume": 1.0},
        }
        result = composite_last_price(tickers)
        expected = (30_000 * 2 + 31_000 * 1) / 3
        assert result == pytest.approx(expected)

    def test_empty_returns_none(self):
        from hogan_bot.multi_exchange import composite_last_price

        assert composite_last_price({}) is None
