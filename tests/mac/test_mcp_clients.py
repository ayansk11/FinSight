"""Tests for MCP data clients (FRED, Finnhub, yfinance, aggregator).

All external API calls are mocked to ensure fast, deterministic tests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _mock_empty_client(*methods):
    """Create an AsyncMock with given methods returning {} and a close()."""
    mock = AsyncMock()
    for m in methods:
        setattr(mock, m, AsyncMock(return_value={}))
    mock.close = AsyncMock()
    return mock


_P = "finsight_mac.mcp.market_data"


def _all_new_patches():
    """Patch FMP, AlphaVantage, Econdb, StockData with empty mocks."""
    fmp = _mock_empty_client(
        "get_income_statement",
        "get_balance_sheet",
        "get_cash_flow",
        "get_key_metrics",
        "get_financial_ratios",
    )
    av = _mock_empty_client(
        "get_company_overview",
        "get_earnings",
        "get_technical_indicators",
    )
    econdb = _mock_empty_client("get_global_macro_snapshot")
    sd = _mock_empty_client("get_stock_news")
    return (
        patch(f"{_P}.FMPClient", return_value=fmp),
        patch(f"{_P}.AlphaVantageClient", return_value=av),
        patch(f"{_P}.EcondbClient", return_value=econdb),
        patch(f"{_P}.StockDataClient", return_value=sd),
    )


# ---------------------------------------------------------------------------
# FRED Client Tests
# ---------------------------------------------------------------------------


class TestFredClient:
    """Tests for the FRED macroeconomic data client."""

    def test_no_api_key_returns_empty(self):
        """When no FRED API key is configured, get_macro_snapshot returns {}."""
        with patch("finsight_mac.mcp.fred.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(fred_api_key=None)
            from finsight_mac.mcp.fred import FredClient

            client = FredClient()
            result = asyncio.get_event_loop().run_until_complete(client.get_macro_snapshot())
            assert result == {}

    def test_get_series_latest_no_key(self):
        """get_series_latest returns [] when no API key."""
        with patch("finsight_mac.mcp.fred.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(fred_api_key=None)
            from finsight_mac.mcp.fred import FredClient

            client = FredClient()
            result = asyncio.get_event_loop().run_until_complete(
                client.get_series_latest("FEDFUNDS")
            )
            assert result == []

    def test_format_macro_value(self):
        """Test the value formatting helper."""
        from finsight_mac.mcp.fred import _format_macro_value

        assert _format_macro_value("fed_funds_rate", 5.33) == "5.33%"
        assert _format_macro_value("gdp_growth_yoy", 2.5) == "2.5%"
        assert _format_macro_value("treasury_10y", 4.35) == "4.35%"
        assert _format_macro_value("some_index", 1234.5) == "1234.50"


# ---------------------------------------------------------------------------
# Finnhub Client Tests
# ---------------------------------------------------------------------------


class TestFinnhubClient:
    """Tests for the Finnhub market data client."""

    def test_no_api_key_returns_empty(self):
        """When no Finnhub API key is configured, all methods return {}."""
        with patch("finsight_mac.mcp.finnhub_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(finnhub_api_key=None)
            from finsight_mac.mcp.finnhub_client import FinnhubClient

            client = FinnhubClient()
            assert client._client is None

            result = asyncio.get_event_loop().run_until_complete(client.get_quote("AAPL"))
            assert result == {}

    def test_get_quote_parses_response(self):
        """get_quote correctly maps Finnhub's response fields."""
        with patch("finsight_mac.mcp.finnhub_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(finnhub_api_key="test_key")
            from finsight_mac.mcp.finnhub_client import FinnhubClient

            client = FinnhubClient()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "c": 189.45,
                "d": 1.23,
                "dp": 0.65,
                "h": 190.50,
                "l": 188.00,
                "o": 188.50,
                "pc": 188.22,
            }
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)
            client._last_request_time = 0.0

            result = asyncio.get_event_loop().run_until_complete(client.get_quote("AAPL"))
            assert result["current_price"] == 189.45
            assert result["change"] == 1.23
            assert result["change_pct"] == 0.65

    def test_get_recommendation_trends(self):
        """get_recommendation_trends computes consensus correctly."""
        with patch("finsight_mac.mcp.finnhub_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(finnhub_api_key="test_key")
            from finsight_mac.mcp.finnhub_client import FinnhubClient

            client = FinnhubClient()

            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    "buy": 20,
                    "strongBuy": 8,
                    "hold": 7,
                    "sell": 1,
                    "strongSell": 1,
                    "period": "2024-12-01",
                }
            ]
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)
            client._last_request_time = 0.0

            result = asyncio.get_event_loop().run_until_complete(
                client.get_recommendation_trends("AAPL")
            )
            assert result["analyst_consensus"] == "Buy"
            assert result["analyst_buy"] == 28  # 20 + 8
            assert result["analyst_sell"] == 2  # 1 + 1


# ---------------------------------------------------------------------------
# yfinance Client Tests
# ---------------------------------------------------------------------------


class TestYFinanceClient:
    """Tests for the yfinance stock data client."""

    def test_get_stock_info_handles_failure(self):
        """get_stock_info returns {} on failure."""
        with patch("finsight_mac.mcp.yfinance_client.get_settings"):
            from finsight_mac.mcp.yfinance_client import YFinanceClient

            client = YFinanceClient()

            mock_target = "finsight_mac.mcp.yfinance_client.asyncio.to_thread"
            with patch(mock_target, side_effect=Exception("network error")):
                result = asyncio.get_event_loop().run_until_complete(
                    client.get_stock_info("INVALID")
                )
                assert result == {}

    def test_get_price_history_handles_failure(self):
        """get_price_history returns {} on failure."""
        with patch("finsight_mac.mcp.yfinance_client.get_settings"):
            from finsight_mac.mcp.yfinance_client import YFinanceClient

            client = YFinanceClient()

            mock_target = "finsight_mac.mcp.yfinance_client.asyncio.to_thread"
            with patch(mock_target, side_effect=Exception("timeout")):
                result = asyncio.get_event_loop().run_until_complete(
                    client.get_price_history("INVALID")
                )
                assert result == {}


# ---------------------------------------------------------------------------
# Market Data Aggregator Tests
# ---------------------------------------------------------------------------


class TestMarketDataAggregator:
    """Tests for the market data aggregator."""

    def test_empty_ticker_returns_empty(self):
        """fetch_market_data('') returns {}."""
        from finsight_mac.mcp.market_data import fetch_market_data

        result = asyncio.get_event_loop().run_until_complete(fetch_market_data(""))
        assert result == {}

    def test_format_large_number(self):
        """_fmt_large_number formats billions/trillions correctly."""
        from finsight_mac.mcp.market_data import _fmt_large_number

        assert _fmt_large_number(2.95e12) == "$2.95T"
        assert _fmt_large_number(385.6e9) == "$385.6B"
        assert _fmt_large_number(42.5e6) == "$42.5M"
        assert _fmt_large_number(None) == "N/A"

    def test_format_pct(self):
        """_fmt_pct always treats values as ratios (× 100)."""
        from finsight_mac.mcp.market_data import _fmt_pct

        assert _fmt_pct(0.4625) == "46.25%"
        assert _fmt_pct(1.52) == "152.00%"  # ROE > 100%
        assert _fmt_pct(0.055, decimals=1) == "5.5%"
        assert _fmt_pct(None) == "N/A"

    def test_format_pct_raw(self):
        """_fmt_pct_raw treats values as already percentages."""
        from finsight_mac.mcp.market_data import _fmt_pct_raw

        assert _fmt_pct_raw(0.4) == "0.40%"
        assert _fmt_pct_raw(2.5) == "2.50%"
        assert _fmt_pct_raw(None) == "N/A"

    def test_aggregator_merges_sources(self):
        """fetch_market_data merges data from all three sources."""
        from finsight_mac.mcp.market_data import fetch_market_data

        mock_fred = AsyncMock()
        mock_fred.get_macro_snapshot = AsyncMock(return_value={"gdp_growth_yoy": "2.5%"})
        mock_fred.close = AsyncMock()

        mock_finnhub = AsyncMock()
        mock_finnhub.get_quote = AsyncMock(
            return_value={"current_price": 189.45, "change": 1.0, "change_pct": 0.5},
        )
        mock_finnhub.get_company_profile = AsyncMock(
            return_value={
                "company_name": "Apple",
                "sector": "Technology",
                "market_cap_millions": 2950000,
            },
        )
        mock_finnhub.get_basic_financials = AsyncMock(
            return_value={"pe_ratio_ttm": 28.5, "beta": 1.24},
        )
        mock_finnhub.get_recommendation_trends = AsyncMock(
            return_value={
                "analyst_consensus": "Buy",
                "analyst_buy": 28,
                "analyst_hold": 7,
                "analyst_sell": 2,
            },
        )
        mock_finnhub.close = AsyncMock()

        mock_yf = AsyncMock()
        mock_yf.get_stock_info = AsyncMock(
            return_value={"dividend_yield": 0.0055, "total_revenue": 385.6e9},
        )
        mock_yf.get_financials = AsyncMock(return_value={})
        mock_yf.get_price_history = AsyncMock(return_value={"period_return": 0.123})
        mock_yf.close = AsyncMock()

        p_fmp, p_av, p_econdb, p_sd = _all_new_patches()
        with (
            patch("finsight_mac.mcp.market_data.FredClient", return_value=mock_fred),
            patch("finsight_mac.mcp.market_data.FinnhubClient", return_value=mock_finnhub),
            patch("finsight_mac.mcp.market_data.YFinanceClient", return_value=mock_yf),
            p_fmp,
            p_av,
            p_econdb,
            p_sd,
        ):
            result = asyncio.get_event_loop().run_until_complete(fetch_market_data("AAPL"))

        assert result["current_price"] == "$189.45"
        assert result["company"] == "Apple"
        assert result["pe_ratio"] == "28.5"
        assert result["macro_gdp_growth_yoy"] == "2.5%"
        assert result["revenue_ttm"] == "$385.6B"
        assert result["1y_return"] == "12.30%"
        assert "Finnhub" in result["_data_sources"]
        assert "yfinance" in result["_data_sources"]
        assert "FRED" in result["_data_sources"]

    def test_partial_failure_still_returns_data(self):
        """If one source fails, others still contribute."""
        from finsight_mac.mcp.market_data import fetch_market_data

        mock_fred = AsyncMock()
        mock_fred.get_macro_snapshot = AsyncMock(side_effect=Exception("FRED down"))
        mock_fred.close = AsyncMock()

        mock_finnhub = AsyncMock()
        mock_finnhub.get_quote = AsyncMock(
            return_value={"current_price": 189.45, "change": 0, "change_pct": 0},
        )
        mock_finnhub.get_company_profile = AsyncMock(return_value={})
        mock_finnhub.get_basic_financials = AsyncMock(return_value={})
        mock_finnhub.get_recommendation_trends = AsyncMock(return_value={})
        mock_finnhub.close = AsyncMock()

        mock_yf = AsyncMock()
        mock_yf.get_stock_info = AsyncMock(return_value={})
        mock_yf.get_financials = AsyncMock(return_value={})
        mock_yf.get_price_history = AsyncMock(return_value={})
        mock_yf.close = AsyncMock()

        p_fmp, p_av, p_econdb, p_sd = _all_new_patches()
        with (
            patch("finsight_mac.mcp.market_data.FredClient", return_value=mock_fred),
            patch("finsight_mac.mcp.market_data.FinnhubClient", return_value=mock_finnhub),
            patch("finsight_mac.mcp.market_data.YFinanceClient", return_value=mock_yf),
            p_fmp,
            p_av,
            p_econdb,
            p_sd,
        ):
            result = asyncio.get_event_loop().run_until_complete(fetch_market_data("AAPL"))

        # Should still have Finnhub data despite FRED failure
        assert result["current_price"] == "$189.45"
        assert "macro_gdp_growth_yoy" not in result
