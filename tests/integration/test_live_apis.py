"""Live API integration tests — only run when API keys are available.

These tests make real HTTP requests to external APIs.
They are skipped in CI unless secrets are configured.
Run locally with: uv run pytest tests/integration/ -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

# Skip entire module if no API keys are set
pytestmark = pytest.mark.integration


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Finnhub
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("FINNHUB_API_KEY"),
    reason="FINNHUB_API_KEY not set",
)
class TestFinnhubLive:
    """Live tests for Finnhub API."""

    def test_get_quote(self):
        from finsight_mac.mcp.finnhub_client import FinnhubClient

        client = FinnhubClient()
        result = _run(client.get_quote("AAPL"))
        _run(client.close())

        assert "current_price" in result
        assert isinstance(result["current_price"], (int, float))
        assert result["current_price"] > 0

    def test_get_company_profile(self):
        from finsight_mac.mcp.finnhub_client import FinnhubClient

        client = FinnhubClient()
        result = _run(client.get_company_profile("AAPL"))
        _run(client.close())

        assert result.get("company_name") == "Apple Inc"

    def test_get_recommendation_trends(self):
        from finsight_mac.mcp.finnhub_client import FinnhubClient

        client = FinnhubClient()
        result = _run(client.get_recommendation_trends("AAPL"))
        _run(client.close())

        assert "analyst_consensus" in result


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("FRED_API_KEY"),
    reason="FRED_API_KEY not set",
)
class TestFredLive:
    """Live tests for FRED API."""

    def test_get_macro_snapshot(self):
        from finsight_mac.mcp.fred import FredClient

        client = FredClient()
        result = _run(client.get_macro_snapshot())
        _run(client.close())

        assert isinstance(result, dict)
        assert len(result) > 0
        # Should have at least fed funds rate
        assert "fed_funds_rate" in result

    def test_get_series_latest(self):
        from finsight_mac.mcp.fred import FredClient

        client = FredClient()
        result = _run(client.get_series_latest("FEDFUNDS"))
        _run(client.close())

        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Financial Modeling Prep
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("FMP_API_KEY"),
    reason="FMP_API_KEY not set",
)
class TestFMPLive:
    """Live tests for Financial Modeling Prep stable API."""

    def test_get_income_statement(self):
        from finsight_mac.mcp.fmp_client import FMPClient

        client = FMPClient()
        result = _run(client.get_income_statement("AAPL"))
        _run(client.close())

        assert isinstance(result, dict)
        assert "revenue" in result
        assert "eps_diluted" in result

    def test_get_balance_sheet(self):
        from finsight_mac.mcp.fmp_client import FMPClient

        client = FMPClient()
        result = _run(client.get_balance_sheet("AAPL"))
        _run(client.close())

        assert isinstance(result, dict)
        assert "total_assets" in result

    def test_get_key_metrics(self):
        from finsight_mac.mcp.fmp_client import FMPClient

        client = FMPClient()
        result = _run(client.get_key_metrics("AAPL"))
        _run(client.close())

        assert isinstance(result, dict)
        assert "ev_to_ebitda" in result

    def test_get_financial_ratios(self):
        from finsight_mac.mcp.fmp_client import FMPClient

        client = FMPClient()
        result = _run(client.get_financial_ratios("AAPL"))
        _run(client.close())

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ALPHA_VANTAGE_API_KEY"),
    reason="ALPHA_VANTAGE_API_KEY not set",
)
class TestAlphaVantageLive:
    """Live tests for Alpha Vantage API."""

    def test_get_company_overview(self):
        from finsight_mac.mcp.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()
        result = _run(client.get_company_overview("AAPL"))
        _run(client.close())

        assert isinstance(result, dict)
        assert "name" in result

    def test_get_earnings(self):
        from finsight_mac.mcp.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()
        result = _run(client.get_earnings("AAPL"))
        _run(client.close())

        assert isinstance(result, dict)

    def test_get_technical_indicators(self):
        from finsight_mac.mcp.alpha_vantage_client import AlphaVantageClient

        client = AlphaVantageClient()
        result = _run(client.get_technical_indicators("AAPL"))
        _run(client.close())

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Econdb (no API key required)
# ---------------------------------------------------------------------------


class TestEcondbLive:
    """Live tests for Econdb API.

    Note: Econdb may require authentication. This test gracefully
    handles empty results if the API returns 401.
    """

    def test_get_global_macro_snapshot(self):
        from finsight_mac.mcp.econdb_client import EcondbClient

        client = EcondbClient()
        result = _run(client.get_global_macro_snapshot())
        _run(client.close())

        # Econdb may return empty if auth is required
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# StockData
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("STOCKDATA_API_KEY"),
    reason="STOCKDATA_API_KEY not set",
)
class TestStockDataLive:
    """Live tests for StockData.org API."""

    def test_get_stock_news(self):
        from finsight_mac.mcp.stockdata_client import StockDataClient

        client = StockDataClient()
        result = _run(client.get_stock_news("AAPL"))
        _run(client.close())

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# yfinance (no API key required)
# ---------------------------------------------------------------------------


class TestYFinanceLive:
    """Live tests for yfinance client (no key needed)."""

    def test_get_stock_info(self):
        from finsight_mac.mcp.yfinance_client import YFinanceClient

        client = YFinanceClient()
        result = _run(client.get_stock_info("AAPL"))

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_get_price_history(self):
        from finsight_mac.mcp.yfinance_client import YFinanceClient

        client = YFinanceClient()
        result = _run(client.get_price_history("AAPL"))

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Full Aggregator
# ---------------------------------------------------------------------------


class TestAggregatorLive:
    """Live test for the full market data aggregator."""

    def test_fetch_market_data(self):
        from finsight_mac.mcp.market_data import fetch_market_data

        result = _run(fetch_market_data("AAPL"))

        assert isinstance(result, dict)
        assert "_data_sources" in result
        # Should have at least yfinance data (no key needed)
        assert len(result) > 1
