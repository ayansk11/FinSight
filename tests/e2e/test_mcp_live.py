"""Live MCP data fetching tests (no API key needed for yfinance).

These tests hit real external APIs. They are marked as e2e
and will skip if the network is unreachable or yfinance fails.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.e2e
class TestYFinanceLive:
    """Live tests against Yahoo Finance (no API key required)."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_yfinance_stock_info(self):
        """Fetch real AAPL stock info via yfinance."""
        from finsight_mac.mcp.yfinance_client import YFinanceClient

        client = YFinanceClient()
        info = self._run(client.get_stock_info("AAPL"))

        if not info:
            pytest.skip("yfinance returned empty data (network issue?)")

        assert info.get("company_name"), "Should have company name"
        assert info.get("market_cap") is not None, "Should have market cap"
        assert info["market_cap"] > 1e11, "AAPL market cap should be > $100B"
        assert info.get("sector"), "Should have sector"

    def test_yfinance_price_history(self):
        """Fetch real AAPL price history."""
        from finsight_mac.mcp.yfinance_client import YFinanceClient

        client = YFinanceClient()
        hist = self._run(client.get_price_history("AAPL", period="1mo"))

        if not hist:
            pytest.skip("yfinance returned empty history (network issue?)")

        assert hist.get("period") == "1mo"
        assert hist.get("period_high") is not None
        assert hist.get("period_low") is not None
        assert hist["period_high"] >= hist["period_low"]


@pytest.mark.e2e
class TestMarketDataAggregatorLive:
    """Live test for the market data aggregator."""

    def test_fetch_market_data(self):
        """Aggregate market data for AAPL — yfinance should always work."""
        from finsight_mac.mcp.market_data import fetch_market_data

        result = asyncio.get_event_loop().run_until_complete(fetch_market_data("AAPL"))

        if not result:
            pytest.skip("All data sources returned empty (network issue?)")

        # yfinance should at least populate some fields
        assert "_data_sources" in result
        assert "yfinance" in result["_data_sources"]

        # Should have basic valuation data
        has_data = any(
            key in result for key in ["pe_ratio", "market_cap", "revenue_ttm", "eps_ttm"]
        )
        assert has_data, f"Expected valuation data, got keys: {list(result.keys())}"
