"""Tests for new MCP data clients (FMP, Alpha Vantage, StockData).

All external API calls are mocked to ensure fast, deterministic tests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# FMP Client Tests
# ---------------------------------------------------------------------------


class TestFMPClient:
    """Tests for the Financial Modeling Prep client."""

    def test_no_api_key_returns_none(self):
        """When no FMP API key is configured, _get returns None."""
        with patch("finsight_mac.mcp.fmp_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(fmp_api_key=None)
            from finsight_mac.mcp.fmp_client import FMPClient

            client = FMPClient()
            assert client._client is None

            result = asyncio.get_event_loop().run_until_complete(
                client.get_income_statement("AAPL")
            )
            assert result == {}

    def test_get_income_statement_parses_response(self):
        """get_income_statement correctly maps stable API fields."""
        with patch("finsight_mac.mcp.fmp_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(fmp_api_key="test_key")
            from finsight_mac.mcp.fmp_client import FMPClient

            client = FMPClient()

            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    "date": "2024-09-28",
                    "revenue": 391035000000,
                    "grossProfit": 180683000000,
                    "operatingIncome": 123216000000,
                    "netIncome": 93736000000,
                    "ebitda": 134661000000,
                    "eps": 6.08,
                    "epsDiluted": 6.07,
                    "weightedAverageShsOutDil": 15408095000,
                    "operatingExpenses": 57414000000,
                    "interestExpense": 3002000000,
                    "incomeTaxExpense": 29749000000,
                }
            ]
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)

            result = asyncio.get_event_loop().run_until_complete(
                client.get_income_statement("AAPL")
            )
            assert result["revenue"] == 391035000000
            assert result["eps_diluted"] == 6.07
            assert result["net_income"] == 93736000000
            assert result["period"] == "2024-09-28"

    def test_get_balance_sheet_parses_response(self):
        """get_balance_sheet correctly maps stable API fields."""
        with patch("finsight_mac.mcp.fmp_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(fmp_api_key="test_key")
            from finsight_mac.mcp.fmp_client import FMPClient

            client = FMPClient()

            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    "date": "2024-09-28",
                    "totalAssets": 364980000000,
                    "totalLiabilities": 308030000000,
                    "totalStockholdersEquity": 56950000000,
                    "totalDebt": 101304000000,
                    "netDebt": 76296000000,
                    "cashAndCashEquivalents": 29943000000,
                }
            ]
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)

            result = asyncio.get_event_loop().run_until_complete(client.get_balance_sheet("AAPL"))
            assert result["total_assets"] == 364980000000
            assert result["total_equity"] == 56950000000
            assert result["net_debt"] == 76296000000

    def test_get_cash_flow_parses_response(self):
        """get_cash_flow correctly maps stable API fields."""
        with patch("finsight_mac.mcp.fmp_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(fmp_api_key="test_key")
            from finsight_mac.mcp.fmp_client import FMPClient

            client = FMPClient()

            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    "date": "2024-09-28",
                    "operatingCashFlow": 118254000000,
                    "capitalExpenditure": -9959000000,
                    "freeCashFlow": 108295000000,
                    "commonDividendsPaid": -15025000000,
                    "commonStockRepurchased": -94949000000,
                    "netCashProvidedByInvestingActivities": -4458000000,
                    "netCashProvidedByFinancingActivities": -107388000000,
                }
            ]
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)

            result = asyncio.get_event_loop().run_until_complete(client.get_cash_flow("AAPL"))
            assert result["free_cash_flow"] == 108295000000
            assert result["dividends_paid"] == -15025000000
            assert result["share_repurchases"] == -94949000000

    def test_get_key_metrics_parses_response(self):
        """get_key_metrics correctly maps stable API fields."""
        with patch("finsight_mac.mcp.fmp_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(fmp_api_key="test_key")
            from finsight_mac.mcp.fmp_client import FMPClient

            client = FMPClient()

            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    "date": "2024-09-28",
                    "marketCap": 3440000000000,
                    "enterpriseValue": 3500000000000,
                    "evToEBITDA": 26.0,
                    "currentRatio": 0.87,
                    "returnOnEquity": 1.61,
                    "returnOnInvestedCapital": 0.72,
                    "earningsYield": 0.027,
                }
            ]
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)

            result = asyncio.get_event_loop().run_until_complete(client.get_key_metrics("AAPL"))
            assert result["ev_to_ebitda"] == 26.0
            assert result["roic"] == 0.72
            assert result["roe"] == 1.61

    def test_empty_response_returns_empty(self):
        """Empty or error responses return {}."""
        with patch("finsight_mac.mcp.fmp_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(fmp_api_key="test_key")
            from finsight_mac.mcp.fmp_client import FMPClient

            client = FMPClient()

            mock_response = MagicMock()
            mock_response.json.return_value = []
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)

            result = asyncio.get_event_loop().run_until_complete(
                client.get_income_statement("INVALID")
            )
            assert result == {}


# ---------------------------------------------------------------------------
# Alpha Vantage Client Tests
# ---------------------------------------------------------------------------


class TestAlphaVantageClient:
    """Tests for the Alpha Vantage client."""

    def test_no_api_key_returns_empty(self):
        """When no API key is configured, all methods return {}."""
        with patch("finsight_mac.mcp.alpha_vantage_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(alpha_vantage_api_key=None)
            from finsight_mac.mcp.alpha_vantage_client import AlphaVantageClient

            client = AlphaVantageClient()
            assert client._client is None

            result = asyncio.get_event_loop().run_until_complete(
                client.get_company_overview("AAPL")
            )
            assert result == {}

    def test_get_company_overview_parses_response(self):
        """get_company_overview correctly maps AV fields."""
        with patch("finsight_mac.mcp.alpha_vantage_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(alpha_vantage_api_key="test_key")
            from finsight_mac.mcp.alpha_vantage_client import AlphaVantageClient

            client = AlphaVantageClient()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "Symbol": "AAPL",
                "Name": "Apple Inc",
                "Sector": "Technology",
                "MarketCapitalization": "3440000000000",
                "PERatio": "36.5",
                "PEGRatio": "2.8",
                "EPS": "6.07",
                "Beta": "1.24",
                "52WeekHigh": "237.49",
                "52WeekLow": "164.08",
                "50DayMovingAverage": "225.30",
                "200DayMovingAverage": "210.50",
                "AnalystTargetPrice": "240.00",
            }
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)
            client._last_request_time = 0.0

            result = asyncio.get_event_loop().run_until_complete(
                client.get_company_overview("AAPL")
            )
            assert result["name"] == "Apple Inc"
            assert result["pe_ratio"] == 36.5
            assert result["beta"] == 1.24
            assert result["50d_moving_avg"] == 225.30
            assert result["analyst_target_price"] == 240.00

    def test_get_earnings_parses_response(self):
        """get_earnings correctly computes beat rate and surprise."""
        with patch("finsight_mac.mcp.alpha_vantage_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(alpha_vantage_api_key="test_key")
            from finsight_mac.mcp.alpha_vantage_client import AlphaVantageClient

            client = AlphaVantageClient()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "quarterlyEarnings": [
                    {
                        "fiscalDateEnding": "2024-09-30",
                        "reportedEPS": "1.64",
                        "estimatedEPS": "1.60",
                        "surprisePercentage": "2.5",
                    },
                    {
                        "fiscalDateEnding": "2024-06-30",
                        "reportedEPS": "1.40",
                        "estimatedEPS": "1.35",
                        "surprisePercentage": "3.7",
                    },
                    {
                        "fiscalDateEnding": "2024-03-31",
                        "reportedEPS": "1.52",
                        "estimatedEPS": "1.50",
                        "surprisePercentage": "1.3",
                    },
                    {
                        "fiscalDateEnding": "2023-12-31",
                        "reportedEPS": "2.18",
                        "estimatedEPS": "2.10",
                        "surprisePercentage": "3.8",
                    },
                ]
            }
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)
            client._last_request_time = 0.0

            result = asyncio.get_event_loop().run_until_complete(client.get_earnings("AAPL"))
            assert result["beat_count"] == 4
            assert result["miss_count"] == 0
            assert len(result["quarters"]) == 4
            assert result["avg_surprise_pct"] > 0

    def test_get_technical_indicators_interprets_rsi(self):
        """get_technical_indicators correctly interprets RSI signals."""
        with patch("finsight_mac.mcp.alpha_vantage_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(alpha_vantage_api_key="test_key")
            from finsight_mac.mcp.alpha_vantage_client import AlphaVantageClient

            client = AlphaVantageClient()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "Technical Analysis: RSI": {
                    "2024-12-20": {"RSI": "72.5"},
                    "2024-12-19": {"RSI": "68.3"},
                }
            }
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)
            client._last_request_time = 0.0

            result = asyncio.get_event_loop().run_until_complete(
                client.get_technical_indicators("AAPL")
            )
            assert result["rsi_14"] == 72.5
            assert result["rsi_signal"] == "overbought"

    def test_safe_float_helper(self):
        """_safe_float handles various edge cases."""
        from finsight_mac.mcp.alpha_vantage_client import _safe_float

        assert _safe_float("123.45") == 123.45
        assert _safe_float("0") == 0.0
        assert _safe_float(None) is None
        assert _safe_float("") is None
        assert _safe_float("None") is None
        assert _safe_float("-") is None
        assert _safe_float("invalid") is None


# ---------------------------------------------------------------------------
# StockData Client Tests
# ---------------------------------------------------------------------------


class TestStockDataClient:
    """Tests for the StockData.org news and sentiment client."""

    def test_no_api_key_returns_empty(self):
        """When no API key is configured, get_stock_news returns {}."""
        with patch("finsight_mac.mcp.stockdata_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(stockdata_api_key=None)
            from finsight_mac.mcp.stockdata_client import StockDataClient

            client = StockDataClient()
            assert client._client is None

            result = asyncio.get_event_loop().run_until_complete(client.get_stock_news("AAPL"))
            assert result == {}

    def test_get_stock_news_parses_response(self):
        """get_stock_news correctly parses articles and computes sentiment."""
        with patch("finsight_mac.mcp.stockdata_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(stockdata_api_key="test_key")
            from finsight_mac.mcp.stockdata_client import StockDataClient

            client = StockDataClient()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": [
                    {
                        "title": "Apple Reports Record Q4",
                        "source": "Reuters",
                        "published_at": "2024-12-20T10:00:00Z",
                        "snippet": "Apple beat expectations...",
                        "sentiment": {"score": 0.8, "label": "positive"},
                    },
                    {
                        "title": "iPhone Sales Decline in China",
                        "source": "Bloomberg",
                        "published_at": "2024-12-19T14:00:00Z",
                        "snippet": "Market share fell...",
                        "sentiment": {"score": -0.3, "label": "negative"},
                    },
                    {
                        "title": "Apple AI Strategy Update",
                        "source": "CNBC",
                        "published_at": "2024-12-18T09:00:00Z",
                        "snippet": "New AI features...",
                        "sentiment": {"score": 0.5, "label": "positive"},
                    },
                ]
            }
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)

            result = asyncio.get_event_loop().run_until_complete(client.get_stock_news("AAPL"))
            assert result["article_count"] == 3
            assert len(result["articles"]) == 3
            assert result["articles"][0]["title"] == "Apple Reports Record Q4"
            assert result["articles"][0]["sentiment_score"] == 0.8
            # Average: (0.8 + -0.3 + 0.5) / 3 ≈ 0.333 → positive
            assert result["overall_sentiment"] == "positive"
            assert result["avg_sentiment"] > 0.2

    def test_negative_sentiment(self):
        """Correctly identifies negative overall sentiment."""
        with patch("finsight_mac.mcp.stockdata_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(stockdata_api_key="test_key")
            from finsight_mac.mcp.stockdata_client import StockDataClient

            client = StockDataClient()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": [
                    {
                        "title": "Bad News",
                        "source": "Reuters",
                        "published_at": "2024-12-20",
                        "snippet": "...",
                        "sentiment": {"score": -0.6, "label": "negative"},
                    },
                    {
                        "title": "More Bad News",
                        "source": "Bloomberg",
                        "published_at": "2024-12-19",
                        "snippet": "...",
                        "sentiment": {"score": -0.4, "label": "negative"},
                    },
                ]
            }
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)

            result = asyncio.get_event_loop().run_until_complete(client.get_stock_news("AAPL"))
            assert result["overall_sentiment"] == "negative"
            assert result["avg_sentiment"] < -0.2

    def test_empty_response_returns_empty(self):
        """Empty API response returns {}."""
        with patch("finsight_mac.mcp.stockdata_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(stockdata_api_key="test_key")
            from finsight_mac.mcp.stockdata_client import StockDataClient

            client = StockDataClient()

            mock_response = MagicMock()
            mock_response.json.return_value = {"data": []}
            mock_response.raise_for_status = MagicMock()

            client._client = AsyncMock()
            client._client.get = AsyncMock(return_value=mock_response)

            result = asyncio.get_event_loop().run_until_complete(client.get_stock_news("INVALID"))
            assert result == {}
