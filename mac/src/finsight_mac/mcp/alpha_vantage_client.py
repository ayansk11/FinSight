"""Alpha Vantage market data client.

Fetches historical stock prices, technical indicators, and earnings data.
Complements yfinance with reliable technical analysis indicators and
earnings surprise data.

API key: free at https://www.alphavantage.co/support/#api-key (25 req/day free)
"""

from __future__ import annotations

import logging
import time

import httpx

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"

# Free tier: 25 requests per day, 5 per minute
RATE_LIMIT_DELAY = 12.5  # ~5 req/min


class AlphaVantageClient:
    """Async client for the Alpha Vantage REST API.

    Example:
        >>> client = AlphaVantageClient()
        >>> overview = await client.get_company_overview("AAPL")
        >>> print(overview["market_cap"])
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.alpha_vantage_api_key
        self._last_request_time = 0.0
        self._client = httpx.AsyncClient(timeout=20) if self._api_key else None

    async def _rate_limit(self) -> None:
        """Enforce Alpha Vantage rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            import asyncio
            await asyncio.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    async def _get(self, params: dict) -> dict | None:
        """Make an authenticated GET request to Alpha Vantage."""
        if not self._client or not self._api_key:
            return None

        await self._rate_limit()

        full_params = {"apikey": self._api_key, **params}

        try:
            response = await self._client.get(BASE_URL, params=full_params)
            response.raise_for_status()
            data = response.json()

            # Alpha Vantage returns error messages in the response body
            if "Error Message" in data:
                logger.warning(f"[AlphaVantage] API error: {data['Error Message']}")
                return None
            if "Note" in data:
                logger.warning(f"[AlphaVantage] Rate limit: {data['Note']}")
                return None

            return data
        except Exception as e:
            logger.warning(f"[AlphaVantage] Request failed: {e}")
            return None

    async def get_company_overview(self, ticker: str) -> dict:
        """Get comprehensive company overview with key metrics.

        Returns:
            Dict with fundamentals: PE, PEG, book value, EPS, revenue, profit margins, etc.
        """
        data = await self._get({"function": "OVERVIEW", "symbol": ticker})
        if not data or not data.get("Symbol"):
            return {}

        return {
            "name": data.get("Name", ""),
            "description": data.get("Description", ""),
            "sector": data.get("Sector", ""),
            "industry": data.get("Industry", ""),
            "market_cap": _safe_float(data.get("MarketCapitalization")),
            "pe_ratio": _safe_float(data.get("PERatio")),
            "peg_ratio": _safe_float(data.get("PEGRatio")),
            "book_value": _safe_float(data.get("BookValue")),
            "eps": _safe_float(data.get("EPS")),
            "dividend_per_share": _safe_float(data.get("DividendPerShare")),
            "dividend_yield": _safe_float(data.get("DividendYield")),
            "revenue_ttm": _safe_float(data.get("RevenueTTM")),
            "gross_profit_ttm": _safe_float(data.get("GrossProfitTTM")),
            "ebitda": _safe_float(data.get("EBITDA")),
            "profit_margin": _safe_float(data.get("ProfitMargin")),
            "operating_margin": _safe_float(data.get("OperatingMarginTTM")),
            "return_on_assets": _safe_float(data.get("ReturnOnAssetsTTM")),
            "return_on_equity": _safe_float(data.get("ReturnOnEquityTTM")),
            "revenue_per_share": _safe_float(data.get("RevenuePerShareTTM")),
            "quarterly_revenue_growth_yoy": _safe_float(data.get("QuarterlyRevenueGrowthYOY")),
            "quarterly_earnings_growth_yoy": _safe_float(data.get("QuarterlyEarningsGrowthYOY")),
            "analyst_target_price": _safe_float(data.get("AnalystTargetPrice")),
            "analyst_rating": data.get("AnalystRatingStrongBuy", ""),
            "forward_pe": _safe_float(data.get("ForwardPE")),
            "price_to_sales": _safe_float(data.get("PriceToSalesRatioTTM")),
            "price_to_book": _safe_float(data.get("PriceToBookRatio")),
            "ev_to_revenue": _safe_float(data.get("EVToRevenue")),
            "ev_to_ebitda": _safe_float(data.get("EVToEBITDA")),
            "beta": _safe_float(data.get("Beta")),
            "52w_high": _safe_float(data.get("52WeekHigh")),
            "52w_low": _safe_float(data.get("52WeekLow")),
            "50d_moving_avg": _safe_float(data.get("50DayMovingAverage")),
            "200d_moving_avg": _safe_float(data.get("200DayMovingAverage")),
            "shares_outstanding": _safe_float(data.get("SharesOutstanding")),
        }

    async def get_earnings(self, ticker: str) -> dict:
        """Get quarterly earnings data with surprise metrics.

        Returns:
            Dict with recent quarterly earnings, estimates, and surprise percentages.
        """
        data = await self._get({"function": "EARNINGS", "symbol": ticker})
        if not data:
            return {}

        quarterly = data.get("quarterlyEarnings", [])
        if not quarterly:
            return {}

        # Return last 4 quarters
        recent = quarterly[:4]
        result = {
            "quarters": [],
            "avg_surprise_pct": 0.0,
            "beat_count": 0,
            "miss_count": 0,
        }

        surprises = []
        for q in recent:
            surprise_pct = _safe_float(q.get("surprisePercentage"))
            estimated = _safe_float(q.get("estimatedEPS"))
            reported = _safe_float(q.get("reportedEPS"))

            quarter_data = {
                "date": q.get("fiscalDateEnding", ""),
                "reported_eps": reported,
                "estimated_eps": estimated,
                "surprise_pct": surprise_pct,
            }
            result["quarters"].append(quarter_data)

            if surprise_pct is not None:
                surprises.append(surprise_pct)
                if surprise_pct > 0:
                    result["beat_count"] += 1
                elif surprise_pct < 0:
                    result["miss_count"] += 1

        if surprises:
            result["avg_surprise_pct"] = sum(surprises) / len(surprises)

        return result

    async def get_technical_indicators(self, ticker: str) -> dict:
        """Get key technical indicators (RSI, SMA, EMA).

        Makes a single call for RSI-14 daily. SMA/EMA are derived
        from the company overview's moving averages to conserve API calls.

        Returns:
            Dict with rsi_14, sma_50, sma_200, and signal interpretation.
        """
        data = await self._get({
            "function": "RSI",
            "symbol": ticker,
            "interval": "daily",
            "time_period": "14",
            "series_type": "close",
        })
        if not data:
            return {}

        rsi_data = data.get("Technical Analysis: RSI", {})
        if not rsi_data:
            return {}

        # Get the most recent RSI value
        latest_date = next(iter(rsi_data), None)
        if not latest_date:
            return {}

        rsi = _safe_float(rsi_data[latest_date].get("RSI"))
        if rsi is None:
            return {}

        # Interpret RSI
        if rsi > 70:
            signal = "overbought"
        elif rsi < 30:
            signal = "oversold"
        elif rsi > 60:
            signal = "bullish"
        elif rsi < 40:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "rsi_14": rsi,
            "rsi_signal": signal,
            "rsi_date": latest_date,
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


def _safe_float(value: str | None) -> float | None:
    """Convert a string to float, returning None for invalid values."""
    if value is None or value == "" or value == "None" or value == "-":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
