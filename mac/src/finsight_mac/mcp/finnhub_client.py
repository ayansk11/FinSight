"""Finnhub market data client.

Fetches real-time quotes, company profiles, basic financials, and
analyst recommendations. Uses raw httpx (already a dependency) rather
than the finnhub-python package.

API key: free at https://finnhub.io/register (60 req/min)
"""

from __future__ import annotations

import logging
import time

import httpx

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://finnhub.io/api/v1"

# Rate limit: 60 requests per minute → 1 req/sec
RATE_LIMIT_DELAY = 1.05


class FinnhubClient:
    """Async client for the Finnhub REST API.

    Example:
        >>> client = FinnhubClient()
        >>> quote = await client.get_quote("AAPL")
        >>> print(quote)
        {'current_price': 189.45, 'change': 1.23, 'change_pct': 0.65, ...}
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.finnhub_api_key
        self._last_request_time = 0.0
        self._client = httpx.AsyncClient(timeout=15) if self._api_key else None

    async def _rate_limit(self) -> None:
        """Enforce Finnhub's 60 req/min rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            import asyncio

            await asyncio.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    async def _get(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        """Make an authenticated GET request to Finnhub."""
        if not self._client or not self._api_key:
            return None

        await self._rate_limit()

        full_params = {"token": self._api_key}
        if params:
            full_params.update(params)

        try:
            response = await self._client.get(f"{BASE_URL}/{endpoint}", params=full_params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"[Finnhub] {endpoint} failed: {e}")
            return None

    async def get_quote(self, ticker: str) -> dict:
        """Get real-time stock quote.

        Returns:
            Dict with current_price, change, change_pct, high, low, open, prev_close, volume.
        """
        data = await self._get("quote", {"symbol": ticker})
        if not data or data.get("c", 0) == 0:
            return {}

        return {
            "current_price": data.get("c", 0),
            "change": data.get("d", 0),
            "change_pct": data.get("dp", 0),
            "day_high": data.get("h", 0),
            "day_low": data.get("l", 0),
            "open": data.get("o", 0),
            "prev_close": data.get("pc", 0),
        }

    async def get_company_profile(self, ticker: str) -> dict:
        """Get company profile information.

        Returns:
            Dict with name, sector, market_cap, ipo_date, exchange, etc.
        """
        data = await self._get("stock/profile2", {"symbol": ticker})
        if not data or not data.get("name"):
            return {}

        market_cap = data.get("marketCapitalization", 0)
        return {
            "company_name": data.get("name", ""),
            "sector": data.get("finnhubIndustry", ""),
            "market_cap_millions": market_cap,
            "ipo_date": data.get("ipo", ""),
            "exchange": data.get("exchange", ""),
            "country": data.get("country", ""),
            "web_url": data.get("weburl", ""),
        }

    async def get_basic_financials(self, ticker: str) -> dict:
        """Get key financial metrics.

        Returns:
            Dict with PE, EPS, margins, 52-week range, etc.
        """
        data = await self._get("stock/metric", {"symbol": ticker, "metric": "all"})
        if not data:
            return {}

        metrics = data.get("metric", {})
        if not metrics:
            return {}

        return {
            "pe_ratio_ttm": metrics.get("peNormalizedAnnual"),
            "eps_ttm": metrics.get("epsNormalizedAnnual"),
            "revenue_per_share_ttm": metrics.get("revenuePerShareTTM"),
            "52w_high": metrics.get("52WeekHigh"),
            "52w_low": metrics.get("52WeekLow"),
            "52w_high_date": metrics.get("52WeekHighDate"),
            "52w_low_date": metrics.get("52WeekLowDate"),
            "beta": metrics.get("beta"),
            "dividend_yield_ttm": metrics.get("dividendYieldIndicatedAnnual"),
            "10d_avg_volume": metrics.get("10DayAverageTradingVolume"),
            "gross_margin_ttm": metrics.get("grossMarginTTM"),
            "operating_margin_ttm": metrics.get("operatingMarginTTM"),
            "net_margin_ttm": metrics.get("netProfitMarginTTM"),
            "roe_ttm": metrics.get("roeTTM"),
            "roa_ttm": metrics.get("roaTTM"),
            "debt_to_equity": metrics.get("totalDebt/totalEquityQuarterly"),
            "current_ratio": metrics.get("currentRatioQuarterly"),
            "pb_ratio": metrics.get("pbAnnual"),
        }

    async def get_recommendation_trends(self, ticker: str) -> dict:
        """Get analyst recommendation trends.

        Returns:
            Dict with buy, hold, sell counts and consensus.
        """
        data = await self._get("stock/recommendation", {"symbol": ticker})
        if not data or not isinstance(data, list) or len(data) == 0:
            return {}

        # Use most recent month
        latest = data[0]
        buy = latest.get("buy", 0) + latest.get("strongBuy", 0)
        hold = latest.get("hold", 0)
        sell = latest.get("sell", 0) + latest.get("strongSell", 0)
        total = buy + hold + sell

        if total == 0:
            return {}

        if buy > hold and buy > sell:
            consensus = "Buy"
        elif sell > hold and sell > buy:
            consensus = "Sell"
        else:
            consensus = "Hold"

        return {
            "analyst_buy": buy,
            "analyst_hold": hold,
            "analyst_sell": sell,
            "analyst_total": total,
            "analyst_consensus": consensus,
            "analyst_period": latest.get("period", ""),
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
