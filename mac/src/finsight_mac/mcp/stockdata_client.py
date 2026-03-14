"""StockData.org market news and sentiment client.

Fetches real-time market news, stock sentiment data, and trending
tickers. Provides news sentiment scoring that feeds into the Risk
Agent for market sentiment analysis.

API key: free at https://www.stockdata.org (100 req/day)
"""

from __future__ import annotations

import logging

import httpx

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.stockdata.org/v1"


class StockDataClient:
    """Async client for the StockData.org REST API.

    Example:
        >>> client = StockDataClient()
        >>> news = await client.get_stock_news("AAPL")
        >>> print(news["articles"][0]["title"])
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.stockdata_api_key
        self._client = httpx.AsyncClient(timeout=15) if self._api_key else None

    async def _get(
        self, endpoint: str, params: dict | None = None
    ) -> dict | None:
        """Make an authenticated GET request to StockData."""
        if not self._client or not self._api_key:
            return None

        full_params = {"api_token": self._api_key}
        if params:
            full_params.update(params)

        try:
            response = await self._client.get(
                f"{BASE_URL}/{endpoint}", params=full_params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"[StockData] {endpoint} failed: {e}")
            return None

    async def get_stock_news(self, ticker: str, limit: int = 5) -> dict:
        """Get recent news articles for a stock.

        Args:
            ticker: Stock ticker symbol.
            limit: Number of articles to return (max 5 on free tier).

        Returns:
            Dict with articles list and sentiment summary.
        """
        data = await self._get(
            "news/all",
            {
                "symbols": ticker,
                "filter_entities": "true",
                "limit": str(limit),
                "language": "en",
            },
        )
        if not data or not data.get("data"):
            return {}

        articles = []
        sentiment_scores = []

        for item in data["data"][:limit]:
            sentiment = item.get("sentiment")
            article = {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "published_at": item.get("published_at", ""),
                "snippet": item.get("snippet", ""),
            }

            if sentiment:
                score = sentiment.get("score", 0)
                article["sentiment_score"] = score
                article["sentiment_label"] = sentiment.get("label", "")
                sentiment_scores.append(score)

            articles.append(article)

        result = {"articles": articles, "article_count": len(articles)}

        if sentiment_scores:
            avg = sum(sentiment_scores) / len(sentiment_scores)
            result["avg_sentiment"] = avg
            if avg > 0.2:
                result["overall_sentiment"] = "positive"
            elif avg < -0.2:
                result["overall_sentiment"] = "negative"
            else:
                result["overall_sentiment"] = "neutral"

        return result

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
