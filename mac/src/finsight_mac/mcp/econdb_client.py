"""Econdb global macroeconomic data client.

Fetches international macro indicators — GDP, inflation, unemployment,
interest rates — for major economies. Complements FRED (US-only) with
global coverage. No API key required.

Data source: https://www.econdb.com/api/
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://www.econdb.com/api/series"

# Key global macro series — ticker format: "TICKER"
# Econdb uses its own ticker system for series
GLOBAL_SERIES = {
    # US (supplements FRED)
    "us_gdp_growth": "RGDP_US",
    "us_inflation": "CPI_US",
    "us_unemployment": "URATE_US",
    "us_interest_rate": "POLIR_US",
    # Eurozone
    "eu_gdp_growth": "RGDP_EA19",
    "eu_inflation": "CPI_EA19",
    "eu_unemployment": "URATE_EA19",
    "eu_interest_rate": "POLIR_EA19",
    # China
    "cn_gdp_growth": "RGDP_CN",
    "cn_inflation": "CPI_CN",
    # Japan
    "jp_gdp_growth": "RGDP_JP",
    "jp_inflation": "CPI_JP",
    "jp_interest_rate": "POLIR_JP",
    # UK
    "uk_gdp_growth": "RGDP_UK",
    "uk_inflation": "CPI_UK",
    "uk_unemployment": "URATE_UK",
    "uk_interest_rate": "POLIR_UK",
}


class EcondbClient:
    """Async client for the Econdb REST API (no auth required).

    Example:
        >>> client = EcondbClient()
        >>> snapshot = await client.get_global_macro_snapshot()
        >>> print(snapshot["eu_inflation"])
        '2.4%'
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=15)

    async def _get_series(self, ticker: str) -> dict | None:
        """Fetch the latest data point for an Econdb series."""
        try:
            response = await self._client.get(
                f"{BASE_URL}/{ticker}/",
                params={"format": "json"},
            )
            response.raise_for_status()
            data = response.json()

            # Econdb returns {ticker, description, data: {dates: [...], values: [...]}}
            series_data = data.get("data", {})
            dates = series_data.get("dates", [])
            values = series_data.get("values", [])

            if not dates or not values:
                return None

            return {
                "date": dates[-1],
                "value": values[-1],
                "description": data.get("description", ""),
            }
        except Exception as e:
            logger.warning(f"[Econdb] Failed to fetch {ticker}: {e}")
            return None

    async def get_series_latest(self, name: str) -> dict:
        """Fetch the latest value for a named macro series.

        Args:
            name: Key from GLOBAL_SERIES (e.g., "eu_inflation").

        Returns:
            Dict with date, value, and formatted string.
        """
        ticker = GLOBAL_SERIES.get(name)
        if not ticker:
            return {}

        result = await self._get_series(ticker)
        if not result:
            return {}

        return {
            "date": result["date"],
            "value": result["value"],
            "formatted": _format_macro_value(name, result["value"]),
        }

    async def get_global_macro_snapshot(self) -> dict:
        """Fetch a snapshot of key global macro indicators.

        Returns:
            Flat dict of indicator_name → formatted value string.
        """
        import asyncio

        tasks = {name: self._get_series(ticker) for name, ticker in GLOBAL_SERIES.items()}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        snapshot = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"[Econdb] {name} failed: {result}")
                continue
            if result and result.get("value") is not None:
                snapshot[name] = _format_macro_value(name, result["value"])

        logger.info(f"[Econdb] Global macro snapshot: {len(snapshot)} indicators fetched")
        return snapshot

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


def _format_macro_value(name: str, value: float) -> str:
    """Format a macro value for display."""
    if "inflation" in name or "interest_rate" in name or "unemployment" in name:
        return f"{value:.1f}%"
    if "gdp" in name:
        return f"{value:.1f}%"
    return f"{value:.2f}"
