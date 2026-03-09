"""FRED (Federal Reserve Economic Data) client.

Fetches macroeconomic indicators — GDP, CPI, federal funds rate,
unemployment, Treasury yields — to provide macro context for the
Quantitative Analysis Agent.

API key: free at https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)

# Standard macro series IDs
MACRO_SERIES = {
    "gdp_growth_yoy": "A191RL1Q225SBEA",  # Real GDP growth rate (quarterly, annualized)
    "cpi_yoy": "CPIAUCSL",  # CPI for all urban consumers (monthly)
    "fed_funds_rate": "FEDFUNDS",  # Effective federal funds rate (monthly)
    "unemployment_rate": "UNRATE",  # Unemployment rate (monthly)
    "treasury_10y": "GS10",  # 10-Year Treasury constant maturity (monthly)
    "treasury_2y": "GS2",  # 2-Year Treasury (monthly, for yield curve)
}


class FredClient:
    """Async wrapper around the fredapi library.

    Example:
        >>> client = FredClient()
        >>> data = await client.get_macro_snapshot()
        >>> print(data)
        {'gdp_growth_yoy': '2.5%', 'fed_funds_rate': '5.33%', ...}
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.fred_api_key
        self._fred = None

    def _get_fred(self):
        """Lazy-initialize fredapi.Fred client."""
        if self._fred is None:
            from fredapi import Fred

            self._fred = Fred(api_key=self._api_key)
        return self._fred

    async def get_series_latest(self, series_id: str, periods: int = 1) -> list[dict]:
        """Fetch the latest observations from a FRED series.

        Args:
            series_id: FRED series ID (e.g., "FEDFUNDS").
            periods: Number of recent observations to return.

        Returns:
            List of {date, value} dicts, most recent first.
        """
        if not self._api_key:
            return []

        def _fetch():
            fred = self._get_fred()
            # Fetch last 2 years to ensure we get enough data
            end = datetime.now()
            start = end - timedelta(days=730)
            series = fred.get_series(series_id, observation_start=start, observation_end=end)
            # Drop NaN and get latest
            series = series.dropna().tail(periods)
            return [
                {"date": idx.strftime("%Y-%m-%d"), "value": float(val)}
                for idx, val in series.items()
            ]

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.warning(f"[FRED] Failed to fetch {series_id}: {e}")
            return []

    async def get_macro_snapshot(self) -> dict:
        """Fetch a standard set of macroeconomic indicators.

        Returns:
            Flat dict of indicator_name → formatted value string.
        """
        if not self._api_key:
            logger.info("[FRED] No API key configured, skipping macro data")
            return {}

        # Fetch all series concurrently
        tasks = {
            name: self.get_series_latest(series_id, periods=1)
            for name, series_id in MACRO_SERIES.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        snapshot = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"[FRED] {name} failed: {result}")
                continue
            if result:
                value = result[-1]["value"]
                snapshot[name] = _format_macro_value(name, value)

        # Compute yield curve spread if both yields available
        if "treasury_10y" in snapshot and "treasury_2y" in snapshot:
            try:
                t10 = float(snapshot["treasury_10y"].rstrip("%"))
                t2 = float(snapshot["treasury_2y"].rstrip("%"))
                spread = t10 - t2
                snapshot["yield_curve_spread"] = f"{spread:+.2f}%"
            except (ValueError, TypeError):
                pass

        logger.info(f"[FRED] Macro snapshot: {len(snapshot)} indicators fetched")
        return snapshot

    async def close(self) -> None:
        """No persistent connections to close for fredapi."""
        pass


def _format_macro_value(name: str, value: float) -> str:
    """Format a macro value for display."""
    if "rate" in name or "treasury" in name or "cpi" in name:
        return f"{value:.2f}%"
    if "gdp" in name:
        return f"{value:.1f}%"
    return f"{value:.2f}"
