"""SEC EDGAR filing downloader.

Downloads SEC filings (10-K, 10-Q, 8-K) from EDGAR's HTTPS interface.
No API key required — just a User-Agent identifying the requester.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

# SEC rate limit: 10 requests/second
SEC_RATE_LIMIT_DELAY = 0.12


class EdgarClient:
    """Client for downloading SEC filings from EDGAR.

    Respects SEC's rate limiting (10 req/sec) and User-Agent requirement.

    Example:
        >>> client = EdgarClient()
        >>> filings = await client.search_filings("AAPL", "10-K", limit=3)
        >>> pdf_path = await client.download_filing(filings[0])
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.user_agent = settings.sec_user_agent
        self.filings_dir = settings.filings_dir
        self._last_request_time = 0.0
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=30,
            follow_redirects=True,
        )

    async def _rate_limit(self) -> None:
        """Enforce SEC's rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < SEC_RATE_LIMIT_DELAY:
            await _async_sleep(SEC_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    async def get_company_cik(self, ticker: str) -> str | None:
        """Look up a company's CIK number from ticker.

        Args:
            ticker: Company ticker symbol (e.g., "AAPL").

        Returns:
            CIK number as a zero-padded string, or None if not found.
        """
        await self._rate_limit()

        try:
            # Try the ticker-to-CIK lookup via company tickers file
            response = await self._client.get("https://www.sec.gov/files/company_tickers.json")
            response.raise_for_status()
            data = response.json()

            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    logger.info(f"Found CIK for {ticker}: {cik}")
                    return cik

        except Exception as e:
            logger.warning(f"CIK lookup failed for {ticker}: {e}")

        return None

    async def search_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        limit: int = 5,
    ) -> list[dict]:
        """Search for recent filings by ticker and type.

        Args:
            ticker: Company ticker symbol.
            filing_type: SEC filing type (10-K, 10-Q, 8-K, etc.).
            limit: Maximum number of results.

        Returns:
            List of filing metadata dicts.
        """
        cik = await self.get_company_cik(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker: {ticker}")
            return []

        await self._rate_limit()

        url = f"{EDGAR_SUBMISSIONS_URL}/CIK{cik}.json"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            data = response.json()

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            results = []
            for i, form in enumerate(forms):
                if form == filing_type and len(results) < limit:
                    results.append(
                        {
                            "ticker": ticker,
                            "company_name": data.get("name", ticker),
                            "filing_type": form,
                            "filing_date": dates[i] if i < len(dates) else "",
                            "accession_number": accessions[i] if i < len(accessions) else "",
                            "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                            "cik": cik,
                        }
                    )

            logger.info(f"Found {len(results)} {filing_type} filings for {ticker}")
            return results

        except Exception as e:
            logger.error(f"Filing search failed: {e}")
            return []

    async def download_filing(
        self,
        filing_meta: dict,
        output_dir: Path | None = None,
    ) -> Path | None:
        """Download a filing document.

        Args:
            filing_meta: Filing metadata from search_filings().
            output_dir: Optional output directory (defaults to filings_dir).

        Returns:
            Path to the downloaded file, or None if download failed.
        """
        output_dir = output_dir or self.filings_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        cik = filing_meta.get("cik", "").lstrip("0")
        accession = filing_meta.get("accession_number", "").replace("-", "")
        primary_doc = filing_meta.get("primary_document", "")

        if not all([cik, accession, primary_doc]):
            logger.error("Incomplete filing metadata for download")
            return None

        url = f"{EDGAR_ARCHIVES_URL}/{cik}/{accession}/{primary_doc}"
        ticker = filing_meta.get("ticker", "unknown")
        filing_type = filing_meta.get("filing_type", "filing")
        filing_date = filing_meta.get("filing_date", "")

        filename = f"{ticker}_{filing_type}_{filing_date}{Path(primary_doc).suffix}"
        output_path = output_dir / filename

        if output_path.exists():
            logger.info(f"Filing already downloaded: {output_path}")
            return output_path

        await self._rate_limit()

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            logger.info(f"Downloaded filing: {output_path} ({len(response.content)} bytes)")
            return output_path
        except Exception as e:
            logger.error(f"Filing download failed: {e}")
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio

    await asyncio.sleep(seconds)
