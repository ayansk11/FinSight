"""yfinance stock data client.

Fetches stock info, financial statements, and price history from
Yahoo Finance. No API key required.

Package: yfinance (pip install yfinance)
"""

from __future__ import annotations

import asyncio
import logging

from finsight_mac.config import get_settings  # noqa: F401

logger = logging.getLogger(__name__)


class YFinanceClient:
    """Async wrapper around the yfinance library.

    All yfinance calls are synchronous — we wrap them in asyncio.to_thread()
    to avoid blocking the event loop.

    Example:
        >>> client = YFinanceClient()
        >>> info = await client.get_stock_info("AAPL")
        >>> print(info["market_cap"])
        '$2.95T'
    """

    async def get_stock_info(self, ticker: str) -> dict:
        """Get key stock information.

        Returns:
            Dict with market_cap, pe_ratio, beta, dividend_yield, sector, etc.
        """

        def _fetch():
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or not info.get("shortName"):
                return {}

            return {
                "company_name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap"),
                "pe_ratio_trailing": info.get("trailingPE"),
                "pe_ratio_forward": info.get("forwardPE"),
                "eps_trailing": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "beta": info.get("beta"),
                "dividend_yield": info.get("dividendYield"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "enterprise_value": info.get("enterpriseValue"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "return_on_equity": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "free_cash_flow": info.get("freeCashflow"),
                "total_revenue": info.get("totalRevenue"),
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash"),
            }

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.warning(f"[yfinance] get_stock_info({ticker}) failed: {e}")
            return {}

    async def get_financials(self, ticker: str) -> dict:
        """Get latest financial statement highlights.

        Returns:
            Dict with key income statement, balance sheet, and cash flow items.
        """

        def _fetch():
            import yfinance as yf

            stock = yf.Ticker(ticker)
            result = {}

            # Income statement
            income = stock.income_stmt
            if income is not None and not income.empty:
                latest = income.iloc[:, 0]  # Most recent period
                col = income.columns[0]
                period = str(col.date()) if hasattr(col, "date") else str(col)
                result["income_period"] = period
                income_rows = [
                    "Total Revenue", "Gross Profit",
                    "Operating Income", "Net Income", "EBITDA",
                ]
                for row in income_rows:
                    if row in latest.index:
                        result[f"income_{row.lower().replace(' ', '_')}"] = float(latest[row])

            # Balance sheet
            bs = stock.balance_sheet
            if bs is not None and not bs.empty:
                latest = bs.iloc[:, 0]
                bs_rows = [
                    "Total Assets",
                    "Total Liabilities Net Minority Interest",
                    "Total Equity Gross Minority Interest",
                    "Total Debt",
                ]
                for row in bs_rows:
                    if row in latest.index:
                        key = (
                            row.lower()
                            .replace(" ", "_")
                            .replace("_net_minority_interest", "")
                            .replace("_gross_minority_interest", "")
                        )
                        result[f"bs_{key}"] = float(latest[row])

            # Cash flow
            cf = stock.cashflow
            if cf is not None and not cf.empty:
                latest = cf.iloc[:, 0]
                for row in ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"]:
                    if row in latest.index:
                        result[f"cf_{row.lower().replace(' ', '_')}"] = float(latest[row])

            return result

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.warning(f"[yfinance] get_financials({ticker}) failed: {e}")
            return {}

    async def get_price_history(self, ticker: str, period: str = "1y") -> dict:
        """Get price history summary stats.

        Args:
            ticker: Stock ticker.
            period: yfinance period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y).

        Returns:
            Dict with 52w_high, 52w_low, ytd_return, avg_volume, etc.
        """

        def _fetch():
            import yfinance as yf

            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist is None or hist.empty:
                return {}

            result = {
                "period": period,
                "period_high": float(hist["High"].max()),
                "period_low": float(hist["Low"].min()),
                "period_start_price": float(hist["Close"].iloc[0]),
                "period_end_price": float(hist["Close"].iloc[-1]),
                "avg_volume": int(hist["Volume"].mean()),
            }

            # Compute return over period
            start_price = result["period_start_price"]
            end_price = result["period_end_price"]
            if start_price > 0:
                result["period_return"] = (end_price - start_price) / start_price

            return result

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.warning(f"[yfinance] get_price_history({ticker}) failed: {e}")
            return {}

    async def close(self) -> None:
        """No persistent connections to close for yfinance."""
        pass
