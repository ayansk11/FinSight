"""Financial Modeling Prep (FMP) client.

Fetches structured financial statements, key metrics, ratios, and
company profiles. Provides detailed income statement, balance sheet,
and cash flow data that complements yfinance.

Uses the stable API (https://financialmodelingprep.com/stable/).
API key: free at https://site.financialmodelingprep.com/developer (250 req/day)
"""

from __future__ import annotations

import logging

import httpx

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable"


class FMPClient:
    """Async client for Financial Modeling Prep REST API.

    Example:
        >>> client = FMPClient()
        >>> stmt = await client.get_income_statement("AAPL")
        >>> print(stmt["revenue"])
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.fmp_api_key
        self._client = httpx.AsyncClient(timeout=15) if self._api_key else None

    async def _get(
        self, endpoint: str, params: dict | None = None
    ) -> list | dict | None:
        """Make an authenticated GET request to FMP."""
        if not self._client or not self._api_key:
            return None

        full_params = {"apikey": self._api_key}
        if params:
            full_params.update(params)

        try:
            response = await self._client.get(
                f"{BASE_URL}/{endpoint}", params=full_params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"[FMP] {endpoint} failed: {e}")
            return None

    async def get_income_statement(self, ticker: str, limit: int = 1) -> dict:
        """Get annual income statement.

        Returns:
            Dict with revenue, gross_profit, operating_income, net_income,
            ebitda, eps.
        """
        data = await self._get(
            "income-statement", {"symbol": ticker, "limit": str(limit)}
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            return {}

        stmt = data[0]
        return {
            "period": stmt.get("date", ""),
            "revenue": stmt.get("revenue"),
            "cost_of_revenue": stmt.get("costOfRevenue"),
            "gross_profit": stmt.get("grossProfit"),
            "operating_income": stmt.get("operatingIncome"),
            "net_income": stmt.get("netIncome"),
            "ebitda": stmt.get("ebitda"),
            "eps": stmt.get("eps"),
            "eps_diluted": stmt.get("epsDiluted"),
            "weighted_avg_shares": stmt.get("weightedAverageShsOutDil"),
            "operating_expenses": stmt.get("operatingExpenses"),
            "interest_expense": stmt.get("interestExpense"),
            "income_tax_expense": stmt.get("incomeTaxExpense"),
        }

    async def get_balance_sheet(self, ticker: str, limit: int = 1) -> dict:
        """Get annual balance sheet.

        Returns:
            Dict with total_assets, total_liabilities, equity, cash, debt, etc.
        """
        data = await self._get(
            "balance-sheet-statement", {"symbol": ticker, "limit": str(limit)}
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            return {}

        stmt = data[0]
        return {
            "period": stmt.get("date", ""),
            "total_assets": stmt.get("totalAssets"),
            "total_liabilities": stmt.get("totalLiabilities"),
            "total_equity": stmt.get("totalStockholdersEquity"),
            "total_debt": stmt.get("totalDebt"),
            "net_debt": stmt.get("netDebt"),
            "cash_and_equivalents": stmt.get("cashAndCashEquivalents"),
            "short_term_investments": stmt.get("shortTermInvestments"),
            "total_current_assets": stmt.get("totalCurrentAssets"),
            "total_current_liabilities": stmt.get("totalCurrentLiabilities"),
            "inventory": stmt.get("inventory"),
            "accounts_receivable": stmt.get("netReceivables"),
            "accounts_payable": stmt.get("accountPayables"),
            "goodwill": stmt.get("goodwill"),
            "intangible_assets": stmt.get("intangibleAssets"),
            "retained_earnings": stmt.get("retainedEarnings"),
        }

    async def get_cash_flow(self, ticker: str, limit: int = 1) -> dict:
        """Get annual cash flow statement.

        Returns:
            Dict with operating_cf, investing_cf, financing_cf, capex,
            free_cash_flow, etc.
        """
        data = await self._get(
            "cash-flow-statement", {"symbol": ticker, "limit": str(limit)}
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            return {}

        stmt = data[0]
        return {
            "period": stmt.get("date", ""),
            "operating_cash_flow": stmt.get("operatingCashFlow"),
            "investing_cash_flow": stmt.get(
                "netCashProvidedByInvestingActivities"
            ),
            "financing_cash_flow": stmt.get(
                "netCashProvidedByFinancingActivities"
            ),
            "capital_expenditure": stmt.get("capitalExpenditure"),
            "free_cash_flow": stmt.get("freeCashFlow"),
            "dividends_paid": stmt.get("commonDividendsPaid"),
            "share_repurchases": stmt.get("commonStockRepurchased"),
            "depreciation": stmt.get("depreciationAndAmortization"),
            "stock_based_compensation": stmt.get("stockBasedCompensation"),
        }

    async def get_key_metrics(self, ticker: str, limit: int = 1) -> dict:
        """Get key financial metrics and ratios.

        Returns:
            Dict with EV/EBITDA, ROE, ROIC, debt ratios, etc.
        """
        data = await self._get(
            "key-metrics", {"symbol": ticker, "limit": str(limit)}
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            return {}

        m = data[0]
        return {
            "period": m.get("date", ""),
            "market_cap": m.get("marketCap"),
            "enterprise_value": m.get("enterpriseValue"),
            "ev_to_ebitda": m.get("evToEBITDA"),
            "ev_to_sales": m.get("evToSales"),
            "ev_to_operating_cf": m.get("evToOperatingCashFlow"),
            "ev_to_free_cf": m.get("evToFreeCashFlow"),
            "current_ratio": m.get("currentRatio"),
            "roe": m.get("returnOnEquity"),
            "roa": m.get("returnOnAssets"),
            "roic": m.get("returnOnInvestedCapital"),
            "return_on_tangible_assets": m.get("returnOnTangibleAssets"),
            "earnings_yield": m.get("earningsYield"),
            "free_cf_yield": m.get("freeCashFlowYield"),
            "income_quality": m.get("incomeQuality"),
            "working_capital": m.get("workingCapital"),
            "invested_capital": m.get("investedCapital"),
        }

    async def get_financial_ratios(self, ticker: str, limit: int = 1) -> dict:
        """Get detailed financial ratios.

        Returns:
            Dict with profitability, liquidity, leverage, and efficiency ratios.
        """
        data = await self._get(
            "ratios", {"symbol": ticker, "limit": str(limit)}
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            return {}

        r = data[0]
        return {
            "period": r.get("date", ""),
            "gross_profit_margin": r.get("grossProfitMargin"),
            "operating_profit_margin": r.get("operatingProfitMargin"),
            "net_profit_margin": r.get("netProfitMargin"),
            "return_on_equity": r.get("returnOnEquity"),
            "return_on_assets": r.get("returnOnAssets"),
            "return_on_capital": r.get("returnOnCapitalEmployed"),
            "debt_equity_ratio": r.get("debtToEquityRatio"),
            "debt_ratio": r.get("debtToAssetsRatio"),
            "current_ratio": r.get("currentRatio"),
            "quick_ratio": r.get("quickRatio"),
            "cash_ratio": r.get("cashRatio"),
            "inventory_turnover": r.get("inventoryTurnover"),
            "receivables_turnover": r.get("receivablesTurnover"),
            "asset_turnover": r.get("assetTurnover"),
            "dividend_payout_ratio": r.get("dividendPayoutRatio"),
            "price_earnings_ratio": r.get("priceToEarningsRatio"),
            "price_to_book": r.get("priceToBookRatio"),
            "price_to_sales": r.get("priceToSalesRatio"),
            "effective_tax_rate": r.get("effectiveTaxRate"),
            "interest_coverage": r.get("interestCoverageRatio"),
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
