"""Market data aggregator.

Single entry point that calls FRED, Finnhub, and yfinance concurrently
and returns a unified flat dict for the Quantitative Analysis Agent.
"""

from __future__ import annotations

import asyncio
import logging

from finsight_mac.mcp.finnhub_client import FinnhubClient
from finsight_mac.mcp.fred import FredClient
from finsight_mac.mcp.yfinance_client import YFinanceClient

logger = logging.getLogger(__name__)


def _fmt_large_number(n: float | int | None) -> str:
    """Format a large number with B/M/T suffix."""
    if n is None:
        return "N/A"
    n = float(n)
    if abs(n) >= 1e12:
        return f"${n / 1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n / 1e9:.1f}B"
    if abs(n) >= 1e6:
        return f"${n / 1e6:.1f}M"
    return f"${n:,.0f}"


def _fmt_pct(v: float | None, decimals: int = 2) -> str:
    """Format a decimal ratio as a percentage string.

    Treats ALL values as ratios and multiplies by 100.
    E.g., 0.47 → '47.00%', 1.52 → '152.00%'.
    """
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def _fmt_pct_raw(v: float | None, decimals: int = 2) -> str:
    """Format a value that is ALREADY a percentage (not a ratio).

    yfinance's dividendYield is inconsistent — sometimes returned
    as a raw percentage (0.4 = 0.4%) rather than a ratio (0.004).
    """
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}%"


async def fetch_market_data(ticker: str) -> dict:
    """Fetch market data from all available sources.

    Calls FRED, Finnhub, and yfinance concurrently and merges
    their results into a single flat dict. Each source failing
    gracefully does not block the others.

    Args:
        ticker: Company ticker symbol (e.g., "AAPL").

    Returns:
        Flat dict of key → formatted value string, suitable for
        passing directly to run_quant_agent(market_data=...).
    """
    if not ticker:
        return {}

    fred = FredClient()
    finnhub = FinnhubClient()
    yf_client = YFinanceClient()

    try:
        # Run all fetches concurrently
        (
            macro,
            quote,
            profile,
            financials,
            recommendations,
            stock_info,
            yf_financials,
            price_hist,
        ) = await asyncio.gather(
            fred.get_macro_snapshot(),
            finnhub.get_quote(ticker),
            finnhub.get_company_profile(ticker),
            finnhub.get_basic_financials(ticker),
            finnhub.get_recommendation_trends(ticker),
            yf_client.get_stock_info(ticker),
            yf_client.get_financials(ticker),
            yf_client.get_price_history(ticker, period="1y"),
            return_exceptions=True,
        )

        # Treat exceptions as empty dicts
        macro = macro if isinstance(macro, dict) else {}
        quote = quote if isinstance(quote, dict) else {}
        profile = profile if isinstance(profile, dict) else {}
        financials = financials if isinstance(financials, dict) else {}
        recommendations = recommendations if isinstance(recommendations, dict) else {}
        stock_info = stock_info if isinstance(stock_info, dict) else {}
        yf_financials = yf_financials if isinstance(yf_financials, dict) else {}
        price_hist = price_hist if isinstance(price_hist, dict) else {}

        result = {}

        # --- Price & Quote ---
        if quote.get("current_price"):
            result["current_price"] = f"${quote['current_price']:.2f}"
            chg = quote.get("change", 0)
            chg_pct = quote.get("change_pct", 0)
            result["daily_change"] = f"{chg:+.2f} ({chg_pct:+.2f}%)"

        # --- Company Profile ---
        if profile.get("company_name"):
            result["company"] = profile["company_name"]
        if profile.get("sector"):
            result["sector"] = profile["sector"]
        if profile.get("market_cap_millions"):
            result["market_cap"] = _fmt_large_number(profile["market_cap_millions"] * 1e6)

        # --- Key Metrics (prefer Finnhub, fallback to yfinance) ---
        pe = financials.get("pe_ratio_ttm") or stock_info.get("pe_ratio_trailing")
        if pe is not None:
            result["pe_ratio"] = f"{pe:.1f}"

        eps = financials.get("eps_ttm") or stock_info.get("eps_trailing")
        if eps is not None:
            result["eps_ttm"] = f"${eps:.2f}"

        beta = financials.get("beta") or stock_info.get("beta")
        if beta is not None:
            result["beta"] = f"{beta:.2f}"

        div_yield = financials.get("dividend_yield_ttm") or stock_info.get("dividend_yield")
        if div_yield is not None:
            # yfinance returns dividendYield inconsistently;
            # if > 0.2, treat as already a percentage (not a ratio)
            if div_yield > 0.2:
                result["dividend_yield"] = _fmt_pct_raw(div_yield)
            else:
                result["dividend_yield"] = _fmt_pct(div_yield)

        # --- 52-Week Range ---
        hi = financials.get("52w_high") or price_hist.get("period_high")
        lo = financials.get("52w_low") or price_hist.get("period_low")
        if hi is not None and lo is not None:
            result["52w_range"] = f"${lo:.2f} - ${hi:.2f}"

        # --- Margins ---
        gm = financials.get("gross_margin_ttm") or stock_info.get("gross_margin")
        if gm is not None:
            result["gross_margin"] = _fmt_pct(gm)

        om = financials.get("operating_margin_ttm") or stock_info.get("operating_margin")
        if om is not None:
            result["operating_margin"] = _fmt_pct(om)

        nm = financials.get("net_margin_ttm") or stock_info.get("profit_margin")
        if nm is not None:
            result["net_margin"] = _fmt_pct(nm)

        # --- Profitability ---
        roe = financials.get("roe_ttm") or stock_info.get("return_on_equity")
        if roe is not None:
            result["roe"] = _fmt_pct(roe)

        dte = financials.get("debt_to_equity") or stock_info.get("debt_to_equity")
        if dte is not None:
            result["debt_to_equity"] = f"{dte:.2f}" if dte < 100 else f"{dte:.0f}"

        cr = financials.get("current_ratio") or stock_info.get("current_ratio")
        if cr is not None:
            result["current_ratio"] = f"{cr:.2f}"

        # --- Growth ---
        rg = stock_info.get("revenue_growth")
        if rg is not None:
            result["revenue_growth_yoy"] = _fmt_pct(rg)

        eg = stock_info.get("earnings_growth")
        if eg is not None:
            result["earnings_growth_yoy"] = _fmt_pct(eg)

        # --- Valuation ---
        ev_ebitda = stock_info.get("ev_to_ebitda")
        if ev_ebitda is not None:
            result["ev_to_ebitda"] = f"{ev_ebitda:.1f}"

        pb = financials.get("pb_ratio") or stock_info.get("price_to_book")
        if pb is not None:
            result["price_to_book"] = f"{pb:.2f}"

        peg = stock_info.get("peg_ratio")
        if peg is not None:
            result["peg_ratio"] = f"{peg:.2f}"

        # --- Revenue & Income ---
        rev = stock_info.get("total_revenue")
        if rev is not None:
            result["revenue_ttm"] = _fmt_large_number(rev)

        fcf = stock_info.get("free_cash_flow")
        if fcf is not None:
            result["free_cash_flow"] = _fmt_large_number(fcf)

        # --- 1Y Price Performance ---
        if price_hist.get("period_return") is not None:
            result["1y_return"] = _fmt_pct(price_hist["period_return"])

        # --- Analyst Recommendations ---
        if recommendations.get("analyst_consensus"):
            buy = recommendations.get("analyst_buy", 0)
            hold = recommendations.get("analyst_hold", 0)
            sell = recommendations.get("analyst_sell", 0)
            result["analyst_consensus"] = (
                f"{recommendations['analyst_consensus']} ({buy} buy, {hold} hold, {sell} sell)"
            )

        # --- Financial Statements (from yfinance) ---
        if yf_financials.get("income_period"):
            result["financial_period"] = yf_financials["income_period"]
        stmt_keys = [
            "income_total_revenue", "income_net_income",
            "income_operating_income", "income_ebitda",
        ]
        for key in stmt_keys:
            if yf_financials.get(key) is not None:
                label = key.replace("income_", "stmt_")
                result[label] = _fmt_large_number(yf_financials[key])

        # --- Macroeconomic Context (from FRED) ---
        for key, value in macro.items():
            result[f"macro_{key}"] = value

        sources = []
        if quote:
            sources.append("Finnhub")
        if stock_info:
            sources.append("yfinance")
        if macro:
            sources.append("FRED")
        result["_data_sources"] = ", ".join(sources) if sources else "none"

        src = result.get("_data_sources", "none")
        logger.info(f"[MarketData] Fetched {len(result)} data points for {ticker} from {src}")
        return result

    finally:
        await asyncio.gather(
            fred.close(),
            finnhub.close(),
            yf_client.close(),
            return_exceptions=True,
        )
