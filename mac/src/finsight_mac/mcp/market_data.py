"""Market data aggregator.

Single entry point that calls FRED, Finnhub, yfinance, Financial Modeling Prep,
Alpha Vantage, and StockData concurrently and returns a unified flat dict
for the Quantitative Analysis Agent.
"""

from __future__ import annotations

import asyncio
import logging

from finsight_mac.mcp.alpha_vantage_client import AlphaVantageClient
from finsight_mac.mcp.finnhub_client import FinnhubClient
from finsight_mac.mcp.fmp_client import FMPClient
from finsight_mac.mcp.fred import FredClient
from finsight_mac.mcp.stockdata_client import StockDataClient
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
    fmp = FMPClient()
    av = AlphaVantageClient()
    stockdata = StockDataClient()

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
            fmp_income,
            fmp_balance,
            fmp_cash_flow,
            fmp_metrics,
            fmp_ratios,
            av_overview,
            av_earnings,
            av_technicals,
            news_sentiment,
        ) = await asyncio.gather(
            fred.get_macro_snapshot(),
            finnhub.get_quote(ticker),
            finnhub.get_company_profile(ticker),
            finnhub.get_basic_financials(ticker),
            finnhub.get_recommendation_trends(ticker),
            yf_client.get_stock_info(ticker),
            yf_client.get_financials(ticker),
            yf_client.get_price_history(ticker, period="1y"),
            fmp.get_income_statement(ticker),
            fmp.get_balance_sheet(ticker),
            fmp.get_cash_flow(ticker),
            fmp.get_key_metrics(ticker),
            fmp.get_financial_ratios(ticker),
            av.get_company_overview(ticker),
            av.get_earnings(ticker),
            av.get_technical_indicators(ticker),
            stockdata.get_stock_news(ticker),
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
        fmp_income = fmp_income if isinstance(fmp_income, dict) else {}
        fmp_balance = fmp_balance if isinstance(fmp_balance, dict) else {}
        fmp_cash_flow = fmp_cash_flow if isinstance(fmp_cash_flow, dict) else {}
        fmp_metrics = fmp_metrics if isinstance(fmp_metrics, dict) else {}
        fmp_ratios = fmp_ratios if isinstance(fmp_ratios, dict) else {}
        av_overview = av_overview if isinstance(av_overview, dict) else {}
        av_earnings = av_earnings if isinstance(av_earnings, dict) else {}
        av_technicals = av_technicals if isinstance(av_technicals, dict) else {}
        news_sentiment = news_sentiment if isinstance(news_sentiment, dict) else {}

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
            "income_total_revenue",
            "income_net_income",
            "income_operating_income",
            "income_ebitda",
        ]
        for key in stmt_keys:
            if yf_financials.get(key) is not None:
                label = key.replace("income_", "stmt_")
                result[label] = _fmt_large_number(yf_financials[key])

        # --- Macroeconomic Context (from FRED) ---
        for key, value in macro.items():
            result[f"macro_{key}"] = value

        # --- FMP: Structured Financial Statements ---
        if fmp_income.get("revenue") is not None:
            result["fmp_revenue"] = _fmt_large_number(fmp_income["revenue"])
            result["fmp_net_income"] = _fmt_large_number(fmp_income.get("net_income"))
            result["fmp_ebitda"] = _fmt_large_number(fmp_income.get("ebitda"))
            if fmp_income.get("eps_diluted") is not None:
                result["fmp_eps"] = f"${fmp_income['eps_diluted']:.2f}"
            if fmp_income.get("period"):
                result["fmp_period"] = fmp_income["period"]

        if fmp_balance.get("total_assets") is not None:
            result["fmp_total_assets"] = _fmt_large_number(fmp_balance["total_assets"])
            result["fmp_total_liabilities"] = _fmt_large_number(
                fmp_balance.get("total_liabilities")
            )
            result["fmp_total_equity"] = _fmt_large_number(fmp_balance.get("total_equity"))
            result["fmp_total_debt"] = _fmt_large_number(fmp_balance.get("total_debt"))
            result["fmp_cash"] = _fmt_large_number(fmp_balance.get("cash_and_equivalents"))
            if fmp_balance.get("net_debt") is not None:
                result["fmp_net_debt"] = _fmt_large_number(fmp_balance["net_debt"])

        if fmp_cash_flow.get("operating_cash_flow") is not None:
            result["fmp_operating_cf"] = _fmt_large_number(fmp_cash_flow["operating_cash_flow"])
            result["fmp_free_cash_flow"] = _fmt_large_number(fmp_cash_flow.get("free_cash_flow"))
            result["fmp_capex"] = _fmt_large_number(fmp_cash_flow.get("capital_expenditure"))
            if fmp_cash_flow.get("dividends_paid") is not None:
                result["fmp_dividends_paid"] = _fmt_large_number(fmp_cash_flow["dividends_paid"])
            if fmp_cash_flow.get("share_repurchases") is not None:
                result["fmp_buybacks"] = _fmt_large_number(fmp_cash_flow["share_repurchases"])

        # FMP key metrics (fill gaps not covered by Finnhub/yfinance)
        if fmp_metrics.get("roic") is not None:
            result["roic"] = _fmt_pct(fmp_metrics["roic"])
        if fmp_metrics.get("earnings_yield") is not None:
            result["earnings_yield"] = _fmt_pct(fmp_metrics["earnings_yield"])
        if fmp_metrics.get("income_quality") is not None:
            result["income_quality"] = f"{fmp_metrics['income_quality']:.2f}"

        # FMP ratios: interest coverage
        if fmp_ratios.get("interest_coverage") is not None:
            result["interest_coverage"] = f"{fmp_ratios['interest_coverage']:.1f}x"

        # FMP detailed ratios (fill gaps)
        if fmp_ratios.get("quick_ratio") is not None:
            result["quick_ratio"] = f"{fmp_ratios['quick_ratio']:.2f}"
        if fmp_ratios.get("cash_ratio") is not None:
            result["cash_ratio"] = f"{fmp_ratios['cash_ratio']:.2f}"
        if fmp_ratios.get("asset_turnover") is not None:
            result["asset_turnover"] = f"{fmp_ratios['asset_turnover']:.2f}"
        if fmp_ratios.get("inventory_turnover") is not None:
            result["inventory_turnover"] = f"{fmp_ratios['inventory_turnover']:.1f}"
        if fmp_ratios.get("effective_tax_rate") is not None:
            result["effective_tax_rate"] = _fmt_pct(fmp_ratios["effective_tax_rate"])

        # --- Alpha Vantage: Technical Indicators & Earnings ---
        if av_technicals.get("rsi_14") is not None:
            result["rsi_14"] = f"{av_technicals['rsi_14']:.1f}"
            result["rsi_signal"] = av_technicals.get("rsi_signal", "")

        if av_overview.get("50d_moving_avg") is not None:
            result["sma_50"] = f"${av_overview['50d_moving_avg']:.2f}"
        if av_overview.get("200d_moving_avg") is not None:
            result["sma_200"] = f"${av_overview['200d_moving_avg']:.2f}"

        if av_overview.get("analyst_target_price") is not None:
            result["analyst_target_price"] = f"${av_overview['analyst_target_price']:.2f}"
        if av_overview.get("forward_pe") is not None:
            result["forward_pe"] = f"{av_overview['forward_pe']:.1f}"
        if av_overview.get("quarterly_revenue_growth_yoy") is not None:
            result["quarterly_revenue_growth"] = _fmt_pct(
                av_overview["quarterly_revenue_growth_yoy"]
            )
        if av_overview.get("quarterly_earnings_growth_yoy") is not None:
            result["quarterly_earnings_growth"] = _fmt_pct(
                av_overview["quarterly_earnings_growth_yoy"]
            )

        # Earnings surprise data
        if av_earnings.get("quarters"):
            result["earnings_beat_rate"] = (
                f"{av_earnings['beat_count']}/{len(av_earnings['quarters'])} quarters"
            )
            if av_earnings.get("avg_surprise_pct") is not None:
                result["avg_earnings_surprise"] = f"{av_earnings['avg_surprise_pct']:+.1f}%"

        # --- StockData: News Sentiment ---
        if news_sentiment.get("overall_sentiment"):
            result["news_sentiment"] = news_sentiment["overall_sentiment"]
            result["news_article_count"] = str(news_sentiment.get("article_count", 0))
            if news_sentiment.get("avg_sentiment") is not None:
                result["news_sentiment_score"] = f"{news_sentiment['avg_sentiment']:+.2f}"
            # Include recent headlines for context
            articles = news_sentiment.get("articles", [])
            headlines = [a["title"] for a in articles[:3] if a.get("title")]
            if headlines:
                result["recent_headlines"] = " | ".join(headlines)

        # --- Data Sources ---
        sources = []
        if quote:
            sources.append("Finnhub")
        if stock_info:
            sources.append("yfinance")
        if macro:
            sources.append("FRED")
        if fmp_income or fmp_metrics:
            sources.append("FMP")
        if av_overview or av_technicals:
            sources.append("AlphaVantage")
        if news_sentiment:
            sources.append("StockData")
        result["_data_sources"] = ", ".join(sources) if sources else "none"

        src = result.get("_data_sources", "none")
        logger.info(f"[MarketData] Fetched {len(result)} data points for {ticker} from {src}")
        return result

    finally:
        await asyncio.gather(
            fred.close(),
            finnhub.close(),
            yf_client.close(),
            fmp.close(),
            av.close(),
            stockdata.close(),
            return_exceptions=True,
        )
