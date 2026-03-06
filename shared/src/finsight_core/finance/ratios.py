"""Financial ratio calculations.

Pure functions for computing standard financial ratios from
extracted financial statement data. No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FinancialRatios:
    """Computed financial ratios for a company/period."""

    # Valuation
    pe_ratio: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None
    ev_to_ebitda: float | None = None

    # Profitability
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    roe: float | None = None  # Return on equity
    roa: float | None = None  # Return on assets
    roic: float | None = None  # Return on invested capital

    # Liquidity
    current_ratio: float | None = None
    quick_ratio: float | None = None
    cash_ratio: float | None = None

    # Leverage
    debt_to_equity: float | None = None
    debt_to_assets: float | None = None
    interest_coverage: float | None = None

    # Efficiency
    asset_turnover: float | None = None
    inventory_turnover: float | None = None
    receivables_turnover: float | None = None

    # Cash Flow
    fcf_yield: float | None = None
    operating_cash_flow_ratio: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def summary(self) -> str:
        """Generate a text summary of computed ratios."""
        parts = []
        ratios = self.to_dict()
        if not ratios:
            return "No financial ratios computed."

        labels = {
            "pe_ratio": "P/E Ratio",
            "price_to_book": "Price/Book",
            "price_to_sales": "Price/Sales",
            "ev_to_ebitda": "EV/EBITDA",
            "gross_margin": "Gross Margin",
            "operating_margin": "Operating Margin",
            "net_margin": "Net Margin",
            "roe": "ROE",
            "roa": "ROA",
            "roic": "ROIC",
            "current_ratio": "Current Ratio",
            "quick_ratio": "Quick Ratio",
            "debt_to_equity": "Debt/Equity",
            "interest_coverage": "Interest Coverage",
            "fcf_yield": "FCF Yield",
        }

        for key, value in ratios.items():
            label = labels.get(key, key.replace("_", " ").title())
            if "margin" in key or "yield" in key or key in ("roe", "roa", "roic"):
                parts.append(f"  {label}: {value:.1%}")
            else:
                parts.append(f"  {label}: {value:.2f}")

        return "\n".join(parts)


# --- Pure calculation functions ---


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Safely divide, returning None if denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def compute_profitability(
    revenue: float,
    cost_of_goods: float,
    operating_income: float,
    net_income: float,
    total_equity: float,
    total_assets: float,
) -> dict[str, float | None]:
    """Compute profitability ratios."""
    return {
        "gross_margin": safe_divide(revenue - cost_of_goods, revenue),
        "operating_margin": safe_divide(operating_income, revenue),
        "net_margin": safe_divide(net_income, revenue),
        "roe": safe_divide(net_income, total_equity),
        "roa": safe_divide(net_income, total_assets),
    }


def compute_liquidity(
    current_assets: float,
    current_liabilities: float,
    cash: float,
    inventory: float = 0,
) -> dict[str, float | None]:
    """Compute liquidity ratios."""
    return {
        "current_ratio": safe_divide(current_assets, current_liabilities),
        "quick_ratio": safe_divide(
            current_assets - inventory, current_liabilities
        ),
        "cash_ratio": safe_divide(cash, current_liabilities),
    }


def compute_leverage(
    total_debt: float,
    total_equity: float,
    total_assets: float,
    ebit: float,
    interest_expense: float,
) -> dict[str, float | None]:
    """Compute leverage ratios."""
    return {
        "debt_to_equity": safe_divide(total_debt, total_equity),
        "debt_to_assets": safe_divide(total_debt, total_assets),
        "interest_coverage": safe_divide(ebit, interest_expense),
    }


def compute_valuation(
    market_cap: float,
    net_income: float,
    book_value: float,
    revenue: float,
    enterprise_value: float,
    ebitda: float,
) -> dict[str, float | None]:
    """Compute valuation ratios."""
    return {
        "pe_ratio": safe_divide(market_cap, net_income),
        "price_to_book": safe_divide(market_cap, book_value),
        "price_to_sales": safe_divide(market_cap, revenue),
        "ev_to_ebitda": safe_divide(enterprise_value, ebitda),
    }
