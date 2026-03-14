"""Tests for financial ratio calculations."""

from __future__ import annotations

import pytest
from finsight_core.finance.ratios import (
    FinancialRatios,
    compute_leverage,
    compute_liquidity,
    compute_profitability,
    compute_valuation,
    safe_divide,
)


class TestSafeDivide:
    def test_normal_division(self) -> None:
        assert safe_divide(10, 2) == 5.0

    def test_division_by_zero(self) -> None:
        assert safe_divide(10, 0) is None

    def test_zero_numerator(self) -> None:
        assert safe_divide(0, 5) == 0.0

    def test_negative_values(self) -> None:
        result = safe_divide(-10, 2)
        assert result == -5.0


class TestComputeProfitability:
    def test_basic_profitability(self) -> None:
        result = compute_profitability(
            revenue=100,
            cost_of_goods=60,
            operating_income=25,
            net_income=20,
            total_equity=50,
            total_assets=200,
        )
        assert result["gross_margin"] == pytest.approx(0.4)
        assert result["operating_margin"] == pytest.approx(0.25)
        assert result["net_margin"] == pytest.approx(0.2)
        assert result["roe"] == pytest.approx(0.4)
        assert result["roa"] == pytest.approx(0.1)

    def test_zero_revenue(self) -> None:
        result = compute_profitability(
            revenue=0,
            cost_of_goods=0,
            operating_income=0,
            net_income=0,
            total_equity=50,
            total_assets=100,
        )
        assert result["gross_margin"] is None
        assert result["net_margin"] is None


class TestComputeLiquidity:
    def test_basic_liquidity(self) -> None:
        result = compute_liquidity(
            current_assets=200,
            current_liabilities=100,
            cash=50,
            inventory=30,
        )
        assert result["current_ratio"] == pytest.approx(2.0)
        assert result["quick_ratio"] == pytest.approx(1.7)
        assert result["cash_ratio"] == pytest.approx(0.5)

    def test_zero_liabilities(self) -> None:
        result = compute_liquidity(
            current_assets=200,
            current_liabilities=0,
            cash=50,
        )
        assert result["current_ratio"] is None
        assert result["cash_ratio"] is None


class TestComputeLeverage:
    def test_basic_leverage(self) -> None:
        result = compute_leverage(
            total_debt=100,
            total_equity=200,
            total_assets=300,
            ebit=50,
            interest_expense=10,
        )
        assert result["debt_to_equity"] == pytest.approx(0.5)
        assert result["debt_to_assets"] == pytest.approx(1 / 3)
        assert result["interest_coverage"] == pytest.approx(5.0)

    def test_zero_interest(self) -> None:
        result = compute_leverage(
            total_debt=100,
            total_equity=200,
            total_assets=300,
            ebit=50,
            interest_expense=0,
        )
        assert result["interest_coverage"] is None


class TestComputeValuation:
    def test_basic_valuation(self) -> None:
        result = compute_valuation(
            market_cap=1000,
            net_income=50,
            book_value=200,
            revenue=500,
            enterprise_value=1200,
            ebitda=80,
        )
        assert result["pe_ratio"] == pytest.approx(20.0)
        assert result["price_to_book"] == pytest.approx(5.0)
        assert result["price_to_sales"] == pytest.approx(2.0)
        assert result["ev_to_ebitda"] == pytest.approx(15.0)

    def test_negative_earnings(self) -> None:
        result = compute_valuation(
            market_cap=1000,
            net_income=-50,
            book_value=200,
            revenue=500,
            enterprise_value=1200,
            ebitda=80,
        )
        # Negative P/E is mathematically valid
        assert result["pe_ratio"] == pytest.approx(-20.0)


class TestFinancialRatios:
    def test_to_dict_excludes_none(self) -> None:
        ratios = FinancialRatios(pe_ratio=20.0, gross_margin=0.4)
        d = ratios.to_dict()
        assert "pe_ratio" in d
        assert "gross_margin" in d
        assert "debt_to_equity" not in d

    def test_summary_formatting(self) -> None:
        ratios = FinancialRatios(
            pe_ratio=20.0,
            gross_margin=0.4,
            net_margin=0.2,
            current_ratio=1.5,
            roe=0.15,
        )
        summary = ratios.summary()
        assert "P/E Ratio" in summary
        assert "20.00" in summary
        assert "Gross Margin" in summary
        assert "40.0%" in summary
        assert "ROE" in summary

    def test_empty_summary(self) -> None:
        ratios = FinancialRatios()
        assert "No financial ratios computed" in ratios.summary()
