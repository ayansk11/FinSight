"""Tests for LangGraph workflow and router."""

from __future__ import annotations

from finsight_mac.graph.router import classify_query


class TestRouter:
    def test_single_doc_route(self) -> None:
        result = classify_query("What is the business description in this 10-K?")
        assert result["route"] == "single_doc"
        assert "doc_intel" in result["agents_to_run"]

    def test_quantitative_route(self) -> None:
        result = classify_query("What is Apple's P/E ratio and revenue margin?")
        assert result["route"] == "quantitative"
        assert "quant" in result["agents_to_run"]

    def test_risk_focused_route(self) -> None:
        result = classify_query("Identify all risk factors and governance concerns")
        assert result["route"] == "risk_focused"
        assert "risk" in result["agents_to_run"]

    def test_multi_doc_route(self) -> None:
        result = classify_query(
            "Compare revenue trends across years",
            num_filings=3,
        )
        assert result["route"] == "multi_doc"
        assert result["needs_rlm"] is True

    def test_default_route(self) -> None:
        result = classify_query("Tell me about this company")
        assert result["route"] in ("single_doc", "multi_doc", "quantitative", "risk_focused")
        assert "agents_to_run" in result
        assert len(result["agents_to_run"]) > 0

    def test_synthesis_always_included(self) -> None:
        result = classify_query("What is Apple's P/E ratio?")
        assert "synthesis" in result["agents_to_run"]
