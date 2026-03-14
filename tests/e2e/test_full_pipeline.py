"""End-to-end tests for the full FinSight LangGraph workflow.

These tests run the real workflow with a real Ollama LLM.
They are skipped automatically if Ollama is not running.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.e2e.conftest import requires_ollama


@pytest.mark.e2e
@requires_ollama
class TestFullPipeline:
    """Full workflow tests with real Ollama inference."""

    def _run(self, workflow, state):
        return asyncio.get_event_loop().run_until_complete(workflow.ainvoke(state))

    def test_risk_query(self, workflow, base_state):
        """Risk-focused query routes correctly and produces findings."""
        base_state["query"] = "Identify all risk factors and governance concerns"
        result = self._run(workflow, base_state)

        assert result["route"] == "risk_focused"
        assert "risk" in result["agents_to_run"]
        assert len(result["findings"]) > 0
        assert len(result["risk_scores"]) > 0

        # Risk agent should have produced findings
        risk_findings = [f for f in result["findings"] if f.get("agent") == "risk"]
        assert len(risk_findings) > 0

    def test_quant_query(self, workflow, base_state):
        """Quant-focused query routes and produces quant findings."""
        base_state["query"] = "What is Apple's revenue and P/E ratio?"
        result = self._run(workflow, base_state)

        assert result["route"] == "quantitative"
        assert "quant" in result["agents_to_run"]
        assert len(result["findings"]) > 0

    def test_doc_intel_query(self, workflow, base_state):
        """Doc intel query produces findings with citations."""
        base_state["query"] = "What is Apple's business description?"
        result = self._run(workflow, base_state)

        assert len(result["findings"]) > 0
        doc_findings = [f for f in result["findings"] if f.get("agent") == "doc_intel"]
        assert len(doc_findings) > 0

    def test_pipeline_produces_report(self, workflow, base_state):
        """Any query produces a non-empty report and executive summary."""
        base_state["query"] = "Summarize Apple's financial position"
        result = self._run(workflow, base_state)

        assert result.get("report"), "Report should not be empty"
        assert len(result["report"]) > 50
        assert result.get("executive_summary"), "Executive summary should not be empty"

    def test_agent_summaries_populated(self, workflow, base_state):
        """Agent summaries are populated for each agent that ran."""
        base_state["query"] = "Identify all risk factors and governance concerns"
        result = self._run(workflow, base_state)

        summaries = result.get("agent_summaries", [])
        assert len(summaries) >= 2  # At least doc_intel + risk

        agents_with_summaries = {s.get("agent") for s in summaries}
        assert "doc_intel" in agents_with_summaries
