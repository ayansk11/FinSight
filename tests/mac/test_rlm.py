"""Tests for RLM integration — client, workflow routing, and synthesis."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# RLM Client Tests
# ---------------------------------------------------------------------------


class TestRLMClient:
    """Tests for the RLMClient fallback and library paths."""

    def test_fallback_when_no_rlm_lib(self):
        """RLMClient falls back gracefully when rlm library is missing."""
        with patch("finsight_mac.llm.rlm_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                ollama_model="qwen3:8b",
                ollama_openai_url="http://localhost:11434/v1",
                max_rlm_iterations=15,
            )
            # Patch the rlm import to fail
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "rlm":
                    raise ImportError("No module named 'rlm'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                from finsight_mac.llm.rlm_client import RLMClient

                client = RLMClient()
                assert client._rlm is None

    def test_analyze_cross_document_fallback(self):
        """Fallback multi-step analysis produces a result."""
        with patch("finsight_mac.llm.rlm_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                ollama_model="qwen3:8b",
                ollama_openai_url="http://localhost:11434/v1",
                max_rlm_iterations=15,
            )
            from finsight_mac.llm.rlm_client import RLMClient

            client = RLMClient()
            client._rlm = None  # Force fallback path

            mock_ollama = AsyncMock()
            mock_ollama.achat = AsyncMock(return_value="Mocked analysis")

            target = "finsight_mac.llm.ollama_client.OllamaClient"
            with patch(target, return_value=mock_ollama):
                result = asyncio.get_event_loop().run_until_complete(
                    client.analyze_cross_document(
                        query="Compare risk factors",
                        filing_names=["AAPL_2022", "AAPL_2023"],
                        tree_outlines={
                            "AAPL_2022": "1. Risk Factors",
                            "AAPL_2023": "1. Risk Factors",
                        },
                        page_texts={},
                    )
                )

            assert result.success is True
            assert result.method == "fallback_multi_step"
            # 2 individual analyses + 1 synthesis = 3 iterations
            assert result.iterations == 3
            assert result.answer == "Mocked analysis"


# ---------------------------------------------------------------------------
# RLM Routing Tests
# ---------------------------------------------------------------------------


class TestRLMRouting:
    """Tests for workflow conditional routing with RLM."""

    def test_rlm_route_when_needs_rlm(self):
        """should_use_rlm returns 'rlm_synthesis' when needs_rlm is True."""
        from finsight_mac.graph.workflow import should_use_rlm

        state = {"needs_rlm": True}
        assert should_use_rlm(state) == "rlm_synthesis"

    def test_standard_route_when_no_rlm(self):
        """should_use_rlm returns 'synthesis' when needs_rlm is False."""
        from finsight_mac.graph.workflow import should_use_rlm

        state = {"needs_rlm": False}
        assert should_use_rlm(state) == "synthesis"

    def test_standard_route_when_missing(self):
        """should_use_rlm returns 'synthesis' when needs_rlm is absent."""
        from finsight_mac.graph.workflow import should_use_rlm

        state = {}
        assert should_use_rlm(state) == "synthesis"

    def test_risk_skips_to_rlm_when_needed(self):
        """should_run_risk routes to rlm_synthesis when no risk agent but needs_rlm."""
        from finsight_mac.graph.workflow import should_run_risk

        state = {"agents_to_run": ["doc_intel", "synthesis"], "needs_rlm": True}
        assert should_run_risk(state) == "rlm_synthesis"

    def test_risk_skips_to_synthesis_normally(self):
        """should_run_risk routes to synthesis when no risk agent and no RLM."""
        from finsight_mac.graph.workflow import should_run_risk

        state = {"agents_to_run": ["doc_intel", "synthesis"], "needs_rlm": False}
        assert should_run_risk(state) == "synthesis"


# ---------------------------------------------------------------------------
# RLM Synthesis Node Tests
# ---------------------------------------------------------------------------


class TestRLMSynthesisNode:
    """Tests for the rlm_synthesis_node workflow node."""

    def test_rlm_synthesis_with_mock_client(self):
        """rlm_synthesis_node produces report using RLM client."""
        from finsight_mac.graph.workflow import rlm_synthesis_node

        # Mock tree
        mock_tree = MagicMock()
        mock_tree.doc_name = "AAPL_10K_2023"
        mock_tree.get_tree_outline.return_value = "1. Risk Factors\n2. Business"

        # Mock RLM result
        mock_rlm_result = MagicMock()
        mock_rlm_result.answer = "Cross-doc analysis: risks evolved."
        mock_rlm_result.method = "fallback_multi_step"
        mock_rlm_result.iterations = 3
        mock_rlm_result.success = True

        mock_rlm_client = AsyncMock()
        mock_rlm_client.analyze_cross_document = AsyncMock(
            return_value=mock_rlm_result,
        )

        # Mock synthesis output
        mock_finding = MagicMock()
        mock_finding.content = "Synthesized report"
        mock_finding.model_dump.return_value = {
            "content": "Synthesized report",
            "agent": "synthesis",
        }

        mock_synthesis_output = MagicMock()
        mock_synthesis_output.summary = "Executive summary"
        mock_synthesis_output.findings = [mock_finding]
        mock_synthesis_output.duration_seconds = 1.5

        state = {
            "query": "Compare risk factors across years",
            "trees": [{"doc_name": "AAPL_10K_2023"}],
            "filings": [{"display_name": "Apple 10-K 2023"}],
            "page_texts_by_filing": {},
            "agent_summaries": [],
            "findings": [],
        }

        with (
            patch(
                "finsight_core.pageindex.parser.load_tree_from_dict",
                return_value=mock_tree,
            ),
            patch(
                "finsight_mac.llm.rlm_client.RLMClient",
                return_value=mock_rlm_client,
            ),
            patch(
                "finsight_mac.graph.workflow.run_synthesis_agent",
                new_callable=AsyncMock,
                return_value=mock_synthesis_output,
            ) as mock_synth,
        ):
            result = asyncio.get_event_loop().run_until_complete(rlm_synthesis_node(state))

        assert result["report"] == "Synthesized report"
        assert result["executive_summary"] == "Executive summary"
        assert result["rlm_result"]["success"] is True
        assert result["rlm_result"]["method"] == "fallback_multi_step"

        # Verify RLM analysis was passed to synthesis
        call_kwargs = mock_synth.call_args[1]
        assert call_kwargs["rlm_analysis"] == "Cross-doc analysis: risks evolved."


# ---------------------------------------------------------------------------
# Synthesis Prompt Tests
# ---------------------------------------------------------------------------


class TestSynthesisPromptRLM:
    """Tests for RLM section in synthesis prompt."""

    def test_prompt_includes_rlm_section(self):
        """build_synthesis_prompt includes RLM analysis when provided."""
        from finsight_core.prompts.synthesis import build_synthesis_prompt

        prompt = build_synthesis_prompt(
            query="Compare trends",
            doc_intel_findings="Doc findings here",
            quant_findings="Quant findings here",
            risk_findings="Risk findings here",
            filing_names=["AAPL_2022", "AAPL_2023"],
            rlm_analysis="Cross-document RLM reasoning output.",
        )

        assert "Cross-Document RLM Analysis" in prompt
        assert "Cross-document RLM reasoning output." in prompt
        assert "primary basis for your synthesis" in prompt

    def test_prompt_excludes_rlm_when_none(self):
        """build_synthesis_prompt omits RLM section when not provided."""
        from finsight_core.prompts.synthesis import build_synthesis_prompt

        prompt = build_synthesis_prompt(
            query="Analyze this filing",
            doc_intel_findings="Doc findings here",
            quant_findings="Quant findings here",
            risk_findings="Risk findings here",
            filing_names=["AAPL_2023"],
        )

        assert "Cross-Document RLM Analysis" not in prompt
        assert "primary basis for your synthesis" not in prompt
