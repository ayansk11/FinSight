"""RLM (Recursive Language Model) client for cross-document reasoning.

Integrates with the official `rlm` library to enable reasoning across
documents that exceed the LLM's context window. Falls back to a
simplified custom implementation if the library is unavailable.

RLM is triggered ONLY for cross-document queries (multi-year, multi-company).
Single-document queries use direct PageIndex tree navigation instead.
"""

from __future__ import annotations

import logging

from finsight_core.prompts.rlm_system import RLM_SYSTEM_PROMPT, build_rlm_context

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)


class RLMClient:
    """Client for RLM-powered cross-document reasoning.

    Uses the official RLM library when available, with a fallback
    to a simplified multi-step analysis approach.

    Example:
        >>> client = RLMClient()
        >>> result = await client.analyze_cross_document(
        ...     query="How has Apple's supply chain risk changed over 3 years?",
        ...     filing_names=["AAPL_10K_2022", "AAPL_10K_2023", "AAPL_10K_2024"],
        ...     tree_outlines={"AAPL_10K_2022": "...", ...},
        ...     page_texts={"AAPL_10K_2022": {1: "text...", ...}, ...},
        ... )
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.ollama_model
        self.base_url = settings.ollama_openai_url
        self.max_iterations = settings.max_rlm_iterations
        self._rlm = None
        self._init_rlm()

    def _init_rlm(self) -> None:
        """Try to initialize the official RLM library."""
        try:
            from rlm import RLM

            self._rlm = RLM(
                backend="openai",
                backend_kwargs={
                    "model_name": self.model,
                    "base_url": self.base_url,
                    "api_key": "ollama",
                },
                max_iterations=self.max_iterations,
                verbose=True,
            )
            logger.info("RLM library initialized successfully")
        except ImportError:
            logger.warning(
                "RLM library not installed. Using fallback multi-step analysis. "
                "Install with: pip install rlm"
            )
        except Exception as e:
            logger.warning(f"RLM initialization failed: {e}. Using fallback.")

    async def analyze_cross_document(
        self,
        query: str,
        filing_names: list[str],
        tree_outlines: dict[str, str],
        page_texts: dict[str, dict[int, str]],
    ) -> RLMResult:
        """Run cross-document analysis using RLM.

        Args:
            query: User's analysis question.
            filing_names: List of filing identifiers.
            tree_outlines: Map of filing name → tree outline text.
            page_texts: Map of filing name → {page_num: text}.

        Returns:
            RLMResult with the analysis and execution trace.
        """
        context = build_rlm_context(query, filing_names, tree_outlines)

        if self._rlm is not None:
            return await self._analyze_with_rlm(query, context)
        else:
            return await self._analyze_fallback(query, filing_names, tree_outlines, page_texts)

    async def _analyze_with_rlm(self, query: str, context: str) -> RLMResult:
        """Use the official RLM library for analysis."""
        try:
            prompt = f"{query}\n\nContext:\n{context}"
            result = self._rlm.completion(prompt)
            return RLMResult(
                answer=result,
                method="rlm_library",
                iterations=0,  # Library doesn't expose this
                success=True,
            )
        except Exception as e:
            logger.error(f"RLM library execution failed: {e}")
            return RLMResult(
                answer="",
                method="rlm_library",
                error=str(e),
                success=False,
            )

    async def _analyze_fallback(
        self,
        query: str,
        filing_names: list[str],
        tree_outlines: dict[str, str],
        page_texts: dict[str, dict[int, str]],
    ) -> RLMResult:
        """Fallback: multi-step analysis without the RLM library.

        Implements a simplified version of the RLM pattern:
        1. Analyze each filing's relevant sections independently
        2. Synthesize across filings
        """
        from finsight_mac.llm.ollama_client import OllamaClient

        client = OllamaClient()
        individual_analyses: dict[str, str] = {}
        iterations = 0

        # Step 1: Analyze each filing independently
        for name in filing_names:
            outline = tree_outlines.get(name, "")
            prompt = (
                f"Analyze the following document structure for information relevant to: {query}\n\n"
                f"Document: {name}\n"
                f"Structure:\n{outline}\n\n"
                f"Provide key findings with section references."
            )
            response = await client.achat(prompt, system_message=RLM_SYSTEM_PROMPT)
            individual_analyses[name] = response
            iterations += 1

        # Step 2: Synthesize across filings
        synthesis_prompt = f"Query: {query}\n\nIndividual filing analyses:\n"
        for name, analysis in individual_analyses.items():
            synthesis_prompt += f"\n=== {name} ===\n{analysis}\n"

        synthesis_prompt += (
            "\nSynthesize these analyses into a comprehensive answer. "
            "Compare across filings, identify trends, and note contradictions."
        )
        final = await client.achat(synthesis_prompt, system_message=RLM_SYSTEM_PROMPT)
        iterations += 1

        return RLMResult(
            answer=final,
            method="fallback_multi_step",
            iterations=iterations,
            individual_analyses=individual_analyses,
            success=True,
        )


class RLMResult:
    """Result of an RLM analysis."""

    def __init__(
        self,
        answer: str,
        method: str,
        iterations: int = 0,
        individual_analyses: dict[str, str] | None = None,
        error: str | None = None,
        success: bool = True,
    ) -> None:
        self.answer = answer
        self.method = method
        self.iterations = iterations
        self.individual_analyses = individual_analyses or {}
        self.error = error
        self.success = success
