"""Quantitative Analysis Agent.

Performs numerical financial analysis using data from MCP servers
(Finnhub, yfinance, FRED) and extracted financial statement data.
Computes ratios, identifies trends, and compares against benchmarks.
"""

from __future__ import annotations

import logging
import time

from finsight_core.models.analysis import (
    AgentOutput,
    ConfidenceLevel,
    Finding,
)
from finsight_core.prompts.quant import SYSTEM_PROMPT

from finsight_mac.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


async def run_quant_agent(
    query: str,
    ticker: str,
    doc_intel_findings: list[dict] | None = None,
    market_data: dict | None = None,
    llm: OllamaClient | None = None,
) -> AgentOutput:
    """Execute the Quantitative Analysis Agent.

    Args:
        query: User's analysis question.
        ticker: Company ticker symbol.
        doc_intel_findings: Findings from the Document Intelligence Agent.
        market_data: Pre-fetched market data from MCP servers.
        llm: Optional OllamaClient.

    Returns:
        AgentOutput with quantitative findings.
    """
    start = time.time()
    client = llm or OllamaClient()

    # Build context from available data
    context_parts = [f"Ticker: {ticker}", f"Query: {query}"]

    if doc_intel_findings:
        context_parts.append("\n## Document Intelligence Findings:")
        for f in doc_intel_findings[:10]:  # Limit context
            context_parts.append(f"- {f.get('content', '')}")

    if market_data:
        context_parts.append("\n## Available Market Data:")
        for key, value in market_data.items():
            context_parts.append(f"- {key}: {value}")

    context = "\n".join(context_parts)

    prompt = (
        f"Perform quantitative analysis for the following:\n\n"
        f"{context}\n\n"
        f"Compute relevant financial ratios, identify trends, "
        f"and compare against sector benchmarks where possible. "
        f"Respond with structured JSON per your system prompt."
    )

    try:
        response = await client.achat_json(prompt, system_message=SYSTEM_PROMPT)

        findings = []
        # Extract metrics as findings
        for metric_name, metric_data in response.get("metrics", {}).items():
            if isinstance(metric_data, dict):
                content = (
                    f"{metric_name}: {metric_data.get('value', 'N/A')} "
                    f"({metric_data.get('interpretation', '')})"
                )
            else:
                content = f"{metric_name}: {metric_data}"

            findings.append(
                Finding(
                    content=content,
                    agent="quant",
                    confidence=ConfidenceLevel.MEDIUM,
                    metadata={"metric_name": metric_name, "raw_data": metric_data},
                )
            )

        # Extract trends as findings
        for trend in response.get("trends", []):
            findings.append(
                Finding(
                    content=(
                        f"Trend: {trend.get('metric', '')} is {trend.get('direction', 'stable')}. "
                        f"{trend.get('interpretation', '')}"
                    ),
                    agent="quant",
                    confidence=ConfidenceLevel.MEDIUM,
                    metadata={"trend_data": trend},
                )
            )

        elapsed = time.time() - start
        return AgentOutput(
            agent_name="quant",
            findings=findings,
            summary=response.get("summary", f"Quantitative analysis for {ticker}"),
            duration_seconds=elapsed,
        )

    except Exception as e:
        logger.error(f"[Quant] Analysis failed: {e}")
        return AgentOutput(
            agent_name="quant",
            error=str(e),
            summary=f"Quantitative analysis failed: {e}",
            duration_seconds=time.time() - start,
        )
