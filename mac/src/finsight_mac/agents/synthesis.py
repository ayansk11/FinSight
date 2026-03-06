"""Synthesis & Report Agent.

Synthesizes findings from all other agents into a comprehensive
investment-grade analysis report. Uses RLM for cross-document
reasoning when multiple filings are involved.
"""

from __future__ import annotations

import logging
import time

from finsight_core.models.analysis import (
    AgentOutput,
    ConfidenceLevel,
    Finding,
)
from finsight_core.prompts.synthesis import SYSTEM_PROMPT, build_synthesis_prompt

from finsight_mac.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


async def run_synthesis_agent(
    query: str,
    doc_intel_output: AgentOutput | None = None,
    quant_output: AgentOutput | None = None,
    risk_output: AgentOutput | None = None,
    filing_names: list[str] | None = None,
    llm: OllamaClient | None = None,
) -> AgentOutput:
    """Execute the Synthesis & Report Agent.

    Args:
        query: User's analysis question.
        doc_intel_output: Output from Document Intelligence Agent.
        quant_output: Output from Quantitative Analysis Agent.
        risk_output: Output from Risk Classification Agent.
        filing_names: List of filing names analyzed.
        llm: Optional OllamaClient.

    Returns:
        AgentOutput with synthesized report.
    """
    start = time.time()
    client = llm or OllamaClient()
    filing_names = filing_names or ["Unknown filing"]

    # Format agent findings for synthesis
    doc_findings_text = _format_agent_findings(doc_intel_output, "Document Intelligence")
    quant_findings_text = _format_agent_findings(quant_output, "Quantitative Analysis")
    risk_findings_text = _format_agent_findings(risk_output, "Risk Classification")

    prompt = build_synthesis_prompt(
        query=query,
        doc_intel_findings=doc_findings_text,
        quant_findings=quant_findings_text,
        risk_findings=risk_findings_text,
        filing_names=filing_names,
    )

    try:
        report = await client.achat(prompt, system_message=SYSTEM_PROMPT)

        # Extract executive summary (first paragraph or section)
        exec_summary = _extract_executive_summary(report)

        # Aggregate all citations from input agents
        all_findings = []
        for output in [doc_intel_output, quant_output, risk_output]:
            if output and output.findings:
                all_findings.extend(output.findings)

        # Create synthesis finding
        synthesis_finding = Finding(
            content=report,
            agent="synthesis",
            confidence=ConfidenceLevel.MEDIUM,
            metadata={
                "report_sections": _count_sections(report),
                "source_agent_count": sum(
                    1
                    for o in [doc_intel_output, quant_output, risk_output]
                    if o and o.succeeded
                ),
            },
        )

        elapsed = time.time() - start
        return AgentOutput(
            agent_name="synthesis",
            findings=[synthesis_finding],
            summary=exec_summary,
            duration_seconds=elapsed,
        )

    except Exception as e:
        logger.error(f"[Synthesis] Report generation failed: {e}")
        return AgentOutput(
            agent_name="synthesis",
            error=str(e),
            summary=f"Synthesis failed: {e}",
            duration_seconds=time.time() - start,
        )


def _format_agent_findings(
    output: AgentOutput | None, agent_label: str
) -> str:
    """Format an agent's output for synthesis input."""
    if output is None:
        return f"No {agent_label} analysis available."

    if not output.succeeded:
        return f"{agent_label} failed: {output.error}"

    parts = [f"### {agent_label} ({len(output.findings)} findings)"]
    parts.append(f"Summary: {output.summary}")

    for i, finding in enumerate(output.findings[:10], 1):
        citations_text = ""
        if finding.citations:
            refs = [c.short_ref for c in finding.citations]
            citations_text = f" {' '.join(refs)}"
        parts.append(f"{i}. [{finding.confidence.value}] {finding.content}{citations_text}")

    if output.risk_scores:
        parts.append(f"\nRisk Scores ({len(output.risk_scores)} classifications):")
        for rs in output.risk_scores:
            parts.append(
                f"  - [{rs.technique_id}] {rs.technique_name}: "
                f"severity={rs.severity:.1f}, confidence={rs.confidence.value}"
            )

    return "\n".join(parts)


def _extract_executive_summary(report: str) -> str:
    """Extract the executive summary from a generated report."""
    lines = report.split("\n")
    in_summary = False
    summary_lines = []

    for line in lines:
        if "executive summary" in line.lower() or "key takeaway" in line.lower():
            in_summary = True
            continue
        if in_summary:
            if line.strip().startswith("#") or line.strip().startswith("##"):
                break
            if line.strip():
                summary_lines.append(line.strip())

    if summary_lines:
        return " ".join(summary_lines[:5])

    # Fallback: first non-header paragraph
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and len(stripped) > 50:
            return stripped[:500]

    return "Analysis complete."


def _count_sections(report: str) -> int:
    """Count the number of sections in the report."""
    return sum(1 for line in report.split("\n") if line.strip().startswith("#"))
