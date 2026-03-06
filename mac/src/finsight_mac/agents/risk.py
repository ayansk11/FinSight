"""Risk Classification Agent.

Classifies financial risks in SEC filings using the FinSight F3-Inspired
Risk Taxonomy (based on MITRE's Fight Financial Fraud framework).
Scans filings for linguistic and structural patterns and assigns
severity scores with confidence levels.
"""

from __future__ import annotations

import logging
import time

from finsight_core.models.analysis import (
    AgentOutput,
    ConfidenceLevel,
    Finding,
    RiskScore,
    RiskTactic,
)
from finsight_core.prompts.risk import SYSTEM_PROMPT
from finsight_core.taxonomy.f3_taxonomy import TECHNIQUE_BY_ID, get_taxonomy_prompt

from finsight_mac.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


async def run_risk_agent(
    query: str,
    doc_intel_findings: list[dict] | None = None,
    filing_display_name: str = "",
    llm: OllamaClient | None = None,
) -> AgentOutput:
    """Execute the Risk Classification Agent.

    Args:
        query: User's analysis question.
        doc_intel_findings: Findings from the Document Intelligence Agent.
        filing_display_name: Filing name for citations.
        llm: Optional OllamaClient.

    Returns:
        AgentOutput with risk classifications.
    """
    start = time.time()
    client = llm or OllamaClient()

    # Build context
    taxonomy_text = get_taxonomy_prompt()
    findings_context = ""
    if doc_intel_findings:
        findings_parts = []
        for f in doc_intel_findings[:15]:
            content = f.get("content", "")
            citations = f.get("citations", [])
            ref = ""
            if citations:
                c = citations[0]
                ref = f" [{c.get('section_title', '')}, p.{c.get('page_start', '')}]"
            findings_parts.append(f"- {content}{ref}")
        findings_context = "\n".join(findings_parts)

    prompt = (
        f"## Risk Taxonomy\n{taxonomy_text}\n\n"
        f"## Document Findings\n{findings_context}\n\n"
        f"## Query: {query}\n\n"
        f"Classify risks found in the document findings using the F3-Inspired taxonomy above. "
        f"For each detected risk, provide the technique_id, severity (0.0-1.0), confidence, "
        f"and supporting evidence. Respond with structured JSON per your system prompt."
    )

    try:
        response = await client.achat_json(prompt, system_message=SYSTEM_PROMPT)

        risk_scores: list[RiskScore] = []
        findings: list[Finding] = []

        for rs_data in response.get("risk_scores", []):
            tech_id = rs_data.get("technique_id", "")
            technique = TECHNIQUE_BY_ID.get(tech_id)

            try:
                tactic = RiskTactic(rs_data.get("tactic", "disclosure_risk"))
            except ValueError:
                tactic = RiskTactic.DISCLOSURE_RISK

            try:
                confidence = ConfidenceLevel(rs_data.get("confidence", "medium"))
            except ValueError:
                confidence = ConfidenceLevel.MEDIUM

            risk_score = RiskScore(
                tactic=tactic,
                technique_id=tech_id,
                technique_name=rs_data.get(
                    "technique_name", technique.name if technique else tech_id
                ),
                severity=min(max(float(rs_data.get("severity", 0.5)), 0.0), 1.0),
                confidence=confidence,
                evidence=rs_data.get("evidence", ""),
            )
            risk_scores.append(risk_score)

            # Also add as a finding
            findings.append(
                Finding(
                    content=f"[{tech_id}] {risk_score.technique_name}: {risk_score.evidence}",
                    agent="risk",
                    confidence=confidence,
                    metadata={
                        "technique_id": tech_id,
                        "tactic": tactic.value,
                        "severity": risk_score.severity,
                    },
                )
            )

        elapsed = time.time() - start

        return AgentOutput(
            agent_name="risk",
            findings=findings,
            risk_scores=risk_scores,
            summary=response.get("summary", f"Risk classification for {filing_display_name}"),
            duration_seconds=elapsed,
        )

    except Exception as e:
        logger.error(f"[Risk] Classification failed: {e}")
        return AgentOutput(
            agent_name="risk",
            error=str(e),
            summary=f"Risk classification failed: {e}",
            duration_seconds=time.time() - start,
        )
