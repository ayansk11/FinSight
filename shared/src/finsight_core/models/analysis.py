"""Analysis output models — findings, citations, risk scores, and agent outputs.

These models are produced by FinSight agents and consumed by the Synthesis agent
and the Streamlit UI. They are also serialized into .finsight export bundles
for the iPhone companion app.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ConfidenceLevel(StrEnum):
    """Confidence levels for findings and risk assessments."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Citation(BaseModel):
    """A precise reference to a location in a source document."""

    filing_display_name: str = Field(
        description="Human-readable filing identifier (e.g., 'AAPL 10-K FY2024')"
    )
    node_id: str = Field(description="PageIndex tree node ID")
    section_title: str = Field(description="Section heading from the tree")
    page_start: int = Field(description="Starting page number")
    page_end: int = Field(description="Ending page number")
    excerpt: str | None = Field(
        default=None,
        description="Short text excerpt supporting the finding (max 500 chars)",
    )

    @property
    def short_ref(self) -> str:
        """Compact citation string for inline use."""
        return f"[{self.filing_display_name}, p.{self.page_start}]"


class Finding(BaseModel):
    """A single analytical finding produced by an agent."""

    content: str = Field(description="The finding text")
    agent: str = Field(
        description="Which agent produced this (doc_intel, quant, risk, synthesis)"
    )
    confidence: ConfidenceLevel = Field(description="Confidence in this finding")
    citations: list[Citation] = Field(
        default_factory=list, description="Supporting source citations"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Agent-specific metadata (ratios, taxonomy IDs, etc.)",
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class RiskTactic(StrEnum):
    """Top-level MITRE F3-inspired risk tactic categories."""

    DISCLOSURE_RISK = "disclosure_risk"
    FINANCIAL_HEALTH = "financial_health"
    GOVERNANCE = "governance"
    MARKET_RISK = "market_risk"


class RiskScore(BaseModel):
    """A risk classification score from the Risk Classification Agent."""

    tactic: RiskTactic = Field(description="F3 tactic category")
    technique_id: str = Field(
        description="F3-style technique ID (e.g., 'DR-T001')"
    )
    technique_name: str = Field(
        description="Human-readable technique name (e.g., 'Material omission in risk factors')"
    )
    severity: float = Field(
        ge=0.0, le=1.0, description="Severity score from 0.0 to 1.0"
    )
    confidence: ConfidenceLevel = Field(description="Confidence in this assessment")
    evidence: str = Field(description="Brief description of the evidence found")
    citations: list[Citation] = Field(
        default_factory=list, description="Where the risk pattern was detected"
    )


class AgentOutput(BaseModel):
    """Standard output format for all FinSight agents.

    Every agent (doc_intel, quant, risk, synthesis) returns this structure.
    The LangGraph workflow merges these into the shared state.
    """

    agent_name: str = Field(description="Agent identifier")
    findings: list[Finding] = Field(
        default_factory=list, description="Analytical findings"
    )
    risk_scores: list[RiskScore] = Field(
        default_factory=list, description="Risk classifications (risk agent only)"
    )
    summary: str = Field(
        default="", description="Brief summary of this agent's analysis"
    )
    error: str | None = Field(
        default=None, description="Error message if the agent failed"
    )
    duration_seconds: float = Field(
        default=0.0, description="Time taken for this agent's execution"
    )

    @property
    def succeeded(self) -> bool:
        """Whether the agent completed without errors."""
        return self.error is None

    @property
    def citation_count(self) -> int:
        """Total citations across all findings."""
        return sum(len(f.citations) for f in self.findings)
