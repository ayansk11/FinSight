"""Export bundle model for Mac → iPhone data transfer.

A .finsight bundle is a JSON file containing a PageIndex tree,
pre-computed analysis findings, risk scores, and metadata.
The iPhone app imports these bundles to display analysis results
and enable on-device follow-up Q&A against the tree.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from finsight_core.models.analysis import Finding, RiskScore
from finsight_core.models.document import Filing, PageIndexTree


class ExportBundle(BaseModel):
    """A complete analysis bundle for transfer to the iPhone companion app.

    Contains everything the iPhone needs to display results and enable
    follow-up Q&A: the tree structure, findings, risk scores, and metadata.
    """

    # --- Metadata ---
    version: str = Field(default="1.0", description="Bundle format version")
    created_at: datetime = Field(default_factory=datetime.now)
    model_used: str = Field(description="LLM model that generated this analysis")

    # --- Source document ---
    filing: Filing = Field(description="The analyzed filing")
    tree: PageIndexTree = Field(description="PageIndex tree structure")

    # --- Analysis results ---
    findings: list[Finding] = Field(
        default_factory=list, description="All findings from all agents"
    )
    risk_scores: list[RiskScore] = Field(
        default_factory=list, description="Risk classifications"
    )
    executive_summary: str | None = Field(
        default=None, description="One-paragraph executive summary"
    )
    full_report: str | None = Field(
        default=None, description="Full synthesis report text"
    )

    def to_json_file(self, path: str) -> None:
        """Write this bundle to a .finsight JSON file."""
        from pathlib import Path

        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, path: str) -> ExportBundle:
        """Load a bundle from a .finsight JSON file."""
        from pathlib import Path

        return cls.model_validate_json(Path(path).read_text())
