"""Document and PageIndex tree models.

These models represent SEC filings and their PageIndex hierarchical tree structures.
The TreeNode model mirrors the output of VectifyAI/PageIndex exactly.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class FilingType(StrEnum):
    """SEC filing types supported by FinSight."""

    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    FORM_DEF14A = "DEF 14A"
    EARNINGS_CALL = "earnings_call"
    OTHER = "other"


class Filing(BaseModel):
    """Represents a single SEC filing or financial document."""

    ticker: str = Field(description="Company ticker symbol (e.g., AAPL)")
    company_name: str = Field(description="Full company name")
    filing_type: FilingType = Field(description="Type of SEC filing")
    filing_date: date = Field(description="Date the filing was submitted")
    fiscal_year: int | None = Field(default=None, description="Fiscal year covered")
    fiscal_quarter: int | None = Field(
        default=None, description="Fiscal quarter (1-4) if applicable"
    )
    pdf_path: str | None = Field(default=None, description="Local path to the PDF file")
    tree_path: str | None = Field(default=None, description="Local path to the PageIndex JSON tree")
    accession_number: str | None = Field(default=None, description="SEC EDGAR accession number")
    url: str | None = Field(default=None, description="URL to the filing on SEC EDGAR")

    @property
    def display_name(self) -> str:
        """Human-readable name for this filing."""
        q = f" Q{self.fiscal_quarter}" if self.fiscal_quarter else ""
        fy = f" FY{self.fiscal_year}" if self.fiscal_year else ""
        return f"{self.ticker} {self.filing_type.value}{fy}{q}"


class TreeNode(BaseModel):
    """A single node in a PageIndex hierarchical tree.

    Mirrors the output structure of VectifyAI/PageIndex exactly:
    {
        "title": "Risk Factors",
        "node_id": "0003",
        "start_index": 15,
        "end_index": 28,
        "summary": "...",
        "text": "...",
        "nodes": [...]
    }
    """

    title: str = Field(description="Section heading from the document")
    node_id: str = Field(description="Sequential 4-digit ID assigned via DFS (e.g., '0003')")
    start_index: int = Field(description="First physical page number (1-indexed, inclusive)")
    end_index: int = Field(description="Last physical page number (1-indexed, inclusive)")
    summary: str | None = Field(default=None, description="LLM-generated summary of this section")
    text: str | None = Field(default=None, description="Raw page text for this section's pages")
    nodes: list[TreeNode] = Field(default_factory=list, description="Child sections (recursive)")

    @property
    def page_count(self) -> int:
        """Number of pages in this section."""
        return self.end_index - self.start_index + 1

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children."""
        return len(self.nodes) == 0

    def all_nodes_flat(self) -> list[TreeNode]:
        """Return all nodes in this subtree as a flat list (DFS order)."""
        result = [self]
        for child in self.nodes:
            result.extend(child.all_nodes_flat())
        return result

    def find_by_id(self, node_id: str) -> TreeNode | None:
        """Find a node by its ID in this subtree."""
        if self.node_id == node_id:
            return self
        for child in self.nodes:
            found = child.find_by_id(node_id)
            if found:
                return found
        return None


class PageIndexTree(BaseModel):
    """Complete PageIndex tree structure for a document.

    This is the top-level output of `pageindex.page_index()`.
    """

    doc_name: str = Field(description="Source document filename")
    doc_description: str | None = Field(
        default=None, description="One-sentence document description"
    )
    structure: list[TreeNode] = Field(description="Top-level sections of the document tree")

    @property
    def total_nodes(self) -> int:
        """Total number of nodes in the tree."""
        return sum(len(node.all_nodes_flat()) for node in self.structure)

    @property
    def leaf_nodes(self) -> list[TreeNode]:
        """All leaf nodes in the tree."""
        leaves = []
        for node in self.structure:
            for n in node.all_nodes_flat():
                if n.is_leaf:
                    leaves.append(n)
        return leaves

    def find_by_id(self, node_id: str) -> TreeNode | None:
        """Find any node in the tree by its ID."""
        for node in self.structure:
            found = node.find_by_id(node_id)
            if found:
                return found
        return None

    def get_all_nodes(self) -> list[TreeNode]:
        """Return all nodes as a flat list in DFS order."""
        result = []
        for node in self.structure:
            result.extend(node.all_nodes_flat())
        return result

    def get_tree_outline(self, max_depth: int = 3, _depth: int = 0) -> str:
        """Generate a text outline of the tree for LLM context.

        Args:
            max_depth: Maximum nesting depth to include.
            _depth: Current depth (internal use).

        Returns:
            Formatted text outline with node IDs and page ranges.
        """
        lines = []
        if _depth == 0:
            lines.append(f"Document: {self.doc_name}")
            if self.doc_description:
                lines.append(f"Description: {self.doc_description}")
            lines.append("")

        for node in self.structure if _depth == 0 else []:
            lines.extend(self._outline_node(node, depth=0, max_depth=max_depth))
        return "\n".join(lines)

    @staticmethod
    def _outline_node(node: TreeNode, depth: int, max_depth: int) -> list[str]:
        """Recursively format a node for the outline."""
        indent = "  " * depth
        page_info = f"[pages {node.start_index}-{node.end_index}]"
        summary_part = f" — {node.summary}" if node.summary else ""
        line = f"{indent}[{node.node_id}] {node.title} {page_info}{summary_part}"
        lines = [line]

        if depth < max_depth:
            for child in node.nodes:
                lines.extend(PageIndexTree._outline_node(child, depth + 1, max_depth))
        elif node.nodes:
            lines.append(f"{indent}  ... ({len(node.nodes)} subsections)")

        return lines
