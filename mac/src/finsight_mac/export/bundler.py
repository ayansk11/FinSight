"""Export bundler — generates .finsight bundles for iPhone.

A .finsight bundle contains:
- PageIndex tree structure
- Pre-computed analysis findings
- Risk scores
- Executive summary and report
- Filing metadata

The iPhone app imports these bundles for offline viewing and follow-up Q&A.
"""

from __future__ import annotations

import logging
from pathlib import Path

from finsight_core.models.analysis import Finding, RiskScore
from finsight_core.models.document import Filing, PageIndexTree
from finsight_core.models.export import ExportBundle

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)


class BundleExporter:
    """Generates .finsight export bundles for iPhone.

    Example:
        >>> exporter = BundleExporter()
        >>> exporter.create_bundle(
        ...     filing=filing,
        ...     tree=tree,
        ...     findings=findings,
        ...     risk_scores=risk_scores,
        ...     output_path="exports/aapl_2024.finsight",
        ... )
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def create_bundle(
        self,
        filing: Filing,
        tree: PageIndexTree,
        findings: list[Finding] | None = None,
        risk_scores: list[RiskScore] | None = None,
        executive_summary: str | None = None,
        full_report: str | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """Create a .finsight export bundle.

        Args:
            filing: Source filing metadata.
            tree: PageIndex tree (without page text to save size).
            findings: Analysis findings from all agents.
            risk_scores: Risk classifications.
            executive_summary: One-paragraph summary.
            full_report: Full synthesis report.
            output_path: Output file path (.finsight extension).

        Returns:
            Path to the created bundle file.
        """
        # Strip text from tree nodes to reduce bundle size
        tree_for_export = self._strip_node_text(tree)

        bundle = ExportBundle(
            model_used=self.settings.ollama_model,
            filing=filing,
            tree=tree_for_export,
            findings=findings or [],
            risk_scores=risk_scores or [],
            executive_summary=executive_summary,
            full_report=full_report,
        )

        if output_path is None:
            export_dir = self.settings.data_dir / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            output_path = export_dir / f"{filing.display_name.replace(' ', '_')}.finsight"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bundle.to_json_file(str(output_path))

        size_kb = output_path.stat().st_size / 1024
        logger.info(f"Export bundle created: {output_path} ({size_kb:.1f} KB)")

        return output_path

    @staticmethod
    def _strip_node_text(tree: PageIndexTree) -> PageIndexTree:
        """Create a copy of the tree with node text removed (for smaller bundles)."""
        import json

        tree_dict = json.loads(tree.model_dump_json())

        def strip_text(nodes: list) -> list:
            for node in nodes:
                node.pop("text", None)
                if "nodes" in node:
                    strip_text(node["nodes"])
            return nodes

        if "structure" in tree_dict:
            strip_text(tree_dict["structure"])

        from finsight_core.pageindex.parser import load_tree_from_dict

        return load_tree_from_dict(tree_dict)
