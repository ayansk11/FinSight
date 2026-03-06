#!/usr/bin/env python3
"""CLI tool to generate .finsight export bundles for the iPhone companion app.

Creates a JSON bundle containing the PageIndex tree, analysis findings,
risk scores, and metadata that the iPhone app can import via AirDrop or Files.

Usage:
    python scripts/export_for_iphone.py --tree trees/aapl_10k_structure.json
    python scripts/export_for_iphone.py --tree trees/aapl_10k_structure.json --findings analysis/aapl_findings.json
    python scripts/export_for_iphone.py --all  # Export all available trees
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mac" / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate .finsight export bundles for the iPhone companion app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --tree data/trees/aapl_10k_structure.json
  %(prog)s --tree data/trees/aapl_10k_structure.json --output ~/Desktop/
  %(prog)s --all --output ~/Desktop/FinSight_Exports/
""",
    )
    parser.add_argument(
        "--tree",
        type=str,
        help="Path to a PageIndex tree JSON file.",
    )
    parser.add_argument(
        "--findings",
        type=str,
        default=None,
        help="Path to analysis findings JSON (optional).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (defaults to data/exports/).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all available trees.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5:9b",
        help="Model name to record in the bundle metadata.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from finsight_core.models.analysis import Finding, RiskScore
    from finsight_core.models.document import Filing, FilingType, PageIndexTree
    from finsight_core.models.export import ExportBundle
    from finsight_core.pageindex.parser import load_tree
    from finsight_mac.config import get_settings

    settings = get_settings()

    # Determine output directory
    output_dir = Path(args.output) if args.output else settings.data_dir / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        # Export all available trees
        from finsight_mac.document.pipeline import DocumentPipeline

        pipeline = DocumentPipeline()
        tree_names = pipeline.list_available_trees()

        if not tree_names:
            print("No trees found to export.")
            sys.exit(1)

        print(f"Exporting {len(tree_names)} trees...")
        for name in tree_names:
            tree = pipeline.load_tree(name)
            if tree:
                _export_tree(tree, name, output_dir, args.model)
            else:
                print(f"  Skipping {name} (failed to load)")

        print(f"\nAll bundles saved to: {output_dir}")
        return

    if not args.tree:
        parser.error("--tree is required (or use --all)")

    tree_path = Path(args.tree)
    if not tree_path.exists():
        # Try looking in the trees directory
        tree_path = settings.trees_dir / args.tree
        if not tree_path.exists():
            print(f"Error: Tree file not found: {args.tree}")
            sys.exit(1)

    tree = load_tree(tree_path)
    doc_name = tree_path.stem.replace("_structure", "")

    # Load findings if provided
    findings: list[Finding] = []
    risk_scores: list[RiskScore] = []

    if args.findings:
        findings_path = Path(args.findings)
        if findings_path.exists():
            try:
                data = json.loads(findings_path.read_text())
                if isinstance(data, dict):
                    findings = [
                        Finding.model_validate(f)
                        for f in data.get("findings", [])
                    ]
                    risk_scores = [
                        RiskScore.model_validate(r)
                        for r in data.get("risk_scores", [])
                    ]
                elif isinstance(data, list):
                    findings = [Finding.model_validate(f) for f in data]
                print(f"Loaded {len(findings)} findings, {len(risk_scores)} risk scores")
            except Exception as e:
                print(f"Warning: Failed to load findings: {e}")

    _export_tree(
        tree,
        doc_name,
        output_dir,
        args.model,
        findings=findings,
        risk_scores=risk_scores,
    )


def _export_tree(
    tree: "PageIndexTree",
    doc_name: str,
    output_dir: Path,
    model_name: str,
    findings: list | None = None,
    risk_scores: list | None = None,
    executive_summary: str | None = None,
    full_report: str | None = None,
) -> None:
    """Create and save an export bundle for a single tree."""
    from finsight_core.models.document import Filing, FilingType
    from finsight_core.models.export import ExportBundle

    # Create a basic filing record from the tree name
    parts = doc_name.upper().split("_")
    ticker = parts[0] if parts else "UNKNOWN"

    filing_type = FilingType.OTHER
    for part in parts:
        if part in ("10K", "10-K"):
            filing_type = FilingType.FORM_10K
        elif part in ("10Q", "10-Q"):
            filing_type = FilingType.FORM_10Q
        elif part in ("8K", "8-K"):
            filing_type = FilingType.FORM_8K

    filing = Filing(
        ticker=ticker,
        company_name=ticker,
        filing_type=filing_type,
        filing_date=date.today(),
    )

    bundle = ExportBundle(
        model_used=model_name,
        filing=filing,
        tree=tree,
        findings=findings or [],
        risk_scores=risk_scores or [],
        executive_summary=executive_summary,
        full_report=full_report,
    )

    output_path = output_dir / f"{doc_name}.finsight"
    bundle.to_json_file(str(output_path))

    size_kb = output_path.stat().st_size / 1024
    print(f"  Exported: {doc_name} ({size_kb:.1f} KB) → {output_path}")


if __name__ == "__main__":
    main()
