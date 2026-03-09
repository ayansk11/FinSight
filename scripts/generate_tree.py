#!/usr/bin/env python3
"""CLI tool to generate a PageIndex tree from a SEC filing PDF.

Usage:
    python scripts/generate_tree.py --pdf path/to/10k.pdf
    python scripts/generate_tree.py --pdf path/to/10k.pdf --name AAPL_10K_2024
    python scripts/generate_tree.py --pdf path/to/10k.pdf --model gpt-4o --no-summaries
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mac" / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PageIndex tree from a SEC filing PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --pdf data/filings/aapl_10k_2024.pdf
  %(prog)s --pdf data/filings/aapl_10k_2024.pdf --name AAPL_10K_2024
  %(prog)s --pdf data/filings/aapl_10k_2024.pdf --model gpt-4o-mini --no-summaries
  %(prog)s --list  # List all generated trees
""",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to the SEC filing PDF file.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Document name (defaults to PDF filename without extension).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="PageIndex model to use (default: gpt-4o, set in config).",
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Skip generating node summaries (faster).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available pre-generated trees.",
    )
    parser.add_argument(
        "--info",
        type=str,
        default=None,
        help="Show info about a specific tree by document name.",
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

    from finsight_mac.document.pipeline import DocumentPipeline

    pipeline = DocumentPipeline()

    # --- List mode ---
    if args.list:
        trees = pipeline.list_available_trees()
        if not trees:
            print("No pre-generated trees found.")
            print(f"Trees directory: {pipeline.settings.trees_dir}")
            return

        print(f"Available trees ({len(trees)}):")
        for name in sorted(trees):
            tree = pipeline.load_tree(name)
            if tree:
                node_count = tree.total_nodes
                leaf_count = len(tree.leaf_nodes)
                desc = tree.doc_description or "No description"
                print(f"  {name:40s} {node_count:4d} nodes ({leaf_count} leaves)")
                print(f"    {desc[:80]}")
            else:
                print(f"  {name:40s} (failed to load)")
        return

    # --- Info mode ---
    if args.info:
        tree = pipeline.load_tree(args.info)
        if not tree:
            print(f"Tree not found: {args.info}")
            sys.exit(1)

        print(f"Document: {tree.doc_name}")
        print(f"Description: {tree.doc_description or 'None'}")
        print(f"Total nodes: {tree.total_nodes}")
        print(f"Leaf nodes: {len(tree.leaf_nodes)}")
        print()
        print("Tree outline:")
        print(tree.get_tree_outline(max_depth=4))
        return

    # --- Generate mode ---
    if not args.pdf:
        parser.error("--pdf is required for tree generation (or use --list / --info)")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"Generating PageIndex tree for: {pdf_path}")
    print(f"Model: {args.model or 'default (gpt-4o)'}")
    print(f"Summaries: {'no' if args.no_summaries else 'yes'}")
    print()

    try:
        tree = pipeline.generate_tree(
            pdf_path=pdf_path,
            doc_name=args.name,
            model=args.model,
            add_summaries=not args.no_summaries,
        )

        print()
        print("Tree generated successfully!")
        print(f"  Document: {tree.doc_name}")
        print(f"  Description: {tree.doc_description or 'None'}")
        print(f"  Total nodes: {tree.total_nodes}")
        print(f"  Leaf nodes: {len(tree.leaf_nodes)}")
        out_name = f"{args.name or pdf_path.stem}_structure.json"
        print(f"  Saved to: {pipeline.settings.trees_dir / out_name}")

    except ImportError as e:
        print(f"Error: {e}")
        print("Install pageindex: pip install pageindex")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating tree: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
