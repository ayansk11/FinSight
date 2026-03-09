#!/usr/bin/env python3
"""Convert a raw PageIndex Cloud API response to FinSight's PageIndexTree format.

Usage:
    python scripts/convert_api_tree.py data/trees/AAPL_10K_2024_raw_api.json
    python scripts/convert_api_tree.py data/trees/AAPL_10K_2024_raw_api.json --name AAPL_10K_2024
    python scripts/convert_api_tree.py data/trees/AAPL_10K_2024_raw_api.json --max-page 108
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mac" / "src"))


def normalize_nodes(nodes: list, max_page: int = 0) -> list:
    """Convert API nodes to our TreeNode format.

    Maps page_index → start_index, computes end_index from siblings,
    and maps prefix_summary → summary.
    """
    result = []
    for i, node in enumerate(nodes):
        children = node.get("nodes", node.get("children", []))
        page_idx = node.get("page_index", 0)

        # Compute end_index from next sibling
        if i + 1 < len(nodes):
            next_page = nodes[i + 1].get("page_index", page_idx)
            end_idx = max(next_page - 1, page_idx)
        elif max_page > 0:
            end_idx = max_page
        else:
            end_idx = page_idx

        normalized_children = normalize_nodes(children, max_page=end_idx)

        # Refine end_index from children
        if normalized_children:
            child_max = max(c.get("end_index", 0) for c in normalized_children)
            if child_max > 0:
                end_idx = max(end_idx, child_max)

        normalized = {
            "title": node.get("title", ""),
            "node_id": node.get("node_id", ""),
            "start_index": node.get("start_index", page_idx),
            "end_index": node.get("end_index", end_idx),
            "summary": node.get("summary", node.get("prefix_summary", "")),
            "text": node.get("text"),
            "nodes": normalized_children,
        }
        result.append(normalized)
    return result


def convert(raw_path: Path, doc_name: str, max_page: int) -> dict:
    """Convert raw API JSON to PageIndexTree dict."""
    with open(raw_path) as f:
        api_response = json.load(f)

    result = api_response.get("result", [])

    # result is a list of root nodes
    if isinstance(result, list):
        api_nodes = result
    elif isinstance(result, dict):
        api_nodes = result.get("structure") or result.get("nodes", [])
    else:
        api_nodes = []

    normalized = normalize_nodes(api_nodes, max_page=max_page)

    # Use root node's prefix_summary as doc description
    doc_description = ""
    if api_nodes and isinstance(api_nodes[0], dict):
        doc_description = api_nodes[0].get("prefix_summary", "")

    return {
        "doc_name": doc_name,
        "doc_description": doc_description,
        "structure": normalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw PageIndex API response to FinSight tree format.",
    )
    parser.add_argument("input", help="Path to raw API JSON file")
    parser.add_argument("--name", default=None, help="Document name")
    parser.add_argument("--max-page", type=int, default=108, help="Total pages in PDF")
    parser.add_argument("--output", default=None, help="Output path (default: auto)")

    args = parser.parse_args()

    raw_path = Path(args.input)
    if not raw_path.exists():
        print(f"Error: File not found: {raw_path}")
        sys.exit(1)

    doc_name = args.name or raw_path.stem.replace("_raw_api", "")

    # Convert
    tree_dict = convert(raw_path, doc_name, args.max_page)

    # Validate with our model
    from finsight_core.pageindex.parser import load_tree_from_dict, tree_to_json

    tree = load_tree_from_dict(tree_dict)

    # Save
    output = args.output or str(
        raw_path.parent / f"{doc_name}_structure.json"
    )
    tree_to_json(tree, output)

    print(f"Converted: {raw_path.name} → {Path(output).name}")
    print(f"  Document: {tree.doc_name}")
    print(f"  Description: {(tree.doc_description or 'None')[:80]}")
    print(f"  Total nodes: {tree.total_nodes}")
    print(f"  Leaf nodes: {len(tree.leaf_nodes)}")

    # Count nodes with text
    text_nodes = sum(1 for n in tree.get_all_nodes() if n.text)
    print(f"  Nodes with text: {text_nodes}")

    print("\nTree outline:")
    print(tree.get_tree_outline(max_depth=2))


if __name__ == "__main__":
    main()
