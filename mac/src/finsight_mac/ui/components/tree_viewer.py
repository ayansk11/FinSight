"""PageIndex tree viewer component for Streamlit.

Renders a collapsible tree view of a PageIndex document structure.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_tree(tree_data: dict[str, Any]) -> None:
    """Render a PageIndex tree as a collapsible Streamlit component.

    Args:
        tree_data: Raw PageIndex tree dict (or PageIndexTree.model_dump()).
    """
    doc_name = tree_data.get("doc_name", "Unknown Document")
    doc_desc = tree_data.get("doc_description", "")
    structure = tree_data.get("structure", [])

    st.markdown(f"**{doc_name}**")
    if doc_desc:
        st.caption(doc_desc)

    total_nodes = _count_nodes(structure)
    st.caption(f"{total_nodes} sections | {len(structure)} top-level")

    for node in structure:
        _render_node(node, depth=0)


def _render_node(node: dict, depth: int) -> None:
    """Recursively render a tree node."""
    indent = "  " * depth
    node_id = node.get("node_id", "?")
    title = node.get("title", "Untitled")
    start = node.get("start_index", "?")
    end = node.get("end_index", "?")
    summary = node.get("summary", "")
    children = node.get("nodes", [])

    # Format the node line
    pages = f"p.{start}-{end}" if start != end else f"p.{start}"
    label = f"`[{node_id}]` **{title}** ({pages})"

    if children:
        with st.expander(f"{indent}{label}", expanded=(depth < 1)):
            if summary:
                st.caption(summary)
            for child in children:
                _render_node(child, depth + 1)
    else:
        st.markdown(f"{indent}📄 {label}")
        if summary:
            st.caption(f"{indent}  {summary[:200]}")


def _count_nodes(nodes: list[dict]) -> int:
    """Count total nodes in a tree structure."""
    count = len(nodes)
    for node in nodes:
        count += _count_nodes(node.get("nodes", []))
    return count
