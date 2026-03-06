"""PageIndex tree navigator — search and traverse hierarchical document trees.

This is the core retrieval interface for FinSight. Instead of vector similarity
search, we navigate the PageIndex tree structure using LLM-guided reasoning
or keyword-based heuristics.

This module is also the primary candidate for porting to Swift for the iPhone app.
Keep logic simple and avoid Python-specific constructs where possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finsight_core.models.document import PageIndexTree, TreeNode


@dataclass
class SearchResult:
    """A node matched by a tree search, with relevance metadata."""

    node: TreeNode
    score: float = 0.0
    match_reason: str = ""
    path: list[str] = field(default_factory=list)  # Path of titles from root


class TreeNavigator:
    """Navigate and search a PageIndex tree structure.

    This class provides multiple search strategies:
    - Keyword search: Find nodes whose titles/summaries match keywords
    - Page range search: Find nodes covering specific pages
    - Section path search: Navigate by section hierarchy (e.g., "Item 1A > Risk Factors")
    - LLM-guided search: (requires external LLM call) Select nodes based on query

    Example:
        >>> nav = TreeNavigator(tree)
        >>> results = nav.search_by_keywords(["supply chain", "risk"])
        >>> for r in results:
        ...     print(f"{r.node.title} (pages {r.node.start_index}-{r.node.end_index})")
    """

    def __init__(self, tree: PageIndexTree) -> None:
        self.tree = tree
        # Build flat index for fast lookups
        self._all_nodes: list[TreeNode] = tree.get_all_nodes()
        self._id_index: dict[str, TreeNode] = {
            n.node_id: n for n in self._all_nodes
        }
        # Build parent path index
        self._paths: dict[str, list[str]] = {}
        for node in tree.structure:
            self._build_paths(node, [])

    def _build_paths(self, node: TreeNode, parent_path: list[str]) -> None:
        """Recursively build title paths for each node."""
        current_path = parent_path + [node.title]
        self._paths[node.node_id] = current_path
        for child in node.nodes:
            self._build_paths(child, current_path)

    def get_by_id(self, node_id: str) -> TreeNode | None:
        """Look up a node by its ID. O(1)."""
        return self._id_index.get(node_id)

    def get_path(self, node_id: str) -> list[str]:
        """Get the title path from root to a given node."""
        return self._paths.get(node_id, [])

    def search_by_keywords(
        self,
        keywords: list[str],
        search_summaries: bool = True,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Find nodes whose titles (and optionally summaries) match keywords.

        Scoring:
        - Title exact match: +3.0 per keyword
        - Title partial match: +1.5 per keyword
        - Summary match: +1.0 per keyword
        - Bonus for leaf nodes (more specific): +0.5

        Args:
            keywords: List of search terms (case-insensitive).
            search_summaries: Whether to also search node summaries.
            max_results: Maximum number of results to return.

        Returns:
            Sorted list of SearchResult, highest score first.
        """
        results: list[SearchResult] = []
        kw_patterns = [re.compile(re.escape(kw), re.IGNORECASE) for kw in keywords]

        for node in self._all_nodes:
            score = 0.0
            reasons: list[str] = []

            title_lower = node.title.lower()
            for i, kw in enumerate(keywords):
                kw_lower = kw.lower()
                if kw_lower == title_lower:
                    score += 3.0
                    reasons.append(f"title exact: '{kw}'")
                elif kw_patterns[i].search(node.title):
                    score += 1.5
                    reasons.append(f"title partial: '{kw}'")

                if search_summaries and node.summary:
                    if kw_patterns[i].search(node.summary):
                        score += 1.0
                        reasons.append(f"summary: '{kw}'")

            if score > 0:
                if node.is_leaf:
                    score += 0.5
                results.append(
                    SearchResult(
                        node=node,
                        score=score,
                        match_reason="; ".join(reasons),
                        path=self.get_path(node.node_id),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    def search_by_page(self, page: int) -> list[SearchResult]:
        """Find all nodes whose page range includes the given page.

        Returns nodes from most specific (deepest) to least specific (shallowest).
        """
        results: list[SearchResult] = []

        for node in self._all_nodes:
            if node.start_index <= page <= node.end_index:
                # Deeper/smaller nodes are more specific
                specificity = 1.0 / max(node.page_count, 1)
                results.append(
                    SearchResult(
                        node=node,
                        score=specificity,
                        match_reason=f"covers page {page}",
                        path=self.get_path(node.node_id),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def search_by_page_range(
        self, start: int, end: int
    ) -> list[SearchResult]:
        """Find nodes overlapping with a page range."""
        results: list[SearchResult] = []

        for node in self._all_nodes:
            # Check for overlap
            overlap_start = max(node.start_index, start)
            overlap_end = min(node.end_index, end)
            if overlap_start <= overlap_end:
                overlap = overlap_end - overlap_start + 1
                total = max(node.page_count, 1)
                coverage = overlap / total
                results.append(
                    SearchResult(
                        node=node,
                        score=coverage,
                        match_reason=f"overlaps pages {start}-{end} ({coverage:.0%})",
                        path=self.get_path(node.node_id),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def get_section_by_path(
        self, section_titles: list[str]
    ) -> TreeNode | None:
        """Navigate to a section by a path of titles.

        Args:
            section_titles: List of section titles from root to target.
                Example: ["Part I", "Item 1A", "Risk Factors"]

        Returns:
            The matching TreeNode, or None if the path doesn't match.
        """
        current_nodes = self.tree.structure

        for title in section_titles:
            title_lower = title.lower().strip()
            match = None
            for node in current_nodes:
                if node.title.lower().strip() == title_lower:
                    match = node
                    break
                # Fuzzy: check if the title contains the search term
                if title_lower in node.title.lower():
                    match = node
                    break

            if match is None:
                return None
            current_nodes = match.nodes

        return match

    def get_context_for_node(
        self, node_id: str, include_siblings: bool = True
    ) -> dict:
        """Get contextual information for a node, useful for LLM prompts.

        Returns:
            Dict with node details, path, siblings, and parent info.
        """
        node = self.get_by_id(node_id)
        if node is None:
            return {"error": f"Node {node_id} not found"}

        path = self.get_path(node_id)
        siblings: list[dict] = []

        if include_siblings:
            # Find siblings by looking at parent's children
            parent_node = self._find_parent(node_id)
            if parent_node:
                for sibling in parent_node.nodes:
                    if sibling.node_id != node_id:
                        siblings.append(
                            {
                                "node_id": sibling.node_id,
                                "title": sibling.title,
                                "pages": f"{sibling.start_index}-{sibling.end_index}",
                            }
                        )

        return {
            "node_id": node.node_id,
            "title": node.title,
            "pages": f"{node.start_index}-{node.end_index}",
            "page_count": node.page_count,
            "summary": node.summary,
            "path": " > ".join(path),
            "is_leaf": node.is_leaf,
            "children_count": len(node.nodes),
            "siblings": siblings,
        }

    def _find_parent(self, node_id: str) -> TreeNode | None:
        """Find the parent of a node by searching the tree."""
        for root_node in self.tree.structure:
            parent = self._find_parent_recursive(root_node, node_id)
            if parent:
                return parent
        return None

    def _find_parent_recursive(
        self, current: TreeNode, target_id: str
    ) -> TreeNode | None:
        """Recursively search for the parent of a node."""
        for child in current.nodes:
            if child.node_id == target_id:
                return current
            found = self._find_parent_recursive(child, target_id)
            if found:
                return found
        return None

    def get_nodes_for_llm_selection(self, max_depth: int = 2) -> str:
        """Format the tree for LLM-based node selection.

        Produces a compact text representation that an LLM can use to
        decide which nodes are relevant to a query. This is the core
        of PageIndex's reasoning-based retrieval.

        Returns:
            Formatted string with node IDs, titles, page ranges, and summaries.
        """
        return self.tree.get_tree_outline(max_depth=max_depth)
