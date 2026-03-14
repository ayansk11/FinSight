"""Tests for PageIndex parser, navigator, and text extractor."""

from __future__ import annotations

from pathlib import Path

import pytest
from finsight_core.models.document import PageIndexTree, TreeNode
from finsight_core.pageindex.navigator import TreeNavigator
from finsight_core.pageindex.parser import load_tree, load_tree_from_dict, tree_to_json
from finsight_core.pageindex.text_extractor import (
    extract_node_text,
    get_node_text_with_context,
)

# ============================================================
# Parser tests
# ============================================================


class TestParser:
    def test_load_tree_from_file(self, sample_tree_path: Path) -> None:
        tree = load_tree(sample_tree_path)
        assert tree.doc_name == "AAPL_10K_2024"
        assert len(tree.structure) > 0

    def test_load_tree_from_dict(self, sample_tree_dict: dict) -> None:
        tree = load_tree_from_dict(sample_tree_dict)
        assert tree.doc_name == "AAPL_10K_2024"

    def test_load_tree_missing_doc_name(self) -> None:
        data = {
            "structure": [
                {
                    "title": "Test",
                    "node_id": "0001",
                    "start_index": 1,
                    "end_index": 5,
                    "nodes": [],
                }
            ]
        }
        tree = load_tree_from_dict(data)
        assert tree.doc_name == "unknown"

    def test_load_tree_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_tree("/nonexistent/path.json")

    def test_tree_to_json_roundtrip(self, tmp_path: Path, sample_tree_dict: dict) -> None:
        tree = load_tree_from_dict(sample_tree_dict)
        out_path = tmp_path / "output.json"
        tree_to_json(tree, out_path)

        # Reload and verify
        reloaded = load_tree(out_path)
        assert reloaded.doc_name == tree.doc_name
        assert reloaded.total_nodes == tree.total_nodes


# ============================================================
# Navigator tests
# ============================================================


class TestNavigator:
    @pytest.fixture
    def navigator(self, sample_tree_dict: dict) -> TreeNavigator:
        tree = PageIndexTree.model_validate(sample_tree_dict)
        return TreeNavigator(tree)

    def test_get_by_id(self, navigator: TreeNavigator) -> None:
        node = navigator.get_by_id("0007")
        assert node is not None
        assert "Risk Factors" in node.title

    def test_get_by_id_not_found(self, navigator: TreeNavigator) -> None:
        assert navigator.get_by_id("9999") is None

    def test_get_path(self, navigator: TreeNavigator) -> None:
        path = navigator.get_path("0009")
        assert len(path) >= 2
        assert "Supply Chain Risks" in path[-1]

    def test_search_by_keywords_title_match(self, navigator: TreeNavigator) -> None:
        results = navigator.search_by_keywords(["Risk Factors"])
        assert len(results) > 0
        # The exact title match should have the highest score
        top = results[0]
        assert "Risk Factors" in top.node.title

    def test_search_by_keywords_partial_match(self, navigator: TreeNavigator) -> None:
        results = navigator.search_by_keywords(["supply chain"])
        assert len(results) > 0
        assert any("Supply Chain" in r.node.title for r in results)

    def test_search_by_keywords_summary_match(self, navigator: TreeNavigator) -> None:
        results = navigator.search_by_keywords(["App Store"])
        assert len(results) > 0
        # Should match via summary even if not in title
        assert any("Services" in r.node.title for r in results)

    def test_search_by_keywords_no_match(self, navigator: TreeNavigator) -> None:
        results = navigator.search_by_keywords(["quantum computing"])
        assert len(results) == 0

    def test_search_by_keywords_max_results(self, navigator: TreeNavigator) -> None:
        results = navigator.search_by_keywords(["risk"], max_results=3)
        assert len(results) <= 3

    def test_search_by_page(self, navigator: TreeNavigator) -> None:
        results = navigator.search_by_page(20)
        assert len(results) > 0
        # All results should include page 20 in their range
        for r in results:
            assert r.node.start_index <= 20 <= r.node.end_index

    def test_search_by_page_range(self, navigator: TreeNavigator) -> None:
        results = navigator.search_by_page_range(13, 28)
        assert len(results) > 0
        # Should find the Risk Factors section
        assert any("Risk Factors" in r.node.title for r in results)

    def test_get_section_by_path(self, navigator: TreeNavigator) -> None:
        node = navigator.get_section_by_path(["Part I", "Item 1A"])
        assert node is not None
        assert "Risk Factors" in node.title

    def test_get_section_by_path_not_found(self, navigator: TreeNavigator) -> None:
        node = navigator.get_section_by_path(["Part IV", "Nonexistent"])
        assert node is None

    def test_get_context_for_node(self, navigator: TreeNavigator) -> None:
        ctx = navigator.get_context_for_node("0009")
        assert ctx.get("title") == "Supply Chain Risks"
        assert "path" in ctx
        assert "siblings" in ctx
        assert len(ctx["siblings"]) >= 1  # Has at least one sibling

    def test_get_context_for_missing_node(self, navigator: TreeNavigator) -> None:
        ctx = navigator.get_context_for_node("9999")
        assert "error" in ctx

    def test_get_nodes_for_llm_selection(self, navigator: TreeNavigator) -> None:
        text = navigator.get_nodes_for_llm_selection(max_depth=2)
        assert "AAPL_10K_2024" in text
        assert "[0001]" in text


# ============================================================
# Text Extractor tests
# ============================================================


class TestTextExtractor:
    def test_extract_node_text_from_pages(self, page_texts: dict[int, str]) -> None:
        node = TreeNode(
            title="Products",
            node_id="0004",
            start_index=3,
            end_index=6,
            nodes=[],
        )
        text = extract_node_text(node, page_texts)
        assert "iPhone" in text
        assert "Page 3" in text

    def test_extract_node_text_with_embedded_text(self) -> None:
        node = TreeNode(
            title="Products",
            node_id="0004",
            start_index=3,
            end_index=6,
            text="This is embedded text from PageIndex",
            nodes=[],
        )
        text = extract_node_text(node, {})
        assert text == "This is embedded text from PageIndex"

    def test_extract_node_text_max_tokens(self, page_texts: dict[int, str]) -> None:
        node = TreeNode(
            title="Risk Factors",
            node_id="0007",
            start_index=13,
            end_index=28,
            nodes=[],
        )
        text = extract_node_text(node, page_texts, max_tokens=50)
        # Should be truncated (50 tokens ≈ 200 chars)
        assert len(text) < 1000

    def test_extract_node_text_embedded_truncation(self) -> None:
        long_text = "x" * 5000
        node = TreeNode(
            title="Long",
            node_id="0001",
            start_index=1,
            end_index=1,
            text=long_text,
            nodes=[],
        )
        text = extract_node_text(node, {}, max_tokens=100)
        assert "[truncated]" in text
        assert len(text) < len(long_text)

    def test_get_node_text_with_context(
        self, sample_tree_dict: dict, page_texts: dict[int, str]
    ) -> None:
        tree = PageIndexTree.model_validate(sample_tree_dict)
        node = tree.find_by_id("0009")  # Supply Chain Risks (pages 18-21)
        assert node is not None

        text = get_node_text_with_context(node, tree, page_texts, context_pages=1)
        # Should include the node's own pages
        assert "Supply Chain" in text

    def test_extract_empty_page_texts(self) -> None:
        node = TreeNode(
            title="Empty",
            node_id="0001",
            start_index=1,
            end_index=5,
            nodes=[],
        )
        text = extract_node_text(node, {})
        assert text == ""
