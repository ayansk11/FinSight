"""Tests for local Ollama-based PageIndex tree generator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from finsight_mac.document.local_tree_generator import LocalTreeGenerator

# ---------------------------------------------------------------------------
# Heading Extraction Tests
# ---------------------------------------------------------------------------


class TestHeadingExtraction:
    """Tests for regex-based heading candidate detection."""

    def _gen(self) -> LocalTreeGenerator:
        return LocalTreeGenerator(llm=MagicMock())

    def test_detects_part_headings(self):
        gen = self._gen()
        page_texts = {
            1: "Cover page content",
            5: "PART I\nSome business description here.",
            30: "PART II\nFinancial statements follow.",
        }
        candidates = gen._extract_heading_candidates(page_texts)
        titles = [c["title"] for c in candidates]
        assert "PART I" in titles
        assert "PART II" in titles
        assert all(
            c["level"] == 1
            for c in candidates
            if c["title"].startswith("PART")
        )

    def test_detects_item_headings(self):
        gen = self._gen()
        page_texts = {
            5: "Item 1. Business\nApple designs and manufactures...",
            13: "Item 1A. Risk Factors\nThe following discussion...",
            40: "Item 7. Management's Discussion and Analysis of...",
        }
        candidates = gen._extract_heading_candidates(page_texts)
        titles = [c["title"] for c in candidates]
        assert any("Item 1." in t for t in titles)
        assert any("Item 1A." in t for t in titles)
        assert any("Item 7." in t for t in titles)
        assert all(
            c["level"] == 2
            for c in candidates
            if c["title"].startswith("Item")
        )

    def test_detects_signatures(self):
        gen = self._gen()
        page_texts = {100: "SIGNATURES\nPursuant to the requirements..."}
        candidates = gen._extract_heading_candidates(page_texts)
        assert any(c["title"] == "SIGNATURES" for c in candidates)

    def test_detects_exhibit_index(self):
        gen = self._gen()
        page_texts = {105: "EXHIBIT INDEX\n\nExhibit No. Description..."}
        candidates = gen._extract_heading_candidates(page_texts)
        assert any("EXHIBIT INDEX" in c["title"] for c in candidates)

    def test_skips_false_positives(self):
        gen = self._gen()
        page_texts = {
            1: "TABLE OF CONTENTS\nPART I\nItem 1. Business",
            2: "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\nWASHINGTON",
        }
        candidates = gen._extract_heading_candidates(page_texts)
        titles = [c["title"] for c in candidates]
        assert "TABLE OF CONTENTS" not in titles
        assert "UNITED STATES" not in titles

    def test_all_caps_subheading(self):
        gen = self._gen()
        page_texts = {
            13: "MACROECONOMIC AND INDUSTRY RISKS\nThe Company is subject to...",
        }
        candidates = gen._extract_heading_candidates(page_texts)
        assert any(
            c["title"] == "MACROECONOMIC AND INDUSTRY RISKS"
            for c in candidates
        )

    def test_empty_pages_handled(self):
        gen = self._gen()
        page_texts = {1: "", 2: "   \n\n  ", 3: "PART I\nContent"}
        candidates = gen._extract_heading_candidates(page_texts)
        assert len(candidates) == 1
        assert candidates[0]["title"] == "PART I"

    def test_sorted_by_page(self):
        gen = self._gen()
        page_texts = {
            30: "PART II\nFinancial data",
            5: "PART I\nBusiness description",
            60: "PART III\nDirectors info",
        }
        candidates = gen._extract_heading_candidates(page_texts)
        pages = [c["page"] for c in candidates]
        assert pages == sorted(pages)

    def test_deduplicates_titles(self):
        gen = self._gen()
        page_texts = {
            1: "PART I\nFirst occurrence",
            2: "PART I\nSecond occurrence on different page",
        }
        candidates = gen._extract_heading_candidates(page_texts)
        part_i = [c for c in candidates if c["title"] == "PART I"]
        assert len(part_i) == 1


# ---------------------------------------------------------------------------
# Node ID Assignment Tests
# ---------------------------------------------------------------------------


class TestNodeIdAssignment:
    """Tests for DFS-order node ID assignment."""

    def test_sequential_ids(self):
        gen = LocalTreeGenerator(llm=MagicMock())
        nodes = [
            {
                "title": "Root",
                "start_index": 1,
                "end_index": 50,
                "nodes": [
                    {
                        "title": "Child A",
                        "start_index": 1,
                        "end_index": 20,
                        "nodes": [],
                    },
                    {
                        "title": "Child B",
                        "start_index": 21,
                        "end_index": 50,
                        "nodes": [],
                    },
                ],
            }
        ]
        gen._assign_node_ids(nodes)
        assert nodes[0]["node_id"] == "0000"
        assert nodes[0]["nodes"][0]["node_id"] == "0001"
        assert nodes[0]["nodes"][1]["node_id"] == "0002"

    def test_four_digit_padding(self):
        gen = LocalTreeGenerator(llm=MagicMock())
        nodes = [{"title": f"Node {i}", "nodes": []} for i in range(12)]
        gen._assign_node_ids(nodes)
        assert nodes[0]["node_id"] == "0000"
        assert nodes[9]["node_id"] == "0009"
        assert nodes[10]["node_id"] == "0010"
        assert nodes[11]["node_id"] == "0011"

    def test_deep_nesting(self):
        gen = LocalTreeGenerator(llm=MagicMock())
        nodes = [
            {
                "title": "L0",
                "nodes": [
                    {
                        "title": "L1",
                        "nodes": [
                            {"title": "L2", "nodes": []},
                        ],
                    },
                ],
            }
        ]
        gen._assign_node_ids(nodes)
        assert nodes[0]["node_id"] == "0000"
        assert nodes[0]["nodes"][0]["node_id"] == "0001"
        assert nodes[0]["nodes"][0]["nodes"][0]["node_id"] == "0002"


# ---------------------------------------------------------------------------
# End Index Computation Tests
# ---------------------------------------------------------------------------


class TestEndIndexComputation:
    """Tests for end_index computation from sibling boundaries."""

    def test_sibling_boundary(self):
        gen = LocalTreeGenerator(llm=MagicMock())
        nodes = [
            {"title": "A", "start_index": 1, "end_index": 1, "nodes": []},
            {"title": "B", "start_index": 10, "end_index": 10, "nodes": []},
            {"title": "C", "start_index": 20, "end_index": 20, "nodes": []},
        ]
        gen._compute_end_indices(nodes, max_page=50)
        assert nodes[0]["end_index"] == 9  # B starts at 10
        assert nodes[1]["end_index"] == 19  # C starts at 20
        assert nodes[2]["end_index"] == 50  # Last gets max_page

    def test_parent_spans_children(self):
        gen = LocalTreeGenerator(llm=MagicMock())
        nodes = [
            {
                "title": "Parent",
                "start_index": 1,
                "end_index": 1,
                "nodes": [
                    {
                        "title": "Child",
                        "start_index": 5,
                        "end_index": 5,
                        "nodes": [],
                    },
                ],
            }
        ]
        gen._compute_end_indices(nodes, max_page=30)
        # Child gets max_page=30 (last child, parent's end)
        assert nodes[0]["nodes"][0]["end_index"] == 30
        # Parent's end is at least child's end
        assert nodes[0]["end_index"] == 30


# ---------------------------------------------------------------------------
# Default Tree Fallback Tests
# ---------------------------------------------------------------------------


class TestDefaultTreeFallback:
    """Tests for level-based tree construction without LLM."""

    def test_basic_sec_structure(self):
        gen = LocalTreeGenerator(llm=MagicMock())
        candidates = [
            {"title": "PART I", "page": 5, "level": 1},
            {"title": "Item 1. Business", "page": 5, "level": 2},
            {"title": "Item 1A. Risk Factors", "page": 13, "level": 2},
            {"title": "PART II", "page": 30, "level": 1},
            {"title": "Item 7. MD&A", "page": 40, "level": 2},
        ]
        result = gen._candidates_to_default_tree(
            candidates, "TEST_10K", max_page=100
        )

        structure = result["structure"]
        assert len(structure) == 1  # One root node
        root = structure[0]
        assert root["title"] == "TEST_10K"

        # PART I and PART II are children of root
        parts = root["nodes"]
        assert len(parts) == 2
        assert parts[0]["title"] == "PART I"
        assert parts[1]["title"] == "PART II"

        # Items are children of their PARTs
        assert len(parts[0]["nodes"]) == 2  # Item 1, Item 1A
        assert len(parts[1]["nodes"]) == 1  # Item 7

    def test_orphan_items_go_to_root(self):
        gen = LocalTreeGenerator(llm=MagicMock())
        candidates = [
            {"title": "Item 1. Business", "page": 5, "level": 2},
        ]
        result = gen._candidates_to_default_tree(
            candidates, "TEST", max_page=50
        )
        root = result["structure"][0]
        # Item should be a direct child of root (no PART parent)
        assert len(root["nodes"]) == 1
        assert root["nodes"][0]["title"] == "Item 1. Business"


# ---------------------------------------------------------------------------
# Full Pipeline Tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Tests for the full generate() pipeline with mocked Ollama."""

    def test_generate_with_mocked_llm(self):
        """Full pipeline produces a valid PageIndexTree."""
        mock_llm = AsyncMock()
        mock_llm.achat_json = AsyncMock(
            return_value={
                "doc_description": "Test SEC filing",
                "structure": [
                    {
                        "title": "TEST DOCUMENT",
                        "page": 1,
                        "children": [
                            {
                                "title": "PART I",
                                "page": 3,
                                "children": [
                                    {
                                        "title": "Item 1. Business",
                                        "page": 3,
                                        "children": [],
                                    },
                                ],
                            },
                        ],
                    }
                ],
            }
        )
        mock_llm.achat = AsyncMock(return_value="Section summary.")

        gen = LocalTreeGenerator(llm=mock_llm)
        page_texts = {
            1: "Cover page\nANNUAL REPORT",
            2: "Table of contents",
            3: "PART I\nItem 1. Business\nCompany description...",
            4: "More business content here.",
            5: "Additional business details.",
        }

        tree = asyncio.get_event_loop().run_until_complete(
            gen.generate(
                pdf_path="/fake/path.pdf",
                doc_name="TEST_10K",
                add_summaries=True,
                page_texts=page_texts,
            )
        )

        assert tree.doc_name == "TEST_10K"
        assert tree.doc_description == "Test SEC filing"
        assert tree.total_nodes >= 3
        # Node IDs should be sequential
        all_nodes = tree.get_all_nodes()
        ids = [n.node_id for n in all_nodes]
        assert ids[0] == "0000"
        assert ids[1] == "0001"
        assert ids[2] == "0002"

    def test_generate_without_summaries(self):
        """Pipeline works with add_summaries=False."""
        mock_llm = AsyncMock()
        mock_llm.achat_json = AsyncMock(
            return_value={
                "doc_description": "Test doc",
                "structure": [
                    {"title": "Root", "page": 1, "children": []},
                ],
            }
        )

        gen = LocalTreeGenerator(llm=mock_llm)

        tree = asyncio.get_event_loop().run_until_complete(
            gen.generate(
                pdf_path="/fake/path.pdf",
                doc_name="TEST",
                add_summaries=False,
                page_texts={1: "Content"},
            )
        )

        assert tree.total_nodes == 1
        # achat should NOT have been called (no summaries)
        mock_llm.achat.assert_not_called()

    def test_fallback_on_llm_failure(self):
        """Falls back to default tree when Ollama fails."""
        mock_llm = AsyncMock()
        mock_llm.achat_json = AsyncMock(
            side_effect=Exception("Ollama down")
        )
        mock_llm.achat = AsyncMock(side_effect=Exception("Ollama down"))

        gen = LocalTreeGenerator(llm=mock_llm)
        page_texts = {
            1: "Cover page",
            5: "PART I\nItem 1. Business\nDescription...",
            13: "Item 1A. Risk Factors\nRisks include...",
        }

        tree = asyncio.get_event_loop().run_until_complete(
            gen.generate(
                pdf_path="/fake/path.pdf",
                doc_name="FALLBACK_TEST",
                add_summaries=False,
                page_texts=page_texts,
            )
        )

        # Should still produce a valid tree
        assert tree.doc_name == "FALLBACK_TEST"
        assert tree.total_nodes >= 1
        # Should have detected headings via heuristics
        all_nodes = tree.get_all_nodes()
        titles = [n.title for n in all_nodes]
        assert any("PART I" in t for t in titles)


# ---------------------------------------------------------------------------
# Prompt Construction Tests
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Tests for tree generation prompt builders."""

    def test_hierarchy_prompt_format(self):
        from finsight_core.prompts.tree_generation import (
            build_hierarchy_prompt,
        )

        candidates = [
            {"title": "PART I", "page": 5, "level": 1},
            {"title": "Item 1. Business", "page": 5, "level": 2},
        ]
        prompt = build_hierarchy_prompt(candidates, "AAPL_10K_2024")
        assert "AAPL_10K_2024" in prompt
        assert "PART I" in prompt
        assert "Item 1. Business" in prompt
        assert "[PART]" in prompt
        assert "[ITEM]" in prompt

    def test_hierarchy_prompt_with_samples(self):
        from finsight_core.prompts.tree_generation import (
            build_hierarchy_prompt,
        )

        candidates = [
            {"title": "PART I", "page": 5, "level": 1},
        ]
        samples = {5: "This is the beginning of Part I content..."}
        prompt = build_hierarchy_prompt(candidates, "TEST", samples)
        assert "Page 5:" in prompt
        assert "This is the beginning" in prompt

    def test_summary_prompt_format(self):
        from finsight_core.prompts.tree_generation import (
            build_summary_prompt,
        )

        prompt = build_summary_prompt(
            "Item 1A. Risk Factors",
            "The Company is subject to various risks...",
        )
        assert "Item 1A. Risk Factors" in prompt
        assert "ONE sentence" in prompt
        assert "various risks" in prompt
