"""Live local tree generator tests.

Tests heading extraction against the real AAPL PDF and (if Ollama
is running) full tree generation with real LLM inference.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.e2e.conftest import requires_ollama


@pytest.mark.e2e
class TestHeadingExtractionRealPDF:
    """Extract headings from the real AAPL 10-K PDF."""

    def test_heading_extraction(self, aapl_pdf_path):
        """Detect SEC headings from the real AAPL PDF."""
        from finsight_core.pageindex.text_extractor import (
            extract_pages_from_pdf,
        )
        from finsight_mac.document.local_tree_generator import (
            LocalTreeGenerator,
        )

        page_texts = extract_pages_from_pdf(str(aapl_pdf_path))
        assert len(page_texts) > 10, "AAPL 10-K should have many pages"

        gen = LocalTreeGenerator(llm=None)
        candidates = gen._extract_heading_candidates(page_texts)

        assert len(candidates) >= 5, f"Expected at least 5 headings, got {len(candidates)}"

        titles = [c["title"] for c in candidates]

        # Should detect standard SEC parts
        has_part_1 = any("PART I" in t for t in titles)
        has_part_2 = any("PART II" in t for t in titles)
        assert has_part_1, f"Should detect PART I. Found: {titles[:10]}"
        assert has_part_2, f"Should detect PART II. Found: {titles[:10]}"

        # Should detect items
        has_items = any("Item" in t for t in titles)
        assert has_items, f"Should detect Item headings. Found: {titles[:10]}"

        # Pages should be in order
        pages = [c["page"] for c in candidates]
        assert pages == sorted(pages), "Candidates should be sorted by page"


@pytest.mark.e2e
@requires_ollama
class TestLocalTreeGenLive:
    """Full tree generation with real Ollama inference."""

    def test_generate_tree_from_real_pdf(self, aapl_pdf_path):
        """Generate a full tree from the AAPL 10-K with Ollama."""
        from finsight_mac.document.local_tree_generator import (
            LocalTreeGenerator,
        )

        gen = LocalTreeGenerator()  # Uses real OllamaClient
        tree = asyncio.get_event_loop().run_until_complete(
            gen.generate(
                pdf_path=str(aapl_pdf_path),
                doc_name="AAPL_10K_2024",
                add_summaries=False,  # Skip summaries to keep test fast
            )
        )

        assert tree.doc_name == "AAPL_10K_2024"
        assert tree.total_nodes >= 5, f"Should have at least 5 nodes, got {tree.total_nodes}"

        # All nodes should have valid IDs
        all_nodes = tree.get_all_nodes()
        ids = [n.node_id for n in all_nodes]
        assert ids[0] == "0000", "First node should be 0000"
        assert len(set(ids)) == len(ids), "Node IDs should be unique"

        # Start/end indices should be valid
        for node in all_nodes:
            assert node.start_index >= 1
            assert node.end_index >= node.start_index
