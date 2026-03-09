"""Document processing pipeline — PDF → PageIndex tree.

Wraps VectifyAI/PageIndex Cloud API and local Ollama-based generation
to create hierarchical tree structures from SEC filing PDFs.
Also handles loading pre-generated trees and managing the document store.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from finsight_core.models.document import PageIndexTree
from finsight_core.pageindex.parser import load_tree, load_tree_from_dict, tree_to_json

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Manages document ingestion, tree generation, and storage.

    Example:
        >>> pipeline = DocumentPipeline()
        >>> tree = pipeline.generate_tree("path/to/10k.pdf")
        >>> tree = pipeline.load_tree("AAPL_10K_2024")
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.settings.ensure_dirs()

    def generate_tree_cloud(
        self,
        pdf_path: str | Path,
        doc_name: str | None = None,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
        add_node_text: bool = True,
    ) -> PageIndexTree:
        """Generate a PageIndex tree using the Cloud API.

        Uses the PageIndex Cloud API (api.pageindex.ai) to process a PDF
        and generate a hierarchical tree structure.

        Args:
            pdf_path: Path to the PDF file.
            doc_name: Optional name for the document (defaults to filename).
            poll_interval: Seconds between status polls.
            max_wait: Maximum seconds to wait for processing.
            add_node_text: Request embedded text in tree nodes.

        Returns:
            Generated and validated PageIndexTree.

        Raises:
            ValueError: If PAGEINDEX_API_KEY is not configured.
            TimeoutError: If processing exceeds max_wait.
        """
        if not self.settings.pageindex_api_key:
            raise ValueError(
                "PAGEINDEX_API_KEY is required for cloud tree generation. "
                "Get one at https://pageindex.ai (free tier: 200 pages)"
            )

        try:
            from pageindex import PageIndexClient
        except ImportError:
            raise ImportError(
                "pageindex is required for cloud tree generation. "
                "Install with: pip install pageindex"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_name = doc_name or pdf_path.stem
        client = PageIndexClient(api_key=self.settings.pageindex_api_key)

        # Step 1: Submit document
        logger.info(f"Submitting {pdf_path.name} to PageIndex Cloud API...")
        submit_result = client.submit_document(str(pdf_path))
        doc_id = submit_result["doc_id"]
        logger.info(f"Document submitted. doc_id={doc_id}")

        # Step 2: Poll for tree completion
        logger.info("Waiting for tree generation...")
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(
                    f"PageIndex tree generation timed out after {max_wait}s"
                )

            try:
                tree_result = client.get_tree(doc_id, node_summary=True)
                status = tree_result.get("status", "")

                if status == "completed":
                    logger.info(
                        f"Tree generation completed in {elapsed:.1f}s"
                    )
                    break
                elif status == "failed":
                    error = tree_result.get("error", "Unknown error")
                    raise RuntimeError(
                        f"PageIndex tree generation failed: {error}"
                    )
                else:
                    logger.debug(
                        f"Status: {status} ({elapsed:.0f}s elapsed)"
                    )
            except Exception as e:
                if "failed" in str(e).lower():
                    raise
                logger.debug(f"Polling... ({elapsed:.0f}s elapsed)")

            time.sleep(poll_interval)

        # Step 3: Get page count from PDF for end_index computation
        max_page = 0
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            max_page = len(doc)
            doc.close()
        except Exception:
            pass

        # Step 4: Extract tree structure from API response
        raw_tree = self._api_response_to_tree_dict(
            tree_result, doc_name, doc_id, max_page=max_page
        )

        # Step 5: Check if tree nodes already have text embedded
        has_text = self._tree_has_text(raw_tree["structure"])

        # Step 6: If no text in nodes, try OCR markdown from the API
        if add_node_text and not has_text:
            page_texts: dict[int, str] = {}
            try:
                ocr_result = client.get_ocr(doc_id, format="page")
                if ocr_result.get("status") == "completed":
                    pages = ocr_result.get("result", [])
                    for page_data in pages:
                        page_num = page_data.get("page_index", 0)
                        text = page_data.get("markdown", "")
                        if page_num > 0 and text:
                            page_texts[page_num] = text
                    logger.info(
                        f"OCR markdown retrieved for {len(page_texts)} pages"
                    )
            except Exception as e:
                logger.warning(f"Could not retrieve OCR text: {e}")

            if page_texts:
                self._embed_texts_in_tree(
                    raw_tree["structure"], page_texts
                )

        # Step 6: Validate and save
        tree = load_tree_from_dict(raw_tree)
        output_path = self.settings.trees_dir / f"{doc_name}_structure.json"
        tree_to_json(tree, output_path)

        # Also save the raw API response for reference
        raw_path = self.settings.trees_dir / f"{doc_name}_raw_api.json"
        with open(raw_path, "w") as f:
            json.dump(tree_result, f, indent=2)

        logger.info(
            f"Tree saved to {output_path} ({tree.total_nodes} nodes)"
        )

        return tree

    async def generate_tree_local(
        self,
        pdf_path: str | Path,
        doc_name: str | None = None,
        add_summaries: bool = True,
    ) -> PageIndexTree:
        """Generate a PageIndex tree using local Ollama.

        No API keys required — works entirely offline.

        Args:
            pdf_path: Path to the PDF file.
            doc_name: Optional name for the document.
            add_summaries: Whether to generate per-node summaries.

        Returns:
            Generated and validated PageIndexTree.
        """
        from finsight_mac.document.local_tree_generator import (
            LocalTreeGenerator,
        )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_name = doc_name or pdf_path.stem
        generator = LocalTreeGenerator()
        tree = await generator.generate(
            pdf_path, doc_name, add_summaries
        )

        output_path = (
            self.settings.trees_dir / f"{doc_name}_structure.json"
        )
        tree_to_json(tree, output_path)
        logger.info(
            f"Local tree saved to {output_path} "
            f"({tree.total_nodes} nodes)"
        )
        return tree

    def generate_tree(
        self,
        pdf_path: str | Path,
        doc_name: str | None = None,
        model: str | None = None,
        add_summaries: bool = True,
    ) -> PageIndexTree:
        """Generate a PageIndex tree from a PDF file.

        Fallback chain:
          1. Cloud API (if PAGEINDEX_API_KEY is set)
          2. Local Ollama (always available when Ollama running)
          3. Legacy page_index() (if installed with OpenAI key)

        Args:
            pdf_path: Path to the PDF file.
            doc_name: Optional name for the document.
            model: Override the PageIndex model (legacy mode only).
            add_summaries: Whether to generate node summaries.

        Returns:
            Generated and validated PageIndexTree.
        """
        # Prefer Cloud API if key is configured
        if self.settings.pageindex_api_key:
            return self.generate_tree_cloud(
                pdf_path=pdf_path,
                doc_name=doc_name,
                add_node_text=True,
            )

        # Try local Ollama generation
        try:
            import asyncio

            from finsight_mac.document.local_tree_generator import (
                LocalTreeGenerator,
            )

            pdf_path_obj = Path(pdf_path)
            if not pdf_path_obj.exists():
                raise FileNotFoundError(
                    f"PDF not found: {pdf_path_obj}"
                )
            name = doc_name or pdf_path_obj.stem

            generator = LocalTreeGenerator()
            tree = asyncio.run(
                generator.generate(
                    pdf_path_obj, name, add_summaries
                )
            )
            output_path = (
                self.settings.trees_dir / f"{name}_structure.json"
            )
            tree_to_json(tree, output_path)
            logger.info(
                f"Local tree saved to {output_path} "
                f"({tree.total_nodes} nodes)"
            )
            return tree
        except Exception as e:
            logger.warning(f"Local Ollama tree generation failed: {e}")

        # Legacy: use page_index() function (requires OpenAI key)
        try:
            from pageindex import page_index  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            raise ImportError(
                "Set PAGEINDEX_API_KEY for cloud tree generation, "
                "ensure Ollama is running for local generation, or "
                "install pageindex with legacy support."
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_name = doc_name or pdf_path.stem
        pi_model = model or self.settings.pageindex_model

        logger.info(
            f"Generating PageIndex tree for {doc_name} "
            f"using {pi_model}..."
        )

        raw_tree = page_index(
            str(pdf_path),
            model=pi_model,
            if_add_node_id="yes",
            if_add_node_summary="yes" if add_summaries else "no",
            if_add_doc_description="yes",
            if_add_node_text="no",
        )

        tree = load_tree_from_dict(raw_tree)

        output_path = (
            self.settings.trees_dir / f"{doc_name}_structure.json"
        )
        tree_to_json(tree, output_path)
        logger.info(
            f"Tree saved to {output_path} ({tree.total_nodes} nodes)"
        )

        return tree

    def _api_response_to_tree_dict(
        self,
        api_response: dict,
        doc_name: str,
        doc_id: str,
        max_page: int = 0,
    ) -> dict:
        """Convert PageIndex Cloud API tree response to our tree dict format.

        The API returns {"result": [root_nodes...]} where each node has
        page_index (single int) and prefix_summary instead of our
        start_index/end_index and summary fields.
        """
        result = api_response.get("result", [])

        # result is a list of root nodes
        if isinstance(result, list):
            api_nodes = result
        elif isinstance(result, dict):
            api_nodes = (
                result.get("structure")
                or result.get("tree")
                or result.get("nodes", [])
            )
        else:
            api_nodes = []

        # Normalize nodes: map page_index → start/end, prefix_summary → summary
        normalized = self._normalize_nodes(api_nodes, max_page=max_page)

        # Use root node's prefix_summary as doc description
        doc_description = ""
        if api_nodes and isinstance(api_nodes[0], dict):
            doc_description = api_nodes[0].get("prefix_summary", "")

        return {
            "doc_name": doc_name,
            "doc_description": doc_description,
            "structure": normalized,
        }

    def _normalize_nodes(
        self, nodes: list, max_page: int = 0
    ) -> list:
        """Normalize API node format to our internal format.

        Maps page_index → start_index and computes end_index from
        the next sibling's page_index (or max_page for the last node).
        Maps prefix_summary → summary.
        """
        result = []
        for i, node in enumerate(nodes):
            children = node.get("nodes", node.get("children", []))
            page_idx = node.get("page_index", 0)

            # Compute end_index: next sibling's page_index - 1, or max_page
            if i + 1 < len(nodes):
                next_page = nodes[i + 1].get("page_index", page_idx)
                end_idx = max(next_page - 1, page_idx)
            elif max_page > 0:
                end_idx = max_page
            else:
                end_idx = page_idx

            # If node has children, end_index is last child's end
            # (will be computed recursively)
            normalized_children = self._normalize_nodes(
                children, max_page=end_idx
            )

            # Refine end_index from children if available
            if normalized_children:
                child_max = max(
                    c.get("end_index", 0) for c in normalized_children
                )
                if child_max > 0:
                    end_idx = max(end_idx, child_max)

            normalized = {
                "title": node.get("title", ""),
                "node_id": node.get("node_id", ""),
                "start_index": node.get("start_index", page_idx),
                "end_index": node.get("end_index", end_idx),
                "summary": node.get(
                    "summary", node.get("prefix_summary", "")
                ),
                "text": node.get("text"),
                "nodes": normalized_children,
            }
            result.append(normalized)
        return result

    def _embed_texts_in_tree(
        self,
        nodes: list[dict],
        page_texts: dict[int, str],
    ) -> None:
        """Embed page texts into leaf nodes that don't have text."""
        for node in nodes:
            children = node.get("nodes", [])
            if children:
                self._embed_texts_in_tree(children, page_texts)
            elif not node.get("text"):
                # Leaf node without text — concatenate page texts
                start = node.get("start_index", 0)
                end = node.get("end_index", start)
                parts = []
                for p in range(start, end + 1):
                    if p in page_texts:
                        parts.append(page_texts[p])
                if parts:
                    node["text"] = "\n\n".join(parts)

    @staticmethod
    def _tree_has_text(nodes: list[dict]) -> bool:
        """Check if any node in the tree has embedded text."""
        for node in nodes:
            if node.get("text"):
                return True
            children = node.get("nodes", [])
            if children and DocumentPipeline._tree_has_text(children):
                return True
        return False

    def load_tree(self, doc_name: str) -> PageIndexTree | None:
        """Load a pre-generated PageIndex tree by document name.

        Args:
            doc_name: Document name (matches filename without _structure.json).

        Returns:
            PageIndexTree if found, None otherwise.
        """
        tree_path = self.settings.trees_dir / f"{doc_name}_structure.json"
        if not tree_path.exists():
            tree_path = self.settings.trees_dir / f"{doc_name}.json"
        if not tree_path.exists():
            logger.warning(f"No tree found for {doc_name}")
            return None

        return load_tree(tree_path)

    def list_available_trees(self) -> list[str]:
        """List all available pre-generated trees."""
        trees_dir = self.settings.trees_dir
        if not trees_dir.exists():
            return []

        return [
            p.stem.replace("_structure", "")
            for p in trees_dir.glob("*.json")
            if not p.stem.endswith("_raw_api")
        ]

    def load_all_trees(self) -> dict[str, PageIndexTree]:
        """Load all available trees into a dictionary."""
        result: dict[str, PageIndexTree] = {}
        for name in self.list_available_trees():
            tree = self.load_tree(name)
            if tree:
                result[name] = tree
        return result

    def get_pdf_path(self, doc_name: str) -> Path | None:
        """Find the PDF file for a document name."""
        for ext in [".pdf", ".PDF"]:
            path = self.settings.filings_dir / f"{doc_name}{ext}"
            if path.exists():
                return path
        return None

    def extract_page_texts(
        self,
        pdf_path: str | Path,
        use_docling: bool = True,
    ) -> dict[int, str]:
        """Extract page texts from a PDF, using Docling if available.

        Docling provides layout-aware text with correct reading order and
        structured table extraction. Falls back to PyMuPDF if unavailable.

        Args:
            pdf_path: Path to the PDF file.
            use_docling: Whether to attempt using Docling (default True).

        Returns:
            Dict mapping page numbers (1-indexed) to extracted text.
        """
        pdf_path = Path(pdf_path)

        if use_docling:
            try:
                from finsight_mac.document.docling_processor import (
                    DoclingProcessor,
                    DoclingResult,
                    is_docling_available,
                )

                if is_docling_available():
                    cache_dir = self.settings.data_dir / "docling_cache"
                    cache_path = (
                        cache_dir / f"{pdf_path.stem}_docling.json"
                    )
                    cached = DoclingResult.load_cache(cache_path)

                    if cached:
                        logger.info(
                            f"Using cached Docling result for "
                            f"{pdf_path.stem}"
                        )
                        return cached.page_texts

                    processor = DoclingProcessor(
                        enable_ocr=False, table_mode="fast"
                    )
                    result = processor.process_pdf(pdf_path)
                    result.save_cache(cache_dir)
                    return result.page_texts
            except Exception as e:
                logger.warning(
                    f"Docling processing failed, "
                    f"falling back to PyMuPDF: {e}"
                )

        from finsight_core.pageindex.text_extractor import (
            extract_pages_from_pdf,
        )
        return extract_pages_from_pdf(str(pdf_path))

    def extract_page_markdowns(
        self,
        pdf_path: str | Path,
    ) -> dict[int, str] | None:
        """Extract structured Markdown per page using Docling.

        Returns Markdown with proper headings, tables, and reading order.
        Only available when Docling is installed.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict mapping page numbers to Markdown, or None if unavailable.
        """
        try:
            from finsight_mac.document.docling_processor import (
                DoclingProcessor,
                DoclingResult,
                is_docling_available,
            )

            if not is_docling_available():
                return None

            cache_dir = self.settings.data_dir / "docling_cache"
            cache_path = (
                cache_dir / f"{Path(pdf_path).stem}_docling.json"
            )
            cached = DoclingResult.load_cache(cache_path)

            if cached:
                return cached.page_markdowns

            processor = DoclingProcessor(
                enable_ocr=False, table_mode="fast"
            )
            result = processor.process_pdf(pdf_path)
            result.save_cache(cache_dir)
            return result.page_markdowns

        except Exception as e:
            logger.warning(
                f"Docling Markdown extraction failed: {e}"
            )
            return None
