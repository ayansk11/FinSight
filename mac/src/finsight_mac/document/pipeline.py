"""Document processing pipeline — PDF → PageIndex tree.

Wraps VectifyAI/PageIndex to generate hierarchical tree structures
from SEC filing PDFs. Also handles loading pre-generated trees
and managing the document store.
"""

from __future__ import annotations

import logging
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

    def generate_tree(
        self,
        pdf_path: str | Path,
        doc_name: str | None = None,
        model: str | None = None,
        add_summaries: bool = True,
    ) -> PageIndexTree:
        """Generate a PageIndex tree from a PDF file.

        Requires an OpenAI API key (PageIndex uses GPT-4o for tree generation).

        Args:
            pdf_path: Path to the PDF file.
            doc_name: Optional name for the document (defaults to filename).
            model: Override the PageIndex model (default: gpt-4o).
            add_summaries: Whether to generate node summaries.

        Returns:
            Generated and validated PageIndexTree.
        """
        try:
            from pageindex import page_index
        except ImportError:
            raise ImportError(
                "pageindex is required for tree generation. "
                "Install with: pip install pageindex"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_name = doc_name or pdf_path.stem
        pi_model = model or self.settings.pageindex_model

        logger.info(f"Generating PageIndex tree for {doc_name} using {pi_model}...")

        # Call PageIndex
        raw_tree = page_index(
            str(pdf_path),
            model=pi_model,
            if_add_node_id="yes",
            if_add_node_summary="yes" if add_summaries else "no",
            if_add_doc_description="yes",
            if_add_node_text="no",  # Text loaded on-demand from PDF
        )

        # Validate and convert
        tree = load_tree_from_dict(raw_tree)

        # Save to trees directory
        output_path = self.settings.trees_dir / f"{doc_name}_structure.json"
        tree_to_json(tree, output_path)
        logger.info(f"Tree saved to {output_path} ({tree.total_nodes} nodes)")

        return tree

    def load_tree(self, doc_name: str) -> PageIndexTree | None:
        """Load a pre-generated PageIndex tree by document name.

        Args:
            doc_name: Document name (matches filename without _structure.json).

        Returns:
            PageIndexTree if found, None otherwise.
        """
        tree_path = self.settings.trees_dir / f"{doc_name}_structure.json"
        if not tree_path.exists():
            # Try without _structure suffix
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

        # Try Docling first (structured extraction with tables)
        if use_docling:
            try:
                from finsight_mac.document.docling_processor import (
                    DoclingProcessor,
                    DoclingResult,
                    is_docling_available,
                )

                if is_docling_available():
                    # Check cache first
                    cache_dir = self.settings.data_dir / "docling_cache"
                    cache_path = cache_dir / f"{pdf_path.stem}_docling.json"
                    cached = DoclingResult.load_cache(cache_path)

                    if cached:
                        logger.info(f"Using cached Docling result for {pdf_path.stem}")
                        return cached.page_texts

                    # Process with Docling
                    processor = DoclingProcessor(enable_ocr=False, table_mode="fast")
                    result = processor.process_pdf(pdf_path)
                    result.save_cache(cache_dir)  # Cache for reuse
                    return result.page_texts
            except Exception as e:
                logger.warning(f"Docling processing failed, falling back to PyMuPDF: {e}")

        # Fallback: PyMuPDF
        from finsight_core.pageindex.text_extractor import extract_pages_from_pdf
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
            Dict mapping page numbers to Markdown, or None if Docling unavailable.
        """
        try:
            from finsight_mac.document.docling_processor import (
                DoclingProcessor,
                DoclingResult,
                is_docling_available,
            )

            if not is_docling_available():
                return None

            # Check cache
            cache_dir = self.settings.data_dir / "docling_cache"
            cache_path = cache_dir / f"{Path(pdf_path).stem}_docling.json"
            cached = DoclingResult.load_cache(cache_path)

            if cached:
                return cached.page_markdowns

            processor = DoclingProcessor(enable_ocr=False, table_mode="fast")
            result = processor.process_pdf(pdf_path)
            result.save_cache(cache_dir)
            return result.page_markdowns

        except Exception as e:
            logger.warning(f"Docling Markdown extraction failed: {e}")
            return None
