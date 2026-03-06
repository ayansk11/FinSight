"""Docling integration — AI-powered document conversion for rich content extraction.

Docling (IBM Research) provides layout analysis, table structure recognition,
reading order detection, and XBRL parsing. It runs entirely locally using
specialized ML models (DocLayNet for layout, TableFormer for tables).

This module is an OPTIONAL enrichment layer that complements PageIndex:
- PageIndex: Semantic hierarchical tree (navigation index)
- Docling: Structured content extraction (tables, layout, reading order)

When Docling is available, agents receive structured Markdown with real tables
instead of raw text blobs. When unavailable, the system falls back to PyMuPDF.

Install with: pip install finsight-mac[docling]
Performance: ~1.3 sec/page on Apple Silicon (one-time processing, results cached)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check if Docling is available
_DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter
    _DOCLING_AVAILABLE = True
except ImportError:
    pass


def is_docling_available() -> bool:
    """Check if Docling is installed and usable."""
    return _DOCLING_AVAILABLE


class DoclingProcessor:
    """Process documents using Docling for rich content extraction.

    Provides structured content (Markdown with tables, proper headings,
    correct reading order) that dramatically improves agent analysis quality
    compared to raw PyMuPDF text extraction.

    Example:
        >>> processor = DoclingProcessor()
        >>> result = processor.process_pdf("path/to/10k.pdf")
        >>> markdown = result.markdown
        >>> tables = result.tables
        >>> page_markdowns = result.page_markdowns
    """

    def __init__(
        self,
        enable_ocr: bool = False,
        table_mode: str = "fast",
    ) -> None:
        """Initialize the Docling processor.

        Args:
            enable_ocr: Whether to run OCR on scanned/image regions.
                       Disabled by default for programmatic PDFs (SEC filings).
            table_mode: Table structure recognition mode: "fast" or "accurate".
                       "fast" is recommended for M4 MacBook Air.
        """
        if not _DOCLING_AVAILABLE:
            raise ImportError(
                "Docling is not installed. Install with: "
                "pip install finsight-mac[docling]"
            )

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
            TableStructureOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # Configure pipeline for SEC filings
        table_modes = {
            "fast": TableFormerMode.FAST,
            "accurate": TableFormerMode.ACCURATE,
        }

        pipeline_options = PdfPipelineOptions(
            do_ocr=enable_ocr,
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                mode=table_modes.get(table_mode, TableFormerMode.FAST),
            ),
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

        logger.info(
            f"Docling processor initialized (OCR={enable_ocr}, tables={table_mode})"
        )

    def process_pdf(self, pdf_path: str | Path) -> DoclingResult:
        """Process a PDF file and extract structured content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            DoclingResult with structured content, tables, and page-level data.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Processing {pdf_path.name} with Docling...")
        result = self._converter.convert(str(pdf_path))
        doc = result.document

        # Extract full document as Markdown
        full_markdown = doc.export_to_markdown()

        # Extract tables
        tables = self._extract_tables(doc)

        # Build page-level Markdown mapping
        page_markdowns = self._build_page_markdowns(doc)

        # Extract page-level text (like PyMuPDF but with correct reading order)
        page_texts = self._build_page_texts(doc)

        total_pages = len(doc.pages) if hasattr(doc, 'pages') else 0

        logger.info(
            f"Docling processed {pdf_path.name}: "
            f"{total_pages} pages, {len(tables)} tables extracted"
        )

        return DoclingResult(
            doc_name=pdf_path.stem,
            markdown=full_markdown,
            tables=tables,
            page_markdowns=page_markdowns,
            page_texts=page_texts,
            total_pages=total_pages,
            raw_document=doc,
        )

    def _extract_tables(self, doc: Any) -> list[dict]:
        """Extract structured tables from the DoclingDocument."""
        tables = []
        try:
            for i, table_item in enumerate(doc.tables):
                table_data = {
                    "index": i,
                    "markdown": "",
                    "page": None,
                    "rows": 0,
                    "cols": 0,
                }

                # Get table as Markdown
                if hasattr(table_item, 'export_to_markdown'):
                    table_data["markdown"] = table_item.export_to_markdown()

                # Get table metadata
                if hasattr(table_item, 'data'):
                    td = table_item.data
                    table_data["rows"] = getattr(td, 'num_rows', 0)
                    table_data["cols"] = getattr(td, 'num_cols', 0)

                # Get page number from provenance
                if hasattr(table_item, 'prov') and table_item.prov:
                    prov = table_item.prov[0]
                    table_data["page"] = getattr(prov, 'page_no', None)

                tables.append(table_data)
        except Exception as e:
            logger.warning(f"Table extraction error: {e}")

        return tables

    def _build_page_markdowns(self, doc: Any) -> dict[int, str]:
        """Build a mapping of page number → Markdown content.

        This provides Docling's structured Markdown (with proper headings,
        tables, and reading order) indexed by page number for integration
        with PageIndex tree nodes.
        """
        page_markdowns: dict[int, str] = {}

        try:
            # Iterate through document elements and group by page
            for item in doc.iterate_items():
                # Handle both tuple (label, item) and direct item
                if isinstance(item, tuple):
                    _, element = item
                else:
                    element = item

                if not hasattr(element, 'prov') or not element.prov:
                    continue

                for prov in element.prov:
                    page_no = getattr(prov, 'page_no', None)
                    if page_no is None:
                        continue

                    # 1-indexed to match PageIndex convention
                    page_num = page_no + 1 if page_no == 0 else page_no

                    text = ""
                    if hasattr(element, 'export_to_markdown'):
                        text = element.export_to_markdown()
                    elif hasattr(element, 'text'):
                        text = element.text or ""

                    if text.strip():
                        if page_num not in page_markdowns:
                            page_markdowns[page_num] = ""
                        page_markdowns[page_num] += text + "\n\n"

        except Exception as e:
            logger.warning(f"Page Markdown extraction error: {e}")

        return page_markdowns

    def _build_page_texts(self, doc: Any) -> dict[int, str]:
        """Build page-text mapping (plain text, correct reading order).

        Compatible with the existing extract_node_text() interface —
        can be used as a drop-in replacement for PyMuPDF page texts.
        """
        page_texts: dict[int, str] = {}

        try:
            for item in doc.iterate_items():
                if isinstance(item, tuple):
                    _, element = item
                else:
                    element = item

                if not hasattr(element, 'prov') or not element.prov:
                    continue

                for prov in element.prov:
                    page_no = getattr(prov, 'page_no', None)
                    if page_no is None:
                        continue

                    page_num = page_no + 1 if page_no == 0 else page_no

                    text = getattr(element, 'text', "") or ""
                    if text.strip():
                        if page_num not in page_texts:
                            page_texts[page_num] = ""
                        page_texts[page_num] += text + "\n"

        except Exception as e:
            logger.warning(f"Page text extraction error: {e}")

        return page_texts


class DoclingResult:
    """Result of Docling document processing."""

    def __init__(
        self,
        doc_name: str,
        markdown: str,
        tables: list[dict],
        page_markdowns: dict[int, str],
        page_texts: dict[int, str],
        total_pages: int,
        raw_document: Any = None,
    ) -> None:
        self.doc_name = doc_name
        self.markdown = markdown
        self.tables = tables
        self.page_markdowns = page_markdowns
        self.page_texts = page_texts
        self.total_pages = total_pages
        self.raw_document = raw_document

    def get_tables_for_pages(self, start: int, end: int) -> list[dict]:
        """Get tables within a page range (for PageIndex node integration)."""
        return [
            t for t in self.tables
            if t.get("page") is not None
            and start <= t["page"] <= end
        ]

    def get_markdown_for_pages(self, start: int, end: int) -> str:
        """Get concatenated Markdown for a page range."""
        parts = []
        for page_num in range(start, end + 1):
            if page_num in self.page_markdowns:
                parts.append(f"--- Page {page_num} ---\n{self.page_markdowns[page_num]}")
        return "\n".join(parts)

    def save_cache(self, cache_dir: str | Path) -> Path:
        """Save processing results to a cache file for reuse."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{self.doc_name}_docling.json"

        cache_data = {
            "doc_name": self.doc_name,
            "markdown": self.markdown,
            "tables": self.tables,
            "page_markdowns": {str(k): v for k, v in self.page_markdowns.items()},
            "page_texts": {str(k): v for k, v in self.page_texts.items()},
            "total_pages": self.total_pages,
        }
        cache_path.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
        logger.info(f"Docling cache saved: {cache_path}")
        return cache_path

    @classmethod
    def load_cache(cls, cache_path: str | Path) -> DoclingResult | None:
        """Load cached Docling results."""
        cache_path = Path(cache_path)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return cls(
                doc_name=data["doc_name"],
                markdown=data["markdown"],
                tables=data["tables"],
                page_markdowns={int(k): v for k, v in data["page_markdowns"].items()},
                page_texts={int(k): v for k, v in data["page_texts"].items()},
                total_pages=data["total_pages"],
            )
        except Exception as e:
            logger.warning(f"Failed to load Docling cache: {e}")
            return None
