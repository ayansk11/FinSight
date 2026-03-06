"""Text extraction utilities for PageIndex tree nodes.

Given a tree node's page range and a source of page texts,
extract the relevant text content for LLM consumption.
"""

from __future__ import annotations

from finsight_core.models.document import PageIndexTree, TreeNode


def extract_node_text(
    node: TreeNode,
    page_texts: dict[int, str],
    max_tokens: int = 0,
) -> str:
    """Extract text for a tree node from a page-text mapping.

    If the node has embedded text (from PageIndex's `if_add_node_text` option),
    returns that directly. Otherwise, concatenates page texts from the
    node's page range.

    Args:
        node: The tree node to extract text for.
        page_texts: Mapping of page number (1-indexed) to page text.
        max_tokens: If > 0, approximate max token limit (rough: 4 chars/token).

    Returns:
        Concatenated text for the node's pages.
    """
    # Prefer embedded text if available
    if node.text:
        text = node.text
        if max_tokens > 0:
            char_limit = max_tokens * 4
            if len(text) > char_limit:
                text = text[:char_limit] + "\n... [truncated]"
        return text

    # Concatenate page texts
    parts: list[str] = []
    total_chars = 0
    char_limit = max_tokens * 4 if max_tokens > 0 else 0

    for page_num in range(node.start_index, node.end_index + 1):
        page_text = page_texts.get(page_num, "")
        if not page_text:
            continue

        if char_limit > 0 and total_chars + len(page_text) > char_limit:
            remaining = char_limit - total_chars
            if remaining > 100:  # Only add if meaningful
                parts.append(f"--- Page {page_num} ---")
                parts.append(page_text[:remaining])
                parts.append("... [truncated]")
            break

        parts.append(f"--- Page {page_num} ---")
        parts.append(page_text)
        total_chars += len(page_text)

    return "\n".join(parts)


def extract_pages_from_pdf(pdf_path: str) -> dict[int, str]:
    """Extract text from all pages of a PDF file.

    Returns a mapping of page number (1-indexed) to page text.
    Uses PyMuPDF (fitz) for extraction.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict mapping page numbers (1-indexed) to extracted text.

    Raises:
        ImportError: If pymupdf is not installed.
        FileNotFoundError: If the PDF doesn't exist.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF text extraction. "
            "Install it with: pip install pymupdf"
        )

    from pathlib import Path

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_texts: dict[int, str] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                page_texts[page_num + 1] = text  # 1-indexed
    finally:
        doc.close()

    return page_texts


def get_node_text_with_context(
    node: TreeNode,
    tree: PageIndexTree,
    page_texts: dict[int, str],
    context_pages: int = 1,
) -> str:
    """Extract node text with surrounding context pages.

    Useful for cases where relevant information spans section boundaries.

    Args:
        node: The target node.
        tree: The full tree (for boundary checking).
        page_texts: Page text mapping.
        context_pages: Number of extra pages to include before/after.

    Returns:
        Text including context pages, with clear boundaries marked.
    """
    all_pages = sorted(page_texts.keys())
    if not all_pages:
        return ""

    min_page = max(node.start_index - context_pages, min(all_pages))
    max_page = min(node.end_index + context_pages, max(all_pages))

    parts: list[str] = []

    # Before-context
    for p in range(min_page, node.start_index):
        if p in page_texts:
            parts.append(f"--- Page {p} (context before) ---")
            parts.append(page_texts[p])

    # Main node text
    for p in range(node.start_index, node.end_index + 1):
        if p in page_texts:
            parts.append(f"--- Page {p} ---")
            parts.append(page_texts[p])

    # After-context
    for p in range(node.end_index + 1, max_page + 1):
        if p in page_texts:
            parts.append(f"--- Page {p} (context after) ---")
            parts.append(page_texts[p])

    return "\n".join(parts)
