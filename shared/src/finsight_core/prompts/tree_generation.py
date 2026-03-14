"""Prompts for local Ollama-based PageIndex tree generation.

Used by LocalTreeGenerator to refine heuristic heading candidates
into a hierarchical document structure and generate node summaries.
"""

from __future__ import annotations

TREE_GEN_SYSTEM_PROMPT = """\
You are a document structure parser for SEC filings (10-K, 10-Q, 8-K).

Your task is to organize a list of detected section headings into
a hierarchical tree structure that accurately represents the document.

SEC 10-K filings typically follow this hierarchy:
  Document root (FORM 10-K or ANNUAL REPORT)
    └─ PART I
    │    ├─ Item 1. Business
    │    ├─ Item 1A. Risk Factors
    │    ├─ Item 1B. Unresolved Staff Comments
    │    └─ Item 2. Properties
    ├─ PART II
    │    ├─ Item 5. Market for Registrant's Common Equity
    │    ├─ Item 6. [Reserved]
    │    ├─ Item 7. Management's Discussion and Analysis
    │    ├─ Item 7A. Quantitative and Qualitative Disclosures
    │    ├─ Item 8. Financial Statements
    │    └─ Item 9. Changes in and Disagreements
    ├─ PART III
    ├─ PART IV
    └─ SIGNATURES

Rules:
- Output valid JSON only, no explanation text
- Preserve exact heading titles from the input
- PART sections are children of the document root
- Item sections are children of their PART
- Sub-headings within Items are children of that Item
- Remove any false positive headings (e.g., table headers, page footers)
- Every node must have: title (str), page (int), children (list)
"""


def build_hierarchy_prompt(
    candidates: list[dict],
    doc_name: str,
    sample_texts: dict[int, str] | None = None,
) -> str:
    """Build prompt asking Ollama to organize headings into a tree.

    Args:
        candidates: List of {title, page, level} heading candidates.
        doc_name: Document name for context.
        sample_texts: Optional {page: first ~150 chars} for context.

    Returns:
        User prompt string.
    """
    heading_list = []
    for c in candidates:
        level_label = {1: "PART", 2: "ITEM", 3: "SUB"}.get(c.get("level", 3), "SUB")
        heading_list.append(f"  - [{level_label}] page {c['page']}: {c['title']}")
    headings_text = "\n".join(heading_list)

    context_text = ""
    if sample_texts:
        snippets = []
        for page_num in sorted(sample_texts.keys())[:20]:
            text = sample_texts[page_num][:150].replace("\n", " ")
            snippets.append(f"  Page {page_num}: {text}")
        context_text = "\n\n## Page Previews (first 150 chars)\n" + "\n".join(snippets)

    return f"""## Document: {doc_name}

## Detected Headings
{headings_text}
{context_text}

## Task
Organize these headings into a hierarchical JSON tree.
Return ONLY valid JSON in this exact format:

{{
  "doc_description": "One-sentence description of this document",
  "structure": [
    {{
      "title": "Root section title",
      "page": 1,
      "children": [
        {{
          "title": "Child section",
          "page": 5,
          "children": []
        }}
      ]
    }}
  ]
}}"""


def build_summary_prompt(title: str, text_snippet: str) -> str:
    """Build prompt asking Ollama for a one-sentence section summary.

    Args:
        title: Section heading.
        text_snippet: First ~500 chars of the section text.

    Returns:
        User prompt string.
    """
    return f"""Summarize this SEC filing section in ONE sentence (max 30 words).

Section: {title}
Content preview:
{text_snippet}

One-sentence summary:"""
