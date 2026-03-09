"""Document Intelligence Agent.

Navigates PageIndex trees to find and analyze relevant sections
in SEC filings. Uses LLM-guided tree traversal (reasoning-based retrieval)
rather than vector similarity search.

This is the first and most critical agent — it provides the raw document
intelligence that all other agents build upon.
"""

from __future__ import annotations

import json
import logging
import time

from finsight_core.models.analysis import (
    AgentOutput,
    Citation,
    ConfidenceLevel,
    Finding,
)
from finsight_core.models.document import PageIndexTree
from finsight_core.pageindex.navigator import TreeNavigator
from finsight_core.pageindex.text_extractor import extract_node_text, extract_pages_from_pdf
from finsight_core.prompts.doc_intel import (
    SYSTEM_PROMPT,
    build_section_analysis_prompt,
    build_tree_navigation_prompt,
)

from finsight_mac.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


async def run_doc_intel_agent(
    query: str,
    tree: PageIndexTree,
    pdf_path: str | None = None,
    page_texts: dict[int, str] | None = None,
    filing_display_name: str = "",
    llm: OllamaClient | None = None,
) -> AgentOutput:
    """Execute the Document Intelligence Agent.

    Steps:
    1. Present tree outline to LLM for node selection
    2. Extract text for selected nodes
    3. Analyze each selected section
    4. Aggregate findings with citations

    Args:
        query: User's analysis question.
        tree: PageIndex tree for the filing.
        pdf_path: Path to PDF (for text extraction if page_texts not provided).
        page_texts: Pre-extracted page texts {page_num: text}.
        filing_display_name: Human-readable filing name for citations.
        llm: Optional OllamaClient (creates one if not provided).

    Returns:
        AgentOutput with findings and citations.
    """
    start = time.time()
    client = llm or OllamaClient()
    navigator = TreeNavigator(tree)
    filing_name = filing_display_name or tree.doc_name

    # Step 1: LLM-guided node selection
    logger.info(f"[DocIntel] Navigating tree for: {query}")
    tree_outline = navigator.get_nodes_for_llm_selection(max_depth=4)

    nav_prompt = build_tree_navigation_prompt(query, tree_outline, filing_name)
    try:
        nav_response = await client.achat_json(nav_prompt, system_message=SYSTEM_PROMPT)
        selected_ids = nav_response.get("selected_nodes", [])
        reasoning = nav_response.get("reasoning", "")
        logger.info(f"[DocIntel] Selected {len(selected_ids)} nodes: {selected_ids}")
        logger.info(f"[DocIntel] Reasoning: {reasoning}")
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(
            f"[DocIntel] Tree navigation failed: {e}. Falling back to keyword search."
        )
        keywords = query.lower().split()[:5]
        results = navigator.search_by_keywords(keywords, max_results=3)
        selected_ids = [r.node.node_id for r in results]

    # Step 1b: Validate LLM selection with keyword re-ranking
    if selected_ids:
        query_keywords = [w for w in query.lower().split() if len(w) > 3][:6]
        keyword_results = navigator.search_by_keywords(
            query_keywords, max_results=5
        )
        keyword_ids = {r.node.node_id for r in keyword_results}

        overlap = set(selected_ids) & keyword_ids
        if not overlap and keyword_results:
            # LLM picked nodes with zero keyword overlap — merge top results
            top_kw_ids = [r.node.node_id for r in keyword_results[:2]]
            selected_ids = top_kw_ids + selected_ids[:3]
            logger.info(
                f"[DocIntel] Re-ranked: merged keyword results {top_kw_ids}"
            )

    if not selected_ids:
        return AgentOutput(
            agent_name="doc_intel",
            summary="No relevant sections found in the document tree.",
            duration_seconds=time.time() - start,
        )

    # Step 2: Extract text for selected nodes
    if page_texts is None and pdf_path:
        logger.info(f"[DocIntel] Extracting text from PDF: {pdf_path}")
        page_texts = extract_pages_from_pdf(pdf_path)
    elif page_texts is None:
        # Use node text if embedded in tree, otherwise we have no text
        page_texts = {}

    # Step 3: Analyze each selected section
    findings: list[Finding] = []

    for node_id in selected_ids[:5]:  # Limit to 5 nodes
        node = navigator.get_by_id(node_id)
        if node is None:
            logger.warning(f"[DocIntel] Node {node_id} not found in tree")
            continue

        # Extract text for this node
        node_text = extract_node_text(node, page_texts, max_tokens=4000)
        if not node_text.strip():
            logger.info(f"[DocIntel] No text available for node {node_id} ({node.title})")
            continue

        # Analyze the section
        analysis_prompt = build_section_analysis_prompt(
            query=query,
            section_title=node.title,
            section_text=node_text,
            filing_name=filing_name,
        )

        try:
            analysis = await client.achat_json(
                analysis_prompt, system_message=SYSTEM_PROMPT
            )

            # Convert to Finding models
            for finding_data in analysis.get("findings", []):
                confidence_str = finding_data.get("confidence", "medium")
                try:
                    confidence = ConfidenceLevel(confidence_str)
                except ValueError:
                    confidence = ConfidenceLevel.MEDIUM

                citation = Citation(
                    filing_display_name=filing_name,
                    node_id=node_id,
                    section_title=node.title,
                    page_start=node.start_index,
                    page_end=node.end_index,
                    excerpt=finding_data.get("excerpt", ""),
                )

                findings.append(
                    Finding(
                        content=finding_data.get("content", ""),
                        agent="doc_intel",
                        confidence=confidence,
                        citations=[citation],
                        metadata={
                            "data_points": analysis.get("data_points", {}),
                            "cross_references": analysis.get("cross_references", []),
                        },
                    )
                )
        except Exception as e:
            logger.warning(f"[DocIntel] Section analysis failed for {node.title}: {e}")

    elapsed = time.time() - start
    summary = (
        f"Analyzed {len(selected_ids)} sections of {filing_name}. "
        f"Found {len(findings)} findings in {elapsed:.1f}s."
    )

    return AgentOutput(
        agent_name="doc_intel",
        findings=findings,
        summary=summary,
        duration_seconds=elapsed,
    )
