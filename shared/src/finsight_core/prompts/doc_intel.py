"""Document Intelligence Agent prompt templates.

These prompts guide the LLM through PageIndex tree navigation
and document analysis for SEC filings.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are the Document Intelligence Agent of FinSight, a financial document analysis system.

Your role is to analyze SEC filings (10-K, 10-Q, 8-K) using a hierarchical document tree structure.
You receive a tree outline showing the document's sections with node IDs, page ranges, and summaries.

## Your Capabilities
1. Navigate the document tree to find relevant sections for a query
2. Extract precise information from specific sections
3. Follow cross-references (e.g., "see Note 12") to related sections
4. Identify tables, financial statements, and their relationships
5. Provide exact citations with page numbers

## Response Format
Always respond with structured JSON:
{
    "relevant_nodes": ["node_id1", "node_id2"],
    "findings": [
        {
            "content": "The finding text with specific data points",
            "confidence": "high|medium|low",
            "citations": [
                {
                    "node_id": "0003",
                    "section_title": "Risk Factors",
                    "page_start": 15,
                    "page_end": 16,
                    "excerpt": "Brief supporting quote"
                }
            ]
        }
    ],
    "cross_references": ["List of cross-references to follow up on"],
    "summary": "Brief summary of analysis"
}

## Rules
- ALWAYS cite specific page numbers and section titles
- If information is not found, say so explicitly — never fabricate
- Prefer specific data points (numbers, dates, percentages) over vague descriptions
- Flag any contradictions between sections
- Note when information may be outdated (e.g., discussing forward-looking estimates from prior years)
"""


def build_tree_navigation_prompt(
    query: str,
    tree_outline: str,
    filing_name: str,
) -> str:
    """Build a prompt for LLM-guided tree navigation.

    The LLM examines the tree outline and selects which nodes
    are most likely to contain the answer to the query.
    """
    return f"""## Task: Select Relevant Document Sections

**Filing:** {filing_name}
**Query:** {query}

## Document Tree Structure
{tree_outline}

## Instructions
Examine the tree structure above and identify which sections (by node_id) are most likely to contain information relevant to the query. Consider:
1. Section titles that directly relate to the query topic
2. Sections that typically contain this type of information in SEC filings
3. Cross-reference sections that may provide supporting data

Return a JSON object:
{{
    "selected_nodes": ["node_id1", "node_id2", ...],
    "reasoning": "Brief explanation of why these sections were selected"
}}

Select 1-5 most relevant sections. Prefer specific subsections over broad parent sections."""


def build_section_analysis_prompt(
    query: str,
    section_title: str,
    section_text: str,
    filing_name: str,
) -> str:
    """Build a prompt for analyzing a specific document section."""
    return f"""## Task: Analyze Document Section

**Filing:** {filing_name}
**Section:** {section_title}
**Query:** {query}

## Section Content
{section_text}

## Instructions
Analyze this section to find information relevant to the query. Extract:
1. Specific data points, figures, and dates
2. Key statements and their implications
3. Any cross-references to other sections (e.g., "see Note 12", "as discussed in Item 7")
4. Tables or structured data with relevant information

Respond with JSON:
{{
    "findings": [
        {{
            "content": "Specific finding with data",
            "confidence": "high|medium|low",
            "excerpt": "Brief supporting text from the section"
        }}
    ],
    "cross_references": ["List of cross-reference targets to follow up"],
    "data_points": {{"metric_name": "value", ...}},
    "summary": "Brief section summary relevant to the query"
}}

If this section does not contain relevant information, return:
{{"findings": [], "cross_references": [], "data_points": {{}}, "summary": "No relevant information found in this section."}}"""
