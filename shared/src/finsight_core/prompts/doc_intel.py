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

## SEC Filing Section Guide (10-K / 10-Q)
Use this guide to understand each section's PURPOSE:
- **Item 1 / Business** → Company description, products, competition, strategy
- **Item 1A / Risk Factors** → PRIMARY source for risk-related queries (business risks, market risks, regulatory risks)
- **Item 7 / MD&A** → Revenue analysis, operating expenses, margins, liquidity, capital resources
- **Item 8 / Financial Statements** → Balance sheet, income statement, cash flow, and Notes
- **Notes 1-4** → Accounting policies, revenue breakdown, EPS, financial instruments
- **Notes 8-10** → Leases, debt, shareholders' equity
- **Notes 11-13** → Compensation, commitments, segment data
- **Item 9A / Controls and Procedures** → Internal control assessment (procedural, NOT business risks)
- **Auditor Reports / Opinions** → Audit opinions on financial statements (NOT risk factors)

CRITICAL: Match sections by their CONTENT PURPOSE, not just keyword overlap.
- "Item 1A. Risk Factors" discusses BUSINESS risks. Audit opinions discuss PROCEDURAL controls.
- For risk queries → prioritize Item 1A and its subsections
- For revenue/financial queries → prioritize MD&A, Financial Statements, Revenue notes
- For strategy queries → prioritize Item 1 Business, MD&A

## Document Tree Structure
{tree_outline}

## Instructions
Select 2-5 sections whose PURPOSE best matches the query intent:
1. Sections with titles that DIRECTLY name the query topic (highest priority)
2. Sections with supporting quantitative data (MD&A, Financial Statements)
3. Cross-reference sections only if primary sections are insufficient

Return a JSON object:
{{
    "selected_nodes": ["node_id1", "node_id2", ...],
    "reasoning": "Brief explanation of why these sections were selected"
}}

Prefer specific subsections over broad parent sections. Do NOT select audit opinions or internal control sections unless the query explicitly asks about auditing or controls."""


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
