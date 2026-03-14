"""Query router — classifies queries and determines which agents to run.

Analyzes the user's query to decide:
1. Route type: single_doc, multi_doc, quantitative, risk_focused
2. Which agents should be activated
3. Whether RLM is needed (cross-document reasoning)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Keywords that signal cross-document queries (triggers RLM)
CROSS_DOC_SIGNALS = [
    "compare",
    "comparison",
    "changed",
    "evolved",
    "trend",
    "over the years",
    "year over year",
    "yoy",
    "across",
    "difference between",
    "how has",
    "how have",
    "growth",
    "decline",
    "increase",
    "decrease",
    "historical",
    "multi-year",
    "three year",
    "five year",
]

# Keywords that signal quantitative focus
QUANT_SIGNALS = [
    "ratio",
    "pe",
    "p/e",
    "debt",
    "revenue",
    "income",
    "margin",
    "ebitda",
    "cash flow",
    "fcf",
    "roi",
    "roe",
    "eps",
    "earnings per share",
    "valuation",
    "price",
    "market cap",
    "dividend",
    "yield",
    "turnover",
    "benchmark",
    "sector",
    "industry average",
]

# Keywords that signal risk focus
RISK_SIGNALS = [
    "risk",
    "fraud",
    "concern",
    "weakness",
    "audit",
    "litigation",
    "lawsuit",
    "regulatory",
    "compliance",
    "governance",
    "related party",
    "impairment",
    "going concern",
    "material weakness",
    "restatement",
    "investigation",
    "enforcement",
    "violation",
]


def classify_query(
    query: str,
    num_filings: int = 1,
) -> dict[str, Any]:
    """Classify a query and determine the execution plan.

    Args:
        query: User's natural language question.
        num_filings: Number of filings loaded/available.

    Returns:
        Dict with route, agents_to_run, and needs_rlm.
    """
    query_lower = query.lower()

    # Detect signals
    is_cross_doc = any(s in query_lower for s in CROSS_DOC_SIGNALS)
    is_quant = any(s in query_lower for s in QUANT_SIGNALS)
    is_risk = any(s in query_lower for s in RISK_SIGNALS)

    # Multiple filings override to cross-doc if any comparison signals
    if num_filings > 1 and is_cross_doc:
        route = "multi_doc"
        needs_rlm = True
    elif is_risk and not is_quant:
        route = "risk_focused"
        needs_rlm = False
    elif is_quant and not is_risk:
        route = "quantitative"
        needs_rlm = False
    elif num_filings > 1:
        route = "multi_doc"
        needs_rlm = True
    else:
        route = "single_doc"
        needs_rlm = False

    # Determine which agents to run
    agents = _select_agents(route, is_quant, is_risk)

    logger.info(
        f"[Router] Query classified as '{route}' | "
        f"Agents: {agents} | RLM: {needs_rlm} | "
        f"Signals: cross_doc={is_cross_doc}, quant={is_quant}, risk={is_risk}"
    )

    return {
        "route": route,
        "agents_to_run": agents,
        "needs_rlm": needs_rlm,
    }


def _select_agents(route: str, is_quant: bool, is_risk: bool) -> list[str]:
    """Select which agents to run based on route and signals."""

    if route == "multi_doc":
        # Cross-document: run all agents, synthesis uses RLM
        return ["doc_intel", "quant", "risk", "synthesis"]

    if route == "risk_focused":
        # Risk-focused: doc intel provides context, risk classifies
        return ["doc_intel", "risk", "synthesis"]

    if route == "quantitative":
        # Quant-focused: doc intel extracts data, quant analyzes
        return ["doc_intel", "quant", "synthesis"]

    # Default single_doc: run all for comprehensive analysis
    agents = ["doc_intel"]
    if is_quant:
        agents.append("quant")
    if is_risk:
        agents.append("risk")
    agents.append("synthesis")

    return agents
