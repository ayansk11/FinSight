"""LangGraph shared state definition.

FinSightState is the TypedDict that flows through the entire LangGraph workflow.
Every agent reads from and writes to this state. It serves as the central
data contract for the multi-agent system.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

# --- LangGraph state (TypedDict for StateGraph compatibility) ---


class FinSightState(TypedDict, total=False):
    """Shared state for the LangGraph orchestrator.

    This TypedDict defines every field that flows through the graph.
    Agents read inputs from this state and append their outputs.

    Fields use LangGraph's Annotated[list, add] pattern for accumulation —
    each agent appends to lists rather than overwriting them.
    """

    # --- Input (set by user/UI) ---
    query: str  # User's natural language question
    ticker: str  # Company ticker symbol (e.g., "AAPL")
    filing_types: list[str]  # Which filing types to analyze (e.g., ["10-K"])
    fiscal_years: list[int]  # Which fiscal years to cover

    # --- Document context (set by document pipeline) ---
    filings: list[dict]  # List of Filing model dicts loaded for this query
    trees: list[dict]  # List of PageIndexTree dicts for loaded filings
    tree_outlines: list[str]  # Text outlines of trees for LLM context

    # --- Router decision ---
    route: str  # "single_doc" | "multi_doc" | "quantitative" | "risk_focused"
    agents_to_run: list[str]  # Which agents the router selected
    needs_rlm: bool  # Whether cross-document RLM reasoning is needed

    # --- Agent outputs (accumulated) ---
    findings: Annotated[list[dict], add]  # Finding dicts from all agents
    risk_scores: Annotated[list[dict], add]  # RiskScore dicts from risk agent
    agent_summaries: Annotated[list[dict], add]  # AgentOutput summary dicts

    # --- RLM cross-document reasoning ---
    page_texts_by_filing: dict  # {filing_name: {page_num: text}} for RLM
    rlm_result: dict  # RLMResult as dict (answer, method, iterations)

    # --- Synthesis output ---
    report: str  # Final synthesized report text
    executive_summary: str  # One-paragraph executive summary

    # --- Metadata ---
    model_name: str  # Which LLM model is being used
    total_duration_seconds: float  # Total query processing time
    error: str | None  # Top-level error if workflow failed
