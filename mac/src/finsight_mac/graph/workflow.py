"""LangGraph workflow — the main orchestration graph.

Defines the StateGraph that routes queries through the appropriate
agents and collects their outputs into the shared state.

Flow:
  user_query → router → [doc_intel] → [quant?] → [risk?] → synthesis → response
"""

from __future__ import annotations

import logging

from finsight_core.models.analysis import AgentOutput, Finding
from finsight_core.models.state import FinSightState
from langgraph.graph import END, StateGraph

from finsight_mac.agents.doc_intel import run_doc_intel_agent
from finsight_mac.agents.quant import run_quant_agent
from finsight_mac.agents.risk import run_risk_agent
from finsight_mac.agents.synthesis import run_synthesis_agent
from finsight_mac.graph.router import classify_query

logger = logging.getLogger(__name__)


# --- Node Functions ---


async def router_node(state: FinSightState) -> dict:
    """Classify the query and decide which agents to run."""
    query = state["query"]
    num_filings = len(state.get("trees", []))

    result = classify_query(query, num_filings)

    return {
        "route": result["route"],
        "agents_to_run": result["agents_to_run"],
    }


async def doc_intel_node(state: FinSightState) -> dict:
    """Run the Document Intelligence Agent."""
    from finsight_core.pageindex.parser import load_tree_from_dict
    from finsight_core.pageindex.text_extractor import extract_pages_from_pdf

    query = state["query"]
    trees_data = state.get("trees", [])

    if not trees_data:
        return {
            "findings": [],
            "agent_summaries": [{"agent": "doc_intel", "summary": "No documents loaded."}],
        }

    # Process first tree (single-doc for now)
    tree_data = trees_data[0]
    tree = load_tree_from_dict(tree_data)

    # Try to get page texts from PDF
    filings = state.get("filings", [])
    pdf_path = filings[0].get("pdf_path") if filings else None
    page_texts = None
    if pdf_path:
        try:
            page_texts = extract_pages_from_pdf(pdf_path)
        except Exception as e:
            logger.warning(f"Could not extract PDF text: {e}")

    filing_name = filings[0].get("display_name", tree.doc_name) if filings else tree.doc_name

    output = await run_doc_intel_agent(
        query=query,
        tree=tree,
        pdf_path=pdf_path,
        page_texts=page_texts,
        filing_display_name=filing_name,
    )

    return {
        "findings": [f.model_dump() for f in output.findings],
        "agent_summaries": [
            {"agent": "doc_intel", "summary": output.summary, "duration": output.duration_seconds}
        ],
    }


async def quant_node(state: FinSightState) -> dict:
    """Run the Quantitative Analysis Agent."""
    query = state["query"]
    ticker = state.get("ticker", "")
    doc_findings = state.get("findings", [])

    output = await run_quant_agent(
        query=query,
        ticker=ticker,
        doc_intel_findings=doc_findings,
    )

    return {
        "findings": [f.model_dump() for f in output.findings],
        "agent_summaries": [
            {"agent": "quant", "summary": output.summary, "duration": output.duration_seconds}
        ],
    }


async def risk_node(state: FinSightState) -> dict:
    """Run the Risk Classification Agent."""
    query = state["query"]
    doc_findings = state.get("findings", [])
    filings = state.get("filings", [])
    filing_name = filings[0].get("display_name", "") if filings else ""

    output = await run_risk_agent(
        query=query,
        doc_intel_findings=doc_findings,
        filing_display_name=filing_name,
    )

    return {
        "findings": [f.model_dump() for f in output.findings],
        "risk_scores": [rs.model_dump() for rs in output.risk_scores],
        "agent_summaries": [
            {"agent": "risk", "summary": output.summary, "duration": output.duration_seconds}
        ],
    }


async def synthesis_node(state: FinSightState) -> dict:
    """Run the Synthesis & Report Agent."""

    query = state["query"]
    summaries = state.get("agent_summaries", [])
    filings = state.get("filings", [])
    filing_names = [f.get("display_name", "") for f in filings] if filings else ["Unknown"]

    # Reconstruct agent outputs from state (simplified)
    doc_output = _find_agent_output(summaries, "doc_intel", state.get("findings", []))
    quant_output = _find_agent_output(summaries, "quant", state.get("findings", []))
    risk_output = _find_agent_output(summaries, "risk", state.get("findings", []))

    output = await run_synthesis_agent(
        query=query,
        doc_intel_output=doc_output,
        quant_output=quant_output,
        risk_output=risk_output,
        filing_names=filing_names,
    )

    report = ""
    exec_summary = output.summary
    if output.findings:
        report = output.findings[0].content

    return {
        "report": report,
        "executive_summary": exec_summary,
        "findings": [f.model_dump() for f in output.findings],
        "agent_summaries": [
            {"agent": "synthesis", "summary": output.summary, "duration": output.duration_seconds}
        ],
    }


def _find_agent_output(
    summaries: list[dict], agent_name: str, all_findings: list[dict]
) -> AgentOutput | None:
    """Reconstruct an AgentOutput from state data."""
    summary_data = next((s for s in summaries if s.get("agent") == agent_name), None)
    if summary_data is None:
        return None

    agent_findings = [
        Finding.model_validate(f) for f in all_findings if f.get("agent") == agent_name
    ]

    return AgentOutput(
        agent_name=agent_name,
        findings=agent_findings,
        summary=summary_data.get("summary", ""),
        duration_seconds=summary_data.get("duration", 0),
    )


# --- Conditional Edges ---


def should_run_quant(state: FinSightState) -> str:
    """Decide whether to run the quant agent."""
    agents = state.get("agents_to_run", [])
    if "quant" in agents:
        return "quant"
    return "check_risk"


def should_run_risk(state: FinSightState) -> str:
    """Decide whether to run the risk agent."""
    agents = state.get("agents_to_run", [])
    if "risk" in agents:
        return "risk"
    return "synthesis"


# --- Graph Construction ---


def build_workflow() -> StateGraph:
    """Build the FinSight LangGraph workflow.

    Returns a compiled StateGraph ready for invocation.
    """
    workflow = StateGraph(FinSightState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("doc_intel", doc_intel_node)
    workflow.add_node("quant", quant_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("synthesis", synthesis_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Router always goes to doc_intel first
    workflow.add_edge("router", "doc_intel")

    # After doc_intel, conditionally run quant
    workflow.add_conditional_edges(
        "doc_intel",
        should_run_quant,
        {"quant": "quant", "check_risk": "risk"},  # will be checked again
    )

    # After quant, conditionally run risk
    workflow.add_conditional_edges(
        "quant",
        should_run_risk,
        {"risk": "risk", "synthesis": "synthesis"},
    )

    # Risk always goes to synthesis
    workflow.add_edge("risk", "synthesis")

    # Synthesis goes to END
    workflow.add_edge("synthesis", END)

    return workflow.compile()


# Singleton compiled graph
_graph = None


def get_workflow():
    """Get the compiled workflow (singleton)."""
    global _graph
    if _graph is None:
        _graph = build_workflow()
    return _graph
