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
        "needs_rlm": result.get("needs_rlm", False),
    }


async def doc_intel_node(state: FinSightState) -> dict:
    """Run the Document Intelligence Agent on all filings."""
    from finsight_core.pageindex.parser import load_tree_from_dict
    from finsight_core.pageindex.text_extractor import extract_pages_from_pdf

    query = state["query"]
    trees_data = state.get("trees", [])
    filings = state.get("filings", [])

    if not trees_data:
        return {
            "findings": [],
            "agent_summaries": [{"agent": "doc_intel", "summary": "No documents loaded."}],
        }

    all_findings = []
    all_summaries = []
    all_page_texts: dict[str, dict[int, str]] = {}

    for i, tree_data in enumerate(trees_data):
        tree = load_tree_from_dict(tree_data)
        pdf_path = filings[i].get("pdf_path") if i < len(filings) else None
        filing_name = (
            filings[i].get("display_name", tree.doc_name) if i < len(filings) else tree.doc_name
        )

        page_texts = None
        if pdf_path:
            try:
                page_texts = extract_pages_from_pdf(pdf_path)
            except Exception as e:
                logger.warning(f"Could not extract PDF text for {filing_name}: {e}")

        if page_texts:
            all_page_texts[filing_name] = page_texts

        output = await run_doc_intel_agent(
            query=query,
            tree=tree,
            pdf_path=pdf_path,
            page_texts=page_texts,
            filing_display_name=filing_name,
        )

        all_findings.extend(f.model_dump() for f in output.findings)
        all_summaries.append(
            {"agent": "doc_intel", "summary": output.summary, "duration": output.duration_seconds}
        )

    return {
        "findings": all_findings,
        "agent_summaries": all_summaries,
        "page_texts_by_filing": all_page_texts,
    }


async def quant_node(state: FinSightState) -> dict:
    """Run the Quantitative Analysis Agent."""
    from finsight_mac.mcp.market_data import fetch_market_data

    query = state["query"]
    ticker = state.get("ticker", "")
    doc_findings = state.get("findings", [])

    # Fetch real market data from MCP servers (FRED, Finnhub, yfinance)
    market_data = None
    if ticker:
        try:
            market_data = await fetch_market_data(ticker)
            logger.info(f"[Quant] Fetched {len(market_data)} market data points for {ticker}")
        except Exception as e:
            logger.warning(f"[Quant] Market data fetch failed: {e}")

    output = await run_quant_agent(
        query=query,
        ticker=ticker,
        doc_intel_findings=doc_findings,
        market_data=market_data,
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


async def rlm_synthesis_node(state: FinSightState) -> dict:
    """Run RLM-powered cross-document synthesis."""
    from finsight_core.pageindex.parser import load_tree_from_dict

    from finsight_mac.llm.rlm_client import RLMClient

    query = state["query"]
    trees_data = state.get("trees", [])
    filings = state.get("filings", [])
    page_texts_by_filing = state.get("page_texts_by_filing", {})
    summaries = state.get("agent_summaries", [])

    # Build tree outlines for RLM
    filing_names: list[str] = []
    tree_outlines: dict[str, str] = {}
    for i, tree_data in enumerate(trees_data):
        tree = load_tree_from_dict(tree_data)
        name = filings[i].get("display_name", tree.doc_name) if i < len(filings) else tree.doc_name
        filing_names.append(name)
        tree_outlines[name] = tree.get_tree_outline(max_depth=4)

    # Run RLM cross-document analysis
    client = RLMClient()
    rlm_result = await client.analyze_cross_document(
        query=query,
        filing_names=filing_names,
        tree_outlines=tree_outlines,
        page_texts=page_texts_by_filing,
    )

    # Feed RLM analysis + upstream findings into synthesis
    doc_output = _find_agent_output(summaries, "doc_intel", state.get("findings", []))
    quant_output = _find_agent_output(summaries, "quant", state.get("findings", []))
    risk_output = _find_agent_output(summaries, "risk", state.get("findings", []))

    output = await run_synthesis_agent(
        query=query,
        doc_intel_output=doc_output,
        quant_output=quant_output,
        risk_output=risk_output,
        filing_names=filing_names,
        rlm_analysis=rlm_result.answer if rlm_result.success else None,
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
            {
                "agent": "rlm_synthesis",
                "summary": output.summary,
                "duration": output.duration_seconds,
            }
        ],
        "rlm_result": {
            "answer": rlm_result.answer,
            "method": rlm_result.method,
            "iterations": rlm_result.iterations,
            "success": rlm_result.success,
        },
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
    if state.get("needs_rlm"):
        return "rlm_synthesis"
    return "synthesis"


def should_use_rlm(state: FinSightState) -> str:
    """After risk, branch on whether RLM cross-doc reasoning is needed."""
    if state.get("needs_rlm"):
        return "rlm_synthesis"
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
    workflow.add_node("rlm_synthesis", rlm_synthesis_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Router always goes to doc_intel first
    workflow.add_edge("router", "doc_intel")

    # After doc_intel, conditionally run quant
    workflow.add_conditional_edges(
        "doc_intel",
        should_run_quant,
        {"quant": "quant", "check_risk": "risk"},
    )

    # After quant, conditionally run risk (or skip to RLM/synthesis)
    workflow.add_conditional_edges(
        "quant",
        should_run_risk,
        {"risk": "risk", "synthesis": "synthesis", "rlm_synthesis": "rlm_synthesis"},
    )

    # After risk, branch on RLM vs standard synthesis
    workflow.add_conditional_edges(
        "risk",
        should_use_rlm,
        {"rlm_synthesis": "rlm_synthesis", "synthesis": "synthesis"},
    )

    # Both synthesis paths go to END
    workflow.add_edge("synthesis", END)
    workflow.add_edge("rlm_synthesis", END)

    return workflow.compile()


# Singleton compiled graph
_graph = None


def get_workflow():
    """Get the compiled workflow (singleton)."""
    global _graph
    if _graph is None:
        _graph = build_workflow()
    return _graph
