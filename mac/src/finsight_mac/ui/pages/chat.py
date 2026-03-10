"""Chat page — main Q&A interface for FinSight.

Users ask questions about loaded SEC filings and the LangGraph
workflow routes them through the appropriate agents.
"""

from __future__ import annotations

import asyncio
import re
import time

import streamlit as st


def render_chat_page() -> None:
    """Render the chat interface."""
    st.title("Financial Document Chat")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "loaded_trees" not in st.session_state:
        st.session_state.loaded_trees = {}

    # Show loaded documents
    trees = st.session_state.loaded_trees
    if trees:
        with st.expander(f"Loaded Documents ({len(trees)})", expanded=False):
            for name, tree in trees.items():
                nodes = tree.get("total_nodes") or _count_tree_nodes(tree)
                st.text(f"  {name} ({nodes} nodes)")
    else:
        st.info(
            "No documents loaded. Go to **Documents** page to upload a PDF "
            "or load a pre-generated PageIndex tree."
        )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "metadata" in message:
                _render_metadata(message["metadata"])

    # Chat input
    if prompt := st.chat_input("Ask about your financial documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Running FinSight analysis...", expanded=True) as status:
                st.write("Routing query...")
                response, metadata = _run_query(prompt)
                status.update(label="Analysis complete", state="complete")

            st.markdown(response)

            if metadata and not metadata.get("error"):
                _render_metadata(metadata)

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "metadata": metadata}
        )


def _render_metadata(meta: dict) -> None:
    """Render analysis metadata below a response."""
    cols = st.columns(4)
    with cols[0]:
        st.caption(f"Route: {meta.get('route', '?')}")
    with cols[1]:
        st.caption(f"Findings: {meta.get('findings_count', 0)}")
    with cols[2]:
        st.caption(f"Risks: {meta.get('risk_count', 0)}")
    with cols[3]:
        st.caption(f"Duration: {meta.get('duration', '?')}s")


def _detect_ticker(trees: dict) -> str:
    """Auto-detect ticker from loaded tree names.

    E.g., "AAPL_10K_2024" → "AAPL"
    """
    for name in trees:
        match = re.match(r"^([A-Z]{1,5})", name.upper())
        if match:
            return match.group(1)
    return ""


def _run_query(query: str) -> tuple[str, dict]:
    """Run a query through the LangGraph workflow.

    Returns (response_text, metadata_dict).
    """
    trees = st.session_state.get("loaded_trees", {})

    if not trees:
        return (
            "No documents loaded yet. Please upload a PDF or load a "
            "PageIndex tree from the **Documents** page first.",
            {},
        )

    try:
        from finsight_mac.config import get_settings
        from finsight_mac.graph.workflow import get_workflow

        settings = get_settings()
        workflow = get_workflow()

        tree_list = list(trees.values())
        ticker = st.session_state.get("ticker") or _detect_ticker(trees)

        filings = [
            {"display_name": t.get("doc_name", name), "pdf_path": None}
            for name, t in trees.items()
        ]

        initial_state = {
            "query": query,
            "ticker": ticker,
            "filings": filings,
            "trees": tree_list,
            "route": "",
            "agents_to_run": [],
            "needs_rlm": False,
            "findings": [],
            "risk_scores": [],
            "agent_summaries": [],
            "page_texts_by_filing": {},
            "rlm_result": {},
            "report": "",
            "executive_summary": "",
            "model_name": st.session_state.get("selected_model", settings.ollama_model),
            "total_duration_seconds": 0.0,
            "error": None,
        }

        start = time.time()
        result = asyncio.run(workflow.ainvoke(initial_state))
        duration = time.time() - start

        # Store result for Analysis page
        st.session_state.last_result = result
        st.session_state.last_query = query
        st.session_state.last_duration = duration

        report = result.get("report", "")
        exec_summary = result.get("executive_summary", "")
        findings = result.get("findings", [])
        risk_scores = result.get("risk_scores", [])
        route = result.get("route", "unknown")
        error = result.get("error")

        if error:
            return f"Analysis error: {error}", {"error": error}

        # Build response text
        if exec_summary and report:
            response = (
                f"**Executive Summary**\n\n{exec_summary}\n\n---\n\n"
                f"**Full Report**\n\n{report}"
            )
        elif report:
            response = report
        elif exec_summary:
            response = exec_summary
        else:
            response = "Analysis complete but no report generated."

        metadata = {
            "duration": f"{duration:.1f}",
            "findings_count": len(findings),
            "risk_count": len(risk_scores),
            "route": route,
        }

        return response, metadata

    except Exception as e:
        return f"Error running analysis: {e}", {"error": str(e)}


def _count_tree_nodes(tree_data: dict) -> int:
    """Count nodes in a raw tree dict."""
    def _count(nodes: list) -> int:
        total = len(nodes)
        for n in nodes:
            total += _count(n.get("nodes", []))
        return total

    return _count(tree_data.get("structure", []))
