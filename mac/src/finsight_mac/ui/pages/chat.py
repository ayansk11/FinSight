"""Chat page — main Q&A interface for FinSight.

Users ask questions about loaded SEC filings and the LangGraph
workflow routes them through the appropriate agents.
"""

from __future__ import annotations

import asyncio
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
    if st.session_state.loaded_trees:
        with st.expander("Loaded Documents", expanded=False):
            for name, tree in st.session_state.loaded_trees.items():
                st.text(f"📄 {name} ({tree.get('total_nodes', '?')} nodes)")
    else:
        st.info(
            "No documents loaded. Go to **Documents** page to upload a PDF "
            "or load a pre-generated PageIndex tree."
        )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]
                cols = st.columns(3)
                with cols[0]:
                    st.caption(f"⏱️ {meta.get('duration', '?')}s")
                with cols[1]:
                    st.caption(f"🔍 {meta.get('findings_count', '?')} findings")
                with cols[2]:
                    st.caption(f"📊 Route: {meta.get('route', '?')}")

    # Chat input
    if prompt := st.chat_input("Ask about your financial documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response, metadata = _run_query(prompt)
                st.markdown(response)

                if metadata:
                    cols = st.columns(3)
                    with cols[0]:
                        st.caption(f"⏱️ {metadata.get('duration', '?')}s")
                    with cols[1]:
                        st.caption(f"🔍 {metadata.get('findings_count', '?')} findings")
                    with cols[2]:
                        st.caption(f"📊 Route: {metadata.get('route', '?')}")

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "metadata": metadata}
        )


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
        from finsight_mac.graph.workflow import get_workflow

        workflow = get_workflow()

        # Build initial state
        tree_list = list(trees.values())
        tree_outlines = []
        for t in tree_list:
            from finsight_core.models.document import PageIndexTree
            tree_obj = PageIndexTree.model_validate(t)
            tree_outlines.append(tree_obj.get_tree_outline())

        initial_state = {
            "query": query,
            "ticker": "",
            "filing_types": [],
            "fiscal_years": [],
            "filings": [],
            "trees": tree_list,
            "tree_outlines": tree_outlines,
            "findings": [],
            "risk_scores": [],
            "agent_summaries": [],
            "report": "",
            "executive_summary": "",
            "model_name": "",
            "total_duration_seconds": 0,
        }

        start = time.time()
        result = asyncio.run(workflow.ainvoke(initial_state))
        duration = time.time() - start

        report = result.get("report", "")
        exec_summary = result.get("executive_summary", "")
        findings = result.get("findings", [])
        route = result.get("route", "unknown")

        response = exec_summary or report or "Analysis complete but no report generated."

        if report and exec_summary:
            response = f"**Executive Summary:** {exec_summary}\n\n---\n\n{report}"

        metadata = {
            "duration": f"{duration:.1f}",
            "findings_count": len(findings),
            "route": route,
        }

        return response, metadata

    except Exception as e:
        return f"Error running analysis: {str(e)}", {"error": str(e)}
