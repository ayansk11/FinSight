"""Analysis page — view findings, risk scores, and reports."""

from __future__ import annotations

import streamlit as st


def render_analysis_page() -> None:
    """Render the analysis dashboard."""
    st.title("Analysis Dashboard")
    st.info(
        "Run a query from the **Chat** page to generate analysis results. "
        "This page will display findings, risk scores, and reports."
    )

    # TODO: Implement in Week 3-4
    # - Display findings with citations
    # - Risk heat map visualization
    # - Financial ratio charts
    # - Export to .finsight bundle button

    st.subheader("Recent Analyses")
    st.caption("No analyses yet. Start by asking a question in Chat.")

    st.subheader("Export")
    st.caption(
        "After running an analysis, you can export results as a "
        "`.finsight` bundle for the iPhone companion app."
    )
