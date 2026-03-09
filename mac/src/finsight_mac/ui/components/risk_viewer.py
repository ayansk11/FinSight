"""Risk score viewer component for Streamlit.

Renders risk scores as a severity-sorted table with colored indicators.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import streamlit as st


def render_risk_table(risk_scores: list[dict[str, Any]]) -> None:
    """Render risk scores as a table sorted by severity."""
    if not risk_scores:
        st.caption("No risk scores.")
        return

    # Sort by severity descending
    sorted_scores = sorted(
        risk_scores, key=lambda r: r.get("severity", 0), reverse=True
    )

    for rs in sorted_scores:
        severity = rs.get("severity", 0)
        technique = rs.get("technique_name", "Unknown")
        tactic = rs.get("tactic", "").replace("_", " ").title()
        evidence = rs.get("evidence", "")
        tech_id = rs.get("technique_id", "")

        # Color based on severity
        if severity >= 0.7:
            bar_color = "red"
            icon = "🔴"
        elif severity >= 0.3:
            bar_color = "orange"
            icon = "🟡"
        else:
            bar_color = "green"
            icon = "🟢"

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"{icon} **{technique}** `{tech_id}`")
        with col2:
            st.markdown(f":{bar_color}[{severity:.0%}]")
        with col3:
            st.caption(tactic)

        if evidence:
            st.caption(f"  {evidence[:300]}")

        citations = rs.get("citations", [])
        if citations:
            refs = []
            for c in citations:
                p_start = c.get("page_start", "?")
                section = c.get("section_title", "")
                refs.append(f"{section} (p.{p_start})" if section else f"p.{p_start}")
            st.caption("  Sources: " + " | ".join(refs))


def render_risk_summary(risk_scores: list[dict[str, Any]]) -> None:
    """Render a tactic-level summary of risk scores."""
    if not risk_scores:
        return

    tactic_counts = Counter(
        rs.get("tactic", "unknown").replace("_", " ").title()
        for rs in risk_scores
    )

    avg_severity = sum(rs.get("severity", 0) for rs in risk_scores) / len(risk_scores)
    max_severity = max(rs.get("severity", 0) for rs in risk_scores)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Risks", len(risk_scores))
    with col2:
        st.metric("Avg Severity", f"{avg_severity:.0%}")
    with col3:
        st.metric("Max Severity", f"{max_severity:.0%}")

    st.caption(
        "By tactic: "
        + ", ".join(f"{t} ({c})" for t, c in tactic_counts.most_common())
    )
