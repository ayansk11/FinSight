"""Findings viewer component for Streamlit.

Renders analysis findings with confidence badges and citations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import streamlit as st

_CONFIDENCE_COLORS = {
    "high": "green",
    "medium": "orange",
    "low": "red",
}


def render_findings(findings: list[dict[str, Any]]) -> None:
    """Render findings grouped by agent with confidence badges."""
    if not findings:
        st.caption("No findings.")
        return

    # Group by agent
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        by_agent[f.get("agent", "unknown")].append(f)

    for agent, agent_findings in by_agent.items():
        label = agent.replace("_", " ").title()
        st.markdown(f"#### {label} ({len(agent_findings)})")

        for i, finding in enumerate(agent_findings):
            confidence = finding.get("confidence", "medium")
            color = _CONFIDENCE_COLORS.get(confidence, "gray")
            content = finding.get("content", "")

            st.markdown(
                f"**{i + 1}.** :{color}[{confidence}] — {content[:500]}"
            )

            citations = finding.get("citations", [])
            if citations:
                render_citations(citations)

            if i < len(agent_findings) - 1:
                st.divider()


def render_citations(citations: list[dict[str, Any]]) -> None:
    """Render citations as compact page references."""
    if not citations:
        return

    parts = []
    for c in citations:
        filing = c.get("filing_display_name", "")
        section = c.get("section_title", "")
        p_start = c.get("page_start", "?")
        p_end = c.get("page_end", "?")

        if p_start == p_end:
            ref = f"p.{p_start}"
        else:
            ref = f"p.{p_start}-{p_end}"

        label = f"{section} ({ref})" if section else ref
        if filing:
            label = f"{filing}: {label}"
        parts.append(label)

    st.caption("Sources: " + " | ".join(parts))
