"""Analysis page — view findings, risk scores, and reports.

Displays results from the last workflow run stored in
``st.session_state.last_result``.
"""

from __future__ import annotations

from datetime import date

import streamlit as st


def render_analysis_page() -> None:
    """Render the analysis dashboard."""
    st.title("Analysis Dashboard")

    result = st.session_state.get("last_result")

    if not result:
        st.info(
            "No analysis results yet. Run a query from the **Chat** page "
            "to generate analysis results."
        )
        return

    query = st.session_state.get("last_query", "")
    duration = st.session_state.get("last_duration", 0)

    # Header metrics
    findings = result.get("findings", [])
    risk_scores = result.get("risk_scores", [])
    route = result.get("route", "unknown")
    summaries = result.get("agent_summaries", [])

    st.caption(f'Query: "{query}"')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Route", route.replace("_", " ").title())
    with col2:
        st.metric("Findings", len(findings))
    with col3:
        st.metric("Risk Scores", len(risk_scores))
    with col4:
        st.metric("Duration", f"{duration:.1f}s")

    st.divider()

    # Tabs
    tabs = st.tabs(["Executive Summary", "Findings", "Risk Scores", "Full Report", "Export"])

    # --- Executive Summary ---
    with tabs[0]:
        exec_summary = result.get("executive_summary", "")
        if exec_summary:
            st.markdown(exec_summary)
        else:
            st.caption("No executive summary generated.")

        if summaries:
            st.subheader("Agent Summaries")
            for s in summaries:
                agent = s.get("agent", "?").replace("_", " ").title()
                summary = s.get("summary", "")
                dur = s.get("duration", 0)
                st.markdown(f"**{agent}** ({dur:.1f}s) — {summary}")

    # --- Findings ---
    with tabs[1]:
        from finsight_mac.ui.components.findings_viewer import render_findings

        render_findings(findings)

    # --- Risk Scores ---
    with tabs[2]:
        from finsight_mac.ui.components.risk_viewer import (
            render_risk_summary,
            render_risk_table,
        )

        if risk_scores:
            render_risk_summary(risk_scores)
            st.divider()
            render_risk_table(risk_scores)
        else:
            st.caption("No risk scores. Run a risk-focused query to generate them.")

    # --- Full Report ---
    with tabs[3]:
        report = result.get("report", "")
        if report:
            st.markdown(report)
        else:
            st.caption("No full report generated.")

    # --- Export ---
    with tabs[4]:
        _render_export(result)


def _render_export(result: dict) -> None:
    """Render export controls for .finsight bundles."""
    st.subheader("Export to .finsight Bundle")
    st.caption("Generate a `.finsight` JSON bundle for the iPhone companion app.")

    trees = st.session_state.get("loaded_trees", {})
    if not trees:
        st.warning("No documents loaded — cannot export.")
        return

    tree_name = list(trees.keys())[0]
    tree_data = trees[tree_name]

    if st.button("Generate Export Bundle"):
        try:
            from finsight_core.models.analysis import Finding, RiskScore
            from finsight_core.models.document import Filing, FilingType
            from finsight_core.models.export import ExportBundle
            from finsight_core.pageindex.parser import load_tree_from_dict

            tree = load_tree_from_dict(tree_data)

            # Detect filing type from name
            parts = tree_name.upper().split("_")
            ticker = parts[0] if parts else "UNKNOWN"
            filing_type = FilingType.OTHER
            for part in parts:
                if part in ("10K", "10-K"):
                    filing_type = FilingType.FORM_10K
                elif part in ("10Q", "10-Q"):
                    filing_type = FilingType.FORM_10Q

            filing = Filing(
                ticker=ticker,
                company_name=ticker,
                filing_type=filing_type,
                filing_date=date.today(),
            )

            findings_objs = [Finding.model_validate(f) for f in result.get("findings", [])]
            risk_objs = [RiskScore.model_validate(r) for r in result.get("risk_scores", [])]

            bundle = ExportBundle(
                model_used=result.get("model_name", "unknown"),
                filing=filing,
                tree=tree,
                findings=findings_objs,
                risk_scores=risk_objs,
                executive_summary=result.get("executive_summary"),
                full_report=result.get("report"),
            )

            bundle_json = bundle.model_dump_json(indent=2)

            st.download_button(
                label="Download .finsight Bundle",
                data=bundle_json,
                file_name=f"{tree_name}.finsight",
                mime="application/json",
            )
            st.success(f"Bundle ready: {len(bundle_json) / 1024:.1f} KB")

        except Exception as e:
            st.error(f"Export failed: {e}")
