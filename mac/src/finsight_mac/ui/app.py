"""FinSight Streamlit UI — main entry point.

Launch with: uv run streamlit run mac/src/finsight_mac/ui/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="FinSight",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main Streamlit application."""

    # Initialize session state
    if "loaded_trees" not in st.session_state:
        st.session_state.loaded_trees = {}
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "ticker" not in st.session_state:
        st.session_state.ticker = ""

    # Sidebar navigation
    st.sidebar.title("FinSight")
    st.sidebar.caption("Agentic Financial Document Intelligence")

    page = st.sidebar.radio(
        "Navigate",
        ["Chat", "Documents", "Analysis"],
        index=0,
    )

    st.sidebar.divider()

    # Ticker input
    st.sidebar.text_input(
        "Ticker Override",
        key="ticker",
        placeholder="e.g. AAPL",
        help="Auto-detected from document name if left blank.",
    )

    # Loaded documents count
    doc_count = len(st.session_state.loaded_trees)
    if doc_count:
        st.sidebar.caption(f"Documents loaded: {doc_count}")

    st.sidebar.divider()

    # Model status indicator
    st.sidebar.subheader("System Status")
    _show_model_status()

    st.sidebar.divider()
    st.sidebar.caption("Built with LangGraph + PageIndex + Qwen3.5")

    # Route to selected page
    if page == "Chat":
        from finsight_mac.ui.pages.chat import render_chat_page
        render_chat_page()
    elif page == "Documents":
        from finsight_mac.ui.pages.documents import render_documents_page
        render_documents_page()
    elif page == "Analysis":
        from finsight_mac.ui.pages.analysis import render_analysis_page
        render_analysis_page()


def _show_model_status() -> None:
    """Display current model and Ollama connection status."""
    from finsight_mac.config import get_settings

    settings = get_settings()
    st.sidebar.text(f"Model: {settings.ollama_model}")
    st.sidebar.text(f"Ollama: {settings.ollama_base_url}")

    # Quick health check
    try:
        import httpx
        response = httpx.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=2,
        )
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if any(settings.ollama_model in n for n in model_names):
                st.sidebar.success("Connected", icon="✅")
            else:
                st.sidebar.warning(
                    f"Model not found. Available: {', '.join(model_names[:3])}",
                    icon="⚠️",
                )
        else:
            st.sidebar.error("Ollama not responding", icon="❌")
    except Exception:
        st.sidebar.error("Ollama not running", icon="❌")


if __name__ == "__main__":
    main()
