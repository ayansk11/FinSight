"""FinSight Streamlit UI — main entry point.

Launch with: uv run streamlit run mac/src/finsight_mac/ui/app.py
"""

import streamlit as st

# Available Ollama models for the Mac platform.
MAC_MODELS: list[tuple[str, str]] = [
    ("Qwen3.5-9B (Default)", "qwen3.5:9b"),
    ("Qwen3-8B (RLM Fine-tuned)", "qwen3:8b-q4_K_M"),
    ("Qwen3.5-4B", "qwen3.5:4b"),
    ("Qwen3.5-2B", "qwen3.5:2b"),
    ("Qwen3.5-0.8B", "qwen3.5:0.8b"),
]

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
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "qwen3.5:9b"

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
    st.sidebar.caption("Built with LangGraph + PageIndex + Ollama")

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
    """Display model selector dropdown and Ollama connection status."""
    from finsight_mac.config import get_settings

    settings = get_settings()

    # Model selector dropdown
    labels = [label for label, _ in MAC_MODELS]
    model_ids = [mid for _, mid in MAC_MODELS]

    current_idx = 0
    if st.session_state.selected_model in model_ids:
        current_idx = model_ids.index(st.session_state.selected_model)

    selected_label = st.sidebar.selectbox(
        "Model",
        labels,
        index=current_idx,
    )
    st.session_state.selected_model = model_ids[labels.index(selected_label)]

    selected_model = st.session_state.selected_model
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
            if any(selected_model in n for n in model_names):
                st.sidebar.success("Ollama connected", icon="✅")
            else:
                st.sidebar.warning(
                    f"Model not found. Pull with: ollama pull {selected_model}",
                    icon="⚠️",
                )
        else:
            st.sidebar.error("Ollama not responding", icon="❌")
    except Exception:
        st.sidebar.error("Ollama not running", icon="❌")

    # Groq fallback status
    if settings.groq_api_key:
        st.sidebar.caption(f"Groq fallback: {settings.groq_model.split('/')[-1]}")
    else:
        st.sidebar.caption("Groq fallback: not configured")


if __name__ == "__main__":
    main()
