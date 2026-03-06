"""Documents page — manage PDFs and PageIndex trees.

Upload new PDFs, generate PageIndex trees, load pre-generated trees,
and view document structures.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def render_documents_page() -> None:
    """Render the document management page."""
    st.title("Document Manager")

    tab1, tab2 = st.tabs(["Load Trees", "Upload PDF"])

    with tab1:
        _render_load_trees()

    with tab2:
        _render_upload_pdf()

    # Show currently loaded documents
    st.divider()
    st.subheader("Loaded Documents")
    _render_loaded_documents()


def _render_load_trees() -> None:
    """Load pre-generated PageIndex tree JSON files."""
    st.subheader("Load Pre-Generated Trees")

    from finsight_mac.config import get_settings
    settings = get_settings()
    trees_dir = settings.trees_dir

    if not trees_dir.exists():
        st.warning(f"Trees directory not found: {trees_dir}")
        return

    # List available trees
    tree_files = list(trees_dir.glob("*.json"))

    if not tree_files:
        st.info(
            f"No tree files found in `{trees_dir}`. "
            "Generate one with: `python scripts/generate_tree.py --pdf <path>`"
        )
        return

    for tree_file in tree_files:
        name = tree_file.stem.replace("_structure", "")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(f"📄 {name}")
        with col2:
            if st.button("Load", key=f"load_{name}"):
                _load_tree_file(tree_file, name)

    # Also allow uploading a tree JSON directly
    st.divider()
    uploaded = st.file_uploader(
        "Or upload a PageIndex tree JSON",
        type=["json"],
        key="tree_upload",
    )
    if uploaded:
        try:
            data = json.loads(uploaded.read())
            name = uploaded.name.replace(".json", "").replace("_structure", "")
            if "loaded_trees" not in st.session_state:
                st.session_state.loaded_trees = {}
            st.session_state.loaded_trees[name] = data
            st.success(f"Loaded tree: {name}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")


def _render_upload_pdf() -> None:
    """Upload a PDF and optionally generate a PageIndex tree."""
    st.subheader("Upload PDF")
    st.info(
        "Upload a SEC filing PDF to generate a PageIndex tree. "
        "Requires an OpenAI API key for tree generation."
    )

    uploaded = st.file_uploader("Upload SEC Filing PDF", type=["pdf"], key="pdf_upload")

    if uploaded:
        from finsight_mac.config import get_settings
        settings = get_settings()

        # Save uploaded file
        save_path = settings.filings_dir / uploaded.name
        settings.filings_dir.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(uploaded.read())
        st.success(f"PDF saved to {save_path}")

        if settings.openai_api_key:
            if st.button("Generate PageIndex Tree"):
                with st.spinner("Generating tree (this may take a few minutes)..."):
                    try:
                        from finsight_mac.document.pipeline import DocumentPipeline
                        pipeline = DocumentPipeline()
                        tree = pipeline.generate_tree(str(save_path))
                        st.success(f"Tree generated! {tree.total_nodes} nodes")

                        if "loaded_trees" not in st.session_state:
                            st.session_state.loaded_trees = {}
                        name = save_path.stem
                        st.session_state.loaded_trees[name] = json.loads(
                            tree.model_dump_json()
                        )
                    except Exception as e:
                        st.error(f"Tree generation failed: {e}")
        else:
            st.warning(
                "Set `OPENAI_API_KEY` in `.env` to enable automatic tree generation. "
                "Alternatively, generate trees using the CLI: "
                "`python scripts/generate_tree.py --pdf <path>`"
            )


def _load_tree_file(path: Path, name: str) -> None:
    """Load a tree file into session state."""
    try:
        data = json.loads(path.read_text())
        if "loaded_trees" not in st.session_state:
            st.session_state.loaded_trees = {}
        st.session_state.loaded_trees[name] = data
        st.success(f"Loaded: {name}")
    except Exception as e:
        st.error(f"Failed to load {name}: {e}")


def _render_loaded_documents() -> None:
    """Display currently loaded documents with tree previews."""
    trees = st.session_state.get("loaded_trees", {})

    if not trees:
        st.caption("No documents loaded yet.")
        return

    for name, tree_data in trees.items():
        with st.expander(f"📄 {name}", expanded=False):
            from finsight_mac.ui.components.tree_viewer import render_tree
            render_tree(tree_data)

            if st.button("Unload", key=f"unload_{name}"):
                del st.session_state.loaded_trees[name]
                st.rerun()
