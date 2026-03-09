"""Shared fixtures for end-to-end tests requiring Ollama."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _check_ollama() -> bool:
    """Check if Ollama is running and responsive."""
    try:
        from finsight_mac.llm.ollama_client import OllamaClient

        client = OllamaClient()
        return asyncio.get_event_loop().run_until_complete(client.check_health())
    except Exception:
        return False


# Session-scoped: check once per test session
_ollama_available: bool | None = None


def is_ollama_available() -> bool:
    global _ollama_available
    if _ollama_available is None:
        _ollama_available = _check_ollama()
    return _ollama_available


requires_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama is not running — skip e2e tests",
)


@pytest.fixture
def aapl_tree_data() -> dict:
    """Load the real AAPL 10-K 2024 tree."""
    path = DATA_DIR / "trees" / "AAPL_10K_2024_structure.json"
    if not path.exists():
        pytest.skip(f"Tree file not found: {path}")
    return json.loads(path.read_text())


@pytest.fixture
def aapl_pdf_path() -> Path:
    """Path to the real AAPL 10-K 2024 PDF."""
    path = DATA_DIR / "filings" / "AAPL_10K_2024.pdf"
    if not path.exists():
        pytest.skip(f"PDF not found: {path}")
    return path


@pytest.fixture
def workflow():
    """Get the compiled LangGraph workflow."""
    from finsight_mac.graph.workflow import build_workflow

    return build_workflow()


@pytest.fixture
def base_state(aapl_tree_data) -> dict:
    """Base FinSightState with AAPL tree loaded."""
    return {
        "query": "",
        "ticker": "AAPL",
        "filings": [{"display_name": aapl_tree_data["doc_name"], "pdf_path": None}],
        "trees": [aapl_tree_data],
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
        "model_name": "qwen3:8b",
        "total_duration_seconds": 0.0,
        "error": None,
    }
