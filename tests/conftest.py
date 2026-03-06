"""Shared test fixtures for the FinSight test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_tree_path() -> Path:
    """Path to the sample PageIndex tree fixture."""
    return FIXTURES_DIR / "sample_tree.json"


@pytest.fixture
def sample_tree_dict() -> dict:
    """Sample tree as a raw dictionary."""
    path = FIXTURES_DIR / "sample_tree.json"
    return json.loads(path.read_text())


@pytest.fixture
def page_texts() -> dict[int, str]:
    """Simulated page texts for the sample tree."""
    return {
        1: "Cover Page. Apple Inc. Form 10-K. Filed October 2024.",
        2: "Part I. Item 1. Business.\nApple designs, manufactures, and markets...",
        3: "Products\niPhone is the company's flagship product line...",
        7: "Services\nApple Services includes the App Store, Apple Music...",
        10: "Competition\nApple faces intense competition across all segments...",
        13: "Item 1A. Risk Factors\nInvesting in Apple involves significant risks...",
        18: "Supply Chain Risks\nApple relies on single or limited sources for components...",
        22: "Legal and Regulatory Risks\nApple is subject to various legal proceedings...",
        37: "Revenue Analysis\nTotal net revenue was $383.3 billion...",
        43: "Operating Expenses\nCost of sales decreased 2% year over year...",
        54: "Balance Sheet\nTotal assets were $352.6 billion...",
    }
