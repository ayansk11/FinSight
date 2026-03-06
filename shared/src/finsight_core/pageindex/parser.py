"""PageIndex JSON tree parser and validator.

Loads PageIndex tree structures from JSON files or dictionaries
and validates them against the TreeNode/PageIndexTree models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finsight_core.models.document import PageIndexTree


def load_tree(path: str | Path) -> PageIndexTree:
    """Load a PageIndex tree from a JSON file.

    Args:
        path: Path to the PageIndex JSON file (output of `pageindex.page_index()`).

    Returns:
        Validated PageIndexTree model.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        pydantic.ValidationError: If the JSON doesn't match the expected schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PageIndex tree file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    return load_tree_from_dict(raw)


def load_tree_from_dict(data: dict[str, Any]) -> PageIndexTree:
    """Load a PageIndex tree from a dictionary.

    Handles both the standard PageIndex output format and variations:
    - Standard: {"doc_name": "...", "structure": [...]}
    - Minimal: {"structure": [...]} (doc_name defaults to "unknown")

    Args:
        data: Dictionary matching PageIndex output format.

    Returns:
        Validated PageIndexTree model.
    """
    # Handle missing doc_name gracefully
    if "doc_name" not in data:
        data = {**data, "doc_name": "unknown"}

    return PageIndexTree.model_validate(data)


def tree_to_json(tree: PageIndexTree, path: str | Path) -> None:
    """Save a PageIndexTree to a JSON file.

    Args:
        tree: The tree to serialize.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")
