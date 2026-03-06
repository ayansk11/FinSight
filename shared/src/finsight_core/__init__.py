"""FinSight Core — shared data models, PageIndex navigation, and financial utilities.

This package has ZERO LLM dependencies. It provides:
- Pydantic data models for the entire FinSight system
- PageIndex tree parsing and navigation
- Financial ratio calculations
- Prompt templates for all agents
- MITRE F3-inspired risk taxonomy definitions
"""

from finsight_core.models.analysis import (
    AgentOutput,
    Citation,
    Finding,
    RiskScore,
)
from finsight_core.models.document import (
    Filing,
    FilingType,
    PageIndexTree,
    TreeNode,
)
from finsight_core.models.state import FinSightState

__all__ = [
    "Filing",
    "FilingType",
    "PageIndexTree",
    "TreeNode",
    "Citation",
    "Finding",
    "RiskScore",
    "AgentOutput",
    "FinSightState",
]
