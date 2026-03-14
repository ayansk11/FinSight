"""Tests for finsight_core data models."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
from finsight_core.models.analysis import (
    AgentOutput,
    Citation,
    ConfidenceLevel,
    Finding,
    RiskScore,
    RiskTactic,
)
from finsight_core.models.document import Filing, FilingType, PageIndexTree, TreeNode
from finsight_core.models.export import ExportBundle

# ============================================================
# TreeNode tests
# ============================================================


class TestTreeNode:
    def test_leaf_node(self) -> None:
        node = TreeNode(
            title="Products",
            node_id="0004",
            start_index=3,
            end_index=6,
            nodes=[],
        )
        assert node.is_leaf is True
        assert node.page_count == 4

    def test_parent_node(self) -> None:
        child = TreeNode(
            title="Products",
            node_id="0004",
            start_index=3,
            end_index=6,
            nodes=[],
        )
        parent = TreeNode(
            title="Item 1. Business",
            node_id="0003",
            start_index=2,
            end_index=12,
            nodes=[child],
        )
        assert parent.is_leaf is False
        assert parent.page_count == 11

    def test_find_by_id(self) -> None:
        child = TreeNode(
            title="Products",
            node_id="0004",
            start_index=3,
            end_index=6,
            nodes=[],
        )
        parent = TreeNode(
            title="Item 1",
            node_id="0003",
            start_index=2,
            end_index=12,
            nodes=[child],
        )
        assert parent.find_by_id("0004") == child
        assert parent.find_by_id("0003") == parent
        assert parent.find_by_id("9999") is None

    def test_all_nodes_flat(self) -> None:
        grandchild = TreeNode(
            title="iPhone",
            node_id="0005",
            start_index=3,
            end_index=4,
            nodes=[],
        )
        child = TreeNode(
            title="Products",
            node_id="0004",
            start_index=3,
            end_index=6,
            nodes=[grandchild],
        )
        parent = TreeNode(
            title="Item 1",
            node_id="0003",
            start_index=2,
            end_index=12,
            nodes=[child],
        )
        flat = parent.all_nodes_flat()
        assert len(flat) == 3
        assert flat[0].node_id == "0003"
        assert flat[1].node_id == "0004"
        assert flat[2].node_id == "0005"


# ============================================================
# PageIndexTree tests
# ============================================================


class TestPageIndexTree:
    def test_from_fixture(self, sample_tree_dict: dict) -> None:
        tree = PageIndexTree.model_validate(sample_tree_dict)
        assert tree.doc_name == "AAPL_10K_2024"
        assert tree.total_nodes > 0

    def test_find_by_id(self, sample_tree_dict: dict) -> None:
        tree = PageIndexTree.model_validate(sample_tree_dict)
        node = tree.find_by_id("0007")
        assert node is not None
        assert "Risk Factors" in node.title

    def test_find_by_id_not_found(self, sample_tree_dict: dict) -> None:
        tree = PageIndexTree.model_validate(sample_tree_dict)
        assert tree.find_by_id("9999") is None

    def test_leaf_nodes(self, sample_tree_dict: dict) -> None:
        tree = PageIndexTree.model_validate(sample_tree_dict)
        leaves = tree.leaf_nodes
        assert len(leaves) > 0
        assert all(n.is_leaf for n in leaves)

    def test_get_all_nodes(self, sample_tree_dict: dict) -> None:
        tree = PageIndexTree.model_validate(sample_tree_dict)
        all_nodes = tree.get_all_nodes()
        assert len(all_nodes) == tree.total_nodes

    def test_tree_outline(self, sample_tree_dict: dict) -> None:
        tree = PageIndexTree.model_validate(sample_tree_dict)
        outline = tree.get_tree_outline(max_depth=2)
        assert "AAPL_10K_2024" in outline
        assert "[0001]" in outline
        assert "Cover Page" in outline


# ============================================================
# Filing tests
# ============================================================


class TestFiling:
    def test_display_name(self) -> None:
        filing = Filing(
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type=FilingType.FORM_10K,
            filing_date=date(2024, 10, 31),
            fiscal_year=2024,
        )
        assert "AAPL" in filing.display_name
        assert "10-K" in filing.display_name
        assert "FY2024" in filing.display_name


# ============================================================
# Analysis model tests
# ============================================================


class TestAnalysisModels:
    def test_citation_short_ref(self) -> None:
        citation = Citation(
            filing_display_name="AAPL 10-K FY2024",
            node_id="0007",
            section_title="Risk Factors",
            page_start=13,
            page_end=28,
        )
        assert "AAPL 10-K FY2024" in citation.short_ref
        assert "p.13" in citation.short_ref

    def test_finding_creation(self) -> None:
        finding = Finding(
            content="Revenue increased 5% YoY",
            agent="doc_intel",
            confidence=ConfidenceLevel.HIGH,
        )
        assert finding.agent == "doc_intel"
        assert finding.confidence == ConfidenceLevel.HIGH
        assert isinstance(finding.timestamp, datetime)

    def test_risk_score_bounds(self) -> None:
        score = RiskScore(
            tactic=RiskTactic.DISCLOSURE_RISK,
            technique_id="DR-T001",
            technique_name="Material omission",
            severity=0.8,
            confidence=ConfidenceLevel.MEDIUM,
            evidence="Risk section shorter than peers",
        )
        assert 0 <= score.severity <= 1

    def test_risk_score_invalid_severity(self) -> None:
        with pytest.raises(Exception):
            RiskScore(
                tactic=RiskTactic.DISCLOSURE_RISK,
                technique_id="DR-T001",
                technique_name="Test",
                severity=1.5,  # Invalid: > 1.0
                confidence=ConfidenceLevel.LOW,
                evidence="test",
            )

    def test_agent_output(self) -> None:
        output = AgentOutput(
            agent_name="doc_intel",
            findings=[
                Finding(
                    content="Test finding",
                    agent="doc_intel",
                    confidence=ConfidenceLevel.HIGH,
                    citations=[
                        Citation(
                            filing_display_name="AAPL 10-K",
                            node_id="0003",
                            section_title="Business",
                            page_start=2,
                            page_end=12,
                        )
                    ],
                )
            ],
            summary="Test summary",
        )
        assert output.succeeded is True
        assert output.citation_count == 1

    def test_agent_output_with_error(self) -> None:
        output = AgentOutput(
            agent_name="quant",
            error="Model unavailable",
        )
        assert output.succeeded is False


# ============================================================
# ExportBundle tests
# ============================================================


class TestExportBundle:
    def test_roundtrip_json(self, tmp_path: Path, sample_tree_dict: dict) -> None:
        tree = PageIndexTree.model_validate(sample_tree_dict)
        filing = Filing(
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type=FilingType.FORM_10K,
            filing_date=date(2024, 10, 31),
        )

        bundle = ExportBundle(
            model_used="qwen3.5:9b",
            filing=filing,
            tree=tree,
            findings=[],
            risk_scores=[],
            executive_summary="Test summary",
        )

        output_path = str(tmp_path / "test.finsight")
        bundle.to_json_file(output_path)

        loaded = ExportBundle.from_json_file(output_path)
        assert loaded.filing.ticker == "AAPL"
        assert loaded.tree.doc_name == "AAPL_10K_2024"
        assert loaded.executive_summary == "Test summary"
        assert loaded.model_used == "qwen3.5:9b"
