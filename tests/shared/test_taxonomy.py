"""Tests for the F3-inspired financial risk taxonomy."""

from __future__ import annotations

from finsight_core.models.analysis import RiskTactic
from finsight_core.taxonomy.f3_taxonomy import (
    ALL_TECHNIQUES,
    DISCLOSURE_TECHNIQUES,
    FINANCIAL_HEALTH_TECHNIQUES,
    GOVERNANCE_TECHNIQUES,
    MARKET_RISK_TECHNIQUES,
    TECHNIQUE_BY_ID,
    TECHNIQUES_BY_TACTIC,
    get_taxonomy_prompt,
)


class TestTaxonomy:
    def test_all_techniques_populated(self) -> None:
        assert len(ALL_TECHNIQUES) == 14

    def test_technique_by_id_lookup(self) -> None:
        tech = TECHNIQUE_BY_ID.get("DR-T001")
        assert tech is not None
        assert tech.name == "Material omission in risk factors"
        assert tech.tactic == RiskTactic.DISCLOSURE_RISK

    def test_technique_by_id_all_exist(self) -> None:
        for tech in ALL_TECHNIQUES:
            assert tech.technique_id in TECHNIQUE_BY_ID

    def test_techniques_by_tactic_complete(self) -> None:
        for tactic in RiskTactic:
            assert tactic in TECHNIQUES_BY_TACTIC
            assert len(TECHNIQUES_BY_TACTIC[tactic]) > 0

    def test_disclosure_techniques_count(self) -> None:
        assert len(DISCLOSURE_TECHNIQUES) == 4

    def test_financial_health_techniques_count(self) -> None:
        assert len(FINANCIAL_HEALTH_TECHNIQUES) == 4

    def test_governance_techniques_count(self) -> None:
        assert len(GOVERNANCE_TECHNIQUES) == 3

    def test_market_risk_techniques_count(self) -> None:
        assert len(MARKET_RISK_TECHNIQUES) == 3

    def test_severity_weights_valid(self) -> None:
        for tech in ALL_TECHNIQUES:
            assert 0.0 <= tech.severity_weight <= 1.0, (
                f"{tech.technique_id} has invalid severity_weight: {tech.severity_weight}"
            )

    def test_all_techniques_have_indicators(self) -> None:
        for tech in ALL_TECHNIQUES:
            assert len(tech.indicators) > 0, f"{tech.technique_id} has no indicators"

    def test_all_techniques_have_sec_sections(self) -> None:
        for tech in ALL_TECHNIQUES:
            assert len(tech.sec_sections) > 0, f"{tech.technique_id} has no SEC sections"

    def test_unique_technique_ids(self) -> None:
        ids = [t.technique_id for t in ALL_TECHNIQUES]
        assert len(ids) == len(set(ids)), "Duplicate technique IDs found"

    def test_technique_id_format(self) -> None:
        """IDs should follow the pattern XX-TXXX."""
        import re

        pattern = re.compile(r"^[A-Z]{2}-T\d{3}$")
        for tech in ALL_TECHNIQUES:
            assert pattern.match(tech.technique_id), (
                f"Invalid technique ID format: {tech.technique_id}"
            )

    def test_get_taxonomy_prompt(self) -> None:
        prompt = get_taxonomy_prompt()
        assert "FinSight Financial Risk Taxonomy" in prompt
        assert "DR-T001" in prompt
        assert "FH-T001" in prompt
        assert "GV-T001" in prompt
        assert "MR-T001" in prompt
        assert len(prompt) > 500  # Should be substantial
