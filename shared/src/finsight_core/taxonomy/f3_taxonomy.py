"""MITRE F3-inspired Financial Risk Taxonomy.

Defines the structured risk classification system used by the
Risk Classification Agent. Based on MITRE's Fight Financial Fraud (F3)
framework announced May 2025, adapted for SEC filing analysis.

Taxonomy Structure (ATT&CK-style):
- Tactics: High-level risk categories (4 total)
- Techniques: Specific risk patterns within each tactic
- Each technique has indicators (linguistic/structural patterns to detect)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from finsight_core.models.analysis import RiskTactic


@dataclass
class RiskTechnique:
    """A specific risk pattern within a tactic category."""

    technique_id: str  # e.g., "DR-T001"
    name: str  # e.g., "Material omission in risk factors"
    description: str
    tactic: RiskTactic
    indicators: list[str] = field(default_factory=list)  # What to look for
    sec_sections: list[str] = field(default_factory=list)  # Where to look
    severity_weight: float = 0.5  # Base severity (0.0-1.0)


# ============================================================
# Tactic 1: Disclosure Risk
# Patterns indicating inadequate, misleading, or evasive disclosure
# ============================================================

DISCLOSURE_TECHNIQUES = [
    RiskTechnique(
        technique_id="DR-T001",
        name="Material omission in risk factors",
        description="Key risk factors mentioned in industry reports or peer filings are absent",
        tactic=RiskTactic.DISCLOSURE_RISK,
        indicators=[
            "Risk factor section significantly shorter than peers",
            "Known industry risks not mentioned",
            "Prior year risks removed without explanation",
        ],
        sec_sections=["Item 1A - Risk Factors"],
        severity_weight=0.8,
    ),
    RiskTechnique(
        technique_id="DR-T002",
        name="Boilerplate risk language",
        description="Risk factors use generic language without company-specific detail",
        tactic=RiskTactic.DISCLOSURE_RISK,
        indicators=[
            "Identical risk language across multiple years",
            "No quantitative thresholds or specific scenarios",
            "Generic phrases like 'may adversely affect'",
        ],
        sec_sections=["Item 1A - Risk Factors"],
        severity_weight=0.4,
    ),
    RiskTechnique(
        technique_id="DR-T003",
        name="Inconsistent narrative across sections",
        description="MD&A narrative contradicts or omits information from financial statements",
        tactic=RiskTactic.DISCLOSURE_RISK,
        indicators=[
            "MD&A tone contradicts financial trends",
            "Revenue discussion omits known headwinds",
            "Forward-looking statements ignore recent negative developments",
        ],
        sec_sections=["Item 7 - MD&A", "Item 8 - Financial Statements"],
        severity_weight=0.7,
    ),
    RiskTechnique(
        technique_id="DR-T004",
        name="Obfuscation through complexity",
        description="Unnecessarily complex language or structure obscures material information",
        tactic=RiskTactic.DISCLOSURE_RISK,
        indicators=[
            "Readability score significantly worse than peers",
            "Excessive use of cross-references",
            "Critical details buried in footnotes",
        ],
        sec_sections=["Item 7 - MD&A", "Item 8 - Financial Statements"],
        severity_weight=0.5,
    ),
]

# ============================================================
# Tactic 2: Financial Health
# Patterns indicating deteriorating or manipulated financial condition
# ============================================================

FINANCIAL_HEALTH_TECHNIQUES = [
    RiskTechnique(
        technique_id="FH-T001",
        name="Revenue recognition irregularities",
        description="Revenue patterns suggesting aggressive recognition practices",
        tactic=RiskTactic.FINANCIAL_HEALTH,
        indicators=[
            "Revenue growth without corresponding cash flow growth",
            "Increasing accounts receivable relative to revenue",
            "Significant revenue from related parties",
            "Quarter-end revenue spikes",
        ],
        sec_sections=["Item 8 - Financial Statements", "Revenue recognition notes"],
        severity_weight=0.9,
    ),
    RiskTechnique(
        technique_id="FH-T002",
        name="Expense deferral patterns",
        description="Evidence of deferring expenses to inflate current-period earnings",
        tactic=RiskTactic.FINANCIAL_HEALTH,
        indicators=[
            "Capitalization policy changes",
            "Growing deferred costs relative to revenue",
            "Unusual asset impairment timing",
        ],
        sec_sections=["Item 8 - Financial Statements"],
        severity_weight=0.7,
    ),
    RiskTechnique(
        technique_id="FH-T003",
        name="Off-balance-sheet exposure",
        description="Material obligations not reflected on the balance sheet",
        tactic=RiskTactic.FINANCIAL_HEALTH,
        indicators=[
            "Variable interest entities (VIEs)",
            "Operating lease obligations (pre-ASC 842 or complex)",
            "Unconsolidated joint ventures with significant activity",
            "Guarantees or commitments in footnotes",
        ],
        sec_sections=["Item 7 - MD&A", "Commitment and contingency notes"],
        severity_weight=0.8,
    ),
    RiskTechnique(
        technique_id="FH-T004",
        name="Going concern indicators",
        description="Signs the company may face viability challenges",
        tactic=RiskTactic.FINANCIAL_HEALTH,
        indicators=[
            "Negative working capital trend",
            "Covenant violation or waiver",
            "Auditor going concern language",
            "Debt maturity wall approaching",
        ],
        sec_sections=["Item 7 - MD&A", "Auditor report", "Debt notes"],
        severity_weight=1.0,
    ),
]

# ============================================================
# Tactic 3: Governance
# Patterns indicating governance weaknesses or conflicts
# ============================================================

GOVERNANCE_TECHNIQUES = [
    RiskTechnique(
        technique_id="GV-T001",
        name="Related-party transactions",
        description="Material transactions with insiders or affiliates",
        tactic=RiskTactic.GOVERNANCE,
        indicators=[
            "Transactions with officers, directors, or their families",
            "Entities controlled by major shareholders",
            "Below-market or above-market pricing",
        ],
        sec_sections=["Item 13 - Related Party Transactions"],
        severity_weight=0.7,
    ),
    RiskTechnique(
        technique_id="GV-T002",
        name="Executive compensation misalignment",
        description="Compensation structures not tied to shareholder value",
        tactic=RiskTactic.GOVERNANCE,
        indicators=[
            "Large discretionary bonuses despite poor performance",
            "Short-term incentives dominating compensation",
            "Repriced stock options",
            "Golden parachutes exceeding 3× base salary",
        ],
        sec_sections=["DEF 14A - Executive Compensation"],
        severity_weight=0.5,
    ),
    RiskTechnique(
        technique_id="GV-T003",
        name="Auditor changes or qualifications",
        description="Unusual changes in external auditors or audit opinions",
        tactic=RiskTactic.GOVERNANCE,
        indicators=[
            "Auditor change without clear explanation",
            "Material weakness disclosures",
            "Qualified audit opinion",
            "Disagreements with auditors",
        ],
        sec_sections=["Item 9A - Controls and Procedures", "Auditor report"],
        severity_weight=0.9,
    ),
]

# ============================================================
# Tactic 4: Market Risk
# Patterns indicating external risk exposure
# ============================================================

MARKET_RISK_TECHNIQUES = [
    RiskTechnique(
        technique_id="MR-T001",
        name="Customer/supplier concentration",
        description="Excessive dependence on a small number of customers or suppliers",
        tactic=RiskTactic.MARKET_RISK,
        indicators=[
            "Single customer > 10% of revenue",
            "Single supplier for critical inputs",
            "Geographic concentration > 40%",
        ],
        sec_sections=["Item 1 - Business", "Item 1A - Risk Factors"],
        severity_weight=0.6,
    ),
    RiskTechnique(
        technique_id="MR-T002",
        name="Regulatory exposure escalation",
        description="Increasing regulatory risk or compliance burden",
        tactic=RiskTactic.MARKET_RISK,
        indicators=[
            "New regulatory investigations mentioned",
            "Compliance cost increases",
            "Industry-wide regulatory changes",
            "Cross-border regulatory conflicts",
        ],
        sec_sections=["Item 1A - Risk Factors", "Legal proceedings"],
        severity_weight=0.6,
    ),
    RiskTechnique(
        technique_id="MR-T003",
        name="Litigation trajectory",
        description="Legal exposure trending upward or with material contingencies",
        tactic=RiskTactic.MARKET_RISK,
        indicators=[
            "Increasing litigation reserves",
            "Class action lawsuits filed",
            "Government enforcement actions",
            "Patent infringement claims on core products",
        ],
        sec_sections=["Item 3 - Legal Proceedings", "Contingency notes"],
        severity_weight=0.7,
    ),
]

# ============================================================
# Complete taxonomy registry
# ============================================================

ALL_TECHNIQUES: list[RiskTechnique] = (
    DISCLOSURE_TECHNIQUES
    + FINANCIAL_HEALTH_TECHNIQUES
    + GOVERNANCE_TECHNIQUES
    + MARKET_RISK_TECHNIQUES
)

TECHNIQUE_BY_ID: dict[str, RiskTechnique] = {t.technique_id: t for t in ALL_TECHNIQUES}

TECHNIQUES_BY_TACTIC: dict[RiskTactic, list[RiskTechnique]] = {
    RiskTactic.DISCLOSURE_RISK: DISCLOSURE_TECHNIQUES,
    RiskTactic.FINANCIAL_HEALTH: FINANCIAL_HEALTH_TECHNIQUES,
    RiskTactic.GOVERNANCE: GOVERNANCE_TECHNIQUES,
    RiskTactic.MARKET_RISK: MARKET_RISK_TECHNIQUES,
}


def get_taxonomy_prompt() -> str:
    """Generate a text description of the full taxonomy for LLM context.

    Used by the Risk Classification Agent to understand what patterns to look for.
    """
    lines = ["# FinSight Financial Risk Taxonomy (F3-Inspired)", ""]

    for tactic in RiskTactic:
        tactic_name = tactic.value.replace("_", " ").title()
        techniques = TECHNIQUES_BY_TACTIC.get(tactic, [])
        lines.append(f"## {tactic_name}")
        lines.append("")

        for tech in techniques:
            lines.append(f"### [{tech.technique_id}] {tech.name}")
            lines.append(f"Description: {tech.description}")
            lines.append(f"Severity weight: {tech.severity_weight}")
            lines.append(f"Look in: {', '.join(tech.sec_sections)}")
            lines.append("Indicators:")
            for indicator in tech.indicators:
                lines.append(f"  - {indicator}")
            lines.append("")

    return "\n".join(lines)
