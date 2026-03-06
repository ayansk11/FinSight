import Foundation

/// Confidence levels for findings and risk assessments.
///
/// Matches Python `finsight_core.models.analysis.ConfidenceLevel`.
enum ConfidenceLevel: String, Codable, CaseIterable {
    case high
    case medium
    case low

    var displayName: String {
        rawValue.capitalized
    }

    var color: String {
        switch self {
        case .high: return "green"
        case .medium: return "orange"
        case .low: return "red"
        }
    }
}

/// A precise reference to a location in a source document.
///
/// Matches Python `finsight_core.models.analysis.Citation`.
struct Citation: Codable, Identifiable {
    let filingDisplayName: String
    let nodeId: String
    let sectionTitle: String
    let pageStart: Int
    let pageEnd: Int
    let excerpt: String?

    var id: String { "\(filingDisplayName)-\(nodeId)-\(pageStart)" }

    enum CodingKeys: String, CodingKey {
        case filingDisplayName = "filing_display_name"
        case nodeId = "node_id"
        case sectionTitle = "section_title"
        case pageStart = "page_start"
        case pageEnd = "page_end"
        case excerpt
    }

    /// Compact citation string for inline use.
    var shortRef: String {
        "[\(filingDisplayName), p.\(pageStart)]"
    }
}

/// A single analytical finding produced by an agent.
///
/// Matches Python `finsight_core.models.analysis.Finding`.
struct Finding: Codable, Identifiable {
    let content: String
    let agent: String
    let confidence: ConfidenceLevel
    let citations: [Citation]
    let metadata: [String: AnyCodable]?
    let timestamp: String?

    var id: String { "\(agent)-\(content.prefix(50))" }

    /// Agent display name.
    var agentDisplayName: String {
        switch agent {
        case "doc_intel": return "Document Intelligence"
        case "quant": return "Quantitative Analysis"
        case "risk": return "Risk Classification"
        case "synthesis": return "Synthesis"
        default: return agent
        }
    }
}

/// Top-level MITRE F3-inspired risk tactic categories.
///
/// Matches Python `finsight_core.models.analysis.RiskTactic`.
enum RiskTactic: String, Codable, CaseIterable {
    case disclosureRisk = "disclosure_risk"
    case financialHealth = "financial_health"
    case governance
    case marketRisk = "market_risk"

    var displayName: String {
        switch self {
        case .disclosureRisk: return "Disclosure Risk"
        case .financialHealth: return "Financial Health"
        case .governance: return "Governance"
        case .marketRisk: return "Market Risk"
        }
    }
}

/// A risk classification score from the Risk Classification Agent.
///
/// Matches Python `finsight_core.models.analysis.RiskScore`.
struct RiskScore: Codable, Identifiable {
    let tactic: RiskTactic
    let techniqueId: String
    let techniqueName: String
    let severity: Double
    let confidence: ConfidenceLevel
    let evidence: String
    let citations: [Citation]

    var id: String { techniqueId }

    enum CodingKeys: String, CodingKey {
        case tactic
        case techniqueId = "technique_id"
        case techniqueName = "technique_name"
        case severity
        case confidence
        case evidence
        case citations
    }

    /// Severity as a percentage string.
    var severityPercent: String {
        String(format: "%.0f%%", severity * 100)
    }

    /// Severity level description.
    var severityLevel: String {
        switch severity {
        case 0..<0.3: return "Low"
        case 0.3..<0.6: return "Medium"
        case 0.6..<0.8: return "High"
        default: return "Critical"
        }
    }
}

// MARK: - AnyCodable helper

/// Type-erased Codable wrapper for heterogeneous metadata dictionaries.
struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let string = try? container.decode(String.self) {
            value = string
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let string = value as? String {
            try container.encode(string)
        } else if let int = value as? Int {
            try container.encode(int)
        } else if let double = value as? Double {
            try container.encode(double)
        } else if let bool = value as? Bool {
            try container.encode(bool)
        } else {
            try container.encodeNil()
        }
    }
}
