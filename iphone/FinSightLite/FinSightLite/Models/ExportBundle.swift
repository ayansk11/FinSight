import Foundation

/// SEC filing types supported by FinSight.
///
/// Matches Python `finsight_core.models.document.FilingType`.
enum FilingType: String, Codable {
    case form10K = "10-K"
    case form10Q = "10-Q"
    case form8K = "8-K"
    case formDEF14A = "DEF 14A"
    case earningsCall = "earnings_call"
    case other
}

/// Represents a single SEC filing or financial document.
///
/// Matches Python `finsight_core.models.document.Filing`.
struct Filing: Codable {
    let ticker: String
    let companyName: String
    let filingType: FilingType
    let filingDate: String
    let fiscalYear: Int?
    let fiscalQuarter: Int?

    enum CodingKeys: String, CodingKey {
        case ticker
        case companyName = "company_name"
        case filingType = "filing_type"
        case filingDate = "filing_date"
        case fiscalYear = "fiscal_year"
        case fiscalQuarter = "fiscal_quarter"
    }

    /// Human-readable name for this filing.
    var displayName: String {
        let q = fiscalQuarter.map { " Q\($0)" } ?? ""
        let fy = fiscalYear.map { " FY\($0)" } ?? ""
        return "\(ticker) \(filingType.rawValue)\(fy)\(q)"
    }
}

/// A complete analysis bundle imported from the Mac app.
///
/// Matches Python `finsight_core.models.export.ExportBundle`.
/// This is the primary data format for Mac → iPhone transfer.
struct ExportBundle: Codable, Identifiable {
    let version: String
    let createdAt: String
    let modelUsed: String
    let filing: Filing
    let tree: PageIndexTreeModel
    let findings: [Finding]
    let riskScores: [RiskScore]
    let executiveSummary: String?
    let fullReport: String?

    var id: String { "\(filing.ticker)-\(createdAt)" }

    enum CodingKeys: String, CodingKey {
        case version
        case createdAt = "created_at"
        case modelUsed = "model_used"
        case filing
        case tree
        case findings
        case riskScores = "risk_scores"
        case executiveSummary = "executive_summary"
        case fullReport = "full_report"
    }

    /// Display name combining filing info and date.
    var displayName: String {
        "\(filing.displayName)"
    }

    /// Total number of findings.
    var findingCount: Int { findings.count }

    /// Total number of risk scores.
    var riskScoreCount: Int { riskScores.count }

    /// Average risk severity across all scores.
    var averageSeverity: Double? {
        guard !riskScores.isEmpty else { return nil }
        return riskScores.reduce(0.0) { $0 + $1.severity } / Double(riskScores.count)
    }

    /// Findings grouped by agent.
    var findingsByAgent: [String: [Finding]] {
        Dictionary(grouping: findings, by: { $0.agent })
    }

    /// Risk scores grouped by tactic.
    var riskScoresByTactic: [RiskTactic: [RiskScore]] {
        Dictionary(grouping: riskScores, by: { $0.tactic })
    }

    // MARK: - File Operations

    /// Load a bundle from a .finsight JSON file.
    static func load(from url: URL) throws -> ExportBundle {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(ExportBundle.self, from: data)
    }

    /// Save this bundle to a .finsight JSON file.
    func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url)
    }
}
