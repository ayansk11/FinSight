import SwiftUI

/// Analysis summary view — display pre-computed findings and risk scores.
///
/// Shows analysis results from imported .finsight bundles, including
/// executive summaries, findings by agent, and risk heat maps.
struct AnalysisSummaryView: View {
    @EnvironmentObject var documentStore: DocumentStore
    @State private var selectedBundle: ExportBundle?

    var body: some View {
        NavigationStack {
            Group {
                if documentStore.bundles.isEmpty {
                    emptyState
                } else {
                    bundleAnalysisView
                }
            }
            .navigationTitle("Analysis")
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 48))
                .foregroundColor(.secondary)

            Text("No Analysis Results")
                .font(.headline)

            Text("Import a .finsight bundle from the Mac app to view analysis results, findings, and risk scores.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
    }

    // MARK: - Bundle Analysis

    private var bundleAnalysisView: some View {
        List {
            // Bundle picker
            if documentStore.bundles.count > 1 {
                Section("Select Analysis") {
                    Picker("Bundle", selection: $selectedBundle) {
                        Text("Select...").tag(nil as ExportBundle?)
                        ForEach(documentStore.bundles) { bundle in
                            Text(bundle.displayName).tag(bundle as ExportBundle?)
                        }
                    }
                }
            }

            let bundle = selectedBundle ?? documentStore.bundles.first

            if let bundle = bundle {
                // Executive Summary
                if let summary = bundle.executiveSummary {
                    Section("Executive Summary") {
                        Text(summary)
                            .font(.body)
                    }
                }

                // Risk Overview
                if !bundle.riskScores.isEmpty {
                    Section("Risk Overview") {
                        riskHeatMap(bundle: bundle)
                    }
                }

                // Findings by Agent
                let byAgent = bundle.findingsByAgent
                ForEach(Array(byAgent.keys.sorted()), id: \.self) { agent in
                    if let findings = byAgent[agent] {
                        Section(agentDisplayName(agent) + " (\(findings.count))") {
                            ForEach(findings) { finding in
                                FindingSummaryRow(finding: finding)
                            }
                        }
                    }
                }

                // Full Report
                if let report = bundle.fullReport {
                    Section("Full Report") {
                        DisclosureGroup("View Full Report") {
                            Text(report)
                                .font(.caption)
                        }
                    }
                }

                // Metadata
                Section("Analysis Info") {
                    LabeledContent("Model", value: bundle.modelUsed)
                    LabeledContent("Version", value: bundle.version)
                    LabeledContent("Created", value: bundle.createdAt)
                }
            }
        }
    }

    // MARK: - Risk Heat Map

    private func riskHeatMap(bundle: ExportBundle) -> some View {
        let byTactic = bundle.riskScoresByTactic

        return VStack(alignment: .leading, spacing: 8) {
            ForEach(RiskTactic.allCases, id: \.self) { tactic in
                if let scores = byTactic[tactic] {
                    let avgSeverity = scores.reduce(0.0) { $0 + $1.severity } / Double(scores.count)

                    HStack {
                        Text(tactic.displayName)
                            .font(.caption)
                            .frame(width: 120, alignment: .leading)

                        ProgressView(value: avgSeverity)
                            .tint(severityColor(avgSeverity))

                        Text(String(format: "%.0f%%", avgSeverity * 100))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .frame(width: 40)
                    }
                }
            }
        }
    }

    // MARK: - Helpers

    private func agentDisplayName(_ agent: String) -> String {
        switch agent {
        case "doc_intel": return "Document Intelligence"
        case "quant": return "Quantitative Analysis"
        case "risk": return "Risk Classification"
        case "synthesis": return "Synthesis"
        default: return agent
        }
    }

    private func severityColor(_ severity: Double) -> Color {
        switch severity {
        case 0..<0.3: return .green
        case 0.3..<0.6: return .orange
        default: return .red
        }
    }
}

/// Compact finding row for the analysis view.
struct FindingSummaryRow: View {
    let finding: Finding

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(finding.content)
                .font(.subheadline)
                .lineLimit(3)

            HStack {
                Text(finding.confidence.displayName)
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(confidenceBackground)
                    .cornerRadius(4)

                if !finding.citations.isEmpty {
                    Text("\(finding.citations.count) citations")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.vertical, 2)
    }

    private var confidenceBackground: Color {
        switch finding.confidence {
        case .high: return .green.opacity(0.15)
        case .medium: return .orange.opacity(0.15)
        case .low: return .red.opacity(0.15)
        }
    }
}

#Preview {
    AnalysisSummaryView()
        .environmentObject(DocumentStore())
}
