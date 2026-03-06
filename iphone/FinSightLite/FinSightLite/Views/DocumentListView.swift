import SwiftUI
import UniformTypeIdentifiers

/// Document management view — list, import, and manage documents.
struct DocumentListView: View {
    @EnvironmentObject var documentStore: DocumentStore
    @EnvironmentObject var syncService: SyncService
    @State private var showingImporter = false
    @State private var showingAlert = false

    var body: some View {
        NavigationStack {
            List {
                // Bundles section
                if !documentStore.bundles.isEmpty {
                    Section("Analysis Bundles") {
                        ForEach(documentStore.bundles) { bundle in
                            NavigationLink(destination: BundleDetailView(bundle: bundle)) {
                                BundleRow(bundle: bundle)
                            }
                        }
                    }
                }

                // Standalone trees section
                if !documentStore.trees.isEmpty {
                    Section("PageIndex Trees") {
                        ForEach(Array(documentStore.trees.keys.sorted()), id: \.self) { name in
                            if let tree = documentStore.trees[name] {
                                NavigationLink(destination: TreeBrowserView(tree: tree, name: name)) {
                                    TreeRow(name: name, tree: tree)
                                }
                            }
                        }
                    }
                }

                // Empty state
                if documentStore.bundles.isEmpty && documentStore.trees.isEmpty {
                    Section {
                        VStack(spacing: 16) {
                            Image(systemName: "doc.text.magnifyingglass")
                                .font(.system(size: 48))
                                .foregroundColor(.secondary)

                            Text("No Documents")
                                .font(.headline)

                            Text("Import a .finsight bundle from the Mac app or load a PageIndex tree JSON file.")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)

                            Button(action: { showingImporter = true }) {
                                Label("Import File", systemImage: "square.and.arrow.down")
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 32)
                    }
                }
            }
            .navigationTitle("Documents")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingImporter = true }) {
                        Image(systemName: "plus")
                    }
                }
            }
            .fileImporter(
                isPresented: $showingImporter,
                allowedContentTypes: [.json, UTType(filenameExtension: "finsight") ?? .json],
                allowsMultipleSelection: true
            ) { result in
                switch result {
                case .success(let urls):
                    for url in urls {
                        Task {
                            await syncService.importBundle(from: url, into: documentStore)
                        }
                    }
                case .failure(let error):
                    documentStore.errorMessage = error.localizedDescription
                }
            }
            .alert("Import Result", isPresented: $showingAlert) {
                Button("OK") {}
            } message: {
                if let result = syncService.lastImportResult {
                    Text(result.message)
                }
            }
            .onChange(of: syncService.lastImportResult?.id) { _, _ in
                showingAlert = syncService.lastImportResult != nil
            }
        }
    }
}

/// Row for displaying a bundle in the list.
struct BundleRow: View {
    let bundle: ExportBundle

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(bundle.displayName)
                .font(.headline)

            HStack(spacing: 12) {
                Label("\(bundle.findingCount)", systemImage: "magnifyingglass")
                Label("\(bundle.riskScoreCount)", systemImage: "exclamationmark.triangle")
                Label("\(bundle.tree.totalNodes) nodes", systemImage: "list.bullet.indent")
            }
            .font(.caption)
            .foregroundColor(.secondary)

            if let severity = bundle.averageSeverity {
                HStack {
                    Text("Avg Risk:")
                        .font(.caption2)
                    ProgressView(value: severity)
                        .tint(severityColor(severity))
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func severityColor(_ severity: Double) -> Color {
        switch severity {
        case 0..<0.3: return .green
        case 0.3..<0.6: return .orange
        default: return .red
        }
    }
}

/// Row for displaying a standalone tree.
struct TreeRow: View {
    let name: String
    let tree: PageIndexTreeModel

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(name)
                .font(.headline)

            HStack(spacing: 12) {
                Label("\(tree.totalNodes) nodes", systemImage: "list.bullet.indent")
                Label("\(tree.leafNodes.count) leaves", systemImage: "leaf")
            }
            .font(.caption)
            .foregroundColor(.secondary)

            if let desc = tree.docDescription {
                Text(desc)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }
}

/// Detail view for an imported bundle.
struct BundleDetailView: View {
    let bundle: ExportBundle

    var body: some View {
        List {
            Section("Filing Info") {
                LabeledContent("Ticker", value: bundle.filing.ticker)
                LabeledContent("Type", value: bundle.filing.filingType.rawValue)
                LabeledContent("Model", value: bundle.modelUsed)
            }

            if let summary = bundle.executiveSummary {
                Section("Executive Summary") {
                    Text(summary)
                        .font(.body)
                }
            }

            if !bundle.findings.isEmpty {
                Section("Findings (\(bundle.findingCount))") {
                    ForEach(bundle.findings) { finding in
                        FindingRow(finding: finding)
                    }
                }
            }

            if !bundle.riskScores.isEmpty {
                Section("Risk Scores (\(bundle.riskScoreCount))") {
                    ForEach(bundle.riskScores) { score in
                        RiskScoreRow(score: score)
                    }
                }
            }

            Section("Document Tree") {
                NavigationLink("Browse Tree (\(bundle.tree.totalNodes) nodes)") {
                    TreeBrowserView(tree: bundle.tree, name: bundle.filing.ticker)
                }
            }
        }
        .navigationTitle(bundle.displayName)
    }
}

/// Row for displaying a finding.
struct FindingRow: View {
    let finding: Finding

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(finding.agentDisplayName)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(4)

                Spacer()

                Text(finding.confidence.displayName)
                    .font(.caption2)
                    .foregroundColor(confidenceColor(finding.confidence))
            }

            Text(finding.content)
                .font(.subheadline)
                .lineLimit(3)

            if !finding.citations.isEmpty {
                Text(finding.citations.map { $0.shortRef }.joined(separator: ", "))
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 2)
    }

    private func confidenceColor(_ level: ConfidenceLevel) -> Color {
        switch level {
        case .high: return .green
        case .medium: return .orange
        case .low: return .red
        }
    }
}

/// Row for displaying a risk score.
struct RiskScoreRow: View {
    let score: RiskScore

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(score.tactic.displayName)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(4)

                Spacer()

                Text(score.severityPercent)
                    .font(.caption)
                    .bold()
            }

            Text(score.techniqueName)
                .font(.subheadline)

            Text(score.evidence)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(2)
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    DocumentListView()
        .environmentObject(DocumentStore())
        .environmentObject(SyncService())
}
