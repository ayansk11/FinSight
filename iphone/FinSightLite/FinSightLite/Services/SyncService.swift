import Foundation
import UniformTypeIdentifiers

/// Service for importing .finsight bundles from various sources.
///
/// Supports:
/// - AirDrop: Receives .finsight files via share sheet
/// - Files app: Open .finsight files from iCloud Drive, local storage, etc.
/// - Direct URL: Import from a file URL
class SyncService: ObservableObject {
    @Published var isImporting: Bool = false
    @Published var lastImportResult: ImportResult?

    /// Result of an import operation.
    struct ImportResult: Identifiable {
        let id = UUID()
        let success: Bool
        let documentName: String
        let message: String
        let timestamp: Date = Date()
    }

    /// The UTType for .finsight bundle files.
    static let finsightUTType = UTType(
        exportedAs: "dev.finsight.bundle",
        conformingTo: .json
    )

    /// Supported file extensions.
    static let supportedExtensions: [String] = ["finsight", "json"]

    /// Import a .finsight bundle from a URL into the document store.
    @MainActor
    func importBundle(from url: URL, into store: DocumentStore) async {
        isImporting = true

        do {
            // Handle security-scoped resources (from Files app, AirDrop, etc.)
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }

            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()

            // Try to decode as ExportBundle first
            if let bundle = try? decoder.decode(ExportBundle.self, from: data) {
                store.importBundle(from: url)

                lastImportResult = ImportResult(
                    success: true,
                    documentName: bundle.displayName,
                    message: "Imported \(bundle.findingCount) findings and \(bundle.riskScoreCount) risk scores"
                )
            }
            // Fall back to standalone PageIndex tree
            else if let _ = try? decoder.decode(PageIndexTreeModel.self, from: data) {
                store.importTree(from: url)

                let name = url.deletingPathExtension().lastPathComponent
                lastImportResult = ImportResult(
                    success: true,
                    documentName: name,
                    message: "Imported PageIndex tree"
                )
            } else {
                lastImportResult = ImportResult(
                    success: false,
                    documentName: url.lastPathComponent,
                    message: "Unrecognized file format"
                )
            }

        } catch {
            lastImportResult = ImportResult(
                success: false,
                documentName: url.lastPathComponent,
                message: "Import failed: \(error.localizedDescription)"
            )
        }

        isImporting = false
    }

    /// Check if a URL points to a file we can import.
    static func canImport(_ url: URL) -> Bool {
        supportedExtensions.contains(url.pathExtension.lowercased())
    }
}
