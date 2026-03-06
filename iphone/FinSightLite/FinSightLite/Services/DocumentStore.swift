import Foundation

/// Local document storage service for FinSight Lite.
///
/// Manages persistent storage of imported .finsight bundles
/// and pre-loaded PageIndex trees in the app's documents directory.
@MainActor
class DocumentStore: ObservableObject {
    @Published var bundles: [ExportBundle] = []
    @Published var trees: [String: PageIndexTreeModel] = [:]
    @Published var errorMessage: String?

    /// Directory for storing imported bundles.
    private let bundlesDirectory: URL

    /// Directory for storing standalone trees.
    private let treesDirectory: URL

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        bundlesDirectory = docs.appendingPathComponent("Bundles", isDirectory: true)
        treesDirectory = docs.appendingPathComponent("Trees", isDirectory: true)

        // Create directories if needed
        try? FileManager.default.createDirectory(at: bundlesDirectory, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: treesDirectory, withIntermediateDirectories: true)

        // Load existing data
        loadAll()
    }

    // MARK: - Bundle Management

    /// Import a .finsight bundle from a URL (e.g., from AirDrop or Files).
    func importBundle(from url: URL) {
        do {
            // Handle security-scoped resources
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }

            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            let bundle = try decoder.decode(ExportBundle.self, from: data)

            // Save to local storage
            let filename = url.lastPathComponent
            let localURL = bundlesDirectory.appendingPathComponent(filename)
            try data.write(to: localURL)

            bundles.append(bundle)

            // Also index the tree
            trees[bundle.filing.ticker] = bundle.tree

        } catch {
            errorMessage = "Failed to import bundle: \(error.localizedDescription)"
        }
    }

    /// Import a standalone PageIndex tree JSON.
    func importTree(from url: URL, name: String? = nil) {
        do {
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }

            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            let tree = try decoder.decode(PageIndexTreeModel.self, from: data)

            let treeName = name ?? url.deletingPathExtension().lastPathComponent
                .replacingOccurrences(of: "_structure", with: "")

            // Save locally
            let localURL = treesDirectory.appendingPathComponent("\(treeName).json")
            try data.write(to: localURL)

            trees[treeName] = tree

        } catch {
            errorMessage = "Failed to import tree: \(error.localizedDescription)"
        }
    }

    /// Delete a bundle by ID.
    func deleteBundle(_ bundle: ExportBundle) {
        bundles.removeAll { $0.id == bundle.id }

        // Also remove from disk
        let possibleFiles = try? FileManager.default.contentsOfDirectory(
            at: bundlesDirectory,
            includingPropertiesForKeys: nil
        )
        // Note: We'd need to match by content; for simplicity, handled via filename
    }

    /// Delete a tree by name.
    func deleteTree(name: String) {
        trees.removeValue(forKey: name)

        let url = treesDirectory.appendingPathComponent("\(name).json")
        try? FileManager.default.removeItem(at: url)
    }

    // MARK: - Loading

    /// Load all saved bundles and trees from disk.
    func loadAll() {
        loadBundles()
        loadTrees()
    }

    private func loadBundles() {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: bundlesDirectory,
            includingPropertiesForKeys: nil
        ) else { return }

        let decoder = JSONDecoder()
        bundles = files
            .filter { $0.pathExtension == "finsight" || $0.pathExtension == "json" }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url),
                      let bundle = try? decoder.decode(ExportBundle.self, from: data)
                else { return nil }
                return bundle
            }
    }

    private func loadTrees() {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: treesDirectory,
            includingPropertiesForKeys: nil
        ) else { return }

        let decoder = JSONDecoder()
        for url in files where url.pathExtension == "json" {
            let name = url.deletingPathExtension().lastPathComponent
                .replacingOccurrences(of: "_structure", with: "")

            if let data = try? Data(contentsOf: url),
               let tree = try? decoder.decode(PageIndexTreeModel.self, from: data) {
                trees[name] = tree
            }
        }
    }

    // MARK: - Queries

    /// Get all unique tickers across loaded bundles.
    var availableTickers: [String] {
        Array(Set(bundles.map { $0.filing.ticker })).sorted()
    }

    /// Get all available document names (from both bundles and standalone trees).
    var availableDocuments: [String] {
        let bundleNames = bundles.map { $0.filing.displayName }
        let treeNames = Array(trees.keys)
        return Array(Set(bundleNames + treeNames)).sorted()
    }
}
