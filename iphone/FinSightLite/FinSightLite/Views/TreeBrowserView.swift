import SwiftUI

/// Hierarchical tree browser for PageIndex structures.
///
/// Displays the document tree in an expandable/collapsible outline view
/// with node details, page ranges, and summaries.
struct TreeBrowserView: View {
    let tree: PageIndexTreeModel
    let name: String

    @State private var searchText: String = ""
    @State private var searchResults: [SearchResult] = []
    @State private var selectedNode: TreeNode?

    private var navigator: TreeNavigator {
        TreeNavigator(tree: tree)
    }

    var body: some View {
        List {
            // Document info
            Section("Document") {
                LabeledContent("Name", value: tree.docName)
                LabeledContent("Nodes", value: "\(tree.totalNodes)")
                LabeledContent("Leaves", value: "\(tree.leafNodes.count)")
                if let desc = tree.docDescription {
                    Text(desc)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            // Search results
            if !searchResults.isEmpty {
                Section("Search Results") {
                    ForEach(searchResults) { result in
                        Button(action: { selectedNode = result.node }) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(result.node.title)
                                    .font(.subheadline)
                                HStack {
                                    Text("Score: \(String(format: "%.1f", result.score))")
                                    Text("Pages \(result.node.startIndex)-\(result.node.endIndex)")
                                }
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                Text(result.path.joined(separator: " > "))
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                    .lineLimit(1)
                            }
                        }
                    }
                }
            }

            // Tree structure
            Section("Structure") {
                ForEach(tree.structure) { node in
                    TreeNodeRow(node: node, depth: 0, selectedNode: $selectedNode)
                }
            }
        }
        .navigationTitle(name)
        .searchable(text: $searchText, prompt: "Search sections...")
        .onChange(of: searchText) { _, newValue in
            if newValue.count >= 2 {
                let keywords = newValue.components(separatedBy: " ")
                    .filter { !$0.isEmpty }
                searchResults = navigator.searchByKeywords(keywords, maxResults: 10)
            } else {
                searchResults = []
            }
        }
        .sheet(item: $selectedNode) { node in
            NodeDetailView(node: node, navigator: navigator)
        }
    }
}

/// Recursive tree node row with expand/collapse.
struct TreeNodeRow: View {
    let node: TreeNode
    let depth: Int
    @Binding var selectedNode: TreeNode?
    @State private var isExpanded: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: {
                if node.isLeaf {
                    selectedNode = node
                } else {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isExpanded.toggle()
                    }
                }
            }) {
                HStack(spacing: 4) {
                    // Indent
                    if depth > 0 {
                        Color.clear.frame(width: CGFloat(depth * 16))
                    }

                    // Expand/collapse indicator
                    if !node.isLeaf {
                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .frame(width: 16)
                    } else {
                        Image(systemName: "doc.text")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .frame(width: 16)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text(node.title)
                            .font(.subheadline)
                            .lineLimit(2)

                        HStack(spacing: 8) {
                            Text("[\(node.nodeId)]")
                                .font(.caption2)
                                .foregroundColor(.blue)
                            Text("pp. \(node.startIndex)-\(node.endIndex)")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }

                    Spacer()

                    // Info button for detailed view
                    Button(action: { selectedNode = node }) {
                        Image(systemName: "info.circle")
                            .font(.caption)
                            .foregroundColor(.accentColor)
                    }
                }
            }
            .buttonStyle(.plain)
            .padding(.vertical, 4)

            // Children
            if isExpanded {
                ForEach(node.nodes) { child in
                    TreeNodeRow(node: child, depth: depth + 1, selectedNode: $selectedNode)
                }
            }
        }
    }
}

/// Detailed view for a single node.
struct NodeDetailView: View {
    let node: TreeNode
    let navigator: TreeNavigator

    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Node Info") {
                    LabeledContent("Title", value: node.title)
                    LabeledContent("Node ID", value: node.nodeId)
                    LabeledContent("Pages", value: "\(node.startIndex) - \(node.endIndex)")
                    LabeledContent("Page Count", value: "\(node.pageCount)")
                    LabeledContent("Children", value: "\(node.nodes.count)")
                    LabeledContent("Is Leaf", value: node.isLeaf ? "Yes" : "No")
                }

                // Path
                let path = navigator.getPath(node.nodeId)
                if !path.isEmpty {
                    Section("Path") {
                        Text(path.joined(separator: " > "))
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                // Summary
                if let summary = node.summary {
                    Section("Summary") {
                        Text(summary)
                            .font(.body)
                    }
                }

                // Text content
                if let text = node.text {
                    Section("Content") {
                        Text(text.prefix(2000))
                            .font(.caption)
                            .lineLimit(50)
                    }
                }

                // Children
                if !node.nodes.isEmpty {
                    Section("Children (\(node.nodes.count))") {
                        ForEach(node.nodes) { child in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(child.title)
                                    .font(.subheadline)
                                Text("[\(child.nodeId)] pages \(child.startIndex)-\(child.endIndex)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Node \(node.nodeId)")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#Preview {
    let sampleTree = PageIndexTreeModel(
        docName: "Sample 10-K",
        docDescription: "Annual report for fiscal year 2024",
        structure: [
            TreeNode(
                title: "Part I",
                nodeId: "0001",
                startIndex: 1,
                endIndex: 50,
                summary: "Business overview and risk factors",
                text: nil,
                nodes: [
                    TreeNode(
                        title: "Item 1 - Business",
                        nodeId: "0002",
                        startIndex: 1,
                        endIndex: 20,
                        summary: "Description of the business",
                        text: nil,
                        nodes: []
                    ),
                    TreeNode(
                        title: "Item 1A - Risk Factors",
                        nodeId: "0003",
                        startIndex: 21,
                        endIndex: 50,
                        summary: "Key risk factors",
                        text: nil,
                        nodes: []
                    ),
                ]
            ),
        ]
    )

    TreeBrowserView(tree: sampleTree, name: "Sample")
}
