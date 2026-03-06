import Foundation

/// A single node in a PageIndex hierarchical tree.
///
/// Mirrors the Python `finsight_core.models.document.TreeNode` model exactly.
/// Generated from `schemas/tree_node.schema.json`.
struct TreeNode: Codable, Identifiable, Hashable {
    /// Section heading from the document.
    let title: String

    /// Sequential 4-digit ID assigned via DFS (e.g., "0003").
    let nodeId: String

    /// First physical page number (1-indexed, inclusive).
    let startIndex: Int

    /// Last physical page number (1-indexed, inclusive).
    let endIndex: Int

    /// LLM-generated summary of this section.
    let summary: String?

    /// Raw page text for this section's pages.
    let text: String?

    /// Child sections (recursive).
    let nodes: [TreeNode]

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case title
        case nodeId = "node_id"
        case startIndex = "start_index"
        case endIndex = "end_index"
        case summary
        case text
        case nodes
    }

    // MARK: - Identifiable

    var id: String { nodeId }

    // MARK: - Hashable

    func hash(into hasher: inout Hasher) {
        hasher.combine(nodeId)
    }

    static func == (lhs: TreeNode, rhs: TreeNode) -> Bool {
        lhs.nodeId == rhs.nodeId
    }

    // MARK: - Computed Properties

    /// Number of pages in this section.
    var pageCount: Int {
        endIndex - startIndex + 1
    }

    /// Whether this node has no children.
    var isLeaf: Bool {
        nodes.isEmpty
    }

    /// All nodes in this subtree as a flat list (DFS order).
    func allNodesFlat() -> [TreeNode] {
        var result = [self]
        for child in nodes {
            result.append(contentsOf: child.allNodesFlat())
        }
        return result
    }

    /// Find a node by its ID in this subtree.
    func findById(_ targetId: String) -> TreeNode? {
        if nodeId == targetId { return self }
        for child in nodes {
            if let found = child.findById(targetId) {
                return found
            }
        }
        return nil
    }

    /// Total node count in this subtree.
    var totalNodeCount: Int {
        1 + nodes.reduce(0) { $0 + $1.totalNodeCount }
    }
}

/// Complete PageIndex tree structure for a document.
///
/// Mirrors the Python `finsight_core.models.document.PageIndexTree`.
struct PageIndexTreeModel: Codable {
    /// Source document filename.
    let docName: String

    /// One-sentence document description.
    let docDescription: String?

    /// Top-level sections of the document tree.
    let structure: [TreeNode]

    enum CodingKeys: String, CodingKey {
        case docName = "doc_name"
        case docDescription = "doc_description"
        case structure
    }

    // MARK: - Computed Properties

    /// Total number of nodes in the tree.
    var totalNodes: Int {
        structure.reduce(0) { $0 + $1.totalNodeCount }
    }

    /// All leaf nodes in the tree.
    var leafNodes: [TreeNode] {
        structure.flatMap { $0.allNodesFlat() }.filter { $0.isLeaf }
    }

    /// Find any node in the tree by its ID.
    func findById(_ targetId: String) -> TreeNode? {
        for node in structure {
            if let found = node.findById(targetId) {
                return found
            }
        }
        return nil
    }

    /// All nodes as a flat list in DFS order.
    func allNodes() -> [TreeNode] {
        structure.flatMap { $0.allNodesFlat() }
    }

    /// Generate a text outline of the tree for LLM context.
    func getTreeOutline(maxDepth: Int = 3) -> String {
        var lines: [String] = []
        lines.append("Document: \(docName)")
        if let desc = docDescription {
            lines.append("Description: \(desc)")
        }
        lines.append("")

        for node in structure {
            lines.append(contentsOf: outlineNode(node, depth: 0, maxDepth: maxDepth))
        }
        return lines.joined(separator: "\n")
    }

    private func outlineNode(_ node: TreeNode, depth: Int, maxDepth: Int) -> [String] {
        let indent = String(repeating: "  ", count: depth)
        let pageInfo = "[pages \(node.startIndex)-\(node.endIndex)]"
        let summaryPart = node.summary.map { " — \($0)" } ?? ""
        let line = "\(indent)[\(node.nodeId)] \(node.title) \(pageInfo)\(summaryPart)"
        var lines = [line]

        if depth < maxDepth {
            for child in node.nodes {
                lines.append(contentsOf: outlineNode(child, depth: depth + 1, maxDepth: maxDepth))
            }
        } else if !node.nodes.isEmpty {
            lines.append("\(indent)  ... (\(node.nodes.count) subsections)")
        }

        return lines
    }
}
