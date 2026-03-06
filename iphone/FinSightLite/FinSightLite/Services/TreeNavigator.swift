import Foundation

/// Result of a tree search, with relevance metadata.
///
/// Swift port of Python `finsight_core.pageindex.navigator.SearchResult`.
struct SearchResult: Identifiable {
    let node: TreeNode
    let score: Double
    let matchReason: String
    let path: [String]

    var id: String { node.nodeId }
}

/// Navigate and search a PageIndex tree structure.
///
/// Swift port of Python `finsight_core.pageindex.navigator.TreeNavigator`.
/// Provides keyword search, page range search, and section path navigation.
///
/// Example:
/// ```swift
/// let nav = TreeNavigator(tree: tree)
/// let results = nav.searchByKeywords(["supply chain", "risk"])
/// for r in results {
///     print("\(r.node.title) (pages \(r.node.startIndex)-\(r.node.endIndex))")
/// }
/// ```
class TreeNavigator {
    let tree: PageIndexTreeModel

    /// All nodes in DFS order for fast iteration.
    private let allNodes: [TreeNode]

    /// O(1) lookup by node ID.
    private let idIndex: [String: TreeNode]

    /// Title path from root to each node.
    private var paths: [String: [String]] = [:]

    init(tree: PageIndexTreeModel) {
        self.tree = tree
        self.allNodes = tree.allNodes()
        self.idIndex = Dictionary(uniqueKeysWithValues: allNodes.map { ($0.nodeId, $0) })

        // Build paths
        for node in tree.structure {
            buildPaths(node: node, parentPath: [])
        }
    }

    private func buildPaths(node: TreeNode, parentPath: [String]) {
        let currentPath = parentPath + [node.title]
        paths[node.nodeId] = currentPath
        for child in node.nodes {
            buildPaths(node: child, parentPath: currentPath)
        }
    }

    // MARK: - Lookup

    /// Look up a node by its ID. O(1).
    func getById(_ nodeId: String) -> TreeNode? {
        idIndex[nodeId]
    }

    /// Get the title path from root to a given node.
    func getPath(_ nodeId: String) -> [String] {
        paths[nodeId] ?? []
    }

    // MARK: - Search

    /// Find nodes whose titles (and optionally summaries) match keywords.
    ///
    /// Scoring:
    /// - Title exact match: +3.0 per keyword
    /// - Title partial match: +1.5 per keyword
    /// - Summary match: +1.0 per keyword
    /// - Bonus for leaf nodes: +0.5
    func searchByKeywords(
        _ keywords: [String],
        searchSummaries: Bool = true,
        maxResults: Int = 10
    ) -> [SearchResult] {
        let lowercasedKeywords = keywords.map { $0.lowercased() }
        var results: [SearchResult] = []

        for node in allNodes {
            var score = 0.0
            var reasons: [String] = []
            let titleLower = node.title.lowercased()

            for kw in lowercasedKeywords {
                if kw == titleLower {
                    score += 3.0
                    reasons.append("title exact: '\(kw)'")
                } else if titleLower.contains(kw) {
                    score += 1.5
                    reasons.append("title partial: '\(kw)'")
                }

                if searchSummaries, let summary = node.summary?.lowercased() {
                    if summary.contains(kw) {
                        score += 1.0
                        reasons.append("summary: '\(kw)'")
                    }
                }
            }

            if score > 0 {
                if node.isLeaf {
                    score += 0.5
                }
                results.append(SearchResult(
                    node: node,
                    score: score,
                    matchReason: reasons.joined(separator: "; "),
                    path: getPath(node.nodeId)
                ))
            }
        }

        results.sort { $0.score > $1.score }
        return Array(results.prefix(maxResults))
    }

    /// Find all nodes whose page range includes the given page.
    func searchByPage(_ page: Int) -> [SearchResult] {
        var results: [SearchResult] = []

        for node in allNodes {
            if node.startIndex <= page && page <= node.endIndex {
                let specificity = 1.0 / Double(max(node.pageCount, 1))
                results.append(SearchResult(
                    node: node,
                    score: specificity,
                    matchReason: "covers page \(page)",
                    path: getPath(node.nodeId)
                ))
            }
        }

        results.sort { $0.score > $1.score }
        return results
    }

    /// Find nodes overlapping with a page range.
    func searchByPageRange(start: Int, end: Int) -> [SearchResult] {
        var results: [SearchResult] = []

        for node in allNodes {
            let overlapStart = max(node.startIndex, start)
            let overlapEnd = min(node.endIndex, end)
            if overlapStart <= overlapEnd {
                let overlap = Double(overlapEnd - overlapStart + 1)
                let total = Double(max(node.pageCount, 1))
                let coverage = overlap / total
                results.append(SearchResult(
                    node: node,
                    score: coverage,
                    matchReason: "overlaps pages \(start)-\(end) (\(Int(coverage * 100))%)",
                    path: getPath(node.nodeId)
                ))
            }
        }

        results.sort { $0.score > $1.score }
        return results
    }

    /// Navigate to a section by a path of titles.
    ///
    /// - Parameter sectionTitles: List of section titles from root to target.
    /// - Returns: The matching TreeNode, or nil if the path doesn't match.
    func getSectionByPath(_ sectionTitles: [String]) -> TreeNode? {
        var currentNodes = tree.structure

        var match: TreeNode?
        for title in sectionTitles {
            let titleLower = title.lowercased().trimmingCharacters(in: .whitespaces)
            match = nil

            for node in currentNodes {
                if node.title.lowercased().trimmingCharacters(in: .whitespaces) == titleLower {
                    match = node
                    break
                }
                if node.title.lowercased().contains(titleLower) {
                    match = node
                    break
                }
            }

            guard let found = match else { return nil }
            currentNodes = found.nodes
        }

        return match
    }

    /// Get contextual information for a node.
    func getContextForNode(_ nodeId: String, includeSiblings: Bool = true) -> [String: Any] {
        guard let node = getById(nodeId) else {
            return ["error": "Node \(nodeId) not found"]
        }

        let path = getPath(nodeId)
        var siblings: [[String: String]] = []

        if includeSiblings {
            if let parent = findParent(nodeId) {
                for sibling in parent.nodes where sibling.nodeId != nodeId {
                    siblings.append([
                        "node_id": sibling.nodeId,
                        "title": sibling.title,
                        "pages": "\(sibling.startIndex)-\(sibling.endIndex)"
                    ])
                }
            }
        }

        return [
            "node_id": node.nodeId,
            "title": node.title,
            "pages": "\(node.startIndex)-\(node.endIndex)",
            "page_count": node.pageCount,
            "summary": node.summary ?? "",
            "path": path.joined(separator: " > "),
            "is_leaf": node.isLeaf,
            "children_count": node.nodes.count,
            "siblings": siblings,
        ]
    }

    /// Format the tree for LLM-based node selection.
    func getNodesForLLMSelection(maxDepth: Int = 2) -> String {
        tree.getTreeOutline(maxDepth: maxDepth)
    }

    // MARK: - Private

    private func findParent(_ nodeId: String) -> TreeNode? {
        for root in tree.structure {
            if let parent = findParentRecursive(current: root, targetId: nodeId) {
                return parent
            }
        }
        return nil
    }

    private func findParentRecursive(current: TreeNode, targetId: String) -> TreeNode? {
        for child in current.nodes {
            if child.nodeId == targetId {
                return current
            }
            if let found = findParentRecursive(current: child, targetId: targetId) {
                return found
            }
        }
        return nil
    }
}
