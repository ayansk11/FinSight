import SwiftUI

/// FinSight Lite — iPhone companion app for financial document analysis.
///
/// Provides on-device Q&A using Qwen3.5 via MLX Swift, tree browsing
/// for PageIndex structures, and viewing of pre-computed analysis bundles
/// exported from the Mac version.
@main
struct FinSightLiteApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
}

/// Global application state shared across all views.
@MainActor
class AppState: ObservableObject {
    @Published var loadedBundles: [ExportBundle] = []
    @Published var loadedTrees: [String: TreeNode] = [:]
    @Published var isModelLoaded: Bool = false
    @Published var modelName: String = "qwen3.5:0.8b"
    @Published var selectedTab: Tab = .documents

    enum Tab: String, CaseIterable {
        case documents = "Documents"
        case chat = "Chat"
        case analysis = "Analysis"
    }

    /// Available device RAM in GB (approximate).
    var deviceRAMGB: Int {
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        return Int(totalMemory / (1024 * 1024 * 1024))
    }

    /// Whether this device supports the larger 2B model.
    var supports2BModel: Bool {
        deviceRAMGB >= 8
    }

    /// Recommended models based on device capability.
    var availableModels: [ModelOption] {
        var models = [
            ModelOption(
                name: "Qwen3.5-0.8B",
                id: "mlx-community/Qwen3.5-0.8B-4bit",
                sizeGB: 0.5,
                description: "Fast, lightweight — works on all devices"
            )
        ]
        if supports2BModel {
            models.append(
                ModelOption(
                    name: "Qwen3.5-2B",
                    id: "mlx-community/Qwen3.5-2B-4bit",
                    sizeGB: 1.2,
                    description: "Better quality — requires 8GB+ RAM"
                )
            )
        }
        return models
    }
}

/// A model option the user can select.
struct ModelOption: Identifiable {
    let name: String
    let id: String
    let sizeGB: Double
    let description: String
}
