import SwiftUI

/// Root view with tab-based navigation.
struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var documentStore = DocumentStore()
    @StateObject private var llmService = LLMService()
    @StateObject private var syncService = SyncService()

    var body: some View {
        TabView(selection: $appState.selectedTab) {
            DocumentListView()
                .environmentObject(documentStore)
                .environmentObject(syncService)
                .tabItem {
                    Label("Documents", systemImage: "doc.text")
                }
                .tag(AppState.Tab.documents)

            ChatView()
                .environmentObject(documentStore)
                .environmentObject(llmService)
                .tabItem {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right")
                }
                .tag(AppState.Tab.chat)

            AnalysisSummaryView()
                .environmentObject(documentStore)
                .tabItem {
                    Label("Analysis", systemImage: "chart.bar.xaxis")
                }
                .tag(AppState.Tab.analysis)
        }
        .onOpenURL { url in
            if SyncService.canImport(url) {
                Task {
                    await syncService.importBundle(from: url, into: documentStore)
                }
                appState.selectedTab = .documents
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(AppState())
}
