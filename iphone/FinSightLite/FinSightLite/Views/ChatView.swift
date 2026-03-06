import SwiftUI

/// Chat interface for on-device Q&A against loaded documents.
struct ChatView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var documentStore: DocumentStore
    @EnvironmentObject var llmService: LLMService

    @State private var inputText: String = ""
    @State private var messages: [ChatMessage] = []
    @State private var selectedDocument: String?
    @State private var showingModelPicker = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Document selector
                documentSelector

                Divider()

                // Chat messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(messages) { message in
                                ChatBubble(message: message)
                                    .id(message.id)
                            }

                            // Streaming output
                            if llmService.isGenerating {
                                StreamingBubble(text: llmService.currentOutput)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: messages.count) { _, _ in
                        if let last = messages.last {
                            withAnimation {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                }

                Divider()

                // Input area
                inputArea
            }
            .navigationTitle("Chat")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingModelPicker = true }) {
                        HStack(spacing: 4) {
                            Circle()
                                .fill(llmService.isReady ? Color.green : Color.red)
                                .frame(width: 8, height: 8)
                            Image(systemName: "cpu")
                        }
                    }
                }
            }
            .sheet(isPresented: $showingModelPicker) {
                ModelPickerView()
                    .environmentObject(appState)
                    .environmentObject(llmService)
            }
        }
    }

    // MARK: - Document Selector

    private var documentSelector: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(documentStore.availableDocuments, id: \.self) { name in
                    Button(action: { selectedDocument = name }) {
                        Text(name)
                            .font(.caption)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(
                                selectedDocument == name
                                    ? Color.accentColor
                                    : Color.secondary.opacity(0.15)
                            )
                            .foregroundColor(selectedDocument == name ? .white : .primary)
                            .cornerRadius(16)
                    }
                }

                if documentStore.availableDocuments.isEmpty {
                    Text("No documents loaded")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 10)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
        }
    }

    // MARK: - Input Area

    private var inputArea: some View {
        HStack(spacing: 8) {
            TextField("Ask about your documents...", text: $inputText, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...4)

            Button(action: sendMessage) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || llmService.isGenerating)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    // MARK: - Actions

    private func sendMessage() {
        let query = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return }

        // Add user message
        let userMessage = ChatMessage(role: .user, content: query)
        messages.append(userMessage)
        inputText = ""

        // Check for loaded documents
        guard !documentStore.trees.isEmpty || !documentStore.bundles.isEmpty else {
            let errorMsg = ChatMessage(
                role: .assistant,
                content: "No documents loaded. Please import a .finsight bundle or PageIndex tree from the Documents tab first."
            )
            messages.append(errorMsg)
            return
        }

        // Check if model is loaded
        guard llmService.isReady else {
            let errorMsg = ChatMessage(
                role: .assistant,
                content: "Model not loaded. Tap the CPU icon in the toolbar to download and load a model first."
            )
            messages.append(errorMsg)
            return
        }

        // Generate response
        Task {
            let treeOutline = getSelectedTreeOutline()

            let response = await llmService.generateWithTreeContext(
                query: query,
                treeOutline: treeOutline
            )

            let assistantMessage = ChatMessage(
                role: .assistant,
                content: response,
                tokensPerSecond: llmService.tokensPerSecond
            )
            messages.append(assistantMessage)
        }
    }

    private func getSelectedTreeOutline() -> String {
        // Try selected document first
        if let selected = selectedDocument, let tree = documentStore.trees[selected] {
            return tree.getTreeOutline(maxDepth: 3)
        }

        // Fall back to first available bundle's tree
        if let bundle = documentStore.bundles.first {
            return bundle.tree.getTreeOutline(maxDepth: 3)
        }

        // Fall back to first available tree
        if let (_, tree) = documentStore.trees.first {
            return tree.getTreeOutline(maxDepth: 3)
        }

        return "No document structure available."
    }
}

// MARK: - Chat Models

/// A single chat message.
struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    let content: String
    let timestamp: Date = Date()
    var tokensPerSecond: Double?

    enum Role {
        case user
        case assistant
    }
}

// MARK: - Chat Bubble Views

/// Styled chat bubble.
struct ChatBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 40) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        message.role == .user
                            ? Color.accentColor
                            : Color.secondary.opacity(0.15)
                    )
                    .foregroundColor(message.role == .user ? .white : .primary)
                    .cornerRadius(16)

                if let tps = message.tokensPerSecond, tps > 0 {
                    Text(String(format: "%.1f tok/s", tps))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }

            if message.role == .assistant { Spacer(minLength: 40) }
        }
    }
}

/// Streaming response bubble with typing indicator.
struct StreamingBubble: View {
    let text: String

    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                if text.isEmpty {
                    HStack(spacing: 4) {
                        ForEach(0..<3, id: \.self) { i in
                            Circle()
                                .fill(Color.secondary)
                                .frame(width: 6, height: 6)
                                .opacity(0.6)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                } else {
                    Text(text)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                }
            }
            .background(Color.secondary.opacity(0.15))
            .cornerRadius(16)

            Spacer(minLength: 40)
        }
    }
}

// MARK: - Model Picker

/// View for selecting and downloading LLM models.
struct ModelPickerView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var llmService: LLMService
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Device Info") {
                    LabeledContent("RAM", value: "\(appState.deviceRAMGB) GB")
                    LabeledContent("Model Loaded", value: llmService.isReady ? "Yes" : "No")
                }

                Section("Available Models") {
                    ForEach(appState.availableModels) { model in
                        Button(action: {
                            Task {
                                await llmService.loadModel(modelId: model.id)
                                appState.modelName = model.name
                            }
                        }) {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(model.name)
                                        .font(.headline)
                                    Spacer()
                                    Text(String(format: "%.1f GB", model.sizeGB))
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Text(model.description)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }

                if llmService.isLoading {
                    Section("Download Progress") {
                        ProgressView(value: llmService.loadProgress) {
                            Text("Downloading model...")
                        }
                    }
                }

                if let error = llmService.errorMessage {
                    Section("Error") {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                    }
                }

                if llmService.isReady {
                    Section {
                        Button("Unload Model", role: .destructive) {
                            llmService.unloadModel()
                            appState.isModelLoaded = false
                        }
                    }
                }
            }
            .navigationTitle("Model Settings")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#Preview {
    ChatView()
        .environmentObject(AppState())
        .environmentObject(DocumentStore())
        .environmentObject(LLMService())
}
