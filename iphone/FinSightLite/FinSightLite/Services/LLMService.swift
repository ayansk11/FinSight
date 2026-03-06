import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// On-device LLM inference service using MLX Swift.
///
/// Provides text generation using Qwen3.5 models running locally
/// on the device's Apple Silicon (A-series or M-series) chip.
@MainActor
class LLMService: ObservableObject {
    @Published var isLoading: Bool = false
    @Published var isGenerating: Bool = false
    @Published var loadProgress: Double = 0.0
    @Published var currentOutput: String = ""
    @Published var tokensPerSecond: Double = 0.0
    @Published var errorMessage: String?

    private var modelContainer: ModelContainer?
    private var currentModelId: String?

    /// Default model configuration for the 0.8B model.
    static let defaultModelId = "mlx-community/Qwen3.5-0.8B-4bit"

    /// Load a model from HuggingFace Hub (downloads on first use).
    func loadModel(modelId: String = defaultModelId) async {
        guard modelId != currentModelId else { return }

        isLoading = true
        loadProgress = 0.0
        errorMessage = nil

        do {
            let modelConfiguration = ModelConfiguration(id: modelId)

            // Download and load the model
            let container = try await MLXLMCommon.loadModelContainer(
                configuration: modelConfiguration
            ) { progress in
                Task { @MainActor in
                    self.loadProgress = progress.fractionCompleted
                }
            }

            self.modelContainer = container
            self.currentModelId = modelId
            self.isLoading = false
            self.loadProgress = 1.0
        } catch {
            self.errorMessage = "Failed to load model: \(error.localizedDescription)"
            self.isLoading = false
        }
    }

    /// Generate a response for the given prompt.
    ///
    /// - Parameters:
    ///   - systemPrompt: System message for the LLM.
    ///   - userMessage: User's question or instruction.
    ///   - maxTokens: Maximum tokens to generate.
    /// - Returns: The generated text response.
    func generate(
        systemPrompt: String = "You are a helpful financial document analyst.",
        userMessage: String,
        maxTokens: Int = 1024
    ) async -> String {
        guard let container = modelContainer else {
            return "Error: Model not loaded. Please load a model first."
        }

        isGenerating = true
        currentOutput = ""
        tokensPerSecond = 0.0

        do {
            let messages: [[String: String]] = [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": userMessage],
            ]

            let result = try await container.perform { context in
                let input = try await context.processor.prepare(
                    input: .init(messages: messages)
                )

                var output = ""
                var tokenCount = 0
                let startTime = Date()

                let result = try MLXLMCommon.generate(
                    input: input,
                    parameters: .init(temperature: 0.7, topP: 0.9),
                    context: context
                ) { tokens in
                    // Streaming callback
                    let text = context.tokenizer.decode(tokens: tokens)
                    output = text
                    tokenCount = tokens.count

                    Task { @MainActor in
                        self.currentOutput = text
                        let elapsed = Date().timeIntervalSince(startTime)
                        if elapsed > 0 {
                            self.tokensPerSecond = Double(tokenCount) / elapsed
                        }
                    }

                    if tokenCount >= maxTokens {
                        return .stop
                    }
                    return .more
                }

                return result.output
            }

            isGenerating = false

            // Strip Qwen3.5 thinking tags if present
            let cleaned = stripThinkingTags(result)
            currentOutput = cleaned
            return cleaned

        } catch {
            isGenerating = false
            let msg = "Generation error: \(error.localizedDescription)"
            errorMessage = msg
            return msg
        }
    }

    /// Generate a response with tree context for document Q&A.
    func generateWithTreeContext(
        query: String,
        treeOutline: String,
        sectionText: String? = nil,
        maxTokens: Int = 1024
    ) async -> String {
        var systemPrompt = """
        You are a financial document analyst. You have access to a hierarchical \
        document structure (PageIndex tree) representing a SEC filing.

        DOCUMENT STRUCTURE:
        \(treeOutline)
        """

        if let text = sectionText {
            systemPrompt += """

            RELEVANT SECTION TEXT:
            \(text.prefix(4000))
            """
        }

        systemPrompt += """

        Instructions:
        - Answer the user's question based on the document structure and any provided text.
        - Reference specific sections by their node IDs and page numbers.
        - If the answer is not in the available content, say so clearly.
        - Be concise and factual.
        """

        return await generate(
            systemPrompt: systemPrompt,
            userMessage: query,
            maxTokens: maxTokens
        )
    }

    /// Check if a model is currently loaded and ready.
    var isReady: Bool {
        modelContainer != nil
    }

    /// Unload the current model to free memory.
    func unloadModel() {
        modelContainer = nil
        currentModelId = nil
        currentOutput = ""
    }

    // MARK: - Private

    /// Strip Qwen3.5's `<think>...</think>` reasoning tags from output.
    private func stripThinkingTags(_ text: String) -> String {
        // Remove <think>...</think> blocks (including multiline)
        let pattern = #"<think>[\s\S]*?</think>\s*"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return text
        }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, range: range, withTemplate: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
