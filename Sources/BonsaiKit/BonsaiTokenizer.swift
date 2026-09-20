import Foundation
import Tokenizers

/// The bundle's tokenizer and chat template, through swift-transformers.
///
/// The reference fixtures were rendered by Hugging Face `transformers` with
/// `apply_chat_template([{"role": "user", ...}], add_generation_prompt=True)` and the
/// template's defaults (thinking on). `chatPrompt` renders the same thing here, and the
/// parity command checks the two id sequences agree before it trusts the graph.
public struct BonsaiTokenizer: Sendable {
    public let tokenizer: any Tokenizer
    /// Ids that end a turn: the tokenizer's eos plus the template's own end marker.
    public let stopIds: Set<Int32>

    public static func load(directory: URL) async throws -> BonsaiTokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        var stop = Set<Int32>()
        if let eos = tokenizer.eosTokenId { stop.insert(Int32(eos)) }
        for marker in ["<|im_end|>", "<|endoftext|>"] {
            if let id = tokenizer.convertTokenToId(marker), tokenizer.convertIdToToken(id) == marker {
                stop.insert(Int32(id))
            }
        }
        return BonsaiTokenizer(tokenizer: tokenizer, stopIds: stop)
    }

    public func chatPrompt(_ user: String, system: String? = nil) throws -> [Int32] {
        var messages: [Message] = []
        if let system { messages.append(["role": "system", "content": system]) }
        messages.append(["role": "user", "content": user])
        return try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: nil, addGenerationPrompt: true,
            truncation: false, maxLength: nil, tools: nil).map(Int32.init)
    }

    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    public func decode(_ ids: [Int32]) -> String {
        tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: false)
    }
}
