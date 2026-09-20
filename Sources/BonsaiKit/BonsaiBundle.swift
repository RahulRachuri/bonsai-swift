import CoreAI
import Foundation

/// The zoo's bundle layout, as written by `conversion/_bundle.py`:
///
///     <dir>/metadata.json                 kind "llm", assets.main, language.{...}
///     <dir>/<name>.aimodel/               the portable source graph (JIT-specialized on load)
///     <dir>/aot_mac/<name>.<arch>.aimodelc  optional ahead-of-time compile for one chip
///     <dir>/tokenizer/                    the HF tokenizer files + chat template
///
/// Nothing here is Bonsai-specific except the expectation of a `prefill` entrypoint next
/// to `main`, which `language.prefill_chunk` announces.
public struct BonsaiBundle: Sendable {
    public struct Metadata: Decodable, Sendable {
        public struct Assets: Decodable, Sendable { public let main: String }
        public struct Language: Decodable, Sendable {
            public let tokenizer: String
            public let vocab_size: Int
            public let max_context_length: Int
            public let prefill_chunk: Int?
            public let prefill_chunks: [Int]?
            public let next_token_output: String?
            public let function_map: [String: [String]]?
        }
        public let name: String
        public let kind: String
        public let assets: Assets
        public let language: Language
    }

    public let directory: URL
    public let metadata: Metadata

    public init(directory: URL) throws {
        self.directory = directory
        let data = try Data(contentsOf: directory.appendingPathComponent("metadata.json"))
        self.metadata = try JSONDecoder().decode(Metadata.self, from: data)
        guard metadata.kind == "llm" else {
            throw BonsaiError.message("\(directory.path): metadata kind is '\(metadata.kind)', expected 'llm'")
        }
    }

    public var name: String { metadata.name }
    public var vocabSize: Int { metadata.language.vocab_size }
    public var maxContextLength: Int { metadata.language.max_context_length }
    /// Static query length of the `prefill` entrypoint, nil for a decode-only bundle.
    public var prefillChunk: Int? { metadata.language.prefill_chunk }
    /// Every static prefill length the bundle carries, largest first: `prefill` at the first,
    /// `prefill<S>` at each of the others. A prompt is fed greedily, largest chunk that fits.
    public var prefillChunks: [Int] {
        if let all = metadata.language.prefill_chunks { return all.sorted(by: >) }
        return prefillChunk.map { [$0] } ?? []
    }
    /// Name of `main`'s greedy-choice output (int32 `[1,1]`), when the graph emits one.
    public var nextTokenOutput: String? { metadata.language.next_token_output }
    public var tokenizerDirectory: URL { directory.appendingPathComponent("tokenizer") }
    public var sourceAsset: URL { directory.appendingPathComponent(metadata.assets.main) }

    /// The ahead-of-time compile for this machine, if the bundle carries one.
    ///
    /// Two reasons to prefer it. It skips the load-time specialization of a 6.7 GB graph,
    /// and on the macOS 27 beta the JIT path with GPU preferred still tries to place the
    /// custom kernels on the Neural Engine and dies in the command buffer; the compile fixed
    /// the placement, so the `.aimodelc` loads with default options and runs.
    public func compiledAsset(architecture: String = AIModel.deviceArchitectureName) -> URL? {
        let url = directory.appendingPathComponent("aot_mac")
            .appendingPathComponent("\(name).\(architecture).aimodelc")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}
