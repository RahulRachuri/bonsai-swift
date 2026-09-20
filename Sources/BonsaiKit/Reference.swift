import Foundation

/// A greedy reference decode from PrismML's MLX runtime, as written by the zoo's
/// `_smoke/bonsai/mlx_reference.py`: the prompt ids, the tokens it generated, and at every
/// step the argmax it chose with the top-5 logits and the top-1/top-2 margin.
///
/// The step index runs over the whole teacher-forced sequence: step `s` fed token `seq[s]`
/// and produced the distribution for position `s+1`. Steps below `ids.count - 1` are prompt
/// positions, the rest are generation.
public struct MLXReference: Decodable, Sendable {
    public struct Step: Decodable, Sendable {
        public let step: Int
        public let token: Int32
        public let argmax: Int32
        public let top5: [Int32]
        public let top5_logits: [Float]
        public let margin: Float
    }
    public let prompt: String
    public let chat: Bool
    public let ids: [Int32]
    public let generated: [Int32]
    public let steps: [Step]

    public static func load(_ url: URL) throws -> MLXReference {
        try JSONDecoder().decode(MLXReference.self, from: Data(contentsOf: url))
    }

    /// How many new tokens the reference generated, from its own step count.
    public var newTokens: Int { steps.count + 1 - ids.count }
}
