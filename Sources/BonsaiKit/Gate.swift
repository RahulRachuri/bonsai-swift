import Darwin
import Foundation

/// The token-identity gate: the bundle teacher-forced through a reference decode, argmax
/// compared at every position.
///
/// Mirrors the zoo's `--check-only --ref-json` check step for step, so the Swift host and
/// the Python runtime are judged by the same rule. The prompt goes through whole `prefill`
/// chunks when `chunked` (the last prompt token always through `main`), the rest of the
/// sequence one step at a time. After the prompt, the reference's own next token is fed
/// regardless of what the bundle chose, so every step compares the same context even after
/// a divergence; `generated` still records what the bundle would have said.
public struct GateReport {
    public struct Miss {
        public let step: Int
        public let ours: Int32
        public let ourMargin: Float
        public let ref: Int32
        public let refMargin: Float
        public let inPrompt: Bool
    }
    public var positions = 0
    public var agreed = 0
    public var misses: [Miss] = []
    public var generated: [Int32] = []
    public var referenceGenerated: [Int32] = []
    public var chunkedTokens = 0
    /// The chunk sizes actually run, in order: the engine's own greedy plan, so the gate
    /// exercises every prefill entrypoint `chat` would use on this prompt.
    public var chunkPlan: [Int] = []
    public var chunkSeconds = 0.0
    public var walkedTokens = 0
    public var walkSeconds = 0.0
    public var decodedTokens = 0
    public var decodeSeconds = 0.0

    /// Walked positions where the graph's own `next_token` (Core AI's reduce_index) chose a
    /// different token than the host scan over the same fp16 logits. Both are argmaxes; they
    /// can only differ on an exact tie, and the generation loop trusts the graph's choice.
    public struct ArgmaxMismatch {
        public let step: Int
        public let host: Int32
        public let graph: Int32
        public let margin: Float
    }
    public var argmaxMismatches: [ArgmaxMismatch] = []

    public var generationIdentical: Bool { generated == referenceGenerated }
}

public func runGate(engine: BonsaiEngine, reference ref: MLXReference, chunked: Bool,
                    log: (String) -> Void = { _ in }) async throws -> GateReport {
    var report = GateReport()
    report.referenceGenerated = ref.generated
    let refSteps = Dictionary(uniqueKeysWithValues: ref.steps.map { ($0.step, $0) })
    let ids = ref.ids
    let new = ref.newTokens
    var seq = ids
    engine.reset()

    func score(_ step: Int, _ top: Logits.Top2) {
        guard let r = refSteps[step] else { return }
        report.positions += 1
        if top.argmax == r.argmax {
            report.agreed += 1
        } else {
            report.misses.append(.init(step: step, ours: top.argmax, ourMargin: top.margin,
                                       ref: r.argmax, refMargin: r.margin, inPrompt: step < ids.count - 1))
        }
    }

    var start = 0
    if chunked, !engine.chunkSizes.isEmpty {
        // The same greedy plan as `BonsaiEngine.prefill` (largest chunk that leaves the last
        // prompt token for `main`), so the gate covers the entrypoints the product runs.
        let t0 = ContinuousClock.now
        while let size = engine.chunkSizes.first(where: { $0 <= ids.count - 1 - start }) {
            let logits = try await engine.prefillChunk(seq[start ..< start + size])
            for j in 0..<size { score(start + j, logits.top2(row: j)) }
            report.chunkPlan.append(size)
            start += size
        }
        report.chunkSeconds = seconds(ContinuousClock.now - t0)
        report.chunkedTokens = start
        log("prefill \(start) tokens in chunks \(report.chunkPlan): "
            + String(format: "%.1f tok/s", Double(start) / max(report.chunkSeconds, 1e-9)))
    }
    report.walkedTokens = ids.count - start
    var walk = 0.0, decode = 0.0
    for step in start ..< (ids.count + new - 1) {
        let t0 = ContinuousClock.now
        let logits = try await engine.step(seq[step])
        let dt = seconds(ContinuousClock.now - t0)
        if step < ids.count { walk += dt } else { decode += dt }
        let top = logits.top2(row: 0)
        score(step, top)
        if let graph = engine.lastNextToken, graph != top.argmax {
            report.argmaxMismatches.append(.init(step: step, host: top.argmax, graph: graph, margin: top.margin))
        }
        if step >= ids.count - 1 {
            report.generated.append(top.argmax)
            seq.append(refSteps[step + 1]?.token ?? top.argmax)
        }
    }
    report.walkSeconds = walk
    report.decodedTokens = new - 1
    report.decodeSeconds = decode
    return report
}


/// Chunked prefill against the same bundle's S=1 walk: per-position argmax agreement and the
/// worst logit difference over the prompt. No reference needed, so it gates the prefill
/// entrypoints on any build, including truncated ones.
public struct SelfCheckReport {
    public struct Disagreement {
        public var position: Int
        public var chunkArgmax: Int32
        public var walkArgmax: Int32
        public var chunkMargin: Float
        public var walkMargin: Float
    }
    public var positions = 0
    public var agreed = 0
    public var worstDelta: Float = 0
    public var worstDeltaPosition = -1
    public var disagreements: [Disagreement] = []
    public var chunkPlan: [Int] = []
    public var walkSeconds = 0.0
    public var chunkSeconds = 0.0
    public var generatedPositions = 0
    public var generatedAgreed = 0
    public var generatedDisagreements: [Disagreement] = []
}

public func selfCheck(engine: BonsaiEngine, ids: [Int32], generatedTokens: Int = 0,
                      chunkSizes requestedChunkSizes: [Int]? = nil) async throws -> SelfCheckReport {
    precondition(!ids.isEmpty)
    var report = SelfCheckReport()
    var scratchPath = Array((NSTemporaryDirectory() + "bonsai-selfcheck.XXXXXX").utf8CString)
    let scratchFile = scratchPath.withUnsafeMutableBufferPointer { mkstemp($0.baseAddress!) }
    guard scratchFile >= 0 else {
        throw BonsaiError.message("create self-check scratch file: \(String(cString: strerror(errno)))")
    }
    // The descriptor remains valid after unlink and is closed on every return path. Avoid
    // populating the unified-memory file cache with hundreds of MB of disposable logits.
    _ = scratchPath.withUnsafeBufferPointer { unlink($0.baseAddress!) }
    _ = fcntl(scratchFile, F_NOCACHE, 1)
    defer { close(scratchFile) }

    engine.reset()
    var walk: [(Int32, Float)] = []                   // argmax, margin; logits are disk-backed
    var walkLast: Logits.Top2?
    let t0 = ContinuousClock.now
    for id in ids {
        let l = try await engine.step(id)
        let top = l.top2(row: 0)
        walkLast = top
        walk.append((top.argmax, top.margin))
        try l.writeFloat16(row: 0, to: scratchFile)
    }
    var walkGenerated: [(Int32, Float)] = []
    for i in 0..<generatedTokens {
        let top = walkLast!
        walkGenerated.append((top.argmax, top.margin))
        if i + 1 < generatedTokens {
            let l = try await engine.step(top.argmax)
            walkLast = l.top2(row: 0)
        }
    }
    report.walkSeconds = seconds(ContinuousClock.now - t0)
    engine.reset()
    var i = 0
    var scratch = [Float16](repeating: 0, count: engine.vocabSize)
    let rowBytes = Int64(engine.vocabSize * MemoryLayout<Float16>.stride)
    let t1 = ContinuousClock.now
    func compare(_ pos: Int, _ logits: Logits, _ row: Int) throws -> Logits.Top2 {
        report.positions += 1
        let top = logits.top2(row: row)
        if top.argmax == walk[pos].0 {
            report.agreed += 1
        } else {
            report.disagreements.append(.init(position: pos, chunkArgmax: top.argmax, walkArgmax: walk[pos].0,
                                              chunkMargin: top.margin, walkMargin: walk[pos].1))
        }
        let worst = try logits.maxAbsDifference(row: row, from: scratchFile,
                                                offset: Int64(pos) * rowBytes, scratch: &scratch)
        if worst > report.worstDelta { report.worstDelta = worst; report.worstDeltaPosition = pos }
        return top
    }
    var chunkLast: Logits.Top2?
    let chunkSizes = requestedChunkSizes ?? engine.chunkSizes
    while let size = chunkSizes.first(where: { $0 <= ids.count - 1 - i }) {
        let l = try await engine.prefillChunk(ids[i ..< i + size])
        for j in 0..<size { chunkLast = try compare(i + j, l, j) }
        report.chunkPlan.append(size)
        i += size
    }
    report.chunkSeconds = seconds(ContinuousClock.now - t1)
    while i < ids.count {
        let l = try await engine.step(ids[i])
        chunkLast = try compare(i, l, 0)
        i += 1
    }
    for j in walkGenerated.indices {
        let top = chunkLast!
        report.generatedPositions += 1
        if top.argmax == walkGenerated[j].0 {
            report.generatedAgreed += 1
        } else {
            report.generatedDisagreements.append(.init(
                position: j, chunkArgmax: top.argmax, walkArgmax: walkGenerated[j].0,
                chunkMargin: top.margin, walkMargin: walkGenerated[j].1))
        }
        if j + 1 < walkGenerated.count {
            // Teacher-force the walk path's choice after a miss so both sides keep the same
            // token context and later positions remain independently meaningful.
            let l = try await engine.step(walkGenerated[j].0)
            chunkLast = l.top2(row: 0)
        }
    }
    return report
}
