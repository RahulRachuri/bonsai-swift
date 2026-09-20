import CoreAI
import Foundation

/// Owns one loaded Bonsai bundle and the four state tensors of a hybrid Qwen3.5-family
/// decoder, and drives its two entrypoints.
///
/// The graph contract (from `coreai-build inspect` on the shipped bundle):
///
///     main     input_ids [1,1]   position_ids [1,?]  -> logits [1,1,V]
///     prefill  input_ids [1,C]   position_ids [1,?]  -> logits [1,C,V]
///     states   keyCache/valueCache [16,1,4,?,256]  convState [48,1,10240,3]  recState [48,1,48,128,128]
///
/// States are mutated in place by the runtime and persist across calls, so this class
/// allocates them once and hands the same buffers to every run. `position_ids` always
/// carries the full `[0, total)`; the graph reads the caches by position and writes the
/// tail, so a call at position p after p tokens have been processed is a decode step, and
/// a `prefill` call at p with C new tokens is one chunk. Everything is single-threaded and
/// strictly ordered: `await` suspends on GPU completion, and no two runs are in flight.
///
/// Why this is a hand-rolled loop and not Apple's pipelined engine: the token-identity
/// gate needs every position's logits, which that engine does not expose, and at 27B the
/// per-token host overhead the engine hides is small next to the 7 GB of weights each token
/// reads. The engine path stays an option for the perf rounds.
public final class BonsaiEngine: @unchecked Sendable {
    public let bundle: BonsaiBundle
    public let assetURL: URL
    public let vocabSize: Int
    /// Static query length of the largest `prefill` entrypoint, nil when the bundle has none.
    public let chunk: Int?
    /// Every prefill length loaded, largest first.
    public let chunkSizes: [Int]
    /// `main`'s greedy-choice output, when the graph emits one; enables the streamed decode.
    public let nextTokenName: String?
    /// The `next_token` output of the last `main` call, when the bundle has one (see `run`).
    public private(set) var lastNextToken: Int32?
    /// Sequence capacity the KV caches were allocated at, between `tracedKVMinimum` and the
    /// bundle's `max_context_length` (the export's traced range for that axis).
    public let kvCapacity: Int
    /// Highest `total` any call may reach: the KV capacity, but never `max_context_length`
    /// itself, because `position_ids` was traced one shorter than the KV axis (positions run
    /// `[0, max_ctx - 1]`, the cache holds `max_ctx` slots).
    public let positionLimit: Int
    /// The export's traced minimum for the KV sequence axis (`TRACE_KV_CACHE_SEQ_LEN`); a
    /// smaller allocation is outside the graph's contract.
    public static let tracedKVMinimum = 2048
    public let loadSeconds: Double
    public private(set) var processed = 0

    private let model: AIModel
    private let main: InferenceFunction

    private var prefills: [Int: Prefill] = [:]
    private struct Prefill {
        let function: InferenceFunction
        var input: NDArray
        var logits: NDArray
    }
    private let names: Names
    private let inputIdsDescriptor: NDArrayDescriptor
    private let positionIdsDescriptor: NDArrayDescriptor
    private let logitsDescriptor: NDArrayDescriptor

    private var keyCache: NDArray
    private var valueCache: NDArray
    private var convState: NDArray
    private var recState: NDArray
    /// Resolved descriptors of the four states, to re-allocate one that the streamed loop
    /// could not hand back (see `generateStreamed`).
    private let stateDescriptors: [String: NDArrayDescriptor]
    private var inputOne: NDArray
    private var logitsOne: NDArray

    private struct Names {
        var inputIds = "input_ids", positionIds = "position_ids", logits = "logits"
        var keyCache = "keyCache", valueCache = "valueCache", convState = "convState", recState = "recState"
    }

    public struct Options: Sendable {
        /// Explicit asset to load instead of the bundle's own choice (the AOT compile for
        /// this chip if present, else the source `.aimodel`).
        public var asset: URL? = nil
        public var kvCapacity = 2048
        public init() {}
    }

    public init(bundle: BonsaiBundle, options: Options = Options()) async throws {
        let start = ContinuousClock.now
        self.bundle = bundle
        self.vocabSize = bundle.vocabSize
        guard options.kvCapacity >= Self.tracedKVMinimum, options.kvCapacity <= bundle.maxContextLength else {
            throw BonsaiError.message("kv capacity \(options.kvCapacity) is outside the graph's traced range "
                + "[\(Self.tracedKVMinimum), \(bundle.maxContextLength)]")
        }
        self.kvCapacity = options.kvCapacity
        self.positionLimit = min(options.kvCapacity, bundle.maxContextLength - 1)

        let asset = options.asset ?? bundle.compiledAsset() ?? bundle.sourceAsset
        self.assetURL = asset
        // A compiled bundle fixed its compute placement at build time and loads with the
        // default options. A source bundle needs the GPU stated, or the runtime tries the
        // Neural Engine first. Reshapes are frequent by construction: position_ids grows
        // by one every step.
        let specialization: SpecializationOptions
        if asset.pathExtension == "aimodelc" {
            specialization = .default
        } else {
            var s = SpecializationOptions(preferredComputeUnitKind: .gpu)
            s.expectFrequentReshapes = true
            specialization = s
        }
        self.model = try await AIModel(contentsOf: asset, options: specialization)

        guard let descriptor = model.functionDescriptor(for: "main"),
              let main = try model.loadFunction(named: "main") else {
            throw BonsaiError.message("\(asset.lastPathComponent): no 'main' function")
        }
        self.main = main
        // Every tensor is looked up by name; the order the runtime reports them in is not
        // part of the contract, and reading the logits as the `[1,1]` next token would not
        // fail, it would decode garbage.
        let names = Names()
        guard descriptor.inputNames.count == 2, (1...2).contains(descriptor.outputNames.count),
              descriptor.stateNames.count == 4 else {
            throw BonsaiError.message("unexpected signature: inputs \(descriptor.inputNames) "
                + "outputs \(descriptor.outputNames) states \(descriptor.stateNames)")
        }
        for expected in [names.inputIds, names.positionIds] where !descriptor.inputNames.contains(expected) {
            throw BonsaiError.message("input '\(expected)' missing; bundle has \(descriptor.inputNames)")
        }
        guard descriptor.outputNames.contains(names.logits) else {
            throw BonsaiError.message("output '\(names.logits)' missing; bundle has \(descriptor.outputNames)")
        }
        for expected in [names.keyCache, names.valueCache, names.convState, names.recState]
        where !descriptor.stateNames.contains(expected) {
            throw BonsaiError.message("state '\(expected)' missing; bundle has \(descriptor.stateNames)")
        }
        // The greedy-choice output is named by the metadata; a second output the metadata
        // does not announce is still accepted, by elimination, for bundles written before
        // `next_token_output` existed.
        let extraOutputs = descriptor.outputNames.filter { $0 != names.logits }
        if let announced = bundle.nextTokenOutput {
            guard extraOutputs == [announced] else {
                throw BonsaiError.message("metadata names next-token output '\(announced)' but main's "
                    + "outputs are \(descriptor.outputNames)")
            }
            self.nextTokenName = announced
        } else {
            self.nextTokenName = extraOutputs.first
        }
        self.names = names

        guard case .ndArray(let inputIdsDescriptor)? = descriptor.inputDescriptor(of: names.inputIds),
              case .ndArray(let positionIdsDescriptor)? = descriptor.inputDescriptor(of: names.positionIds),
              case .ndArray(let logitsDescriptor)? = descriptor.outputDescriptor(of: names.logits) else {
            throw BonsaiError.message("inputs/outputs are not NDArrays")
        }
        guard logitsDescriptor.scalarType == .float16 else {
            throw BonsaiError.message("logits are \(logitsDescriptor.scalarType), expected float16")
        }
        self.inputIdsDescriptor = inputIdsDescriptor
        self.positionIdsDescriptor = positionIdsDescriptor
        self.logitsDescriptor = logitsDescriptor

        var stateDescriptors: [String: NDArrayDescriptor] = [:]
        func state(_ name: String) throws -> NDArray {
            guard case .ndArray(let d)? = descriptor.stateDescriptor(of: name) else {
                throw BonsaiError.message("state '\(name)' is not an NDArray")
            }
            guard d.scalarType == .float16 else {
                throw BonsaiError.message("state '\(name)' is \(d.scalarType), expected float16")
            }
            // Dynamic dimensions come back negative. The KV sequence axis is the only one the
            // export leaves dynamic; anything else would be a contract change, so refuse it
            // rather than silently size it to the KV capacity.
            let dynamic = d.shape.filter { $0 < 0 }.count
            guard dynamic <= 1 else {
                throw BonsaiError.message("state '\(name)' has \(dynamic) dynamic dimensions; expected at most one")
            }
            let resolved = d.resolvingDynamicDimensions(d.shape.map { $0 < 0 ? options.kvCapacity : $0 })
            stateDescriptors[name] = resolved
            var array = NDArray(descriptor: resolved)
            zeroFloat16(&array)
            return array
        }
        self.keyCache = try state(names.keyCache)
        self.valueCache = try state(names.valueCache)
        self.convState = try state(names.convState)
        self.recState = try state(names.recState)
        self.stateDescriptors = stateDescriptors

        self.inputOne = NDArray(descriptor: inputIdsDescriptor.resolvingDynamicDimensions([1, 1]))
        self.logitsOne = NDArray(descriptor: logitsDescriptor.resolvingDynamicDimensions([1, 1, vocabSize]))

        var sizes: [Int] = []
        for (i, size) in bundle.prefillChunks.enumerated() {
            let fname = i == 0 ? "prefill" : "prefill\(size)"
            guard model.functionNames.contains(fname),
                  let pf = try model.loadFunction(named: fname),
                  let pd = model.functionDescriptor(for: fname),
                  case .ndArray(let pin)? = pd.inputDescriptor(of: names.inputIds),
                  case .ndArray(let plog)? = pd.outputDescriptor(of: names.logits) else {
                throw BonsaiError.message("metadata lists prefill chunk \(size) but function '\(fname)' is missing")
            }
            guard pin.shape == [1, size] else {
                throw BonsaiError.message("\(fname) input_ids is \(pin.shape); metadata says chunk \(size)")
            }
            prefills[size] = Prefill(function: pf, input: NDArray(descriptor: pin),
                                     logits: NDArray(descriptor: plog.resolvingDynamicDimensions([1, size, vocabSize])))
            sizes.append(size)
        }
        self.chunkSizes = sizes.sorted(by: >)
        self.chunk = chunkSizes.first
        self.loadSeconds = seconds(ContinuousClock.now - start)
    }

    public var functionNames: [String] { model.functionNames }

    /// The chip the runtime reports, which names the AOT compile it would load.
    public static var architecture: String { AIModel.deviceArchitectureName }

    /// One line per input, state and output of a function, the way `coreai-build inspect`
    /// prints them (dynamic dimensions as `?`).
    public func signature(of function: String) -> [String] {
        guard let d = model.functionDescriptor(for: function) else { return [] }
        func describe(_ v: InferenceValue.Descriptor?) -> String {
            guard case .ndArray(let a)? = v else { return "?" }
            let dims = a.shape.map { $0 < 0 ? "?" : String($0) }.joined(separator: " x ")
            return "\(a.scalarType) [\(dims)]"
        }
        var lines: [String] = []
        for n in d.inputNames { lines.append("  in    \(n): \(describe(d.inputDescriptor(of: n)))") }
        for n in d.stateNames { lines.append("  state \(n): \(describe(d.stateDescriptor(of: n)))") }
        for n in d.outputNames { lines.append("  out   \(n): \(describe(d.outputDescriptor(of: n)))") }
        return lines
    }

    /// Forget the sequence: zero the four states and rewind the position counter.
    public func reset() {
        zeroFloat16(&keyCache)
        zeroFloat16(&valueCache)
        zeroFloat16(&convState)
        zeroFloat16(&recState)
        processed = 0
    }

    // MARK: - The two entrypoints

    /// One decode step: `token` at position `processed`. Returns its next-token logits.
    public func step(_ token: Int32) async throws -> Logits {
        try ensureCapacity(processed + 1)
        let t0 = ContinuousClock.now
        fillInt32(&inputOne, CollectionOfOne(token))
        try await withLocal(&logitsOne) { out in
            try await run(main, input: inputOne, output: &out, total: processed + 1)
        }
        processed += 1
        Self.timing?.runSeconds += seconds(ContinuousClock.now - t0)
        Self.timing?.steps += 1
        return Logits(array: logitsOne, rows: 1, vocab: vocabSize)
    }

    /// Optional wall-time accounting for the decode loop (`BONSAI_TIMING=1`): how much of a
    /// step is the runtime call itself versus the host's argmax over the logits.
    public final class Timing: @unchecked Sendable {
        public var runSeconds = 0.0
        public var argmaxSeconds = 0.0
        public var steps = 0
    }
    public static let timing: Timing? = ProcessInfo.processInfo.environment["BONSAI_TIMING"] != nil ? Timing() : nil

    /// One prefill chunk: exactly one of the bundle's static lengths, starting at position
    /// `processed`. Returns the logits of every position in the chunk, row `j` for token `j`.
    public func prefillChunk(_ tokens: ArraySlice<Int32>) async throws -> Logits {
        let n = tokens.count
        guard prefills[n] != nil else {
            throw BonsaiError.message("no prefill entrypoint for \(n) tokens (have \(chunkSizes))")
        }
        try ensureCapacity(processed + n)
        fillInt32(&prefills[n]!.input, tokens)
        let function = prefills[n]!.function
        let input = prefills[n]!.input
        try await withLocal(&prefills[n]!.logits) { out in
            try await run(function, input: input, output: &out, total: processed + n)
        }
        processed += n
        return Logits(array: prefills[n]!.logits, rows: n, vocab: vocabSize)
    }

    private func ensureCapacity(_ total: Int) throws {
        guard total <= positionLimit else {
            throw BonsaiError.message("position \(total) exceeds the limit \(positionLimit) "
                + "(kv capacity \(kvCapacity), max context \(bundle.maxContextLength))")
        }
    }

    /// The runtime's mutable views are non-escapable and must outlive the `await` on the
    /// run, which a view taken on a class property cannot (its scoped access ends at the
    /// statement). So every buffer the runtime writes is swapped out of its stored slot
    /// into a local for the duration of the call and swapped back afterwards. A swap moves
    /// the handle, never the bytes; the placeholder left in the slot is one fp16 scalar.
    private func withLocal(_ slot: inout NDArray,
                           _ body: (inout NDArray) async throws -> Void) async throws {
        var local = NDArray(shape: [1], scalarType: .float16)
        swap(&local, &slot)
        defer { swap(&local, &slot) }
        try await body(&local)
    }

    private func run(_ function: InferenceFunction, input: NDArray, output: inout NDArray,
                     total: Int) async throws {
        var positions = NDArray(descriptor: positionIdsDescriptor.resolvingDynamicDimensions([1, total]))
        fillPositions(&positions, count: total)
        var k = NDArray(shape: [1], scalarType: .float16)
        var v = NDArray(shape: [1], scalarType: .float16)
        var c = NDArray(shape: [1], scalarType: .float16)
        var r = NDArray(shape: [1], scalarType: .float16)
        swap(&k, &keyCache); swap(&v, &valueCache); swap(&c, &convState); swap(&r, &recState)
        defer { swap(&k, &keyCache); swap(&v, &valueCache); swap(&c, &convState); swap(&r, &recState) }
        var states = InferenceFunction.MutableViews()
        states.insert(&k, for: names.keyCache)
        states.insert(&v, for: names.valueCache)
        states.insert(&c, for: names.convState)
        states.insert(&r, for: names.recState)
        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&output, for: names.logits)
        var outs = try await function.run(
            inputs: [names.inputIds: input, names.positionIds: positions],
            states: consume states,
            outputViews: consume outputs)
        // `main` also emits its greedy choice; keep it so the sync loop can skip the host argmax
        if let nextTokenName, let nd = outs.remove(nextTokenName)?.ndArray {
            lastNextToken = readInt32(nd)
        } else {
            lastNextToken = nil
        }
    }

    // MARK: - Prompt and generation

    public struct PrefillResult {
        /// Next-token logits after the whole prompt.
        public let last: Logits
        public let chunkedTokens: Int
        public let walkedTokens: Int
        public let chunkSeconds: Double
        public let walkSeconds: Double
    }

    /// Process a prompt: whole chunks through `prefill` when `chunked` and the bundle has
    /// the entrypoint, then the remainder one token at a time through `main`. The last
    /// prompt token always goes through `main`, so the returned logits are a `[1,1,V]` row
    /// whichever path ran; that keeps the decode loop's first step identical to a pure walk.
    public func prefill(_ tokens: [Int32], chunked: Bool = true) async throws -> PrefillResult {
        precondition(!tokens.isEmpty)
        var i = 0
        var chunkSeconds = 0.0
        if chunked, !chunkSizes.isEmpty {
            let t0 = ContinuousClock.now
            // Greedy: the largest chunk that leaves at least the last prompt token for `main`.
            while let size = chunkSizes.first(where: { $0 <= tokens.count - 1 - i }) {
                _ = try await prefillChunk(tokens[i ..< i + size])
                i += size
            }
            chunkSeconds = seconds(ContinuousClock.now - t0)
        }
        let chunkedTokens = i
        let t1 = ContinuousClock.now
        var last: Logits? = nil
        while i < tokens.count {
            last = try await step(tokens[i])
            i += 1
        }
        return PrefillResult(last: last!, chunkedTokens: chunkedTokens,
                             walkedTokens: tokens.count - chunkedTokens,
                             chunkSeconds: chunkSeconds, walkSeconds: seconds(ContinuousClock.now - t1))
    }

    public struct Generation {
        public var tokens: [Int32] = []
        public var prefill: PrefillResult
        public var decodeSeconds = 0.0
        public var stoppedOnEOS = false
    }

    /// Greedy generation with the graph choosing each token on the GPU and the host encoding
    /// the next step before the current one finishes.
    ///
    /// The sync loop pays a host round trip per token (await completion, read the choice,
    /// encode the next step) during which the GPU idles; measured at 3.9 ms of a 46 ms token.
    /// Here `main`'s `next_token` output is handed to the next step's `input_ids` as an async
    /// value, so step N+1 is encoded onto the compute stream while N runs and the GPU never
    /// waits on the host. The host reads each token one step late for streaming and the stop
    /// check, so at most two steps are in flight; on a stop, one extra step has already been
    /// encoded and its position is discarded (the sequence is not continued afterwards).
    ///
    /// The prompt goes through the sync path first; the returned tokens are the same greedy
    /// choices the sync loop makes (a tie between two logits could in principle resolve
    /// differently on the GPU; `chat --compare` checks the two streams agree).
    ///
    /// Measured outcome on the M4 Pro (2026-09-18): no gain. `encode(to:)` returns only when
    /// the previous step's GPU work is done (46 ms per call, the token await then 0.02 ms),
    /// with tracked or untracked state buffers, one or two loaded instances of `main`, one or
    /// two streams, chained or fresh inputs. The runtime paces its encoding by the GPU for a
    /// graph this size, so the 3.9 ms per-step setup inside it cannot be hidden from here.
    /// Kept as an option (`chat --stream`) because it is validated and costs nothing.
    public func generateStreamed(prompt: [Int32], maxNew: Int, chunked: Bool = true,
                                 stop: Set<Int32> = [], onToken: ((Int32) -> Void)? = nil) async throws -> Generation {
        guard let nextTokenName else { throw BonsaiError.message("bundle's main has no next_token output") }
        reset()
        let pre = try await prefill(prompt, chunked: chunked)
        var gen = Generation(prefill: pre)
        let t0 = ContinuousClock.now
        let first = pre.last.argmax(row: 0)
        gen.tokens.append(first)
        onToken?(first)
        if stop.contains(first) { gen.stoppedOnEOS = true }
        if gen.stoppedOnEOS || maxNew <= 1 {
            gen.decodeSeconds = seconds(ContinuousClock.now - t0)
            return gen
        }
        let stream = ComputeStream()

        // States move out of their slots into async mutable values for the loop, and back after.
        var k = NDArray(shape: [1], scalarType: .float16)
        var v = NDArray(shape: [1], scalarType: .float16)
        var c = NDArray(shape: [1], scalarType: .float16)
        var r = NDArray(shape: [1], scalarType: .float16)
        swap(&k, &keyCache); swap(&v, &valueCache); swap(&c, &convState); swap(&r, &recState)
        var kA = InferenceFunction.AsyncMutableValue(consume k)
        var vA = InferenceFunction.AsyncMutableValue(consume v)
        var cA = InferenceFunction.AsyncMutableValue(consume c)
        var rA = InferenceFunction.AsyncMutableValue(consume r)

        // `ids` is the input of the next step: the last token, resolved on the host or still
        // an unread choice of the step before (`unresolved`, the same value).
        var ids = InferenceFunction.AsyncValue(NDArray(scalars: [first], shape: [1, 1]))
        var unresolved: InferenceFunction.AsyncValue? = nil
        var encoded = 0
        var stopped = false
        var thrown: Error? = nil
        do {
            while !stopped && gen.tokens.count + (unresolved == nil ? 0 : 1) < maxNew {
                let total = processed + encoded + 1
                try ensureCapacity(total)
                var positions = NDArray(descriptor: positionIdsDescriptor.resolvingDynamicDimensions([1, total]))
                fillPositions(&positions, count: total)
                let tE = ContinuousClock.now
                var states = InferenceFunction.AsyncMutableViews()
                states.insert(&kA, for: names.keyCache)
                states.insert(&vA, for: names.valueCache)
                states.insert(&cA, for: names.convState)
                states.insert(&rA, for: names.recState)
                let outs = try main.encode(
                    inputs: [names.inputIds: ids, names.positionIds: InferenceFunction.AsyncValue(consume positions)],
                    states: consume states, to: stream)
                Self.timing?.runSeconds += seconds(ContinuousClock.now - tE)
                Self.timing?.steps += 1
                guard let next = outs[nextTokenName] else { throw BonsaiError.message("no '\(nextTokenName)' output") }
                encoded += 1
                // The step just encoded consumes the previous step's choice; read that choice
                // now, one step late, so two steps stay in flight.
                if let u = unresolved {
                    let tA = ContinuousClock.now
                    guard let nd = try await u.ndArray else { throw BonsaiError.message("next_token came back empty") }
                    Self.timing?.argmaxSeconds += seconds(ContinuousClock.now - tA)
                    let value = readInt32(nd)
                    gen.tokens.append(value)
                    onToken?(value)
                    if stop.contains(value) { stopped = true; gen.stoppedOnEOS = true }
                }
                unresolved = next
                ids = next
            }
            if !stopped, let u = unresolved {
                guard let nd = try await u.ndArray else { throw BonsaiError.message("next_token came back empty") }
                let value = readInt32(nd)
                gen.tokens.append(value)
                onToken?(value)
                if stop.contains(value) { gen.stoppedOnEOS = true }
            }
        } catch { thrown = error }
        await stream.currentWorkCompleted()
        // Move the states back into their slots whatever happened. On a stop, one extra step
        // was encoded; its position is counted so the caller sees where the state really is.
        // A state the stream cannot hand back (it faulted mid-loop) would otherwise leave the
        // one-scalar placeholder in its slot, which `reset` zeroes happily and the next run
        // feeds to the graph: re-allocate it, forget the sequence, and report the loss.
        var lost: [String] = []
        func restored(_ value: consuming InferenceFunction.AsyncMutableValue, _ name: String) async -> NDArray {
            if let nd = try? await value.ndArray { return nd }
            lost.append(name)
            var fresh = NDArray(descriptor: stateDescriptors[name]!)
            zeroFloat16(&fresh)
            return fresh
        }
        keyCache = await restored(kA, names.keyCache)
        valueCache = await restored(vA, names.valueCache)
        convState = await restored(cA, names.convState)
        recState = await restored(rA, names.recState)
        processed += encoded
        if !lost.isEmpty {
            reset()
            throw thrown ?? BonsaiError.message("streamed loop lost states \(lost); the sequence was reset")
        }
        if let thrown { throw thrown }
        gen.decodeSeconds = seconds(ContinuousClock.now - t0)
        if let t = Self.timing {
            print(String(format: "[timing/stream] %d encodes: encode %.2f ms/step, await token %.2f ms/step, loop total %.2f ms/token",
                         t.steps, t.runSeconds / Double(max(t.steps, 1)) * 1e3, t.argmaxSeconds / Double(max(t.steps, 1)) * 1e3,
                         gen.decodeSeconds / Double(max(gen.tokens.count, 1)) * 1e3))
        }
        return gen
    }

    /// `BONSAI_CHECK_ARGMAX=1`: also scan every step's logits on the host and report where the
    /// graph's `next_token` differs. Two argmaxes over the same fp16 row can only disagree on
    /// an exact tie, which the fp16 rounding makes routine near a margin of 0.01; the
    /// generation trusts the graph, the parity gate trusts the host, so this is how the two
    /// tie-breakers are checked against each other on a real run.
    public static let checkArgmax = ProcessInfo.processInfo.environment["BONSAI_CHECK_ARGMAX"] != nil

    /// Greedy generation on a fresh sequence. `onToken` sees each token as it is chosen.
    public func generate(prompt: [Int32], maxNew: Int, chunked: Bool = true,
                         stop: Set<Int32> = [], onToken: ((Int32) -> Void)? = nil) async throws -> Generation {
        reset()
        let pre = try await prefill(prompt, chunked: chunked)
        var gen = Generation(prefill: pre)
        var next = pre.last.argmax(row: 0)
        let t0 = ContinuousClock.now
        var checked = 0, mismatched = 0
        for _ in 0..<maxNew {
            gen.tokens.append(next)
            onToken?(next)
            if stop.contains(next) { gen.stoppedOnEOS = true; break }
            if gen.tokens.count == maxNew { break }
            let logits = try await step(next)
            let t1 = ContinuousClock.now
            next = lastNextToken ?? logits.argmax(row: 0)     // the graph's argmax when it has one
            Self.timing?.argmaxSeconds += seconds(ContinuousClock.now - t1)
            if Self.checkArgmax, let graph = lastNextToken {
                let host = logits.top2(row: 0)
                checked += 1
                if host.argmax != graph {
                    mismatched += 1
                    print(String(format: "[argmax] step %d: host %d vs graph %d (host margin %.4f)",
                                 processed, host.argmax, graph, host.margin))
                }
            }
        }
        if Self.checkArgmax { print("[argmax] graph vs host: \(mismatched) mismatches over \(checked) steps") }
        gen.decodeSeconds = seconds(ContinuousClock.now - t0)
        if let t = Self.timing {
            print(String(format: "[timing] %d steps: run %.2f ms/step, argmax %.2f ms/step, loop total %.2f ms/step",
                         t.steps, t.runSeconds / Double(max(t.steps, 1)) * 1e3, t.argmaxSeconds / Double(max(gen.tokens.count, 1)) * 1e3,
                         gen.decodeSeconds / Double(max(gen.tokens.count, 1)) * 1e3))
        }
        return gen
    }
}
