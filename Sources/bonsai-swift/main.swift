import BonsaiKit
import Foundation

// bonsai-swift: a native Swift host for the Bonsai 2 27B Core AI bundle.
//
//   probe                          print the bundle's entrypoints, states and chosen asset
//   parity  --ref FILE [--walk]    teacher-forced token gate against an MLX reference decode
//   chat    --prompt TEXT [--new N] [--walk]   greedy answer through the chat template
//   bench   [--prompt TEXT] [--new N] [--rounds R]  prefill/decode tok/s, chunked vs walk
//
// The bundle directory comes from `--bundle DIR` or `BONSAI_BUNDLE`; nothing is compiled in.
// `--asset PATH` overrides the bundle's own choice of graph (the AOT compile for this chip
// when present, else the source .aimodel). `--kv N` sets the KV capacity (default 2048).

// Line-buffered stdout even when redirected to a file: gate reports are long-running and a
// crash (or a reboot) must not take the finished lines with it.
setlinebuf(stdout)

let argv = Array(CommandLine.arguments.dropFirst())
func flag(_ name: String) -> Bool { argv.contains("--\(name)") }
func option(_ name: String) -> String? {
    guard let i = argv.firstIndex(of: "--\(name)"), i + 1 < argv.count else { return nil }
    return argv[i + 1]
}
func intOption(_ name: String, _ fallback: Int) -> Int {
    guard let raw = option(name) else { return fallback }
    guard let value = Int(raw) else { fail("--\(name) needs an integer, got '\(raw)'") }
    return value
}
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("bonsai-swift: \(message)\n".utf8))
    exit(2)
}
func url(_ path: String) -> URL { URL(fileURLWithPath: (path as NSString).expandingTildeInPath) }
struct BatteryPrompt: Decodable { let name: String; let prompt: String }

let usage = """
bonsai-swift: Bonsai 2 27B on Apple Core AI

  probe                                   entrypoints, states, and which asset would load
  prompt-info --prompts FILE              tokenize a prompt battery without loading the model
  parity --ref FILE [--walk]              token-identity gate vs an MLX reference decode
  parity --self --ref FILE                chunked prefill vs this bundle's own S=1 walk
  parity --self --prompts FILE --only NAME [--chunk-size N]
                                           one bounded battery case; use a fresh process per case
  chat --prompt TEXT [--new N] [--walk] [--stream | --compare]
  bench [--prompt TEXT] [--new N] [--rounds R]

  common: --bundle DIR (or BONSAI_BUNDLE)  --asset PATH  --kv N (2048 ... max context)
  env:    BONSAI_TIMING=1 per-step timing   BONSAI_CHECK_ARGMAX=1 graph vs host argmax on chat
"""

guard let command = argv.first, command != "--help", command != "-h" else {
    print(usage)
    exit(0)
}

func loadBundle() -> BonsaiBundle {
    guard let dir = option("bundle") ?? ProcessInfo.processInfo.environment["BONSAI_BUNDLE"] else {
        fail("no bundle: pass --bundle DIR or set BONSAI_BUNDLE")
    }
    do { return try BonsaiBundle(directory: url(dir)) } catch { fail("\(error)") }
}

func loadEngine(_ bundle: BonsaiBundle) async -> BonsaiEngine {
    var options = BonsaiEngine.Options()
    options.asset = option("asset").map(url)
    options.kvCapacity = intOption("kv", 2048)
    let asset = options.asset ?? bundle.compiledAsset() ?? bundle.sourceAsset
    print("loading \(asset.lastPathComponent) (kv capacity \(options.kvCapacity)) ...")
    do {
        let engine = try await BonsaiEngine(bundle: bundle, options: options)
        print(String(format: "loaded in %.1f s; functions %@", engine.loadSeconds,
                     engine.functionNames.joined(separator: ", ")))
        return engine
    } catch { fail("load failed: \(error)") }
}

func rate(_ n: Int, _ s: Double) -> String { String(format: "%.1f tok/s", Double(n) / max(s, 1e-9)) }

let bundle = loadBundle()

switch command {
case "probe":
    print("bundle   \(bundle.name)  vocab \(bundle.vocabSize)  max context \(bundle.maxContextLength)")
    print("prefill  \(bundle.prefillChunk.map { "chunk \($0)" } ?? "none")")
    print("arch     \(BonsaiEngine.architecture)")
    print("asset    \((option("asset").map(url) ?? bundle.compiledAsset() ?? bundle.sourceAsset).path)")
    let engine = await loadEngine(bundle)
    for name in engine.functionNames {
        print("function \(name)")
        for line in engine.signature(of: name) { print(line) }
    }

case "prompt-info":
    guard let batteryPath = option("prompts") else { fail("prompt-info needs --prompts FILE") }
    let items: [BatteryPrompt]
    do { items = try JSONDecoder().decode([BatteryPrompt].self, from: Data(contentsOf: url(batteryPath))) }
    catch { fail("bad battery file: \(error)") }
    let tok: BonsaiTokenizer
    do { tok = try await BonsaiTokenizer.load(directory: bundle.tokenizerDirectory) }
    catch { fail("tokenizer: \(error)") }
    for item in items {
        let ids: [Int32]
        do { ids = try tok.chatPrompt(item.prompt) } catch { fail("chat template: \(error)") }
        print(String(format: "%-14@ %5d ids (max id %6d)", item.name, ids.count, ids.max() ?? 0))
    }

case "parity":
    if flag("self"), let batteryPath = option("prompts") {
        // Oracle-free battery: every prompt through the chunk plan and through the S=1 walk,
        // greedy tokens compared per position. This is the chunked-vs-walk gate the zoo asks
        // for on the GDN family, over content chosen to stress the scan (repetition, numbers,
        // code, non-Latin scripts), where a reference decode is not needed to spot a divergence.
        let decoded: [BatteryPrompt]
        do { decoded = try JSONDecoder().decode([BatteryPrompt].self, from: Data(contentsOf: url(batteryPath))) }
        catch { fail("bad battery file: \(error)") }
        guard let only = option("only") else {
            fail("battery cases must run in separate processes; pass --only NAME (list with prompt-info)")
        }
        let items = decoded.filter { $0.name == only }
        if items.isEmpty { fail("battery has no prompt named '\(only)'") }
        let tok: BonsaiTokenizer
        do { tok = try await BonsaiTokenizer.load(directory: bundle.tokenizerDirectory) } catch { fail("tokenizer: \(error)") }
        let engine = await loadEngine(bundle)
        guard !engine.chunkSizes.isEmpty else { fail("bundle has no prefill entrypoint") }
        let selectedChunkSizes: [Int]?
        if let raw = option("chunk-size") {
            guard let size = Int(raw), engine.chunkSizes.contains(size) else {
                fail("--chunk-size must name one of the bundle's chunk sizes: \(engine.chunkSizes)")
            }
            selectedChunkSizes = [size]
        } else {
            selectedChunkSizes = nil
        }
        var failed = 0, totalPositions = 0, totalAgreed = 0
        var totalGeneratedPositions = 0, totalGeneratedAgreed = 0
        for item in items {
            let ids: [Int32]
            do { ids = try tok.chatPrompt(item.prompt) } catch { fail("chat template: \(error)") }
            let r: SelfCheckReport
            do {
                r = try await selfCheck(engine: engine, ids: ids, generatedTokens: 8,
                                        chunkSizes: selectedChunkSizes)
            }
            catch { fail("self-check failed on \(item.name): \(error)") }
            totalPositions += r.positions; totalAgreed += r.agreed
            totalGeneratedPositions += r.generatedPositions; totalGeneratedAgreed += r.generatedAgreed
            let maxId = ids.max() ?? 0
            print(String(format: "%-14@ %5d ids (max id %6d) plan %@: prompt %d/%d, generated %d/%d, worst max|delta| %.4f @%d",
                         item.name, ids.count, maxId, "\(r.chunkPlan)", r.agreed, r.positions,
                         r.generatedAgreed, r.generatedPositions, r.worstDelta, r.worstDeltaPosition))
            for d in r.disagreements {
                print(String(format: "    position %d: chunk %d (margin %.4f) vs walk %d (margin %.4f)",
                             d.position, d.chunkArgmax, d.chunkMargin, d.walkArgmax, d.walkMargin))
            }
            for d in r.generatedDisagreements {
                print(String(format: "    generated %d: chunk %d (margin %.4f) vs walk %d (margin %.4f)",
                             d.position, d.chunkArgmax, d.chunkMargin, d.walkArgmax, d.walkMargin))
            }
            if r.agreed != r.positions || r.generatedAgreed != r.generatedPositions { failed += 1 }
        }
        print("battery: \(items.count - failed)/\(items.count) cases exact; prompt \(totalAgreed)/\(totalPositions), "
              + "generated \(totalGeneratedAgreed)/\(totalGeneratedPositions)")
        exit(failed == 0 ? 0 : 1)
    }
    guard let refPath = option("ref") else { fail("parity needs --ref FILE") }
    let ref: MLXReference
    do { ref = try MLXReference.load(url(refPath)) } catch { fail("bad reference: \(error)") }
    if flag("self") {
        let engine = await loadEngine(bundle)
        guard !engine.chunkSizes.isEmpty else { fail("bundle has no prefill entrypoint") }
        let r: SelfCheckReport
        do { r = try await selfCheck(engine: engine, ids: ref.ids) } catch { fail("self-check failed: \(error)") }
        let chunkedTokens = r.chunkPlan.reduce(0, +)
        print("chunk plan \(r.chunkPlan) + \(ref.ids.count - chunkedTokens) walked; chunks @ \(rate(chunkedTokens, r.chunkSeconds)), walk @ \(rate(ref.ids.count, r.walkSeconds))")
        print(String(format: "prefill vs walk over %d prompt positions: argmax %d/%d, worst max|delta| %.4f (position %d)", r.positions, r.agreed, r.positions, r.worstDelta, r.worstDeltaPosition))
        for d in r.disagreements {
            print(String(format: "  position %d: chunk argmax %d (margin %.4f) vs walk argmax %d (margin %.4f)",
                         d.position, d.chunkArgmax, d.chunkMargin, d.walkArgmax, d.walkMargin))
        }
        exit(r.agreed == r.positions ? 0 : 1)
    }
    let chunked = !flag("walk")
    print("reference: \(ref.ids.count) prompt ids, \(ref.generated.count) generated, "
          + "\(ref.steps.count) steps; prompt \(ref.prompt.prefix(60).debugDescription)")
    // Tokenizer gate first: the Swift chat template must reproduce the reference's ids.
    if ref.chat {
        do {
            let tok = try await BonsaiTokenizer.load(directory: bundle.tokenizerDirectory)
            let ours = try tok.chatPrompt(ref.prompt)
            if ours == ref.ids {
                print("tokenizer: chat template renders the reference's \(ref.ids.count) ids exactly")
            } else {
                let first = zip(ours, ref.ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                    ?? min(ours.count, ref.ids.count)
                print("tokenizer: MISMATCH, \(ours.count) ids vs reference \(ref.ids.count), first at \(first)")
            }
        } catch { print("tokenizer: could not load (\(error)); gating the graph on the reference ids only") }
    }
    let engine = await loadEngine(bundle)
    if chunked && engine.chunk == nil { print("bundle has no prefill entrypoint; walking") }
    let report: GateReport
    do {
        report = try await runGate(engine: engine, reference: ref, chunked: chunked) { print($0) }
    } catch { fail("gate failed: \(error)") }
    print("chunk plan \(report.chunkPlan); walk \(report.walkedTokens) prompt tokens @ \(rate(report.walkedTokens, report.walkSeconds)); "
          + "decode \(report.decodedTokens) @ \(rate(report.decodedTokens, report.decodeSeconds))")
    for m in report.argmaxMismatches {
        print(String(format: "  step %3d graph next_token %d vs host argmax %d (host margin %.4f)",
                     m.step, m.graph, m.host, m.margin))
    }
    print("graph vs host argmax: \(report.argmaxMismatches.count) mismatches over \(report.walkedTokens + report.decodedTokens) walked steps")
    for m in report.misses {
        print(String(format: "  step %3d %@: ours %d (margin %.3f) vs mlx %d (margin %.3f)",
                     m.step, m.inPrompt ? "prompt" : "gen   ", m.ours, m.ourMargin, m.ref, m.refMargin))
    }
    print("argmax agreement vs MLX reference: \(report.agreed)/\(report.positions)")
    print("generated \(report.generationIdentical ? "IDENTICAL" : "DIFFERS"): \(report.generated)")
    if !report.generationIdentical { print("reference: \(report.referenceGenerated)") }
    if let tok = try? await BonsaiTokenizer.load(directory: bundle.tokenizerDirectory) {
        print("text: \(tok.decode(report.generated).debugDescription)")
    }
    exit(report.generationIdentical ? 0 : 1)

case "chat":
    guard let prompt = option("prompt") else { fail("chat needs --prompt TEXT") }
    let tok: BonsaiTokenizer
    do { tok = try await BonsaiTokenizer.load(directory: bundle.tokenizerDirectory) } catch { fail("tokenizer: \(error)") }
    let ids: [Int32]
    do { ids = try tok.chatPrompt(prompt) } catch { fail("chat template: \(error)") }
    let engine = await loadEngine(bundle)
    print("prompt: \(ids.count) tokens")
    let streamed = flag("stream") || flag("compare")
    if streamed && engine.nextTokenName == nil { fail("bundle's main has no next_token output; the streamed loop needs it") }
    func report(_ gen: BonsaiEngine.Generation, _ label: String) {
        let p = gen.prefill
        print("\(label): prefill \(p.chunkedTokens) chunked @ \(rate(p.chunkedTokens, p.chunkSeconds)), "
              + "\(p.walkedTokens) walked @ \(rate(p.walkedTokens, p.walkSeconds)); "
              + "decode \(gen.tokens.count) @ \(rate(gen.tokens.count, gen.decodeSeconds))"
              + (gen.stoppedOnEOS ? " (eos)" : ""))
    }
    let new = intOption("new", 128)
    let printer: (Int32) -> Void = { t in print(tok.decode([t]), terminator: ""); fflush(stdout) }
    do {
        if flag("compare") {
            let sync = try await engine.generate(prompt: ids, maxNew: new, chunked: !flag("walk"), stop: tok.stopIds)
            let stream = try await engine.generateStreamed(prompt: ids, maxNew: new, chunked: !flag("walk"), stop: tok.stopIds)
            report(sync, "sync  ")
            report(stream, "stream")
            print("streamed tokens \(sync.tokens == stream.tokens ? "IDENTICAL" : "DIFFER") to the sync loop (\(sync.tokens.count) vs \(stream.tokens.count))")
            if sync.tokens != stream.tokens {
                print("sync:   \(sync.tokens)"); print("stream: \(stream.tokens)")
                exit(1)
            }
            print("text: \(tok.decode(stream.tokens).debugDescription)")
        } else if streamed {
            let gen = try await engine.generateStreamed(prompt: ids, maxNew: new, chunked: !flag("walk"), stop: tok.stopIds, onToken: printer)
            print(); report(gen, "stream")
        } else {
            let gen = try await engine.generate(prompt: ids, maxNew: new, chunked: !flag("walk"), stop: tok.stopIds, onToken: printer)
            print(); report(gen, "sync")
        }
    } catch { fail("generation failed: \(error)") }

case "bench":
    let tok: BonsaiTokenizer
    do { tok = try await BonsaiTokenizer.load(directory: bundle.tokenizerDirectory) } catch { fail("tokenizer: \(error)") }
    let prompt = option("prompt") ?? """
    Here is a short passage. The lighthouse keeper climbed the spiral stairs every evening, \
    counting each of the hundred and twelve steps as he had for thirty years, and lit the lamp \
    just as the last colour drained from the sea. Summarize the passage in one sentence and then \
    name the number of steps.
    """
    let ids: [Int32]
    do { ids = try tok.chatPrompt(prompt) } catch { fail("chat template: \(error)") }
    let new = intOption("new", 32)
    let rounds = intOption("rounds", 3)
    let engine = await loadEngine(bundle)
    print("prompt \(ids.count) tokens, \(new) new, \(rounds) rounds, arms interleaved")
    var arms: [(String, Bool, Bool)] = [("walk", false, false)]
    if engine.chunk != nil { arms.insert(("chunk", true, false), at: 0) }
    if engine.nextTokenName != nil { arms.insert(("stream", true, true), at: 0) }
    for round in 1...rounds {
        for (label, chunked, streamed) in arms {
            let gen: BonsaiEngine.Generation
            do {
                gen = streamed ? try await engine.generateStreamed(prompt: ids, maxNew: new, chunked: chunked)
                               : try await engine.generate(prompt: ids, maxNew: new, chunked: chunked)
            } catch { fail("bench failed: \(error)") }
            let p = gen.prefill
            let promptSeconds = p.chunkSeconds + p.walkSeconds
            var detail = "walked \(p.walkedTokens) @ \(rate(p.walkedTokens, p.walkSeconds))"
            if p.chunkedTokens > 0 {
                detail = "chunked \(p.chunkedTokens) @ \(rate(p.chunkedTokens, p.chunkSeconds)), " + detail
            }
            print(String(format: "round %d %-7@ prompt %@ (%@)  decode %@",
                         round, label, rate(ids.count, promptSeconds), detail,
                         rate(gen.tokens.count, gen.decodeSeconds)))
        }
    }

default:
    fail("unknown command '\(command)'\n\(usage)")
}
