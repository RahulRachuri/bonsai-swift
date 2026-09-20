# bonsai-swift

A native Swift host for
[Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), PrismML's
ternary build of Qwen3.8-27B, running on Apple's Core AI runtime with no Python in the loop
at run time. The verified target is an Apple silicon Mac running macOS 27.

The host loads the three-function bundle, runs chunked prefill and decode, and generates
token-identically to PrismML's own MLX runtime on all three reference prompts. The published
bundle decodes at approximately 23.6 tok/s on an M4 Pro.

## Components

- **This repository:** bundle loading, tokenizer and chat-template integration, chunked prefill,
  greedy decoding, state management for the hybrid GatedDeltaNet layers, parity checks, and
  performance measurement.
- **Core AI model bundle:** the converted weights, tokenizer assets, portable `.aimodel`, and
  hardware-specific ahead-of-time compile are distributed separately on Hugging Face.
- **Conversion recipe:** Core AI embeds the custom ternary and Hadamard Metal kernels in the
  exported bundle. Their source and the export pipeline live in the
  [community Core AI model zoo](https://github.com/john-rocky/coreai-model-zoo).
- **Related project:** [parakeet-swift](https://github.com/RahulRachuri/parakeet-swift) provides
  a native Swift Core AI host for Parakeet ASR.

## Setup

### Requirements

- An Apple silicon Mac running macOS 27.
- The Xcode 27 toolchain and Swift 6.4 or newer.
- Roughly 14 GB of disk for the published directory: the 6.7 GB portable source graph plus
  an ahead-of-time M4 Pro compile. The running host is about 7.5 to 8 GB resident, before other
  applications and system caches.
- The converted Core AI bundle. Model weights are never stored in this Git repository.

The release build and measurements in this repository used an M4 Pro, macOS 27.0 build `26A428`,
Xcode 27.0 build `27A5237l`, and Apple Swift 6.4.

### Build

Build with the Xcode 27 toolchain on macOS 27:

```bash
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
swift build -c release
```

The binary is `.build/release/bonsai-swift`.

### Get the model

The release bundle is published separately at
[`rahulrachuri/ternary-bonsai-2-27b-coreai`](https://huggingface.co/rahulrachuri/ternary-bonsai-2-27b-coreai).
It is derived from PrismML's PQ2_0 checkpoint and contains the tokenizer, the portable `.aimodel`,
and an `h16s` ahead-of-time compile for the M4 Pro. The Hugging Face model card records the exact
source revisions, conversion recipe, checksums, and gate results.

This release is pinned to immutable artifact revision
[`0391d83323b76e58fe3167caef5a3ce994e5d4aa`](https://huggingface.co/rahulrachuri/ternary-bonsai-2-27b-coreai/tree/0391d83323b76e58fe3167caef5a3ce994e5d4aa):

```bash
hf download rahulrachuri/ternary-bonsai-2-27b-coreai \
  --revision 0391d83323b76e58fe3167caef5a3ce994e5d4aa \
  --local-dir bonsai2_27b_decode_pq2_0_pf64
```

The expected directory is the model repository root:

```text
bonsai2_27b_decode_pq2_0_pf64/
  metadata.json
  bonsai2_27b_decode_pq2_0_pf64.aimodel/
  aot_mac/
    bonsai2_27b_decode_pq2_0_pf64.h16s.aimodelc/
  tokenizer/
```

You can also reproduce it from the conversion recipe in the
[community Core AI model zoo](https://github.com/john-rocky/coreai-model-zoo). The host accepts
either bundle by `--bundle DIR` or `BONSAI_BUNDLE`; it never downloads weights implicitly.

### Run

```bash
export BONSAI_BUNDLE=/path/to/bonsai2_27b_decode_pq2_0_pf64
.build/release/bonsai-swift chat --prompt "The capital of France is" --new 64
```

Commands:

```
probe                                      entrypoints, states and selected asset
chat --prompt TEXT [--new N]               greedy text generation
bench [--prompt TEXT] [--new N]            chunked/walked prefill and decode speed
prompt-info --prompts FILE                  tokenize a prompt battery without loading the model
parity --ref FILE                           gate against an MLX reference fixture
parity --self --prompts FILE --only NAME    one bounded chunk-vs-walk battery case
```

The host prefers the ahead-of-time compile for this chip when the bundle carries one. On the
macOS 27 beta that is not optional: the JIT load of the source graph still tries to place the
custom kernels on the Neural Engine and dies in the command buffer. Compile once with
`coreai-build compile --platform macOS --architecture <chip> --preferred-compute gpu
--expect-frequent-reshapes` and the `.aimodelc` loads in about a second after the first run.

### Use as a package

`BonsaiKit` is a library product, so another Swift package can use the host without invoking the
CLI. The application still supplies an explicitly downloaded bundle directory.

```swift
.package(url: "https://github.com/RahulRachuri/bonsai-swift", from: "0.1.0")
// target dependency: .product(name: "BonsaiKit", package: "bonsai-swift")
```

```swift
import BonsaiKit
import Foundation

let bundle = try BonsaiBundle(directory: URL(fileURLWithPath: "/path/to/bundle"))
let tokenizer = try await BonsaiTokenizer.load(directory: bundle.tokenizerDirectory)
let engine = try await BonsaiEngine(bundle: bundle)
let prompt = try tokenizer.chatPrompt("The capital of France is")
let result = try await engine.generate(prompt: prompt, maxNew: 64, stop: tokenizer.stopIds)
let text = tokenizer.decode(result.tokens)
```

## How the host drives the bundle

The bundle has three functions over one set of packed weights and four state tensors:

```
main     input_ids [1,1]    position_ids [1,?]  -> logits [1,1,V]
prefill  input_ids [1,64]   position_ids [1,?]  -> logits [1,64,V]
prefill16 input_ids [1,16]  position_ids [1,?]  -> logits [1,16,V]
states   keyCache, valueCache [16,1,4,?,256]   convState [48,1,10240,3]   recState [48,1,48,128,128]
```

`BonsaiEngine` allocates the states once (the KV sequence axis at `--kv`, default 2048; the
export traced that axis between 2048 and the bundle's max context, and the host refuses
anything outside it) and hands the same buffers to every run; the graph reads its caches by
position and writes the tail. A prompt goes through the largest prefill chunks that fit (64,
then 16), then the remainder one token at a time through `main`, and the last prompt token
always through `main` so decode starts from the same row either way. Greedy decode reads the fp16
logits directly from the runtime buffer for the argmax.

The host uses the low-level `CoreAI` framework rather than Apple's pipelined engine. This exposes
every position's logits for parity checks and adds little overhead relative to reading roughly
7 GB of packed weights for each decoded token.

With Swift 6.4, the runtime's mutable views cannot be taken on a class property across the
`await` on a run
("lifetime-dependent variable escapes its scope"), even though Apple's own sequential engine
is written that way. The engine swaps each state buffer into a local for the duration of the
call and back afterwards; a swap moves the handle, not the bytes.

## Validation

`parity` teacher-forces the bundle through a greedy decode recorded from PrismML's MLX runtime.
The fixtures under `reference/` are produced by the model zoo's `mlx_reference.py`. The chat
template must render the reference prompt IDs exactly, then every position's argmax is compared.
After the prompt, the reference token is fed regardless of the bundle's prediction so every step
uses the same context. The gate uses the same 64-token, 16-token, and single-token chunk planner
as `chat`. Each single-token step also compares the graph's `next_token` output with the host's
argmax over the same logits.

| prompt (chat template, thinking on) | prefill | positions | generated |
|---|---|---:|---|
| "The capital of France is", 24 new | 16+16+16, 9 walked | 77/80 | all 24 identical: `Paris` |
| train time arithmetic, 48 new | 64+16, 3 walked | 127/130 | all 48 identical: `6:15 pm` |
| lighthouse passage, 125 tokens, 40 new | 64+16+16+16, 13 walked | 161/164 | all 40 identical |
| same, prompt walked at S=1 | walk | 161/164 | all 40 identical |

Graph and host argmax agreed at every walked step of every fixture (134 steps), and on a
24-token chat run under `BONSAI_CHECK_ARGMAX=1`. The published bundle retains the GDN decay
parameters (`A_log`, `dt_bias`) in fp32; the table above reports results from that bundle.

The three misses occur at the same positions in every run, all inside the chat template's
system prompt, where the reference's top-two margin is 0.002 to 0.024: the fp16 tie band.
The Python check of the same bundle reports the same positions.

### Broad chunk-vs-walk battery

The self-check also covers ten chat-templated prompts from 53 to 848 tokens: prose, code,
numbers, repeated words and characters, newlines, CJK, mixed scripts, and emoji. Each case runs
in a fresh process because a full-vocabulary comparison has 248,320 logits per position and Core
AI retains some per-run allocations; the fp16 reference rows are spooled to a non-cached scratch
file and compared through one reusable row buffer instead of being widened and retained in RAM.

Across 4,405 prompt positions, chunked prefill and an S=1 walk agree on 4,347 argmaxes. The
differences concentrate in repeated/newline/high-ID-token stress cases and show the expected
numeric drift between the chunk scan and fused step paths. The final states produce **80/80
identical continuation tokens** (eight per prompt, teacher-forced after any miss
so each later comparison keeps the same context). On the M4 Pro each isolated case held the host
near 7.5 GB resident and did not grow swap. The three MLX-reference gates above remain the
authoritative cross-runtime correctness result.

The 419-position `mixed_scripts` case was also evaluated with controlled state and execution-path
variants. The chunk scan retains recurrent GDN state in fp32 for the full chunk before writing
fp16 state, while an S=1 walk writes fp16 state after every token. Explicit per-token rounding
improved a 4-layer S=64 control from 418/419 to 419/419. On the full 64-layer model it improved
the fused-main comparison from 399/419 to 407/419; using the same GDN path on both sides reached
413/419 at S=64 and 410/419 at S=16. Every variant still agreed on 8/8 continuation tokens.
The remaining non-monotonic differences are consistent with shape-dependent floating-point
reduction order, including q/k normalization. Per-token rounding is therefore retained as a
diagnostic rather than enabled in the distributed graph.

## Performance

Measurements use the release build on an M4 Pro with a Python-free generation loop.

| workload | result |
|---|---:|
| decode | 23.6 tok/s |
| 114-token prompt, end to end | 48 tok/s |
| S=64 prefill chunk | 59 tok/s |
| S=1 prompt walk | 23.6 tok/s |

The packed ternary matvec reads at approximately 220 to 240 GB/s. Four 2-bit codes are converted
per instruction through the GPU's unorm unpack path, and the ternary mapping's subtraction is
folded into one activation sum. Decode also fuses the GatedDeltaNet scan and its small projections,
residual normalization with the Hadamard transform, SwiGLU with its transform, and matrix groups
that share an activation.

The 16-token and 64-token prefill entrypoints reduce prompt-tail work: a 125-token prompt is split
into 64 + 16 + 16 + 16 tokens followed by 13 single-token steps. For greedy decode, the graph's
`next_token` output avoids a separate host-side scan of the vocabulary logits.

A decoded token takes approximately 42 ms: about 30 ms for ternary matvecs, about 2 ms for fused
kernels, approximately 3.5 ms at the tail of Core AI's CPU-side encode work, and the remaining
time in runtime and command-buffer overhead. The theoretical matvec bandwidth floor on the tested
M4 Pro is approximately 28 ms per token.

## Limitations

- The published `h16s` AOT asset targets the M4 Pro. Another Apple silicon architecture can use a
  matching AOT compile produced from the portable source graph; on the tested macOS 27 beta, the
  source graph's JIT path does not place the custom kernels correctly.
- The exported language model is text-only. PrismML's optional vision projection was not ported.
- The exported context limit is 4,096 tokens, not the upstream model's maximum. The host's KV
  capacity defaults to and cannot be lower than 2,048.
- Generation is greedy. This is a low-level host and parity harness, not an OpenAI-compatible
  server or a general sampling framework.

## Licence

Apache License 2.0. See `NOTICE` for what this repository does and does not distribute.
