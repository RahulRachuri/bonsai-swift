# Security

This repository is a native Swift host for the Bonsai 2 27B Core AI port: bundle loading,
the tokenizer and chat template, the prefill and decode loop, and the sampling around
them. It distributes no weights and no bundles; the conversion recipe that produces the
bundles is published separately in the community Core AI model zoo.

As with parakeet-swift, there are two questions: whether this code decodes correctly,
and whether the artifacts it loaded are the ones that were gated.

## Artifact provenance

The canonical artifact is published separately at
`rahulrachuri/ternary-bonsai-2-27b-coreai`. Each host release identifies the immutable Hugging Face
revision it was verified against. The artifact is derived from:

- `prism-ml/Ternary-Bonsai-2-27B-gguf`, revision
  `6ed5e12bf84b7a63069882c91dd9e9218647d17b`, PQ2_0 language weights.
- `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, tokenizer and
  chat template lineage.
- `coreai-core 1.0.0b2` for the portable graph and `coreai-build-3600.82.1` for the published
  M4 Pro `h16s` compile.

This host does not download, authenticate, or sandbox model artifacts. `--bundle` and
`BONSAI_BUNDLE` are trusted local inputs, and a Core AI bundle contains executable Metal code.
Download from the named repository, select the immutable revision recorded by the release, and
verify the Hub-provided hashes before loading it. A later push to the model repository must not
silently replace the revision a host release names.

Correctness evidence and its limits are recorded in the README. The authoritative public checks
are the token gate against PrismML's MLX runtime and the bounded chunk-vs-walk battery; plausible
output by itself is not evidence that a converted graph is correct.

## Reporting

Open a GitHub issue for anything that is not sensitive. For anything that is, use
GitHub's private vulnerability reporting on this repository.
