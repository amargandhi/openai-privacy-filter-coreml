# GitHub Project Backlog

These items are intended to become GitHub issues and Project items.

## Milestone 1: Conversion Proof

1. Create Python 3.12 conversion environment
   - Acceptance: `scripts/check_environment.py` passes with conversion dependencies.

2. Convert fixed 128-token Core ML package
   - Acceptance: `build/OpenAIPrivacyFilterLogits_128.mlpackage` and provenance sidecar are generated.

3. Compare PyTorch vs Core ML logits
   - Acceptance: parity JSON reports max/mean absolute differences and argmax agreement for all fixtures.

## Milestone 2: Baselines

4. Run MLX MXFP8 fixture inference
   - Acceptance: `reports/mlx-mxfp8.json` contains spans and expected/actual category comparison.

5. Run MLX BF16 fixture inference
   - Acceptance: `reports/mlx-bf16.json` exists and can be compared to MXFP8.

6. Add official CLI/Transformers fixture baseline
   - Acceptance: PyTorch or official OPF output is stored in the same JSON schema.
   - Status: Done with `scripts/run_transformers_privacy_filter.py`.

## Milestone 3: Native Runtime

7. Port tokenizer/offset handling to Swift
   - Acceptance: token IDs and offsets match Python tokenizer fixtures.
   - Status: Done with `PrivacyFilterTokenizer` SwiftPM target and Python-generated fixture tests.

8. Port BIOES/Viterbi decode to Swift
   - Acceptance: Swift decoder matches Python decoder on fixture logits.

9. Implement Manifold CoreMLPrivacyBackend
   - Acceptance: backend conforms to `PrivacyBackend` and returns `PrivacyScanResult`.

## Milestone 4: Product Readiness

10. Benchmark Core ML compute units
    - Acceptance: cold load, warm latency, and memory are reported for `.all`, `.cpuAndGPU`, and `.cpuAndNeuralEngine`.

11. Add install/provenance lifecycle
    - Acceptance: Manifold records model revision, conversion hash, and installed artifact state.

12. Document production limitations
    - Acceptance: README documents chunking, long-context limits, fallback behavior, and human-review caveats.
