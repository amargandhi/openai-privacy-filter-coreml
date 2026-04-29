# Mac Performance Summary

Short version: MLX MXFP8 is the best macOS runtime today. Core ML is correct,
portable, and useful for Apple-platform packaging, but the current Core ML graph
is still a correctness baseline, not the fast path.

## What We Measured

The measured machine was an M1 Max MacBook Pro with 32 GB memory. All numbers
below use 128-token chunks and the fixture scanner in this repo.

| Runtime | Artifact size | Warm latency | Read |
| --- | ---: | ---: | --- |
| MLX MXFP8 | about 1.47 GB | 16.84 ms/sample | Fastest path today. Best default for a macOS app that can ship an MLX runtime. |
| MLX BF16 | about 2.83 GB | 44.61 ms/sample | Accurate and simple, but larger and slower than MXFP8. |
| Core ML dense fp16 | about 2.80 GB | 96.88 ms/sample | Correct, but about 5.75x slower than MLX MXFP8 on this graph. |
| Core ML 8-bit palettized | about 1.40 GB | 148.97 ms/sample | Keeps agreement, but is slower. Useful only when package size matters more than latency. |
| Core ML linear int8 | about 1.40 GB | rejected | Accelerated execution failed; CPU-only parity was poor. |

All accepted Core ML runs kept token argmax agreement at `1.0` against MLX on
the fixture set. The problem is speed, not correctness.

## Batch Scanning

The scanner should process many emails or files as chunks:

```text
files -> text extraction -> tokenize/chunk -> microbatch -> model -> decode -> merge spans
```

For 100 short emails on the measured M1 Max:

| Runtime | Practical expectation |
| --- | --- |
| MLX MXFP8 | roughly 2.5-3 seconds after warmup for 100 short 128-token chunks |
| Core ML dense fp16 | roughly 9-10 seconds after warmup for the same workload |

True Core ML tensor batches were tested at `[2, 128]`, `[4, 128]`, and
`[8, 128]`. They stayed correct, but did not make the dense Core ML graph fast:

| Core ML tensor batch | Core ML mean/sample | MLX mean/sample | Agreement |
| ---: | ---: | ---: | ---: |
| `[2, 128]` | 214.04 ms | 26.98 ms | 1.0 |
| `[4, 128]` | 186.72 ms | 32.10 ms | 1.0 |
| `[8, 128]` | 192.03 ms | 25.49 ms | 1.0 |

The fixed-batch package is not the bottleneck. The dense expert block is.

## Expected Behavior Across Macs

These are planning estimates, not measured benchmarks. The safest rule is:
memory size decides whether the experience is smooth; memory bandwidth and GPU
size decide how fast the scan feels.

| Mac class | MLX MXFP8 expectation | Core ML dense expectation |
| --- | --- | --- |
| M1 Max, 32 GB | measured fast path, about 2.5-3 seconds for 100 short emails | measured slower path, about 9-10 seconds |
| Base M3, 8-16 GB | likely usable, but 8 GB may hit memory pressure on large scans | usable for small scans, less attractive for batch scanning |
| Base M4, 16 GB+ | likely a good local scanner, faster than base M3 | still limited by dense MoE export |
| M5, 16 GB+ | likely the best normal-chip experience; should narrow the gap with older Max chips | faster than old Core ML runs, but still structurally behind MLX MXFP8 unless sparse MoE lands |
| Pro/Max chips | best for heavy folder scans and long documents | benefit from bandwidth, but dense expert cost remains |

## Install Size In A macOS App

A user-friendly macOS app should not ask the user to install Python, Homebrew,
Xcode, `pip`, or the Hugging Face CLI. The app should include the native runtime
and download model packs itself, like Xcode downloading optional simulators.

Recommended downloadable packs:

| Pack | Approx size | Best use |
| --- | ---: | --- |
| MLX MXFP8 | about 1.47 GB | Default macOS download. Fastest and smaller than BF16/Core ML fp16. |
| MLX BF16 | about 2.83 GB | Debug/reference option. |
| Core ML fp16 dense | about 2.80 GB | Apple-platform compatibility and Core ML experiments. |
| Core ML 8-bit palettized | about 1.40 GB | Smaller Core ML option, but slower. |

The app needs to provide:

- a native MLX Swift runtime path, not an embedded Python script;
- the privacy-filter model architecture implemented against MLX Swift;
- tokenizer files and host-side BIOES/Viterbi decode;
- a model downloader with resume, checksums, versioning, and deletion;
- model storage under app support, for example
  `~/Library/Application Support/<App>/Models/`.

## Current Recommendation

Use MLX MXFP8 as the downloadable "Fast Local Scanner" on macOS. Keep Core ML as
the compatibility path and conversion target. The next meaningful Core ML speed
step is sparse MoE export or another graph design that avoids computing all 128
experts for every token.
