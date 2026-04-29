# Results

These numbers are a snapshot, not a benchmark suite. They answer two early
questions:

1. Does the converted Core ML graph behave like the public model?
2. How far is the current dense Core ML proof from the MLX runtime?

Environment:

- Date: 2026-04-28
- Machine: macOS arm64
- Python: 3.12.13
- Core ML Tools: 9.0
- Torch: 2.11.0
- Transformers: 5.7.0

## Generated Packages

Two local Core ML packages were generated:

| Package | Sequence length | Precision | Export | Mask | Expert mode |
| --- | ---: | --- | --- | --- | --- |
| `OpenAIPrivacyFilterLogits_16_dense.mlpackage` | 16 | fp16 | `torch.export` | 4D additive | dense |
| `OpenAIPrivacyFilterLogits_128_dense.mlpackage` | 128 | fp16 | `torch.export` | 4D additive | dense |

The 128-token provenance hash was:

```text
876f1672ff2c87352e953de234bc89ac83600380a6e6b1fe732caaa6daa89137
```

Generated packages and reports are ignored by Git. Rebuild them locally when you
need them.

## Core ML Parity

All 128-token fixture comparisons had token `argmax_agreement: 1.0`.

| Comparison | Worst max abs diff | Worst mean abs diff |
| --- | ---: | ---: |
| Core ML vs source Transformers | 3.4082 | 0.2639 |
| Core ML vs MLX BF16 | 11.9753 | 0.4878 |
| Core ML vs MLX MXFP8 | 16.4822 | 1.2737 |

The absolute differences are larger for MLX BF16 and MXFP8 because the runtime
and quantization differ. The thing that matters for the first pass is the tag
chosen for each non-padding token. Those tags matched across all fixtures.

## Core ML vs MLX Performance

The benchmark command ran the seven fixture texts at sequence length 128 with
three warmup passes and ten measured passes per fixture. Core ML used
`compute_units=all`.

```bash
python scripts/benchmark_coreml_mlx.py \
  --coreml-model build/OpenAIPrivacyFilterLogits_128_dense.mlpackage \
  --mlx-model mlx-community/openai-privacy-filter-bf16 \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --warmup 3 \
  --iterations 10 \
  --json-out reports/benchmark-coreml-vs-mlx-bf16-128.json
```

| Comparison | Core ML warm mean | MLX warm mean | Core ML / MLX latency | Core ML load | MLX load | Core ML package | MLX model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Core ML dense vs MLX BF16 | 88.09 ms | 44.61 ms | 1.97x | 47.86 s | 2.31 s | 2.80 GB | 2.83 GB |
| Core ML dense vs MLX MXFP8 | 96.88 ms | 16.84 ms | 5.75x | 47.45 s | 2.11 s | 2.80 GB | 1.47 GB |

Both benchmark runs had token argmax agreement of `1.0` on every fixture.

The read is straightforward: the current Core ML package is a correctness proof,
not the fast path yet. The dense expert patch runs every expert, while the model
was designed as a sparse mixture-of-experts network. MLX also has an MXFP8
checkpoint that is much smaller and faster. The next performance phase should
attack the expert block and package shape, not the tokenizer or decoder.

## Batch Throughput

The first batch pass added two benchmark modes:

- `api`: pass a list of batch-1 inputs through Core ML's batch prediction API.
- `tensor`: pass a real `[batch_size, sequence_length]` tensor into a package
  exported for that batch size.

Measured snapshots:

| Package | Mode | Batch | Sequence | Core ML mean/sample | MLX mean/sample | Core ML tokens/sec | MLX tokens/sec | Agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `OpenAIPrivacyFilterLogits_128_dense.mlpackage` | Core ML API batch | 4 | 128 | 96.06 ms | 10.29 ms | 180 | 1,680 | 1.0 |
| `OpenAIPrivacyFilterLogits_b2_16_dense.mlpackage` | Core ML tensor batch | 2 | 16 | 20.23 ms | 9.48 ms | 720 | 1,537 | 1.0 |

The API-batch result is mainly an integration check; it did not materially beat
the earlier single-sample Core ML latency. The tensor-batch result proves the
batch-shaped conversion path works, but it was only validated on a small
sequence-16 package. The next useful Core ML measurement is a true tensor-batch
128-token package at batch sizes 2, 4, and 8.

## 2026-04-29 Speed Sweep

The next sweep generated true 128-token tensor-batch packages for batch sizes 2,
4, and 8. Each package preserved token argmax agreement of `1.0` against MLX
MXFP8, but none improved throughput.

| Core ML package | Batch | Core ML mean/sample | Core ML tokens/sec | MLX mean/sample | MLX tokens/sec | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `OpenAIPrivacyFilterLogits_b2_128_dense.mlpackage` | 2 | 214.04 ms | 81 | 26.98 ms | 641 | 1.0 |
| `OpenAIPrivacyFilterLogits_b4_128_dense.mlpackage` | 4 | 186.72 ms | 93 | 32.10 ms | 538 | 1.0 |
| `OpenAIPrivacyFilterLogits_b8_128_dense.mlpackage` | 8 | 192.03 ms | 98 | 25.49 ms | 695 | 1.0 |

Compression results were mixed:

| Package | Method | Size | Accuracy | Runtime |
| --- | --- | ---: | --- | --- |
| `OpenAIPrivacyFilterLogits_128_dense_int8.mlpackage` | linear int8 | 1.40 GB | rejected, minimum agreement `0.27` on CPU-only smoke | accelerated execution failed with an MPSGraph MLIR error |
| `OpenAIPrivacyFilterLogits_128_dense_palettize8_uniform.mlpackage` | uniform 8-bit palettization | 1.40 GB | agreement `1.0` | 148.97 ms/sample, slower than fp16 dense |

The conclusion is not subtle: easy Core ML levers are not enough. Fixed tensor
batching is correct but slower for this dense graph, and compression can reduce
artifact size but does not improve latency. The next real performance path is
to remove the dense expert patch or use MLX MXFP8 on macOS where that runtime is
available.

## Fixture Baselines

The official Transformers baseline recovered the expected label set for every
fixture using:

- model: `openai/privacy-filter`
- decode: `viterbi`
- max length: 128
- calibration: `viterbi_calibration.json`

The MLX BF16 and MXFP8 baselines also recovered the expected label set for every
fixture. The first run is slower because MLX and Core ML both compile kernels and
runtime caches on first use.

## Swift Tokenizer

The SwiftPM target `PrivacyFilterTokenizer` loads the official Hugging Face
`tokenizer.json` and implements the byte-level BPE path used by
`openai/privacy-filter`.

`swift test` passed against a Python-generated fixture with:

- all privacy sample texts at `max_length=128`
- padded `input_ids`
- `attention_mask`
- tokenizer offset mappings
- one extra Unicode sample with non-ASCII text and emoji

The Swift tokenizer is now suitable for host-side experiments. The Swift
BIOES/Viterbi decoder still needs to be ported.
