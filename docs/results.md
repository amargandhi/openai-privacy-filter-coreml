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
