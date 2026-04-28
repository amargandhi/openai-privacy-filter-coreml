# Results

These numbers are a snapshot, not a benchmark suite. They answer the first
question: does the converted Core ML graph behave like the public model?

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
