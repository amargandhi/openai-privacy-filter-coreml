# Current Results

Generated on 2026-04-28 on macOS arm64 with Python 3.12.13, Core ML Tools 9.0,
Torch 2.11.0, and Transformers 5.7.0.

## Conversion

- `build/OpenAIPrivacyFilterLogits_16_dense.mlpackage`
  - `sequence_length`: 16
  - `mask_mode`: `4d`
  - `export_method`: `export`
  - `expert_mode`: `dense`
  - `provenance_hash`: `de0768a7409e439e23d27fb113cc74e8754cb97a76736234f98077a6c859852f`

- `build/OpenAIPrivacyFilterLogits_128_dense.mlpackage`
  - `sequence_length`: 128
  - `mask_mode`: `4d`
  - `export_method`: `export`
  - `expert_mode`: `dense`
  - `provenance_hash`: `876f1672ff2c87352e953de234bc89ac83600380a6e6b1fe732caaa6daa89137`

Generated model artifacts and JSON reports stay ignored by Git.

## Parity Summary

All 128-token fixture comparisons had `argmax_agreement: 1.0`.

| Comparison | Worst max abs diff | Worst mean abs diff |
| --- | ---: | ---: |
| Core ML vs source PyTorch | 3.4082 | 0.2639 |
| Core ML vs MLX BF16 | 11.9753 | 0.4878 |
| Core ML vs MLX MXFP8 | 16.4822 | 1.2737 |

The larger absolute differences in MLX comparisons are expected from BF16/MXFP8
runtime and quantization differences. The token argmax labels matched on every
non-padding token across all fixtures.

## MLX Fixture Baseline

Both MLX community checkpoints recovered the expected label set for every
128-token fixture:

- `mlx-community/openai-privacy-filter-bf16`
- `mlx-community/openai-privacy-filter-mxfp8`

The first decoded run is slower because the MLX/Metal kernels and Core ML model
runtime compile on first use.

## Official Transformers Baseline

`scripts/run_transformers_privacy_filter.py` generated
`reports/transformers-viterbi-128.json` using:

- `model`: `openai/privacy-filter`
- `decode`: `viterbi`
- `max_length`: 128
- `torch_dtype`: `float32`
- `viterbi_calibration`: Hugging Face `viterbi_calibration.json` from revision
  `7ffa9a043d54d1be65afb281eddf0ffbe629385b`

The official baseline recovered the expected label set for every fixture and
uses the same JSON schema as the MLX reports.

## Swift Tokenizer Baseline

The SwiftPM target `PrivacyFilterTokenizer` loads the official Hugging Face
`tokenizer.json` and implements the byte-level BPE tokenizer path needed by
`openai/privacy-filter`.

`swift test` passed against `tests/PrivacyFilterTokenizerTests/Resources/tokenizer_baseline_128.json`.
The fixture was generated from the Python Transformers tokenizer and covers:

- all privacy sample texts at `max_length=128`
- padded `input_ids`
- `attention_mask`
- tokenizer offset mappings
- an extra Unicode scalar offset sample with non-ASCII text and emoji
