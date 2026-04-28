# OpenAI Privacy Filter Core ML

Core ML conversion and validation tooling for `openai/privacy-filter`.

The goal is a native Apple runtime artifact that Manifold and other macOS/iOS apps can use without a Python privacy-filter subprocess. The Core ML model should only emit token-classification logits. Tokenization, BIOES/Viterbi decoding, offset mapping, redaction, and policy remain in the host app.

## Status

- No public Core ML conversion found as of 2026-04-28.
- Official Hugging Face artifacts exist, including tokenizer, `viterbi_calibration.json`, safetensors weights, and ONNX variants.
- MLX community conversions exist for BF16 and MXFP8.
- This repo starts with conversion/parity tooling. It does not vendor model weights.

## Quick Start

Use Python 3.12 or earlier for conversion. The current Core ML Tools install docs list wheels through Python 3.12 for version 8.0.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[convert,mlx,dev]"
```

Check the local environment:

```bash
python scripts/check_environment.py
```

Run the MLX metadata/inference smoke path after dependencies are installed:

```bash
python scripts/run_mlx_privacy_filter.py \
  --model mlx-community/openai-privacy-filter-mxfp8 \
  --fixtures fixtures/privacy_samples.json \
  --json-out reports/mlx-mxfp8.json
```

Attempt a small fixed-shape Core ML conversion first:

```bash
python scripts/convert_privacy_filter_coreml.py \
  --model-id openai/privacy-filter \
  --sequence-length 128 \
  --output build/OpenAIPrivacyFilterLogits_128.mlpackage
```

Compare PyTorch and Core ML logits:

```bash
python scripts/compare_coreml_logits.py \
  --model-id openai/privacy-filter \
  --coreml-model build/OpenAIPrivacyFilterLogits_128.mlpackage \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --json-out reports/coreml-128-parity.json
```

## Target Artifact

`OpenAIPrivacyFilterLogits.mlpackage`

Inputs:

- `input_ids`: `int32`, shape `[1, sequence_length]`
- `attention_mask`: `int32`, shape `[1, sequence_length]`

Output:

- `logits`: `float16` or `float32`, shape `[1, sequence_length, 33]`

The first conversion should use fixed shapes (`128`, then `512`). Enumerated shapes can follow once small-shape conversion and parity are proven.

## Work Plan

1. Prove PyTorch-to-Core ML conversion at fixed shape 128.
2. Compare PyTorch vs Core ML logits on fixtures.
3. Run MLX BF16/MXFP8 fixture inference for baseline behavior.
4. Implement or port tokenizer and Viterbi decode in Swift.
5. Add a `CoreMLPrivacyBackend` in Manifold that conforms to the existing `PrivacyBackend` protocol.
6. Benchmark compute units and sequence lengths.
7. Add provenance records for model revision, Core ML Tools version, Torch version, Transformers version, shape set, precision, and conversion hash.

## References

- OpenAI release: https://openai.com/index/introducing-openai-privacy-filter/
- Hugging Face model: https://huggingface.co/openai/privacy-filter
- Official ONNX artifacts: https://huggingface.co/openai/privacy-filter/tree/main/onnx
- MLX BF16: https://huggingface.co/mlx-community/openai-privacy-filter-bf16
- Core ML PyTorch conversion: https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html
- Core ML ML Program docs: https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html
