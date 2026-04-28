# OpenAI Privacy Filter Core ML

Core ML conversion and validation tooling for `openai/privacy-filter`.

The goal is a native Apple runtime artifact that Manifold and other macOS/iOS apps can use without a Python privacy-filter subprocess. The Core ML model should only emit token-classification logits. Tokenization, BIOES/Viterbi decoding, offset mapping, redaction, and policy remain in the host app.

## Status

- No public Core ML conversion found as of 2026-04-28.
- Official Hugging Face artifacts exist, including tokenizer, `viterbi_calibration.json`, safetensors weights, and ONNX variants.
- MLX community conversions exist for BF16 and MXFP8.
- Fixed-shape Core ML export is proven at sequence lengths 16 and 128 using `--mask-mode 4d`, `--export-method export`, and `--expert-mode dense`.
- 128-token Core ML parity reports show `argmax_agreement: 1.0` across all fixtures against the source model, MLX BF16, and MLX MXFP8.
- This repo does not vendor model weights or generated `.mlpackage` artifacts.

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
python scripts/check_environment.py --profile convert
python scripts/check_environment.py --profile mlx
```

Run the MLX metadata/inference smoke path after dependencies are installed:

```bash
python scripts/run_mlx_privacy_filter.py \
  --model mlx-community/openai-privacy-filter-mxfp8 \
  --fixtures fixtures/privacy_samples.json \
  --json-out reports/mlx-mxfp8.json
```

Run the official Transformers fixture baseline:

```bash
python scripts/run_transformers_privacy_filter.py \
  --model-id openai/privacy-filter \
  --fixtures fixtures/privacy_samples.json \
  --max-length 128 \
  --decode viterbi \
  --json-out reports/transformers-viterbi-128.json
```

Attempt a small fixed-shape Core ML conversion first:

```bash
python scripts/convert_privacy_filter_coreml.py \
  --model-id openai/privacy-filter \
  --sequence-length 128 \
  --mask-mode 4d \
  --export-method export \
  --expert-mode dense \
  --output build/OpenAIPrivacyFilterLogits_128.mlpackage
```

Compare PyTorch and Core ML logits:

```bash
python scripts/compare_coreml_logits.py \
  --model-id openai/privacy-filter \
  --coreml-model build/OpenAIPrivacyFilterLogits_128.mlpackage \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --mask-mode 4d \
  --expert-mode dense \
  --json-out reports/coreml-128-parity.json
```

Compare Core ML against the MLX community checkpoints:

```bash
python scripts/compare_coreml_mlx_logits.py \
  --mlx-model mlx-community/openai-privacy-filter-bf16 \
  --coreml-model build/OpenAIPrivacyFilterLogits_128.mlpackage \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --json-out reports/coreml-vs-mlx-bf16-128.json

python scripts/compare_coreml_mlx_logits.py \
  --mlx-model mlx-community/openai-privacy-filter-mxfp8 \
  --coreml-model build/OpenAIPrivacyFilterLogits_128.mlpackage \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --json-out reports/coreml-vs-mlx-mxfp8-128.json
```

## Target Artifact

`OpenAIPrivacyFilterLogits.mlpackage`

Inputs:

- `input_ids`: `int32`, shape `[1, sequence_length]`
- `attention_mask`: either tokenizer mask `int32` shape `[1, sequence_length]` or precomputed additive mask `float32` shape `[1, 1, sequence_length, sequence_length]`

Output:

- `logits`: `float16` or `float32`, shape `[1, sequence_length, 33]`

The first conversion should use fixed shapes (`128`, then `512`). Enumerated shapes can follow once small-shape conversion and parity are proven.

The current conversion default uses a precomputed 4D additive attention mask (`--mask-mode 4d`) instead of the tokenizer's 2D mask. This avoids a Transformers 5.7 TorchScript tracing issue in the bidirectional sliding-window mask helper. A production host can generate the same 4D mask from token padding plus the model's `sliding_window`.

The upstream sparse MoE expert loop is data-dependent, which blocks `torch.export`. `--expert-mode dense` patches the expert block to compute every expert and mask/sum the router top-k experts. This proves Core ML conversion feasibility and parity, but a performant production path still needs a sparse or custom decomposition.

The MLX fixture scripts use a project-local `openai_privacy_filter` MLX implementation because `mlx-embeddings` 0.1.0 does not register this model type. Its RoPE implementation intentionally follows the Transformers privacy-filter YaRN formula, including `truncate: false`.

## Work Plan

1. Prove PyTorch-to-Core ML conversion at fixed shape 128. Done.
2. Compare PyTorch vs Core ML logits on fixtures. Done.
3. Run MLX BF16/MXFP8 fixture inference for baseline behavior. Done.
4. Add an official Transformers fixture baseline. Done.
5. Implement or port tokenizer and Viterbi decode in Swift.
6. Add a `CoreMLPrivacyBackend` in Manifold that conforms to the existing `PrivacyBackend` protocol.
7. Benchmark compute units and sequence lengths.
8. Add provenance records for model revision, Core ML Tools version, Torch version, Transformers version, shape set, precision, and conversion hash.

## References

- OpenAI release: https://openai.com/index/introducing-openai-privacy-filter/
- Hugging Face model: https://huggingface.co/openai/privacy-filter
- Official ONNX artifacts: https://huggingface.co/openai/privacy-filter/tree/main/onnx
- MLX BF16: https://huggingface.co/mlx-community/openai-privacy-filter-bf16
- Core ML PyTorch conversion: https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html
- Core ML ML Program docs: https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html
