# OpenAI Privacy Filter Core ML

Run `openai/privacy-filter` on Apple platforms with Core ML.

This repository is a small, reproducible conversion lab. It takes the public
OpenAI Privacy Filter model from Hugging Face, converts the neural forward pass
to a Core ML package, and checks the result against three reference paths:
Transformers, MLX BF16, and MLX MXFP8.

The useful mental model is:

```text
text -> tokenizer -> token ids -> Core ML logits -> BIOES/Viterbi decode -> spans
```

Core ML only does the neural network. Tokenization, offsets, decoding, redaction,
policy, and UI stay in the host app where they are easier to inspect and test.

## What Works

- Fixed-shape Core ML export at sequence lengths 16 and 128.
- 128-token parity with the source Transformers model: token argmax agreement is
  1.0 on every fixture.
- 128-token parity with MLX BF16 and MXFP8 checkpoints: token argmax agreement is
  1.0 on every fixture.
- Swift byte-level BPE tokenizer with offset parity against the official Python
  tokenizer fixtures.
- Python BIOES and constrained Viterbi decoding helpers.

The repo does not store model weights or generated `.mlpackage` files. Those are
large artifacts that should be generated locally from pinned upstream revisions.

## Current Performance Read

On the measured M1 Max machine, MLX MXFP8 is the clear macOS fast path: about
`16.84 ms` per warm 128-token sample, versus `96.88 ms` for the current dense
Core ML fp16 package. Both paths match token decisions on the fixture set, but
Core ML is slowed down by the dense expert export. For a batch of 100 short
emails, the practical expectation is roughly 2.5-3 seconds with MLX MXFP8 after
warmup and roughly 9-10 seconds with the current Core ML dense package.

For normal M3, M4, and M5 Macs, treat MLX MXFP8 as the recommended downloadable
runtime for a macOS app. Core ML remains valuable for compatibility and
packaging, but the next real speed step is sparse MoE export, not larger fixed
batches. See the short [Mac performance summary](docs/mac-performance-summary.md).

## Why This Exists

The privacy filter is useful in desktop and mobile apps, but a Python or server
side model path is awkward for local-first software. Apple platforms already have
a native model runtime in Core ML. The interesting question is whether we can
make the model boring: deterministic conversion, explicit provenance, and parity
tests that fail loudly.

This repo is not an official OpenAI or Apple project. It is an engineering
bridge for people who want to run the public model locally on macOS or iOS.

## Install

Use macOS on Apple Silicon. Python 3.12 is the current least surprising choice
for Core ML Tools, Torch, Transformers, and MLX in this repo.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[convert,mlx,dev]"
```

Check the environment:

```bash
python scripts/check_environment.py --profile convert
python scripts/check_environment.py --profile mlx
```

## Convert

The proven path uses a fixed shape, a precomputed 4D attention mask, and an
export-friendly dense expert patch:

```bash
python scripts/convert_privacy_filter_coreml.py \
  --model-id openai/privacy-filter \
  --sequence-length 128 \
  --batch-size 1 \
  --mask-mode 4d \
  --export-method export \
  --expert-mode dense \
  --output build/OpenAIPrivacyFilterLogits_128_dense.mlpackage
```

The script writes a sidecar provenance file next to the generated package. It
records the upstream model revision, tool versions, shape, precision, and a hash
of the conversion recipe.

## Validate

Compare Core ML logits with the source Transformers model:

```bash
python scripts/compare_coreml_logits.py \
  --model-id openai/privacy-filter \
  --coreml-model build/OpenAIPrivacyFilterLogits_128_dense.mlpackage \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --mask-mode 4d \
  --expert-mode dense \
  --json-out reports/coreml-128-dense-parity.json
```

Compare the same Core ML package with MLX checkpoints:

```bash
python scripts/compare_coreml_mlx_logits.py \
  --mlx-model mlx-community/openai-privacy-filter-bf16 \
  --coreml-model build/OpenAIPrivacyFilterLogits_128_dense.mlpackage \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --json-out reports/coreml-vs-mlx-bf16-128.json

python scripts/compare_coreml_mlx_logits.py \
  --mlx-model mlx-community/openai-privacy-filter-mxfp8 \
  --coreml-model build/OpenAIPrivacyFilterLogits_128_dense.mlpackage \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --json-out reports/coreml-vs-mlx-mxfp8-128.json
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

Run the MLX fixture baseline:

```bash
python scripts/run_mlx_privacy_filter.py \
  --model mlx-community/openai-privacy-filter-mxfp8 \
  --fixtures fixtures/privacy_samples.json \
  --max-length 128 \
  --json-out reports/mlx-mxfp8-128.json
```

Benchmark Core ML against an MLX checkpoint:

```bash
python scripts/benchmark_coreml_mlx.py \
  --coreml-model build/OpenAIPrivacyFilterLogits_128_dense.mlpackage \
  --mlx-model mlx-community/openai-privacy-filter-bf16 \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --batch-size 1 \
  --warmup 3 \
  --iterations 10 \
  --json-out reports/benchmark-coreml-vs-mlx-bf16-128.json
```

For throughput experiments, use `--batch-size` with repeated fixtures or a
batch-shaped package. See [Performance](docs/performance.md).

Run Swift tokenizer tests:

```bash
swift test
```

The Swift test target uses `OPENAI_PRIVACY_FILTER_TOKENIZER_JSON` if set. If it
is not set, it falls back to the pinned Hugging Face cache path for
`openai/privacy-filter`.

## Artifact Contract

The target Core ML package is intentionally simple:

- `input_ids`: `int32`, shape `[1, sequence_length]`
- `attention_mask`: `float32`, shape `[1, 1, sequence_length, sequence_length]`
- `logits`: `float16` or `float32`, shape `[1, sequence_length, 33]`

The 4D attention mask is an additive bidirectional sliding-window mask. A host
app can build it from the tokenizer padding mask and the model `sliding_window`.

## Limitations

- The current Core ML proof uses `--expert-mode dense`. It computes every expert
  and masks the router top-k result. This is good for correctness, but it is
  slower than MLX today and is not the final performance path.
- The generated 128-token package is large, roughly the size of the public model
  weights. This repo intentionally does not commit it.
- Shape support is fixed today. Enumerated shapes and longer chunks are the next
  step.
- Batch-shaped exports are supported for experiments, but each batch size is a
  separate fixed-shape package today.
- The Swift tokenizer is present. Swift BIOES/Viterbi decode is still in Python
  and is the next native runtime piece.
- A privacy classifier is not a privacy guarantee. Apps still need policy,
  auditability, fallbacks, and careful UX.

## Repository Map

- `scripts/convert_privacy_filter_coreml.py`: Core ML conversion entry point.
- `scripts/compare_coreml_logits.py`: Core ML vs Transformers logits.
- `scripts/compare_coreml_mlx_logits.py`: Core ML vs MLX logits.
- `scripts/benchmark_coreml_mlx.py`: Core ML vs MLX latency and parity snapshot.
- `scripts/compress_coreml_model.py`: Core ML weight compression experiments.
- `scripts/run_transformers_privacy_filter.py`: official Python fixture baseline.
- `scripts/run_mlx_privacy_filter.py`: MLX fixture baseline.
- `Sources/PrivacyFilterTokenizer`: Swift tokenizer package.
- `src/privacy_filter_coreml`: Python decode, Viterbi, MLX loader, and conversion helpers.
- `fixtures/privacy_samples.json`: small human-readable validation samples.
- `docs/results.md`: current measurements and parity numbers.
- `docs/technical-notes.md`: conversion details and known blockers.
- `docs/roadmap.md`: public project direction.

## Docs

- [Results](docs/results.md)
- [Performance](docs/performance.md)
- [Mac performance summary](docs/mac-performance-summary.md)
- [Technical notes](docs/technical-notes.md)
- [Roadmap](docs/roadmap.md)

## References

- [OpenAI Privacy Filter announcement](https://openai.com/index/introducing-openai-privacy-filter/)
- [Hugging Face model](https://huggingface.co/openai/privacy-filter)
- [Official ONNX artifacts](https://huggingface.co/openai/privacy-filter/tree/main/onnx)
- [MLX BF16 checkpoint](https://huggingface.co/mlx-community/openai-privacy-filter-bf16)
- [MLX MXFP8 checkpoint](https://huggingface.co/mlx-community/openai-privacy-filter-mxfp8)
- [Core ML PyTorch conversion guide](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)
