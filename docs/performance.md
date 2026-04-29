# Performance

The current Core ML package is a validated baseline, not the fastest runtime.
It proves that the graph can be converted and compared, then shows exactly where
the speed work should go.

## What The Research Says

Core ML has three levers that matter here:

1. Batch predictions. `MLModel` can make multiple predictions from an
   `MLBatchProvider`, and `MLArrayBatchProvider` is the standard wrapper for an
   array of feature providers.
2. Shape discipline. Core ML Tools recommends finite fixed or enumerated shapes
   for best performance because the runtime can optimize the finite set of input
   shapes. Range shapes are useful, but bounded and enumerated shapes are easier
   for the runtime to specialize.
3. Compression. Core ML Tools supports post-training weight quantization and
   palettization for `mlprogram` packages. This can reduce package size, load
   cost, and memory traffic. It still needs parity testing because compression
   changes logits.

Those levers do not remove the biggest repo-specific cost: the dense expert
patch computes all 128 experts for every token, while the model routes each
token to four experts. That makes sparse MoE export the main long-term
performance task.

## What Is Implemented

- `scripts/convert_privacy_filter_coreml.py --batch-size N` exports fixed
  batch-shaped packages such as `[4, 128]` or `[8, 128]`.
- `scripts/benchmark_coreml_mlx.py --batch-size N` measures microbatch
  throughput and reports per-sample latency plus non-padding tokens per second.
- `--coreml-batch-mode api` uses Core ML's batch prediction path with a list of
  batch-1 inputs, which works with the existing package.
- `--coreml-batch-mode tensor` sends a true `[N, sequence_length]` tensor and
  requires a package converted with the same `--batch-size`.
- `scripts/compress_coreml_model.py` applies Core ML Tools weight compression
  experiments: int8, int4, and 8/6/4-bit palettization.

## Batch Benchmark

Existing batch-1 package, Core ML API batching:

```bash
python scripts/benchmark_coreml_mlx.py \
  --coreml-model build/OpenAIPrivacyFilterLogits_128_dense.mlpackage \
  --mlx-model mlx-community/openai-privacy-filter-mxfp8 \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --batch-size 4 \
  --repeat-fixtures 8 \
  --coreml-batch-mode api \
  --warmup 2 \
  --iterations 10 \
  --json-out reports/benchmark-coreml-api-batch4-vs-mlx-mxfp8-128.json
```

True tensor-batched Core ML package:

```bash
python scripts/convert_privacy_filter_coreml.py \
  --model-id openai/privacy-filter \
  --sequence-length 128 \
  --batch-size 4 \
  --mask-mode 4d \
  --export-method export \
  --expert-mode dense \
  --output build/OpenAIPrivacyFilterLogits_b4_128_dense.mlpackage

python scripts/benchmark_coreml_mlx.py \
  --coreml-model build/OpenAIPrivacyFilterLogits_b4_128_dense.mlpackage \
  --mlx-model mlx-community/openai-privacy-filter-mxfp8 \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --batch-size 4 \
  --repeat-fixtures 8 \
  --coreml-batch-mode tensor \
  --warmup 2 \
  --iterations 10 \
  --json-out reports/benchmark-coreml-tensor-batch4-vs-mlx-mxfp8-128.json
```

Initial measurements from the first batch pass:

| Package | Mode | Batch | Sequence | Core ML mean/sample | MLX mean/sample | Core ML tokens/sec | MLX tokens/sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `OpenAIPrivacyFilterLogits_128_dense.mlpackage` | API batch | 4 | 128 | 96.06 ms | 10.29 ms | 180 | 1,680 |
| `OpenAIPrivacyFilterLogits_b2_16_dense.mlpackage` | Tensor batch | 2 | 16 | 20.23 ms | 9.48 ms | 720 | 1,537 |

Both runs kept token argmax agreement at `1.0` against MLX MXFP8 on the fixture
set. API batching is compatible with the existing package, but the 128-token
measurement did not improve average Core ML latency. Tensor batching is the
more important path; the next benchmark should build `[2, 128]`, `[4, 128]`,
and `[8, 128]` packages and compare throughput.

## 128-Token Tensor Batch Sweep

The 2026-04-29 sweep generated fixed tensor-batch packages for batch sizes 2, 4,
and 8 at sequence length 128. Each package was benchmarked against MLX MXFP8
with `--repeat-fixtures 4`, `--warmup 2`, and `--iterations 5`.

| Package | Batch | Core ML mean/sample | Core ML p50/sample | Core ML tokens/sec | MLX mean/sample | MLX tokens/sec | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `OpenAIPrivacyFilterLogits_b2_128_dense.mlpackage` | 2 | 214.04 ms | 225.82 ms | 81 | 26.98 ms | 641 | 1.0 |
| `OpenAIPrivacyFilterLogits_b4_128_dense.mlpackage` | 4 | 186.72 ms | 195.78 ms | 93 | 32.10 ms | 538 | 1.0 |
| `OpenAIPrivacyFilterLogits_b8_128_dense.mlpackage` | 8 | 192.03 ms | 155.54 ms | 98 | 25.49 ms | 695 | 1.0 |

Tensor batching did not make the dense Core ML graph faster. The outputs still
match MLX decisions, but the dense expert patch dominates runtime and scales
poorly with batch size. This rules out fixed batch shape as the main speed lever
for the current dense export.

## Compression Experiments

Start with int8 because it is fast and data-free:

```bash
python scripts/compress_coreml_model.py \
  --input build/OpenAIPrivacyFilterLogits_128_dense.mlpackage \
  --output build/OpenAIPrivacyFilterLogits_128_dense_int8.mlpackage \
  --method linear-int8
```

Then compare logits and benchmark:

```bash
python scripts/compare_coreml_mlx_logits.py \
  --mlx-model mlx-community/openai-privacy-filter-mxfp8 \
  --coreml-model build/OpenAIPrivacyFilterLogits_128_dense_int8.mlpackage \
  --fixtures fixtures/privacy_samples.json \
  --sequence-length 128 \
  --json-out reports/coreml-int8-vs-mlx-mxfp8-128.json
```

The 2026-04-29 compression sweep tried two data-free 8-bit methods:

| Package | Method | Size | Result |
| --- | --- | ---: | --- |
| `OpenAIPrivacyFilterLogits_128_dense_int8.mlpackage` | linear int8 | 1.40 GB | Rejected. It is 2.0x smaller, but the accelerated backend failed with an MPSGraph MLIR error. CPU-only execution worked, but fixture agreement fell to `0.27`. |
| `OpenAIPrivacyFilterLogits_128_dense_palettize8_uniform.mlpackage` | uniform 8-bit palettization | 1.40 GB | Correct but slower. It kept fixture agreement at `1.0`, but warm Core ML latency was 148.97 ms/sample versus 96.88 ms/sample for the dense fp16 package. |

Compression is useful for package size, not speed, on the current graph. Linear
int8 is not acceptable without a better quantization recipe. Uniform 8-bit
palettization is a viable size-only artifact if a roughly 1.4 GB package is more
important than latency.

## Target Batch Scanner

A fast file scanner should be a pipeline:

```text
file walk -> text extract -> tokenize/chunk -> microbatch queue -> model -> decode -> span merge
```

Rules:

- Load the model once per scan.
- Tokenize and read files concurrently on CPU workers.
- Keep the accelerator fed with fixed-size microbatches.
- Cache attention masks by `(batch_size, sequence_length, valid_token_count)`.
- Decode and merge spans off the inference path.
- Report throughput as files/sec, chunks/sec, non-padding tokens/sec, and MB/sec.

## References

- Apple Core ML `MLModel` batch prediction APIs:
  https://developer.apple.com/documentation/coreml/mlmodel
- Apple `MLBatchProvider`:
  https://developer.apple.com/documentation/coreml/mlbatchprovider
- Apple `MLArrayBatchProvider`:
  https://developer.apple.com/documentation/coreml/mlarraybatchprovider
- Core ML Tools flexible shapes and enumerated shapes:
  https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html
- Core ML Tools quantization:
  https://apple.github.io/coremltools/source/coremltools.optimize.coreml.quantization.html
- Core ML Tools palettization:
  https://apple.github.io/coremltools/source/coremltools.optimize.coreml.palettization.html
