# Technical Notes

This project keeps the model boundary narrow.

Core ML gets:

- token IDs
- an attention mask
- a fixed sequence length

Core ML returns:

- token-classification logits for 33 BIOES labels

Everything else stays in normal app code. This makes the system easier to test:
the tokenizer can be compared against Python, the decoder can be compared
against fixture logits, and the Core ML graph can be compared directly with
Transformers and MLX.

## Why A 4D Mask

The model uses a bidirectional sliding-window attention mask. In Transformers,
the normal path starts from a 2D tokenizer mask and constructs the 4D additive
mask inside the model.

That path is not friendly to the current export stack. TorchScript tracing with
the 2D mask runs into shape handling inside the Transformers mask helper.

The workaround is simple: build the 4D additive mask outside the model and pass
it in as an input.

The converted model therefore expects:

```text
attention_mask: float32 [1, 1, sequence_length, sequence_length]
```

Allowed attention positions are `0.0`. Blocked positions are a large negative
float. This is the same representation used by many transformer exports.

## Why Dense Experts

The upstream model uses sparse mixture-of-experts routing. In eager PyTorch this
is natural: route each token to the top-k experts and only run those experts.

`torch.export` does not like the data-dependent expert loop in the upstream
implementation. The loop asks a graph capture system to reason about which
experts happen to be active for this input. That set changes with the data.

For the first Core ML proof, the repo patches the expert block into a dense
version:

1. Run every expert.
2. Build a top-k routing weight mask.
3. Sum the active expert outputs.

This is mathematically useful for parity. It is not the final performance
strategy. A production path should either find an exportable sparse
decomposition or use a custom Core ML-friendly expert implementation.

## MLX Loader

The public MLX checkpoints exist, but `mlx-embeddings` 0.1.0 does not register
`model_type = openai_privacy_filter`.

The repo includes a small MLX implementation for fixture testing. It mirrors the
architecture needed for parity and uses the same privacy-filter YaRN RoPE formula
as Transformers, including `truncate: false`.

This MLX path is a validation tool. It is not trying to replace the upstream MLX
ecosystem.

## Swift Tokenizer

The Swift tokenizer is intentionally small:

- read the official `tokenizer.json`
- apply the tokenizer regex split
- apply byte-level encoding
- run BPE merges
- return token IDs, attention mask, and offsets

The offsets match the Python tokenizer fixture for the sample suite. The tests
also cover a Unicode scalar sample because offset bugs usually hide there first.

## Current Artifact Shape

The proven package shape is fixed:

```text
input_ids: int32 [1, 128]
attention_mask: float32 [1, 1, 128, 128]
logits: fp16 [1, 128, 33]
```

The next useful shape work is:

1. 512-token fixed package.
2. Enumerated shapes for common chunk sizes.
3. Host-side chunking rules for longer text.
