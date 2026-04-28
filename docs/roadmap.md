# Roadmap

The project is moving from "can this convert?" to "can this be a boring native
runtime component?"

## Shipped

- Reproducible Python conversion environment.
- Fixed 128-token Core ML package.
- Core ML vs Transformers parity report.
- Core ML vs MLX BF16 and MXFP8 parity reports.
- Core ML vs MLX benchmark harness and first 128-token performance snapshot.
- Fixed batch-size conversion and microbatch benchmark support.
- Official Transformers fixture baseline.
- Swift tokenizer and offset handling.

## Next

### Swift BIOES and Viterbi Decode

The Python decoder already handles BIOES spans and constrained Viterbi decoding.
The next native piece is a Swift decoder that matches the Python fixture output.

Acceptance criteria:

- Swift decodes fixture logits into the same label IDs as Python.
- Swift reconstructs the same spans from token offsets.
- The decoder supports calibration values from `viterbi_calibration.json`.

### Core ML Runtime Adapter

The package needs a small Swift runtime wrapper that owns:

- tokenizer
- 4D attention mask construction
- Core ML model invocation
- decoder
- span output

For app integration, this wrapper should expose one boring method:

```swift
scan(text: String) throws -> [PrivacySpan]
```

### Performance

Correctness came first. The first benchmark is now in `docs/results.md`: the
dense Core ML package is accurate, but slower than MLX BF16 and much slower than
MLX MXFP8. That points at the real work.

Next measure:

- peak memory
- package size
- compute unit choice: `.all`, `.cpuAndGPU`, `.cpuAndNeuralEngine`
- sequence lengths: 128 and 512
- true tensor batch sizes: 2, 4, 8, and 16
- compressed package variants: int8, int4, and palettized weights

### Smaller Or Sparse Expert Export

The dense expert patch is the main known cost. A practical package likely needs
one of:

- an exportable sparse expert decomposition
- a custom Core ML-friendly expert block
- a smaller quantized package
- direct use of an official ONNX or MLX path where that is a better fit

### Distribution

Before this is an installable dependency, it needs:

- a documented model download/conversion command
- provenance checks
- a versioned artifact naming scheme
- clear license and attribution notes

## Non-Goals

- Commit model weights to Git.
- Pretend a classifier is a complete privacy system.
- Hide policy decisions inside the model artifact.
- Make a server dependency for a local-first runtime.
