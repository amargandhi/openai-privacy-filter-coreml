# Conversion Plan

## Discovery

Targeted searches did not find a public Core ML package for `openai/privacy-filter`.

Related work that matters:

- OpenAI publishes official safetensors, tokenizer, Viterbi calibration, and ONNX artifacts.
- Hugging Face has Transformers.js support and quantized ONNX files.
- MLX community publishes BF16 and MXFP8 conversions.
- A Rust/Burn weight package and ONNX-based hooks exist, but they do not solve Core ML deployment.

## Architecture Boundary

Keep this split:

- Core ML: neural forward pass from token IDs to 33 token-label logits.
- Host runtime: tokenizer, offsets, Viterbi/BIOES decode, redaction, policy, audit, and UI.

This keeps Core ML conversion small and lets Manifold preserve its existing `PrivacyBackend` contract.

## Conversion Path

Proven path:

1. Load `AutoModelForTokenClassification.from_pretrained("openai/privacy-filter")`.
2. Wrap forward to return only `logits`.
3. Disable cache and return dictionaries.
4. Use a precomputed 4D additive sliding-window attention mask.
5. Patch the sparse MoE block into a dense all-expert computation for export.
6. Export a fixed-shape graph with `torch.export.export`.
7. Convert with Core ML Tools Unified Conversion API as an ML Program.
8. Save `.mlpackage` plus a provenance sidecar.
9. Compare logits against PyTorch and MLX BF16/MXFP8.

Observed blockers:

- TorchScript tracing with the normal 2D tokenizer mask fails in the Transformers bidirectional sliding-window mask helper.
- `torch.export.export` of the upstream sparse MoE loop fails on data-dependent expert routing.
- The dense expert patch proves conversion and parity, but it is not the production performance target.
- `mlx-embeddings` 0.1.0 does not register `openai_privacy_filter`, so MLX fixture tests use a project-local model class.

## First Milestones

1. Environment proof: Python 3.12, Torch, Transformers 5.6+, Core ML Tools.
2. Fixed 128-token `.mlpackage`. Done.
3. PyTorch vs Core ML logit parity report. Done.
4. MLX BF16/MXFP8 fixture output reports. Done.
5. Fixed 512-token `.mlpackage`.
6. Enumerated-shape conversion.
7. Swift tokenizer/Viterbi port.
8. Manifold `CoreMLPrivacyBackend`.
