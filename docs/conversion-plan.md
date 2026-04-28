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

Primary path:

1. Load `AutoModelForTokenClassification.from_pretrained("openai/privacy-filter")`.
2. Wrap forward to return only `logits`.
3. Disable cache and return dictionaries.
4. Trace or export a fixed-shape graph.
5. Convert with Core ML Tools Unified Conversion API as an ML Program.
6. Save `.mlpackage`.
7. Compare logits against PyTorch.

Fallback path:

- Try `torch.export.export` if TorchScript tracing fails.
- Use ONNX only as a reference or last-resort experiment.

## First Milestones

1. Environment proof: Python 3.12, Torch, Transformers 5.6+, Core ML Tools.
2. Fixed 128-token `.mlpackage`.
3. PyTorch vs Core ML logit parity report.
4. MLX MXFP8 fixture output report.
5. Fixed 512-token `.mlpackage`.
6. Enumerated-shape conversion.
7. Swift tokenizer/Viterbi port.
8. Manifold `CoreMLPrivacyBackend`.
