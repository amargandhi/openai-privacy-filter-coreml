#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert openai/privacy-filter token logits to a fixed-shape Core ML package."
    )
    parser.add_argument("--model-id", default="openai/privacy-filter")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--export-method",
        choices=["trace", "export"],
        default="trace",
        help="Capture the PyTorch graph with TorchScript trace or torch.export.",
    )
    parser.add_argument(
        "--expert-mode",
        choices=["eager", "dense"],
        default="eager",
        help=(
            "Use upstream sparse eager experts or an export-friendly dense expert patch. "
            "Dense mode computes all experts and masks to top-k, so it is for feasibility tests."
        ),
    )
    parser.add_argument(
        "--mask-mode",
        choices=["2d", "4d"],
        default="4d",
        help=(
            "Use the normal 2D tokenizer mask or a precomputed 4D additive mask. "
            "The 4D mask avoids a TorchScript tracing bug in Transformers 5.7."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true")
    return parser.parse_args()


def require_dependencies():
    try:
        import coremltools as ct
        import numpy as np
        import torch
        import transformers
        from huggingface_hub import model_info
        from transformers import AutoModelForTokenClassification
    except ImportError as exc:
        raise SystemExit(
            "Missing conversion dependencies. Install with "
            '`python -m pip install -e ".[convert]"`.'
        ) from exc
    return ct, np, torch, transformers, model_info, AutoModelForTokenClassification


def main() -> int:
    args = parse_args()
    ct, np, torch, transformers, model_info, AutoModelForTokenClassification = require_dependencies()
    if args.expert_mode == "dense":
        from privacy_filter_coreml.transformers_patch import patch_openai_privacy_filter_dense_experts

        patch_openai_privacy_filter_dense_experts()

    torch_dtype = torch.float32 if args.torch_dtype == "float32" else torch.bfloat16
    compute_precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32

    class PrivacyFilterLogits(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=False,
            )
            return outputs[0]

    print(f"Loading {args.model_id}", file=sys.stderr)
    torch_model = AutoModelForTokenClassification.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    torch_model.config.use_cache = False
    torch_model.eval()
    wrapped = PrivacyFilterLogits(torch_model).eval()

    example_input_ids = torch.zeros((1, args.sequence_length), dtype=torch.int32)
    if args.mask_mode == "4d":
        example_attention_mask = make_4d_attention_mask(
            torch,
            args.sequence_length,
            sliding_window=int(torch_model.config.sliding_window),
            dtype=torch.float32,
        )
        attention_mask_dtype = np.float32
    else:
        example_attention_mask = torch.ones((1, args.sequence_length), dtype=torch.int32)
        attention_mask_dtype = np.int32

    with torch.no_grad():
        logits = wrapped(example_input_ids, example_attention_mask)
    if tuple(logits.shape) != (1, args.sequence_length, 33):
        raise SystemExit(f"Unexpected logits shape: {tuple(logits.shape)}")

    if args.export_method == "export":
        print("Exporting PyTorch graph with torch.export", file=sys.stderr)
        source_model = torch.export.export(
            wrapped,
            (example_input_ids, example_attention_mask),
            strict=False,
        )
        source_model = source_model.run_decompositions({})
    else:
        print("Tracing PyTorch graph", file=sys.stderr)
        source_model = torch.jit.trace(
            wrapped,
            (example_input_ids, example_attention_mask),
            strict=False,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting to {args.output}", file=sys.stderr)
    mlmodel = ct.convert(
        source_model,
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.macOS12,
        skip_model_load=args.skip_model_load,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=example_input_ids.shape,
                dtype=np.int32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=example_attention_mask.shape,
                dtype=attention_mask_dtype,
            ),
        ],
        outputs=[ct.TensorType(name="logits")],
    )
    mlmodel.short_description = "OpenAI Privacy Filter token-classification logits"
    mlmodel.input_description["input_ids"] = "Token IDs shaped [1, sequence_length]"
    mlmodel.input_description["attention_mask"] = "Attention mask shaped [1, sequence_length]"
    mlmodel.output_description["logits"] = "BIOES token classification logits shaped [1, sequence_length, 33]"
    mlmodel.save(str(args.output))

    info = model_info(args.model_id)
    provenance = {
        "model_id": args.model_id,
        "model_sha": info.sha,
        "sequence_length": args.sequence_length,
        "export_method": args.export_method,
        "expert_mode": args.expert_mode,
        "mask_mode": args.mask_mode,
        "precision": args.precision,
        "torch_dtype": args.torch_dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "coremltools_version": ct.__version__,
        "output": str(args.output),
    }
    provenance["provenance_hash"] = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode("utf-8")
    ).hexdigest()
    args.output.with_suffix(args.output.suffix + ".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


def make_4d_attention_mask(torch, sequence_length: int, sliding_window: int, dtype):
    positions = torch.arange(sequence_length)
    allowed = torch.abs(positions[:, None] - positions[None, :]) <= sliding_window
    mask = torch.where(
        allowed,
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(torch.finfo(dtype).min, dtype=dtype),
    )
    return mask.reshape(1, 1, sequence_length, sequence_length)


if __name__ == "__main__":
    raise SystemExit(main())
