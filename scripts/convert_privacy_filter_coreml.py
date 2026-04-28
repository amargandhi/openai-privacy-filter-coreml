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
    example_attention_mask = torch.ones((1, args.sequence_length), dtype=torch.int32)

    with torch.no_grad():
        logits = wrapped(example_input_ids, example_attention_mask)
    if tuple(logits.shape) != (1, args.sequence_length, 33):
        raise SystemExit(f"Unexpected logits shape: {tuple(logits.shape)}")

    print("Tracing PyTorch graph", file=sys.stderr)
    traced = torch.jit.trace(wrapped, (example_input_ids, example_attention_mask), strict=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting to {args.output}", file=sys.stderr)
    mlmodel = ct.convert(
        traced,
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
                dtype=np.int32,
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


if __name__ == "__main__":
    raise SystemExit(main())
