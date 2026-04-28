#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and Core ML logits on fixtures.")
    parser.add_argument("--model-id", default="openai/privacy-filter")
    parser.add_argument("--coreml-model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/privacy_samples.json"))
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--mask-mode", choices=["2d", "4d"], default="4d")
    parser.add_argument("--expert-mode", choices=["eager", "dense"], default="eager")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def require_dependencies():
    try:
        import coremltools as ct
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing comparison dependencies. Install with "
            '`python -m pip install -e ".[convert]"`.'
        ) from exc
    return ct, torch, AutoModelForTokenClassification, AutoTokenizer


def main() -> int:
    args = parse_args()
    ct, torch, AutoModelForTokenClassification, AutoTokenizer = require_dependencies()
    if args.expert_mode == "dense":
        from privacy_filter_coreml.transformers_patch import patch_openai_privacy_filter_dense_experts

        patch_openai_privacy_filter_dense_experts()
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    torch_model = AutoModelForTokenClassification.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    torch_model.config.use_cache = False
    torch_model.eval()
    coreml_model = ct.models.MLModel(str(args.coreml_model), compute_units=ct.ComputeUnit.ALL)

    results = []
    for item in fixtures:
        encoded = tokenizer(
            item["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.sequence_length,
        )
        input_ids = encoded["input_ids"].to(torch.int32)
        attention_mask_2d = encoded["attention_mask"].to(torch.int32)
        if args.mask_mode == "4d":
            attention_mask = make_4d_attention_mask(
                torch,
                attention_mask_2d,
                sliding_window=int(torch_model.config.sliding_window),
            )
        else:
            attention_mask = attention_mask_2d

        started = perf_counter()
        with torch.no_grad():
            torch_logits = torch_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits.detach().cpu().numpy()
        torch_ms = (perf_counter() - started) * 1000.0

        coreml_inputs = {
            "input_ids": input_ids.numpy().astype(np.int32),
            "attention_mask": attention_mask.numpy().astype(np.float32 if args.mask_mode == "4d" else np.int32),
        }
        started = perf_counter()
        prediction = coreml_model.predict(coreml_inputs)
        coreml_ms = (perf_counter() - started) * 1000.0
        coreml_logits = np.asarray(prediction["logits"])

        non_padding = attention_mask_2d.numpy().astype(bool)
        abs_diff = np.abs(torch_logits - coreml_logits)
        torch_argmax = np.argmax(torch_logits, axis=-1)
        coreml_argmax = np.argmax(coreml_logits, axis=-1)
        argmax_agreement = float(np.mean(torch_argmax[non_padding] == coreml_argmax[non_padding]))
        results.append(
            {
                "id": item["id"],
                "max_abs_diff": float(np.max(abs_diff)),
                "mean_abs_diff": float(np.mean(abs_diff)),
                "argmax_agreement": argmax_agreement,
                "torch_ms": torch_ms,
                "coreml_ms": coreml_ms,
            }
        )

    payload = {
        "model_id": args.model_id,
        "coreml_model": str(args.coreml_model),
        "sequence_length": args.sequence_length,
        "mask_mode": args.mask_mode,
        "expert_mode": args.expert_mode,
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def make_4d_attention_mask(torch, attention_mask_2d, sliding_window: int):
    batch_size, sequence_length = attention_mask_2d.shape
    positions = torch.arange(sequence_length, device=attention_mask_2d.device)
    local = torch.abs(positions[:, None] - positions[None, :]) <= sliding_window
    key_padding = attention_mask_2d.to(torch.bool)[:, None, None, :]
    allowed = local[None, None, :, :] & key_padding
    dtype = torch.float32
    return torch.where(
        allowed,
        torch.tensor(0.0, device=attention_mask_2d.device, dtype=dtype),
        torch.tensor(torch.finfo(dtype).min, device=attention_mask_2d.device, dtype=dtype),
    ).expand(batch_size, 1, sequence_length, sequence_length)


if __name__ == "__main__":
    raise SystemExit(main())
