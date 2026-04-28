#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Core ML logits against an MLX checkpoint.")
    parser.add_argument("--mlx-model", default="mlx-community/openai-privacy-filter-mxfp8")
    parser.add_argument("--coreml-model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/privacy_samples.json"))
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--sliding-window", type=int, default=128)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def require_dependencies():
    try:
        import coremltools as ct
        import mlx.core as mx
        from mlx_embeddings.tokenizer_utils import load_tokenizer
        from mlx_embeddings.utils import get_model_path, load_model
    except ImportError as exc:
        raise SystemExit(
            "Missing comparison dependencies. Install with "
            '`python -m pip install -e ".[convert,mlx]"`.'
        ) from exc
    return ct, mx, get_model_path, load_model, load_tokenizer


def load_privacy_filter_model(model_ref: str, get_model_path, load_model, load_tokenizer):
    from privacy_filter_coreml.mlx_openai_privacy_filter import Model, ModelArgs

    def get_model_classes(config):
        return Model, ModelArgs, None, None

    model_path = get_model_path(model_ref)
    model = load_model(
        model_path,
        get_model_classes=get_model_classes,
        path_to_repo=model_ref,
    )
    tokenizer = load_tokenizer(model_path, {})
    return model, tokenizer


def main() -> int:
    args = parse_args()
    ct, mx, get_model_path, load_model, load_tokenizer = require_dependencies()
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))

    started_load = perf_counter()
    mlx_model, tokenizer = load_privacy_filter_model(
        args.mlx_model,
        get_model_path,
        load_model,
        load_tokenizer,
    )
    mlx_load_ms = (perf_counter() - started_load) * 1000.0
    coreml_model = ct.models.MLModel(str(args.coreml_model), compute_units=ct.ComputeUnit.ALL)

    results = []
    for item in fixtures:
        encoded = tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=args.sequence_length,
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int32)[None, :]
        attention_mask_2d = np.asarray(encoded["attention_mask"], dtype=np.int32)[None, :]
        coreml_attention_mask = make_4d_attention_mask(
            attention_mask_2d,
            args.sequence_length,
            args.sliding_window,
        )

        started = perf_counter()
        mlx_outputs = mlx_model(mx.array(input_ids), attention_mask=mx.array(attention_mask_2d))
        mx.eval(mlx_outputs.logits)
        mlx_ms = (perf_counter() - started) * 1000.0
        mlx_logits = np.asarray(mlx_outputs.logits, dtype=np.float32)

        started = perf_counter()
        prediction = coreml_model.predict(
            {
                "input_ids": input_ids,
                "attention_mask": coreml_attention_mask,
            }
        )
        coreml_ms = (perf_counter() - started) * 1000.0
        coreml_logits = np.asarray(prediction["logits"], dtype=np.float32)

        non_padding = attention_mask_2d.astype(bool)
        abs_diff = np.abs(mlx_logits - coreml_logits)
        mlx_argmax = np.argmax(mlx_logits, axis=-1)
        coreml_argmax = np.argmax(coreml_logits, axis=-1)
        results.append(
            {
                "id": item["id"],
                "max_abs_diff": float(np.max(abs_diff)),
                "mean_abs_diff": float(np.mean(abs_diff)),
                "argmax_agreement": float(np.mean(mlx_argmax[non_padding] == coreml_argmax[non_padding])),
                "mlx_ms": mlx_ms,
                "coreml_ms": coreml_ms,
            }
        )

    payload = {
        "mlx_model": args.mlx_model,
        "coreml_model": str(args.coreml_model),
        "sequence_length": args.sequence_length,
        "sliding_window": args.sliding_window,
        "mlx_load_ms": mlx_load_ms,
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def make_4d_attention_mask(
    attention_mask_2d: np.ndarray,
    sequence_length: int,
    sliding_window: int,
) -> np.ndarray:
    positions = np.arange(sequence_length)
    local = np.abs(positions[:, None] - positions[None, :]) <= sliding_window
    key_padding = attention_mask_2d.astype(bool)[:, None, None, :]
    allowed = local[None, None, :, :] & key_padding
    return np.where(allowed, 0.0, np.finfo(np.float32).min).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
