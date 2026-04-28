#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from privacy_filter_coreml.bioes import decode_bioes_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLX Privacy Filter fixtures and emit JSON.")
    parser.add_argument("--model", default="mlx-community/openai-privacy-filter-mxfp8")
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/privacy_samples.json"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def require_dependencies():
    try:
        import mlx.core as mx
        from mlx_embeddings.utils import load
    except ImportError as exc:
        raise SystemExit(
            "Missing MLX dependencies. Install with "
            '`python -m pip install -e ".[mlx]"`.'
        ) from exc
    return mx, load


def main() -> int:
    args = parse_args()
    mx, load = require_dependencies()
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))

    started_load = perf_counter()
    model, tokenizer = load(args.model)
    load_ms = (perf_counter() - started_load) * 1000.0
    id2label = model.config.id2label

    results = []
    for item in fixtures:
        text = item["text"]
        encode_kwargs = {
            "return_tensors": "mlx",
            "padding": "max_length",
            "truncation": True,
            "max_length": args.max_length,
        }
        try:
            inputs = tokenizer(text, return_offsets_mapping=True, **encode_kwargs)
            offsets = [tuple(pair) for pair in inputs.pop("offset_mapping")[0].tolist()]
        except TypeError:
            inputs = tokenizer(text, **encode_kwargs)
            offsets = None

        started = perf_counter()
        outputs = model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        mx.eval(outputs.logits)
        elapsed_ms = (perf_counter() - started) * 1000.0

        predictions = mx.argmax(outputs.logits, axis=-1)[0].tolist()
        attention_mask = inputs.get("attention_mask")
        special_mask = None
        if attention_mask is not None:
            special_mask = [value == 0 for value in attention_mask[0].tolist()]
        spans = decode_bioes_predictions(
            predictions,
            id2label,
            offsets=offsets,
            text=text,
            special_token_mask=special_mask,
        )
        labels = sorted({span.label for span in spans})
        results.append(
            {
                "id": item["id"],
                "expected_labels": item.get("expected_labels", []),
                "labels": labels,
                "spans": [span.to_dict() for span in spans],
                "elapsed_ms": elapsed_ms,
            }
        )

    payload = {
        "backend": "mlx",
        "model": args.model,
        "load_ms": load_ms,
        "max_length": args.max_length,
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
