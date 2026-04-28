#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from privacy_filter_coreml.bioes import decode_bioes_predictions, normalize_id2label
from privacy_filter_coreml.viterbi import constrained_viterbi, load_viterbi_biases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Transformers Privacy Filter fixtures.")
    parser.add_argument("--model-id", default="openai/privacy-filter")
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/privacy_samples.json"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--decode", choices=["argmax", "viterbi"], default="viterbi")
    parser.add_argument("--operating-point", default="default")
    parser.add_argument("--viterbi-calibration", type=Path)
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def require_dependencies():
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing Transformers dependencies. Install with "
            '`python -m pip install -e ".[convert]"`.'
        ) from exc
    return torch, hf_hub_download, AutoModelForTokenClassification, AutoTokenizer


def main() -> int:
    args = parse_args()
    torch, hf_hub_download, AutoModelForTokenClassification, AutoTokenizer = require_dependencies()
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))

    started_load = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    torch_dtype = torch.float32 if args.torch_dtype == "float32" else torch.bfloat16
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model.eval()
    load_ms = (perf_counter() - started_load) * 1000.0

    id2label = normalize_id2label(model.config.id2label)
    biases = None
    calibration_path = None
    if args.decode == "viterbi":
        calibration_path = args.viterbi_calibration
        if calibration_path is None:
            calibration_path = Path(
                hf_hub_download(
                    repo_id=args.model_id,
                    filename="viterbi_calibration.json",
                )
            )
        biases = load_viterbi_biases(calibration_path, args.operating_point)

    results = []
    for item in fixtures:
        encoded = tokenizer(
            item["text"],
            return_tensors="pt",
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        started = perf_counter()
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits.detach().cpu().float().numpy()[0]
        elapsed_ms = (perf_counter() - started) * 1000.0

        predictions = decode_predictions(logits, attention_mask[0].tolist(), id2label, args.decode, biases)
        special_mask = [value == 0 for value in attention_mask[0].tolist()]
        spans = decode_bioes_predictions(
            predictions,
            id2label,
            offsets=offsets,
            text=item["text"],
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
        "backend": "transformers",
        "model": args.model_id,
        "decode": args.decode,
        "operating_point": args.operating_point if args.decode == "viterbi" else None,
        "viterbi_calibration": str(calibration_path) if calibration_path else None,
        "load_ms": load_ms,
        "max_length": args.max_length,
        "torch_dtype": args.torch_dtype,
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def decode_predictions(
    logits: np.ndarray,
    attention_mask: list[int],
    id2label: dict[int, str],
    decode: str,
    biases: dict[str, float] | None,
) -> list[int]:
    if decode == "argmax":
        return np.argmax(logits, axis=-1).astype(int).tolist()

    valid_length = int(sum(attention_mask))
    path = constrained_viterbi(logits[:valid_length], id2label, biases=biases)
    return path + [0] * (len(attention_mask) - valid_length)


if __name__ == "__main__":
    raise SystemExit(main())
