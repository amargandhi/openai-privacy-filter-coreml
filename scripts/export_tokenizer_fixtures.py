#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXTRA_SAMPLES = [
    {
        "id": "unicode_scalar_offsets",
        "text": "Zoë paid £20 and waved 👋🏽.",
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Python tokenizer fixtures for Swift tests.")
    parser.add_argument("--model-id", default="openai/privacy-filter")
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/privacy_samples.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def require_dependencies():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing Transformers dependencies. Install with "
            '`python -m pip install -e ".[convert]"`.'
        ) from exc
    return AutoTokenizer


def main() -> int:
    args = parse_args()
    AutoTokenizer = require_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    samples = json.loads(args.fixtures.read_text(encoding="utf-8")) + EXTRA_SAMPLES

    encoded_samples = []
    for item in samples:
        encoded = tokenizer(
            item["text"],
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        encoded_samples.append(
            {
                "id": item["id"],
                "text": item["text"],
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "offsets": [
                    {"start": int(start), "end": int(end)}
                    for start, end in encoded["offset_mapping"]
                ],
            }
        )

    payload = {
        "model_id": args.model_id,
        "max_length": args.max_length,
        "samples": encoded_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
