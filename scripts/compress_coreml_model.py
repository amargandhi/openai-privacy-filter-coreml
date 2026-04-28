#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress an OpenAI Privacy Filter Core ML package.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["linear-int8", "linear-int4", "palettize-8", "palettize-6", "palettize-4"],
        default="linear-int8",
    )
    parser.add_argument("--weight-threshold", type=int, default=2048)
    parser.add_argument("--palettize-mode", choices=["kmeans", "uniform"], default="kmeans")
    parser.add_argument("--num-kmeans-workers", type=int, default=1)
    return parser.parse_args()


def require_dependencies():
    try:
        import coremltools as ct
        import coremltools.optimize.coreml as cto
    except ImportError as exc:
        raise SystemExit(
            "Missing Core ML dependencies. Install with "
            '`python -m pip install -e ".[convert]"`.'
        ) from exc
    return ct, cto


def main() -> int:
    args = parse_args()
    ct, cto = require_dependencies()
    started = perf_counter()
    model = ct.models.MLModel(str(args.input), skip_model_load=True)
    load_ms = elapsed_ms(started)

    started = perf_counter()
    compressed = compress_model(cto, model, args)
    compress_ms = elapsed_ms(started)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    compressed.save(str(args.output))
    save_ms = elapsed_ms(started)

    payload = {
        "input": str(args.input),
        "output": str(args.output),
        "method": args.method,
        "input_size_bytes": path_size(args.input),
        "output_size_bytes": path_size(args.output),
        "compression_ratio": safe_ratio(path_size(args.input), path_size(args.output)),
        "load_ms": load_ms,
        "compress_ms": compress_ms,
        "save_ms": save_ms,
    }
    args.output.with_suffix(args.output.suffix + ".compression.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def compress_model(cto, model, args):
    if args.method.startswith("linear"):
        dtype = "int8" if args.method == "linear-int8" else "int4"
        config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=dtype,
                weight_threshold=args.weight_threshold,
            )
        )
        return cto.linear_quantize_weights(model, config)

    nbits = int(args.method.rsplit("-", 1)[1])
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(
            mode=args.palettize_mode,
            nbits=nbits,
            num_kmeans_workers=args.num_kmeans_workers,
            weight_threshold=args.weight_threshold,
        )
    )
    return cto.palettize_weights(model, config)


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000.0


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


if __name__ == "__main__":
    raise SystemExit(main())
