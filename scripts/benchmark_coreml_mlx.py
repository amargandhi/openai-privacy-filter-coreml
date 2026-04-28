#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np


COMPUTE_UNITS = {
    "all": "ALL",
    "cpuAndGPU": "CPU_AND_GPU",
    "cpuOnly": "CPU_ONLY",
    "cpuAndNeuralEngine": "CPU_AND_NE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Core ML against an MLX checkpoint.")
    parser.add_argument("--coreml-model", type=Path, required=True)
    parser.add_argument("--mlx-model", default="mlx-community/openai-privacy-filter-mxfp8")
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/privacy_samples.json"))
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--sliding-window", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--repeat-fixtures",
        type=int,
        default=1,
        help="Repeat the fixture set to create enough chunks for throughput benchmarks.",
    )
    parser.add_argument(
        "--coreml-batch-mode",
        choices=["auto", "api", "tensor"],
        default="auto",
        help=(
            "`api` uses Core ML batch prediction with a list of batch-1 inputs. "
            "`tensor` sends a real batch tensor and requires a batch-shaped mlpackage. "
            "`auto` uses tensor for batch size 1 and api otherwise."
        ),
    )
    parser.add_argument("--coreml-compute-unit", choices=sorted(COMPUTE_UNITS), default="all")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
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
            "Missing benchmark dependencies. Install with "
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
    return model, tokenizer, Path(model_path)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.repeat_fixtures < 1:
        raise SystemExit("--repeat-fixtures must be >= 1")
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    ct, mx, get_model_path, load_model, load_tokenizer = require_dependencies()
    fixtures = expand_fixtures(json.loads(args.fixtures.read_text(encoding="utf-8")), args.repeat_fixtures)

    started = perf_counter()
    mlx_model, tokenizer, mlx_model_path = load_privacy_filter_model(
        args.mlx_model,
        get_model_path,
        load_model,
        load_tokenizer,
    )
    mlx_load_ms = elapsed_ms(started)

    started = perf_counter()
    compute_unit = getattr(ct.ComputeUnit, COMPUTE_UNITS[args.coreml_compute_unit])
    coreml_model = ct.models.MLModel(str(args.coreml_model), compute_units=compute_unit)
    coreml_load_ms = elapsed_ms(started)

    samples = tokenize_samples(tokenizer, fixtures, args.sequence_length)
    coreml_batch_mode = resolve_coreml_batch_mode(args.coreml_batch_mode, args.batch_size)

    rows = []
    all_coreml_batch_times = []
    all_mlx_batch_times = []
    all_coreml_sample_times = []
    all_mlx_sample_times = []
    total_non_padding_tokens = 0
    pad_final_batch = coreml_batch_mode == "tensor"
    for batch_index, (batch, real_sample_count) in enumerate(
        batched(samples, args.batch_size, args.sequence_length, pad_final=pad_final_batch)
    ):
        input_ids = np.stack([item["input_ids"] for item in batch]).astype(np.int32)
        attention_mask_2d = np.stack([item["attention_mask"] for item in batch]).astype(np.int32)
        total_non_padding_tokens += int(attention_mask_2d[:real_sample_count].sum())
        coreml_attention_mask = make_4d_attention_mask(
            attention_mask_2d,
            args.sequence_length,
            args.sliding_window,
        )

        mlx_inputs = {
            "input_ids": mx.array(input_ids),
            "attention_mask": mx.array(attention_mask_2d),
        }
        coreml_inputs = make_coreml_inputs(input_ids, coreml_attention_mask, coreml_batch_mode)

        for _ in range(args.warmup):
            run_mlx(mx, mlx_model, mlx_inputs)
            run_coreml(coreml_model, coreml_inputs, coreml_batch_mode)

        mlx_logits, mlx_times = time_mlx(mx, mlx_model, mlx_inputs, args.iterations)
        coreml_logits, coreml_times = time_coreml(
            coreml_model,
            coreml_inputs,
            args.iterations,
            coreml_batch_mode,
        )
        all_mlx_batch_times.extend(mlx_times)
        all_coreml_batch_times.extend(coreml_times)
        all_mlx_sample_times.extend(time / real_sample_count for time in mlx_times)
        all_coreml_sample_times.extend(time / real_sample_count for time in coreml_times)

        rows.append(
            summarize_batch(
                batch_index,
                [str(item["id"]) for item in batch[:real_sample_count]],
                [int(mask.sum()) for mask in attention_mask_2d[:real_sample_count]],
                real_sample_count,
                mlx_logits,
                coreml_logits,
                attention_mask_2d.astype(bool),
                mlx_times,
                coreml_times,
            )
        )

    payload = {
        "coreml_model": str(args.coreml_model),
        "coreml_compute_unit": args.coreml_compute_unit,
        "coreml_load_ms": coreml_load_ms,
        "coreml_package_size_bytes": path_size(args.coreml_model),
        "mlx_model": args.mlx_model,
        "mlx_model_path": str(mlx_model_path),
        "mlx_load_ms": mlx_load_ms,
        "mlx_model_size_bytes": path_size(mlx_model_path),
        "sample_count": len(samples),
        "batch_count": len(rows),
        "batch_size": args.batch_size,
        "coreml_batch_mode": coreml_batch_mode,
        "sequence_length": args.sequence_length,
        "sliding_window": args.sliding_window,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "summary": {
            "coreml_batch_ms": summarize_times(all_coreml_batch_times),
            "mlx_batch_ms": summarize_times(all_mlx_batch_times),
            "coreml_ms": summarize_times(all_coreml_sample_times),
            "mlx_ms": summarize_times(all_mlx_sample_times),
            "coreml_ms_per_sample": summarize_times(all_coreml_sample_times),
            "mlx_ms_per_sample": summarize_times(all_mlx_sample_times),
            "coreml_non_padding_tokens_per_second": tokens_per_second(
                total_non_padding_tokens * args.iterations,
                sum(all_coreml_batch_times),
            ),
            "mlx_non_padding_tokens_per_second": tokens_per_second(
                total_non_padding_tokens * args.iterations,
                sum(all_mlx_batch_times),
            ),
            "coreml_speedup_vs_mlx_mean": safe_ratio(
                statistics.fmean(all_mlx_sample_times),
                statistics.fmean(all_coreml_sample_times),
            ),
            "coreml_to_mlx_mean_latency_ratio": safe_ratio(
                statistics.fmean(all_coreml_sample_times),
                statistics.fmean(all_mlx_sample_times),
            ),
            "min_argmax_agreement": min(row["argmax_agreement"] for row in rows),
            "max_abs_diff": max(row["max_abs_diff"] for row in rows),
            "max_mean_abs_diff": max(row["mean_abs_diff"] for row in rows),
        },
        "results": rows,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def expand_fixtures(fixtures: list[dict[str, object]], repeat_count: int) -> list[dict[str, object]]:
    if repeat_count == 1:
        return fixtures
    expanded = []
    for repeat_index in range(repeat_count):
        for item in fixtures:
            expanded.append(
                {
                    **item,
                    "id": f"{item['id']}#{repeat_index + 1}",
                }
            )
    return expanded


def tokenize_samples(tokenizer, fixtures: list[dict[str, object]], sequence_length: int) -> list[dict[str, object]]:
    samples = []
    for item in fixtures:
        encoded = tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
        )
        samples.append(
            {
                "id": str(item["id"]),
                "input_ids": np.asarray(encoded["input_ids"], dtype=np.int32),
                "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int32),
                "is_padding": False,
            }
        )
    return samples


def batched(items: list[dict[str, object]], batch_size: int, sequence_length: int, pad_final: bool):
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        real_sample_count = len(batch)
        if pad_final and real_sample_count < batch_size:
            batch = [*batch]
            for pad_index in range(batch_size - real_sample_count):
                batch.append(make_padding_sample(sequence_length, pad_index))
        yield batch, real_sample_count


def make_padding_sample(sequence_length: int, index: int) -> dict[str, object]:
    return {
        "id": f"__padding__{index}",
        "input_ids": np.zeros(sequence_length, dtype=np.int32),
        "attention_mask": np.zeros(sequence_length, dtype=np.int32),
        "is_padding": True,
    }


def resolve_coreml_batch_mode(mode: str, batch_size: int) -> str:
    if mode != "auto":
        return mode
    return "tensor" if batch_size == 1 else "api"


def make_coreml_inputs(input_ids: np.ndarray, attention_mask: np.ndarray, batch_mode: str):
    if batch_mode == "tensor":
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    if batch_mode == "api":
        return [
            {
                "input_ids": input_ids[index : index + 1],
                "attention_mask": attention_mask[index : index + 1],
            }
            for index in range(input_ids.shape[0])
        ]
    raise ValueError(f"Unknown Core ML batch mode: {batch_mode}")


def run_mlx(mx, model, inputs):
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    mx.eval(outputs.logits)
    return np.asarray(outputs.logits, dtype=np.float32)


def run_coreml(model, inputs, batch_mode: str):
    prediction = model.predict(inputs)
    if batch_mode == "api":
        return np.concatenate(
            [np.asarray(item["logits"], dtype=np.float32) for item in prediction],
            axis=0,
        )
    return np.asarray(prediction["logits"], dtype=np.float32)


def time_mlx(mx, model, inputs, iterations: int):
    times = []
    logits = None
    for _ in range(iterations):
        started = perf_counter()
        logits = run_mlx(mx, model, inputs)
        times.append(elapsed_ms(started))
    if logits is None:
        raise RuntimeError("No MLX benchmark iterations ran")
    return logits, times


def time_coreml(model, inputs, iterations: int, batch_mode: str):
    times = []
    logits = None
    for _ in range(iterations):
        started = perf_counter()
        logits = run_coreml(model, inputs, batch_mode)
        times.append(elapsed_ms(started))
    if logits is None:
        raise RuntimeError("No Core ML benchmark iterations ran")
    return logits, times


def summarize_batch(
    batch_index: int,
    sample_ids: list[str],
    token_counts: list[int],
    real_sample_count: int,
    mlx_logits: np.ndarray,
    coreml_logits: np.ndarray,
    non_padding: np.ndarray,
    mlx_times: list[float],
    coreml_times: list[float],
) -> dict[str, object]:
    abs_diff = np.abs(mlx_logits - coreml_logits)
    mlx_argmax = np.argmax(mlx_logits, axis=-1)
    coreml_argmax = np.argmax(coreml_logits, axis=-1)
    return {
        "id": f"batch-{batch_index:04d}",
        "sample_ids": sample_ids,
        "sample_count": real_sample_count,
        "token_count": int(sum(token_counts)),
        "token_counts": token_counts,
        "argmax_agreement": float(np.mean(mlx_argmax[non_padding] == coreml_argmax[non_padding])),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "mlx_batch_ms": summarize_times(mlx_times),
        "coreml_batch_ms": summarize_times(coreml_times),
        "mlx_ms": summarize_times([time / real_sample_count for time in mlx_times]),
        "coreml_ms": summarize_times([time / real_sample_count for time in coreml_times]),
        "mlx_ms_per_sample": summarize_times([time / real_sample_count for time in mlx_times]),
        "coreml_ms_per_sample": summarize_times([time / real_sample_count for time in coreml_times]),
        "coreml_speedup_vs_mlx_mean": safe_ratio(
            statistics.fmean(time / real_sample_count for time in mlx_times),
            statistics.fmean(time / real_sample_count for time in coreml_times),
        ),
        "coreml_to_mlx_mean_latency_ratio": safe_ratio(
            statistics.fmean(time / real_sample_count for time in coreml_times),
            statistics.fmean(time / real_sample_count for time in mlx_times),
        ),
    }


def summarize_times(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        "min": float(sorted_values[0]),
        "p50": percentile(sorted_values, 50),
        "p90": percentile(sorted_values, 90),
        "max": float(sorted_values[-1]),
        "mean": float(statistics.fmean(sorted_values)),
    }


def percentile(sorted_values: list[float], percent: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = (len(sorted_values) - 1) * percent / 100.0
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = index - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


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


def tokens_per_second(token_count: int, elapsed_ms_value: float) -> float | None:
    if elapsed_ms_value == 0:
        return None
    return float(token_count / (elapsed_ms_value / 1000.0))


if __name__ == "__main__":
    raise SystemExit(main())
