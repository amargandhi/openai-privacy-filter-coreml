from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .bioes import normalize_id2label, split_label


DEFAULT_BIASES = {
    "transition_bias_background_stay": 0.0,
    "transition_bias_background_to_start": 0.0,
    "transition_bias_end_to_background": 0.0,
    "transition_bias_end_to_start": 0.0,
    "transition_bias_inside_to_continue": 0.0,
    "transition_bias_inside_to_end": 0.0,
}


def load_viterbi_biases(path: str | Path, operating_point: str = "default") -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    biases = (
        payload.get("operating_points", {})
        .get(operating_point, {})
        .get("biases", {})
    )
    merged = dict(DEFAULT_BIASES)
    merged.update({key: float(value) for key, value in biases.items()})
    return merged


def constrained_viterbi(
    logits: np.ndarray,
    id2label: Mapping[str | int, str],
    *,
    biases: Mapping[str, float] | None = None,
) -> list[int]:
    """Decode logits with BIOES transition constraints.

    Args:
        logits: Array shaped `[tokens, labels]`.
        id2label: Label map from model config.
        biases: Optional transition biases from `viterbi_calibration.json`.
    """

    scores = np.asarray(logits, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"logits must be rank 2 [tokens, labels], got {scores.shape}")
    token_count, label_count = scores.shape
    if token_count == 0:
        return []

    labels = normalize_id2label(id2label)
    if len(labels) != label_count:
        raise ValueError(f"id2label has {len(labels)} labels but logits have {label_count}")

    transition = _transition_matrix(labels, biases or DEFAULT_BIASES)
    start_mask = np.array([_allowed_start(labels[index]) for index in range(label_count)])
    end_mask = np.array([_allowed_end(labels[index]) for index in range(label_count)])
    impossible = np.float32(-1.0e9)

    dp = np.full((token_count, label_count), impossible, dtype=np.float32)
    backpointers = np.zeros((token_count, label_count), dtype=np.int32)
    dp[0] = scores[0] + np.where(start_mask, 0.0, impossible)

    for token_index in range(1, token_count):
        previous = dp[token_index - 1][:, None] + transition
        backpointers[token_index] = np.argmax(previous, axis=0)
        dp[token_index] = scores[token_index] + np.max(previous, axis=0)

    final_scores = dp[-1] + np.where(end_mask, 0.0, impossible)
    last_label = int(np.argmax(final_scores))
    path = [last_label]
    for token_index in range(token_count - 1, 0, -1):
        last_label = int(backpointers[token_index, last_label])
        path.append(last_label)
    path.reverse()
    return path


def _transition_matrix(labels: Mapping[int, str], biases: Mapping[str, float]) -> np.ndarray:
    count = len(labels)
    matrix = np.full((count, count), -1.0e9, dtype=np.float32)
    for previous in range(count):
        for current in range(count):
            if _allowed_transition(labels[previous], labels[current]):
                matrix[previous, current] = _transition_bias(labels[previous], labels[current], biases)
    return matrix


def _allowed_start(label: str) -> bool:
    tag, _ = split_label(label)
    return tag in {"O", "B", "S"}


def _allowed_end(label: str) -> bool:
    tag, _ = split_label(label)
    return tag in {"O", "E", "S"}


def _allowed_transition(previous: str, current: str) -> bool:
    prev_tag, prev_category = split_label(previous)
    curr_tag, curr_category = split_label(current)

    if prev_tag == "O":
        return curr_tag in {"O", "B", "S"}
    if prev_tag in {"B", "I"}:
        return curr_category == prev_category and curr_tag in {"I", "E"}
    if prev_tag in {"E", "S"}:
        return curr_tag in {"O", "B", "S"}
    return False


def _transition_bias(previous: str, current: str, biases: Mapping[str, float]) -> float:
    prev_tag, _ = split_label(previous)
    curr_tag, _ = split_label(current)

    if prev_tag == "O" and curr_tag == "O":
        return float(biases.get("transition_bias_background_stay", 0.0))
    if prev_tag == "O" and curr_tag in {"B", "S"}:
        return float(biases.get("transition_bias_background_to_start", 0.0))
    if prev_tag in {"B", "I"} and curr_tag == "I":
        return float(biases.get("transition_bias_inside_to_continue", 0.0))
    if prev_tag in {"B", "I"} and curr_tag == "E":
        return float(biases.get("transition_bias_inside_to_end", 0.0))
    if prev_tag in {"E", "S"} and curr_tag == "O":
        return float(biases.get("transition_bias_end_to_background", 0.0))
    if prev_tag in {"E", "S"} and curr_tag in {"B", "S"}:
        return float(biases.get("transition_bias_end_to_start", 0.0))
    return 0.0
