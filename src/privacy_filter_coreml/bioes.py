from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DecodedSpan:
    label: str
    start: int | None
    end: int | None
    text: str | None
    token_start: int
    token_end: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def split_label(label: str) -> tuple[str, str | None]:
    if label == "O":
        return "O", None
    if "-" not in label:
        raise ValueError(f"Invalid BIOES label: {label!r}")
    tag, category = label.split("-", 1)
    if tag not in {"B", "I", "E", "S"}:
        raise ValueError(f"Invalid BIOES tag in label: {label!r}")
    return tag, category


def normalize_id2label(id2label: Mapping[str | int, str]) -> dict[int, str]:
    return {int(key): value for key, value in id2label.items()}


def decode_bioes_predictions(
    predictions: Sequence[int],
    id2label: Mapping[str | int, str],
    *,
    offsets: Sequence[tuple[int, int]] | None = None,
    text: str | None = None,
    special_token_mask: Iterable[bool] | None = None,
) -> list[DecodedSpan]:
    """Decode token-class predictions into privacy spans.

    This decoder is intentionally tolerant: malformed B/I/E sequences are
    repaired into the closest contiguous span so parity scripts keep producing
    inspectable output while conversion work is in progress.
    """

    labels = normalize_id2label(id2label)
    mask = list(special_token_mask) if special_token_mask is not None else [False] * len(predictions)
    if len(mask) != len(predictions):
        raise ValueError("special_token_mask length must match predictions length")
    if offsets is not None and len(offsets) != len(predictions):
        raise ValueError("offsets length must match predictions length")

    spans: list[DecodedSpan] = []
    active_category: str | None = None
    active_start: int | None = None

    def close(end_token_exclusive: int) -> None:
        nonlocal active_category, active_start
        if active_category is None or active_start is None:
            return
        spans.append(
            _make_span(
                active_category,
                active_start,
                end_token_exclusive,
                offsets=offsets,
                text=text,
            )
        )
        active_category = None
        active_start = None

    for index, pred in enumerate(predictions):
        if mask[index]:
            close(index)
            continue

        tag, category = split_label(labels[int(pred)])
        if tag == "O":
            close(index)
            continue

        if tag == "S":
            close(index)
            spans.append(_make_span(category or "", index, index + 1, offsets=offsets, text=text))
            continue

        if tag == "B":
            close(index)
            active_category = category
            active_start = index
            continue

        if active_category != category:
            close(index)
            active_category = category
            active_start = index

        if tag == "E":
            close(index + 1)

    close(len(predictions))
    return spans


def _make_span(
    label: str,
    token_start: int,
    token_end: int,
    *,
    offsets: Sequence[tuple[int, int]] | None,
    text: str | None,
) -> DecodedSpan:
    start: int | None = None
    end: int | None = None
    span_text: str | None = None
    if offsets is not None:
        usable_offsets = [item for item in offsets[token_start:token_end] if item[1] > item[0]]
        if usable_offsets:
            start = usable_offsets[0][0]
            end = usable_offsets[-1][1]
            if text is not None:
                span_text = text[start:end]
    return DecodedSpan(
        label=label,
        start=start,
        end=end,
        text=span_text,
        token_start=token_start,
        token_end=token_end,
    )
