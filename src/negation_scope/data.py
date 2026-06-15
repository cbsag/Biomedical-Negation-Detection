"""Dataset loading and span-to-BIO conversion helpers."""

from __future__ import annotations

import ast
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import DATASET_FILES, LABEL_TO_ID

WORD_PATTERN = re.compile(r"\S+")


def is_missing(value: object) -> bool:
    """Return True when a spreadsheet cell should be treated as empty."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none"}


def parse_cue_spans(raw_value: object) -> list[tuple[int, int]]:
    """Parse a cue span string such as '[(74, 78)]' into tuples."""
    if is_missing(raw_value):
        return []

    parsed = ast.literal_eval(str(raw_value))
    if isinstance(parsed, tuple) and len(parsed) == 2:
        parsed = [parsed]

    spans: list[tuple[int, int]] = []
    for item in parsed:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = int(item[0]), int(item[1])
            spans.append((start, end))
    return sorted(spans)


def parse_scope_span(raw_value: object) -> tuple[int, int] | None:
    """Parse a scope span string such as '[74, 110]' into a tuple."""
    if is_missing(raw_value):
        return None

    parsed = ast.literal_eval(str(raw_value))
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        return int(parsed[0]), int(parsed[1])
    if (
        isinstance(parsed, list)
        and len(parsed) == 1
        and isinstance(parsed[0], (list, tuple))
        and len(parsed[0]) == 2
    ):
        return int(parsed[0][0]), int(parsed[0][1])
    raise ValueError(f"Could not parse scope span from: {raw_value!r}")


def find_word_offsets(sentence: str) -> list[tuple[str, tuple[int, int]]]:
    """Split a sentence on whitespace while keeping character offsets."""
    return [(match.group(0), match.span()) for match in WORD_PATTERN.finditer(sentence)]


def has_overlap(
    token_start: int,
    token_end: int,
    span_start: int,
    span_end: int,
) -> bool:
    """Return True when a token and character span overlap at least one character."""
    return max(token_start, span_start) < min(token_end, span_end)


def apply_bio_labels(labels: list[str], indices: list[int], entity_name: str) -> None:
    """Apply BIO tags to the specified token indices in-place."""
    if not indices:
        return
    first_index = indices[0]
    for index in indices:
        prefix = "B" if index == first_index else "I"
        labels[index] = f"{prefix}-{entity_name}"


def build_word_labels(
    sentence: str,
    cue_spans: list[tuple[int, int]],
    scope_span: tuple[int, int] | None,
) -> tuple[list[str], list[tuple[int, int]], list[str]]:
    """Convert character spans to word-level BIO tags."""
    words_and_offsets = find_word_offsets(sentence)
    tokens = [token for token, _ in words_and_offsets]
    offsets = [offset for _, offset in words_and_offsets]
    labels = ["O"] * len(tokens)

    cue_indices_by_span: list[list[int]] = []
    cue_index_set: set[int] = set()
    for cue_start, cue_end in cue_spans:
        span_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if has_overlap(token_start, token_end, cue_start, cue_end)
        ]
        if span_indices:
            cue_indices_by_span.append(span_indices)
            cue_index_set.update(span_indices)

    if scope_span is not None:
        scope_start, scope_end = scope_span
        scope_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if has_overlap(token_start, token_end, scope_start, scope_end)
            and index not in cue_index_set
        ]
        apply_bio_labels(labels, scope_indices, "SCOPE")

    for cue_indices in cue_indices_by_span:
        apply_bio_labels(labels, cue_indices, "CUE")

    return tokens, offsets, labels


def row_to_record(row: pd.Series, source_name: str) -> dict[str, object]:
    """Convert one BIOSCOPE row into a JSON-serializable experiment record."""
    sentence = str(row["sentence"])
    cue_spans = parse_cue_spans(row["cue_span"])
    scope_span = parse_scope_span(row["scope_span"])
    tokens, token_offsets, word_labels = build_word_labels(sentence, cue_spans, scope_span)

    return {
        "sentence": sentence,
        "sentence_id": str(row["sentence_id"]),
        "source": source_name,
        "target": int(row["target"]),
        "cue_spans": cue_spans,
        "scope_span": scope_span,
        "tokens": tokens,
        "token_offsets": token_offsets,
        "word_labels": word_labels,
    }


def load_bioscope_records(
    data_dir: str | Path,
    dataset: str = "all",
) -> list[dict[str, object]]:
    """Load the selected BIOSCOPE subset and build token-level labels."""
    data_path = Path(data_dir)
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unknown dataset selection: {dataset}")

    records: list[dict[str, object]] = []
    for filename in DATASET_FILES[dataset]:
        frame = pd.read_excel(data_path / filename)
        for _, row in frame.iterrows():
            records.append(row_to_record(row, source_name=filename))
    return records


def summarize_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    """Summarize dataset size and label balance for quick inspection."""
    cached_records = list(records)
    sentence_counter = Counter(int(record["target"]) for record in cached_records)
    label_counter = Counter()
    for record in cached_records:
        label_counter.update(record["word_labels"])

    return {
        "num_sentences": len(cached_records),
        "num_negated": sentence_counter.get(1, 0),
        "num_non_negated": sentence_counter.get(0, 0),
        "word_label_counts": dict(label_counter),
    }


def stratified_split(
    records: list[dict[str, object]],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    seed: int = 13,
) -> dict[str, list[dict[str, object]]]:
    """Create deterministic train/dev/test splits while preserving sentence balance."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= dev_ratio < 1:
        raise ValueError("dev_ratio must be between 0 and 1")
    if train_ratio + dev_ratio >= 1:
        raise ValueError("train_ratio + dev_ratio must be < 1")

    rng = random.Random(seed)
    buckets: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for record in records:
        buckets[int(record["target"])].append(record)

    train_records: list[dict[str, object]] = []
    dev_records: list[dict[str, object]] = []
    test_records: list[dict[str, object]] = []

    for bucket in buckets.values():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_end = int(total * train_ratio)
        dev_end = train_end + int(total * dev_ratio)
        train_records.extend(shuffled[:train_end])
        dev_records.extend(shuffled[train_end:dev_end])
        test_records.extend(shuffled[dev_end:])

    rng.shuffle(train_records)
    rng.shuffle(dev_records)
    rng.shuffle(test_records)

    return {"train": train_records, "dev": dev_records, "test": test_records}


def write_jsonl(records: Iterable[dict[str, object]], output_path: str | Path) -> None:
    """Write records as newline-delimited JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def export_splits(
    data_dir: str | Path,
    output_dir: str | Path,
    dataset: str = "all",
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    seed: int = 13,
) -> dict[str, object]:
    """Load raw spreadsheets, create splits, and save them to disk."""
    records = load_bioscope_records(data_dir, dataset=dataset)
    splits = stratified_split(records, train_ratio=train_ratio, dev_ratio=dev_ratio, seed=seed)

    output_path = Path(output_dir)
    for split_name, split_records in splits.items():
        write_jsonl(split_records, output_path / f"{split_name}.jsonl")

    summary = summarize_records(records)
    summary["dataset"] = dataset
    summary["seed"] = seed
    summary["split_sizes"] = {name: len(items) for name, items in splits.items()}
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def continuation_label(label: str) -> str:
    """Convert B-tags to I-tags when a word is split into multiple wordpieces."""
    if label.startswith("B-"):
        return "I-" + label.split("-", maxsplit=1)[1]
    return label


def encode_record_for_transformer(
    record: dict[str, object],
    tokenizer: object,
    max_length: int = 128,
) -> dict[str, object]:
    """Align word-level labels to tokenizer wordpieces."""
    tokens = list(record["tokens"])
    word_labels = list(record["word_labels"])
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    word_ids = encoding.word_ids()

    labels: list[int] = []
    previous_word_id: int | None = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
            continue

        label = word_labels[word_id]
        if previous_word_id == word_id:
            label = continuation_label(label)
        labels.append(LABEL_TO_ID[label])
        previous_word_id = word_id

    encoded = {key: value for key, value in encoding.items()}
    encoded["labels"] = labels
    encoded["tokens"] = tokens
    return encoded

