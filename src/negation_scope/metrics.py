"""Pure-Python token and span metrics used across experiments."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .constants import ID_TO_LABEL, LABELS, MINORITY_LABELS


def safe_div(numerator: float, denominator: float) -> float:
    """Return a stable division result."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 from confusion counts."""
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def flatten_sequences(sequences: Iterable[Iterable[str]]) -> list[str]:
    """Flatten nested label sequences into one list."""
    flattened: list[str] = []
    for sequence in sequences:
        flattened.extend(sequence)
    return flattened


def compute_classification_metrics(
    true_sequences: list[list[str]],
    predicted_sequences: list[list[str]],
    labels: tuple[str, ...] = LABELS,
) -> dict[str, object]:
    """Compute token-level metrics in the style used in the notebook."""
    true_flat = flatten_sequences(true_sequences)
    predicted_flat = flatten_sequences(predicted_sequences)
    if len(true_flat) != len(predicted_flat):
        raise ValueError("True and predicted label sequences must have the same length")

    total = len(true_flat)
    accuracy = safe_div(sum(1 for true, pred in zip(true_flat, predicted_flat) if true == pred), total)

    observed_labels = tuple(label for label in labels if label in set(true_flat))
    if not observed_labels:
        observed_labels = ("O",)

    per_label: dict[str, dict[str, float]] = {}
    supports: list[int] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in observed_labels:
        tp = sum(1 for true, pred in zip(true_flat, predicted_flat) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(true_flat, predicted_flat) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(true_flat, predicted_flat) if true == label and pred != label)
        support = sum(1 for true in true_flat if true == label)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }

        supports.append(support)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    weighted_denominator = sum(supports) or 1
    micro_precision, micro_recall, micro_f1 = precision_recall_f1(total_tp, total_fp, total_fn)

    return {
        "accuracy": accuracy,
        "precision_macro": sum(precision_values) / len(precision_values),
        "recall_macro": sum(recall_values) / len(recall_values),
        "f1_macro": sum(f1_values) / len(f1_values),
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
        "f1_micro": micro_f1,
        "precision_weighted": sum(value * support for value, support in zip(precision_values, supports)) / weighted_denominator,
        "recall_weighted": sum(value * support for value, support in zip(recall_values, supports)) / weighted_denominator,
        "f1_weighted": sum(value * support for value, support in zip(f1_values, supports)) / weighted_denominator,
        "per_label": per_label,
    }


def extract_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    """Convert BIO tags into inclusive token spans."""
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    current_entity: str | None = None

    def close_span(end_index: int) -> None:
        nonlocal start, current_entity
        if start is not None and current_entity is not None:
            spans.append((start, end_index, current_entity))
        start = None
        current_entity = None

    for index, label in enumerate(labels):
        if label == "O":
            close_span(index - 1)
            continue

        prefix, entity = label.split("-", maxsplit=1)
        if prefix == "B":
            close_span(index - 1)
            start = index
            current_entity = entity
            continue

        if prefix == "I" and current_entity == entity and start is not None:
            continue

        close_span(index - 1)
        start = index
        current_entity = entity

    close_span(len(labels) - 1)
    return spans


def spans_overlap(left: tuple[int, int, str], right: tuple[int, int, str]) -> bool:
    """Return True when two spans of the same type overlap."""
    same_entity = left[2] == right[2]
    interval_overlap = left[0] <= right[1] and left[1] >= right[0]
    return same_entity and interval_overlap


def compute_span_metrics(
    true_sequences: list[list[str]],
    predicted_sequences: list[list[str]],
) -> dict[str, dict[str, float]]:
    """Compute strict and overlap span matching metrics."""
    strict_tp = strict_fp = strict_fn = 0
    overlap_tp = overlap_fp = overlap_fn = 0

    for true_labels, predicted_labels in zip(true_sequences, predicted_sequences):
        true_spans = extract_spans(true_labels)
        predicted_spans = extract_spans(predicted_labels)

        true_counter = Counter(true_spans)
        predicted_counter = Counter(predicted_spans)

        strict_matches = true_counter & predicted_counter
        strict_tp += sum(strict_matches.values())
        strict_fp += len(predicted_spans) - sum(strict_matches.values())
        strict_fn += len(true_spans) - sum(strict_matches.values())

        matched_true: set[int] = set()
        matched_pred: set[int] = set()
        for pred_index, predicted_span in enumerate(predicted_spans):
            for true_index, true_span in enumerate(true_spans):
                if true_index in matched_true:
                    continue
                if spans_overlap(predicted_span, true_span):
                    matched_true.add(true_index)
                    matched_pred.add(pred_index)
                    overlap_tp += 1
                    break

        overlap_fp += len(predicted_spans) - len(matched_pred)
        overlap_fn += len(true_spans) - len(matched_true)

    strict_precision, strict_recall, strict_f1 = precision_recall_f1(strict_tp, strict_fp, strict_fn)
    overlap_precision, overlap_recall, overlap_f1 = precision_recall_f1(overlap_tp, overlap_fp, overlap_fn)

    return {
        "strict": {
            "precision": strict_precision,
            "recall": strict_recall,
            "f1": strict_f1,
        },
        "overlap": {
            "precision": overlap_precision,
            "recall": overlap_recall,
            "f1": overlap_f1,
        },
    }


def decode_label_id_sequences(
    true_label_ids: list[list[int]],
    predicted_label_ids: list[list[int]],
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert numeric label IDs into string labels while dropping ignored positions."""
    decoded_true: list[list[str]] = []
    decoded_pred: list[list[str]] = []

    for true_ids, predicted_ids in zip(true_label_ids, predicted_label_ids):
        true_labels: list[str] = []
        predicted_labels: list[str] = []
        for true_id, predicted_id in zip(true_ids, predicted_ids):
            if true_id == -100:
                continue
            true_labels.append(ID_TO_LABEL[int(true_id)])
            predicted_labels.append(ID_TO_LABEL[int(predicted_id)])
        decoded_true.append(true_labels)
        decoded_pred.append(predicted_labels)

    return decoded_true, decoded_pred


def invalid_bio_transitions(labels: list[str]) -> int:
    """Count BIO mistakes such as I-tags that do not follow the right entity."""
    invalid = 0
    previous_entity: str | None = None

    for label in labels:
        if label == "O":
            previous_entity = None
            continue

        prefix, entity = label.split("-", maxsplit=1)
        if prefix == "B":
            previous_entity = entity
            continue

        if prefix == "I" and previous_entity == entity:
            continue

        invalid += 1
        previous_entity = entity

    return invalid


def minority_recall(metrics: dict[str, object]) -> float:
    """Average recall over the minority labels."""
    per_label = metrics["per_label"]
    recalls = [per_label[label]["recall"] for label in MINORITY_LABELS]
    return sum(recalls) / len(recalls)
