"""CRF baseline for word-level cue and scope tagging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .features import sentence_features
from .metrics import compute_classification_metrics, compute_span_metrics


def _require_crf_dependencies() -> Any:
    """Import sklearn-crfsuite lazily so preprocessing still works without it."""
    try:
        import sklearn_crfsuite
    except ImportError as error:
        raise ImportError(
            "sklearn-crfsuite is required for the CRF baseline. Install the project dependencies first."
        ) from error
    return sklearn_crfsuite


def prepare_crf_inputs(records: list[dict[str, object]]) -> tuple[list[list[dict[str, object]]], list[list[str]]]:
    """Convert records into CRF features and labels."""
    features = [sentence_features(list(record["tokens"])) for record in records]
    labels = [list(record["word_labels"]) for record in records]
    return features, labels


def train_crf(
    train_records: list[dict[str, object]],
    c1: float = 0.1,
    c2: float = 0.1,
    max_iterations: int = 100,
) -> object:
    """Fit a CRF model over word-level features."""
    sklearn_crfsuite = _require_crf_dependencies()
    train_x, train_y = prepare_crf_inputs(train_records)
    model = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
    )
    model.fit(train_x, train_y)
    return model


def predict_crf(model: object, records: list[dict[str, object]]) -> list[list[str]]:
    """Generate CRF predictions for a list of records."""
    features, _ = prepare_crf_inputs(records)
    return list(model.predict(features))


def evaluate_crf(model: object, records: list[dict[str, object]]) -> dict[str, object]:
    """Evaluate the CRF baseline on token and span metrics."""
    predicted = predict_crf(model, records)
    true = [list(record["word_labels"]) for record in records]
    return {
        "token": compute_classification_metrics(true, predicted),
        "span": compute_span_metrics(true, predicted),
    }


def save_crf_model(model: object, output_path: str | Path) -> None:
    """Serialize the fitted CRF using pickle."""
    import pickle

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(model, handle)

