"""Hierarchical sentence-first, tag-second negation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .crf_baseline import predict_crf, train_crf
from .metrics import compute_classification_metrics, compute_span_metrics, precision_recall_f1, safe_div


def _require_hierarchical_dependencies() -> tuple[object, object]:
    """Import sklearn pieces lazily."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for the hierarchical pipeline. Install the project dependencies first."
        ) from error
    return TfidfVectorizer, LogisticRegression


@dataclass
class HierarchicalNegationPipeline:
    """Two-stage model: sentence negation filter, then sequence tagging."""

    min_df: int = 2
    max_features: int = 20000
    max_iter: int = 1000
    sentence_vectorizer: object | None = None
    sentence_classifier: object | None = None
    sequence_tagger: object | None = None

    def fit(self, train_records: list[dict[str, object]]) -> "HierarchicalNegationPipeline":
        """Train the sentence classifier and positive-sentence tagger."""
        TfidfVectorizer, LogisticRegression = _require_hierarchical_dependencies()

        self.sentence_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            min_df=self.min_df,
            max_features=self.max_features,
        )
        sentence_matrix = self.sentence_vectorizer.fit_transform(
            [str(record["sentence"]) for record in train_records]
        )
        targets = [int(record["target"]) for record in train_records]
        self.sentence_classifier = LogisticRegression(
            max_iter=self.max_iter,
            class_weight="balanced",
        )
        self.sentence_classifier.fit(sentence_matrix, targets)

        positive_records = [record for record in train_records if int(record["target"]) == 1]
        self.sequence_tagger = train_crf(positive_records)
        return self

    def predict(self, records: list[dict[str, object]]) -> tuple[list[int], list[list[str]]]:
        """Run sentence prediction followed by conditional sequence tagging."""
        if self.sentence_vectorizer is None or self.sentence_classifier is None or self.sequence_tagger is None:
            raise RuntimeError("The hierarchical pipeline must be fit before prediction.")

        sentence_matrix = self.sentence_vectorizer.transform([str(record["sentence"]) for record in records])
        sentence_predictions = [int(value) for value in self.sentence_classifier.predict(sentence_matrix)]

        positive_records = [
            record for record, prediction in zip(records, sentence_predictions) if prediction == 1
        ]
        positive_predictions = predict_crf(self.sequence_tagger, positive_records) if positive_records else []

        final_predictions: list[list[str]] = []
        positive_index = 0
        for record, sentence_prediction in zip(records, sentence_predictions):
            if sentence_prediction == 1:
                final_predictions.append(positive_predictions[positive_index])
                positive_index += 1
            else:
                final_predictions.append(["O"] * len(record["tokens"]))

        return sentence_predictions, final_predictions

    def evaluate(self, records: list[dict[str, object]]) -> dict[str, object]:
        """Evaluate both stages together."""
        sentence_predictions, sequence_predictions = self.predict(records)
        sentence_targets = [int(record["target"]) for record in records]
        true_sequences = [list(record["word_labels"]) for record in records]

        sentence_tp = sum(1 for true, pred in zip(sentence_targets, sentence_predictions) if true == 1 and pred == 1)
        sentence_fp = sum(1 for true, pred in zip(sentence_targets, sentence_predictions) if true == 0 and pred == 1)
        sentence_fn = sum(1 for true, pred in zip(sentence_targets, sentence_predictions) if true == 1 and pred == 0)
        sentence_precision, sentence_recall, sentence_f1 = precision_recall_f1(
            sentence_tp,
            sentence_fp,
            sentence_fn,
        )

        sentence_accuracy = safe_div(
            sum(1 for true, pred in zip(sentence_targets, sentence_predictions) if true == pred),
            len(sentence_targets),
        )

        return {
            "sentence": {
                "accuracy": sentence_accuracy,
                "precision": sentence_precision,
                "recall": sentence_recall,
                "f1": sentence_f1,
            },
            "token": compute_classification_metrics(true_sequences, sequence_predictions),
            "span": compute_span_metrics(true_sequences, sequence_predictions),
        }

