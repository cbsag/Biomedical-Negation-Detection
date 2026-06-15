from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from negation_scope.metrics import compute_classification_metrics, compute_span_metrics, invalid_bio_transitions


class MetricTests(unittest.TestCase):
    def test_token_metrics_are_perfect_for_identical_sequences(self) -> None:
        true_sequences = [["O", "B-CUE", "B-SCOPE", "I-SCOPE"]]
        predicted_sequences = [["O", "B-CUE", "B-SCOPE", "I-SCOPE"]]
        metrics = compute_classification_metrics(true_sequences, predicted_sequences)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_macro"], 1.0)

    def test_span_overlap_handles_partial_match(self) -> None:
        true_sequences = [["O", "B-CUE", "B-SCOPE", "I-SCOPE", "O"]]
        predicted_sequences = [["O", "B-CUE", "B-SCOPE", "O", "O"]]
        span_metrics = compute_span_metrics(true_sequences, predicted_sequences)
        self.assertLess(span_metrics["strict"]["f1"], 1.0)
        self.assertGreater(span_metrics["overlap"]["f1"], 0.0)

    def test_invalid_bio_transition_detection(self) -> None:
        self.assertEqual(invalid_bio_transitions(["O", "I-SCOPE", "I-SCOPE"]), 1)


if __name__ == "__main__":
    unittest.main()
