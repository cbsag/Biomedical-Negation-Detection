from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from negation_scope.data import build_word_labels, parse_cue_spans, parse_scope_span, stratified_split


class DataPipelineTests(unittest.TestCase):
    def test_parse_spans(self) -> None:
        self.assertEqual(parse_cue_spans("[(8, 11), (17, 20)]"), [(8, 11), (17, 20)])
        self.assertEqual(parse_scope_span("[8, 19]"), (8, 19))
        self.assertEqual(parse_cue_spans(float("nan")), [])
        self.assertIsNone(parse_scope_span(None))

    def test_build_word_labels_excludes_cue_from_scope(self) -> None:
        sentence = "This is not good at all."
        tokens, _, labels = build_word_labels(
            sentence=sentence,
            cue_spans=[(8, 11)],
            scope_span=(8, 19),
        )
        self.assertEqual(tokens, ["This", "is", "not", "good", "at", "all."])
        self.assertEqual(labels, ["O", "O", "B-CUE", "B-SCOPE", "I-SCOPE", "O"])

    def test_stratified_split_preserves_total_size(self) -> None:
        records = [
            {"target": 0, "tokens": ["a"], "word_labels": ["O"]},
            {"target": 0, "tokens": ["b"], "word_labels": ["O"]},
            {"target": 0, "tokens": ["c"], "word_labels": ["O"]},
            {"target": 1, "tokens": ["d"], "word_labels": ["B-CUE"]},
            {"target": 1, "tokens": ["e"], "word_labels": ["B-CUE"]},
            {"target": 1, "tokens": ["f"], "word_labels": ["B-CUE"]},
        ]
        splits = stratified_split(records, train_ratio=0.5, dev_ratio=0.25, seed=7)
        total = len(splits["train"]) + len(splits["dev"]) + len(splits["test"])
        self.assertEqual(total, len(records))


if __name__ == "__main__":
    unittest.main()

