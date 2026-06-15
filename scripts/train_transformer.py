"""Train a transformer baseline for cue and scope detection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from negation_scope.data import load_bioscope_records, stratified_split
from negation_scope.transformer import train_transformer_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="artifacts/transformer/run")
    parser.add_argument("--dataset", choices=("all", "full", "abstract"), default="all")
    parser.add_argument("--model-name", required=True, help="Hugging Face model name or local checkpoint.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--use-class-weights", action="store_true")
    return parser.parse_args()


def maybe_trim_records(records: list[dict[str, object]], max_samples: int, seed: int) -> list[dict[str, object]]:
    if max_samples <= 0 or max_samples >= len(records):
        return records
    rng = random.Random(seed)
    trimmed = list(records)
    rng.shuffle(trimmed)
    return trimmed[:max_samples]


def main() -> None:
    args = parse_args()
    records = load_bioscope_records(PROJECT_ROOT / args.data_dir, dataset=args.dataset)
    records = maybe_trim_records(records, args.max_samples, args.seed)
    splits = stratified_split(records, seed=args.seed)

    report = train_transformer_model(
        train_records=splits["train"],
        dev_records=splits["test"],
        output_dir=PROJECT_ROOT / args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_class_weights=args.use_class_weights,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

