"""Train and evaluate the CRF baseline."""

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

from negation_scope.crf_baseline import evaluate_crf, save_crf_model, train_crf
from negation_scope.data import load_bioscope_records, stratified_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="artifacts/crf")
    parser.add_argument("--dataset", choices=("all", "full", "abstract"), default="all")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick experiments.")
    parser.add_argument("--c1", type=float, default=0.1)
    parser.add_argument("--c2", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=100)
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

    model = train_crf(
        splits["train"],
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iterations,
    )
    report = evaluate_crf(model, splits["test"])

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_crf_model(model, output_dir / "model.pkl")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

