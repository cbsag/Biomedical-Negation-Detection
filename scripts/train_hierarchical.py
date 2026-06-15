"""Train the hierarchical sentence classifier plus sequence tagger pipeline."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from negation_scope.data import load_bioscope_records, stratified_split
from negation_scope.hierarchical import HierarchicalNegationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="artifacts/hierarchical")
    parser.add_argument("--dataset", choices=("all", "full", "abstract"), default="all")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--max-iter", type=int, default=1000)
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

    pipeline = HierarchicalNegationPipeline(
        min_df=args.min_df,
        max_features=args.max_features,
        max_iter=args.max_iter,
    ).fit(splits["train"])
    report = pipeline.evaluate(splits["test"])

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pipeline.pkl").open("wb") as handle:
        pickle.dump(pipeline, handle)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

