"""Export deterministic train/dev/test splits from the raw BIOSCOPE spreadsheets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from negation_scope.data import export_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing BIOSCOPE spreadsheets.")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for JSONL splits.")
    parser.add_argument(
        "--dataset",
        choices=("all", "full", "abstract"),
        default="all",
        help="Which BIOSCOPE subset to export.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_splits(
        PROJECT_ROOT / args.data_dir,
        PROJECT_ROOT / args.output_dir,
        dataset=args.dataset,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()

