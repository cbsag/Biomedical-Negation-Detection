"""Run PPO-style refinement on top of a supervised token classifier."""

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
from negation_scope.ppo import PPOTrainingConfig, train_ppo_refinement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="artifacts/ppo/run")
    parser.add_argument("--dataset", choices=("all", "full", "abstract"), default="all")
    parser.add_argument("--model-name", required=True, help="Base model name or local checkpoint.")
    parser.add_argument("--warmstart-checkpoint", default=None, help="Optional supervised checkpoint directory.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ppo-steps-per-batch", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
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

    config = PPOTrainingConfig(
        model_name=args.model_name,
        output_dir=PROJECT_ROOT / args.output_dir,
        warmstart_checkpoint=args.warmstart_checkpoint,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        ppo_steps_per_batch=args.ppo_steps_per_batch,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        seed=args.seed,
    )
    report = train_ppo_refinement(splits["train"], splits["test"], config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

