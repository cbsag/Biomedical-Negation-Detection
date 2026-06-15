"""PPO-style refinement for token classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .constants import LABELS
from .data import encode_record_for_transformer
from .metrics import (
    compute_classification_metrics,
    compute_span_metrics,
    decode_label_id_sequences,
    invalid_bio_transitions,
    minority_recall,
)
from .transformer import EncodedTokenDataset


@dataclass
class PPOTrainingConfig:
    """Configuration for PPO-style sequence refinement."""

    model_name: str
    output_dir: str | Path
    warmstart_checkpoint: str | None = None
    max_length: int = 128
    batch_size: int = 4
    epochs: int = 3
    ppo_steps_per_batch: int = 3
    learning_rate: float = 1e-5
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 13


def _require_ppo_dependencies() -> tuple[object, object, object, object]:
    """Import torch and transformers lazily."""
    try:
        import torch
        from torch import nn
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "torch and transformers are required for PPO refinement."
        ) from error
    return torch, nn, AutoModelForTokenClassification, AutoTokenizer


def build_policy_dataset(records: list[dict[str, object]], tokenizer: object, max_length: int) -> EncodedTokenDataset:
    """Encode records for PPO training."""
    encodings = [encode_record_for_transformer(record, tokenizer, max_length=max_length) for record in records]
    return EncodedTokenDataset(encodings)


def reward_from_predictions(true_labels: list[str], predicted_labels: list[str]) -> float:
    """Reward cue/scope quality more than the dominant O label."""
    token_metrics = compute_classification_metrics([true_labels], [predicted_labels])
    span_metrics = compute_span_metrics([true_labels], [predicted_labels])
    minority_score = minority_recall(token_metrics)
    invalid_penalty = invalid_bio_transitions(predicted_labels) / max(len(predicted_labels), 1)

    reward = (
        0.35 * token_metrics["f1_macro"]
        + 0.35 * span_metrics["overlap"]["f1"]
        + 0.25 * minority_score
        - 0.10 * invalid_penalty
    )
    return max(-1.0, min(1.0, reward))


def train_ppo_refinement(
    train_records: list[dict[str, object]],
    dev_records: list[dict[str, object]],
    config: PPOTrainingConfig,
) -> dict[str, object]:
    """Run PPO-style updates on top of a token classifier."""
    torch, nn, AutoModelForTokenClassification, AutoTokenizer = _require_ppo_dependencies()
    from torch.distributions import Categorical
    from torch.utils.data import DataLoader

    class PolicyWithValue(nn.Module):
        def __init__(self, source_name: str) -> None:
            super().__init__()
            self.model = AutoModelForTokenClassification.from_pretrained(
                source_name,
                num_labels=len(LABELS),
            )
            self.model.config.output_hidden_states = True
            hidden_size = self.model.config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]
            values = self.value_head(hidden_states).squeeze(-1)
            return outputs.logits, values

        def save(self, output_dir: str | Path, tokenizer: object) -> None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            torch.save(self.value_head.state_dict(), output_path / "value_head.pt")

    def collate_reward_batch(logits_tensor, true_label_tensor):
        sampled_actions = Categorical(logits=logits_tensor).sample()
        rewards: list[float] = []
        for sampled, true_ids in zip(sampled_actions.tolist(), true_label_tensor.tolist()):
            valid_true = []
            valid_pred = []
            for predicted_id, true_id in zip(sampled, true_ids):
                if true_id == -100:
                    continue
                valid_true.append(int(true_id))
                valid_pred.append(int(predicted_id))
            decoded_true, decoded_pred = decode_label_id_sequences([valid_true], [valid_pred])
            rewards.append(reward_from_predictions(decoded_true[0], decoded_pred[0]))
        return sampled_actions, torch.tensor(rewards, dtype=logits_tensor.dtype, device=logits_tensor.device)

    tokenizer_source = config.warmstart_checkpoint or config.model_name
    model_source = config.warmstart_checkpoint or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    train_dataset = build_policy_dataset(train_records, tokenizer, max_length=config.max_length)
    dev_dataset = build_policy_dataset(dev_records, tokenizer, max_length=config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyWithValue(model_source).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        policy.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            mask = batch["labels"].ne(-100)

            with torch.no_grad():
                old_logits, _ = policy(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("token_type_ids"),
                )
                sampled_actions, sequence_rewards = collate_reward_batch(old_logits, batch["labels"])
                old_distribution = Categorical(logits=old_logits)
                old_log_probs = old_distribution.log_prob(sampled_actions)

            reward_targets = sequence_rewards.unsqueeze(1).expand_as(batch["labels"]).to(device)
            reward_targets = torch.where(mask, reward_targets, torch.zeros_like(reward_targets))

            for _ in range(config.ppo_steps_per_batch):
                logits, values = policy(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("token_type_ids"),
                )
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(sampled_actions)
                entropy = distribution.entropy()

                advantages = (reward_targets - values).detach()
                ratios = (new_log_probs - old_log_probs).exp()
                clipped_ratios = ratios.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)

                surrogate_one = ratios * advantages
                surrogate_two = clipped_ratios * advantages
                policy_loss = -torch.min(surrogate_one, surrogate_two)
                policy_loss = policy_loss.masked_select(mask).mean()

                value_loss = ((values - reward_targets) ** 2).masked_select(mask).mean()
                entropy_bonus = entropy.masked_select(mask).mean()

                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy_bonus
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

    policy.eval()
    true_ids: list[list[int]] = []
    predicted_ids: list[list[int]] = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits, _ = policy(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("token_type_ids"),
            )
            predictions = logits.argmax(dim=-1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            predicted_ids.extend(predictions)
            true_ids.extend(labels)

    decoded_true, decoded_pred = decode_label_id_sequences(true_ids, predicted_ids)
    report = {
        "token": compute_classification_metrics(decoded_true, decoded_pred),
        "span": compute_span_metrics(decoded_true, decoded_pred),
    }

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    policy.save(output_path, tokenizer)
    (output_path / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
