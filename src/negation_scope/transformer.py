"""Transformer-based token classification training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .constants import ID_TO_LABEL, LABELS, LABEL_TO_ID
from .data import encode_record_for_transformer
from .metrics import compute_classification_metrics, compute_span_metrics, decode_label_id_sequences


class EncodedTokenDataset:
    """Thin list-backed dataset compatible with Hugging Face Trainer."""

    def __init__(self, encodings: list[dict[str, object]]) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        record = self.encodings[index]
        return {
            key: torch.tensor(value)
            for key, value in record.items()
            if key in {"input_ids", "attention_mask", "token_type_ids", "labels"}
        }


def _require_transformer_dependencies() -> tuple[object, object, object]:
    """Import torch and transformers lazily."""
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as error:
        raise ImportError(
            "torch, transformers, and accelerate are required for transformer training."
        ) from error
    return torch, AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments


def compute_label_weights(records: list[dict[str, object]]) -> list[float]:
    """Compute inverse-frequency weights for token labels."""
    label_counter = Counter()
    for record in records:
        label_counter.update(record["word_labels"])

    total = sum(label_counter.values())
    weights: list[float] = []
    for label in LABELS:
        count = label_counter[label]
        weights.append(total / (len(LABELS) * count) if count else 0.0)
    return weights


def build_encoded_dataset(
    records: list[dict[str, object]],
    tokenizer: object,
    max_length: int,
) -> EncodedTokenDataset:
    """Tokenize and align labels for a list of records."""
    encodings = [encode_record_for_transformer(record, tokenizer, max_length=max_length) for record in records]
    return EncodedTokenDataset(encodings)


def trainer_metrics_from_predictions(
    true_label_ids: list[list[int]],
    predicted_label_ids: list[list[int]],
) -> dict[str, float]:
    """Convert trainer outputs into scalar metrics for checkpoint selection."""
    decoded_true, decoded_pred = decode_label_id_sequences(true_label_ids, predicted_label_ids)
    token_metrics = compute_classification_metrics(decoded_true, decoded_pred)
    span_metrics = compute_span_metrics(decoded_true, decoded_pred)
    return {
        "token_accuracy": token_metrics["accuracy"],
        "token_f1_macro": token_metrics["f1_macro"],
        "token_f1_weighted": token_metrics["f1_weighted"],
        "span_strict_f1": span_metrics["strict"]["f1"],
        "span_overlap_f1": span_metrics["overlap"]["f1"],
    }


def train_transformer_model(
    train_records: list[dict[str, object]],
    dev_records: list[dict[str, object]],
    output_dir: str | Path,
    model_name: str,
    max_length: int = 128,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    seed: int = 13,
    use_class_weights: bool = False,
) -> dict[str, object]:
    """Fine-tune a transformer model for token classification."""
    torch, AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments = (
        _require_transformer_dependencies()
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_dataset = build_encoded_dataset(train_records, tokenizer, max_length=max_length)
    dev_dataset = build_encoded_dataset(dev_records, tokenizer, max_length=max_length)
    class_weights = compute_label_weights(train_records) if use_class_weights else None

    class WeightedTokenTrainer(Trainer):
        def __init__(self, *args, class_weights: list[float] | None = None, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            if not self.class_weights:
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            weights = torch.tensor(self.class_weights, device=outputs.logits.device, dtype=outputs.logits.dtype)
            loss_function = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
            loss = loss_function(
                outputs.logits.view(-1, model.config.num_labels),
                inputs["labels"].view(-1),
            )
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_token_f1_macro",
        greater_is_better=True,
        report_to="none",
        seed=seed,
    )

    def compute_metrics(eval_prediction: object) -> dict[str, float]:
        logits = eval_prediction.predictions
        label_ids = eval_prediction.label_ids
        predicted_label_ids = logits.argmax(axis=-1)
        return trainer_metrics_from_predictions(label_ids.tolist(), predicted_label_ids.tolist())

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    prediction_output = trainer.predict(dev_dataset)
    predicted_label_ids = prediction_output.predictions.argmax(axis=-1).tolist()
    true_label_ids = prediction_output.label_ids.tolist()
    decoded_true, decoded_pred = decode_label_id_sequences(true_label_ids, predicted_label_ids)
    report = {
        "token": compute_classification_metrics(decoded_true, decoded_pred),
        "span": compute_span_metrics(decoded_true, decoded_pred),
    }

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(target_dir))
    tokenizer.save_pretrained(str(target_dir))
    (target_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
