"""Project-wide constants."""

from __future__ import annotations

LABELS = ("O", "B-CUE", "I-CUE", "B-SCOPE", "I-SCOPE")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
MINORITY_LABELS = tuple(label for label in LABELS if label != "O")

DATASET_FILES = {
    "all": ("bioscope_full.xlsx", "bioscope_abstract.xlsx"),
    "full": ("bioscope_full.xlsx",),
    "abstract": ("bioscope_abstract.xlsx",),
}

DEFAULT_MODELS = {
    "bert": "bert-base-uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}

