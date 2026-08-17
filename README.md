# Biomedical Negation Detection

Biomedical negation detection using transformer models, CRFs, hierarchical classification, and PPO-based sequence-labeling refinement on the BioScope corpus.

The task is to identify **negation cues** and their corresponding **scope** in biomedical text using token-level BIO labels:

* `O`
* `B-CUE`, `I-CUE`
* `B-SCOPE`, `I-SCOPE`

This repository is a cleaned and reproducible version of work I developed during my graduate research. The original project was developed across Colab and a lab-hosted GitLab environment that is no longer accessible, so I reconstructed the public repository from saved notebooks, dataset snapshots, experiment artifacts, and the documented project workflow.

## Project Overview

Negation detection is important in biomedical NLP because a small linguistic change can completely reverse the meaning of a clinical statement.

For example:

> **No evidence of pneumonia was observed.**

A biomedical system needs to recognize both the negation cue (`No`) and the text affected by it (`evidence of pneumonia`).

The BioScope data used in this project contains **14,462 sentences**, including:

* 12,368 non-negated sentences
* 2,094 negated sentences

The resulting class imbalance makes token-level accuracy alone misleading, especially for relatively rare cue and scope labels.

This project explores several modeling strategies for improving minority-label and span-level behavior.

## Approaches

### 1. Transformer Sequence Labeling

Fine-tuned transformer models for token-level BIO tagging:

* BERT
* BioBERT
* ClinicalBERT

The models jointly predict cue and scope labels and use class weighting to reduce the impact of label imbalance.

### 2. CRF Baseline

Implemented a Conditional Random Field baseline using word-level contextual and linguistic features.

The CRF provides a structured sequence-modeling comparison to transformer-based approaches and explicitly models dependencies between neighboring BIO labels.

### 3. Hierarchical Negation Pipeline

Developed a two-stage pipeline that separates:

1. sentence-level negation detection
2. token-level cue and scope extraction

The goal is to avoid forcing the sequence labeler to learn primarily from the large number of sentences containing no negation.

### 4. PPO-Based Sequence-Labeling Refinement

Extended the supervised sequence-labeling pipeline with a PPO-based policy trained using rewards derived from human-annotated negation spans.

The reward formulation emphasizes sequence-level behavior that standard token-level cross-entropy does not directly optimize, including:

* cue detection
* scope detection
* minority-label recovery
* span-level agreement

In the experiments, PPO-based refinement improved minority-label detection relative to the supervised baseline, motivating the use of sequence-level feedback for highly imbalanced biomedical labeling tasks.

## Repository Structure

```text
.
├── configs/                  # Reusable experiment configurations
├── data/raw/                 # BioScope dataset snapshots
├── docs/
│   └── project_poster.pdf    # Project poster
├── notebooks/                # Recovered original Colab workflow
├── scripts/                  # Experiment entry points
├── src/negation_scope/       # Reusable project modules
└── tests/                    # Parsing and metric smoke tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Alternatively:

```bash
pip install -r requirements.txt
```

## Data Preparation

The BioScope snapshots are stored under:

```text
data/raw/bioscope_full.xlsx
data/raw/bioscope_abstract.xlsx
```

The preprocessing pipeline converts character-level cue and scope annotations into word-level BIO labels and handles overlapping cue/scope spans.

Prepare the dataset with:

```bash
python scripts/prepare_data.py --dataset all
```

## Running Experiments

Experiment presets are provided under `configs/`.

### Transformer baseline

```bash
python scripts/run_experiment.py \
  configs/train_transformer_biobert_weighted.json
```

Or directly:

```bash
python scripts/train_transformer.py \
  --dataset all \
  --model-name dmis-lab/biobert-base-cased-v1.1 \
  --use-class-weights \
  --output-dir artifacts/transformer/biobert_weighted
```

### CRF baseline

```bash
python scripts/train_crf.py \
  --dataset all \
  --output-dir artifacts/crf
```

### Hierarchical pipeline

```bash
python scripts/train_hierarchical.py \
  --dataset all \
  --output-dir artifacts/hierarchical
```

### PPO refinement

```bash
python scripts/train_ppo.py \
  --dataset all \
  --model-name dmis-lab/biobert-base-cased-v1.1 \
  --warmstart-checkpoint artifacts/transformer/biobert_weighted \
  --output-dir artifacts/ppo/biobert_refined
```

## Public Repository Reconstruction

The original experiments were developed using a combination of Colab notebooks and lab infrastructure.

When the original lab-hosted repository became unavailable, I recovered the project from:

* the saved Colab workflow
* local BioScope dataset snapshots
* project documentation and poster material
* experiment configurations and recorded results

The public `src/` structure reorganizes that work into a cleaner and more reproducible codebase.

Some components were rewritten from the original experimental workflow rather than recovered byte-for-byte. The goal of this repository is to preserve the underlying methods and experimental design accurately while making the project easier to inspect and rerun.

## What This Project Demonstrates

This project focuses on problems that appear frequently in applied biomedical NLP:

* highly imbalanced sequence-labeling datasets
* cue and scope span extraction
* structured prediction
* transformer fine-tuning
* hierarchical modeling
* sequence-level reinforcement learning
* evaluation beyond raw token accuracy

It also explores how feedback-based optimization can complement conventional supervised learning when the labels that matter most are relatively rare.

## References

**BioScope Corpus**
Vincze et al. (2008), *The BioScope Corpus: Biomedical Texts Annotated for Uncertainty, Negation and Their Scopes.*

Model families used in the experiments include BERT, BioBERT, and ClinicalBERT.
