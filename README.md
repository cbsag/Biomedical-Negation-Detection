
# Biomedical Negation Detection

This repository contains a recovered and cleaned-up version of my BIOSCOPE negation project. The original lab-hosted GitLab node is currently unavailable, so I rebuilt the repo from a saved Colab export, local dataset snapshots, the project poster, and the experiment path I had already documented in my CV. I did not want to leave this work as a broken notebook with Drive paths and one-off cells, so this version turns it into something I can actually rerun, explain, and extend.

The task is token-level negation cue and scope detection in biomedical text using BIO sequence labels:

- `O`
- `B-CUE`, `I-CUE`
- `B-SCOPE`, `I-SCOPE`

## Why This Project Matters

Biomedical negation is a small labeling problem on paper, but it is annoying in practice:

- the dataset is heavily imbalanced
- cue and scope spans partially overlap
- token-level accuracy can look good even when minority-label behavior is weak

The recovered BIOSCOPE snapshots in this repo contain 14,462 sentences in total, with 12,368 non-negated and 2,094 negated examples. That imbalance is why I kept both the weighted transformer baseline and the sentence-first hierarchical pipeline, and why the PPO-style extension focuses on minority-label and span-quality rewards instead of raw accuracy alone.

## What Is In This Repo

- recovered artifacts from the original work in `notebooks/` and `docs/`
- a reusable preprocessing pipeline for the BIOSCOPE spreadsheets in `data/raw/`
- transformer training code for BERT, BioBERT, and ClinicalBERT baselines
- a CRF baseline over word-level features
- a hierarchical pipeline that first predicts whether a sentence is negated, then tags cue/scope spans
- a proposed PPO-style refinement stage built on top of the supervised token classifier
- experiment preset files in `configs/` for repeatable runs

## Recovery Boundary

I want to be explicit about what this repository is and is not.

- `notebooks/recovered_colab_export.ipynb` is the saved version of the original Colab workflow
- `src/` is a reconstruction of that workflow into a proper package
- the transformer baseline follows the original project direction closely
- the CRF, hierarchical, and PPO pieces are rebuilt so the repo matches the work I described and remains runnable
- the PPO module is a working reconstruction of the idea I explored, not a byte-for-byte recovery of the lost lab GitLab code

That is the most accurate way I can present the project without pretending I still have the original server-side history.

## Repository Layout

```text
.
├── configs/                  # Reusable experiment presets
├── data/raw/                 # Local BIOSCOPE spreadsheet snapshots
├── docs/project_poster.pdf   # Original poster
├── notebooks/                # Saved Colab export
├── scripts/                  # CLI entry points and config runner
├── src/negation_scope/       # Reconstructed package
└── tests/                    # Smoke tests for parsing and metrics
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you do not want an editable install, `pip install -r requirements.txt` is enough.

## Data

The local snapshots are already placed in:

- `data/raw/bioscope_full.xlsx`
- `data/raw/bioscope_abstract.xlsx`

The preprocessing code reads character-level cue and scope spans from those spreadsheets, converts them to word-level BIO tags, and excludes cue tokens from the scope label when the spans overlap.

To export train/dev/test JSONL splits:

```bash
python scripts/prepare_data.py --dataset all
```

## Running Experiments

### Preset configs

I added JSON presets for the main experiment paths so the repo is easier to scan and rerun.

```bash
python scripts/run_experiment.py configs/prepare_all.json
python scripts/run_experiment.py configs/train_transformer_biobert_weighted.json
python scripts/run_experiment.py configs/train_crf_baseline.json
python scripts/run_experiment.py configs/train_hierarchical_pipeline.json
python scripts/run_experiment.py configs/train_ppo_biobert_refinement.json
```

### Direct CLI examples

Supervised transformer baseline:

```bash
python scripts/train_transformer.py \
  --dataset all \
  --model-name dmis-lab/biobert-base-cased-v1.1 \
  --use-class-weights \
  --output-dir artifacts/transformer/biobert_weighted
```

You can swap the model name with:

- `bert-base-uncased`
- `dmis-lab/biobert-base-cased-v1.1`
- `emilyalsentzer/Bio_ClinicalBERT`

CRF baseline:

```bash
python scripts/train_crf.py \
  --dataset all \
  --output-dir artifacts/crf
```

Hierarchical pipeline:

```bash
python scripts/train_hierarchical.py \
  --dataset all \
  --output-dir artifacts/hierarchical
```

PPO-style refinement:

```bash
python scripts/train_ppo.py \
  --dataset all \
  --model-name dmis-lab/biobert-base-cased-v1.1 \
  --warmstart-checkpoint artifacts/transformer/biobert_weighted \
  --output-dir artifacts/ppo/biobert_refined
```

## What Changed From The Notebook

- removed hard-coded Google Drive paths
- moved data, poster, and notebook into a clearer layout
- split preprocessing, metrics, training, and evaluation into modules
- added a real CRF baseline instead of only CRF-style reporting
- added a hierarchical pipeline to separate sentence-level imbalance handling from token tagging
- rebuilt the PPO idea as a standalone experiment instead of leaving it as a missing claim
- added repeatable experiment presets for cleaner reruns

## What This Repo Shows

If someone is reviewing this repo as part of an application, the main things I would want them to take away are:

- I worked on sequence labeling for biomedical NLP, not just generic text classification
- I dealt with imbalance, span alignment, and evaluation beyond plain accuracy
- I can turn a rough research notebook into a cleaner, reusable codebase
- I am being explicit about what was recovered and what was reconstructed

## Short Version I Would Use In An Interview

> I originally built the project in Colab, then lost access to the lab-hosted GitLab node. I recovered the saved notebook and local artifacts, rebuilt the repo into a proper package, and reconstructed the missing extensions so the project could be rerun and discussed honestly. The transformer baseline comes from the original workflow, while the CRF, hierarchical, and PPO pieces are presented as clean rebuilds of the direction I had explored.

## References

- BIOSCOPE: Vincze et al., 2008
- Model families used here: BERT, BioBERT, and ClinicalBERT

