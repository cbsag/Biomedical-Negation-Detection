
# Biomedical Negation Detection

This repository is a cleaned-up recovery of my BIOSCOPE negation project. The original lab GitLab node is down right now, so I rebuilt the codebase from the saved Colab notebook export, the local dataset snapshots, the poster, and the project description on my CV. I did not want to leave the repo as a single broken notebook with hard-coded Drive paths, so this version turns it into something that is easier to read, rerun, and discuss in interviews.

The core task is token-level negation cue and scope detection in biomedical text using BIO labels:

- `O`
- `B-CUE`, `I-CUE`
- `B-SCOPE`, `I-SCOPE`

The project now contains:

- recovered artifacts from the original work in `notebooks/` and `docs/`
- a reusable preprocessing pipeline for the BIOSCOPE spreadsheets in `data/raw/`
- transformer training code for BERT, BioBERT, and ClinicalBERT baselines
- a CRF baseline over word-level features
- a hierarchical pipeline that first predicts whether a sentence is negated, then tags cue/scope spans
- a proposed PPO-style refinement stage built on top of the supervised token classifier

## Recovery Note

I want to be explicit about what this repo is.

- The notebook in `notebooks/recovered_colab_export.ipynb` is the saved version of the original Colab workflow.
- The package code in `src/` is reconstructed from that notebook and from the project direction I described in my CV.
- The PPO module is a working reconstruction of the idea I explored. It is not claimed to be a byte-for-byte recovery of the lost lab GitLab code.

That felt like the most honest way to present the project while still making the repository useful.

## Repository Layout

```text
.
├── data/raw/                 # Local BIOSCOPE spreadsheet snapshots
├── docs/project_poster.pdf   # Original poster
├── notebooks/                # Saved Colab export
├── scripts/                  # CLI entry points for experiments
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

The preprocessing code reads the character-level cue and scope spans from those spreadsheets, converts them to word-level BIO tags, and keeps cue tokens out of the scope label when the spans overlap.

To export train/dev/test JSONL splits:

```bash
python scripts/prepare_data.py --dataset all
```

## Running Experiments

### 1. Supervised transformer baseline

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

### 2. CRF baseline

```bash
python scripts/train_crf.py \
  --dataset all \
  --output-dir artifacts/crf
```

### 3. Hierarchical pipeline

The hierarchical version uses a sentence-level negation classifier first, then runs span tagging only on sentences predicted as negated. I kept this because the data is heavily imbalanced and this was one of the cleaner ways to separate sentence-level filtering from token labeling.

```bash
python scripts/train_hierarchical.py \
  --dataset all \
  --output-dir artifacts/hierarchical
```

### 4. Proposed PPO refinement

The motivation here is straightforward: the plain supervised model gets pulled toward the dominant `O` label, so the PPO-style refinement uses a reward that cares more about minority labels, span overlap quality, and invalid BIO transitions.

```bash
python scripts/train_ppo.py \
  --dataset all \
  --model-name dmis-lab/biobert-base-cased-v1.1 \
  --warmstart-checkpoint artifacts/transformer/biobert_weighted \
  --output-dir artifacts/ppo/biobert_refined
```

## What I Changed From The Notebook

- removed hard-coded Google Drive paths
- moved data, poster, and notebook into a clearer layout
- split preprocessing, metrics, training, and evaluation into real modules
- added a real CRF baseline instead of only CRF-style reporting
- added a hierarchical pipeline that directly addresses sentence-level imbalance
- rebuilt the PPO idea as a standalone experiment instead of leaving it as a missing claim

## Interview-Friendly Summary

If I need to explain this project quickly, the honest version is:

> I originally built the project in Colab, lost access to the lab-hosted GitLab node, and then reconstructed the repo locally from the saved notebook, data snapshots, and the experiment direction I had documented in my CV. The baseline transformer experiments are recovered from the original workflow, and the CRF, hierarchical, and PPO pieces are rebuilt into a cleaner codebase that I can now rerun and extend.

## References

- BIOSCOPE: Vincze et al., 2008
- Model families used here: BERT, BioBERT, and ClinicalBERT

