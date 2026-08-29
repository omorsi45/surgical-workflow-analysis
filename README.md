# Multi-Task Surgical Workflow Analysis

Joint Phase Recognition and Tool Tracking in Laparoscopic Surgery

**Author:** Omar Morsi (40236376)
**Course:** COMP 432 — Machine Learning

## Overview

A multi-task deep learning system that simultaneously recognizes surgical phases and detects instrument presence in laparoscopic cholecystectomy videos. Uses the [Cholec80 dataset](http://camma.u-strasbg.fr/datasets), sampled at 1 fps.

Experiments here run on a **compute-constrained 17-video subset** (10 train / 2 val / 5 test, ~27,000 training frames) rather than the canonical 40/40 split, because ResNet-50 feature extraction over all 80 videos exceeds a free-tier Colab session. The pipeline scales to the full dataset without code changes — see §3.2 of the report notebook.

The system compares four model variants:
- **Baseline** — frame-wise classification (no temporal context)
- **LSTM** — bidirectional LSTM for sequence modeling
- **MS-TCN** — multi-stage temporal convolutional network
- **MS-TCN + Correlation Loss** — MS-TCN with a novel loss that penalizes impossible tool-phase combinations

## Project Structure

```
├── src/
│   ├── dataset.py            # Cholec80 data loading and preprocessing
│   ├── train.py              # Training loop with early stopping
│   ├── evaluate.py           # Metrics and visualization
│   ├── utils.py              # Reproducibility and config utilities
│   └── models/
│       ├── backbone.py       # ResNet-50 feature extractor
│       ├── temporal.py       # Baseline, LSTM, MS-TCN architectures
│       └── multitask.py      # Multi-task heads and correlation loss
├── notebooks/
│   └── project_report.ipynb  # Main report notebook (run on Colab)
├── configs/
│   └── default.yaml          # Hyperparameters
├── tests/                    # pytest suite (run: python -m pytest tests/ -q)
└── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

Request access to Cholec80 at [camma.u-strasbg.fr/datasets](http://camma.u-strasbg.fr/datasets) and extract into `data/cholec80/`.

### 3. Run

Open `notebooks/project_report.ipynb` in Google Colab (GPU runtime) and run all cells. The notebook handles feature extraction, training, evaluation, and visualization.

## Methods

- **Feature extraction:** Pretrained ResNet-50 (ImageNet), frozen, features extracted once and cached to disk
- **Temporal modeling:** MS-TCN with dilated (acausal) convolutions for long-range dependencies
- **Multi-task learning:** Joint phase (7-class softmax) and tool (7-class sigmoid) prediction
- **Class imbalance:** Weighted cross-entropy loss
- **Novel contribution:** Correlation loss enforcing tool-phase co-occurrence priors

Note: the augmentation pipeline in `src/dataset.py` is **not** used in the reported experiments — with a frozen backbone and one-shot cached features there is nothing for per-epoch augmentation to vary. It is retained for a future fine-tuning experiment.

## Evaluation Metrics

| Task | Metric |
|---|---|
| Phase recognition | Frame-wise F1 (macro), per-phase F1, accuracy |
| Tool detection | Mean Average Precision (mAP), per-tool AP |
| Temporal consistency | Segment-level edit score |

mAP excludes tools with no positive examples in the split, since average precision is undefined for an absent class.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

64 tests cover the metrics, losses, collate/masking contract, annotation parsing, checkpoint selection, and the shape contract shared by all three temporal models.

## References

1. Twinanda et al. (2016). EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. *IEEE TMI*.
2. Jin et al. (2018). SV-RCNet: Workflow Recognition from Surgical Videos Using Recurrent Convolutional Network. *IEEE TMI*.
3. Czempiel et al. (2020). TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks. *MICCAI*.
4. Farha & Gall (2019). MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation. *CVPR*.
