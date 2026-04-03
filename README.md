# Multi-Task Surgical Workflow Analysis

Joint Phase Recognition and Tool Tracking in Laparoscopic Surgery

**Author:** Omar Morsi (40236376)
**Course:** COMP 432 — Machine Learning

## Overview

A multi-task deep learning system that simultaneously recognizes surgical phases and detects instrument presence in laparoscopic cholecystectomy videos. Uses the [Cholec80 dataset](http://camma.u-strasbg.fr/datasets) (80 videos, ~15,000 frames at 1 fps).

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

- **Feature extraction:** Pretrained ResNet-50 (ImageNet), features cached to disk
- **Temporal modeling:** MS-TCN with dilated convolutions for long-range dependencies
- **Multi-task learning:** Joint phase (7-class softmax) and tool (7-class sigmoid) prediction
- **Class imbalance:** Weighted cross-entropy loss
- **Novel contribution:** Correlation loss enforcing tool-phase co-occurrence priors

## Evaluation Metrics

| Task | Metric |
|---|---|
| Phase recognition | Frame-wise F1 (macro), per-phase F1, accuracy |
| Tool detection | Mean Average Precision (mAP), per-tool AP |
| Temporal consistency | Segment-level edit score |

## References

1. Twinanda et al. (2016). EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. *IEEE TMI*.
2. Jin et al. (2018). SV-RCNet: Workflow Recognition from Surgical Videos Using Recurrent Convolutional Network. *IEEE TMI*.
3. Czempiel et al. (2020). TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks. *MICCAI*.
4. Farha & Gall (2019). MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation. *CVPR*.
