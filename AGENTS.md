# Surgical Workflow Analysis agent instructions

## Scope and purpose
These instructions apply to the whole repository. Read additional `AGENTS.md` files in the directories you work in; nested instructions add local requirements and the closest applicable file governs directory-specific work.

This is a research and coursework project for offline multi-task surgical workflow analysis on laparoscopic cholecystectomy videos from Cholec80. It predicts one of seven surgical phases and the presence of seven tools for each sampled frame. Do not describe this prototype as validated for real-time clinical decisions or patient care.

## Repository map
- `configs/default.yaml`: experiment settings, video-level split, model parameters, and training configuration.
- `src/dataset.py`: annotation parsing, 1-fps frame loading, augmentation helpers, cached features, and sequence collation.
- `src/models/`: frozen ResNet-50 feature extraction, baseline/LSTM/MS-TCN temporal models, multi-task heads, and correlation loss.
- `src/train.py`: training, validation, checkpointing, and early stopping.
- `src/evaluate.py`: phase and tool metrics, segment edit score, and report plots.
- `src/utils.py`: shared utilities.
- `notebooks/project_report.ipynb`: Colab experiment workflow and project report.
- `tests/`: automated unit tests.

## Local setup and validation
From the repository root:
```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest tests/ -q
```
The full training and evaluation workflow is run in `notebooks/project_report.ipynb` with a Colab GPU and Cholec80 data or cached features. Do not invent a CLI entry point or imply the model can be trained without its dataset and weights.

## Research integrity
- Make only task-relevant edits; inspect the implementation, configuration, tests, and relevant notebook cells before changing an experiment.
- Preserve video-level split isolation. Keep training-derived weights and tool-phase co-occurrence priors out of validation and test data.
- Preserve the distinction between reported experiments on the configured 17-video subset and hypothetical full-Cholec80 experiments. Never relabel subset results as full-dataset results.
- Do not fabricate metrics, ablation outcomes, checkpoints, or runs. State clearly when a model or notebook was not executed.
- Keep Cholec80 videos, frames, annotations, cached features, checkpoints, and other large or sensitive clinical media out of commits. They belong in ignored local or external data locations.
- Maintain reproducibility: record changes to seeds, splits, sampling rates, architecture, losses, preprocessing, or metric definitions when reporting results.
- Preserve relevant tests, fix root causes, and report exactly which checks passed, failed, or could not be run.
