# Data, training, and evaluation instructions

Applies to `src/` and its descendants; the repository-level instructions still apply.

## Data contracts
- `dataset.py` supports Cholec80 phase annotation names and integer indices and tool-presence vectors. Keep unknown phase labels explicit errors rather than silently turning them into "Preparation."
- Preserve source-frame numbering (25 fps) and the 1-fps tool-annotation grid when matching video frames to labels. Keep separate train/validation/test video IDs.
- `Cholec80FeatureDataset` yields per-video `(features, phases, tools)` sequences loaded from cached feature tensors. `collate_sequences` returns `(features, phases, tools, mask)` where `mask` is true only for valid timesteps; padded phase labels are -1 and padded tool/features entries are zero.
- Keep feature extraction and cache reuse consistent. The current reported training uses frozen ResNet-50 features cached once. Per-epoch image augmentation does not affect that training path.

## Training and metrics
- `train.py` uses AdamW, cosine warm restarts, gradient clipping, early stopping, and checkpoints selected on validation phase macro F1. Restore the best checkpoint before final test evaluation; do not substitute final-epoch weights.
- Apply loss and evaluation metrics only to valid timesteps. Respect phase `ignore_index=-1` and correctly normalize masked tool and correlation losses per valid element.
- `evaluate.py` computes phase macro F1 over all seven classes, tool mAP over only the tools with positives in the evaluated split, and segment edit score on phase sequences. An absent tool's individual AP is undefined (NaN), not zero; mAP is zero if all tools are absent.
- Keep model-specific behavior and output shapes compatible across training, evaluation, and plotting.
- Prefer deterministic unit tests with synthetic inputs before expensive dataset or GPU execution.

## Targeted tests
From the repository root:
```bash
python -m pytest tests/test_dataset.py tests/test_train.py tests/test_evaluate.py -q
```
