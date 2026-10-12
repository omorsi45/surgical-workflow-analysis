# Model and loss instructions

Applies to `src/models/` alongside the root and `src/AGENTS.md` rules.

## Architecture contracts
- `backbone.py` uses pretrained ResNet-50 as a feature extractor, producing cached 2048-dimensional frame features for the reported experiment.
- `temporal.py` offers a frame-wise baseline, bidirectional LSTM, and simplified multi-stage TCN. Every temporal model accepts features shaped `(B, T, feature_dim)`, an optional boolean `(B, T)` validity mask, returns `(B, T, output_dim)`, and exposes `output_dim`.
- The LSTM packs masked sequences. The MS-TCN uses acausal dilated convolutions and processes full offline sequences; it is not a streaming/causal surgical monitoring model. The current TCN passes hidden features between stages and supervises the final stage only, unlike the original deeply supervised MS-TCN.
- The TCN applies masking after each stage, but its BatchNorm can include padded frames for variable-length batches. Do not claim full padding invariance in batched training without addressing that behavior.

## Prediction and loss contracts
- `MultiTaskModel` returns raw phase logits and tool logits, each shaped `(B, T, 7)` with the current labels. Phase probabilities use softmax; independent tool probabilities use sigmoid.
- `build_cooccurrence_matrix` estimates `P(tool | phase)` from training videos only. Keep validation and test labels out of this prior. Smoothing defaults to 0; positive smoothing is an explicit experimental choice.
- `CorrelationLoss` uses soft phase probabilities so gradients reach both task heads. Preserve its differentiability and masking. `MultiTaskLoss` combines weighted phase cross entropy, tool BCE with logits, and optional correlation loss under the configured weights.
- When masks are present, normalize tool and correlation losses by valid frame-by-tool elements, so a fully true mask matches the unmasked mean-loss scale.

## Checks
```bash
python -m pytest tests/test_temporal.py tests/test_multitask.py -q
```
Use small synthetic tensors and verify shapes, gradient propagation, masking, and loss scale.
