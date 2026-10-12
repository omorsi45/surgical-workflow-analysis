# Test instructions

Applies to `tests/` in addition to the root instructions.

- Run `python -m pytest tests/ -q` from the repository root. The test files cover dataset parsing/collation, training checkpoint selection, evaluation metrics, temporal shape/masking, and multi-task losses.
- Prefer lightweight synthetic torch tensors, temporary files, and fixtures over Cholec80 downloads, Colab, GPU dependencies, or full ResNet-50 feature extraction.
- Use `tmp_path` for feature files and checkpoints. Never modify real dataset annotations, experiment outputs, or shared checkpoints in tests.
- Protect the `collate_sequences` contract: valid-mask shape `(B,T)`, phase padding -1, tool/feature padding zero, and unmodified valid content.
- Protect every temporal model's optional-mask and `output_dim` contracts. Exercise both masked and unmasked loss paths, correlation-loss gradient propagation, and video-length differences.
- Keep metric semantics: macro phase F1 includes all seven classes; absent tools contribute NaN to per-tool AP and are excluded from mAP; segment edit score is based on phase sequences.
- Check that training restores the checkpoint chosen by best validation F1. Do not weaken assertions or skip tests to work around regressions.
- Report the actual tests executed and any environment limitations. Do not claim an end-to-end Colab experiment was covered by unit tests.
