# Experiment configuration instructions

Applies to `configs/` in addition to the repository-level instructions.

- `default.yaml` is the shared source for data paths, label definitions, video splits, model hyperparameters, optimizer settings, loss weights, and augmentation options.
- The current reported experiment uses 17 Cholec80 videos: training [1,2,3,4,5,6,7,8,9,10], validation [11,12], and test [13,14,16,17,18]. Keep the split at the video level, with no overlap. Do not silently replace it with the canonical full-data split.
- Sampling is 1 fps. Source video frame numbers use the original 25-fps indexing, while tool annotations are available at 1 fps. Changes to sampling require corresponding annotation, caching, evaluation, and notebook changes.
- Seven surgical phase labels and seven tool labels have a defined order. Coordinate any reordering with annotation decoding, classifier dimensions, test fixtures, metrics, and visualizations.
- The default feature representation is 2048-dimensional ResNet-50 output. Changes to feature size or model dimensions must remain compatible with cached features and the temporal architectures.
- The augmentation options are documented but are not used for the published frozen-backbone cached-feature experiments. Do not claim that changing them alters already cached features or reproduces a new augmented training run.
- When changing experimental settings, update the report or reproducibility notes and add targeted tests for altered contracts. Do not modify configuration solely to improve a displayed metric.
