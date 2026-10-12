# Experiment notebook instructions

Applies to `notebooks/` in addition to the repository-level instructions.

- `project_report.ipynb` is both the written research report and the Google Colab workflow for feature extraction, training, evaluation, and plots. Preserve its scientific meaning, explanatory text, and experiment order.
- The notebook targets a GPU Colab runtime and uses Cholec80 annotations plus pre-extracted feature files when available. Read the notebook setup and its actual paths before editing cells; do not hardcode credentials or replace data sources without documenting the change.
- Keep the reported 17-video experiment separate from the full 80-video dataset description. Clearly distinguish measured results, literature references, design motivations, and future work.
- Respect `configs/default.yaml` for video splits, model settings, loss weights, and seed. Update notebook explanations when a code or configuration change alters the reproducibility story.
- Do not manufacture plots, output cells, validation metrics, or execution counts. Do not state that the notebook has been run unless it was actually executed with the required data and environment.
- Avoid committing dataset files, cached feature tensors, weights, large generated outputs, or potentially identifiable surgical imagery. Preserve useful Markdown and source references when making targeted notebook edits.
- Notebook JSON is easy to churn. Keep cell IDs, metadata, and unaffected outputs unchanged where practical and inspect the diff before committing.
