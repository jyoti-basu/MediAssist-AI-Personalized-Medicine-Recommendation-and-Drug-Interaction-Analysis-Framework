# Code Cleanup and Restructuring Log

## Project restructuring

The previous `mediassist_final_dataset/` directory was moved—without duplicating dataset files—to this project root. Dataset artifacts are now isolated beneath `data/`:

| Previous location | Final location |
|---|---|
| `01_model_data/` | `data/model_data/` |
| `02_vocabulary_and_mapping/` | `data/vocabulary_and_mapping/` |
| `03_ddi/` | `data/ddi/` |
| `04_synthetic_testing/` | `data/synthetic_testing/` |
| `05_clinical_reference/` | `data/clinical_reference/` |
| `manifest.csv` | `data/manifest.csv` |
| implementation, model, reports, tests, notebook, root scripts | project-root counterparts |

All affected code, test, report, manifest, notebook, and README references were updated to use the new relative locations. No original clinical source file was added, removed, or changed.

## Removed

- `code/create_phase1_notebook.py` — one-off generator for a finalized notebook. The notebook is now maintained directly and is executed by `run_phase1.sh`.
- Python `__pycache__/` files — generated interpreter artifacts; they do not support reproducibility or implementation behavior.
- Unused `candidate_models` collection in `code/run_phase1.py` — candidate models were never read after validation selection.

## Retained

- `code/build_package.py` and `code/write_reports.py` — required to rebuild/audit the data package when the authorized original source bundle is available.
- `code/run_phase1.py` — trains and evaluates the real recommendation model and regenerates model/report artifacts.
- `integration/` safety, NLP, feature, end-to-end, and synthetic-test modules — actual Phase I behavior.
- `models/` — verified trained model plus feature/evaluation metadata.
- `reports/` — required audit/provenance/limitation documentation, including unresolved source licence status.
- `data/` — the sole retained dataset copy, including reference templates and synthetic software tests.

## Leakage-filter fix

`integration.phase1_core.medication_like_procedure_ids()` and `filter_procedure_indices()` now define the one shared audited filter. Training and `make_inference_features()` both remove all 839 `flag_any_medication_like` procedure IDs; invalid procedure IDs raise an error. The trained model schema remains 4,635 features, so no retraining was required.

`tests/test_phase1.py::test_inference_cannot_bypass_medication_like_procedure_filter` verifies that an audited medication-like procedure ID yields exactly the same feature vector as no procedure and that an out-of-range ID is rejected.

## Changes deliberately avoided

- The trained model and verified held-out evaluation metrics were preserved; smoke-test results do not replace them.
- The DDI source, mappings, reference template, and synthetic test records were not rewritten or populated with unverified clinical content.
- `ddi_mask_H.npy` remains unused because its substructure-column mapping is unresolved.
