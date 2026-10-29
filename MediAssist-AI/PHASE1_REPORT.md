# MediAssist-AI — Phase I Implementation Report

## Outcome

Phase I is implemented as a reproducible, executable research prototype. The primary deliverable is `MediAssist_AI_Phase1.ipynb`; it completed all 14 code cells without errors in this environment. It models historical ICU medication **classes**, not prescriptions or validated clinical decisions.

The final project layout keeps code, notebook, models, tests, and reports at the project root; all dataset artifacts are under `data/`. The notebook now displays the same structured decision-support report produced by `integration/run_end_to_end_demo.py`.

## Dataset and leakage controls

- Used audited derived tables: `data/model_data/model_inputs.csv`, `model_targets_multihot.npz`, `splits_patient_level.csv`, vocabulary mapping, and DDI matrix.
- Verified shape: 22,988 visits, 10,526 patients, 107 target classes, and a 107×107 aligned symmetric DDI matrix.
- Used the supplied patient-level seed-42 split: train 16,110 visits / 7,368 patients; validation 3,431 / 1,578; test 3,447 / 1,580.
- Features are diagnosis indicators, age bins, sex/missingness flags, and procedure indicators. The target, identifiers, admission offset, and undocumented field are excluded.
- To address procedure leakage, the implementation begins with the supplied explicit-medication-segment filtered procedure column and additionally removes every one of the 839 procedure vocabulary entries flagged `flag_any_medication_like`. This affects 14,528 visits with at least one such flag; it is intentionally conservative because the broader heuristic can remove non-leaking information.
- The full-data co-occurrence matrix is never used. Its train-only counterpart was not needed by the baseline. `ddi_mask_H` is disabled because its 285 columns do not have a verified substructure ordering.

## Model and measured results

- Baseline: train+validation medication-class frequency ranker.
- Main model: one-vs-rest sparse `SGDClassifier(loss='log_loss')` on 4,635 binary/input features. Alphas 1e-5 and 1e-4 were compared only on validation micro-F1; 1e-4 was selected. A fixed top-15 candidate set size was selected on validation, then the selected model was refit on train+validation.

| held-out test model | micro precision | micro recall | micro F1 | samples Jaccard | recorded DDI pair rate |
|---|---:|---:|---:|---:|---:|
| Frequency baseline | 0.3304 | 0.5350 | 0.4085 | 0.2464 | 0.4286 |
| Sparse linear model | 0.3535 | 0.5725 | 0.4371 | 0.2694 | 0.3457 |

Scores are uncalibrated rankings. The DDI rate only counts pairs recorded by the supplied source; a zero is not safety evidence. Neither result establishes clinical effectiveness, safety, or prescribing readiness.

## Mapping, DDI, and safety behavior

- Model columns, mapping `source_medication_id`, and DDI indices were checked as 0–106 in identical order.
- Outputs remain ATC level-4 classes. All individual-drug/generic/brand mappings require review and are not invented.
- The DDI checker reports recorded interactions, non-records that are not proof of safety, coverage-unverified classes, unknown IDs, and duplicates; severity and mechanisms are absent.
- Existing synthetic safety tests passed: 370 age/reference cases and 61 DDI cases, 0 mismatches. These are software tests, not medical evidence.
- For real classes, the reference template has no verified dosage or age facts; the real-workflow result is `reference_unavailable` (or clinician review for minors).

## NLP and end-to-end workflow

`integration/patient_nlp.py` supplies a transparent rule-based extractor for fictional input: age, symptoms, duration, free-text current medicines, allergies, conditions, and explicitly supplied reviewed diagnosis indices. It cannot map symptoms to the dataset’s undocumented diagnosis vocabulary, and it marks missing/unknown information for follow-up. `integration/run_end_to_end_demo.py` ran actual extraction, validation, model inference, mapping, DDI checking, and age/reference checks; its output is `reports/end_to_end_demo.json`.

## Tests and reproducibility

- `python3 -m unittest discover -s tests -v`: 8/8 passed, including the shared inference leakage-filter test and a patient-disjoint fresh-model smoke test on real supplied data.
- `python3 integration/run_synthetic_tests.py`: 431 synthetic checks passed (370 age/reference + 61 DDI).
- `jupyter nbconvert --to notebook --execute --inplace MediAssist_AI_Phase1.ipynb --ExecutePreprocessor.timeout=300`: 14/14 code cells executed, 0 errors.
- `python3 code/run_phase1.py` regenerates model, metrics, report, and plot deterministically (seed 42).

The package rebuild script expects these unavailable originals under the argument passed to it: `README.txt`, `records_final.csv`, `voc_final.csv`, `data_final.csv`, `SMILES.csv`, `substructure_smiles.csv`, `ddi_A_final.csv`, `ehr_adj_final.csv`, and `ddi_mask_H.csv`. The historical audit cites `/home/claude`; that directory does not exist in this workspace, so `reproduce.sh` cannot be run successfully here.

## Unresolved limits and Phase II hand-off

The original data provenance/licence is unconfirmed; the data are predominantly adult US ICU records; the text column is a constant placeholder; individual medication names, doses, contraindications, and DDI severity/mechanism are unverified; and the model has no paediatric, outpatient, India-specific, or clinical validation. Phase II should only add interface/infrastructure after qualified clinical data/reference governance and prospective evaluation are available.
