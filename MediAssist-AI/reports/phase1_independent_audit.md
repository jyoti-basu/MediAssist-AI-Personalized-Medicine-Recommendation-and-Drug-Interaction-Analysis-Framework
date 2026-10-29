# MediAssist-AI Phase I — Independent Implementation Audit

**Audit date:** 2026-10-28  
**Scope:** Read-only inspection and verification of the existing Phase I implementation. The only audit-created artifact is this report. No model, dataset, notebook, or implementation file was changed.

> **Post-audit remediation note:** The subsequent final-fixes pass moved dataset artifacts under `data/`, centralized the procedure leakage filter for training and inference, added its test and a fresh-model smoke test, and integrated the structured end-to-end report into the notebook. See `reports/code_cleanup_log.md` for the change record. The findings below describe the pre-remediation audit evidence.

## A. Executive summary

**Overall status: READY WITH DOCUMENTED LIMITATIONS.**

The workspace contains a genuinely implemented and executable academic research prototype. It uses the supplied derived ICU research tables for a 107-label multi-label medication-*class* task; uses an actual saved one-vs-rest sparse linear model for inference; provides class/index mapping, conservative source-recorded DDI checking, synthetic-rule age/reference validation, rule-based fictional-text extraction, input validation, and an integrated fictional demonstration. The stored final-model and baseline metrics were independently recalculated exactly from the saved model, held-out labels, and DDI matrix.

The most important limitations are: (1) the original CSV source bundle expected at `/home/claude` is unavailable, so a full package rebuild/provenance check is **UNVERIFIED**; (2) generic/brand/drug-level mappings and real dosage/age references are absent by design; (3) the notebook displays the integrated stages but does not itself assemble the same structured final report written by `integration/run_end_to_end_demo.py`; and (4) `make_inference_features()` accepts caller-supplied procedure indices without applying the training-time medication-like procedure filter.

### Files inspected

- `MediAssist_AI_Phase1.ipynb` (all 16 required numbered sections, 14 executed code cells, no stored cell errors)
- `integration/phase1_core.py`, `patient_nlp.py`, `mediassist_safety.py`, `run_end_to_end_demo.py`, `run_synthetic_tests.py`
- `code/run_phase1.py`, `code/build_package.py`, `code/create_phase1_notebook.py`
- `data/model_data/*`, `data/vocabulary_and_mapping/*`, `data/ddi/*`, `data/synthetic_testing/*`, `data/clinical_reference/*`, `manifest.csv`
- `models/recommendation_model.joblib`, `models/feature_metadata.json`, `models/evaluation_metrics.json`
- `reports/dataset_audit_report.md`, `data_quality_report.md`, `model_evaluation_report.md`, `end_to_end_demo.json`, `audit_findings.json`
- `tests/test_phase1.py`, `README.md`, `requirements.txt`, `run_phase1.sh`, `reproduce.sh`

Important missing inputs: `/home/claude/{README.txt,records_final.csv,voc_final.csv,data_final.csv,SMILES.csv,substructure_smiles.csv,ddi_A_final.csv,ehr_adj_final.csv,ddi_mask_H.csv}`. These are the input bundle expected by `reproduce.sh`.

## B. Requirements verification

| Requirement | Status | Evidence | Issue and recommended action |
|---|---|---|---|
| Real supplied research data, not fabricated clinical records | PASS | `phase1_core.load_phase1_data()` reads model tables/NPZ; `synthetic_data_excluded_from_training_and_evaluation: true` in `models/evaluation_metrics.json` | Synthetic files are separate and not loaded by model code. Preserve separation. |
| Multi-label recommendation task and baseline | PASS | `model_targets_multihot.npz` is 22,988×107; `FrequencyBaseline` in `integration/phase1_core.py` | Target is historical administration of ATC level-4 classes, not a clinical recommendation ground truth. |
| Final trained recommendation model and reusable inference | PASS | Saved `OneVsRestClassifier` with 107 `SGDClassifier` estimators; `recommend_from_feature_row()` | Inference returns class index/ATC class/ranking score/review flag, not drug prescriptions. |
| Patient-level split without patient overlap | PASS | `splits_patient_level.csv`; independent overlap calculation: train–val 0, train–test 0, val–test 0 | None found. |
| Leakage-controlled feature construction | PARTIAL | `load_phase1_data()` starts from `procedure_idx_excluding_medications_segment` and excludes all 839 `flag_any_medication_like` procedure codes; targets/identifiers/offset/additional field absent from feature matrix | `make_inference_features()` permits caller-provided procedure indices without applying the 839-code filter. Apply the same filter or reject procedures in the reusable inference path. |
| Validation-only model/threshold selection | PASS | `code/run_phase1.py:65-75`; alpha and top-k selected on validation; test evaluated after final train+validation refit | Fixed top-15 is applied to baseline and final model. |
| DDI checker with conservative semantics | PASS | `DDIChecker.check_pair()`; DDI 0 returns `no_interaction_record_not_proof_of_safety` or coverage-unverified | Source contains no severity/mechanism; do not promote DDI rate to clinical safety. |
| Medication mapping integrity | PASS | `MedicationRegistry`; mapping index/order checks in `load_phase1_data()` | Only class-level ATC mapping exists; individual drugs/names remain unresolved. |
| Age/dosage validation separated from synthetic rules | PASS | `ReferenceTable.from_real_template()` only loads verified rows; none are verified; synthetic rules loaded only in `run_synthetic_tests.py` | Passing tests demonstrate software behavior, not clinical validity. |
| NLP and patient-input validation | PASS | `extract_patient_information()` and `validate_patient_information()` in `patient_nlp.py`; fictional examples independently exercised | Transparent rule-based prototype only; no labelled clinical-NLP evaluation exists. |
| Notebook executes its required pipeline stages | PARTIAL | All 16 sections present; 14/14 stored executions, no stored errors; notebook loads actual model and calls actual mapping/DDI/age/NLP functions | It shows candidates with statuses as a DataFrame, but does not assemble the final structured report produced by `run_end_to_end_demo.py`. Add a call to that report function or construct the report in the notebook. |
| Automated test coverage for important behavior | PARTIAL | `tests/test_phase1.py`: 6 tests passed; synthetic runner: 370 age + 61 DDI cases, 0 mismatches | Unit tests load a saved model but do not train/evaluate it in-test; add a small deterministic train/evaluate smoke test and an inference procedure-filter test. |
| Full source-package rebuild reproducibility | UNVERIFIED | `reproduce.sh` requires an external original CSV folder; all nine expected `/home/claude` files are absent | Obtain authorized source bundle/provenance, then run `bash reproduce.sh <source-dir>` and verify hashes. |
| Model/artifact regeneration from available derived data | PASS | `code/run_phase1.py`; `run_phase1.sh` calls training, demo, tests, notebook execution | Model training produced a convergence warning for the non-selected alpha=1e-5 candidate in prior execution; record/tune this candidate if retained. |

## C. Model evaluation verification

### Verified implementation

- **Features:** 2,572 diagnosis indicators; 2,052 procedure positions after conservative filtering; five age bins; three gender categories; and three flags, total 4,635 sparse features (`phase1_core.load_phase1_data()`). Inputs exclude medication target lists, patient/stay IDs, visit ID, hospital offset, and `undocumented_additional_field`.
- **Targets:** `model_targets_multihot.npz`, 22,988×107 binary target matrix; column *j* corresponds to mapping/DDI class ID *j*.
- **Models:** `FrequencyBaseline` and `OneVsRestClassifier(SGDClassifier(loss='log_loss', alpha=1e-4))`. Candidate alphas 1e-5/1e-4 were compared on validation micro-F1; alpha 1e-4 and top-15 were selected.
- **Partition:** supplied seed-42 patient split: 16,110 train / 3,431 validation / 3,447 test visits and 7,368 / 1,578 / 1,580 patients respectively. Independent set intersections are all zero.
- **Inference:** the persisted object has 107 fitted binary estimators. Calling it on a constructed one-row feature matrix yielded a `(1, 107)` score matrix; `recommend_from_feature_row()` returns 15 ranked class-level records when configured with top-15.

### Independent metric recomputation

I loaded `models/recommendation_model.joblib`, reconstructed the documented feature matrix from the supplied derived files, applied the persisted top-k to the test rows, rebuilt the frequency baseline on train+validation, and called `metric_report()` with `data/ddi/ddi_matrix.npy`. Every persisted value matched exactly.

| Metric | Frequency baseline | Sparse linear model | Verification |
|---|---:|---:|---|
| Micro precision | 0.33035489797891887 | 0.35354414466686007 | Exact match |
| Micro recall | 0.5349514563106796 | 0.57250234888819300 | Exact match |
| Micro F1 | 0.40846535541340345 | 0.43713756202546783 | Exact match |
| Samples Jaccard | 0.24636904229838522 | 0.26943839187500030 | Exact match |
| Recorded DDI-pair rate | 0.42857142857142855 | 0.34573334991089560 | Exact match |
| Recorded / all predicted pairs | 155,115 / 361,935 | 125,133 / 361,935 | Exact match |

Recalculated absolute differences are **+0.028672206612064377 micro-F1** and **0.08283807866053294 recorded-DDI-pair-rate reduction**. The rounded claims (+0.0287 and −0.0828) are consistent. Both models use the same 3,447 held-out patient-disjoint test visits, the same 107-label target matrix, the same top-15 candidate set size, and the same metric function/DDI matrix.

`metric_report()` counts only upper-triangle class pairs in each predicted set. Therefore the DDI result is a **source-recorded class-pair rate**, not a clinical safety measurement; a zero matrix entry explicitly does not prove safety.

## D. Data leakage and dataset integrity

- **Procedure leakage:** `procedure_vocab_leakage_flags.csv` contains 839 `flag_any_medication_like` codes. Training feature construction excludes all of them after starting from the package’s explicit-medication-segment-filtered list. The resulting feature policy is stored in `models/feature_metadata.json`; 14,528 visits have at least one broad heuristic flag. This is more conservative than the previously reported 8,781 explicit medication-segment-affected visits.
- **Target leakage:** target lists are read only from the NPZ target matrix; no target column enters the feature `hstack`. Identifier/offset/undocumented columns are also excluded.
- **Patient leakage:** no supplied patient appears in more than one split. Duplicate content rows remain, but the observed duplicate-content flags do not create cross-patient leakage by themselves; their potential effect on model optimism is not separately quantified.
- **Co-occurrence:** the training-only matrix is 107×107, symmetric, zero diagonal. Independently recomputing the binary same-visit co-occurrence relation from the supplied training target rows gives an exact match (9,870 directed off-diagonal ones). It differs from `ehr_cooccurrence_full_data_LEAKS_TEST.npy` in 40 cells, all full-data-only edges. The Phase I model does not use either co-occurrence matrix. This verifies train-split consistency from the derived target/split artifacts, but cannot independently establish that the missing original inputs were not altered before packaging.
- **Substructure:** the ambiguous `ddi_mask_H.npy` component is not used. The reported unresolved 194/285 column alignment remains a valid limitation.

## E. Module audit

### Mapping

The mapping table has 107 unique ordered `source_medication_id` values 0–106 and 107 matching DDI indices. Every row contains an ATC level-4 class identifier and `mapping_status=class_level_atc_code_only`; all 107 have `drug_level_mapping_status=unmapped`, `requires_review=True`, and blank generic/brand names. No code inspected invents those names or treats a class as an individual medicine.

### DDI

`ddi_matrix.npy` is 107×107, symmetric, and accessed only after numeric ID validation. `DDIChecker` returns unknown-ID for invalid IDs, duplicate for self-pairs, recorded only when matrix value is 1, and conservative non-record/coverage-unverified responses otherwise. `check_list()` produces each unordered pair once. Synthetic DDI coverage is meaningful for software semantics: 25 recorded, 20 no-record-not-safe, 8 coverage-unverified, 5 unknown-ID, and 3 duplicate cases. It does not validate source completeness, severity, mechanism, or clinical management.

### Age/dosage references

The 370 synthetic age/reference cases are in a separate directory and not model inputs. Their categories cover invalid/missing ages (24), real-class age boundaries/no reference (125), synthetic fake-reference windows (150), identifiers/conflicts/ATC-level mismatches, and dosage availability behavior. `validate_age_and_reference()` returns `invalid_age_input` for missing/invalid age and `reference_unavailable` for a real adult class with no verified reference; both behaviors were independently exercised. The real reference template has 107 rows and **zero verified rows**, so no real dose, contraindication, or age restriction is fabricated. Synthetic success is software-rule coverage only.

### NLP and patient validation

`extract_patient_information()` is a non-hard-coded regex implementation for labelled age, symptoms, duration, current medications, allergies, conditions, and explicitly supplied research diagnosis indices. Independent fictional probes showed missing age/current medication/research diagnosis indices produce follow-up questions; `NOT-A-CODE` is reported as unmapped and blocks completion. It does not infer medications/allergies/history absent from text. It cannot map symptoms to diagnosis vocabulary and has no labelled clinical NLP evaluation; its output is appropriate only as a transparent prototype.

## F. Notebook and reproducibility

The notebook contains the required numbered sections 1–16. Stored execution state shows 14 code cells executed and zero error outputs. It loads the actual model from `models/recommendation_model.joblib`, reconstructs actual data, displays persisted test metrics, and invokes `MedicationRegistry`, `DDIChecker`, `validate_age_and_reference`, `extract_patient_information`, and `validate_patient_information` in the fictional demonstration.

During this audit, read-only verification commands completed:

```text
python3 -m unittest discover -s tests -v
Ran 6 tests ... OK

python3 integration/run_synthetic_tests.py
age/reference cases: 370 run, 0 mismatches
ddi cases: 61 run, 0 mismatches
production-mode guarantees: OK
```

The original rebuild cannot be executed with available files because `reproduce.sh` needs the nine original CSV/text files supplied as its positional source directory. `run_phase1.sh` can regenerate the model/artifacts from the derived dataset, but it overwrites current outputs and was not run during this read-only audit.

## G. Prioritized issues

### High

1. **Missing original bundle/provenance evidence.** Full rebuild and independent validation of original-file lineage/licence are unavailable. Obtain authorized sources, run `reproduce.sh`, and preserve hashes/licence evidence.

### Medium

1. **Inference procedure filter mismatch.** Training removes 839 medication-like procedure codes; `make_inference_features()` accepts arbitrary procedure indices. Centralize/apply the same filter at inference or disallow procedure input until an audited transformer is used.
2. **Notebook final report is not structured.** The actual structured report is generated by a separate script. Call it from Section 15 or construct/display the report dictionary in the notebook.
3. **Tests do not train/evaluate a fresh model.** Add a compact deterministic model-fit/evaluation smoke test plus a procedure-filter assertion on inference input.

### Low

1. The non-selected alpha=1e-5 candidate emitted an SGD convergence warning in prior execution. Either raise/tune iterations/tolerance or report why the selection remains adequate.
2. `candidate_models` in `code/run_phase1.py` is retained but unused.

## H. Final recommendation

### READY WITH DOCUMENTED LIMITATIONS

The core research demonstration is technically honest and evidence-backed: data/model artifacts are real, the model evaluation exactly reproduces, train/test patient separation and train-only co-occurrence consistency are verified from available derived files, conservative DDI/age behavior is implemented, and all available tests pass. It should be demonstrated as a class-level historical ICU research prototype only.

Before supervisor presentation, the next actions are:

1. State prominently that outputs are ATC classes, not individual-drug prescriptions or clinical advice.
2. Add the same procedure leakage filter to inference and show the structured decision-support report directly in the notebook.
3. Add a fresh-model training/evaluation smoke test.
4. Obtain or document the authorized source bundle, licence, and provenance; rerun `reproduce.sh` when available.
5. Retain explicit warnings that DDI non-records are not safe, real dosage/reference fields are unavailable, and synthetic tests are not clinical validation.
