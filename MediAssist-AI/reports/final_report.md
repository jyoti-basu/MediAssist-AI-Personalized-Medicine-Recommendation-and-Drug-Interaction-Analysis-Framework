# Final Report

## 1. What was already available
Eight CSV conversions (plus the bundle README): 22,988 ICU visits from 10,526 patients, vocabularies (diagnosis 2,572, procedure 2,052, medication 107), a 107x107 DDI matrix (1,246 recorded pairs), a co-occurrence matrix, SMILES for each class and a 107x285 substructure mask. All identifiers align across files except the substructure mask. The eight `.pkl` originals were **not** in the working directory.

## 2. What was missing
Source/licence documentation; ATC/generic/brand names; any age restriction, dose, contraindication or interaction severity; clinical free text for NLP; any paediatric data; patient-level split; leakage-free co-occurrence graph; consistent substructure ordering; the original pickles.

## 3. What was generated synthetically
`synthetic_age_validation.csv` (370 cases), `synthetic_reference_rules.csv` (6 fake medicines), `synthetic_ddi_check_cases.csv` (61 cases), `age_group_config.json`. All labelled synthetic/test-only; no real medicine received an invented restriction or dose.

## 4. Mappings verified vs unresolved
- **Verified (locally):** vocabulary index <-> ATC level-4 code <-> DDI row/column <-> SMILES row, for all 107 classes; ATC code format.
- **Unresolved (left blank, `requires_review`):** ATC class names, generic names, brand names, class -> individual-drug mapping (107/107). The WHO ATC index could not be reached from this environment.
- **Unresolved (data issue):** `ddi_mask_H` column <-> `substructure_smiles` index (194 of 285 columns ambiguous).

## 5. What depends on external medical references
Everything in `medication_safety_reference_template.csv` (age limits, doses, routes, renal/hepatic adjustments, contraindications), class/drug names, and interaction severity/mechanism. See `data/clinical_reference/reference_sources_needed.md`.

## 6. Are the data sufficient to train and evaluate the recommendation model?
**For a research baseline: yes, with conditions.** Use diagnosis (+ demographics, filtered procedures) -> 107 ATC-4 classes, patient-level splits, train-only co-occurrence graph. **Not sufficient** for: NLP extraction, paediatric behaviour, drug-level output, India-specific claims, or any claim that outputs are clinically correct (labels are historical administrations). Expect heavy class imbalance (min 15, max 14,168 visits per class).

## 7. Leakage, licensing, provenance, quality concerns
See `dataset_audit_report.md` sections 6-8 and `data_quality_report.md`. Headlines: medication administrations inside procedure codes (8,781 visits); `ehr_adj` equals all-data co-occurrence; multi-visit patients; undocumented `additional_field`; placeholder text; undocumented provenance and licence; US-ICU-style population.

## 8. Exact commands to reproduce
```bash
cd mediassist_final_dataset
pip install -r requirements.txt
bash reproduce.sh /home/claude
```
(equivalently `code/build_package.py --source-dir /home/claude`, then `code/write_reports.py`, then `integration/run_synthetic_tests.py`). Seed = 42; outputs are deterministic.

## 9. Final file tree and purpose
- `data/model_data/model_inputs.csv` - inputs only; leakage-flagged (transformed)
- `data/model_data/model_targets.csv` - targets (medication indices) (transformed)
- `data/model_data/model_targets_multihot.npz` - sparse CSR multi-hot targets (transformed)
- `data/model_data/splits_patient_level.csv` - seeded patient-level 70/15/15 (derived)
- `data/model_data/procedure_vocab_leakage_flags.csv` - leakage flags; keyword rule is heuristic (derived)
- `data/vocabulary_and_mapping/medication_vocab_mapping.csv` - class-level only; names blank (transformed)
- `data/vocabulary_and_mapping/safety_status_codes.json` - status/action vocabulary (derived)
- `data/ddi/ddi_matrix.npy` - values identical to original (transformed)
- `data/ddi/ddi_index_map.csv` - ddi_checking (derived)
- `data/ddi/ddi_edge_list.csv` - no severity available (derived)
- `data/ddi/ddi_coverage_per_drug.csv` - ddi_checking (derived)
- `data/ddi/ddi_validation_report.json` - machine-readable validation (derived)
- `data/ddi/ehr_cooccurrence_train_only.npy` - safe co-occurrence graph (derived)
- `data/ddi/ehr_cooccurrence_full_data_LEAKS_TEST.npy` - built from all visits (transformed)
- `data/ddi/ddi_mask_H.npy` - column order unverified (transformed)
- `data/ddi/h_column_alignment_diagnostic.csv` - inferred, unverified (derived)
- `data/synthetic_testing/synthetic_age_validation.csv` - NOT for training/evaluation (synthetic)
- `data/synthetic_testing/synthetic_reference_rules.csv` - fake medicines, fake windows (synthetic)
- `data/synthetic_testing/synthetic_ddi_check_cases.csv` - NOT for training/evaluation (synthetic)
- `data/synthetic_testing/synthetic_case_catalogue.md` - documentation (documentation)
- `data/clinical_reference/medication_safety_reference_template.csv` - no medical content (template)
- `data/clinical_reference/reference_sources_needed.md` - candidate sources, unverified (documentation)
- `integration/mediassist_safety.py` - identifier/age/DDI safety module (code)
- `integration/age_group_config.json` - synthetic test parameters, not clinical (synthetic)
- `integration/run_synthetic_tests.py` - synthetic_software_testing (code)
- `code/build_package.py` - reproducibility (code)
- `code/write_reports.py` - reproducibility (code)
- `reports/audit_findings.json` - source of every reported number (derived)
- `reports/dataset_audit_report.md` - documentation (documentation)
- `reports/data_quality_report.md` - documentation (documentation)
- `reports/data_dictionary.md` - documentation (documentation)
- `reports/final_report.md` - documentation (documentation)
- `README.md` - documentation (documentation)
- `requirements.txt` - reproducibility (code)
- `reproduce.sh` - reproducibility (code)

Original files (referenced, not copied; SHA-256 in `data/manifest.csv`): `README.txt`, `records_final.csv`, `voc_final.csv`, `data_final.csv`, `SMILES.csv`, `substructure_smiles.csv`, `ddi_A_final.csv`, `ehr_adj_final.csv`, `ddi_mask_H.csv`.

## 10. What the package can and cannot safely demonstrate
**Can:** a leakage-aware evaluation protocol for a class-level medication recommender on historical ICU data; safe handling of unknown/unmapped identifiers, missing/invalid age, absent reference data and absent interaction records (verified by 431 test cases); a traceable path from model output index to ATC class to DDI row.
**Cannot:** show that any recommendation, age rule or dose is clinically correct; support paediatric or India-specific claims; evaluate NLP extraction; identify individual drugs or brands; interpret "no recorded interaction" as safety; support redistribution of the data without checking licences.
