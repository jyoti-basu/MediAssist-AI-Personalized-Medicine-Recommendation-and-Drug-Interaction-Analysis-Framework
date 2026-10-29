# Data Dictionary

"Documented" = stated in the bundle's own `README.txt`. "Observed" = verified by this audit. Nothing else is claimed.

## A. Original files (referenced, unchanged; see `data/manifest.csv` for paths and hashes)
| file | columns / structure | status |
|---|---|---|
| `records_final.csv` | `patient_record_index` (int, 0..10,525), `visit_index` (int, 0-based within patient; observed), `diagnosis_codes` / `procedure_codes` / `medication_codes` (JSON list of vocabulary indices), `additional_field` (int; documented only as "4th item of each visit", meaning unknown) | observed |
| `voc_final.csv` | `vocab` in {diag_voc, pro_voc, med_voc}, `index` (0-based, contiguous per vocab), `code` (text). For `med_voc`, `code` is an ATC level-4 string (documented as ATC; format verified) | documented + observed |
| `data_final.csv` | `patienthealthsystemstayid`, `patientunitstayid`, `gender`, `age` (text; numeric or `> 89`), `hospitaladmitoffset` (int, units undocumented), `unitvisitnumber`, `DrugBank_Names` (JSON list of **ATC codes**, despite the name), `diagnosisstring` (JSON list of diagnosis strings), `treatmentstring` (JSON list of procedure/treatment strings), `text` (constant placeholder) | observed |
| `SMILES.csv` | `atc_code`, `med_index` (= `med_voc` index), `smiles_json` (JSON list of 1-4 SMILES strings) | observed |
| `substructure_smiles.csv` | `index`, `smiles` (BRICS-style fragments with `[n*]` attachment points). Rows are alphabetically sorted | observed |
| `ddi_A_final.csv` | first column `atc_code` (row label) + 107 ATC columns; 0/1 symmetric, zero diagonal | observed |
| `ehr_adj_final.csv` | same layout; equals co-occurrence recomputed from all visits | observed |
| `ddi_mask_H.csv` | `atc_code` + `sub_0..sub_284` (0/1). Column order does **not** match `substructure_smiles.csv` row order | observed |

## B. `data/model_data/`
| file | key columns | notes |
|---|---|---|
| `model_inputs.csv` (22,988 rows) | `row_id` (= original row order), `patient_record_index`, `visit_index`, stay ids, `hospitaladmitoffset`, `gender`, `gender_missing_flag`, `age_raw`, `age_years` (blank for top-coded), `age_top_coded_gt89_flag`, `diagnosis_idx` (JSON), `n_diagnosis`, `diagnosis_empty_flag`, `procedure_idx_all` (JSON), `procedure_idx_excluding_medications_segment` (JSON; **use this one**), `n_procedure_medications_segment`, `n_procedure_any_medication_like`, `duplicate_visit_content_flag`, `undocumented_additional_field` (do not use as a feature) | inputs only - **no medication information** |
| `model_targets.csv` | `row_id`, `patient_record_index`, `visit_index`, `medication_idx` (JSON list), `n_medications` | targets |
| `model_targets_multihot.npz` | scipy CSR, shape (22988, 107), int8, column j = `med_voc` index j; row i = `row_id` i | targets as a matrix |
| `splits_patient_level.csv` | `patient_record_index`, `split` (train/val/test), `split_seed` | 70/15/15 by patient, seed 42 |
| `procedure_vocab_leakage_flags.csv` | `index`, `code`, `flag_medication_segment`, `flag_drug_class_keyword`, `flag_any_medication_like`, `leakage_note` | keyword flag is heuristic |

Feature rules: inputs = diagnosis (+ optionally age/gender, filtered procedures). Never use `medication_idx` of the same visit. Fit anything data-dependent (encoders, co-occurrence graphs, thresholds) on the train split only.

## C. `data/vocabulary_and_mapping/`
`medication_vocab_mapping.csv` - one row per `med_voc` entry, **original index order preserved**.
| column | meaning |
|---|---|
| `source_medication_id` | vocabulary index = model output column = DDI matrix index (all verified equal) |
| `source_medication_class` / `atc_code` | the ATC level-4 string from `med_voc` |
| `atc_level`, `atc_level1_anatomical_group_letter`, `atc_level2_code`, `atc_level3_code` | derived purely from the code string (no lookup) |
| `generic_name`, `brand_name` | **blank on purpose** - not verified |
| `mapping_source` | where the code came from |
| `mapping_status` | `class_level_atc_code_only` for all rows |
| `drug_level_mapping_status` | `unmapped` for all rows (class -> individual drug is not guessed) |
| `confidence_or_verification_status` | what was and was not verified |
| `requires_review` | True for all rows |
| `n_source_structures_in_SMILES_csv` | number of SMILES strings listed for the code (meaning undocumented) |
| `ddi_matrix_index`, `ddi_labels_match_vocab`, `n_recorded_ddi_partners`, `has_recorded_ddi_edges` | link to the DDI matrix |
| `n_visits_with_drug` | label frequency in the data |

`safety_status_codes.json` - status -> action vocabulary used by the safety module.

## D. `data/ddi/`
| file | content |
|---|---|
| `ddi_matrix.npy` | int8 107x107, a copy of `ddi_A_final` values; index = `source_medication_id`; **0 = no record, never "safe"** |
| `ddi_index_map.csv` | index <-> ATC <-> original row/column labels (+ match flag) |
| `ddi_edge_list.csv` | 1,246 undirected pairs; `severity`/`mechanism` = `not_provided_by_source` |
| `ddi_coverage_per_drug.csv` | recorded partners per class; zero-degree classes labelled coverage-unknown |
| `ddi_validation_report.json` | machine-readable validation results |
| `ehr_cooccurrence_train_only.npy` | co-occurrence recomputed from training visits only (use this) |
| `ehr_cooccurrence_full_data_LEAKS_TEST.npy` | original `ehr_adj` values; **contains test information**; for reference only |
| `ddi_mask_H.npy` | uint8 107x285 copy of the original mask (column order unverified vs `substructure_smiles`) |
| `h_column_alignment_diagnostic.csv` | inferred, unverified re-alignment hints; not used elsewhere |

## E. `data/synthetic_testing/` and `data/clinical_reference/`
See `synthetic_case_catalogue.md`. Columns of `synthetic_age_validation.csv`: `synthetic_case_id`, `age_input_raw`, `age_years`, `age_group`, `medication_class_id`, `medication_identifier`, `test_rule_id`, `test_category`, `expected_validation_status`, `expected_system_action`, `synthetic_record` (always `SYNTHETIC_TEST_ONLY`), `rule_source_status`, `notes`.
`medication_safety_reference_template.csv` columns: `medication_identifier` (= vocabulary index), `atc_code`, `generic_name`, `patient_age_min`, `patient_age_max`, `age_bound_convention`, `dose_value`, `dose_unit`, `dose_frequency`, `route`, `indication`, `renal_or_hepatic_adjustment`, `contraindications`, `reference_source`, `reference_version_or_date`, `verification_status`, `reviewer`, `notes` - all blank except identifiers and `verification_status`.
