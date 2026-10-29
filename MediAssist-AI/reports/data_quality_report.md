# Data Quality Report

Nothing was deleted, imputed or silently corrected. Each issue is **flagged** in the cleaned tables so you can decide.

| # | issue | count | handling in this package |
|---|---|---|---|
| 1 | Visits with empty diagnosis list | 15 | kept; `diagnosis_empty_flag=1` in `model_inputs.csv` |
| 2 | Visits with empty procedure list | 2,646 | kept (plausible: no procedure recorded) |
| 3 | Visits with empty medication list | 0 | none |
| 4 | Out-of-range / duplicate indices inside a visit | 0 / 0 | none found |
| 5 | Missing `gender` | 2 (one patient) | kept; `gender_missing_flag=1`; not imputed |
| 6 | Age top-coded as "> 89" | 544 | `age_years` left blank, `age_top_coded_gt89_flag=1`; choose your own handling (e.g. cap) and document it |
| 7 | Age unparseable (other) | 0 | none |
| 8 | Age < 18 (min 14) | 10 rows (2 under 15) | kept; paediatric behaviour **cannot** be learned or evaluated from this data |
| 9 | Exact duplicate visits (same diagnosis + procedure + medication content) | 19 extra copies (37 rows involved) | kept; `duplicate_visit_content_flag=1` (could be genuinely identical stays; not removed) |
| 10 | Identical medication sets (common, expected) | 3,241 extra copies | informational |
| 11 | Duplicate `patientunitstayid` | 0 | none |
| 12 | `text` column is one constant placeholder | 1 distinct value | not usable for NLP |
| 13 | `additional_field` undocumented | all rows | kept under an explicit name, excluded from features |
| 14 | Procedure entries that are medication administrations | 374 vocab entries / 8,781 visits | flagged; filtered column provided |
| 15 | `ehr_adj` built from all data | 1 matrix | train-only replacement provided |
| 16 | `ddi_mask_H` vs `substructure_smiles` column order | 91/285 columns recoverable | flagged; diagnostic only |
| 17 | Classes with zero recorded DDIs | 7 (A12CA, A11DA, J01CG, A11GA, G01AD, B02AA, A11HA) | `ddi_coverage_per_drug.csv`; checker reports coverage as unverified |
| 18 | Column `DrugBank_Names` holds ATC codes, not names | all rows | documented in data dictionary |
| 19 | Rare classes (< 100 visits) | 8 | informational; consider class-imbalance handling |

## Other checks that passed
- All code indices are in range and every vocabulary entry is used at least once (diagnosis 2,572/2,572, procedure 2,052/2,052, medication 107/107).
- Mean list length: diagnoses 4.91, procedures 7.22, medications 9.54 per visit.
- Age is constant within each patient (True).
- Cross-file alignment checks all pass except the H-mask/substructure ordering (see audit report section 4).

## Not done on purpose
No imputation, no row removal, no re-encoding of the original vocabularies, no inferred meaning for `additional_field`, no names or brands added to the medication vocabulary.
