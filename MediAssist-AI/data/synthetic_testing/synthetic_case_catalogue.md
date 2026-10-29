# Synthetic Test Data - Catalogue and Rationale

> **Everything in this folder is SYNTHETIC and exists only to test software.** Age groups, thresholds, "reference" windows and expected outcomes are
> *test parameters*, not clinical guidance. No real medicine has an invented age restriction, contraindication or dose. No dose values exist anywhere in this folder.

## Files
| file | purpose |
|---|---|
| `synthetic_age_validation.csv` | 370 cases for the age/identifier/reference validation module |
| `synthetic_reference_rules.csv` | six **fake** medicines `SYN-REF-A..F` with fake windows (`rule_source_status = synthetic_test_rule_not_clinical`) |
| `synthetic_ddi_check_cases.csv` | 61 DDI-handling cases; expected values are *derived from the original matrix*, nothing is invented |
| `../integration/age_group_config.json` | configurable synthetic age groups (child <12, adolescent 12-<18, adult 18-<65, older adult 65-120) |

`rule_source_status` values: `synthetic_test_rule_not_clinical` (fake reference windows), `software_policy_rule_not_clinical` (e.g. minors routed to review, absence of reference), `software_input_validation_rule` (input checks).
`expected_*` columns come from an independent specification in `code/build_package.py`; `integration/run_synthetic_tests.py` runs the real module against them (**370 + 61 cases, 0 mismatches** at build time).

## Decision order (specification)
1. Age missing/invalid -> `invalid_age_input` (`request_valid_age`).
2. Medication identifier problems -> missing / unknown / unmapped / ATC-level mismatch / conflict (`withhold_medication_output_and_flag_for_review`).
3. No verified reference for a real medicine -> minors: `requires_clinician_review`; others: `reference_unavailable`.
4. Reference present (synthetic only): always-review flag -> `requires_clinician_review`; age outside window -> `age_outside_reference_range`; no dose reference -> `dose_reference_unavailable`; otherwise `age_check_passed_synthetic_rule`.

## Age/identifier cases by category
| category | cases | rule ids | what is varied | what it tests in the software |
|---|---|---|---|---|
| `synthetic_reference_age_window_and_dose_availability` | 150 | AGE-020..025 | 25 ages x 6 fake reference medicines (`SYN-REF-A..F`) | The positive path of the rule engine: age inside/outside a window, upper-bound exclusivity, missing dose reference, always-review flag. Fake identifiers only - no real medicine gets an invented restriction. |
| `age_group_boundaries_real_medication_no_reference` | 125 | AGE-010 | 25 ages x 5 real classes: 0, 0.08, 0.5, 11.99, 12, 17.99, 18, 64.99, 65, 89, `> 89`, 120 | Boundary handling of the (synthetic) age groups, inclusive-lower/exclusive-upper convention, and the safe default for real medicines: no reference -> `reference_unavailable` (adult) or `requires_clinician_review` (<18). |
| `invalid_or_missing_age` | 24 | AGE-001/002 | blank, `None`, text, negative, NaN, inf, `12 years`, >120, top-coded > max | Input validation: the module must reject every malformed age and ask for a valid one rather than guess. |
| `unknown_identifier` | 18 | ID-002..006, 015 | index 107, -1, 9999, `abc`, 3.5, `NOTACODE` | Unknown identifiers are withheld and flagged, never coerced. |
| `invalid_age_takes_precedence` | 12 | AGE-003 | invalid age + unknown medication | Precedence: age error is reported first, so the user fixes input before identifier problems. |
| `atc_level_mismatch` | 12 | ID-011..014 | ATC levels 1, 2, 3 and 5 (`R`, `R05`, `R05C`, `R05CA01`) | Enforces *no guessing between ATC levels*: even a level-5 code whose parent exists is withheld. |
| `identifier_normalisation` | 6 | ID-020/021 | `r05ca`, ` R05CA ` | Case/whitespace normalisation only; nothing else is normalised. |
| `identifier_conflict` | 6 | ID-031/041 | class id 0 + `A07AA`; class id 3 + `SYN-REF-A` | Contradictory identifiers are never resolved silently. |
| `missing_identifier` | 3 | ID-001 | both identifiers blank | Missing identifier is distinct from an unknown one. |
| `unmapped_identifier` | 3 | ID-010 | well-formed ATC-4 `Z99ZZ` not in vocabulary | A valid-looking code outside the vocabulary is `unmapped`, not accepted. |
| `consistent_identifier_pair` | 3 | ID-030 | class id 0 + `R05CA` | Both identifier types agreeing is accepted. |
| `id_is_vocabulary_index_not_atc` | 3 | ID-032 | class id 5 alone | A bare integer is a *vocabulary index* and is resolved as such. |
| `synthetic_identifier` | 3 | ID-040 | `SYN-REF-A` | Synthetic ids resolve only when the synthetic reference is loaded (test mode); in production they are unknown. |
| `dose_reference_unavailable_real_medication` | 1 | REF-001 | real class, adult | No real dose reference exists; module returns `reference_unavailable`. |
| `minor_with_real_medication_requires_review` | 1 | REF-002 | age 16 + real class | Software policy flag routes minors to clinician review. |

Expected-status distribution:

| status | cases |
|---|---|
| `requires_clinician_review` | 95 |
| `reference_unavailable` | 69 |
| `age_check_passed_synthetic_rule` | 54 |
| `age_outside_reference_range` | 49 |
| `invalid_age_input` | 36 |
| `dose_reference_unavailable` | 25 |
| `unknown_medication_identifier` | 18 |
| `unmapped_atc_level_mismatch` | 12 |
| `identifier_conflict` | 6 |
| `missing_medication_identifier` | 3 |
| `unmapped_medication_identifier` | 3 |

Age-group coverage: {'child': 107, 'older_adult': 96, 'adult': 75, 'adolescent': 56, 'unclassified': 36}.

## DDI-handling cases
| status | cases |
|---|---|
| `interaction_recorded` | 25 |
| `no_interaction_record_not_proof_of_safety` | 20 |
| `no_interaction_record_coverage_unverified` | 8 |
| `unknown_medication_identifier` | 5 |
| `duplicate_medication_identifier` | 3 |

They test that: recorded pairs raise a warning *without* a severity; an absent record yields a **not-proof-of-safety** notice; drugs with no recorded edges are reported as *coverage unverified*; unknown identifiers and duplicate classes are never treated as "no interaction".

## Why synthetic rows must not enter training or evaluation
- Their labels are outputs of software rules, not clinical outcomes; a model would learn the rule, not medicine.
- Synthetic ages / identifiers have no real prescribing context (the real data contain no child under 14).
- Mixing them into evaluation would inflate or distort metrics and break comparability with real data.
- They are therefore kept in a separate folder and are not referenced by `data/model_data/`.
If you later want synthetic augmentation, justify it, keep it out of validation/test, and report results with and without it.
