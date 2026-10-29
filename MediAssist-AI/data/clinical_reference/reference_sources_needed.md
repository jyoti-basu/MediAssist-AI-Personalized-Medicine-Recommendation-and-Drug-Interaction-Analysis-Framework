# Reference Sources Needed to Populate `medication_safety_reference_template.csv`

`medication_safety_reference_template.csv` is a **schema with 107 class rows and no medical content**. Every row is `verification_status = not_populated_unverified`.
Nothing here was retrieved from, or checked against, any of the sources below - they are **candidates to evaluate**, listed by purpose. Confirm the *current edition* of each.

## Why the template cannot be filled from what we have
- Rows are ATC level-4 **classes**. Age limits, doses, routes, renal/hepatic adjustments and contraindications are specific to an individual drug, formulation and indication; a class-wide value would be wrong.
- An ATC code is a classification, and a **Defined Daily Dose (DDD)** is a drug-utilisation measurement unit. Neither is a patient-specific prescribing dose, and neither may be used as one.
- The ICU data in the bundle record what was administered, not what is appropriate.

## Candidate references for an Indian target population
| need | candidate source(s) to evaluate |
|---|---|
| Approved indications, age limits, contraindications per product | CDSCO-approved prescribing information / package inserts for the specific marketed products |
| Which medicines are in scope nationally | National List of Essential Medicines (NLEM, Ministry of Health & Family Welfare) |
| Dose, route, frequency, renal/hepatic notes | National Formulary of India and Indian Pharmacopoeia (Indian Pharmacopoeia Commission) |
| Condition-specific treatment protocols | ICMR Standard Treatment Workflows |
| Paediatric dosing | IAP (Indian Academy of Pediatrics) drug formulary / guidelines; WHO Model Formulary for Children |
| General international cross-check | WHO Model List of Essential Medicines and WHO Model Formulary |
| ATC class and substance names (names only, not doses) | WHO Collaborating Centre ATC/DDD Index (check its licence terms) |
| Older-adult appropriateness | Published potentially-inappropriate-medication criteria (e.g. Beers, STOPP/START) - check applicability to Indian practice |
| Interaction severity and mechanism | A licensed/curated interaction knowledge base or product labels; confirm the licence and the version of whatever produced `ddi_A_final` |

## Workflow to keep it honest
1. Choose the *individual drugs* in scope per class (a clinician/pharmacist decision, recorded in the `notes` column).
2. For each drug, copy values **only from a cited source**; fill `reference_source` and `reference_version_or_date`.
3. State `age_bound_convention` (is `patient_age_max` inclusive or exclusive, in years/months?).
4. Status progression: `not_populated_unverified` -> `drafted_from_source` -> `verified` (second person, with `reviewer`). The module only trusts rows with `verification_status = verified`.
5. Leave anything unknown **blank**. Blank means `reference_unavailable`, which the system handles safely.
