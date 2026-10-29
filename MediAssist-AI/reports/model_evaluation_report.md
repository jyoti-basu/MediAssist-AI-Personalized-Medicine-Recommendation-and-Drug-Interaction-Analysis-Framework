# Phase I Model Evaluation Report

## Scope
This is a reproducible, leakage-aware research baseline for multi-label prediction of historical ICU medication **classes** (107 ATC level-4 labels). It does not prescribe medicines or establish clinical effectiveness.

## Data and leakage control
- Inputs: diagnosis indices, conservatively filtered procedure indices, age bands, sex, and missingness flags.
- Excluded: medication targets, identifiers, offsets, undocumented `additional_field`, and all 839 procedure vocabulary entries flagged medication-like. This affected 14,528 visits in the supplied audit flag.
- Split: supplied patient-level 70/15/15 split (seed 42), with {'train': 16110, 'val': 3431, 'test': 3447} visits. Training transforms and model fitting used training data only during selection; test labels were not used for selection.
- The full-data co-occurrence matrix and unresolved substructure mask were not features. The train-only co-occurrence matrix was verified as 107x107 but was not needed by this baseline.

## Models
- Baseline: train+validation class-prevalence ranker.
- Main model: one-vs-rest `SGDClassifier(loss='log_loss')`, sparse binary feature inputs. Alpha was selected only by validation micro-F1 from [1e-05, 0.0001].
- Prediction set size: validation-selected top-15; model scores are uncalibrated ranking scores, not clinical probabilities.

## Held-out test metrics

| model | micro precision | micro recall | micro F1 | samples Jaccard | recorded DDI pair rate |
|---|---:|---:|---:|---:|---:|
| Frequency baseline | 0.3304 | 0.5350 | 0.4085 | 0.2464 | 0.4286 |
| Sparse linear model | 0.3535 | 0.5725 | 0.4371 | 0.2694 | 0.3457 |

The DDI rate counts only source-recorded class pairs. A zero DDI-matrix entry is **not** evidence of safety, and rate differences are not clinical benefit or safety evidence.

## Validation-only selection
```json
{
  "1e-05": {
    "top_k": 15,
    "selected_validation_metrics": {
      "micro_precision": 0.3313708345477509,
      "micro_recall": 0.5113949862060694,
      "micro_f1": 0.4021553299612088,
      "samples_jaccard": 0.24228045903388346,
      "predicted_set_size": 15,
      "recorded_ddi_pair_rate": 0.3294166631969022,
      "recorded_ddi_pairs": 118674,
      "all_predicted_pairs": 360255,
      "ddi_interpretation": "recorded source interactions only; a non-recorded pair is not proof of safety"
    }
  },
  "0.0001": {
    "top_k": 15,
    "selected_validation_metrics": {
      "micro_precision": 0.36224618672884484,
      "micro_recall": 0.5590440206309224,
      "micro_f1": 0.43962600073102,
      "samples_jaccard": 0.2713602689889941,
      "predicted_set_size": 15,
      "recorded_ddi_pair_rate": 0.34687929383353455,
      "recorded_ddi_pairs": 124965,
      "all_predicted_pairs": 360255,
      "ddi_interpretation": "recorded source interactions only; a non-recorded pair is not proof of safety"
    }
  }
}
```

## Limitations
Historical ICU administrations are labels, not validated treatment recommendations. ATC level-4 classes are not individual drugs; names, doses, contraindications, severity, mechanisms, and clinical references remain unverified. This is not suitable for paediatric, outpatient, India-specific, or prescribing use.
