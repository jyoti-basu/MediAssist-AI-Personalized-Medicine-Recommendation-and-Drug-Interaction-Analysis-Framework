#!/usr/bin/env python3
"""Train and evaluate the Phase I leakage-aware recommendation prototype."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
from phase1_core import (FrequencyBaseline, fit_main_model, load_phase1_data,
                         metric_report, prediction_at_k, save_artifacts, scores,
                         select_top_k_on_validation, split_indices)


def write_report(path: Path, result: dict) -> None:
    val = result["validation_model_selection"]
    test = result["test_metrics"]
    baseline = result["baseline_test_metrics"]
    text = f"""# Phase I Model Evaluation Report

## Scope
This is a reproducible, leakage-aware research baseline for multi-label prediction of historical ICU medication **classes** (107 ATC level-4 labels). It does not prescribe medicines or establish clinical effectiveness.

## Data and leakage control
- Inputs: diagnosis indices, conservatively filtered procedure indices, age bands, sex, and missingness flags.
- Excluded: medication targets, identifiers, offsets, undocumented `additional_field`, and all {result['feature_metadata']['procedure_codes_removed_as_medication_like']} procedure vocabulary entries flagged medication-like. This affected {result['feature_metadata']['procedure_visits_with_any_medication_like_flag']:,} visits in the supplied audit flag.
- Split: supplied patient-level 70/15/15 split (seed 42), with {result['split_counts']} visits. Training transforms and model fitting used training data only during selection; test labels were not used for selection.
- The full-data co-occurrence matrix and unresolved substructure mask were not features. The train-only co-occurrence matrix was verified as 107x107 but was not needed by this baseline.

## Models
- Baseline: train+validation class-prevalence ranker.
- Main model: one-vs-rest `SGDClassifier(loss='log_loss')`, sparse binary feature inputs. Alpha was selected only by validation micro-F1 from {result['alphas_considered']}.
- Prediction set size: validation-selected top-{result['top_k']}; model scores are uncalibrated ranking scores, not clinical probabilities.

## Held-out test metrics

| model | micro precision | micro recall | micro F1 | samples Jaccard | recorded DDI pair rate |
|---|---:|---:|---:|---:|---:|
| Frequency baseline | {baseline['micro_precision']:.4f} | {baseline['micro_recall']:.4f} | {baseline['micro_f1']:.4f} | {baseline['samples_jaccard']:.4f} | {baseline['recorded_ddi_pair_rate']:.4f} |
| Sparse linear model | {test['micro_precision']:.4f} | {test['micro_recall']:.4f} | {test['micro_f1']:.4f} | {test['samples_jaccard']:.4f} | {test['recorded_ddi_pair_rate']:.4f} |

The DDI rate counts only source-recorded class pairs. A zero DDI-matrix entry is **not** evidence of safety, and rate differences are not clinical benefit or safety evidence.

## Validation-only selection
```json
{json.dumps(val, indent=2)}
```

## Limitations
Historical ICU administrations are labels, not validated treatment recommendations. ATC level-4 classes are not individual drugs; names, doses, contraindications, severity, mechanisms, and clinical references remain unverified. This is not suitable for paediatric, outpatient, India-specific, or prescribing use.
"""
    path.write_text(text)


def main() -> None:
    data = load_phase1_data(ROOT)
    indices = split_indices(data.split)
    ddi = np.load(ROOT / "data" / "ddi/ddi_matrix.npy")
    train, val, test = indices["train"], indices["val"], indices["test"]
    candidate_results = {}
    for alpha in (1e-5, 1e-4):
        model = fit_main_model(data.feature_matrix[train], data.targets[train], alpha)
        val_scores = scores(model, data.feature_matrix[val])
        top_k, k_metrics = select_top_k_on_validation(val_scores, data.targets[val], ddi)
        candidate_results[str(alpha)] = {"top_k": top_k, "selected_validation_metrics": k_metrics[str(top_k)]}
    selected_alpha = max((1e-5, 1e-4), key=lambda a: candidate_results[str(a)]["selected_validation_metrics"]["micro_f1"])
    top_k = candidate_results[str(selected_alpha)]["top_k"]
    # Refit after choice on train+validation; the test split remains untouched.
    final_train = np.concatenate([train, val])
    model = fit_main_model(data.feature_matrix[final_train], data.targets[final_train], selected_alpha)
    test_prediction = prediction_at_k(scores(model, data.feature_matrix[test]), top_k)
    baseline = FrequencyBaseline().fit(data.targets[final_train], top_k)
    baseline_prediction = baseline.predict(len(test))
    result = {
        "seed": 42,
        "target": "multi-label historical ATC level-4 medication class prediction",
        "feature_metadata": data.feature_metadata,
        "split_counts": {name: int(len(value)) for name, value in indices.items()},
        "patient_counts": {name: int(data.inputs.iloc[value].patient_record_index.nunique()) for name, value in indices.items()},
        "alphas_considered": [1e-5, 1e-4],
        "selected_alpha": selected_alpha,
        "top_k": top_k,
        "validation_model_selection": candidate_results,
        "baseline_test_metrics": metric_report(data.targets[test], baseline_prediction, ddi),
        "test_metrics": metric_report(data.targets[test], test_prediction, ddi),
        "ddi_alignment": {"shape": list(ddi.shape), "mapping_indices_match": bool(data.mapping.ddi_matrix_index.tolist() == list(range(107)))},
        "synthetic_data_excluded_from_training_and_evaluation": True,
    }
    model_dir = ROOT / "models"
    save_artifacts(model, {**data.feature_metadata, "top_k": top_k, "selected_alpha": selected_alpha,
                           "selection_split": "validation", "training_after_selection": "train+validation"}, model_dir)
    (model_dir / "evaluation_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(ROOT / "reports" / "model_evaluation_report.md", result)

    plt.figure(figsize=(9, 4))
    prevalence = np.asarray(data.targets[train].mean(axis=0)).ravel()
    plt.bar(np.arange(15), np.sort(prevalence)[-15:][::-1])
    plt.title("Top 15 medication-class prevalences (training split)")
    plt.xlabel("Ranked medication class")
    plt.ylabel("Fraction of training visits")
    plt.tight_layout()
    (ROOT / "reports" / "figures").mkdir(exist_ok=True)
    plt.savefig(ROOT / "reports" / "figures" / "top_medication_prevalence.png", dpi=150)
    plt.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
