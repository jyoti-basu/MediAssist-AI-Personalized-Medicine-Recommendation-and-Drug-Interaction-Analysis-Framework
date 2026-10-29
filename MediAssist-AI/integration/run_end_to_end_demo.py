#!/usr/bin/env python3
"""Run a clearly fictional, research-only Phase I decision-support demonstration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from mediassist_safety import DDIChecker, MedicationRegistry, validate_age_and_reference
from patient_nlp import extract_patient_information, validate_patient_information
from phase1_core import make_inference_features, recommend_from_feature_row


def build_decision_support_report(fictional_text: str | None = None) -> dict:
    """Run the real Phase I components and return a structured research report."""
    # Research diagnosis indices are supplied explicitly because the dataset has
    # no usable free-text training corpus or verified diagnosis-name mapping.
    fictional_text = fictional_text or ("Fictional research case: 61 years old. Symptoms: cough for 3 days. "
                                        "Current medications: R05CA. Allergies: none. Conditions: none. "
                                        "Research diagnosis indices: 0, 1, 2.")
    mapping = pd.read_csv(ROOT / "data" / "vocabulary_and_mapping" / "medication_vocab_mapping.csv")
    known_codes = set(mapping.atc_code.astype(str))
    extracted = extract_patient_information(fictional_text)
    input_validation = validate_patient_information(extracted, known_codes)
    feature_row = make_inference_features(extracted["research_diagnosis_indices"], extracted["age"])
    model = joblib.load(ROOT / "models" / "recommendation_model.joblib")
    config = json.loads((ROOT / "models" / "feature_metadata.json").read_text())
    candidates = recommend_from_feature_row(model, feature_row, mapping, config["top_k"])
    registry = MedicationRegistry()
    current = registry.resolve(atc=extracted["current_medications_raw"][0])
    checker = DDIChecker()
    for candidate in candidates:
        candidate["ddi_against_current_class"] = checker.check_pair(current["class_id"], candidate["class_id"])
        candidate["age_and_reference_status"] = validate_age_and_reference(extracted["age"], class_id=candidate["class_id"])
        candidate["dosage_reference"] = "unavailable: the supplied clinical-reference template contains no verified dosage facts"
    ddi_statuses = sorted({candidate["ddi_against_current_class"]["status"] for candidate in candidates})
    reference_statuses = sorted({candidate["age_and_reference_status"]["status"] for candidate in candidates})
    report = {
        "prototype_notice": "Fictional research demonstration only; not diagnosis, prescribing, or clinical validation.",
        "original_fictional_input": fictional_text,
        "nlp_extraction": extracted,
        "patient_information_validation": input_validation,
        "current_medication_mapping": current,
        "candidate_class_count": len(candidates),
        "candidate_medication_classes": candidates,
        "decision_support_status": {
            "status": "research_candidates_require_review",
            "all_candidates_require_mapping_review": all(candidate["requires_review"] for candidate in candidates),
            "age_reference_statuses": reference_statuses,
            "ddi_statuses_against_current_medication": ddi_statuses,
            "clinical_action_withheld": True,
        },
        "limitations": [
            "Candidate scores are model rankings, not calibrated clinical probabilities.",
            "ATC classes are not individual drugs; generic and brand mappings are unverified.",
            "DDI source only reports recorded class-level pairs; no record is not proof of safety.",
            "Age and dosage references are unavailable for real classes in this package.",
        ],
    }
    return report


def main() -> None:
    report = build_decision_support_report()
    out = ROOT / "reports" / "end_to_end_demo.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
