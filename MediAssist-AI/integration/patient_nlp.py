"""Transparent, fictional-text-only NLP and patient input validation for Phase I."""
from __future__ import annotations

import re


AGE_RE = re.compile(r"\b(?:age\s*[:=]?\s*)?(\d{1,3})\s*(?:years? old|yo|y/o)\b", re.I)
MED_RE = re.compile(r"(?:current medications?|medications?)\s*[:=]\s*([^.;]+)", re.I)
ALLERGY_RE = re.compile(r"(?:allerg(?:y|ies))\s*[:=]?\s*([^.;]+)", re.I)
CONDITION_RE = re.compile(r"(?:history of|conditions?)\s*[:=]?\s*([^.;]+)", re.I)
SYMPTOM_RE = re.compile(r"(?:symptoms?)\s*[:=]?\s*([^.;]+)", re.I)
DURATION_RE = re.compile(r"\bfor\s+(\d+\s*(?:hours?|days?|weeks?))\b", re.I)
DIAG_RE = re.compile(r"research diagnosis indices?\s*[:=]\s*([0-9,\s]+)", re.I)


def _split(value: str | None) -> list[str]:
    return [x.strip() for x in re.split(r",|\band\b", value or "") if x.strip() and x.strip().lower() not in {"none", "no known"}]


def extract_patient_information(text: str) -> dict:
    """Rule-based extraction; absent fields remain None, not guessed."""
    age_match = AGE_RE.search(text)
    age = int(age_match.group(1)) if age_match else None
    uncertain = []
    if age is not None and not 0 <= age <= 120:
        uncertain.append("age outside accepted range")
        age = None
    medicines = _split((MED_RE.search(text) or [None, None])[1])
    diag_match = DIAG_RE.search(text)
    diagnosis_indices = [int(x) for x in re.findall(r"\d+", diag_match.group(1))] if diag_match else []
    return {
        "source_text": text,
        "age": age,
        "symptoms": _split((SYMPTOM_RE.search(text) or [None, None])[1]),
        "symptom_duration": (DURATION_RE.search(text).group(1) if DURATION_RE.search(text) else None),
        "current_medications_raw": medicines,
        "allergies": _split((ALLERGY_RE.search(text) or [None, None])[1]),
        "conditions": _split((CONDITION_RE.search(text) or [None, None])[1]),
        "research_diagnosis_indices": diagnosis_indices,
        "uncertain_extractions": uncertain,
        "extraction_method": "transparent_rule_based_prototype_not_trained_on_clinical_dataset",
    }


def validate_patient_information(patient: dict, known_atc_codes: set[str] | None = None) -> dict:
    missing, questions, unmapped = [], [], []
    if patient.get("age") is None:
        missing.append("age")
        questions.append("What is the patient's age in years?")
    if not patient.get("symptoms"):
        missing.append("symptoms")
        questions.append("What symptoms are present, and how long have they lasted?")
    if not patient.get("current_medications_raw"):
        missing.append("current_medications")
        questions.append("What medicines or ATC codes is the patient currently taking?")
    if not patient.get("research_diagnosis_indices"):
        missing.append("research_diagnosis_indices")
        questions.append("Provide reviewed diagnosis-vocabulary indices for this research model; symptoms cannot be mapped automatically.")
    for item in patient.get("current_medications_raw", []):
        if known_atc_codes is not None and item.upper() not in known_atc_codes:
            unmapped.append(item)
    if unmapped:
        questions.append("Please provide verified ATC level-4 codes for the listed current medicines; free-text names are not mapped by this prototype.")
    return {"is_complete_for_demo": not missing and not unmapped and not patient.get("uncertain_extractions"),
            "missing_fields": missing, "unmapped_medications": unmapped,
            "follow_up_questions": questions}
