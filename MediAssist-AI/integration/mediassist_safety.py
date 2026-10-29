"""
mediassist_safety.py - safe handling of medication identifiers, age input and DDI lookups
for the MediAssist-AI decision-support prototype.

ACADEMIC RESEARCH PROTOTYPE. NOT clinical guidance. Nothing here is a validated prescribing rule.

Design principles
-----------------
* A medication identifier is only trusted if it resolves explicitly. Two identifier types exist and are
  never mixed up:  `class_id`  (integer vocabulary index, 0..106)  and  `atc`  (ATC level-4 string).
* An absent interaction record is NEVER reported as "safe".
* Age groups and thresholds in `age_group_config.json` are SOFTWARE TEST PARAMETERS, not clinical rules.
* No clinical age restriction, contraindication or dose is stored in this module. Real reference data must
come from `data/clinical_reference/medication_safety_reference_template.csv` once it has been populated
  and marked verified by a qualified reviewer. Until then every real medication yields `reference_unavailable`
  (or `requires_clinician_review` for minors).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "age_group_config.json"

# ---- status vocabularies (kept in one place; also exported to safety_status_codes.json) -----------------
ST_INVALID_AGE = "invalid_age_input"
ST_MISSING_MED = "missing_medication_identifier"
ST_UNKNOWN_MED = "unknown_medication_identifier"
ST_UNMAPPED_MED = "unmapped_medication_identifier"
ST_LEVEL_MISMATCH = "unmapped_atc_level_mismatch"
ST_CONFLICT = "identifier_conflict"
ST_REVIEW = "requires_clinician_review"
ST_REF_UNAVAILABLE = "reference_unavailable"
ST_AGE_OUTSIDE = "age_outside_reference_range"
ST_DOSE_UNAVAILABLE = "dose_reference_unavailable"
ST_PASS_SYNTH = "age_check_passed_synthetic_rule"

ACTION = {
    ST_INVALID_AGE: "request_valid_age",
    ST_MISSING_MED: "withhold_medication_output_and_flag_for_review",
    ST_UNKNOWN_MED: "withhold_medication_output_and_flag_for_review",
    ST_UNMAPPED_MED: "withhold_medication_output_and_flag_for_review",
    ST_LEVEL_MISMATCH: "withhold_medication_output_and_flag_for_review",
    ST_CONFLICT: "withhold_medication_output_and_flag_for_review",
    ST_REVIEW: "show_with_mandatory_clinician_review_notice",
    ST_REF_UNAVAILABLE: "show_with_warning_age_suitability_not_assessed",
    ST_AGE_OUTSIDE: "flag_age_unsuitable_per_reference_and_require_review",
    ST_DOSE_UNAVAILABLE: "show_without_dose_information_and_warn",
    ST_PASS_SYNTH: "proceed_with_standard_decision_support_disclaimer",
}

DDI_RECORDED = "interaction_recorded"
DDI_NO_RECORD = "no_interaction_record_not_proof_of_safety"
DDI_NO_RECORD_NO_EDGES = "no_interaction_record_coverage_unverified"
DDI_UNKNOWN_ID = "unknown_medication_identifier"
DDI_DUPLICATE = "duplicate_medication_identifier"
DDI_ACTION = {
    DDI_RECORDED: "show_interaction_warning_severity_not_available",
    DDI_NO_RECORD: "show_notice_no_record_is_not_evidence_of_safety",
    DDI_NO_RECORD_NO_EDGES: "show_notice_no_record_is_not_evidence_of_safety_and_drug_has_no_recorded_interactions_in_source",
    DDI_UNKNOWN_ID: "withhold_and_flag_for_review",
    DDI_DUPLICATE: "show_notice_duplicate_class_not_assessed",
}

_ATC_L1 = re.compile(r"^[A-Z]$")
_ATC_L2 = re.compile(r"^[A-Z]\d{2}$")
_ATC_L3 = re.compile(r"^[A-Z]\d{2}[A-Z]$")
_ATC_L4 = re.compile(r"^[A-Z]\d{2}[A-Z]{2}$")
_ATC_L5 = re.compile(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$")


def load_age_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------------------------------------
# Age handling
# ----------------------------------------------------------------------------------------------------------
def parse_age(raw, config: dict | None = None):
    """Return (age_years or None, issue or None). Never guesses: anything unparseable is invalid."""
    cfg = config or load_age_config()
    if raw is None:
        return None, "missing"
    if isinstance(raw, bool):
        return None, "invalid_type"
    if isinstance(raw, (int, float, np.integer, np.floating)):
        val = float(raw)
    else:
        s = str(raw).strip()
        if s == "":
            return None, "missing"
        m = re.fullmatch(r">\s*(\d+)", s)  # source data stores top-coded ages as '> 89'
        if m:
            val = float(m.group(1)) + 1.0
            if val > cfg["max_valid_age_years"]:
                return None, "out_of_range"
            return val, "top_coded_source_value"
        try:
            val = float(s)
        except ValueError:
            return None, "not_numeric"
    if math.isnan(val) or math.isinf(val):
        return None, "not_finite"
    if val < cfg["min_valid_age_years"] or val > cfg["max_valid_age_years"]:
        return None, "out_of_range"
    return val, None


def age_group(age: float | None, config: dict | None = None) -> str:
    """SYNTHETIC test grouping: lower bound inclusive, upper bound exclusive (last group inclusive of max)."""
    if age is None:
        return "unclassified"
    cfg = config or load_age_config()
    for g in cfg["age_groups"]:
        if g["min_inclusive"] <= age < g["max_exclusive"]:
            return g["name"]
    if age == cfg["max_valid_age_years"]:
        return cfg["age_groups"][-1]["name"]
    return "unclassified"


# ----------------------------------------------------------------------------------------------------------
# Medication registry
# ----------------------------------------------------------------------------------------------------------
class MedicationRegistry:
    def __init__(self, mapping_csv: Path | None = None, synthetic_ids: set[str] | None = None):
        mapping_csv = mapping_csv or (PKG / "data" / "vocabulary_and_mapping" / "medication_vocab_mapping.csv")
        df = pd.read_csv(mapping_csv, dtype={"atc_code": str})
        df = df.sort_values("source_medication_id").reset_index(drop=True)
        assert df.source_medication_id.tolist() == list(range(len(df))), "vocabulary order must be preserved"
        self.n = len(df)
        self.atc_by_id = dict(zip(df.source_medication_id, df.atc_code))
        self.id_by_atc = {v: k for k, v in self.atc_by_id.items()}
        self.synthetic_ids = set(synthetic_ids or [])

    @staticmethod
    def _blank(x) -> bool:
        return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x.strip() == "")

    def _resolve_class_id(self, cid):
        if isinstance(cid, bool):
            return ST_UNKNOWN_MED, None
        if isinstance(cid, (int, np.integer)):
            v = int(cid)
        elif isinstance(cid, (float, np.floating)):
            if math.isnan(cid) or cid != int(cid):
                return ST_UNKNOWN_MED, None
            v = int(cid)
        else:
            s = str(cid).strip()
            if not re.fullmatch(r"-?\d+", s):
                return ST_UNKNOWN_MED, None
            v = int(s)
        if 0 <= v < self.n:
            return None, v
        return ST_UNKNOWN_MED, None

    def _resolve_atc(self, atc):
        s = str(atc).strip().upper()
        if s in self.synthetic_ids:
            return None, s
        if _ATC_L4.match(s):
            if s in self.id_by_atc:
                return None, self.id_by_atc[s]
            return ST_UNMAPPED_MED, None
        if _ATC_L1.match(s) or _ATC_L2.match(s) or _ATC_L3.match(s) or _ATC_L5.match(s):
            # parent/child level of a code we hold: the level mapping is NOT guessed
            return ST_LEVEL_MISMATCH, None
        return ST_UNKNOWN_MED, None

    def resolve(self, class_id=None, atc=None) -> dict:
        has_id, has_atc = not self._blank(class_id), not self._blank(atc)
        if not has_id and not has_atc:
            return {"status": ST_MISSING_MED, "class_id": None, "atc": None, "synthetic": False}
        rid = ratc = None
        if has_id:
            err, rid = self._resolve_class_id(class_id)
            if err:
                return {"status": err, "class_id": None, "atc": None, "synthetic": False}
        if has_atc:
            err, ratc = self._resolve_atc(atc)
            if err:
                return {"status": err, "class_id": None, "atc": None, "synthetic": False}
        if has_id and has_atc:
            if isinstance(ratc, str):  # synthetic id cannot be combined with a numeric class id
                return {"status": ST_CONFLICT, "class_id": rid, "atc": ratc, "synthetic": False}
            if rid != ratc:
                return {"status": ST_CONFLICT, "class_id": rid, "atc": self.atc_by_id[rid], "synthetic": False}
        if has_atc and isinstance(ratc, str):
            return {"status": "ok", "class_id": None, "atc": ratc, "synthetic": True}
        cid = rid if has_id else ratc
        return {"status": "ok", "class_id": cid, "atc": self.atc_by_id[cid], "synthetic": False}


# ----------------------------------------------------------------------------------------------------------
# Reference lookup (real template is deliberately empty -> reference_unavailable)
# ----------------------------------------------------------------------------------------------------------
class ReferenceTable:
    """rows: dict key -> {patient_age_min, patient_age_max (exclusive, test convention), dose_ref_available, review_flag}"""

    def __init__(self, rows: dict | None = None):
        self.rows = rows or {}

    @classmethod
    def from_real_template(cls, path: Path | None = None) -> "ReferenceTable":
        path = path or (PKG / "data" / "clinical_reference" / "medication_safety_reference_template.csv")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        rows = {}
        for _, r in df.iterrows():
            if r.get("verification_status", "").strip().lower() == "verified" and r["patient_age_min"] != "":
                rows[int(r["medication_identifier"])] = {
                    "patient_age_min": float(r["patient_age_min"]),
                    "patient_age_max": float(r["patient_age_max"]) if r["patient_age_max"] != "" else None,
                    "dose_ref_available": r["dose_value"] != "",
                    "review_flag": False,
                }
        return cls(rows)

    @classmethod
    def from_synthetic_rules(cls, path: Path) -> "ReferenceTable":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        rows = {}
        for _, r in df.iterrows():
            rows[r["medication_identifier"]] = {
                "patient_age_min": float(r["patient_age_min"]),
                "patient_age_max": float(r["patient_age_max_exclusive"]),
                "dose_ref_available": r["dose_reference_available"].upper() == "Y",
                "review_flag": r["always_require_review"].upper() == "Y",
            }
        return cls(rows)

    def lookup(self, resolved: dict):
        key = resolved["atc"] if resolved["synthetic"] else resolved["class_id"]
        return self.rows.get(key)


# ----------------------------------------------------------------------------------------------------------
# Age / reference validation (steps 6 of the pipeline)
# ----------------------------------------------------------------------------------------------------------
def validate_age_and_reference(age_raw, class_id=None, atc=None, registry: MedicationRegistry | None = None,
                               reference: ReferenceTable | None = None, config: dict | None = None) -> dict:
    cfg = config or load_age_config()
    registry = registry or MedicationRegistry()
    reference = reference or ReferenceTable.from_real_template()
    age, issue = parse_age(age_raw, cfg)
    grp = age_group(age, cfg)
    out = {"age_years": age, "age_group": grp, "age_issue": issue}

    if age is None:
        return {**out, "status": ST_INVALID_AGE, "action": ACTION[ST_INVALID_AGE], "medication": None}
    med = registry.resolve(class_id, atc)
    out["medication"] = med
    if med["status"] != "ok":
        return {**out, "status": med["status"], "action": ACTION[med["status"]]}

    ref = reference.lookup(med)
    if ref is None:  # no verified reference exists for this identifier
        if cfg.get("policy_minors_require_review", True) and age < cfg["adult_age_years"]:
            return {**out, "status": ST_REVIEW, "action": ACTION[ST_REVIEW]}
        return {**out, "status": ST_REF_UNAVAILABLE, "action": ACTION[ST_REF_UNAVAILABLE]}

    if ref["review_flag"]:
        return {**out, "status": ST_REVIEW, "action": ACTION[ST_REVIEW]}
    upper = ref["patient_age_max"]
    if age < ref["patient_age_min"] or (upper is not None and age >= upper):
        return {**out, "status": ST_AGE_OUTSIDE, "action": ACTION[ST_AGE_OUTSIDE]}
    if not ref["dose_ref_available"]:
        return {**out, "status": ST_DOSE_UNAVAILABLE, "action": ACTION[ST_DOSE_UNAVAILABLE]}
    return {**out, "status": ST_PASS_SYNTH, "action": ACTION[ST_PASS_SYNTH]}


# ----------------------------------------------------------------------------------------------------------
# DDI lookup (step 5 of the pipeline)
# ----------------------------------------------------------------------------------------------------------
class DDIChecker:
    """Lookup over the original ddi_A matrix. 0 means 'no recorded interaction' - NOT 'checked and safe'."""

    def __init__(self, matrix_path: Path | None = None):
        matrix_path = matrix_path or (PKG / "data" / "ddi" / "ddi_matrix.npy")
        self.A = np.load(matrix_path)
        self.n = self.A.shape[0]
        self.degree = self.A.sum(axis=1)

    def _valid(self, i) -> bool:
        if isinstance(i, bool) or i is None:
            return False
        if isinstance(i, (int, np.integer)):
            return 0 <= int(i) < self.n
        return False

    def check_pair(self, a, b) -> dict:
        if not (self._valid(a) and self._valid(b)):
            return {"status": DDI_UNKNOWN_ID, "action": DDI_ACTION[DDI_UNKNOWN_ID]}
        a, b = int(a), int(b)
        if a == b:
            return {"status": DDI_DUPLICATE, "action": DDI_ACTION[DDI_DUPLICATE]}
        if self.A[a, b] == 1:
            st = DDI_RECORDED
        elif self.degree[a] == 0 or self.degree[b] == 0:
            st = DDI_NO_RECORD_NO_EDGES
        else:
            st = DDI_NO_RECORD
        return {"status": st, "action": DDI_ACTION[st]}

    def check_list(self, ids) -> list[dict]:
        res = []
        ids = list(ids)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                r = self.check_pair(ids[x], ids[y])
                res.append({"a": ids[x], "b": ids[y], **r})
        return res
