#!/usr/bin/env python3
"""Runs integration/mediassist_safety.py against every synthetic case and reports mismatches.

The expected values in the CSVs come from an independent specification inside code/build_package.py
(and, for DDI, from the ORIGINAL ddi matrix). A mismatch therefore means the module deviates from spec.
Synthetic reference rules are loaded ONLY here (test mode); production code uses the real, unpopulated template.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mediassist_safety as ms  # noqa: E402

PKG = HERE.parent
syn_ref_path = PKG / "data" / "synthetic_testing" / "synthetic_reference_rules.csv"
synth_ids = set(pd.read_csv(syn_ref_path, dtype=str).medication_identifier)
registry = ms.MedicationRegistry(synthetic_ids=synth_ids)
reference = ms.ReferenceTable.from_synthetic_rules(syn_ref_path)
cfg = ms.load_age_config()


def nb(x):
    return None if (x is None or str(x).strip() == "") else str(x)


def main() -> int:
    age_df = pd.read_csv(PKG / "data" / "synthetic_testing" / "synthetic_age_validation.csv", dtype=str, keep_default_na=False)
    bad = []
    for _, r in age_df.iterrows():
        res = ms.validate_age_and_reference(nb(r.age_input_raw), nb(r.medication_class_id), nb(r.medication_identifier),
                                            registry=registry, reference=reference, config=cfg)
        exp_group = r.age_group
        if (res["status"], res["action"], res["age_group"]) != (r.expected_validation_status, r.expected_system_action, exp_group):
            bad.append((r.synthetic_case_id, r.test_rule_id, r.age_input_raw, r.medication_class_id, r.medication_identifier,
                        (r.expected_validation_status, r.expected_system_action, exp_group),
                        (res["status"], res["action"], res["age_group"])))
    print(f"age/reference cases: {len(age_df)} run, {len(bad)} mismatches")
    for b in bad[:25]:
        print("  MISMATCH", b)

    ddi_df = pd.read_csv(PKG / "data" / "synthetic_testing" / "synthetic_ddi_check_cases.csv", dtype=str, keep_default_na=False)
    chk = ms.DDIChecker()
    bad2 = []

    def conv(x):
        if x == "":
            return None
        try:
            return int(x)
        except ValueError:
            return x  # e.g. 'x' -> must be rejected as unknown
    for _, r in ddi_df.iterrows():
        res = chk.check_pair(conv(r.med_a_class_id), conv(r.med_b_class_id))
        if (res["status"], res["action"]) != (r.expected_ddi_status, r.expected_system_action):
            bad2.append((r.synthetic_case_id, r.med_a_class_id, r.med_b_class_id, r.expected_ddi_status, res["status"]))
    print(f"ddi cases: {len(ddi_df)} run, {len(bad2)} mismatches")
    for b in bad2[:25]:
        print("  MISMATCH", b)

    # production-mode guarantees (no synthetic reference loaded)
    prod = ms.validate_age_and_reference("30", None, "SYN-REF-A")
    assert prod["status"] == ms.ST_UNKNOWN_MED, prod  # synthetic ids must not resolve in production
    prod2 = ms.validate_age_and_reference("30", "0", None)
    assert prod2["status"] == ms.ST_REF_UNAVAILABLE, prod2  # every real medication: reference unavailable
    # absence of a DDI record must never be reported as safe
    allowed = {ms.DDI_RECORDED, ms.DDI_NO_RECORD, ms.DDI_NO_RECORD_NO_EDGES}
    for a in range(chk.n):
        for b in range(a + 1, chk.n):
            s = chk.check_pair(a, b)["status"]
            assert s in allowed, s  # no status that asserts safety / "no interaction" exists in the vocabulary
    print("production-mode guarantees: OK")
    return 1 if (bad or bad2) else 0


if __name__ == "__main__":
    sys.exit(main())
