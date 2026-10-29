#!/usr/bin/env python3
"""
build_package.py - MediAssist-AI dataset audit + package builder (fully reproducible, seeded).

Usage:
    python3 code/build_package.py --source-dir /path/to/original/csvs [--out /path/to/mediassist_final_dataset]

Original files are only READ. Their SHA-256 hashes are recorded before and after the run and the run aborts
if they differ. Nothing medical is invented: age groups / thresholds / reference rules used for software
testing are explicitly synthetic and live only under data/synthetic_testing/ and integration/.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SEED = 42
ORIGINALS = ["README.txt", "records_final.csv", "voc_final.csv", "data_final.csv", "SMILES.csv",
             "substructure_smiles.csv", "ddi_A_final.csv", "ehr_adj_final.csv", "ddi_mask_H.csv"]
EXPECTED_PICKLES = ["data_final.pkl", "records_final.pkl", "voc_final.pkl", "ddi_A_final.pkl", "ddi_mask_H.pkl",
                    "ehr_adj_final.pkl", "SMILES.pkl", "substructure_smiles.pkl"]
DRUG_KEYWORDS = (r"antibiotic|antibacterial|antiviral|antifungal|analgesic|antiarrhythmic|insulin|heparin|diuretic|"
                 r"vasopressor|steroid|glucocorticoid|antihypertensive|anticoagul|sedat|antiemetic|bronchodilator|"
                 r"antidepress|antipsychotic|beta blocker|ace inhibitor|statin|chemotherapy")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(obj, path: Path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()
    src, out = args.source_dir, args.out
    for d in ["data/model_data", "data/vocabulary_and_mapping", "data/ddi", "data/synthetic_testing",
              "data/clinical_reference", "reports", "integration"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    F = {"seed": SEED, "source_dir": str(src)}

    # ------------------------------------------------------------------ 0. integrity of originals
    present = {f: (src / f).exists() for f in ORIGINALS}
    F["files_present"] = present
    F["pickles_present"] = {f: (src / f).exists() for f in EXPECTED_PICKLES}
    missing = [f for f, ok in present.items() if not ok]
    if missing:
        sys.exit(f"missing original files: {missing}")
    hashes_before = {f: sha256(src / f) for f in ORIGINALS}
    F["original_sha256"] = hashes_before

    # ------------------------------------------------------------------ 1. load
    rec = pd.read_csv(src / "records_final.csv")
    voc = pd.read_csv(src / "voc_final.csv")
    dat = pd.read_csv(src / "data_final.csv")
    sm = pd.read_csv(src / "SMILES.csv")
    sub = pd.read_csv(src / "substructure_smiles.csv")
    A_df = pd.read_csv(src / "ddi_A_final.csv", index_col=0)
    E_df = pd.read_csv(src / "ehr_adj_final.csv", index_col=0)
    H_df = pd.read_csv(src / "ddi_mask_H.csv", index_col=0)

    med = voc[voc.vocab == "med_voc"].sort_values("index").reset_index(drop=True)
    diag = voc[voc.vocab == "diag_voc"].sort_values("index").reset_index(drop=True)
    pro = voc[voc.vocab == "pro_voc"].sort_values("index").reset_index(drop=True)
    NMED = len(med)
    atc = med.code.tolist()

    # per-file structural audit ------------------------------------------------------------------------
    def tab_audit(df, name, extra=None):
        d = {"rows": int(df.shape[0]), "cols": int(df.shape[1]),
             "dtypes": {c: str(t) for c, t in df.dtypes.items()} if df.shape[1] <= 20 else
             {"unique_dtypes": sorted({str(t) for t in df.dtypes})},
             "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
             "exact_duplicate_rows": int(df.duplicated().sum())}
        if extra:
            d.update(extra)
        return d

    F["tables"] = {
        "records_final.csv": tab_audit(rec, "records"),
        "voc_final.csv": tab_audit(voc, "voc", {"vocab_sizes": voc.vocab.value_counts().to_dict(),
                                               "index_contiguous": {v: bool((g.sort_values("index")["index"].values == np.arange(len(g))).all())
                                                                    for v, g in voc.groupby("vocab")},
                                               "duplicate_codes_within_vocab": int(voc.duplicated(["vocab", "code"]).sum())}),
        "data_final.csv": tab_audit(dat, "data"),
        "SMILES.csv": tab_audit(sm, "smiles"),
        "substructure_smiles.csv": tab_audit(sub, "sub", {"alphabetically_sorted": bool(list(sub.smiles) == sorted(sub.smiles)),
                                                          "unique_smiles": int(sub.smiles.nunique())}),
        "ddi_A_final.csv": tab_audit(A_df.reset_index(), "A"),
        "ehr_adj_final.csv": tab_audit(E_df.reset_index(), "E"),
        "ddi_mask_H.csv": tab_audit(H_df.reset_index(), "H"),
    }

    # parse list columns --------------------------------------------------------------------------------
    for c in ["diagnosis_codes", "procedure_codes", "medication_codes"]:
        rec[c + "_l"] = rec[c].apply(json.loads)
    dat["DB_l"] = dat.DrugBank_Names.apply(json.loads)
    dat["DG_l"] = dat.diagnosisstring.apply(json.loads)

    # records structure ---------------------------------------------------------------------------------
    rs = {}
    for c, v in [("diagnosis_codes", len(diag)), ("procedure_codes", len(pro)), ("medication_codes", NMED)]:
        L = rec[c + "_l"]
        allv = [x for l in L for x in l]
        rs[c] = {"vocab_size": v, "visits_with_empty_list": int((L.apply(len) == 0).sum()),
                 "min_idx": int(min(allv)), "max_idx": int(max(allv)),
                 "out_of_range_values": int(sum(1 for x in allv if x < 0 or x >= v)),
                 "duplicate_codes_within_a_visit": int(sum(len(l) != len(set(l)) for l in L)),
                 "mean_list_length": float(L.apply(len).mean()), "distinct_codes_used": int(len(set(allv)))}
    F["records"] = rs
    F["patients"] = int(rec.patient_record_index.nunique())
    F["visits_per_patient"] = {int(k): int(v) for k, v in rec.groupby("patient_record_index").size().value_counts().sort_index().items()}
    F["additional_field_values"] = {int(k): int(v) for k, v in rec.additional_field.value_counts().sort_index().items()}

    # cross-file alignment ------------------------------------------------------------------------------
    idx_of = {c: i for i, c in enumerate(atc)}
    F["alignment"] = {
        "records_rows_equal_data_rows": bool(len(rec) == len(dat)),
        "medication_idx_matches_data_final_ATC_list_rowwise": bool(all([idx_of.get(x, -1) for x in db] == l for l, db in zip(rec.medication_codes_l, dat.DB_l))),
        "diagnosis_idx_matches_data_final_diagnosis_strings_rowwise": bool(all([diag.code[x] for x in l] == g for l, g in zip(rec.diagnosis_codes_l, dat.DG_l))),
        "ddi_A_row_labels_equal_med_voc_order": bool(list(A_df.index) == atc),
        "ddi_A_col_labels_equal_med_voc_order": bool(list(A_df.columns) == atc),
        "ehr_adj_labels_equal_med_voc_order": bool(list(E_df.index) == atc and list(E_df.columns) == atc),
        "ddi_mask_H_row_labels_equal_med_voc_order": bool(list(H_df.index) == atc),
        "SMILES_med_index_to_atc_matches_med_voc": bool(sm.set_index("med_index").sort_index().atc_code.tolist() == atc),
        "substructure_count_equals_H_columns": bool(len(sub) == H_df.shape[1]),
        "all_med_codes_are_ATC_level4_format": bool(med.code.str.match(r"^[A-Z]\d{2}[A-Z]{2}$").all()),
        "distinct_ATC_level3_groups": int(med.code.str[:4].nunique()),
    }
    dat["pidx"] = rec.patient_record_index
    # empirical ordering checks on FILE order (no re-sorting, so the checks are not tautological)
    F["alignment"]["patients_contiguous_and_non_decreasing_in_file"] = bool((rec.patient_record_index.diff().dropna() >= 0).all())
    F["alignment"]["visit_index_is_0_1_2_.._within_each_patient_in_file_order"] = bool(
        (rec.groupby("patient_record_index").visit_index.apply(lambda s: list(s) == list(range(len(s))))).all())
    F["alignment"]["share_patients_hospitaladmitoffset_non_increasing_in_file_order"] = float(
        dat.groupby("pidx").hospitaladmitoffset.apply(lambda s: bool((s.diff().dropna() <= 0).all())).mean())
    F["alignment"]["share_patients_unitvisitnumber_strictly_increasing_in_file_order"] = float(
        dat.groupby("pidx").unitvisitnumber.apply(lambda s: bool((s.diff().dropna() > 0).all())).mean())
    F["alignment"]["additional_field_equals_unitvisitnumber_minus_1_share"] = float((rec.additional_field == dat.unitvisitnumber - 1).mean())
    F["alignment"]["additional_field_equals_visit_index_share"] = float((rec.additional_field == rec.visit_index).mean())

    # demographics -----------------------------------------------------------------------------------------
    age_num = pd.to_numeric(dat.age, errors="coerce")
    top_coded = dat.age.astype(str).str.strip().eq("> 89")
    F["demographics"] = {
        "gender_counts": {str(k): int(v) for k, v in dat.gender.value_counts(dropna=False).items()},
        "gender_missing_rows": int(dat.gender.isna().sum()),
        "age_top_coded_'> 89'_rows": int(top_coded.sum()),
        "age_unparseable_rows_excluding_top_coded": int((age_num.isna() & ~top_coded).sum()),
        "age_min": float(age_num.min()), "age_max_numeric": float(age_num.max()),
        "age_median_numeric": float(age_num.median()),
        "rows_age_under_18": int((age_num < 18).sum()), "rows_age_under_15": int((age_num < 15).sum()),
        "rows_age_18_to_64": int(((age_num >= 18) & (age_num < 65)).sum()),
        "rows_age_65_plus_incl_top_coded": int(((age_num >= 65) | top_coded).sum()),
        "age_constant_within_patient": bool((dat.groupby("pidx").age.nunique() == 1).all()),
        "text_column_distinct_values": int(dat.text.nunique()), "text_column_value": str(dat.text.iloc[0]),
        "treatmentstring_nonempty_rows": int((dat.treatmentstring != "[]").sum()),
        "hospitaladmitoffset_min": int(dat.hospitaladmitoffset.min()), "hospitaladmitoffset_max": int(dat.hospitaladmitoffset.max()),
    }

    # duplicates ----------------------------------------------------------------------------------------------
    sig = rec.apply(lambda r: (tuple(sorted(r.diagnosis_codes_l)), tuple(sorted(r.procedure_codes_l)), tuple(sorted(r.medication_codes_l))), axis=1)
    dup_content = sig.duplicated(keep=False)
    F["duplicates"] = {"visits_sharing_identical_diag_proc_med_content": int(dup_content.sum()),
                       "extra_copies": int(sig.duplicated().sum()),
                       "identical_medication_sets_extra_copies": int(rec.medication_codes_l.apply(lambda l: tuple(sorted(l))).duplicated().sum()),
                       "duplicate_patientunitstayid": int(dat.patientunitstayid.duplicated().sum())}

    # ------------------------------------------------------------------ 2. leakage analysis
    pro["segments"] = pro.code.str.split("|").apply(lambda s: [x.strip().lower() for x in s])
    pro["flag_medication_segment"] = pro.segments.apply(lambda s: "medications" in s)
    pro["flag_drug_class_keyword"] = pro.code.str.contains(DRUG_KEYWORDS, case=False, regex=True)
    pro["flag_any_medication_like"] = pro.flag_medication_segment | pro.flag_drug_class_keyword
    medseg = set(pro.loc[pro.flag_medication_segment, "index"])
    anyflag = set(pro.loc[pro.flag_any_medication_like, "index"])
    rec["n_proc_medseg"] = rec.procedure_codes_l.apply(lambda l: sum(1 for x in l if x in medseg))
    rec["n_proc_anyflag"] = rec.procedure_codes_l.apply(lambda l: sum(1 for x in l if x in anyflag))
    pro_out = pro[["index", "code", "flag_medication_segment", "flag_drug_class_keyword", "flag_any_medication_like"]].copy()
    pro_out["leakage_note"] = np.where(pro_out.flag_medication_segment,
                                       "explicit medication administration entry; remove from model INPUT features",
                                       np.where(pro_out.flag_drug_class_keyword, "drug-class keyword match (heuristic); review before using as input", ""))
    pro_out.to_csv(out / "data/model_data" / "procedure_vocab_leakage_flags.csv", index=False)

    # co-occurrence recomputation (leakage of ehr_adj)
    C_full = np.zeros((NMED, NMED), dtype=np.int8)
    for l in rec.medication_codes_l:
        for i in l:
            for j in l:
                if i != j:
                    C_full[i, j] = 1
    ehr_equals = bool((C_full == E_df.values).all())

    F["leakage"] = {
        "procedure_vocab_entries_total": int(len(pro)),
        "procedure_vocab_entries_with_medications_segment": int(pro.flag_medication_segment.sum()),
        "procedure_vocab_entries_with_drug_class_keyword": int(pro.flag_drug_class_keyword.sum()),
        "procedure_vocab_entries_any_medication_like": int(pro.flag_any_medication_like.sum()),
        "visits_with_nonempty_procedures": int((rec.procedure_codes_l.apply(len) > 0).sum()),
        "visits_with_>=1_medications_segment_procedure": int((rec.n_proc_medseg > 0).sum()),
        "visits_with_>=1_any_medication_like_procedure": int((rec.n_proc_anyflag > 0).sum()),
        "ehr_adj_equals_cooccurrence_recomputed_from_ALL_visits": ehr_equals,
        "ehr_adj_offdiag_density": float((E_df.values.sum()) / (NMED * (NMED - 1))),
        "same_patient_has_multiple_visits_share": float((rec.groupby("patient_record_index").size() > 1).mean()),
        "data_final_column_named_DrugBank_Names_actually_holds": "ATC level-4 code strings (identical to med_voc codes)",
    }

    # ------------------------------------------------------------------ 3. patient-level splits + train-only co-occurrence
    patients = np.sort(rec.patient_record_index.unique())
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(patients)
    n = len(perm)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    split_of = {}
    for i, p in enumerate(perm):
        split_of[int(p)] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    splits = pd.DataFrame({"patient_record_index": patients})
    splits["split"] = splits.patient_record_index.map(split_of)
    splits["split_seed"] = SEED
    splits.to_csv(out / "data/model_data" / "splits_patient_level.csv", index=False)
    rec["split"] = rec.patient_record_index.map(split_of)
    F["splits"] = {"patients": {k: int(v) for k, v in splits.split.value_counts().items()},
                   "visits": {k: int(v) for k, v in rec.split.value_counts().items()}, "seed": SEED, "ratio": "70/15/15 by patient"}

    C_train = np.zeros((NMED, NMED), dtype=np.int8)
    for l in rec.loc[rec.split == "train", "medication_codes_l"]:
        for i in l:
            for j in l:
                if i != j:
                    C_train[i, j] = 1
    np.save(out / "data/ddi" / "ehr_cooccurrence_full_data_LEAKS_TEST.npy", E_df.values.astype(np.int8))
    np.save(out / "data/ddi" / "ehr_cooccurrence_train_only.npy", C_train)
    F["leakage"]["pairs_cooccurring_in_full_data_but_not_in_train_split(offdiag cells)"] = int(((E_df.values == 1) & (C_train == 0)).sum())

    # ------------------------------------------------------------------ 4. model inputs / targets
    age_years = age_num.copy()
    inputs = pd.DataFrame({
        "row_id": np.arange(len(rec)),
        "patient_record_index": rec.patient_record_index,
        "visit_index": rec.visit_index,
        "patienthealthsystemstayid": dat.patienthealthsystemstayid,
        "patientunitstayid": dat.patientunitstayid,
        "hospitaladmitoffset": dat.hospitaladmitoffset,
        "gender": dat.gender,
        "gender_missing_flag": dat.gender.isna().astype(int),
        "age_raw": dat.age,
        "age_years": age_years,
        "age_top_coded_gt89_flag": top_coded.astype(int),
        "diagnosis_idx": rec.diagnosis_codes,
        "n_diagnosis": rec.diagnosis_codes_l.apply(len),
        "diagnosis_empty_flag": (rec.diagnosis_codes_l.apply(len) == 0).astype(int),
        "procedure_idx_all": rec.procedure_codes,
        "procedure_idx_excluding_medications_segment": rec.procedure_codes_l.apply(lambda l: json.dumps([x for x in l if x not in medseg])),
        "n_procedure_medications_segment": rec.n_proc_medseg,
        "n_procedure_any_medication_like": rec.n_proc_anyflag,
        "duplicate_visit_content_flag": dup_content.astype(int).values,
        "undocumented_additional_field": rec.additional_field,
    })
    inputs.to_csv(out / "data/model_data" / "model_inputs.csv", index=False)
    targets = pd.DataFrame({"row_id": np.arange(len(rec)), "patient_record_index": rec.patient_record_index,
                            "visit_index": rec.visit_index, "medication_idx": rec.medication_codes,
                            "n_medications": rec.medication_codes_l.apply(len)})
    targets.to_csv(out / "data/model_data" / "model_targets.csv", index=False)
    rows, cols = zip(*[(r, c) for r, l in enumerate(rec.medication_codes_l) for c in l])
    Y = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(rec), NMED))
    sparse.save_npz(out / "data/model_data" / "model_targets_multihot.npz", Y)
    F["targets"] = {"multihot_shape": list(Y.shape), "nnz": int(Y.nnz), "mean_meds_per_visit": float(Y.sum(1).mean())}

    # label distribution
    freq = np.asarray(Y.sum(0)).ravel()
    F["label_frequency"] = {"min_visits_per_drug": int(freq.min()), "max_visits_per_drug": int(freq.max()),
                            "median": float(np.median(freq)), "drugs_in_<100_visits": int((freq < 100).sum()),
                            "top5": [(atc[i], int(freq[i])) for i in np.argsort(-freq)[:5]]}

    # ------------------------------------------------------------------ 5. DDI validation + artifacts
    A = A_df.values.astype(np.int8)
    deg = A.sum(1)
    E = E_df.values
    eye = np.eye(NMED, dtype=bool)
    zero_deg = [atc[i] for i in np.where(deg == 0)[0]]
    ddi_pairs_cooccur = int(np.triu(A * C_full, 1).sum())
    # per-visit "ground truth" DDI rate (pairs with a recorded DDI / all pairs) as a data-property statistic
    vis_with_ddi, rates = 0, []
    for l in rec.medication_codes_l:
        pairs = [(i, j) for x, i in enumerate(l) for j in l[x + 1:]]
        if not pairs:
            continue
        k = sum(int(A[i, j]) for i, j in pairs)  # int(): A is int8 and would overflow past 127 pairs
        vis_with_ddi += int(k > 0)
        rates.append(k / len(pairs))
    ddi_rep = {
        "shape": list(A.shape), "dtype_after_load": str(A_df.values.dtype), "values_present": sorted(np.unique(A).tolist()),
        "symmetric": bool((A == A.T).all()), "diagonal_sum": int(np.trace(A)),
        "undirected_edges": int(np.triu(A, 1).sum()), "offdiag_density": float(A.sum() / (NMED * (NMED - 1))),
        "row_labels_equal_med_voc_order": F["alignment"]["ddi_A_row_labels_equal_med_voc_order"],
        "col_labels_equal_med_voc_order": F["alignment"]["ddi_A_col_labels_equal_med_voc_order"],
        "duplicate_labels": int(A_df.index.duplicated().sum()),
        "drugs_with_zero_recorded_interactions": zero_deg,
        "degree_min": int(deg.min()), "degree_median": float(np.median(deg)), "degree_max": int(deg.max()),
        "ddi_pairs_that_are_also_coprescribed_in_records": ddi_pairs_cooccur,
        "visits_containing_at_least_one_recorded_ddi_pair": int(vis_with_ddi),
        "mean_per_visit_share_of_pairs_with_recorded_ddi": float(np.mean(rates)),
        "ddi_edges_not_in_ehr_adj": int(((A == 1) & (E == 0)).sum()),
        "ddi_semantics": {
            "meaning_of_1": "an interaction is recorded between the two ATC level-4 classes in the bundle's DDI source",
            "meaning_of_0": "NOT DETERMINABLE from the bundle: could be 'no interaction recorded', 'not assessed' or 'not in source'. Treated as 'no record'.",
            "severity_or_mechanism_available": False,
            "source_database_documented_in_bundle": False,
            "level_of_assertion": "class (ATC level-4) level; bundle does not document how molecule-level data were aggregated to class level",
        },
        "safe_use_rule": "absence of a recorded interaction must never be displayed as 'no interaction' or 'safe'",
    }
    np.save(out / "data/ddi" / "ddi_matrix.npy", A)
    jdump(ddi_rep, out / "data/ddi" / "ddi_validation_report.json")
    F["ddi"] = ddi_rep
    F["ddi"]["ehr_adj"] = {"symmetric": bool((E == E.T).all()), "diagonal_sum": int(np.trace(E)),
                           "equals_cooccurrence_recomputed_from_all_visits": ehr_equals}

    pd.DataFrame({"ddi_matrix_index": np.arange(NMED), "atc_code": atc,
                  "vocab_index": med["index"], "row_label_in_ddi_A": list(A_df.index),
                  "col_label_in_ddi_A": list(A_df.columns), "labels_match_vocab": [a == b == c for a, b, c in zip(atc, A_df.index, A_df.columns)]
                  }).to_csv(out / "data/ddi" / "ddi_index_map.csv", index=False)
    edges = [(i, j) for i in range(NMED) for j in range(i + 1, NMED) if A[i, j]]
    pd.DataFrame({"idx_a": [e[0] for e in edges], "idx_b": [e[1] for e in edges],
                  "atc_a": [atc[e[0]] for e in edges], "atc_b": [atc[e[1]] for e in edges],
                  "recorded_interaction": 1, "severity": "not_provided_by_source", "mechanism": "not_provided_by_source",
                  "source_status": "source_database_undocumented_in_bundle"}).to_csv(out / "data/ddi" / "ddi_edge_list.csv", index=False)
    pd.DataFrame({"ddi_matrix_index": np.arange(NMED), "atc_code": atc, "n_recorded_ddi_partners": deg,
                  "coverage_status": np.where(deg == 0, "no_recorded_interactions_assessment_status_unknown", "has_recorded_interactions"),
                  }).to_csv(out / "data/ddi" / "ddi_coverage_per_drug.csv", index=False)
    np.save(out / "data/ddi" / "ddi_mask_H.npy", H_df.values.astype(np.uint8))

    # H <-> substructure_smiles column alignment diagnostic (needs RDKit; skipped gracefully if missing)
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import BRICS
        import rdkit
        RDLogger.DisableLog("rdApp.*")
        sub_idx = {s: i for i, s in enumerate(sub.smiles)}
        smap = sm.set_index("atc_code").smiles_json.apply(json.loads)
        R = np.zeros((NMED, len(sub)), dtype=int)
        not_in_vocab = 0
        for ai, a in enumerate(atc):
            for s in smap[a]:
                for f in BRICS.BRICSDecompose(Chem.MolFromSmiles(s), returnMols=False):
                    can = Chem.MolToSmiles(Chem.MolFromSmiles(f))
                    if can in sub_idx:
                        R[ai, sub_idx[can]] = 1
                    else:
                        not_in_vocab += 1
        Hv = H_df.values.astype(int)
        sigR, sigH = collections.defaultdict(list), collections.defaultdict(list)
        for j in range(R.shape[1]):
            sigR[tuple(np.where(R[:, j])[0])].append(j)
        for k in range(Hv.shape[1]):
            sigH[tuple(np.where(Hv[:, k])[0])].append(k)
        diag_rows = []
        for k in range(Hv.shape[1]):
            s = tuple(np.where(Hv[:, k])[0])
            cands = sigR.get(s, [])
            if len(cands) == 1 and len(sigH[s]) == 1:
                status, cand = "unique_signature_match_inferred_unverified", cands[0]
            elif len(cands) == 0:
                status, cand = "no_candidate", ""
            else:
                status, cand = "ambiguous_multiple_candidates", ""
            diag_rows.append({"H_column": k, "n_drugs_flagged": int(Hv[:, k].sum()), "candidate_substructure_index": cand,
                              "candidate_smiles": sub.smiles[cand] if cand != "" else "", "status": status,
                              "identity_mapping_holds": (cand == k) if cand != "" else ""})
        dg = pd.DataFrame(diag_rows)
        dg.to_csv(out / "data/ddi" / "h_column_alignment_diagnostic.csv", index=False)
        F["h_alignment"] = {
            "rdkit_version": rdkit.__version__, "method": "BRICS decomposition of SMILES.csv structures compared with ddi_mask_H columns by drug-set signature",
            "recomputed_per_drug_fragment_counts_equal_H_row_sums": bool((R.sum(1) == Hv.sum(1)).all()),
            "all_recomputed_fragments_found_in_substructure_smiles": not_in_vocab == 0,
            "recomputed_column_sums_multiset_equals_H": bool(sorted(R.sum(0)) == sorted(Hv.sum(0))),
            "columns_uniquely_matched": int((dg.status == "unique_signature_match_inferred_unverified").sum()),
            "columns_ambiguous": int((dg.status == "ambiguous_multiple_candidates").sum()),
            "columns_where_identity_mapping_holds_among_uniquely_matched": int((dg.identity_mapping_holds == True).sum()),  # noqa: E712
            "identity_mapping_share_of_H_positive_cells_confirmed": float((R & Hv).sum() / Hv.sum()),
            "conclusion": "H column order is NOT the row order of substructure_smiles.csv; index-based join is unsafe",
        }
    except ImportError:
        F["h_alignment"] = {"skipped": "rdkit not installed (pip install rdkit)"}

    # ------------------------------------------------------------------ 6. medication mapping table
    n_struct = sm.set_index("med_index").sort_index().smiles_json.apply(lambda s: len(json.loads(s))).tolist()
    visits_with = freq.tolist()
    mp = pd.DataFrame({
        "source_medication_id": med["index"],
        "source_medication_class": med.code,
        "atc_code": med.code,
        "atc_level": 4,
        "atc_level1_anatomical_group_letter": med.code.str[0],
        "atc_level2_code": med.code.str[:3],
        "atc_level3_code": med.code.str[:4],
        "generic_name": "",
        "brand_name": "",
        "mapping_source": "voc_final.csv (med_voc); README.txt of the bundle states med codes are ATC codes",
        "mapping_status": "class_level_atc_code_only",
        "drug_level_mapping_status": "unmapped",
        "confidence_or_verification_status": "atc_code_format_verified_locally; atc_class_name_not_verified; generic_name_not_verified; not_clinically_validated",
        "requires_review": True,
        "n_source_structures_in_SMILES_csv": n_struct,
        "ddi_matrix_index": np.arange(NMED),
        "ddi_labels_match_vocab": [a == b == c for a, b, c in zip(atc, A_df.index, A_df.columns)],
        "n_recorded_ddi_partners": deg,
        "has_recorded_ddi_edges": deg > 0,
        "n_visits_with_drug": visits_with,
        "notes": "Each model class is an ATC level-4 chemical/pharmacological subgroup, not a single drug. Individual drugs, "
                 "names and brands are intentionally blank until verified against the WHO ATC index / a national formulary.",
    })
    mp.to_csv(out / "data/vocabulary_and_mapping" / "medication_vocab_mapping.csv", index=False)
    F["mapping"] = {"rows": NMED, "atc_level4_format_valid": int(med.code.str.match(r"^[A-Z]\d{2}[A-Z]{2}$").sum()),
                    "generic_names_verified": 0, "brand_names_verified": 0, "class_names_verified": 0,
                    "all_rows_require_review": True,
                    "structures_per_code_distribution": {int(k): int(v) for k, v in pd.Series(n_struct).value_counts().sort_index().items()}}

    status_codes = {
        "_label": "Status vocabulary used by integration/mediassist_safety.py. Software behaviour only; not clinical content.",
        "age_validation": {k: v for k, v in [
            ("invalid_age_input", "request_valid_age"),
            ("missing_medication_identifier", "withhold_medication_output_and_flag_for_review"),
            ("unknown_medication_identifier", "withhold_medication_output_and_flag_for_review"),
            ("unmapped_medication_identifier", "withhold_medication_output_and_flag_for_review"),
            ("unmapped_atc_level_mismatch", "withhold_medication_output_and_flag_for_review"),
            ("identifier_conflict", "withhold_medication_output_and_flag_for_review"),
            ("requires_clinician_review", "show_with_mandatory_clinician_review_notice"),
            ("reference_unavailable", "show_with_warning_age_suitability_not_assessed"),
            ("age_outside_reference_range", "flag_age_unsuitable_per_reference_and_require_review"),
            ("dose_reference_unavailable", "show_without_dose_information_and_warn"),
            ("age_check_passed_synthetic_rule", "proceed_with_standard_decision_support_disclaimer")]},
        "priority_order": ["invalid_age_input", "medication identifier failures", "reference-based outcomes"],
        "ddi": {k: v for k, v in [
            ("interaction_recorded", "show_interaction_warning_severity_not_available"),
            ("no_interaction_record_not_proof_of_safety", "show_notice_no_record_is_not_evidence_of_safety"),
            ("no_interaction_record_coverage_unverified", "show_notice_no_record_is_not_evidence_of_safety_and_drug_has_no_recorded_interactions_in_source"),
            ("unknown_medication_identifier", "withhold_and_flag_for_review"),
            ("duplicate_medication_identifier", "show_notice_duplicate_class_not_assessed")]},
    }
    jdump(status_codes, out / "data/vocabulary_and_mapping" / "safety_status_codes.json")

    # ------------------------------------------------------------------ 7. clinical reference template (schema only)
    tcols = ["medication_identifier", "atc_code", "generic_name", "patient_age_min", "patient_age_max", "age_bound_convention",
             "dose_value", "dose_unit", "dose_frequency", "route", "indication", "renal_or_hepatic_adjustment",
             "contraindications", "reference_source", "reference_version_or_date", "verification_status", "reviewer", "notes"]
    tpl = pd.DataFrame("", index=range(NMED), columns=tcols)
    tpl["medication_identifier"] = med["index"].astype(str)
    tpl["atc_code"] = med.code
    tpl["verification_status"] = "not_populated_unverified"
    tpl["notes"] = "Class-level (ATC-4) row. Do not fill with a class-wide dose: doses are drug-, route- and patient-specific."
    tpl.to_csv(out / "data/clinical_reference" / "medication_safety_reference_template.csv", index=False)

    # ------------------------------------------------------------------ 8. synthetic test data (software tests only)
    cfg = json.load(open(out / "integration" / "age_group_config.json"))
    groups = cfg["age_groups"]
    MAXA, MINA, ADULT = cfg["max_valid_age_years"], cfg["min_valid_age_years"], cfg["adult_age_years"]

    syn_ref = pd.DataFrame([
        ("SYN-REF-A", 0, 120.0000001, "Y", "N", "all valid ages allowed, dose reference present"),
        ("SYN-REF-B", 18, 120.0000001, "Y", "N", "adults only (synthetic)"),
        ("SYN-REF-C", 12, 65, "Y", "N", "12 <= age < 65 (synthetic)"),
        ("SYN-REF-D", 0, 120.0000001, "N", "N", "age allowed but dose reference missing"),
        ("SYN-REF-E", 0, 120.0000001, "Y", "Y", "always routed to clinician review"),
        ("SYN-REF-F", 0, 2, "Y", "N", "0 <= age < 2 (synthetic small-age window)"),
    ], columns=["medication_identifier", "patient_age_min", "patient_age_max_exclusive", "dose_reference_available",
                "always_require_review", "synthetic_rule_description"])
    syn_ref["rule_source_status"] = "synthetic_test_rule_not_clinical"
    syn_ref["is_real_medication"] = "N"
    syn_ref.to_csv(out / "data/synthetic_testing" / "synthetic_reference_rules.csv", index=False)

    def group_of(a):
        for g_ in groups:
            if g_["min_inclusive"] <= a < g_["max_exclusive"]:
                return g_["name"]
        return "unclassified"

    def oracle(age_raw, cid, atc_s):
        """Independent specification of expected behaviour (does not import the module under test)."""
        import math as _m, re as _re
        s = None if age_raw is None else str(age_raw).strip()
        age = None
        if s not in (None, ""):
            mm = _re.fullmatch(r">\s*(\d+)", s)
            try:
                age = float(mm.group(1)) + 1 if mm else float(s)
            except ValueError:
                age = None
            if age is not None and (_m.isnan(age) or _m.isinf(age) or age < MINA or age > MAXA):
                age = None
        if age is None:
            return "invalid_age_input", "request_valid_age", None, "unclassified"
        grp = group_of(age)
        # medication resolution
        def blank(x): return x is None or str(x).strip() == ""
        fail = "withhold_medication_output_and_flag_for_review"
        if blank(cid) and blank(atc_s):
            return "missing_medication_identifier", fail, age, grp
        rid = None
        if not blank(cid):
            t = str(cid).strip()
            if not _re.fullmatch(r"-?\d+", t) or not (0 <= int(t) < NMED):
                return "unknown_medication_identifier", fail, age, grp
            rid = int(t)
        ratc, synth = None, False
        if not blank(atc_s):
            t = str(atc_s).strip().upper()
            if t in set(syn_ref.medication_identifier):
                synth = True
                ratc = t
            elif _re.fullmatch(r"[A-Z]\d{2}[A-Z]{2}", t):
                if t not in idx_of:
                    return "unmapped_medication_identifier", fail, age, grp
                ratc = t
            elif _re.fullmatch(r"[A-Z]|[A-Z]\d{2}|[A-Z]\d{2}[A-Z]|[A-Z]\d{2}[A-Z]{2}\d{2}", t):
                return "unmapped_atc_level_mismatch", fail, age, grp
            else:
                return "unknown_medication_identifier", fail, age, grp
        if synth:
            if rid is not None:
                return "identifier_conflict", fail, age, grp
            r = syn_ref[syn_ref.medication_identifier == ratc].iloc[0]
            if r.always_require_review == "Y":
                return "requires_clinician_review", "show_with_mandatory_clinician_review_notice", age, grp
            if age < float(r.patient_age_min) or age >= float(r.patient_age_max_exclusive):
                return "age_outside_reference_range", "flag_age_unsuitable_per_reference_and_require_review", age, grp
            if r.dose_reference_available == "N":
                return "dose_reference_unavailable", "show_without_dose_information_and_warn", age, grp
            return "age_check_passed_synthetic_rule", "proceed_with_standard_decision_support_disclaimer", age, grp
        if rid is not None and ratc is not None and idx_of[ratc] != rid:
            return "identifier_conflict", fail, age, grp
        if age < ADULT:
            return "requires_clinician_review", "show_with_mandatory_clinician_review_notice", age, grp
        return "reference_unavailable", "show_with_warning_age_suitability_not_assessed", age, grp

    boundary_ages = ["0", "0.08", "0.5", "1", "2", "5", "11", "11.99", "12", "12.01", "14", "17", "17.99", "18", "18.01",
                     "30", "64", "64.99", "65", "65.01", "89", "> 89", "90", "120", "120.0"]
    invalid_ages = [("", "age missing (blank)"), ("None", "literal text 'None'"), ("abc", "non-numeric text"),
                    ("-1", "negative age"), ("-0.5", "negative fractional age"), ("120.01", "just above maximum valid age"),
                    ("200", "implausible age"), ("1e9", "absurd magnitude"), ("nan", "NaN text"), ("inf", "infinity text"),
                    ("12 years", "unit text not accepted"), ("> 200", "top-coded value beyond maximum")]
    real_ids = [atc.index(a) for a in ["R05CA", "N02AA", "A12CA", "C09AA", "A11HA"]]
    real_cat = {atc.index("A12CA"): "real_class_no_recorded_ddi_edges", atc.index("A11HA"): "real_class_no_recorded_ddi_edges"}

    cases = []

    def add(age_raw, cid, atc_s, rule, cat, note=""):
        st, act, age, grp = oracle(age_raw, cid, atc_s)
        is_syn = (atc_s or "").upper().startswith("SYN-REF")
        cases.append({
            "age_input_raw": age_raw, "age_years": "" if age is None else age, "age_group": grp,
            "medication_class_id": "" if cid is None else cid, "medication_identifier": "" if atc_s is None else atc_s,
            "test_rule_id": rule, "test_category": cat, "expected_validation_status": st, "expected_system_action": act,
            "synthetic_record": "SYNTHETIC_TEST_ONLY",
            "rule_source_status": "synthetic_test_rule_not_clinical" if (is_syn or st in ("age_outside_reference_range",)) else
                                  "software_policy_rule_not_clinical" if st in ("requires_clinician_review", "reference_unavailable") else
                                  "software_input_validation_rule", "notes": note})

    for a, note in invalid_ages:  # 1) age invalid x (real known / synthetic ref / unknown medication)
        add(a, str(real_ids[0]), atc[real_ids[0]], "AGE-001", "invalid_or_missing_age", note + " with valid real medication")
        add(a, "", "SYN-REF-A", "AGE-002", "invalid_or_missing_age", note + " with synthetic reference medication")
        add(a, "9999", "", "AGE-003", "invalid_age_takes_precedence", note + " with unknown medication (age error reported first)")
    for a in boundary_ages:  # 2) boundaries x real known meds (no reference exists -> policy outcome)
        for cid in real_ids:
            add(a, str(cid), atc[cid], "AGE-010", "age_group_boundaries_real_medication_no_reference",
                "real class; no verified reference -> reference_unavailable (adult) or requires_clinician_review (<18)")
    for a in boundary_ages:  # 3) boundaries x synthetic reference rules
        for sid, rid_ in [("SYN-REF-A", "AGE-020"), ("SYN-REF-B", "AGE-021"), ("SYN-REF-C", "AGE-022"),
                          ("SYN-REF-D", "AGE-023"), ("SYN-REF-E", "AGE-024"), ("SYN-REF-F", "AGE-025")]:
            add(a, "", sid, rid_, "synthetic_reference_age_window_and_dose_availability")
    id_cases = [  # 4) identifier states
        ("", "", "ID-001", "missing_identifier", "both identifiers blank"),
        ("107", "", "ID-002", "unknown_identifier", "class id one past vocabulary end"),
        ("-1", "", "ID-003", "unknown_identifier", "negative class id"),
        ("9999", "", "ID-004", "unknown_identifier", "class id far out of range"),
        ("abc", "", "ID-005", "unknown_identifier", "non-integer class id"),
        ("3.5", "", "ID-006", "unknown_identifier", "fractional class id"),
        ("", "Z99ZZ", "ID-010", "unmapped_identifier", "well-formed ATC-4 code not in model vocabulary"),
        ("", "R05C", "ID-011", "atc_level_mismatch", "ATC level-3 code: level mapping not guessed"),
        ("", "R05CA01", "ID-012", "atc_level_mismatch", "ATC level-5 (substance) code whose parent exists: mapping not guessed"),
        ("", "R05", "ID-013", "atc_level_mismatch", "ATC level-2 code"),
        ("", "R", "ID-014", "atc_level_mismatch", "ATC level-1 code"),
        ("", "NOTACODE", "ID-015", "unknown_identifier", "malformed string"),
        ("", "r05ca", "ID-020", "identifier_normalisation", "lower-case ATC accepted after normalisation"),
        ("", " R05CA ", "ID-021", "identifier_normalisation", "whitespace-padded ATC accepted after trimming"),
        ("0", "R05CA", "ID-030", "consistent_identifier_pair", "class id and ATC agree"),
        ("0", "A07AA", "ID-031", "identifier_conflict", "class id 0 is R05CA but ATC given is A07AA"),
        ("5", "", "ID-032", "id_is_vocabulary_index_not_atc", "index 5 resolved as vocabulary index only"),
        ("", "SYN-REF-A", "ID-040", "synthetic_identifier", "synthetic id valid only when synthetic reference loaded"),
        ("3", "SYN-REF-A", "ID-041", "identifier_conflict", "numeric class id combined with synthetic id"),
    ]
    for age_raw in ["30", "5", "70"]:
        for cid, a_, rule, cat, note in id_cases:
            add(age_raw, cid, a_, rule, cat, note)
    add("30", str(atc.index("A12CA")), "", "REF-001", "dose_reference_unavailable_real_medication",
        "no dose reference exists for any real medication in this package")
    add("16", str(atc.index("A12CA")), "", "REF-002", "minor_with_real_medication_requires_review", "software policy: minors routed to review")

    syn = pd.DataFrame(cases)
    syn.insert(0, "synthetic_case_id", [f"SYN-AGE-{i + 1:04d}" for i in range(len(syn))])
    syn.to_csv(out / "data/synthetic_testing" / "synthetic_age_validation.csv", index=False)
    F["synthetic_age"] = {"rows": int(len(syn)), "by_category": syn.test_category.value_counts().to_dict(),
                          "by_status": syn.expected_validation_status.value_counts().to_dict(),
                          "by_age_group": syn.age_group.value_counts().to_dict()}

    # synthetic DDI-handling cases (expected values derived from the ORIGINAL matrix, not invented)
    rng2 = np.random.RandomState(SEED)
    edges_arr = np.array(edges)
    pos = edges_arr[rng2.choice(len(edges_arr), 25, replace=False)]
    nz = [i for i in range(NMED) if deg[i] > 0]
    neg = []
    while len(neg) < 20:
        i, j = rng2.choice(nz, 2, replace=False)
        if A[i, j] == 0 and (i, j) not in neg and (j, i) not in neg:
            neg.append((int(i), int(j)))
    zd = [i for i in range(NMED) if deg[i] == 0]
    zero_pairs = [(zd[k % len(zd)], int(nz[k * 7 % len(nz)])) for k in range(8)]
    ddi_cases = []

    def dadd(a, b, rule, note):
        def ok(x):
            return isinstance(x, int) and 0 <= x < NMED
        if not (ok(a) and ok(b)):
            st = "unknown_medication_identifier"
        elif a == b:
            st = "duplicate_medication_identifier"
        elif A[a, b] == 1:
            st = "interaction_recorded"
        elif deg[a] == 0 or deg[b] == 0:
            st = "no_interaction_record_coverage_unverified"
        else:
            st = "no_interaction_record_not_proof_of_safety"
        act = status_codes["ddi"][st]
        ddi_cases.append({"med_a_class_id": "" if a is None else a, "med_b_class_id": "" if b is None else b,
                          "atc_a": atc[a] if ok(a) else "", "atc_b": atc[b] if ok(b) else "", "test_rule_id": rule,
                          "expected_ddi_status": st, "expected_system_action": act,
                          "expected_basis": "derived_from_original_ddi_A_matrix" if st in ("interaction_recorded", "no_interaction_record_not_proof_of_safety", "no_interaction_record_coverage_unverified") else "software_input_rule",
                          "synthetic_record": "SYNTHETIC_TEST_ONLY", "notes": note})

    for a, b in pos:
        dadd(int(a), int(b), "DDI-001", "recorded pair in source matrix")
    for a, b in neg:
        dadd(a, b, "DDI-002", "no record although both drugs have other recorded interactions: NOT proof of safety")
    for a, b in zero_pairs:
        dadd(a, b, "DDI-003", "one drug has no recorded interactions at all: coverage unverified")
    for a, b in [(107, 0), (0, -1), (None, 3), (9999, 10), (0, "x")]:
        dadd(a, b, "DDI-004", "unknown identifier handling")
    for a in [0, 3, zd[0]]:
        dadd(a, a, "DDI-005", "same class listed twice: diagonal is 0 but pair is not assessable")
    dd = pd.DataFrame(ddi_cases)
    dd.insert(0, "synthetic_case_id", [f"SYN-DDI-{i + 1:04d}" for i in range(len(dd))])
    dd.to_csv(out / "data/synthetic_testing" / "synthetic_ddi_check_cases.csv", index=False)
    F["synthetic_ddi"] = {"rows": int(len(dd)), "by_status": dd.expected_ddi_status.value_counts().to_dict()}

    # ------------------------------------------------------------------ 9. verify originals untouched
    hashes_after = {f: sha256(src / f) for f in ORIGINALS}
    F["originals_unchanged"] = hashes_before == hashes_after
    if not F["originals_unchanged"]:
        sys.exit("ERROR: original files changed during run")
    jdump(F, out / "reports" / "audit_findings.json")
    print("build complete; findings ->", out / "reports" / "audit_findings.json")
    print(json.dumps({k: F[k] for k in ["splits", "targets", "leakage", "synthetic_age", "synthetic_ddi", "h_alignment"]}, indent=1, default=str)[:3500])


if __name__ == "__main__":
    main()
