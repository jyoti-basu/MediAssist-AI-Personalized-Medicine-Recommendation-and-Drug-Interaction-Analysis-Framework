"""Leakage-aware Phase I data, modelling and decision-support helpers.

This module is intentionally a research prototype.  It returns ATC level-4
classes, never prescriptions, and treats a missing DDI record as unknown.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _lists(series: pd.Series) -> list[list[int]]:
    return [json.loads(value) if isinstance(value, str) else [] for value in series]


def _indicator(rows: list[list[int]], width: int) -> sparse.csr_matrix:
    r, c = [], []
    for i, values in enumerate(rows):
        for value in values:
            if 0 <= int(value) < width:
                r.append(i)
                c.append(int(value))
    return sparse.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(len(rows), width))


def medication_like_procedure_ids(root: Path | str = ROOT) -> frozenset[int]:
    """Return the audited medication-like procedure vocabulary IDs.

    This is the single filter definition shared by training and inference.
    """
    flags = pd.read_csv(Path(root) / "data" / "model_data" / "procedure_vocab_leakage_flags.csv")
    return frozenset(flags.loc[flags.flag_any_medication_like, "index"].astype(int))


def filter_procedure_indices(procedure_indices: list[int] | None,
                             medication_like: frozenset[int] | None = None) -> tuple[list[int], list[int]]:
    """Validate procedure IDs and remove IDs audited as medication-like.

    Invalid IDs are rejected rather than silently encoded. The returned second
    list records values excluded for leakage prevention.
    """
    values = procedure_indices or []
    medication_like = medication_like if medication_like is not None else medication_like_procedure_ids()
    clean, excluded = [], []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < 2052:
            raise ValueError("procedure indices must be integers in 0..2051")
        value = int(value)
        if value in medication_like:
            excluded.append(value)
        else:
            clean.append(value)
    return clean, excluded


@dataclass
class Phase1Data:
    inputs: pd.DataFrame
    targets: sparse.csr_matrix
    split: np.ndarray
    feature_matrix: sparse.csr_matrix
    feature_metadata: dict
    mapping: pd.DataFrame


def load_phase1_data(root: Path | str = ROOT) -> Phase1Data:
    """Load the derived audited tables and create leakage-controlled features.

    Procedure entries marked ``flag_any_medication_like`` are excluded.  This is
    more conservative than the supplied explicit-medication-segment filter and
    may remove some non-leaking clinical information; it avoids treating an
    unresolved medication-like procedure as a model input.
    """
    root = Path(root)
    data_dir = root / "data"
    inputs = pd.read_csv(data_dir / "model_data/model_inputs.csv")
    targets = sparse.load_npz(data_dir / "model_data/model_targets_multihot.npz").tocsr().astype(np.int8)
    splits = pd.read_csv(data_dir / "model_data/splits_patient_level.csv")
    mapping = pd.read_csv(data_dir / "vocabulary_and_mapping/medication_vocab_mapping.csv")

    if len(inputs) != targets.shape[0] or inputs.row_id.tolist() != list(range(len(inputs))):
        raise ValueError("model input rows do not align with multi-hot target rows")
    if mapping.source_medication_id.tolist() != list(range(targets.shape[1])):
        raise ValueError("medication mapping order does not match target columns")
    split_map = splits.set_index("patient_record_index").split
    split = inputs.patient_record_index.map(split_map).to_numpy()
    if pd.isna(split).any() or set(split) != {"train", "val", "test"}:
        raise ValueError("patient-level split is missing or invalid")

    diag = _indicator(_lists(inputs.diagnosis_idx), 2572)
    # Start from the package's explicit medication-segment-filtered field, then
    # remove all remaining heuristic medication-like procedure codes.
    medication_like = medication_like_procedure_ids(root)
    procedures = [filter_procedure_indices(values, medication_like)[0]
                  for values in _lists(inputs.procedure_idx_excluding_medications_segment)]
    proc = _indicator(procedures, 2052)

    age = inputs.age_years.fillna(-1).to_numpy()
    age_bins = np.select([age < 0, age < 18, age < 45, age < 65, age < 90], [0, 1, 2, 3, 4], default=0)
    age_block = _indicator([[int(v)] for v in age_bins], 5)
    sex_block = _indicator([[0 if str(v) == "Female" else 1 if str(v) == "Male" else 2]
                            for v in inputs.gender], 3)
    top_code = sparse.csr_matrix(inputs[["age_top_coded_gt89_flag", "gender_missing_flag", "diagnosis_empty_flag"]].to_numpy(dtype=np.float32))
    features = sparse.hstack([diag, proc, age_block, sex_block, top_code], format="csr")
    metadata = {
        "diagnosis_features": 2572,
        "procedure_features_before_conservative_filter": 2052,
        "procedure_codes_removed_as_medication_like": len(medication_like),
        "demographic_features": 11,
        "total_features": int(features.shape[1]),
        "procedure_visits_with_any_medication_like_flag": int(inputs.n_procedure_any_medication_like.gt(0).sum()),
        "feature_policy": "diagnosis + conservatively filtered procedures + age bands + gender/missing flags; excludes identifiers, offsets, undocumented field, targets and all medication-like procedure codes",
    }
    return Phase1Data(inputs, targets, split, features, metadata, mapping)


def split_indices(split: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.flatnonzero(split == name) for name in ("train", "val", "test")}


class FrequencyBaseline:
    """Train-only label-prevalence ranker; returns a fixed validation-selected k."""
    def fit(self, y: sparse.csr_matrix, top_k: int):
        self.prevalence_ = np.asarray(y.mean(axis=0)).ravel()
        self.top_k_ = int(top_k)
        return self

    def predict(self, n_rows: int) -> np.ndarray:
        out = np.zeros((n_rows, len(self.prevalence_)), dtype=np.uint8)
        out[:, np.argsort(-self.prevalence_)[: self.top_k_]] = 1
        return out


def prediction_at_k(scores: np.ndarray, top_k: int) -> np.ndarray:
    out = np.zeros(scores.shape, dtype=np.uint8)
    ix = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
    out[np.arange(len(out))[:, None], ix] = 1
    return out


def metric_report(y_true: sparse.csr_matrix | np.ndarray, prediction: np.ndarray, ddi_matrix: np.ndarray) -> dict:
    truth = y_true.toarray() if sparse.issparse(y_true) else y_true
    pairs = 0
    ddi_pairs = 0
    for row in prediction:
        selected = np.flatnonzero(row)
        if len(selected) > 1:
            sub = ddi_matrix[np.ix_(selected, selected)]
            ddi_pairs += int(np.triu(sub, k=1).sum())
            pairs += len(selected) * (len(selected) - 1) // 2
    return {
        "micro_precision": float(precision_score(truth, prediction, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(truth, prediction, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(truth, prediction, average="micro", zero_division=0)),
        "samples_jaccard": float(jaccard_score(truth, prediction, average="samples", zero_division=0)),
        "predicted_set_size": int(prediction.sum(axis=1)[0]),
        "recorded_ddi_pair_rate": float(ddi_pairs / pairs) if pairs else 0.0,
        "recorded_ddi_pairs": int(ddi_pairs),
        "all_predicted_pairs": int(pairs),
        "ddi_interpretation": "recorded source interactions only; a non-recorded pair is not proof of safety",
    }


def fit_main_model(x_train: sparse.csr_matrix, y_train: sparse.csr_matrix, alpha: float = 1e-4):
    estimator = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=100, tol=1e-3, random_state=SEED, n_jobs=None)
    model = OneVsRestClassifier(estimator, n_jobs=-1)
    return model.fit(x_train, y_train.toarray())


def scores(model, x: sparse.csr_matrix) -> np.ndarray:
    return model.predict_proba(x)


def select_top_k_on_validation(scores_val: np.ndarray, y_val: sparse.csr_matrix, ddi_matrix: np.ndarray) -> tuple[int, dict]:
    """Choose top-k by validation micro-F1, never using test labels."""
    candidates = range(1, min(16, scores_val.shape[1] + 1))
    all_results = {k: metric_report(y_val, prediction_at_k(scores_val, k), ddi_matrix) for k in candidates}
    selected = max(candidates, key=lambda k: (all_results[k]["micro_f1"], -k))
    return selected, {str(k): value for k, value in all_results.items()}


def save_artifacts(model, metadata: dict, out_dir: Path | str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "recommendation_model.joblib")
    (out / "feature_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def recommend_from_feature_row(model, feature_row: sparse.csr_matrix, mapping: pd.DataFrame, top_k: int) -> list[dict]:
    """Return ranked, class-level research candidates, never drug prescriptions."""
    row_scores = scores(model, feature_row)[0]
    order = np.argsort(-row_scores)[:top_k]
    return [{"class_id": int(i), "atc_code": str(mapping.iloc[i].atc_code),
             "model_score": float(row_scores[i]), "mapping_status": str(mapping.iloc[i].mapping_status),
             "requires_review": bool(mapping.iloc[i].requires_review)} for i in order]


def make_inference_features(diagnosis_indices: list[int], age: int | None, gender: str | None = None,
                            procedure_indices: list[int] | None = None) -> sparse.csr_matrix:
    """Build one feature row for a research-coded input.

    Free-text symptoms cannot be converted to the dataset's undocumented
    diagnosis vocabulary, so callers must supply reviewed diagnosis indices.
    Medication-like procedure entries must not be supplied here.
    """
    if any(not 0 <= int(v) < 2572 for v in diagnosis_indices):
        raise ValueError("diagnosis indices must be in 0..2571")
    procedure_indices, excluded_procedure_indices = filter_procedure_indices(procedure_indices)
    diag = _indicator([diagnosis_indices], 2572)
    proc = _indicator([procedure_indices], 2052)
    if age is None:
        age_bucket, top_code = 0, 0
    elif age < 18:
        age_bucket, top_code = 1, 0
    elif age < 45:
        age_bucket, top_code = 2, 0
    elif age < 65:
        age_bucket, top_code = 3, 0
    elif age < 90:
        age_bucket, top_code = 4, 0
    else:
        age_bucket, top_code = 0, 1
    sex = 0 if gender == "Female" else 1 if gender == "Male" else 2
    age_block = _indicator([[age_bucket]], 5)
    sex_block = _indicator([[sex]], 3)
    flags = sparse.csr_matrix([[top_code, int(gender not in {"Female", "Male"}), int(not diagnosis_indices)]], dtype=np.float32)
    features = sparse.hstack([diag, proc, age_block, sex_block, flags], format="csr")
    if excluded_procedure_indices:
        # The vector is deliberately identical to one built without those
        # inputs; callers cannot bypass the audited leakage filter.
        assert features.shape[1] == 4635
    return features
