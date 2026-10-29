import sys
import unittest
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
from patient_nlp import extract_patient_information, validate_patient_information
from phase1_core import (filter_procedure_indices, fit_main_model, load_phase1_data,
                         make_inference_features, metric_report, prediction_at_k,
                         recommend_from_feature_row, scores, split_indices)
from mediassist_safety import DDIChecker, MedicationRegistry, validate_age_and_reference
from run_end_to_end_demo import build_decision_support_report


class Phase1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_phase1_data(ROOT)

    def test_dataset_shape_and_patient_split(self):
        self.assertEqual(self.data.feature_matrix.shape, (22988, 4635))
        self.assertEqual(self.data.targets.shape, (22988, 107))
        split = split_indices(self.data.split)
        patient_sets = [set(self.data.inputs.iloc[i].patient_record_index) for i in split.values()]
        self.assertFalse(patient_sets[0] & patient_sets[1])
        self.assertFalse(patient_sets[0] & patient_sets[2])
        self.assertFalse(patient_sets[1] & patient_sets[2])

    def test_mapping_and_ddi_alignment(self):
        self.assertEqual(self.data.mapping.source_medication_id.tolist(), list(range(107)))
        self.assertEqual(self.data.mapping.ddi_matrix_index.tolist(), list(range(107)))
        checker = DDIChecker()
        self.assertEqual(checker.A.shape, (107, 107))
        self.assertTrue(np.array_equal(checker.A, checker.A.T))
        self.assertIn(checker.check_pair(0, 1)["status"], {
            "interaction_recorded", "no_interaction_record_not_proof_of_safety",
            "no_interaction_record_coverage_unverified",
        })

    def test_conservative_leakage_filter_and_inference_shape(self):
        self.assertEqual(self.data.feature_metadata["procedure_codes_removed_as_medication_like"], 839)
        row = make_inference_features([0, 1, 2], 61)
        self.assertEqual(row.shape, (1, self.data.feature_matrix.shape[1]))
        with self.assertRaises(ValueError):
            make_inference_features([9999], 61)

    def test_inference_cannot_bypass_medication_like_procedure_filter(self):
        medication_like_id = 0
        clean, excluded = filter_procedure_indices([medication_like_id])
        self.assertEqual(clean, [])
        self.assertEqual(excluded, [medication_like_id])
        without_procedure = make_inference_features([0, 1, 2], 61)
        with_medication_like_procedure = make_inference_features([0, 1, 2], 61, procedure_indices=[medication_like_id])
        self.assertEqual((without_procedure != with_medication_like_procedure).nnz, 0)
        with self.assertRaises(ValueError):
            make_inference_features([0], 61, procedure_indices=[2052])

    def test_fresh_model_training_smoke_test(self):
        """Small patient-disjoint real-data smoke test; not official evaluation."""
        split = split_indices(self.data.split)
        # Retain at least one training occurrence of every observed label so
        # the tiny smoke fit does not reduce rare labels to constant negatives.
        required_rows = set()
        for label in range(self.data.targets.shape[1]):
            positive = split["train"][self.data.targets[split["train"], label].nonzero()[0]]
            self.assertGreater(len(positive), 0, f"label {label} absent from official training split")
            required_rows.add(int(positive[0]))
        train_rows = np.array(sorted(set(split["train"][:256]) | required_rows))
        eval_rows = split["val"][:128]
        self.assertFalse(set(self.data.inputs.iloc[train_rows].patient_record_index) &
                         set(self.data.inputs.iloc[eval_rows].patient_record_index))
        model = fit_main_model(self.data.feature_matrix[train_rows], self.data.targets[train_rows], alpha=1e-4)
        probabilities = scores(model, self.data.feature_matrix[eval_rows])
        prediction = prediction_at_k(probabilities, 5)
        ddi = np.load(ROOT / "data" / "ddi" / "ddi_matrix.npy")
        metrics = metric_report(self.data.targets[eval_rows], prediction, ddi)
        self.assertEqual(probabilities.shape, (len(eval_rows), 107))
        self.assertEqual(prediction.shape, (len(eval_rows), 107))
        self.assertTrue(0.0 <= metrics["micro_f1"] <= 1.0)

    def test_prediction_shape(self):
        pred = prediction_at_k(np.array([[0.1, 0.9, 0.2], [0.4, 0.3, 0.2]]), 2)
        self.assertEqual(pred.shape, (2, 3))
        self.assertTrue(np.all(pred.sum(axis=1) == 2))

    def test_saved_model_and_end_to_end_components(self):
        model = joblib.load(ROOT / "models" / "recommendation_model.joblib")
        config = __import__("json").loads((ROOT / "models" / "feature_metadata.json").read_text())
        candidates = recommend_from_feature_row(model, make_inference_features([0, 1, 2], 61),
                                                self.data.mapping, config["top_k"])
        self.assertEqual(len(candidates), config["top_k"])
        self.assertTrue(all(c["mapping_status"] == "class_level_atc_code_only" for c in candidates))
        self.assertTrue(all(c["requires_review"] for c in candidates))
        report = build_decision_support_report()
        self.assertEqual(report["decision_support_status"]["status"], "research_candidates_require_review")
        self.assertTrue(report["decision_support_status"]["clinical_action_withheld"])

    def test_age_and_nlp_missing_information(self):
        self.assertEqual(validate_age_and_reference(None, class_id=0)["status"], "invalid_age_input")
        patient = extract_patient_information("Symptoms: cough.")
        result = validate_patient_information(patient, set(self.data.mapping.atc_code))
        self.assertIn("age", result["missing_fields"])
        self.assertIn("research_diagnosis_indices", result["missing_fields"])
        self.assertFalse(result["is_complete_for_demo"])
        self.assertEqual(MedicationRegistry().resolve(atc="NOT-A-CODE")["status"], "unknown_medication_identifier")


if __name__ == "__main__":
    unittest.main(verbosity=2)
