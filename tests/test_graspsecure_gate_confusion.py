import unittest

import numpy as np

from scripts.evaluate_graspsecure_gate_confusion import (
    confusion_metrics,
    cross_validated_logistic,
    threshold_scan,
)


class GraspSecureGateConfusionTests(unittest.TestCase):
    def test_confusion_matrix_and_conditionals(self):
        result = confusion_metrics(
            np.asarray([True, True, False, False, False]),
            np.asarray([True, False, True, False, False]),
        )
        self.assertEqual(
            {key: result[key] for key in ("TP", "FP", "FN", "TN")},
            {"TP": 1, "FP": 1, "FN": 1, "TN": 2},
        )
        self.assertAlmostEqual(
            result["precision_P_lift_success_given_gate_pass"], 0.5
        )
        self.assertAlmostEqual(result["P_lift_success_given_gate_fail"], 1 / 3)
        self.assertAlmostEqual(result["recall_sensitivity"], 0.5)
        self.assertAlmostEqual(result["specificity"], 2 / 3)

    def test_threshold_scan_retains_current_threshold(self):
        rows = threshold_scan(
            np.asarray([0.2, 0.5, 1.0, 1.5]),
            np.asarray([False, False, True, True]),
            current_threshold=1.1775,
        )
        current = [row for row in rows if row["is_current_threshold"]]
        self.assertEqual(len(current), 1)
        self.assertAlmostEqual(current[0]["threshold_rad"], 1.1775)

    def test_numpy_logistic_finds_predictive_feature(self):
        rng = np.random.default_rng(4)
        first = np.linspace(-2.0, 2.0, 60)
        second = rng.normal(0.0, 1.0, size=60)
        labels = first > 0.0
        result = cross_validated_logistic(
            np.c_[first, second], labels, ("signal", "noise"), seed=2
        )
        self.assertTrue(result["available"])
        self.assertGreater(result["out_of_fold"]["roc_auc"], 0.95)
        self.assertEqual(result["full_fit_coefficients"][0]["feature"], "signal")


if __name__ == "__main__":
    unittest.main()
