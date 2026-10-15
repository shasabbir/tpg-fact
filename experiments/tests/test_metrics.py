import unittest

import numpy as np

from evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    macro_ovr_auroc,
    multiclass_brier,
)
from evaluation.bootstrap import event_bootstrap_ci
from evaluation.retrieval import ndcg_at_k, recall_at_k
from training.calibration import apply_temperature, select_threshold


class MetricTest(unittest.TestCase):
    def test_perfect_classification(self):
        labels = ["supported", "refuted", "insufficient_evidence"]
        metrics = classification_metrics(labels, labels)
        self.assertEqual(metrics["macro_f1"], 1.0)
        probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        self.assertGreater(multiclass_brier(labels, probabilities), 0.0)
        self.assertAlmostEqual(expected_calibration_error(labels, probabilities, bins=5), 0.2)
        self.assertEqual(macro_ovr_auroc(labels, probabilities), 1.0)

    def test_event_bootstrap_preserves_perfect_score(self):
        labels = ["supported", "refuted", "insufficient_evidence"]
        interval = event_bootstrap_ci(
            labels,
            labels,
            ["event-a", "event-b", "event-c"],
            lambda truth, prediction: classification_metrics(truth, prediction)["macro_f1"],
            replicates=100,
            seed=7,
        )
        self.assertEqual(interval["estimate"], 1.0)
        self.assertGreaterEqual(interval["ci_low"], 1.0 / 3.0)
        self.assertEqual(interval["ci_high"], 1.0)

    def test_retrieval_metrics(self):
        ranking = ["a", "b", "c"]
        self.assertEqual(recall_at_k(ranking, {"a", "c"}, 2), 0.5)
        self.assertGreater(ndcg_at_k(ranking, {"a", "c"}, 3), 0.8)

    def test_calibration_utilities(self):
        probabilities = apply_temperature(np.array([[2.0, 0.0, -1.0]]), 1.0)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        threshold = select_threshold(np.array([0.1, 0.4, 0.8, 0.9, 1.0]), 0.8)
        self.assertEqual(threshold, 0.1)


if __name__ == "__main__":
    unittest.main()

