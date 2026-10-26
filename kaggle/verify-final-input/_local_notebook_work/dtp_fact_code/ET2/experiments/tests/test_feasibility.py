import unittest

from data.feasibility import audit_feasibility
from data.synthetic import build_synthetic_records
from provenance.clustering import assign_source_clusters


class FeasibilityAuditTest(unittest.TestCase):
    def test_synthetic_fixture_triggers_fallback(self):
        claims, evidence = build_synthetic_records()
        clustered = assign_source_clusters(evidence)
        report = audit_feasibility(claims, clustered)
        self.assertEqual(report["status"], "stress_test_fallback")
        self.assertFalse(report["checks"]["total_cases"])


if __name__ == "__main__":
    unittest.main()
