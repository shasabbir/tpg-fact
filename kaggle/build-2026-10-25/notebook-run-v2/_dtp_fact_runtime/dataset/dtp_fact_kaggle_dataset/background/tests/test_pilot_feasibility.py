import unittest

from data.pilot_feasibility import audit_pilot
from data.synthetic import build_synthetic_records


class PilotFeasibilityTest(unittest.TestCase):
    def test_pilot_report_is_not_final_benchmark_gate(self):
        claims, evidence = build_synthetic_records()
        rows = [
            {"decision": "keep", "exclusion_reason": ""}
            for _ in claims
        ] + [{"decision": "reject", "exclusion_reason": "TEXT_ONLY"}]
        report = audit_pilot(
            rows,
            claims,
            evidence,
            target_cases=len(claims),
            final_target_cases=2000,
        )
        self.assertIn("pilot", report["report_scope"])
        self.assertEqual(report["screening"]["decisions"]["keep"], len(claims))
        self.assertGreater(report["screening"]["projected_candidates_for_final_target"], 2000)


if __name__ == "__main__":
    unittest.main()
