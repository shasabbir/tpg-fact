import importlib.util
from pathlib import Path
import unittest

from data.synthetic import build_synthetic_records


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_interventions.py"
SPEC = importlib.util.spec_from_file_location("evaluate_interventions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvaluateInterventionsTest(unittest.TestCase):
    def test_builds_every_preregistered_scenario(self):
        claims, evidence = build_synthetic_records()
        evidence_by_id = {item.evidence_id: item for item in evidence}
        scenario_claims, scenario_evidence, aliases, image_aliases = MODULE._scenario_records(
            claims, claims, evidence_by_id
        )
        self.assertEqual(set(scenario_claims), set(MODULE.SCENARIOS))
        self.assertEqual(set(scenario_evidence), set(MODULE.SCENARIOS))
        self.assertEqual(len(scenario_claims["caption_swap"]), len(claims))
        self.assertTrue(any("dependent_conflict" in key for key in aliases))
        self.assertTrue(any(name == "caption_swap" for name, _ in image_aliases))


if __name__ == "__main__":
    unittest.main()
