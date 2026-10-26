import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_screening_review.py"
SPEC = importlib.util.spec_from_file_location("merge_screening_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MergeScreeningReviewTest(unittest.TestCase):
    def test_merges_only_review_fields(self):
        master = [
            {
                "candidate_id": "c1",
                "review_url": "https://example.com/1",
                "claim_text": "claim",
                "textual_rating": "False",
                "decision": "pending",
                "normalized_label": "",
            }
        ]
        queue = [
            {
                **master[0],
                "decision": "keep",
                "normalized_label": "refuted",
                "screening_rank": "1",
            }
        ]
        merged, counts = MODULE.merge_rows(master, queue)
        self.assertEqual(merged[0]["decision"], "keep")
        self.assertEqual(merged[0]["normalized_label"], "refuted")
        self.assertNotIn("screening_rank", merged[0])
        self.assertEqual(counts["changed_rows"], 1)

    def test_rejects_identity_mutation(self):
        master = [
            {
                "candidate_id": "c1",
                "review_url": "https://example.com/1",
                "claim_text": "claim",
                "textual_rating": "False",
            }
        ]
        queue = [{**master[0], "claim_text": "different"}]
        with self.assertRaisesRegex(ValueError, "Identity field"):
            MODULE.merge_rows(master, queue)


if __name__ == "__main__":
    unittest.main()
