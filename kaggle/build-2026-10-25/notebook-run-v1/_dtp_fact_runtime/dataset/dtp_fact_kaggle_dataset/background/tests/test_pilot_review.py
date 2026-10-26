import tempfile
import unittest
from pathlib import Path

from data.pilot_review import (
    candidate_to_review_row,
    read_review_csv,
    validate_review_rows,
    write_review_csv,
)


class PilotReviewTest(unittest.TestCase):
    def test_csv_round_trip_and_strict_keep_validation(self):
        candidate = {
            "candidate_id": "candidate-1",
            "claim_text": "Example claim",
            "review_url": "https://example.org/check",
            "textual_rating": "False",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            write_review_csv(path, [candidate])
            rows = read_review_csv(path)
        self.assertEqual(rows[0]["decision"], "pending")
        self.assertEqual(rows[0]["suggested_label"], "refuted")

        kept = candidate_to_review_row(candidate)
        kept.update(
            {
                "decision": "keep",
                "us_relevant": "yes",
                "multimodal": "yes",
                "target_image_verified": "yes",
                "rating_verified": "yes",
                "claim_image_url": "https://example.org/image.jpg",
                "claim_time": "2025-01-01T00:00:00Z",
                "decision_cutoff": "2025-01-02T00:00:00Z",
                "normalized_label": "refuted",
                "event_id": "event-1",
            }
        )
        report = validate_review_rows([kept])
        self.assertEqual(report["status"], "valid")

    def test_rejected_row_requires_reason(self):
        row = candidate_to_review_row({"candidate_id": "candidate-1"})
        row["decision"] = "reject"
        report = validate_review_rows([row])
        self.assertEqual(report["status"], "needs_correction")


if __name__ == "__main__":
    unittest.main()
