import unittest

from data.screening import build_screening_queue, priority_signals


class ScreeningQueueTest(unittest.TestCase):
    def test_queue_balances_labels_and_is_deterministic(self):
        rows = []
        for label in ("supported", "refuted", "insufficient_evidence"):
            for index, year in enumerate((2022, 2022, 2024, 2025, 2026)):
                rows.append(
                    {
                        "candidate_id": f"{label}-{index}",
                        "suggested_label": label,
                        "api_review_date": f"{year}-01-01T00:00:00Z",
                        "api_claim_date": f"{year}-01-01T00:00:00Z",
                        "claim_text": "US photo claim",
                        "review_title": "Image fact check",
                        "textual_rating": "False",
                        "claimant": "post",
                        "collected_publisher_filter": f"publisher-{index % 2}.example",
                    }
                )
        first = build_screening_queue(rows, target_size=12, seed=9)
        second = build_screening_queue(rows, target_size=12, seed=9)
        self.assertEqual(
            [row["candidate_id"] for row in first],
            [row["candidate_id"] for row in second],
        )
        counts = {
            label: sum(row["suggested_label"] == label for row in first)
            for label in ("supported", "refuted", "insufficient_evidence")
        }
        self.assertEqual(set(counts.values()), {4})

    def test_priority_signals_are_not_decisions(self):
        score, signals = priority_signals(
            {
                "claim_text": "A photo allegedly shows the U.S. president",
                "api_review_date": "2025-01-01",
                "api_claim_date": "2025-01-01",
                "claimant": "social post",
            }
        )
        self.assertGreater(score, 0)
        self.assertIn("multimodal_term", signals)
        self.assertIn("us_term", signals)


if __name__ == "__main__":
    unittest.main()
