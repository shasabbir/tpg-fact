import unittest
from datetime import timedelta

from data.synthetic import build_synthetic_records
from data.temporal import filter_time_valid


class TemporalFilterTest(unittest.TestCase):
    def test_removes_post_cutoff_evidence(self):
        claims, evidence = build_synthetic_records()
        claim = claims[0]
        candidates = [item for item in evidence if item.evidence_id in claim.evidence_ids]
        valid = filter_time_valid(candidates, claim.decision_cutoff)
        self.assertEqual(len(valid), 2)
        self.assertNotIn(f"e-{claim.claim_id}-late", {item.evidence_id for item in valid})

    def test_rejects_invalid_archive_order(self):
        claims, evidence = build_synthetic_records()
        claim = claims[0]
        item = evidence[0]
        altered = item.__class__(
            **{
                **item.to_dict(),
                "archived_at": item.published_at - timedelta(hours=1),
            }
        )
        self.assertEqual(filter_time_valid([altered], claim.decision_cutoff), [])


if __name__ == "__main__":
    unittest.main()

