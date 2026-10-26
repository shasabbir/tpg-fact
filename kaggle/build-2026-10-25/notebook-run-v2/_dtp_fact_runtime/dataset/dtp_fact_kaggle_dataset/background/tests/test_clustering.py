import unittest

from data.synthetic import build_synthetic_records
from data.temporal import filter_time_valid
from provenance.clustering import assign_source_clusters


class ClusteringTest(unittest.TestCase):
    def test_syndicated_copy_shares_cluster(self):
        claims, evidence = build_synthetic_records()
        claim = claims[0]
        candidates = [item for item in evidence if item.evidence_id in claim.evidence_ids]
        valid = filter_time_valid(candidates, claim.decision_cutoff)
        clustered = assign_source_clusters(valid)
        self.assertEqual(len({item.source_cluster_id for item in clustered}), 1)


if __name__ == "__main__":
    unittest.main()

