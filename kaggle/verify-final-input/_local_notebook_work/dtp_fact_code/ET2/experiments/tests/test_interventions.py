import unittest

from data.synthetic import build_synthetic_records
from data.temporal import filter_time_valid
from evaluation.interventions import (
    inject_post_cutoff_verdict,
    inject_syndicated_copies,
    remove_earliest_source,
)
from provenance.clustering import assign_source_clusters, cluster_normalized_weights


class InterventionTest(unittest.TestCase):
    def setUp(self):
        claims, evidence = build_synthetic_records()
        self.claim = claims[0]
        self.evidence = [
            item for item in evidence if item.evidence_id in self.claim.evidence_ids
        ]

    def test_post_cutoff_verdict_is_rejected(self):
        modified = inject_post_cutoff_verdict(self.claim, self.evidence)
        valid = filter_time_valid(modified, self.claim.decision_cutoff)
        self.assertNotIn(
            f"{self.claim.claim_id}__post_cutoff_verdict",
            {item.evidence_id for item in valid},
        )

    def test_duplicate_injection_preserves_one_cluster_mass(self):
        valid = filter_time_valid(self.evidence, self.claim.decision_cutoff)
        modified = inject_syndicated_copies(valid, valid[0].evidence_id, 5)
        expected_cluster = (
            valid[0].source_cluster_id
            or f"intervention-cluster:{valid[0].evidence_id}"
        )
        self.assertTrue(
            all(
                item.source_cluster_id == expected_cluster
                for item in modified
                if "__copy_" in item.evidence_id
            )
        )
        clustered = assign_source_clusters(modified)
        weights = cluster_normalized_weights(clustered)
        cluster = clustered[0].source_cluster_id
        mass = sum(
            weights[item.evidence_id]
            for item in clustered
            if item.source_cluster_id == cluster
        )
        self.assertAlmostEqual(mass, 1.0)

    def test_remove_earliest_source(self):
        modified = remove_earliest_source(self.evidence)
        self.assertEqual(len(modified), len(self.evidence) - 1)


if __name__ == "__main__":
    unittest.main()
