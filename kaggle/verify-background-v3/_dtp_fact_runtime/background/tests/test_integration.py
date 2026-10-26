import unittest

from data.synthetic import build_synthetic_records
from data.temporal import filter_time_valid
from models.smoke import SmokeHierarchicalClassifier
from provenance.clustering import assign_source_clusters
from provenance.graph import build_provenance_graph


class IntegrationTest(unittest.TestCase):
    def test_claim_to_prediction(self):
        claims, evidence = build_synthetic_records()
        claim = claims[1]
        candidates = [item for item in evidence if item.evidence_id in claim.evidence_ids]
        clustered = assign_source_clusters(filter_time_valid(candidates, claim.decision_cutoff))
        graph = build_provenance_graph(claim, clustered)
        prediction = SmokeHierarchicalClassifier().predict(claim, clustered)
        self.assertEqual(prediction.predicted_label, "refuted")
        self.assertGreater(len(graph.nodes), 3)
        self.assertGreater(len(graph.edges), 3)


if __name__ == "__main__":
    unittest.main()

