import unittest

import numpy as np

from data.synthetic import build_synthetic_records
from training.dataset import EmbeddingStore, RELATIONS, build_model_example


class TrainingDatasetTest(unittest.TestCase):
    def test_builds_graph_and_matched_baseline_features(self):
        claims, evidence = build_synthetic_records()
        claim = claims[0]
        identifiers = [claim.claim_id, *claim.evidence_ids]
        text = {
            identifier: np.full(4, index + 1, dtype=np.float32)
            for index, identifier in enumerate(identifiers)
        }
        image = {claim.claim_id: np.ones(3, dtype=np.float32)}
        store = EmbeddingStore(text=text, image=image, text_dim=4, image_dim=3)
        example = build_model_example(claim, evidence, store)

        self.assertEqual(example.node_features.shape[1], store.common_dim)
        self.assertEqual(example.edge_index.shape[0], 2)
        self.assertEqual(len(example.edge_types), example.edge_index.shape[1])
        self.assertGreater(len(RELATIONS), 8)
        self.assertEqual(example.baseline_features["text_only"].shape, (4,))
        self.assertEqual(example.baseline_features["image_text"].shape, (7,))
        self.assertEqual(
            example.baseline_features["provenance_pool"].shape,
            (7 + store.common_dim,),
        )

    def test_no_provenance_separates_dependency_clusters(self):
        claims, evidence = build_synthetic_records()
        claim = claims[0]
        text = {
            identifier: np.ones(2, dtype=np.float32)
            for identifier in [claim.claim_id, *claim.evidence_ids]
        }
        store = EmbeddingStore(text=text, image={}, text_dim=2, image_dim=0)
        with_provenance = build_model_example(claim, evidence, store, use_provenance=True)
        without_provenance = build_model_example(claim, evidence, store, use_provenance=False)
        self.assertLess(
            len({item.source_cluster_id for item in with_provenance.evidence}),
            len({item.source_cluster_id for item in without_provenance.evidence}),
        )


if __name__ == "__main__":
    unittest.main()
