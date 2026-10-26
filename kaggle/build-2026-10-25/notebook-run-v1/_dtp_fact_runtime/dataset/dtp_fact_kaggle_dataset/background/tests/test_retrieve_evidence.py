import importlib.util
from pathlib import Path
import unittest

import numpy as np

from data.synthetic import build_synthetic_records
from training.dataset import EmbeddingStore


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retrieve_evidence.py"
SPEC = importlib.util.spec_from_file_location("retrieve_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RetrieveEvidenceTest(unittest.TestCase):
    def test_rank_claim_filters_post_cutoff_and_limits_output(self):
        claims, evidence = build_synthetic_records()
        claim = claims[0]
        text = {
            claim.claim_id: np.asarray([1.0, 0.0], dtype=np.float32),
            **{
                item.evidence_id: np.asarray([1.0, index / 10], dtype=np.float32)
                for index, item in enumerate(evidence)
            },
        }
        store = EmbeddingStore(text=text, image={}, text_dim=2, image_dim=0)
        selected, log = MODULE._rank_claim(
            claim,
            evidence,
            store,
            {item.evidence_id: item.title for item in evidence},
            rrf_k=60,
            candidate_limit=100,
            retained_limit=2,
            reranker=None,
            reranker_batch_size=2,
        )
        evidence_by_id = {item.evidence_id: item for item in evidence}
        self.assertLessEqual(len(selected), 2)
        self.assertTrue(
            all(
                MODULE.is_time_valid(evidence_by_id[item_id], claim.decision_cutoff)
                for item_id in selected
            )
        )
        self.assertEqual(selected, log["selected_evidence_ids"])


if __name__ == "__main__":
    unittest.main()
