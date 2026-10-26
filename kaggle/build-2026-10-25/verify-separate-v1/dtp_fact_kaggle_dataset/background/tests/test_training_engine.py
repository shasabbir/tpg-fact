import importlib.util
import tempfile
import unittest

import numpy as np


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch optional extra not installed")
class TrainingEngineTest(unittest.TestCase):
    def test_hierarchical_probabilities_sum_to_one(self):
        import torch

        from training.engine import hierarchical_log_probabilities

        outputs = {
            "sufficiency_logit": torch.tensor([0.0, 1.0]),
            "veracity_logits": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        probabilities = hierarchical_log_probabilities(outputs).exp().numpy()
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)

    def test_cpu_train_checkpoint_calibrate_predict_cycle(self):
        from data.synthetic import build_synthetic_records
        from training.dataset import EmbeddingStore, build_examples
        from training.engine import TrainingSettings, predict_from_checkpoint, train_model

        claims, evidence = build_synthetic_records()
        text = {}
        for index, claim in enumerate(claims):
            text[claim.claim_id] = np.full(6, index + 1, dtype=np.float32)
        for index, item in enumerate(evidence):
            text[item.evidence_id] = np.full(6, (index % 5) + 1, dtype=np.float32)
        image = {
            claim.claim_id: np.full(4, index + 1, dtype=np.float32)
            for index, claim in enumerate(claims)
        }
        store = EmbeddingStore(text=text, image=image, text_dim=6, image_dim=4)
        examples = build_examples(claims, evidence, store)
        settings = TrainingSettings(
            variant="provenance_pool",
            seed=7,
            hidden_dimension=8,
            batch_size=3,
            epochs=2,
            early_stopping_patience=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = train_model(
                examples,
                examples,
                examples,
                settings,
                directory,
                device_name="cpu",
            )
            self.assertEqual(result["status"], "completed_training_run")
            self.assertEqual(result["counts"]["test"], len(examples))
            with open(f"{directory}/predictions.jsonl", encoding="utf-8") as stream:
                predictions = [line for line in stream if line.strip()]
            self.assertEqual(len(predictions), len(examples))
            repeated = predict_from_checkpoint(
                examples,
                settings,
                result["checkpoint"],
                temperature=float(result["temperature"]),
                abstention_threshold=float(result["abstention_threshold"]),
                device_name="cpu",
            )
            self.assertEqual(len(repeated), len(examples))


if __name__ == "__main__":
    unittest.main()
