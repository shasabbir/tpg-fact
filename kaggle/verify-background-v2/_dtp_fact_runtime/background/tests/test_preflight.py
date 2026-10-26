import tempfile
import unittest
from pathlib import Path

from training.preflight import inspect_full_run_readiness


class FullRunPreflightTest(unittest.TestCase):
    def test_missing_data_and_trainer_are_blockers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs" / "base.yaml").write_text(
                """
data:
  audit_sample_size: 200
retrieval:
  dense_text_revision: null
  reranker_revision: null
  visual_model_revision: null
model:
  comparison_mllm_revision: null
""".strip(),
                encoding="utf-8",
            )
            report = inspect_full_run_readiness(root, api_status="error")
        self.assertEqual(report["status"], "not_ready")
        self.assertIn("200-case pilot claim manifest is incomplete", report["blockers"])
        self.assertIn(
            "audited semantic-conflict donor manifest is missing or invalid",
            report["blockers"],
        )
        self.assertTrue(
            any("final-run scripts are missing" in blocker for blocker in report["blockers"])
        )


if __name__ == "__main__":
    unittest.main()
