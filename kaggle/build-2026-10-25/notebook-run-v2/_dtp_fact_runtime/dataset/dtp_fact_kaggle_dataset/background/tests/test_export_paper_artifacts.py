import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_paper_artifacts.py"
SPEC = importlib.util.spec_from_file_location("export_paper_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _metrics():
    return {
        name: {"mean": 0.5, "sample_std": 0.01}
        for name in ("macro_f1", "balanced_accuracy", "brier", "ece", "aurc")
    }


class ExportPaperArtifactsTest(unittest.TestCase):
    def test_exports_retrieval_and_model_tables_from_checked_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variants = ("majority", *MODULE.MAIN_VARIANTS[1:], *MODULE.ABLATION_VARIANTS[:-1])
            comparison = {
                "status": "checked_matrix_aggregation",
                "test_claims": 9,
                "test_events": 3,
                "evaluations": [
                    {"variant": variant, "aggregate": _metrics()} for variant in dict.fromkeys(variants)
                ],
            }
            dataset = {
                "status": "frozen_dataset",
                "claim_count": 15,
                "evidence_count": 30,
                "splits": {
                    split: {
                        "claims": 5,
                        "events": 2,
                        "labels": {"supported": 2, "refuted": 2, "insufficient_evidence": 1},
                    }
                    for split in ("train", "validation", "test")
                },
            }
            retrieval = {
                "status": "completed_frozen_retrieval",
                "diagnostic_skip_reranker": False,
                "metrics_by_split": {
                    "test": {"recall_at_10": 0.75, "ndcg_at_10": 0.70}
                },
            }
            interventions = {
                "status": "checked_intervention_evaluation",
                "diagnostic_skip_synthetic_encoding": False,
                "conflict_donor_policy": "audited_manifest",
                "aggregate": [
                    {
                        "variant": variant,
                        "scenario": scenario,
                        "macro_f1": {"mean": 0.5, "sample_std": 0.01},
                        "prediction_change_rate": {"mean": 0.1, "sample_std": 0.01},
                    }
                    for variant in ("full", "vanilla_rag")
                    for scenario in (
                        "duplicate_1",
                        "duplicate_5",
                        "duplicate_10",
                        "post_cutoff_verdict",
                        "remove_earliest_source",
                        "dependent_conflict",
                        "independent_conflict",
                        "caption_swap",
                        "historical_image_reuse",
                    )
                ],
            }
            paths = {}
            for name, value in (
                ("comparison", comparison),
                ("dataset", dataset),
                ("retrieval", retrieval),
                ("interventions", interventions),
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths[name] = path
            output = root / "tables"
            argv = [
                str(SCRIPT),
                "--comparison",
                str(paths["comparison"]),
                "--dataset-manifest",
                str(paths["dataset"]),
                "--retrieval-report",
                str(paths["retrieval"]),
                "--intervention-report",
                str(paths["interventions"]),
                "--output-directory",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                MODULE.main()
            text = (output / "generated_retrieval_results.tex").read_text(encoding="utf-8")
            self.assertIn("0.750", text)
            self.assertIn("0.700", text)
            self.assertTrue((output / "generated_main_results.tex").exists())
            self.assertTrue((output / "generated_intervention_results.tex").exists())


if __name__ == "__main__":
    unittest.main()
