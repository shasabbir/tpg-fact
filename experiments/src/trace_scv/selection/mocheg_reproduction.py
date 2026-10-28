from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trace_scv.data.io import sha256_file


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _numeric_close(left: object, right: object, *, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numeric_close(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        ignored = {"runtime", "artifacts", "path", "bytes", "sha256"}
        keys = (set(left) | set(right)) - ignored
        return all(
            key in left and key in right and _numeric_close(left[key], right[key], tolerance=tolerance)
            for key in keys
        )
    return left == right


def compare_mocheg_clean_reproduction(
    reference_feature: Path,
    rerun_feature: Path,
    reference_selector: Path,
    rerun_selector: Path,
    reference_verifier: Path,
    rerun_verifier: Path,
    output_path: Path,
    *,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    discrepancies: list[str] = []
    reference_arrays = np.load(reference_feature, allow_pickle=False)
    rerun_arrays = np.load(rerun_feature, allow_pickle=False)
    if not np.array_equal(reference_arrays["gold"], rerun_arrays["gold"]):
        discrepancies.append("feature gold arrays differ")
    if reference_arrays["features"].shape != rerun_arrays["features"].shape or not np.allclose(
        reference_arrays["features"], rerun_arrays["features"], rtol=tolerance, atol=tolerance
    ):
        discrepancies.append("feature arrays differ beyond tolerance")

    reference_sidecar = json.loads(reference_feature.with_suffix(".json").read_text(encoding="utf-8"))
    rerun_sidecar = json.loads(rerun_feature.with_suffix(".json").read_text(encoding="utf-8"))
    if reference_sidecar["records"] != rerun_sidecar["records"]:
        discrepancies.append("feature record identities differ")
    if reference_sidecar["full_gold_evidence_ids_by_claim"] != rerun_sidecar["full_gold_evidence_ids_by_claim"]:
        discrepancies.append("full gold maps differ")
    reference_rankings = _read_jsonl(reference_feature.with_suffix(".rankings.jsonl"))
    rerun_rankings = _read_jsonl(rerun_feature.with_suffix(".rankings.jsonl"))
    if len(reference_rankings) != len(rerun_rankings):
        discrepancies.append("ranking record counts differ")
    else:
        for reference, rerun in zip(reference_rankings, rerun_rankings, strict=True):
            reference_ids = [row["evidence_id"] for row in reference["ranking"]]
            rerun_ids = [row["evidence_id"] for row in rerun["ranking"]]
            reference_scores = [float(row["score"]) for row in reference["ranking"]]
            rerun_scores = [float(row["score"]) for row in rerun["ranking"]]
            if reference_ids != rerun_ids or not np.allclose(reference_scores, rerun_scores, rtol=tolerance, atol=tolerance):
                discrepancies.append(
                    f"ranking differs for {reference['split']}:{reference['claim_id']}:{reference['modality']}"
                )
                break

    reference_selector_summary = json.loads((reference_selector / "summary.json").read_text(encoding="utf-8"))
    rerun_selector_summary = json.loads((rerun_selector / "summary.json").read_text(encoding="utf-8"))
    if not _numeric_close(reference_selector_summary, rerun_selector_summary, tolerance=tolerance):
        discrepancies.append("selector development table differs")
    reference_verifier_summary = json.loads((reference_verifier / "summary.json").read_text(encoding="utf-8"))
    rerun_verifier_summary = json.loads((rerun_verifier / "summary.json").read_text(encoding="utf-8"))
    if not _numeric_close(reference_verifier_summary, rerun_verifier_summary, tolerance=tolerance):
        discrepancies.append("verifier development table differs")
    formal_flags = [
        reference_selector_summary["formal_test_annotations_accessed"],
        rerun_selector_summary["formal_test_annotations_accessed"],
        reference_verifier_summary["formal_test_annotations_accessed"],
        rerun_verifier_summary["formal_test_annotations_accessed"],
    ]
    if any(formal_flags):
        discrepancies.append("a reproduction artifact reports formal-test access")
    result = {
        "profile": "mocheg_clean_development_reproduction",
        "status": "pass" if not discrepancies else "fail",
        "tolerance": tolerance,
        "discrepancies": discrepancies,
        "reference": {
            "features_sha256": sha256_file(reference_feature),
            "selector_summary_sha256": sha256_file(reference_selector / "summary.json"),
            "verifier_summary_sha256": sha256_file(reference_verifier / "summary.json"),
        },
        "rerun": {
            "features_sha256": sha256_file(rerun_feature),
            "selector_summary_sha256": sha256_file(rerun_selector / "summary.json"),
            "verifier_summary_sha256": sha256_file(rerun_verifier / "summary.json"),
        },
        "formal_test_annotations_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
