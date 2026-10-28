from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from trace_scv.data.io import sha256_file


def evaluate_mocheg_development_gate(
    selector_audit_path: Path,
    verifier_audit_path: Path,
    retrieval_audit_path: Path,
    reproduction_audit_path: Path,
    test_record_path: Path,
    output_path: Path,
) -> dict[str, object]:
    selector = json.loads(selector_audit_path.read_text(encoding="utf-8"))
    verifier = json.loads(verifier_audit_path.read_text(encoding="utf-8"))
    retrieval = json.loads(retrieval_audit_path.read_text(encoding="utf-8"))
    reproduction = json.loads(reproduction_audit_path.read_text(encoding="utf-8"))
    tests = json.loads(test_record_path.read_text(encoding="utf-8"))
    selector_seeds = list(selector["seeds"])
    verifier_seeds = list(verifier["recomputed"]["seeds"])
    selector_gains = [float(row["cefes_minus_fixed_f1"]) for row in selector_seeds]
    verifier_effects = [
        float(row["validation"]["cefes"]["macro_f1"])
        - float(row["validation"]["fixed"]["macro_f1"])
        for row in verifier_seeds
    ]
    formal_flags = [
        selector["recomputed"]["formal_test_annotations_accessed"],
        verifier["recomputed"]["formal_test_annotations_accessed"],
        retrieval["recomputed"]["formal_test_annotations_accessed"],
        reproduction["formal_test_annotations_accessed"],
    ]
    conditions = {
        "1_dataset_and_release_gates": selector["status"] == "pass" and retrieval["status"] == "pass",
        "2_scientific_tests": tests.get("status") == "pass" and int(tests.get("exit_code", 1)) == 0,
        "3_mean_evidence_f1_gain_at_least_0_015": fmean(selector_gains) >= 0.015,
        "4_positive_evidence_effect_each_seed": all(value > 0.0 for value in selector_gains),
        "5_verdict_macro_f1_no_more_than_0_01_below_fixed": fmean(verifier_effects) >= -0.01,
        "6_calibrated_brier_nonworse_each_seed": all(bool(row["calibrated_brier_not_worse"]) for row in selector_seeds),
        "7_matched_artifacts_and_shared_verifier": (
            verifier["status"] == "pass"
            and verifier["recomputed"]["shared_checkpoint_each_seed"]
            and selector["feature_identity"]["sha256"]
            == json.loads((verifier_audit_path.parent / "summary.json").read_text(encoding="utf-8"))["feature_sha256"]
        ),
        "8_clean_end_to_end_reproduction": reproduction["status"] == "pass",
    }
    full_pass = all(conditions.values()) and not any(formal_flags)
    result = {
        "profile": "mocheg_section_13_development_gate",
        "status": "pass" if full_pass else "fail",
        "conditions": conditions,
        "checked_values": {
            "selector_seed_gains": selector_gains,
            "selector_mean_gain": fmean(selector_gains),
            "verifier_seed_effects": verifier_effects,
            "verifier_mean_effect": fmean(verifier_effects),
        },
        "artifact_hashes": {
            "selector_audit": sha256_file(selector_audit_path),
            "verifier_audit": sha256_file(verifier_audit_path),
            "retrieval_audit": sha256_file(retrieval_audit_path),
            "reproduction_audit": sha256_file(reproduction_audit_path),
            "test_record": sha256_file(test_record_path),
        },
        "formal_test_annotations_accessed": False,
        "formal_test_unlocked": full_pass,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
