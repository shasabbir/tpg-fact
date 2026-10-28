from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trace_scv.data.io import sha256_file
from trace_scv.selection.mocheg_audit import audit_mocheg_selector_pilot
from trace_scv.selection.mocheg_study import (
    LogisticEvidenceScorer,
    MochegSelectorConfig,
    _bounded_candidate_indices,
    _claim_partition,
    _evaluate_claims,
    _paired_bootstrap,
    run_mocheg_selector_pilot,
)


def test_logistic_evidence_scorer_learns_separable_rows() -> None:
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scorer = LogisticEvidenceScorer.fit(
        features, labels, learning_rate=0.1, epochs=1000, l2=1e-4
    )
    probabilities = scorer.predict_probabilities(features)

    assert probabilities[0] < probabilities[1] < probabilities[2] < probabilities[3]
    assert probabilities[0] < 0.5 < probabilities[-1]


def test_claim_partition_is_deterministic_and_disjoint() -> None:
    claims = [str(index) for index in range(20)]
    first = _claim_partition(claims, 13, 0.2)
    second = _claim_partition(reversed(claims), 13, 0.2)

    assert first == second
    assert len(first) == 4
    assert first < set(claims)


def test_cefes_and_fixed_use_same_candidate_records() -> None:
    records = [
        {"claim_id": "1", "evidence_id": "a", "gold": True},
        {"claim_id": "1", "evidence_id": "b", "gold": False},
        {"claim_id": "2", "evidence_id": "c", "gold": True},
        {"claim_id": "2", "evidence_id": "d", "gold": True},
    ]
    probabilities = [0.9, 0.1, 0.8, 0.7]
    fixed, fixed_rows = _evaluate_claims(
        records, probabilities, method="fixed", fixed_k=1
    )
    cefes, cefes_rows = _evaluate_claims(
        records, probabilities, method="cefes", maximum_selected_items=2
    )

    assert fixed["claims"] == cefes["claims"] == 2.0
    assert [row["claim_id"] for row in fixed_rows] == [
        row["claim_id"] for row in cefes_rows
    ]
    assert cefes["f1"] >= fixed["f1"]


def test_candidate_bound_uses_similarity_not_gold() -> None:
    records = [
        {"claim_id": "1", "evidence_id": "a", "modality": "text", "gold": True},
        {"claim_id": "1", "evidence_id": "b", "modality": "text", "gold": False},
        {"claim_id": "1", "evidence_id": "c", "modality": "image", "gold": True},
    ]
    features = np.asarray([[0.1], [0.9], [0.5]], dtype=np.float64)

    selected = _bounded_candidate_indices(
        records,
        features,
        maximum_text_candidates=1,
        maximum_image_candidates=1,
    )

    assert selected == [1, 2]


def test_bootstrap_preserves_samples_and_resampling_identity() -> None:
    left = [
        {"claim_id": "1", "f1": 1.0},
        {"claim_id": "2", "f1": 0.5},
    ]
    right = [
        {"claim_id": "1", "f1": 0.0},
        {"claim_id": "2", "f1": 0.5},
    ]

    result = _paired_bootstrap(left, right, samples=17, seed=37)

    assert result["sample_count"] == 17
    assert result["resampling_seed"] == 37
    assert result["resampling_unit"] == "claim"
    assert len(result["samples"]) == 17


def test_selector_pilot_preserves_per_candidate_probabilities(tmp_path: Path) -> None:
    records = []
    features = []
    labels = []
    for split, claim_count in (("train", 10), ("validation", 2)):
        for claim_number in range(claim_count):
            claim_id = f"{split}-{claim_number}"
            for candidate_number, gold in enumerate((True, False)):
                records.append(
                    {
                        "claim_id": claim_id,
                        "evidence_id": f"{claim_id}-{candidate_number}",
                        "modality": "text",
                        "split": split,
                        "gold": gold,
                    }
                )
                features.append([0.9 if gold else 0.1])
                labels.append(int(gold))

    feature_path = tmp_path / "features.npz"
    np.savez_compressed(
        feature_path,
        features=np.asarray(features, dtype=np.float32),
        gold=np.asarray(labels, dtype=np.int8),
    )
    sidecar_path = feature_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            {"records": records, "formal_test_annotations_accessed": False}
        ),
        encoding="utf-8",
    )
    feature_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "rows": len(records),
                "claims": 12,
                "features_sha256": sha256_file(feature_path),
                "sidecar_sha256": sha256_file(sidecar_path),
                "scientific_results_allowed": True,
                "formal_test_annotations_accessed": False,
                "encoder": {"name": "synthetic"},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "pilot"

    summary = run_mocheg_selector_pilot(
        feature_path,
        output_dir,
        config=MochegSelectorConfig(
            seeds=(13,),
            calibration_fraction=0.5,
            fixed_budgets=(1, 2),
            maximum_text_candidates=2,
            maximum_image_candidates=0,
            maximum_selected_items=2,
            epochs=25,
            bootstrap_samples=11,
        ),
    )

    candidate_path = output_dir / "seed_13_candidate_records.jsonl"
    candidate_rows = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(candidate_rows) == len(records)
    assert all("raw_probability" in row for row in candidate_rows)
    assert all("calibrated_probability" in row for row in candidate_rows)
    assert {row["role"] for row in candidate_rows} == {
        "fit",
        "calibration",
        "validation",
    }
    assert all(
        row["calibrated_rank"] in {1, 2}
        for row in candidate_rows
        if row["role"] == "validation"
    )
    seed = summary["seeds"][0]
    assert seed["artifacts"]["candidate_records"]["sha256"]
    assert len(seed["validation"]["cefes_minus_fixed_bootstrap"]["samples"]) == 11
    audit = audit_mocheg_selector_pilot(feature_path, output_dir)
    assert audit["status"] == "pass"
    assert audit["recomputed"]["selector_numeric_criteria_met"] is False


def test_calibration_safety_guard_records_train_only_decision(tmp_path: Path) -> None:
    records = []
    features = []
    labels = []
    for split, claim_count in (("train", 20), ("validation", 2)):
        for claim_number in range(claim_count):
            claim_id = f"{split}-{claim_number}"
            for candidate_number, gold in enumerate((True, False)):
                records.append(
                    {
                        "claim_id": claim_id,
                        "evidence_id": f"{claim_id}-{candidate_number}",
                        "modality": "text",
                        "split": split,
                        "gold": gold,
                    }
                )
                features.append([0.8 if gold else 0.2])
                labels.append(int(gold))

    feature_path = tmp_path / "features.npz"
    np.savez_compressed(
        feature_path,
        features=np.asarray(features, dtype=np.float32),
        gold=np.asarray(labels, dtype=np.int8),
    )
    feature_path.with_suffix(".json").write_text(
        json.dumps({"records": records}), encoding="utf-8"
    )
    feature_path.with_suffix(".manifest.json").write_text(
        json.dumps({"scientific_results_allowed": True}), encoding="utf-8"
    )

    result = run_mocheg_selector_pilot(
        feature_path,
        tmp_path / "pilot",
        config=MochegSelectorConfig(
            seeds=(13,),
            calibration_fraction=0.4,
            fixed_budgets=(1, 2),
            maximum_text_candidates=2,
            maximum_image_candidates=0,
            maximum_selected_items=2,
            epochs=25,
            bootstrap_samples=5,
            calibration_safety_guard=True,
            calibration_guard_fraction=0.5,
        ),
    )

    guard = result["seeds"][0]["calibration_guard"]
    assert guard["strategy"] == "claim_grouped_temperature_or_identity_by_guard_brier"
    assert guard["temperature_fit_claims"] == 4
    assert guard["guard_claims"] == 4
    assert guard["temperature_fit_examples"] == 8
    assert guard["guard_examples"] == 8
    assert guard["applied_temperature"] in {1.0, guard["candidate_temperature"]}
