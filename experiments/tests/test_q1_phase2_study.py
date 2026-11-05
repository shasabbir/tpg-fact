from __future__ import annotations

import numpy as np
from pathlib import Path

from trace_scv.q1_phase2_study import (
    Phase2StudyConfig,
    _claim_profile,
    _evaluable_record_mask,
    _penalized_prefix,
    _retained_packet_metrics,
    _set_metrics,
    _temperature_transform,
)


def test_size_penalty_reduces_selected_prefix() -> None:
    probabilities = [0.8, 0.7, 0.6, 0.55, 0.5]
    groups = [str(index) for index in range(len(probabilities))]
    unpenalized = _penalized_prefix(
        probabilities,
        groups,
        maximum_items=5,
        size_penalty=0.0,
        redundancy_penalty=0.0,
    )
    penalized = _penalized_prefix(
        probabilities,
        groups,
        maximum_items=5,
        size_penalty=0.05,
        redundancy_penalty=0.0,
    )
    assert len(penalized) <= len(unpenalized)
    assert penalized


def test_redundancy_penalty_discourages_duplicate_source_prefix() -> None:
    probabilities = [0.8, 0.79, 0.4]
    no_penalty = _penalized_prefix(
        probabilities,
        ["same", "same", "different"],
        maximum_items=3,
        size_penalty=0.0,
        redundancy_penalty=0.0,
    )
    penalty = _penalized_prefix(
        probabilities,
        ["same", "same", "different"],
        maximum_items=3,
        size_penalty=0.0,
        redundancy_penalty=0.2,
    )
    assert len(penalty) <= len(no_penalty)


def test_temperature_transform_identity_and_bounds() -> None:
    values = np.asarray([0.1, 0.5, 0.9])
    transformed = _temperature_transform(values, 1.0)
    np.testing.assert_allclose(transformed, values, atol=1e-12)
    assert np.all((transformed > 0.0) & (transformed < 1.0))


def test_claim_profile_has_fixed_width() -> None:
    profile = _claim_profile([0.8, 0.2], ["text", "image"], 5)
    assert profile.shape == (11,)
    assert profile[0] == np.float32(0.8)


def test_set_metrics_use_complete_gold() -> None:
    metrics = _set_metrics({"a"}, {"a", "missing"})
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 2 / 3


def test_evaluable_record_mask_uses_split_qualified_complete_gold_keys() -> None:
    records = [
        {"split": "train", "claim_id": "1"},
        {"split": "validation", "claim_id": "1"},
        {"split": "train", "claim_id": "2"},
    ]
    full_gold = {
        "train:1": {"text:a"},
        "validation:1": {"image:b"},
    }
    np.testing.assert_array_equal(
        _evaluable_record_mask(records, full_gold),
        np.asarray([True, True, False]),
    )


def test_retained_packet_metrics_use_the_selector_packet_and_eligible_claims() -> None:
    records = [
        {"split": "validation", "claim_id": "1", "evidence_id": "text:a"},
        {"split": "validation", "claim_id": "1", "evidence_id": "image:b"},
        {"split": "validation", "claim_id": "2", "evidence_id": "text:x"},
    ]
    metrics = _retained_packet_metrics(
        records,
        {
            "validation:1": {"text:a", "image:missing"},
            "validation:2": {"text:missing"},
        },
    )
    assert metrics["text_recall_at_40"] == 0.5
    assert metrics["image_recall_at_20"] == 0.0
    assert metrics["complete_gold_claim_coverage"] == 0.0
    assert metrics["eligible_claims_with_zero_reachable_gold"] == 1.0


def test_corrected_protocol_drives_runtime_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    config = Phase2StudyConfig.from_protocol(
        root / "configs" / "calibration_corrected_study.yaml"
    )
    assert config.size_penalties[-1] == 0.1
    assert "platt" in config.calibrator_kinds
    assert "cross_encoder_augmented_logistic_unweighted" in config.scorer_families
