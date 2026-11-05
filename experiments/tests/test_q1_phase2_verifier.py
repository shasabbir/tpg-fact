from __future__ import annotations

import numpy as np

from trace_scv.q1_phase2_verifier import (
    _aggregate,
    _class_metrics,
    _seed_aligned_paired_bootstrap,
)


def test_verifier_class_metrics_are_macro_averaged() -> None:
    metrics = _class_metrics(
        ["nei", "refuted", "supported"],
        ["nei", "refuted", "refuted"],
    )
    assert metrics["accuracy"] == 2 / 3
    assert 0.0 < metrics["macro_f1"] < 1.0
    assert metrics["nei_f1"] == 1.0


def test_aggregate_has_fixed_width() -> None:
    rows = [
        {
            "split": "train",
            "claim_id": "1",
            "evidence_id": "text:a",
            "modality": "text",
            "calibrated_probability": 0.8,
            "raw_probability": 0.7,
        },
        {
            "split": "train",
            "claim_id": "1",
            "evidence_id": "image:b",
            "modality": "image",
            "calibrated_probability": 0.6,
            "raw_probability": 0.5,
        },
    ]
    feature_index = {
        ("train", "1", "text:a"): np.asarray([1.0, 2.0]),
        ("train", "1", "image:b"): np.asarray([3.0, 4.0]),
    }
    aggregate = _aggregate(rows, feature_index)
    assert aggregate.shape == (13,)


def test_seed_aligned_verifier_bootstrap_reuses_claim_draws() -> None:
    gold = ["nei", "refuted", "supported", "supported"]
    result = _seed_aligned_paired_bootstrap(
        [
            (gold, gold, ["nei", "refuted", "refuted", "supported"]),
            (gold, gold, ["nei", "nei", "supported", "supported"]),
        ],
        samples=200,
        seed=7,
    )
    assert result["claims"] == 4
    assert result["seed_count"] == 2
    assert result["mean"] > 0.0
    assert len(result["samples"]) == 200
