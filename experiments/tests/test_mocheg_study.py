from __future__ import annotations

import numpy as np

from trace_scv.selection.mocheg_study import (
    LogisticEvidenceScorer,
    _bounded_candidate_indices,
    _claim_partition,
    _evaluate_claims,
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
