from __future__ import annotations

import numpy as np

from trace_scv.q1_phase2_retrieval import (
    _claim_role,
    _packet_ceiling,
    _ranking_metrics,
    image_filename_text,
    reciprocal_rank_fusion,
)


def test_claim_partition_is_deterministic_and_complete() -> None:
    first = [_claim_role(str(index), 20261102) for index in range(1000)]
    second = [_claim_role(str(index), 20261102) for index in range(1000)]
    assert first == second
    assert set(first) == {"fit", "calibration", "model_selection", "internal_holdout"}
    assert 500 <= first.count("fit") <= 600


def test_image_filename_text_removes_release_prefix_and_extension() -> None:
    assert (
        image_filename_text("00110-00551-01-Notre_Dame_fire_Eiffel_Tower.jpg")
        == "Notre Dame fire Eiffel Tower"
    )
    assert image_filename_text("00001-00002-01-123456.jpg") == "image"


def test_reciprocal_rank_fusion_is_stable_and_rewards_agreement() -> None:
    indices, scores = reciprocal_rank_fusion(
        ([3, 1, 2], [2, 3, 4]), constant=60, depth=4
    )
    assert indices.tolist()[:2] == [3, 2]
    assert np.all(np.diff(scores) <= 0)
    assert len(set(indices.tolist())) == 4


def test_ranking_metrics_use_complete_gold_denominator() -> None:
    metrics = _ranking_metrics(["a", "x", "b"], {"a", "b", "missing"})
    assert metrics["recall_at_5"] == 2 / 3
    assert metrics["map_at_100"] < 1.0
    assert metrics["ndcg_at_100"] < 1.0


def test_packet_ceiling_excludes_claims_without_released_gold() -> None:
    ceiling = _packet_ceiling(
        [
            {
                "gold_count": 0,
                "reachable_gold_count": 0,
                "complete_gold_reachable": False,
                "oracle_f1": 1.0,
            },
            {
                "gold_count": 2,
                "reachable_gold_count": 1,
                "complete_gold_reachable": False,
                "oracle_f1": 2.0 / 3.0,
            },
            {
                "gold_count": 1,
                "reachable_gold_count": 1,
                "complete_gold_reachable": True,
                "oracle_f1": 1.0,
            },
        ]
    )
    assert ceiling["eligible_claims"] == 2
    assert ceiling["claims_without_released_gold"] == 1
    assert ceiling["mean_oracle_f1"] == (2.0 / 3.0 + 1.0) / 2.0
    assert ceiling["complete_gold_claim_coverage"] == 0.5
