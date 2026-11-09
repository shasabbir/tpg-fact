from __future__ import annotations

from pathlib import Path

import yaml

import numpy as np

from trace_scv.publication_retrieval import metric_csv_fields, weighted_rank_fusion
from trace_scv.publication_candidates import load_npz_arrays, rank_map
from trace_scv.publication_selector import config_from_publication_protocol
from trace_scv.publication_completeness import (
    completeness_feature_vector,
    paired_bootstrap_rows,
    serialize_completeness_model,
)
from trace_scv.selection.mocheg_study import LogisticEvidenceScorer


PROJECT_ROOT = Path(__file__).parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "experiments" / "configs" / "publication_confirmation.yaml"


def test_publication_protocol_keeps_formal_test_locked() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["study"]["formal_test_locked"] is True
    assert protocol["study"]["formal_test_annotations_allowed"] is False
    assert protocol["budgets"]["formal_test_runs_maximum_after_unlock"] == 1


def test_publication_protocol_uses_isolated_outputs_and_strongest_comparator() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["study"]["output_namespace"] == "outputs/publication_confirmation"
    assert protocol["selection"]["primary_comparator"] == "calibrated_threshold"
    assert protocol["selection"]["superiority_gate"][
        "minimum_mean_f1_gain_over_calibrated_threshold"
    ] == 0.01


def test_publication_protocol_prespecifies_reachability_and_matched_packet() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["retrieval"]["retained_text_candidates"] == 40
    assert protocol["retrieval"]["retained_image_candidates"] == 20
    assert protocol["statistics"]["reachability_strata"] == [
        "none",
        "partial",
        "complete",
    ]


def test_weighted_rank_fusion_is_deterministic_and_weight_sensitive() -> None:
    first = [4, 3, 2, 1]
    second = [1, 2, 3, 4]
    first_heavy, first_scores = weighted_rank_fusion(
        first, second, first_weight=0.75, constant=10, depth=4
    )
    second_heavy, second_scores = weighted_rank_fusion(
        first, second, first_weight=0.25, constant=10, depth=4
    )
    assert first_heavy.tolist()[0] == 4
    assert second_heavy.tolist()[0] == 1
    assert np.all(first_scores[:-1] >= first_scores[1:])
    assert np.all(second_scores[:-1] >= second_scores[1:])


def test_equal_weight_fusion_breaks_score_ties_by_identifier() -> None:
    indices, _ = weighted_rank_fusion(
        [2, 1], [1, 2], first_weight=0.5, constant=60, depth=2
    )
    assert indices.tolist() == [1, 2]


def test_retrieval_metric_csv_accepts_both_operating_depths() -> None:
    fields = metric_csv_fields(40, 20)
    assert "recall_at_40" in fields
    assert "recall_at_20" in fields


def test_rank_map_preserves_rank_and_score_alignment() -> None:
    indices = np.asarray([[7, 2, 9]], dtype=np.int64)
    scores = np.asarray([[0.8, 0.5, 0.1]], dtype=np.float32)
    mapped = rank_map(indices, scores, 0)
    assert mapped[7][0] == 1
    assert mapped[2][0] == 2
    assert mapped[9][0] == 3
    assert np.isclose(mapped[2][1], 0.5)


def test_npz_arrays_are_materialized_before_archive_closes(tmp_path: Path) -> None:
    archive = tmp_path / "rankings.npz"
    np.savez_compressed(archive, ranks=np.asarray([[3, 2, 1]], dtype=np.int64))
    loaded = load_npz_arrays(archive, ("ranks",))
    archive.unlink()
    assert loaded["ranks"].tolist() == [[3, 2, 1]]


def test_selector_engine_config_uses_frozen_thresholds_and_no_penalty_search() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    config = config_from_publication_protocol(protocol)
    assert config.thresholds == tuple(protocol["selection"]["thresholds"])
    assert config.size_penalties == (0.0,)
    assert config.redundancy_penalties == (0.0,)
    assert config.bootstrap_samples == 10_000


def test_completeness_features_match_frozen_schema() -> None:
    rows = [
        {
            "calibrated_probability": probability,
            "source_group": f"g{index}",
            "modality": "text" if index < 2 else "image",
            "dense_rank": index + 1,
            "lexical_rank": index + 2,
        }
        for index, probability in enumerate((0.7, 0.2, 0.08, 0.02))
    ]
    features = completeness_feature_vector(rows, threshold=0.1, maximum_items=60)
    assert features.shape == (10,)
    assert np.isfinite(features).all()
    assert np.isclose(features[0], 1.0)
    assert features[5] == 2.0


def test_completeness_bootstrap_rows_use_engine_key_schema() -> None:
    rows = [{"claim_id": "c1", "switch_f1": 0.4}]
    assert paired_bootstrap_rows(rows, "switch") == [{"key": "c1", "f1": 0.4}]


def test_completeness_model_uses_dataclass_serialization() -> None:
    model = LogisticEvidenceScorer(
        mean=(0.0,), scale=(1.0,), weights=(0.5,), bias=-0.25
    )
    serialized = serialize_completeness_model(model)
    assert serialized["weights"] == (0.5,)
    assert serialized["bias"] == -0.25

