from __future__ import annotations

from trace_scv.q1_phase2_robustness import _transform_rows


def _rows() -> list[dict[str, object]]:
    return [
        {"evidence_id": "a", "gold": True, "final_rank": 1, "modality": "text", "calibrated_probability": 0.8, "source_group": "x"},
        {"evidence_id": "b", "gold": False, "final_rank": 2, "modality": "image", "calibrated_probability": 0.4, "source_group": "y"},
    ]


def test_duplicate_condition_adds_non_gold_ids() -> None:
    rows, probabilities = _transform_rows(_rows(), "duplicate_top_3", seed=13, key="validation:1")
    assert len(rows) == 4
    assert len(probabilities) == 4
    assert rows[-1]["gold"] is False
    assert "#duplicate-" in str(rows[-1]["evidence_id"])


def test_temperature_condition_preserves_order() -> None:
    _, probabilities = _transform_rows(_rows(), "temperature_2.0", seed=13, key="validation:1")
    assert probabilities[0] > probabilities[1]
