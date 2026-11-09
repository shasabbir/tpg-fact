from __future__ import annotations

from pathlib import Path

import yaml


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

