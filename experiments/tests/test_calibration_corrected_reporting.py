from __future__ import annotations

from pathlib import Path
import csv
import json

import pytest

from trace_scv.calibration_corrected_reporting import (
    METHOD_LABELS,
    _reachability_stratum,
)


def test_corrected_reporting_has_labels_for_every_selector() -> None:
    assert set(METHOD_LABELS) == {
        "fixed",
        "threshold",
        "count",
        "cefes",
        "size_penalized",
        "rc_cefes",
    }


def test_main_paper_does_not_use_phase_language() -> None:
    root = Path(__file__).resolve().parents[2]
    paper_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [root / "paper" / "paper.tex", *(root / "paper" / "sections").glob("*.tex")]
    ).lower()
    assert "phase 1" not in paper_text
    assert "phase-1" not in paper_text
    assert "phase 2" not in paper_text
    assert "phase-2" not in paper_text
    assert "phase 3" not in paper_text
    assert "phase-3" not in paper_text


def test_reachability_strata_cover_valid_gold_packets() -> None:
    assert _reachability_stratum(0, 3) == "none"
    assert _reachability_stratum(1, 3) == "partial"
    assert _reachability_stratum(3, 3) == "complete"
    with pytest.raises(ValueError):
        _reachability_stratum(1, 0)
    with pytest.raises(ValueError):
        _reachability_stratum(4, 3)


def test_reported_result_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    results = json.loads(
        (root / "research" / "calibration_corrected_study" / "final_results.json").read_text(
            encoding="utf-8"
        )
    )
    means = {row["method"]: row for row in results["method_means"]}
    effects = {row["method"]: row for row in results["effects_vs_fixed"]}
    assert results["population"]["validation_claims"] == 1490
    assert results["retained_packet"]["packet_oracle_f1"] == 0.35604839918841624
    assert means["threshold"]["f1"] > means["cefes"]["f1"] > means["fixed"]["f1"]
    assert effects["cefes"]["lower"] > 0
    assert results["verifier_noninferiority_pass"] is False
    assert results["formal_test_annotations_accessed"] is False
    assert sum(row["claims"] for row in results["validation_reachability"]) == 1490


def test_depth_tables_trace_to_frozen_results() -> None:
    root = Path(__file__).resolve().parents[2]
    research = root / "research" / "calibration_corrected_study"
    with (research / "partition_selector_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        partitions = list(csv.DictReader(handle))
    with (research / "robustness_absolute.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        robustness = list(csv.DictReader(handle))
    assert [int(row["partition"]) for row in partitions] == [13, 37, 73, 101, 137]
    assert all(float(row["expected_minus_fixed"]) > 0 for row in partitions)
    assert len(robustness) == 12
    temperature = {row["condition"]: row for row in robustness}
    assert float(temperature["temperature_0.5"]["difference"]) < 0
    assert float(temperature["temperature_2.0"]["difference"]) < 0
