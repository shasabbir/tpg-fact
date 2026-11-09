from __future__ import annotations

import json
from pathlib import Path

import yaml

from trace_scv.q1_phase2_study import (
    Phase2StudyConfig,
    audit_phase2_selector_study,
    run_phase2_selector_study,
)


def config_from_publication_protocol(protocol: dict[str, object]) -> Phase2StudyConfig:
    study = protocol["study"]
    data = protocol["data"]
    selection = protocol["selection"]
    statistics = protocol["statistics"]
    calibration = protocol["calibration"]
    return Phase2StudyConfig(
        seeds=tuple(map(int, study["seeds"])),
        partition_seed=int(data["train_partition_seed"]),
        fixed_budgets=tuple(map(int, selection["fixed_budgets"])),
        thresholds=tuple(map(float, selection["thresholds"])),
        size_penalties=(0.0,),
        redundancy_penalties=(0.0,),
        calibrator_kinds=tuple(map(str, calibration["candidates"])),
        baseline_text_recall=0.3616018276517159,
        baseline_image_recall=0.21384297520661158,
        maximum_selected_items=int(selection["maximum_selected_items"]),
        bootstrap_samples=int(statistics["bootstrap_samples"]),
    )


def run_publication_selector_base(
    retrieval_root: Path,
    output_root: Path,
    protocol_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["study"]["formal_test_annotations_allowed"]:
        raise RuntimeError("formal-test annotations are not allowed")
    config = config_from_publication_protocol(protocol)
    summary = run_phase2_selector_study(
        retrieval_root,
        output_root,
        protocol_path,
        config=config,
    )
    audit = audit_phase2_selector_study(output_root)
    if summary.get("formal_test_annotations_accessed") or summary.get("formal_test_unlocked"):
        raise RuntimeError("selector engine violated the formal-test lock")
    audit["profile"] = "publication_confirmation_selector_base_audit"
    audit["formal_test_annotations_accessed"] = False
    audit_path = output_root / "selector.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return summary, audit

