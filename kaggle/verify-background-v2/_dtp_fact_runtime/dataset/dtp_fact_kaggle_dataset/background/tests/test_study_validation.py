from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from evaluation.study_validation import (
    REQUIRED_INTERVENTION_VARIANTS,
    REQUIRED_METRICS,
    REQUIRED_SCENARIOS,
    REQUIRED_TABLES,
    validate_study_artifacts,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_complete_tree(root: Path) -> tuple[Path, Path, Path, Path]:
    config_path = root / "configs" / "base.yaml"
    config_path.parent.mkdir(parents=True)
    config = {
        "data": {
            "test_start_year": 2025,
            "test_end_year": 2026,
            "minimum_total_cases": 3,
            "minimum_per_label": 1,
            "minimum_test_events": 1,
            "minimum_independent_cluster_fraction": 0.5,
        },
        "training": {
            "variants": ["vanilla_rag", "full"],
            "seeds": [11, 23],
            "tuning_trials": 2,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    frozen = root / "frozen"
    claims = [
        {
            "claim_id": "c1",
            "claim_text": "train",
            "claim_image_url": "https://example.com/1.jpg",
            "image_sha256": "1",
            "image_phash": "1",
            "claim_time": "2023-01-01T00:00:00Z",
            "decision_cutoff": "2023-01-02T00:00:00Z",
            "fact_check_url": "https://example.com/f1",
            "source_rating": "true",
            "normalized_label": "supported",
            "event_id": "e1",
            "evidence_ids": ["d1"],
        },
        {
            "claim_id": "c2",
            "claim_text": "validation",
            "claim_image_url": "https://example.com/2.jpg",
            "image_sha256": "2",
            "image_phash": "2",
            "claim_time": "2024-01-01T00:00:00Z",
            "decision_cutoff": "2024-01-02T00:00:00Z",
            "fact_check_url": "https://example.com/f2",
            "source_rating": "false",
            "normalized_label": "refuted",
            "event_id": "e2",
            "evidence_ids": ["d1"],
        },
        {
            "claim_id": "c3",
            "claim_text": "test",
            "claim_image_url": "https://example.com/3.jpg",
            "image_sha256": "3",
            "image_phash": "3",
            "claim_time": "2025-01-01T00:00:00Z",
            "decision_cutoff": "2025-01-02T00:00:00Z",
            "fact_check_url": "https://example.com/f3",
            "source_rating": "unproven",
            "normalized_label": "insufficient_evidence",
            "event_id": "e3",
            "evidence_ids": ["d1"],
        },
    ]
    evidence = [{"evidence_id": "d1"}]
    claims_path = frozen / "claims.jsonl"
    evidence_path = frozen / "evidence.jsonl"
    _write_jsonl(claims_path, claims)
    _write_jsonl(evidence_path, evidence)
    splits: dict[str, dict[str, object]] = {}
    for name, row in zip(("train", "validation", "test"), claims, strict=True):
        path = frozen / f"{name}_claims.jsonl"
        _write_jsonl(path, [row])
        splits[name] = {"claims": 1, "sha256": _sha256(path)}
    _write_json(
        frozen / "dataset_manifest.json",
        {
            "status": "frozen_dataset",
            "claims_sha256": _sha256(claims_path),
            "evidence_sha256": _sha256(evidence_path),
            "independent_source_cluster_fraction": 1.0,
            "group_leakage_checked": ["event_id", "image_phash"],
            "splits": splits,
        },
    )

    results = root / "results"
    _write_json(
        results / "orchestration_manifest.json",
        {
            "status": "complete",
            "resolved_device": "cuda",
            "cpu_diagnostic_override": False,
            "stages": [{"return_code": 0}],
        },
    )
    _write_json(
        results / "retrieval" / "retrieval_report.json",
        {
            "status": "completed_frozen_retrieval",
            "diagnostic_skip_reranker": False,
            "reranker": "pinned",
            "metrics_by_split": {"train": {}, "validation": {}, "test": {}},
        },
    )
    variants = config["training"]["variants"]
    seeds = config["training"]["seeds"]
    for variant in variants:
        _write_json(
            results / "selection" / variant / "selection.json",
            {
                "status": "selection_complete",
                "variant": variant,
                "budget": 2,
                "selection_metric": "validation_macro_f1",
                "test_data_used_for_selection": False,
                "trials": [{"return_code": 0}, {"return_code": 0}],
            },
        )
    runs = [{"variant": "majority", "seed": None, "return_code": 0}]
    evaluations = []
    aggregate = {metric: {"mean": 0.5, "sample_std": 0.1} for metric in REQUIRED_METRICS}
    for variant in ("majority", *variants):
        evaluations.append({"variant": variant, "return_code": 0})
        _write_json(
            results / "matrix" / variant / "evaluation.json",
            {
                "status": "measured_from_supplied_prediction_artifacts",
                "evaluated_split": "test",
                "seed_count": 1 if variant == "majority" else 2,
                "aggregate": aggregate,
            },
        )
    for variant in variants:
        for seed in seeds:
            run_root = results / "matrix" / variant / f"seed_{seed}"
            _write_json(
                run_root / "run_summary.json",
                {
                    "status": "completed_training_run",
                    "settings": {"variant": variant, "seed": seed},
                },
            )
            (run_root / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
            (run_root / "best_model.pt").write_bytes(b"checkpoint")
            runs.append({"variant": variant, "seed": seed, "return_code": 0})
    _write_json(
        results / "matrix" / "matrix_manifest.json",
        {
            "status": "complete",
            "variants": variants,
            "seeds": seeds,
            "runs": runs,
            "evaluations": evaluations,
        },
    )
    intervention_runs = [
        {"variant": variant, "scenario": scenario, "seed": seed}
        for variant in REQUIRED_INTERVENTION_VARIANTS
        for scenario in REQUIRED_SCENARIOS
        for seed in seeds
    ]
    _write_json(
        results / "interventions" / "intervention_report.json",
        {
            "status": "checked_intervention_evaluation",
            "diagnostic_skip_synthetic_encoding": False,
            "conflict_donor_policy": "audited_manifest",
            "variants": list(REQUIRED_INTERVENTION_VARIANTS),
            "scenarios": list(REQUIRED_SCENARIOS),
            "runs": intervention_runs,
        },
    )
    comparison_evaluations = [
        {"variant": variant, "aggregate": aggregate}
        for variant in ("majority", *variants)
    ]
    _write_json(
        results / "comparison.json",
        {
            "status": "checked_matrix_aggregation",
            "reference_variant": "full",
            "test_claims": 1,
            "test_events": 1,
            "evaluations": comparison_evaluations,
            "paired_event_bootstrap": [
                {"comparison": "majority", "holm_adjusted_p": 0.1},
                {"comparison": "vanilla_rag", "holm_adjusted_p": 0.2},
            ],
        },
    )
    tables = root / "tables"
    tables.mkdir(parents=True)
    for name in REQUIRED_TABLES:
        (tables / name).write_text("checked\n", encoding="utf-8")
    return config_path, frozen, results, tables


def test_complete_artifact_tree_is_validated(tmp_path: Path) -> None:
    paths = _build_complete_tree(tmp_path)
    report = validate_study_artifacts(*paths)
    assert report["status"] == "scientific_artifacts_validated"
    assert report["scientific_claims_allowed"] is True
    assert not report["blockers"]


def test_cpu_diagnostic_and_missing_seed_are_rejected(tmp_path: Path) -> None:
    config, frozen, results, tables = _build_complete_tree(tmp_path)
    orchestration_path = results / "orchestration_manifest.json"
    orchestration = json.loads(orchestration_path.read_text(encoding="utf-8"))
    orchestration["resolved_device"] = "cpu"
    orchestration["cpu_diagnostic_override"] = True
    _write_json(orchestration_path, orchestration)
    (results / "matrix" / "full" / "seed_23" / "predictions.jsonl").unlink()
    report = validate_study_artifacts(config, frozen, results, tables)
    assert report["status"] == "invalid_or_incomplete_final_artifacts"
    assert report["scientific_claims_allowed"] is False
    assert "cuda_orchestration_complete" in report["blockers"]
    assert "complete_five_seed_matrix" in report["blockers"]
