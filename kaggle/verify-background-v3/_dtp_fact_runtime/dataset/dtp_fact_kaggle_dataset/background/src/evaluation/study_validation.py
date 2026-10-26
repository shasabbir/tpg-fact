"""Strict validation of final-study artifacts before scientific use.

This module intentionally validates provenance and protocol completion rather than
recomputing model metrics.  The latter remains the responsibility of the checked
evaluation scripts that produced the artifacts inspected here.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from data.io import read_jsonl
from data.schema import ClaimRecord
from data.splits import find_group_leakage


REQUIRED_TABLES = (
    "generated_main_results.tex",
    "generated_ablation_results.tex",
    "generated_dataset_summary.tex",
    "generated_retrieval_results.tex",
    "generated_intervention_results.tex",
    "generated_result_macros.tex",
)
REQUIRED_INTERVENTION_VARIANTS = (
    "vanilla_rag",
    "time_filtered_rag",
    "provenance_pool",
    "full",
)
REQUIRED_SCENARIOS = (
    "duplicate_1",
    "duplicate_5",
    "duplicate_10",
    "post_cutoff_verdict",
    "remove_earliest_source",
    "dependent_conflict",
    "independent_conflict",
    "caption_swap",
    "historical_image_reuse",
)
REQUIRED_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "macro_ovr_auroc",
    "brier",
    "nll",
    "ece",
    "aurc",
    "risk_at_target_coverage",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def validate_study_artifacts(
    config_path: str | Path,
    frozen_data_directory: str | Path,
    results_directory: str | Path,
    paper_table_directory: str | Path,
) -> dict[str, Any]:
    """Return an evidence-boundary report for a purported final scientific run."""

    config_path = Path(config_path).resolve()
    frozen_root = Path(frozen_data_directory).resolve()
    results_root = Path(results_directory).resolve()
    table_root = Path(paper_table_directory).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variants = [str(value) for value in config["training"]["variants"]]
    seeds = [int(value) for value in config["training"]["seeds"]]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if not results_root.is_dir():
        check("results_directory", False, f"Missing results directory: {results_root}")
    if not frozen_root.is_dir():
        check("frozen_data_directory", False, f"Missing frozen data directory: {frozen_root}")

    dataset_path = frozen_root / "dataset_manifest.json"
    claims_path = frozen_root / "claims.jsonl"
    evidence_path = frozen_root / "evidence.jsonl"
    dataset = _load_json(dataset_path)
    check(
        "frozen_dataset_manifest",
        dataset is not None and dataset.get("status") == "frozen_dataset",
        str(dataset_path),
    )
    if dataset is not None:
        check(
            "frozen_manifest_hashes",
            claims_path.is_file()
            and evidence_path.is_file()
            and _sha256(claims_path) == dataset.get("claims_sha256")
            and _sha256(evidence_path) == dataset.get("evidence_sha256"),
            "Frozen claim and evidence files must match their recorded SHA-256 values.",
        )
        required_group_fields = {"event_id", "image_phash"}
        check(
            "declared_group_leakage_controls",
            required_group_fields.issubset(set(dataset.get("group_leakage_checked", []))),
            "Required grouping fields: event_id and image_phash.",
        )
        claims: list[ClaimRecord] = []
        try:
            claims = [ClaimRecord.from_dict(row) for row in read_jsonl(claims_path)]
            leakage = find_group_leakage(claims)
        except (OSError, KeyError, TypeError, ValueError) as error:
            leakage = {"read_error": {"error": [str(error)]}}
        check(
            "observed_group_isolation",
            bool(claims) and not any(leakage.values()),
            "No event or perceptual-image group may cross temporal partitions.",
        )
        label_counts = Counter(claim.normalized_label for claim in claims)
        test_events = {
            claim.event_id
            for claim in claims
            if config["data"]["test_start_year"]
            <= claim.claim_time.year
            <= config["data"]["test_end_year"]
        }
        thresholds = {
            "claims": len(claims) >= int(config["data"]["minimum_total_cases"]),
            "labels": all(
                label_counts[label] >= int(config["data"]["minimum_per_label"])
                for label in ("supported", "refuted", "insufficient_evidence")
            ),
            "test_events": len(test_events) >= int(config["data"]["minimum_test_events"]),
            "independent_clusters": float(
                dataset.get("independent_source_cluster_fraction", 0.0)
            )
            >= float(config["data"]["minimum_independent_cluster_fraction"]),
        }
        check(
            "preregistered_benchmark_gate",
            all(thresholds.values()),
            json.dumps(thresholds, sort_keys=True),
        )
        splits = dataset.get("splits", {})
        split_ok = set(splits) == {"train", "validation", "test"}
        for split_name in ("train", "validation", "test"):
            split = splits.get(split_name, {})
            split_file = frozen_root / f"{split_name}_claims.jsonl"
            split_ok = (
                split_ok
                and split_file.is_file()
                and int(split.get("claims", 0)) > 0
                and _sha256(split_file) == split.get("sha256")
            )
        check(
            "frozen_temporal_splits",
            split_ok,
            "Train, validation, and test manifests must be non-empty and hash-matched.",
        )

    orchestration_path = results_root / "orchestration_manifest.json"
    orchestration = _load_json(orchestration_path)
    orchestration_ok = (
        orchestration is not None
        and orchestration.get("status") == "complete"
        and str(orchestration.get("resolved_device", "")).startswith("cuda")
        and orchestration.get("cpu_diagnostic_override") is False
        and all(int(stage.get("return_code", 1)) == 0 for stage in orchestration.get("stages", []))
    )
    check(
        "cuda_orchestration_complete",
        orchestration_ok,
        "Final orchestration must complete on CUDA without the CPU diagnostic override.",
    )

    retrieval_path = results_root / "retrieval" / "retrieval_report.json"
    retrieval = _load_json(retrieval_path)
    retrieval_ok = (
        retrieval is not None
        and retrieval.get("status") == "completed_frozen_retrieval"
        and retrieval.get("diagnostic_skip_reranker") is False
        and retrieval.get("reranker")
        and set(retrieval.get("metrics_by_split", {})) == {"train", "validation", "test"}
    )
    check(
        "checked_retrieval",
        bool(retrieval_ok),
        "Frozen retrieval must include the pinned reranker and all temporal splits.",
    )

    selection_ok = True
    selection_errors: list[str] = []
    expected_budget = int(config["training"]["tuning_trials"])
    for variant in variants:
        selection_path = results_root / "selection" / variant / "selection.json"
        selection = _load_json(selection_path)
        valid = (
            selection is not None
            and selection.get("status") == "selection_complete"
            and selection.get("variant") == variant
            and int(selection.get("budget", 0)) == expected_budget
            and selection.get("selection_metric") == "validation_macro_f1"
            and selection.get("test_data_used_for_selection") is False
            and len(selection.get("trials", [])) == expected_budget
            and all(int(row.get("return_code", 1)) == 0 for row in selection.get("trials", []))
        )
        if not valid:
            selection_ok = False
            selection_errors.append(variant)
    check(
        "validation_only_model_selection",
        selection_ok,
        "Invalid or incomplete selections: " + (", ".join(selection_errors) or "none"),
    )

    matrix_root = results_root / "matrix"
    matrix_path = matrix_root / "matrix_manifest.json"
    matrix = _load_json(matrix_path)
    matrix_ok = matrix is not None and matrix.get("status") == "complete"
    matrix_detail: list[str] = []
    if matrix_ok and matrix is not None:
        matrix_ok = set(matrix.get("variants", [])) == set(variants) and set(matrix.get("seeds", [])) == set(seeds)
        runs = matrix.get("runs", [])
        run_keys = Counter((row.get("variant"), row.get("seed")) for row in runs if int(row.get("return_code", 1)) == 0)
        expected_run_keys = {("majority", None), *((variant, seed) for variant in variants for seed in seeds)}
        if set(run_keys) != expected_run_keys or any(count != 1 for count in run_keys.values()):
            matrix_ok = False
            matrix_detail.append("run/seed coverage mismatch")
        evaluations = matrix.get("evaluations", [])
        evaluated = Counter(
            row.get("variant")
            for row in evaluations
            if int(row.get("return_code", 1)) == 0
        )
        expected_evaluations = {"majority", *variants}
        if set(evaluated) != expected_evaluations or any(count != 1 for count in evaluated.values()):
            matrix_ok = False
            matrix_detail.append("evaluation coverage mismatch")
        for variant in ("majority", *variants):
            evaluation = _load_json(matrix_root / variant / "evaluation.json")
            expected_seed_count = 1 if variant == "majority" else len(seeds)
            if not (
                evaluation is not None
                and evaluation.get("status") == "measured_from_supplied_prediction_artifacts"
                and evaluation.get("evaluated_split") == "test"
                and int(evaluation.get("seed_count", 0)) == expected_seed_count
                and set(REQUIRED_METRICS).issubset(evaluation.get("aggregate", {}))
            ):
                matrix_ok = False
                matrix_detail.append(f"invalid evaluation: {variant}")
        for variant in variants:
            for seed in seeds:
                run_root = matrix_root / variant / f"seed_{seed}"
                summary = _load_json(run_root / "run_summary.json")
                if not (
                    summary is not None
                    and summary.get("status") == "completed_training_run"
                    and int(summary.get("settings", {}).get("seed", -1)) == seed
                    and summary.get("settings", {}).get("variant") == variant
                    and (run_root / "predictions.jsonl").is_file()
                    and (run_root / "best_model.pt").is_file()
                ):
                    matrix_ok = False
                    matrix_detail.append(f"invalid run: {variant}/seed_{seed}")
    check(
        "complete_five_seed_matrix",
        bool(matrix_ok),
        "; ".join(matrix_detail) or "All configured variants, seeds, and evaluations are present.",
    )

    intervention_path = results_root / "interventions" / "intervention_report.json"
    interventions = _load_json(intervention_path)
    intervention_ok = (
        interventions is not None
        and interventions.get("status") == "checked_intervention_evaluation"
        and interventions.get("diagnostic_skip_synthetic_encoding") is False
        and interventions.get("conflict_donor_policy") == "audited_manifest"
        and set(interventions.get("variants", [])) == set(REQUIRED_INTERVENTION_VARIANTS)
        and set(interventions.get("scenarios", [])) == set(REQUIRED_SCENARIOS)
        and len(interventions.get("runs", []))
        == len(REQUIRED_INTERVENTION_VARIANTS) * len(REQUIRED_SCENARIOS) * len(seeds)
    )
    check(
        "audited_intervention_evaluation",
        bool(intervention_ok),
        "All nine scenarios must use five seeds and the audited conflict-donor manifest.",
    )

    comparison_path = results_root / "comparison.json"
    comparison = _load_json(comparison_path)
    comparison_ok = (
        comparison is not None
        and comparison.get("status") == "checked_matrix_aggregation"
        and comparison.get("reference_variant") == "full"
        and {row.get("variant") for row in comparison.get("evaluations", [])}
        == {"majority", *variants}
        and len(comparison.get("paired_event_bootstrap", [])) == len(variants)
        and all("holm_adjusted_p" in row for row in comparison.get("paired_event_bootstrap", []))
    )
    check(
        "paired_event_comparison",
        bool(comparison_ok),
        "Checked comparison must include every variant, paired event bootstrap, and Holm correction.",
    )

    tables_ok = all((table_root / name).is_file() for name in REQUIRED_TABLES)
    check(
        "generated_paper_artifacts",
        tables_ok,
        "Required generated files: " + ", ".join(REQUIRED_TABLES),
    )

    blockers = [row["name"] for row in checks if not row["passed"]]
    if not results_root.is_dir():
        status = "blocked_missing_final_results"
    elif not frozen_root.is_dir() or dataset is None:
        status = "blocked_missing_audited_frozen_data"
    elif blockers:
        status = "invalid_or_incomplete_final_artifacts"
    else:
        status = "scientific_artifacts_validated"
    report: dict[str, Any] = {
        "status": status,
        "scientific_claims_allowed": status == "scientific_artifacts_validated",
        "validated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": str(config_path),
        "frozen_data_directory": str(frozen_root),
        "results_directory": str(results_root),
        "paper_table_directory": str(table_root),
        "expected_variants": variants,
        "expected_seeds": seeds,
        "checks": checks,
        "blockers": blockers,
    }
    if status == "scientific_artifacts_validated" and comparison is not None:
        report["validated_result_summary"] = {
            "test_claims": comparison["test_claims"],
            "test_events": comparison["test_events"],
            "evaluations": comparison["evaluations"],
            "paired_event_bootstrap": comparison["paired_event_bootstrap"],
        }
    return report

