from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

from data.io import write_json


def resolve_device(requested: str, allow_cpu: bool) -> str:
    """Resolve the orchestration device without silently downgrading final runs."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if allow_cpu:
            return "cpu"
        raise RuntimeError(
            "CUDA is required for the final study; use --allow-cpu only for diagnostics"
        )
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-safe orchestration for feature extraction, tuning, final seeds, and paper tables"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--frozen-data-directory", required=True)
    parser.add_argument("--claim-image-manifest", required=True)
    parser.add_argument("--evidence-image-manifest")
    parser.add_argument("--evidence-text")
    parser.add_argument(
        "--conflict-manifest",
        required=True,
        help="Audited JSONL mapping every test claim to semantically conflicting donor evidence",
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--paper-table-directory", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tuning-trials", type=int)
    parser.add_argument("--tuning-epochs", type=int)
    return parser.parse_args()


def _run(command: list[str], log_path: Path, cwd: Path) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "command": command,
        "return_code": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "log": str(log_path),
    }


def _require_success(record: dict[str, object], stage: str) -> None:
    if int(record["return_code"]) != 0:
        raise RuntimeError(f"{stage} failed; see {record['log']}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = resolve_device(args.device, args.allow_cpu)
    frozen_root = Path(args.frozen_data_directory).resolve()
    dataset_manifest = frozen_root / "dataset_manifest.json"
    if not dataset_manifest.exists():
        raise FileNotFoundError(f"Freeze the dataset first: {dataset_manifest}")
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if dataset.get("status") != "frozen_dataset":
        raise ValueError("Dataset manifest is not frozen")
    claims = frozen_root / "claims.jsonl"
    evidence = frozen_root / "evidence.jsonl"
    output_root = Path(args.output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "orchestration_manifest.json"
    state: dict[str, object] = {
        "status": "running",
        "config": str(config_path),
        "dataset_manifest": str(dataset_manifest),
        "requested_device": args.device,
        "resolved_device": device,
        "cpu_diagnostic_override": bool(args.allow_cpu and device == "cpu"),
        "stages": [],
    }

    feature_root = output_root / "features"
    feature_manifest = feature_root / "feature_manifest.json"
    if not (args.resume and feature_manifest.exists()):
        command = [
            sys.executable,
            str(root / "scripts" / "build_feature_store.py"),
            "--config",
            str(config_path),
            "--claims",
            str(claims),
            "--evidence",
            str(evidence),
            "--claim-image-manifest",
            str(Path(args.claim_image_manifest).resolve()),
            "--output-directory",
            str(feature_root),
            "--device",
            device,
        ]
        if args.evidence_image_manifest:
            command.extend(("--evidence-image-manifest", str(Path(args.evidence_image_manifest).resolve())))
        if args.evidence_text:
            command.extend(("--evidence-text", str(Path(args.evidence_text).resolve())))
        record = _run(command, feature_root / "feature_build.log", root)
        state["stages"].append({"stage": "features", **record})
        write_json(state_path, state)
        _require_success(record, "Feature extraction")

    retrieval_root = output_root / "retrieval"
    retrieval_report = retrieval_root / "retrieval_report.json"
    if not (
        args.resume
        and retrieval_report.exists()
        and json.loads(retrieval_report.read_text(encoding="utf-8")).get("status")
        == "completed_frozen_retrieval"
    ):
        command = [
            sys.executable,
            str(root / "scripts" / "retrieve_evidence.py"),
            "--config",
            str(config_path),
            "--claims",
            str(claims),
            "--evidence",
            str(evidence),
            "--text-features",
            str(feature_root / "text_embeddings.npz"),
            "--image-features",
            str(feature_root / "image_embeddings.npz"),
            "--output-directory",
            str(retrieval_root),
        ]
        if args.evidence_text:
            command.extend(("--evidence-text", str(Path(args.evidence_text).resolve())))
        record = _run(command, retrieval_root / "retrieval.log", root)
        state["stages"].append({"stage": "retrieval", **record})
        write_json(state_path, state)
        _require_success(record, "Frozen evidence retrieval")
    retrieved_claims = retrieval_root / "retrieved_claims.jsonl"
    if not retrieved_claims.exists():
        raise FileNotFoundError(retrieved_claims)

    selection_root = output_root / "selection"
    for variant in config["training"]["variants"]:
        selection_path = selection_root / variant / "selection.json"
        if args.resume and selection_path.exists():
            existing = json.loads(selection_path.read_text(encoding="utf-8"))
            if existing.get("status") == "selection_complete":
                continue
        command = [
            sys.executable,
            str(root / "scripts" / "tune.py"),
            "--config",
            str(config_path),
            "--claims",
            str(retrieved_claims),
            "--evidence",
            str(evidence),
            "--text-features",
            str(feature_root / "text_embeddings.npz"),
            "--image-features",
            str(feature_root / "image_embeddings.npz"),
            "--variant",
            str(variant),
            "--output-directory",
            str(selection_root / variant),
            "--device",
            device,
        ]
        if args.tuning_trials:
            command.extend(("--trials", str(args.tuning_trials)))
        if args.tuning_epochs:
            command.extend(("--epochs", str(args.tuning_epochs)))
        record = _run(command, selection_root / variant / "tuning.log", root)
        state["stages"].append({"stage": f"tune:{variant}", **record})
        write_json(state_path, state)
        _require_success(record, f"Tuning {variant}")

    matrix_root = output_root / "matrix"
    matrix_manifest = matrix_root / "matrix_manifest.json"
    if not (args.resume and matrix_manifest.exists() and json.loads(matrix_manifest.read_text(encoding="utf-8")).get("status") == "complete"):
        command = [
            sys.executable,
            str(root / "scripts" / "run_final_matrix.py"),
            "--config",
            str(config_path),
            "--claims",
            str(retrieved_claims),
            "--evidence",
            str(evidence),
            "--text-features",
            str(feature_root / "text_embeddings.npz"),
            "--image-features",
            str(feature_root / "image_embeddings.npz"),
            "--selection-root",
            str(selection_root),
            "--output-directory",
            str(matrix_root),
            "--device",
            device,
        ]
        record = _run(command, matrix_root / "matrix.log", root)
        state["stages"].append({"stage": "matrix", **record})
        write_json(state_path, state)
        _require_success(record, "Final experiment matrix")

    intervention_root = output_root / "interventions"
    intervention_report = intervention_root / "intervention_report.json"
    if not (
        args.resume
        and intervention_report.exists()
        and json.loads(intervention_report.read_text(encoding="utf-8")).get("status")
        == "checked_intervention_evaluation"
    ):
        record = _run(
            [
                sys.executable,
                str(root / "scripts" / "evaluate_interventions.py"),
                "--config",
                str(config_path),
                "--claims",
                str(retrieved_claims),
                "--evidence",
                str(evidence),
                "--text-features",
                str(feature_root / "text_embeddings.npz"),
                "--image-features",
                str(feature_root / "image_embeddings.npz"),
                "--matrix-root",
                str(matrix_root),
                "--output-directory",
                str(intervention_root),
                "--conflict-manifest",
                str(Path(args.conflict_manifest).resolve()),
                "--device",
                device,
            ],
            intervention_root / "intervention.log",
            root,
        )
        state["stages"].append({"stage": "interventions", **record})
        write_json(state_path, state)
        _require_success(record, "Intervention evaluation")

    comparison = output_root / "comparison.json"
    record = _run(
        [
            sys.executable,
            str(root / "scripts" / "compare_experiments.py"),
            "--claims",
            str(retrieved_claims),
            "--matrix-root",
            str(matrix_root),
            "--bootstrap-replicates",
            str(config["training"]["bootstrap_replicates"]),
            "--output",
            str(comparison),
        ],
        output_root / "comparison.log",
        root,
    )
    state["stages"].append({"stage": "comparison", **record})
    write_json(state_path, state)
    _require_success(record, "Comparison aggregation")

    record = _run(
        [
            sys.executable,
            str(root / "scripts" / "export_paper_artifacts.py"),
            "--comparison",
            str(comparison),
            "--dataset-manifest",
            str(dataset_manifest),
            "--retrieval-report",
            str(retrieval_report),
            "--intervention-report",
            str(intervention_report),
            "--output-directory",
            str(Path(args.paper_table_directory).resolve()),
        ],
        output_root / "paper_artifacts.log",
        root,
    )
    state["stages"].append({"stage": "paper_artifacts", **record})
    _require_success(record, "Paper artifact export")
    state["status"] = "complete"
    write_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
