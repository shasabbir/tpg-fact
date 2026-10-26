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
from training.engine import SUPPORTED_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete five-seed model matrix on the frozen 200-case pilot"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--pilot-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variant", action="append", choices=sorted(SUPPORTED_VARIANTS))
    parser.add_argument("--tuning-trials", type=int)
    parser.add_argument("--tuning-epochs", type=int)
    return parser.parse_args()


def run(command: list[str], log: Path, cwd: Path) -> dict[str, object]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8", newline="\n") as stream:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    result = {
        "command": command,
        "return_code": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "log": str(log),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed; see {log}")
    return result


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pilot_root = Path(args.pilot_directory).resolve()
    manifest_path = pilot_root / "pilot_manifest.json"
    frozen_root = pilot_root / "frozen"
    image_manifest = pilot_root / "image_manifest.csv"
    required = (
        manifest_path,
        frozen_root / "dataset_manifest.json",
        frozen_root / "claims.jsonl",
        frozen_root / "evidence.jsonl",
        image_manifest,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Pilot package is incomplete: {missing}")
    pilot_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if pilot_manifest.get("status") != "audited_feasibility_pilot_evidence_gate_passed":
        raise ValueError("The pilot evidence gate has not passed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before running the pilot experiments")

    variants = args.variant or [str(value) for value in config["training"]["variants"]]
    output_root = Path(args.output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "pilot_orchestration_manifest.json"
    state: dict[str, object] = {
        "status": "running",
        "scientific_scope": "200-case feasibility pilot only; not final-study evidence",
        "config": str(config_path),
        "pilot_manifest": str(manifest_path),
        "device": args.device,
        "variants": variants,
        "seeds": config["training"]["seeds"],
        "stages": [],
    }

    feature_root = output_root / "features"
    feature_manifest = feature_root / "feature_manifest.json"
    if not (args.resume and feature_manifest.is_file()):
        state["stages"].append(
            {
                "stage": "features",
                **run(
                    [
                        sys.executable,
                        str(root / "scripts" / "build_feature_store.py"),
                        "--config",
                        str(config_path),
                        "--claims",
                        str(frozen_root / "claims.jsonl"),
                        "--evidence",
                        str(frozen_root / "evidence.jsonl"),
                        "--claim-image-manifest",
                        str(image_manifest),
                        "--output-directory",
                        str(feature_root),
                        "--device",
                        args.device,
                    ],
                    feature_root / "feature_build.log",
                    root,
                ),
            }
        )
        write_json(state_path, state)

    retrieval_root = output_root / "retrieval"
    retrieval_report = retrieval_root / "retrieval_report.json"
    if not (args.resume and retrieval_report.is_file()):
        state["stages"].append(
            {
                "stage": "retrieval",
                **run(
                    [
                        sys.executable,
                        str(root / "scripts" / "retrieve_evidence.py"),
                        "--config",
                        str(config_path),
                        "--claims",
                        str(frozen_root / "claims.jsonl"),
                        "--evidence",
                        str(frozen_root / "evidence.jsonl"),
                        "--text-features",
                        str(feature_root / "text_embeddings.npz"),
                        "--image-features",
                        str(feature_root / "image_embeddings.npz"),
                        "--output-directory",
                        str(retrieval_root),
                    ],
                    retrieval_root / "retrieval.log",
                    root,
                ),
            }
        )
        write_json(state_path, state)

    retrieved_claims = retrieval_root / "retrieved_claims.jsonl"
    selection_root = output_root / "selection"
    for variant in variants:
        selection_path = selection_root / variant / "selection.json"
        if args.resume and selection_path.is_file():
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
            str(frozen_root / "evidence.jsonl"),
            "--text-features",
            str(feature_root / "text_embeddings.npz"),
            "--image-features",
            str(feature_root / "image_embeddings.npz"),
            "--variant",
            variant,
            "--output-directory",
            str(selection_root / variant),
            "--device",
            args.device,
        ]
        if args.tuning_trials:
            command.extend(("--trials", str(args.tuning_trials)))
        if args.tuning_epochs:
            command.extend(("--epochs", str(args.tuning_epochs)))
        state["stages"].append(
            {
                "stage": f"tune:{variant}",
                **run(command, selection_root / variant / "tuning.log", root),
            }
        )
        write_json(state_path, state)

    matrix_root = output_root / "matrix"
    matrix_manifest = matrix_root / "matrix_manifest.json"
    if not (
        args.resume
        and matrix_manifest.is_file()
        and json.loads(matrix_manifest.read_text(encoding="utf-8")).get("status") == "complete"
    ):
        command = [
            sys.executable,
            str(root / "scripts" / "run_final_matrix.py"),
            "--config",
            str(config_path),
            "--claims",
            str(retrieved_claims),
            "--evidence",
            str(frozen_root / "evidence.jsonl"),
            "--text-features",
            str(feature_root / "text_embeddings.npz"),
            "--image-features",
            str(feature_root / "image_embeddings.npz"),
            "--selection-root",
            str(selection_root),
            "--output-directory",
            str(matrix_root),
            "--device",
            args.device,
        ]
        for variant in variants:
            command.extend(("--variant", variant))
        state["stages"].append(
            {"stage": "five_seed_matrix", **run(command, matrix_root / "matrix.log", root)}
        )
        write_json(state_path, state)

    comparison = output_root / "comparison.json"
    state["stages"].append(
        {
            "stage": "paired_comparisons",
            **run(
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
            ),
        }
    )
    state["status"] = "pilot_experiment_complete"
    state["scientific_results_validated"] = False
    state["next_gate"] = (
        "Use these outputs for pipeline/model debugging only; scale or adopt the preregistered "
        "existing-dataset fallback before making final paper claims."
    )
    write_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
