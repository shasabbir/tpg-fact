from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from data.io import write_json
from training.engine import SUPPORTED_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen baseline/ablation matrix for all configured seeds"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--text-features", required=True)
    parser.add_argument("--image-features")
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--variant", action="append", choices=sorted(SUPPORTED_VARIANTS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def _run(command: list[str], log_path: Path, cwd: Path) -> tuple[int, float]:
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
    return completed.returncode, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["training"]["seeds"]]
    variants = args.variant or [str(value) for value in config["training"]["variants"]]
    unsupported = sorted(set(variants) - SUPPORTED_VARIANTS)
    if unsupported:
        raise ValueError(f"Unsupported configured variants: {unsupported}")
    output_root = Path(args.output_directory).resolve()
    selection_root = Path(args.selection_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": "running",
        "config": str(config_path),
        "claims": str(Path(args.claims).resolve()),
        "evidence": str(Path(args.evidence).resolve()),
        "seeds": seeds,
        "variants": variants,
        "runs": [],
        "evaluations": [],
    }

    majority_root = output_root / "majority"
    majority_command = [
        sys.executable,
        str(root / "scripts" / "predict_majority.py"),
        "--claims",
        str(Path(args.claims).resolve()),
        "--output-directory",
        str(majority_root),
    ]
    majority_return_code, elapsed = _run(majority_command, majority_root / "run.log", root)
    manifest["runs"].append(
        {"variant": "majority", "seed": None, "return_code": majority_return_code, "elapsed_seconds": elapsed}
    )
    if majority_return_code != 0 and not args.keep_going:
        raise RuntimeError(f"Majority baseline failed; see {majority_root / 'run.log'}")

    for variant in variants:
        selection_path = selection_root / variant / "selection.json"
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Missing validation selection for {variant}: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "selection_complete":
            raise ValueError(f"Selection is incomplete for {variant}: {selection_path}")
        parameters = selection["selected_parameters"]
        prediction_paths: list[Path] = []
        for seed in seeds:
            run_root = output_root / variant / f"seed_{seed}"
            command = [
                sys.executable,
                str(root / "scripts" / "train.py"),
                "--config",
                str(config_path),
                "--claims",
                str(Path(args.claims).resolve()),
                "--evidence",
                str(Path(args.evidence).resolve()),
                "--text-features",
                str(Path(args.text_features).resolve()),
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--output-dir",
                str(run_root),
                "--device",
                args.device,
                "--learning-rate",
                str(parameters["learning_rate"]),
                "--weight-decay",
                str(parameters["weight_decay"]),
                "--hidden-dimension",
                str(parameters["hidden_dimension"]),
                "--dropout",
                str(parameters["dropout"]),
                "--graph-layers",
                str(parameters["graph_layers"]),
                "--batch-size",
                str(parameters["batch_size"]),
            ]
            if args.image_features:
                command.extend(("--image-features", str(Path(args.image_features).resolve())))
            return_code, elapsed = _run(command, run_root / "run.log", root)
            run_record = {
                "variant": variant,
                "seed": seed,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "run_directory": str(run_root),
                "selection": str(selection_path),
            }
            manifest["runs"].append(run_record)
            write_json(output_root / "matrix_manifest.json", manifest)
            if return_code != 0:
                if not args.keep_going:
                    raise RuntimeError(f"{variant} seed {seed} failed; see {run_root / 'run.log'}")
                continue
            prediction_paths.append(run_root / "predictions.jsonl")

        if len(prediction_paths) != len(seeds):
            continue
        evaluation_path = output_root / variant / "evaluation.json"
        command = [
            sys.executable,
            str(root / "scripts" / "evaluate_predictions.py"),
            "--claims",
            str(Path(args.claims).resolve()),
            "--expected-seeds",
            str(len(seeds)),
            "--split",
            "test",
            "--bootstrap-replicates",
            str(config["training"]["bootstrap_replicates"]),
            "--target-coverage",
            str(config["model"]["target_answer_coverage"]),
            "--output",
            str(evaluation_path),
        ]
        for prediction_path in prediction_paths:
            command.extend(("--predictions", str(prediction_path)))
        return_code, elapsed = _run(
            command, output_root / variant / "evaluation.log", root
        )
        manifest["evaluations"].append(
            {
                "variant": variant,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "output": str(evaluation_path),
            }
        )
        write_json(output_root / "matrix_manifest.json", manifest)
        if return_code != 0 and not args.keep_going:
            raise RuntimeError(f"Evaluation failed for {variant}")

    if majority_return_code == 0:
        majority_evaluation = majority_root / "evaluation.json"
        command = [
            sys.executable,
            str(root / "scripts" / "evaluate_predictions.py"),
            "--claims",
            str(Path(args.claims).resolve()),
            "--predictions",
            str(majority_root / "predictions.jsonl"),
            "--expected-seeds",
            "1",
            "--split",
            "test",
            "--bootstrap-replicates",
            str(config["training"]["bootstrap_replicates"]),
            "--output",
            str(majority_evaluation),
        ]
        code, elapsed = _run(command, majority_root / "evaluation.log", root)
        manifest["evaluations"].append(
            {
                "variant": "majority",
                "return_code": code,
                "elapsed_seconds": elapsed,
                "output": str(majority_evaluation),
            }
        )
    manifest["status"] = (
        "complete"
        if all(int(row["return_code"]) == 0 for row in manifest["runs"])
        and all(int(row["return_code"]) == 0 for row in manifest["evaluations"])
        else "completed_with_failures"
    )
    write_json(output_root / "matrix_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
