from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

from data.io import write_json
from training.engine import SUPPORTED_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic validation-only hyperparameter budget"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--text-features", required=True)
    parser.add_argument("--image-features")
    parser.add_argument("--variant", choices=sorted(SUPPORTED_VARIANTS), required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    return parser.parse_args()


def _variant_seed(base_seed: int, variant: str) -> int:
    value = int(hashlib.sha256(variant.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + value) % (2**32)


def _sample_trials(count: int, seed: int) -> list[dict[str, object]]:
    random = np.random.default_rng(seed)
    choices: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    while len(choices) < count:
        candidate = {
            "learning_rate": float(math.exp(random.uniform(math.log(1e-5), math.log(5e-4)))),
            "weight_decay": float(random.choice([0.0, 1e-5, 1e-4, 1e-3])),
            "hidden_dimension": int(random.choice([128, 256, 384])),
            "dropout": float(random.choice([0.1, 0.2, 0.3, 0.4])),
            "graph_layers": int(random.choice([1, 2, 3])),
            "batch_size": int(random.choice([8, 16, 32])),
        }
        key = tuple(candidate.values())
        if key not in seen:
            seen.add(key)
            choices.append(candidate)
    return choices


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    trial_count = args.trials or int(config["training"]["tuning_trials"])
    if trial_count < 1:
        raise ValueError("trials must be positive")
    selection_seed = int(config["project"]["seed"])
    trials = _sample_trials(trial_count, _variant_seed(selection_seed, args.variant))
    output_root = Path(args.output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, parameters in enumerate(trials, start=1):
        trial_directory = output_root / f"trial_{index:02d}"
        command = [
            sys.executable,
            str(Path(__file__).with_name("train.py")),
            "--config",
            str(config_path),
            "--claims",
            str(Path(args.claims).resolve()),
            "--evidence",
            str(Path(args.evidence).resolve()),
            "--text-features",
            str(Path(args.text_features).resolve()),
            "--variant",
            args.variant,
            "--seed",
            str(selection_seed),
            "--output-dir",
            str(trial_directory),
            "--device",
            args.device,
            "--selection-only",
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
        if args.epochs:
            command.extend(("--epochs", str(args.epochs)))
        if args.precision:
            command.extend(("--precision", args.precision))
        log_path = trial_directory / "selection.log"
        trial_directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(
                command,
                cwd=config_path.parent.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        record: dict[str, object] = {
            "trial": index,
            "parameters": parameters,
            "return_code": completed.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "log": str(log_path),
        }
        summary_path = trial_directory / "run_summary.json"
        if completed.returncode == 0 and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            record["validation_macro_f1"] = summary["best_validation_macro_f1"]
            record["best_epoch"] = summary["best_epoch"]
        else:
            record["validation_macro_f1"] = None
        records.append(record)
        write_json(
            output_root / "selection.json",
            {
                "status": "running",
                "variant": args.variant,
                "selection_seed": selection_seed,
                "budget": trial_count,
                "trials": records,
            },
        )

    successful = [row for row in records if row["validation_macro_f1"] is not None]
    if not successful:
        raise RuntimeError(f"All {trial_count} selection trials failed")
    best = max(
        successful,
        key=lambda row: (float(row["validation_macro_f1"]), -int(row["trial"])),
    )
    report = {
        "status": "selection_complete",
        "variant": args.variant,
        "selection_seed": selection_seed,
        "budget": trial_count,
        "selection_metric": "validation_macro_f1",
        "best_trial": best["trial"],
        "best_validation_macro_f1": best["validation_macro_f1"],
        "selected_parameters": best["parameters"],
        "trials": records,
        "test_data_used_for_selection": False,
    }
    write_json(output_root / "selection.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
