from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from data.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command once for every configured experiment seed"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-directory", default="outputs/seed_runs")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command arguments after --; use {seed} and {run_dir} placeholders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command_template = list(args.command)
    if command_template and command_template[0] == "--":
        command_template = command_template[1:]
    if not command_template:
        raise ValueError("Provide a command after --")
    if not any("{seed}" in part for part in command_template):
        raise ValueError("The command must include a {seed} placeholder")

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    output_root = Path(args.output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "config": str(config_path),
        "seeds": seeds,
        "command_template": command_template,
        "runs": [],
    }

    for seed in seeds:
        run_directory = output_root / f"seed_{seed}"
        run_directory.mkdir(parents=True, exist_ok=True)
        replacements = {"{seed}": str(seed), "{run_dir}": str(run_directory)}
        command = [
            part.replace("{seed}", replacements["{seed}"]).replace(
                "{run_dir}", replacements["{run_dir}"]
            )
            for part in command_template
        ]
        environment = dict(os.environ)
        environment["TPG_FACT_SEED"] = str(seed)
        started = time.perf_counter()
        log_path = run_directory / "run.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(
                command,
                cwd=config_path.parent.parent,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        run = {
            "seed": seed,
            "command": command,
            "return_code": completed.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "run_directory": str(run_directory),
            "log": str(log_path),
        }
        manifest["runs"].append(run)
        write_json(output_root / "manifest.json", manifest)
        if completed.returncode != 0 and not args.keep_going:
            raise RuntimeError(f"Seed {seed} failed; see {log_path}")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
