from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from trace_scv.data.io import sha256_file
from trace_scv.data.mocheg import assert_mocheg_development_sealed


def _command(*parts: str) -> str | None:
    try:
        result = subprocess.run(
            list(parts),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _git_state(project_root: Path) -> dict[str, object]:
    commit = _command("git", "-C", str(project_root), "rev-parse", "HEAD")
    status = _command("git", "-C", str(project_root), "status", "--porcelain")
    return {
        "commit": commit,
        "working_tree_clean": status == "",
        "status_lines": [] if not status else status.splitlines(),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "sklearn", "torch", "transformers", "yaml"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = None
    return versions


def _gpu_state() -> list[dict[str, str]]:
    output = _command(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    if not output:
        return []
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        name, memory_mib, driver = [value.strip() for value in line.split(",", 2)]
        rows.append({"name": name, "memory_mib": memory_mib, "driver": driver})
    return rows


def freeze(project_root: Path, protocol_path: Path, output_root: Path) -> None:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    study = protocol["study"]
    if not study.get("formal_test_locked"):
        raise RuntimeError("publication protocol must keep the formal test locked")
    if study.get("formal_test_annotations_allowed"):
        raise RuntimeError("publication protocol unexpectedly allows formal-test annotations")

    experiment_root = project_root / "experiments"
    dataset_root = experiment_root / protocol["data"]["dataset_root"]
    assert_mocheg_development_sealed(dataset_root)
    forbidden = [
        dataset_root / "test",
        dataset_root / "mocheg-test",
        dataset_root / "metadata" / "test",
        dataset_root / "metadata" / "mocheg-test",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"formal-test material is present: {present}")

    output_root.mkdir(parents=True, exist_ok=True)
    checked_at = datetime.now(timezone.utc).isoformat()
    git = _git_state(project_root)
    lock = {
        "profile": "publication_confirmation_test_lock",
        "status": "locked",
        "checked_at_utc": checked_at,
        "protocol_sha256": sha256_file(protocol_path),
        "baseline_commit": study["baseline_commit"],
        "current_git": git,
        "dataset_root": str(dataset_root.resolve()),
        "forbidden_paths_checked_absent": [str(path.resolve()) for path in forbidden],
        "formal_test_annotations_accessed": False,
        "formal_test_unlocked": False,
        "author_authorization_recorded": False,
        "maximum_runs_after_future_unlock": int(
            protocol["budgets"]["formal_test_runs_maximum_after_unlock"]
        ),
    }
    (output_root / "test_lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )

    memory_bytes: int | None = None
    try:
        import psutil

        memory_bytes = int(psutil.virtual_memory().total)
    except Exception:
        pass
    environment = {
        "profile": "publication_confirmation_environment",
        "captured_at_utc": checked_at,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "ram_bytes": memory_bytes,
        "packages": _package_versions(),
        "gpus": _gpu_state(),
        "git": git,
        "protocol_sha256": sha256_file(protocol_path),
        "energy_instrumentation": None,
        "energy_claim_allowed": False,
    }
    (output_root / "hardware_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )

    freeze_manifest = {
        "profile": "publication_confirmation_protocol_freeze",
        "status": "pass",
        "checked_at_utc": checked_at,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256_file(protocol_path),
        },
        "test_lock": {
            "path": str((output_root / "test_lock.json").resolve()),
            "sha256": sha256_file(output_root / "test_lock.json"),
        },
        "environment": {
            "path": str((output_root / "hardware_environment.json").resolve()),
            "sha256": sha256_file(output_root / "hardware_environment.json"),
        },
        "development_seal_passed": True,
        "formal_test_annotations_accessed": False,
    }
    (output_root / "protocol_freeze.json").write_text(
        json.dumps(freeze_manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the test-sealed publication confirmation protocol."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/configs/publication_confirmation.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/publication_confirmation"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    protocol = args.protocol if args.protocol.is_absolute() else project_root / args.protocol
    output = args.output if args.output.is_absolute() else project_root / args.output
    freeze(project_root, protocol.resolve(), output.resolve())


if __name__ == "__main__":
    main()

