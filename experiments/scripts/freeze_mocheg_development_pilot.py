from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from importlib.metadata import distributions
from pathlib import Path

from trace_scv.data.io import sha256_file


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _environment_packages() -> dict[str, str]:
    return dict(
        sorted(
            (
                distribution.metadata["Name"],
                distribution.version,
            )
            for distribution in distributions()
            if distribution.metadata["Name"]
        )
    )


def _write_deterministic_archive(
    root: Path, files: list[Path], archive_path: Path
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an audited MOCHEG development selector pilot."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("outputs/mocheg/archives/selector_pilot_trial_001.zip"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/mocheg/archives/selector_pilot_trial_001.freeze.json"),
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    audit_path = root / "outputs/mocheg/selector_pilot/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("selector pilot must pass independent audit before freezing")
    if audit["recomputed"].get("formal_test_annotations_accessed"):
        raise RuntimeError("refusing to freeze a development archive with formal-test access")

    required = [
        root / "configs/mocheg_development.yaml",
        root / "configs/workstation_4090.yaml",
        root / "outputs/preflight/workstation.json",
        root / "outputs/mocheg/pilot_features.npz",
        root / "outputs/mocheg/pilot_features.json",
        root / "outputs/mocheg/pilot_features.manifest.json",
        root / "data/processed/mocheg_development/preflight.json",
        root / "data/processed/mocheg_development/sentence_text_audit.json",
        root / "data/processed/mocheg_development/pilot_packet_manifest.json",
        root / "data/processed/mocheg_development/train_pilot_packets.jsonl",
        root / "data/processed/mocheg_development/validation_pilot_packets.jsonl",
        root / "data/public/mocheg/media/pilot_media_manifest.json",
        *sorted((root / "outputs/mocheg/selector_pilot").glob("*")),
    ]
    files = sorted(set(path.resolve() for path in required if path.is_file()), key=str)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing freeze inputs: {missing}")

    archive_path = args.archive.resolve()
    _write_deterministic_archive(root, files, archive_path)
    members = [
        {
            "path": str(path.relative_to(root).as_posix()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "profile": "mocheg_selector_development_pilot_trial_001",
        "status": "frozen_checked_development_result",
        "source_commit": _git_commit(root),
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
        "members": members,
        "packages": _environment_packages(),
        "selector_numeric_criteria_met": audit["recomputed"][
            "selector_numeric_criteria_met"
        ],
        "full_development_gate_pass": False,
        "formal_test_annotations_accessed": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
