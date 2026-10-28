from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from importlib.metadata import distributions
from pathlib import Path

from trace_scv.data.io import sha256_file


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the checked full-corpus MOCHEG development study.")
    parser.add_argument("--archive", type=Path, default=Path("outputs/mocheg/archives/full_corpus_development_trial_002.zip"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/mocheg/archives/full_corpus_development_trial_002.freeze.json"))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    gate_path = root / "outputs/mocheg/development_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["formal_test_annotations_accessed"]:
        raise RuntimeError("refusing development artifacts that accessed formal-test annotations")
    paths = [
        root / "configs/mocheg_development.yaml",
        root / "configs/workstation_4090.yaml",
        root / "outputs/preflight/workstation.json",
        root / "outputs/tests/full_suite.json",
        root / "data/processed/mocheg_development/preflight.json",
        root / "data/public/mocheg/media/full_development_media_manifest.json",
        root / "data/public/mocheg/media/full_media_lfs_import.json",
        root / "data/public/mocheg/media/full_media_transport_log.json",
        root / "outputs/mocheg/full_corpus_features.npz",
        root / "outputs/mocheg/full_corpus_features.json",
        root / "outputs/mocheg/full_corpus_features.manifest.json",
        root / "outputs/mocheg/full_corpus_features.rankings.jsonl",
        root / "outputs/mocheg/full_corpus_features.audit.json",
        root / "outputs/mocheg/clean_rerun/audit.json",
        gate_path,
        *sorted((root / "outputs/mocheg/full_corpus_selector_trial_002").glob("*")),
        *sorted((root / "outputs/mocheg/full_corpus_verifier_trial_002").glob("*")),
        *sorted((root / "outputs/mocheg/clean_rerun/selector_trial_002").glob("*")),
        *sorted((root / "outputs/mocheg/clean_rerun/verifier_trial_002").glob("*")),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing full-development freeze inputs: {missing}")
    files = sorted({path.resolve() for path in paths}, key=lambda path: path.as_posix())
    target = args.archive.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    manifest = {
        "profile": "mocheg_full_corpus_development_trial_002",
        "status": "frozen_checked_development_result",
        "source_commit": _git_commit(root),
        "archive": {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target)},
        "members": [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
        "packages": dict(sorted((item.metadata["Name"], item.version) for item in distributions() if item.metadata["Name"])),
        "development_gate": gate,
        "formal_test_annotations_accessed": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": manifest["archive"], "members": len(files)}, indent=2))


if __name__ == "__main__":
    main()
