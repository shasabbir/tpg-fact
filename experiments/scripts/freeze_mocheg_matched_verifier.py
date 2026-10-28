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


def _archive(root: Path, files: list[Path], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in files:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, path.read_bytes(), compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze checked MOCHEG matched-verifier outputs.")
    parser.add_argument("--verifier-dir", type=Path, default=Path("outputs/mocheg/matched_verifier_trial_002"))
    parser.add_argument("--selector-dir", type=Path, default=Path("outputs/mocheg/selector_pilot_trial_002"))
    parser.add_argument("--archive", type=Path, default=Path("outputs/mocheg/archives/matched_verifier_trial_002.zip"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/mocheg/archives/matched_verifier_trial_002.freeze.json"))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    verifier_dir = args.verifier_dir.resolve()
    audit = json.loads((verifier_dir / "audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "pass" or audit["recomputed"]["formal_test_annotations_accessed"]:
        raise RuntimeError("matched verifier must pass audit with the formal test sealed")
    required = [
        root / "configs/mocheg_development.yaml",
        root / "configs/workstation_4090.yaml",
        root / "outputs/preflight/workstation.json",
        root / "outputs/mocheg/pilot_features.npz",
        root / "outputs/mocheg/pilot_features.json",
        root / "outputs/mocheg/pilot_features.manifest.json",
        root / "data/processed/mocheg_development/train_claims.jsonl",
        root / "data/processed/mocheg_development/validation_claims.jsonl",
        *sorted(args.selector_dir.resolve().glob("*")),
        *sorted(verifier_dir.glob("*")),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing verifier freeze inputs: {missing}")
    files = sorted({path.resolve() for path in required}, key=lambda path: path.as_posix())
    target = args.archive.resolve()
    _archive(root, files, target)
    summary = json.loads((verifier_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = {
        "profile": "mocheg_matched_verifier_development_trial_002",
        "status": "frozen_checked_development_result",
        "source_commit": _git_commit(root),
        "archive": {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target)},
        "members": [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
        "packages": dict(sorted((item.metadata["Name"], item.version) for item in distributions() if item.metadata["Name"])),
        "mean_cefes_minus_fixed_macro_f1": summary["mean_cefes_minus_fixed_macro_f1"],
        "verdict_noninferiority_point_pass": summary["verdict_noninferiority_point_pass"],
        "full_development_gate_pass": False,
        "formal_test_annotations_accessed": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": manifest["archive"], "members": len(files)}, indent=2))


if __name__ == "__main__":
    main()
