from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


EXCLUDED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    ".venv-workstation",
    "__pycache__",
    "build",
    "dist",
    "tmp",
}
EXCLUDED_PATH_PREFIXES = {
    "experiments/cache/",
    "experiments/checkpoints/",
    "experiments/data/",
    "experiments/outputs/",
    "paper/build/",
}
EXCLUDED_FILE_NAMES = {".env", "credentials.json", "secrets.json"}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".key",
    ".log",
    ".out",
    ".pem",
    ".pyc",
    ".synctex",
    ".toc",
    ".zip",
}
ALLOWED_ROOT_FILES = {
    ".gitignore",
    "README.md",
    "RUN_ON_4090_WORKSTATION.md",
    "plan.md",
}
ALLOWED_ROOT_DIRS = {
    "Base papers",
    "Reference_good_papers",
    "experiments",
    "paper",
    "research",
}
REQUIRED_BUNDLE_FILES = {
    "plan.md",
    "RUN_ON_4090_WORKSTATION.md",
    "paper/paper.tex",
    "experiments/pyproject.toml",
    "experiments/scripts/setup_workstation.ps1",
    "experiments/scripts/run_workstation_pilot.ps1",
    "experiments/src/trace_scv/data/__init__.py",
    "experiments/src/trace_scv/data/io.py",
    "experiments/src/trace_scv/data/mocheg.py",
    "experiments/src/trace_scv/cefes.py",
    "experiments/src/trace_scv/mocheg_features.py",
    "experiments/src/trace_scv/selection/mocheg_study.py",
}


def included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    relative_posix = relative.as_posix()
    if relative.parts[0] not in ALLOWED_ROOT_DIRS and relative_posix not in ALLOWED_ROOT_FILES:
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
        return False
    if any(relative_posix.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
        return False
    if path.name in EXCLUDED_FILE_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a secret-free workstation bundle.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output or root / "dist" / "cefes-workstation-source.zip").resolve()
    if root not in output.parents:
        raise ValueError("output must stay inside the project root")
    output.parent.mkdir(parents=True, exist_ok=True)

    candidates = [root / name for name in ALLOWED_ROOT_FILES]
    for directory in sorted(ALLOWED_ROOT_DIRS):
        base = root / directory
        if base.is_dir():
            candidates.extend(base.rglob("*"))
    files = sorted(
        path
        for path in candidates
        if path.is_file() and path != output and included(path, root)
    )
    relative_files = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(REQUIRED_BUNDLE_FILES - relative_files)
    if missing:
        raise RuntimeError(f"workstation bundle is incomplete; missing: {missing}")
    manifest_files: list[dict[str, object]] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            archive.writestr(relative, content)
            manifest_files.append(
                {
                    "path": relative,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        manifest = {
            "profile": "cefes-workstation-source",
            "files": manifest_files,
            "public_data_included": False,
            "outputs_included": False,
            "credentials_included": False,
        }
        archive.writestr(
            "WORKSTATION_BUNDLE_MANIFEST.json",
            json.dumps(manifest, indent=2) + "\n",
        )
    print(
        json.dumps(
            {
                "archive": str(output),
                "files": len(files),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
