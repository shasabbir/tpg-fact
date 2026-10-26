from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
SNAPSHOT = EXPERIMENTS / "data" / "private" / "pilot_claims_2026-10-24"


def ignored(_directory: str, names: list[str]) -> set[str]:
    blocked = {".venv", ".pytest_cache", "__pycache__", ".secrets"}
    return {name for name in names if name in blocked or name.endswith((".pyc", ".pyo"))}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the key-free Kaggle background Dataset ZIP")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dtp-background-") as temporary:
        bundle = Path(temporary) / "dtp_fact_kaggle_dataset"
        background = bundle / "background"
        data = bundle / "data" / "us_tpmfc_candidate_snapshot"
        background.mkdir(parents=True)
        data.mkdir(parents=True)

        for directory in ("configs", "src", "scripts", "tests"):
            shutil.copytree(EXPERIMENTS / directory, background / directory, ignore=ignored)
        for filename in ("pyproject.toml", "README.md"):
            shutil.copy2(EXPERIMENTS / filename, background / filename)
        for filename in (
            "candidates.jsonl", "candidate_review.csv", "collection_manifest.json",
            "screening_queue.csv", "screening_queue_manifest.json",
        ):
            shutil.copy2(SNAPSHOT / filename, data / filename)
        shutil.copytree(SNAPSHOT / "raw", data / "raw")
        shutil.copy2(ROOT / "kaggle" / "README.md", bundle / "README.md")
        shutil.copy2(ROOT / "kaggle" / "FINAL_DATA_LAYOUT.md", bundle / "FINAL_DATA_LAYOUT.md")

        if any(path.name == ".secrets" for path in bundle.rglob("*")):
            raise SystemExit("Secret directory entered the bundle")
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle.parent))
    print(output)
    print(sha256(output))


if __name__ == "__main__":
    main()
