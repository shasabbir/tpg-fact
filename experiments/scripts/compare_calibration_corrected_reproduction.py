from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEEDS = (13, 37, 73, 101, 137)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_status(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["status"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare clean calibration-corrected selector and verifier records."
    )
    parser.add_argument(
        "--original", type=Path, default=Path("outputs/calibration_corrected_study")
    )
    parser.add_argument(
        "--clean",
        type=Path,
        default=Path("outputs/calibration_corrected_study/clean_rerun"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../research/calibration_corrected_study/reproduction_manifest.json"),
    )
    args = parser.parse_args()

    relative_paths: list[Path] = []
    for seed in SEEDS:
        relative_paths.extend(
            [
                Path(f"selector/seed_{seed}_candidate_records.jsonl"),
                Path(f"selector/seed_{seed}_claim_records.jsonl"),
                Path(f"verifier/seed_{seed}_verdict_records.jsonl"),
                Path(f"verifier/seed_{seed}_verifier.json"),
            ]
        )

    comparisons = []
    for relative in relative_paths:
        original = args.original / relative
        clean = args.clean / relative
        original_hash = sha256(original)
        clean_hash = sha256(clean)
        comparisons.append(
            {
                "path": relative.as_posix(),
                "original_sha256": original_hash,
                "clean_sha256": clean_hash,
                "exact": original_hash == clean_hash,
            }
        )

    exact_count = sum(row["exact"] for row in comparisons)
    manifest = {
        "profile": "calibration_corrected_clean_reproduction",
        "status": "pass" if exact_count == len(comparisons) else "fail",
        "original_root": args.original.as_posix(),
        "clean_root": args.clean.as_posix(),
        "retrieval_policy": "frozen retrieval artifacts reused unchanged",
        "selector_original_audit": load_status(args.original / "selector/audit.json"),
        "selector_clean_audit": load_status(args.clean / "selector/audit.json"),
        "verifier_original_audit": load_status(args.original / "verifier/audit.json"),
        "verifier_clean_audit": load_status(args.clean / "verifier/audit.json"),
        "exact_core_artifacts": exact_count,
        "core_artifacts_compared": len(comparisons),
        "note": (
            "Runtime-bearing summaries are audited separately and are intentionally "
            "excluded from byte-for-byte comparison."
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "exact_core_artifacts", "core_artifacts_compared")}, indent=2))
    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
