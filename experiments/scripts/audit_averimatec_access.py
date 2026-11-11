from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

from trace_scv.data.averimatec import load_annotations, normalize_source_url
from trace_scv.data.io import sha256_file


def _git_revision(path: Path) -> str | None:
    if not path.is_dir():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _annotation_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    labels = Counter(str(row["label"]) for row in rows)
    question_counts = [len(row.get("questions") or []) for row in rows]
    source_urls = [
        str(source)
        for row in rows
        for source in row.get("gold_source_urls", [])
    ]
    claim_images = [
        str(image)
        for row in rows
        for image in (row.get("claim_images") or [])
    ]
    return {
        "claims": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "questions": {
            "total": sum(question_counts),
            "mean_per_claim": sum(question_counts) / len(question_counts),
            "minimum_per_claim": min(question_counts),
            "maximum_per_claim": max(question_counts),
        },
        "normalized_gold_sources": len(source_urls),
        "unique_normalized_gold_sources": len(set(source_urls)),
        "duplicate_gold_source_occurrences": len(source_urls) - len(set(source_urls)),
        "claims_without_web_source": sum(not row.get("gold_source_urls") for row in rows),
        "claim_image_references": len(claim_images),
        "unique_claim_image_references": len(set(claim_images)),
    }


def _file_record(path: Path, *, hash_complete_archives: bool) -> dict[str, object]:
    record: dict[str, object] = {"path": path.as_posix(), "exists": path.is_file()}
    if not path.is_file():
        return record
    record["bytes"] = path.stat().st_size
    in_progress = path.with_name(path.name + ".aria2").exists()
    record["download_complete"] = not in_progress
    if not in_progress and (hash_complete_archives or path.suffix.lower() != ".zip"):
        record["sha256"] = sha256_file(path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AVerImaTeC access without opening test data.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/public/averimatec/raw"))
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("cache/external_confirmation/averimatec_official"),
    )
    parser.add_argument(
        "--shared-task-root",
        type=Path,
        default=Path("cache/external_confirmation/averimatec_shared_task"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/access_audit.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-complete-archives", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite explicitly")

    train_path = args.raw_root / "train.json"
    validation_path = args.raw_root / "val.json"
    train = load_annotations(train_path, split="train")
    validation = load_annotations(validation_path, split="validation")
    image_root = args.raw_root / "claim_images"
    referenced_images = [
        str(image)
        for row in train + validation
        for image in (row.get("claim_images") or [])
    ]
    available_images = {path.name for path in image_root.rglob("*") if path.is_file()}
    missing_images = [name for name in referenced_images if name not in available_images]
    archives = [
        args.raw_root / "images.zip",
        args.raw_root / "test_data.zip",
        args.raw_root / "knowledge_store/train_image_related.zip",
        args.raw_root / "knowledge_store/train_text_related.zip",
        args.raw_root / "knowledge_store/val.zip",
        args.raw_root
        / "humane_filled/Knowledge_Store/val/text_related/val_text_related_filled.zip",
    ]
    file_records = [
        _file_record(path, hash_complete_archives=args.hash_complete_archives) for path in archives
    ]
    required_store_records = file_records[2:5]
    status = (
        "fail"
        if missing_images
        else "pass"
        if all(row.get("exists") and row.get("download_complete") for row in required_store_records)
        else "in_progress"
    )
    audit = {
        "status": status,
        "test_access_boundary": {
            "test_archive_exists": (args.raw_root / "test_data.zip").is_file(),
            "test_archive_opened_by_this_audit": False,
            "test_annotations_accessed": False,
            "test_submission_performed": False,
        },
        "annotations": {
            "train": _annotation_summary(train),
            "validation": _annotation_summary(validation),
        },
        "claim_images": {
            "root": image_root.as_posix(),
            "referenced": len(referenced_images),
            "missing": len(missing_images),
            "missing_examples": missing_images[:20],
        },
        "files": file_records,
        "repositories": {
            "official": _git_revision(args.official_root),
            "shared_task": _git_revision(args.shared_task_root),
        },
        "official_evaluator": {
            "model": "google/gemma-3-27b-it",
            "published_launcher_gpus": 2,
            "published_launcher_memory_gb": 50,
            "local_execution_attempted": False,
            "reason": "deferred until predictions exist; evaluator loading is separately audited",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "normalization_policy": {
            "example": normalize_source_url("HTTPS://www.Example.com/a/?utm_source=x&b=2#fragment"),
            "exact_url_is_diagnostic_not_official_semantic_endpoint": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
