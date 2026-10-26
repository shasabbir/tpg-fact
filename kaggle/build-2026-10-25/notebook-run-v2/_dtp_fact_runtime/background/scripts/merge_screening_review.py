from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from data.io import write_json


REVIEW_FIELDS = (
    "decision",
    "exclusion_reason",
    "us_relevant",
    "multimodal",
    "target_image_verified",
    "rating_verified",
    "claim_image_url",
    "claim_time",
    "decision_cutoff",
    "cutoff_basis",
    "cutoff_confidence",
    "normalized_label",
    "event_id",
    "evidence_query",
    "sufficiency_target",
    "sufficiency_target_source",
    "reviewer_notes",
)
IDENTITY_FIELDS = ("review_url", "claim_text", "textual_rating")


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _index(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = row.get("candidate_id", "").strip()
        if not identifier:
            raise ValueError(f"{name} contains a blank candidate_id")
        if identifier in result:
            raise ValueError(f"{name} contains duplicate candidate_id {identifier}")
        result[identifier] = row
    return result


def merge_rows(
    master_rows: list[dict[str, str]], queue_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    master = _index(master_rows, "master")
    queue = _index(queue_rows, "queue")
    unknown = sorted(set(queue) - set(master))
    if unknown:
        raise ValueError(f"Queue contains unknown candidate IDs: {unknown[:5]}")
    counts = {"queue_rows": len(queue), "changed_rows": 0, "keep": 0, "reject": 0, "pending": 0}
    for identifier, reviewed in queue.items():
        target = master[identifier]
        for field in IDENTITY_FIELDS:
            if reviewed.get(field, "") != target.get(field, ""):
                raise ValueError(f"Identity field {field} changed for {identifier}")
        changed = False
        for field in REVIEW_FIELDS:
            value = reviewed.get(field, "")
            if target.get(field, "") != value:
                target[field] = value
                changed = True
        counts[reviewed.get("decision", "pending").strip().casefold() or "pending"] = (
            counts.get(reviewed.get("decision", "pending").strip().casefold() or "pending", 0)
            + 1
        )
        counts["changed_rows"] += int(changed)
    return master_rows, counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely merge reviewed screening-queue fields into the full candidate CSV"
    )
    parser.add_argument("--master", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    master_path = Path(args.master).resolve()
    queue_path = Path(args.queue).resolve()
    fields, master_rows = _read(master_path)
    _, queue_rows = _read(queue_path)
    merged, counts = merge_rows(master_rows, queue_rows)
    output_path = Path(args.output).resolve()
    if output_path in {master_path, queue_path}:
        raise ValueError("Output must be a new file; source CSVs are immutable audit inputs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    report = {
        "status": "screening_review_merged",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "master": str(master_path),
        "master_sha256": _sha256(master_path),
        "queue": str(queue_path),
        "queue_sha256": _sha256(queue_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        **counts,
    }
    write_json(args.manifest, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
