from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from data.io import write_json
from data.pilot_review import read_review_csv
from data.screening import build_screening_queue


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic balanced queue for manual pilot screening"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-size", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    rows = build_screening_queue(
        read_review_csv(input_path), target_size=args.target_size, seed=args.seed
    )
    if not rows:
        raise ValueError("No candidates met the queue prerequisites")
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    label_counts = Counter(row["suggested_label"] for row in rows)
    split_counts = Counter(row["temporal_split"] for row in rows)
    publisher_counts = Counter(row["collected_publisher_filter"] for row in rows)
    manifest = {
        "status": "screening_queue_ready",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": str(input_path),
        "source_sha256": _sha256(input_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "seed": args.seed,
        "requested_size": args.target_size,
        "actual_size": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "temporal_split_counts": dict(sorted(split_counts.items())),
        "publisher_counts": dict(sorted(publisher_counts.items())),
        "interpretation": (
            "Priority signals order manual work only. They do not verify US relevance, "
            "multimodality, target-image identity, timestamps, ratings, or labels."
        ),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
