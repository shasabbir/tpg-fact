from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from retrieval.open_archives import find_pre_cutoff_observation, list_common_crawl_indexes


OUTPUT_FIELDS = (
    "evidence_id",
    "claim_id",
    "url",
    "published_at",
    "decision_cutoff",
    "status",
    "archived_at",
    "common_crawl_index",
    "queried_indexes",
    "error",
)


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-check evidence URLs for a pre-cutoff Common Crawl observation"
    )
    parser.add_argument("--input", required=True, help="CSV with evidence_id, claim_id, url, published_at, decision_cutoff")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-indexes", type=int, default=12)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexes = list_common_crawl_indexes()
    results: list[dict[str, str]] = []
    for position, row in enumerate(rows):
        result = {field: "" for field in OUTPUT_FIELDS}
        for field in ("evidence_id", "claim_id", "url", "published_at", "decision_cutoff"):
            result[field] = row.get(field, "").strip()
        try:
            observation, queried = find_pre_cutoff_observation(
                result["url"],
                _datetime(result["published_at"]),
                _datetime(result["decision_cutoff"]),
                indexes=indexes,
                max_indexes=args.max_indexes,
            )
            result["queried_indexes"] = "|".join(queried)
            if observation:
                observed_at = datetime.strptime(
                    observation.timestamp, "%Y%m%d%H%M%S"
                ).replace(tzinfo=timezone.utc)
                result["status"] = "observed_pre_cutoff"
                result["archived_at"] = observed_at.isoformat().replace("+00:00", "Z")
                result["common_crawl_index"] = queried[-1] if queried else ""
            else:
                result["status"] = "no_pre_cutoff_observation"
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
        results.append(result)
        if position + 1 < len(rows) and args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"processed": len(results), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
