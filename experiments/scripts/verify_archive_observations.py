from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from retrieval.open_archives import (
    find_pre_cutoff_observation,
    list_common_crawl_indexes,
    wayback_pre_cutoff_observation,
)


OUTPUT_FIELDS = (
    "evidence_id",
    "claim_id",
    "url",
    "published_at",
    "decision_cutoff",
    "status",
    "archived_at",
    "common_crawl_index",
    "archive_source",
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Keep completed rows in an existing output and retry only rows whose status is error",
    )
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = Path(args.output)
    completed: list[dict[str, str]] = []
    if args.retry_errors and output.exists():
        with output.open("r", encoding="utf-8-sig", newline="") as stream:
            previous = list(csv.DictReader(stream))
        retry_ids = {
            row.get("evidence_id", "")
            for row in previous
            if row.get("status") == "error"
        }
        completed = [row for row in previous if row.get("status") != "error"]
        rows = [row for row in rows if row.get("evidence_id", "") in retry_ids]
        print(f"retrying {len(rows)} archive errors; preserving {len(completed)} completed rows", flush=True)
    indexes = list_common_crawl_indexes() if args.max_indexes > 0 else []
    def check_row(row: dict[str, str]) -> dict[str, str]:
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
                result["archive_source"] = "common_crawl"
            else:
                wayback = wayback_pre_cutoff_observation(
                    result["url"],
                    _datetime(result["published_at"]),
                    _datetime(result["decision_cutoff"]),
                )
                if wayback:
                    observed_at = datetime.strptime(
                        wayback.timestamp, "%Y%m%d%H%M%S"
                    ).replace(tzinfo=timezone.utc)
                    result["status"] = "observed_pre_cutoff"
                    result["archived_at"] = observed_at.isoformat().replace("+00:00", "Z")
                    result["archive_source"] = "internet_archive_cdx"
                else:
                    result["status"] = "no_pre_cutoff_observation"
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)
        return result

    results: list[dict[str, str]] = list(completed)
    if args.workers <= 1:
        for position, row in enumerate(rows, start=1):
            results.append(check_row(row))
            if position % 25 == 0:
                print(f"checked {position}/{len(rows)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(check_row, row) for row in rows]
            for position, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if position % 25 == 0:
                    print(f"checked {position}/{len(rows)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"processed": len(results), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
