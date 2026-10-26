from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import time

from data.io import write_json
from data.pilot_review import read_review_csv
from retrieval.evidence_candidates import default_evidence_query, gdelt_articles_to_review_rows
from retrieval.open_archives import query_gdelt, utc_collection_time


EVIDENCE_REVIEW_FIELDS = (
    "evidence_id",
    "claim_id",
    "decision",
    "exclusion_reason",
    "url",
    "domain",
    "title",
    "gdelt_seen_at",
    "published_at",
    "decision_cutoff",
    "archived_at",
    "timestamp_confidence",
    "relevance_verified",
    "publication_time_verified",
    "relation",
    "media_type",
    "source_cluster_id",
    "cluster_verified",
    "text_sha256",
    "image_phash",
    "outgoing_links",
    "retrieval_query",
    "reviewer_notes",
)


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _gdelt_time(value: datetime) -> str:
    return value.strftime("%Y%m%d%H%M%S")


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect GDELT evidence candidates for kept pilot claims"
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--exclude-domain", action="append", default=[])
    args = parser.parse_args()

    output_root = Path(args.output_directory)
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    requests: list[dict[str, object]] = []
    kept = [
        row
        for row in read_review_csv(args.review_csv)
        if row.get("decision", "").strip().casefold() == "keep"
    ]
    for position, claim in enumerate(kept):
        claim_id = claim["candidate_id"].strip()
        query = claim.get("evidence_query", "").strip() or default_evidence_query(
            claim.get("claim_text", "")
        )
        cutoff = _datetime(claim["decision_cutoff"])
        claim_time = _datetime(claim["claim_time"])
        start = min(claim_time, cutoff) - timedelta(days=args.lookback_days)
        request_record: dict[str, object] = {
            "claim_id": claim_id,
            "query": query,
            "start_datetime": _gdelt_time(start),
            "end_datetime": _gdelt_time(cutoff),
        }
        try:
            articles = query_gdelt(
                query,
                max_records=args.max_records,
                start_datetime=_gdelt_time(start),
                end_datetime=_gdelt_time(cutoff),
            )
            write_json(
                raw_root / f"{_safe_name(claim_id)}.json",
                {
                    "claim_id": claim_id,
                    "query": query,
                    "collected_at": utc_collection_time(),
                    "articles": articles,
                },
            )
            candidate_rows = gdelt_articles_to_review_rows(
                claim_id,
                claim["decision_cutoff"].strip(),
                query,
                articles,
                excluded_domains=set(args.exclude_domain),
            )
            rows.extend(candidate_rows)
            request_record["status"] = "ok"
            request_record["candidate_count"] = len(candidate_rows)
        except Exception as exc:
            request_record["status"] = "error"
            request_record["error"] = str(exc)
        requests.append(request_record)
        if position + 1 < len(kept) and args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    review_path = output_root / "evidence_review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVIDENCE_REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_root / "collection_manifest.json",
        {
            "source": "GDELT DOC 2.0 candidate discovery",
            "collected_at": utc_collection_time(),
            "request_count": len(requests),
            "evidence_candidate_count": len(rows),
            "requests": requests,
            "warning": "GDELT results are candidates; publication times, relevance, and independence require audit.",
        },
    )
    print(json.dumps({"claims": len(kept), "candidates": len(rows), "review_csv": str(review_path)}, indent=2))


if __name__ == "__main__":
    main()
