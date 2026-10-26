from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from data.factcheck_api import (
    FactCheckApiClient,
    candidates_from_pages,
    utc_collection_time,
)
from data.io import write_json, write_jsonl
from data.pilot_review import write_review_csv


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "all"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect paginated Google Fact Check API candidates for manual pilot review"
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--publisher-site", action="append", default=[])
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--api-key-env", default="GOOGLE_FACTCHECK_API_KEY")
    parser.add_argument(
        "--api-key-file",
        default=".secrets/google_factcheck_api_key.txt",
        help="Ignored local text file used only when the environment variable is unset",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    key_file = Path(args.api_key_file)
    if not api_key and key_file.exists():
        api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit(
            f"Missing environment variable {args.api_key_env} and local key file {key_file}. "
            "Set one locally; never place the key in a command, CSV, or tracked Git file."
        )
    if not args.publisher_site and not args.query:
        raise SystemExit("Provide at least one --publisher-site or --query")

    output_root = Path(args.output_directory)
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    client = FactCheckApiClient(api_key)
    combinations: list[tuple[str, str]] = []
    if args.publisher_site:
        queries = args.query or [""]
        combinations.extend((publisher, query) for publisher in args.publisher_site for query in queries)
    else:
        combinations.extend(("", query) for query in args.query)

    candidates_by_id: dict[str, dict[str, str]] = {}
    requests: list[dict[str, object]] = []
    for request_index, (publisher, query) in enumerate(combinations, start=1):
        pages = client.search_pages(
            query=query,
            publisher_site=publisher,
            language_code=args.language_code,
            page_size=args.page_size,
            max_pages=args.max_pages,
        )
        request_slug = f"{request_index:03d}-{_slug(publisher)}-{_slug(query)}"
        for page_index, payload in enumerate(pages, start=1):
            write_json(raw_root / f"{request_slug}-page-{page_index:04d}.json", payload)
        candidates = candidates_from_pages(
            pages,
            collected_query=query,
            collected_publisher_filter=publisher,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        for candidate in candidates:
            candidates_by_id[candidate["candidate_id"]] = candidate
        requests.append(
            {
                "publisher_site": publisher,
                "query": query,
                "page_count": len(pages),
                "candidate_count_before_cross_request_deduplication": len(candidates),
            }
        )

    candidates = [candidates_by_id[key] for key in sorted(candidates_by_id)]
    write_jsonl(output_root / "candidates.jsonl", candidates)
    write_review_csv(output_root / "candidate_review.csv", candidates)
    write_json(
        output_root / "collection_manifest.json",
        {
            "source": "Google Fact Check Tools API v1alpha1 claims.search",
            "collected_at": utc_collection_time(),
            "language_code": args.language_code,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "candidate_count_after_deduplication": len(candidates),
            "requests": requests,
            "api_key_recorded": False,
        },
    )
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "review_csv": str(output_root / "candidate_review.csv"),
                "raw_directory": str(raw_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
