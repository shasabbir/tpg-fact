from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from data.factcheck_api import FactCheckApiClient, FactCheckApiError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely verify Fact Check Tools API access without printing the key"
    )
    parser.add_argument("--api-key-env", default="GOOGLE_FACTCHECK_API_KEY")
    parser.add_argument("--api-key-file", default=".secrets/google_factcheck_api_key.txt")
    parser.add_argument("--publisher-site", default="politifact.com")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    key_file = Path(args.api_key_file)
    if not api_key and key_file.exists():
        api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        print(json.dumps({"status": "missing_key"}, indent=2))
        raise SystemExit(1)
    try:
        pages = FactCheckApiClient(api_key, retries=0).search_pages(
            publisher_site=args.publisher_site,
            page_size=1,
            max_pages=1,
        )
    except FactCheckApiError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "status": "ok",
                "publisher_site": args.publisher_site,
                "returned_claims": len(pages[0].get("claims") or []),
                "api_key_printed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
