from __future__ import annotations

import argparse

from retrieval.open_archives import query_gdelt, utc_collection_time
from data.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GDELT candidate metadata for one query")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=100)
    args = parser.parse_args()

    articles = query_gdelt(args.query, max_records=args.max_records)
    write_json(
        args.output,
        {
            "query": args.query,
            "collected_at": utc_collection_time(),
            "source": "GDELT DOC 2.0",
            "articles": articles,
        },
    )
    print(f"saved={len(articles)} output={args.output}")


if __name__ == "__main__":
    main()

