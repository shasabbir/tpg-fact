from __future__ import annotations

import argparse
import json

from data.pilot_review import read_review_csv, validate_review_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the human pilot-screening CSV")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    report = validate_review_rows(read_review_csv(args.input))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
