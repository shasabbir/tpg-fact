from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.io import write_json
from evaluation.study_validation import validate_study_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refuse scientific result use unless every frozen-run artifact passes"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--frozen-data-directory", required=True)
    parser.add_argument("--results-directory", required=True)
    parser.add_argument("--paper-table-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a blocked report without returning a non-zero exit code",
    )
    args = parser.parse_args()
    report = validate_study_artifacts(
        args.config,
        args.frozen_data_directory,
        args.results_directory,
        args.paper_table_directory,
    )
    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["scientific_claims_allowed"] and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
