from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data.io import read_jsonl, write_json
from data.pilot_feasibility import audit_pilot
from data.pilot_review import read_review_csv
from data.schema import ClaimRecord, EvidenceRecord


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the 200-case pilot separately from the final 2,000-case benchmark gate"
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = read_review_csv(args.review_csv)
    claims = [ClaimRecord.from_dict(row) for row in read_jsonl(args.claims)]
    evidence = (
        [EvidenceRecord.from_dict(row) for row in read_jsonl(args.evidence)]
        if args.evidence
        else []
    )
    report = audit_pilot(
        rows,
        claims,
        evidence,
        target_cases=config["data"].get("audit_sample_size", 200),
        final_target_cases=config["data"]["minimum_total_cases"],
        independent_cluster_target=config["data"]["minimum_independent_cluster_fraction"],
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
