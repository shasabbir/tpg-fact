from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data.feasibility import audit_feasibility
from data.io import read_jsonl, write_json
from data.schema import ClaimRecord, EvidenceRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen benchmark gate")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_config = config["data"]
    claims = [ClaimRecord.from_dict(row) for row in read_jsonl(args.claims)]
    evidence = [EvidenceRecord.from_dict(row) for row in read_jsonl(args.evidence)]
    report = audit_feasibility(
        claims,
        evidence,
        minimum_total_cases=data_config["minimum_total_cases"],
        minimum_per_label=data_config["minimum_per_label"],
        minimum_test_events=data_config["minimum_test_events"],
        minimum_independent_cluster_fraction=data_config[
            "minimum_independent_cluster_fraction"
        ],
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
