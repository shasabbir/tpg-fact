from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data.io import read_jsonl
from data.schema import ClaimRecord
from data.splits import assert_no_group_leakage, partition


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate claim records and temporal split isolation")
    parser.add_argument("--claims", required=True)
    args = parser.parse_args()

    claims = [ClaimRecord.from_dict(row) for row in read_jsonl(Path(args.claims))]
    assert_no_group_leakage(claims)
    splits = partition(claims)
    summary = {
        "claim_count": len(claims),
        "split_counts": {name: len(records) for name, records in splits.items()},
        "label_counts": dict(Counter(claim.normalized_label for claim in claims)),
        "event_count": len({claim.event_id for claim in claims}),
        "status": "valid",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

