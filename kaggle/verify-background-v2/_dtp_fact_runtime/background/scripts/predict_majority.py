from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from data.io import read_jsonl, write_json
from data.schema import ClaimRecord, LABELS
from data.splits import assert_no_group_leakage, partition


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the frozen majority-class baseline")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    claims = [ClaimRecord.from_dict(row) for row in read_jsonl(args.claims)]
    assert_no_group_leakage(claims)
    splits = partition(claims)
    counts = Counter(row.normalized_label for row in splits["train"])
    if not counts:
        raise ValueError("Training split is empty")
    majority = max(LABELS, key=lambda label: (counts[label], -LABELS.index(label)))
    total = sum(counts.values()) + len(LABELS)
    probabilities = {label: (counts[label] + 1) / total for label in LABELS}
    output_root = Path(args.output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for claim in splits["test"]:
            row = {
                "claim_id": claim.claim_id,
                "predicted_label": majority,
                "class_probabilities": probabilities,
                "sufficient_evidence_score": 1.0 - probabilities["insufficient_evidence"],
                "answer_confidence": probabilities[majority],
                "abstained": False,
                "abstention_threshold": 0.0,
                "selected_evidence_ids": [],
                "source_cluster_ids": [],
                "decision_cutoff": claim.decision_cutoff.isoformat().replace("+00:00", "Z"),
            }
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "status": "completed_deterministic_baseline",
        "majority_label": majority,
        "training_counts": dict(counts),
        "probabilities": probabilities,
        "test_count": len(splits["test"]),
        "predictions": str(prediction_path.resolve()),
    }
    write_json(output_root / "run_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
