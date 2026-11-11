from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.data.averimatec import load_annotations
from trace_scv.external_output import build_official_predictions, write_official_predictions
from trace_scv.external_study import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pinned AVerImaTeC development predictions.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/public/averimatec/raw"))
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/prepared"),
    )
    parser.add_argument(
        "--locked-root",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/locked_development"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", default="completeness_aware_switch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    diagnostic_path = args.output.with_suffix(".diagnostic.json")
    if (args.output.exists() or diagnostic_path.exists()) and not args.overwrite:
        raise FileExistsError("refusing to overwrite an existing export")
    annotations = load_annotations(args.raw_root / "val.json", split="validation")
    candidates = [
        row
        for row in read_jsonl(args.prepared_root / "candidate_records.jsonl")
        if row["split"] == "validation"
    ]
    policy_rows = [
        row
        for row in read_jsonl(args.locked_root / "claim_records.jsonl")
        if int(row["seed"]) == args.seed and str(row["policy"]) == args.policy
    ]
    predictions, diagnostics = build_official_predictions(
        annotations, candidates, policy_rows
    )
    write_official_predictions(
        args.output, diagnostic_path, predictions, diagnostics
    )
    print(
        json.dumps(
            {
                "status": "exported_development_predictions",
                "claims": len(predictions),
                "output": str(args.output),
                "diagnostics": str(diagnostic_path),
                "test_annotations_accessed": False,
                "submission_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
