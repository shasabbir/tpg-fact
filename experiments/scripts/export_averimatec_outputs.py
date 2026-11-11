from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trace_scv.data.averimatec import load_annotations
from trace_scv.external_output import build_official_predictions, write_official_predictions
from trace_scv.external_study import read_jsonl
from trace_scv.external_verifier import aggregate_evidence_features, verifier_from_dict


def _attach_frozen_verdicts(
    candidates: list[dict[str, object]],
    policy_rows: list[dict[str, object]],
    freeze: dict[str, object],
    *,
    seed: int,
) -> list[dict[str, object]]:
    seed_rows = [row for row in freeze["seeds"] if int(row["seed"]) == seed]
    if len(seed_rows) != 1:
        raise ValueError(f"freeze does not contain exactly one row for seed {seed}")
    verifier_freeze = seed_rows[0].get("shared_verifier", {})
    if verifier_freeze.get("status") != "frozen_train_only":
        raise RuntimeError(f"shared verifier is not frozen for seed {seed}")
    verifier = verifier_from_dict(verifier_freeze["model"])
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in candidates:
        if str(row["split"]) != "validation":
            continue
        grouped.setdefault(str(row["claim_id"]), []).append(
            {
                "evidence_id": row["evidence_id"],
                "features": row["features"],
                "modality": row["modality"],
                "source_domain": row.get("source_domain", ""),
            }
        )
    ordered = sorted((dict(row) for row in policy_rows), key=lambda row: str(row["claim_id"]))
    aggregates = []
    for row in ordered:
        claim_id = str(row["claim_id"])
        if claim_id not in grouped:
            raise ValueError(f"candidate rows are missing claim {claim_id}")
        aggregates.append(
            aggregate_evidence_features(grouped[claim_id], row["selected_evidence_ids"])
        )
    probabilities = verifier.predict_probabilities(np.stack(aggregates))
    for index, row in enumerate(ordered):
        row["predicted_verdict"] = verifier.labels[int(probabilities[index].argmax())]
        row["verdict_probabilities"] = {
            label: float(probabilities[index, label_index])
            for label_index, label in enumerate(verifier.labels)
        }
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pinned AVerImaTeC development predictions.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/public/averimatec/raw"))
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/prepared_validation"),
    )
    parser.add_argument(
        "--locked-root",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/locked_development"),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/train_only.freeze.json"),
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
    if any("predicted_verdict" not in row for row in policy_rows):
        freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
        policy_rows = _attach_frozen_verdicts(
            candidates, policy_rows, freeze, seed=args.seed
        )
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
