from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.io import read_jsonl, write_json
from data.schema import ClaimRecord, LABELS
from data.splits import split_name
from evaluation.bootstrap import event_bootstrap_ci
from evaluation.metrics import (
    area_under_risk_coverage,
    classification_metrics,
    expected_calibration_error,
    macro_ovr_auroc,
    multiclass_brier,
    negative_log_likelihood,
    risk_at_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more seeded prediction files against a frozen claim manifest"
    )
    parser.add_argument("--claims", required=True, help="Claim JSONL with labels and event IDs")
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Prediction JSONL; repeat once per seed",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-seeds", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        help="Optionally evaluate only one preregistered temporal split",
    )
    return parser.parse_args()


def macro_f1(truth: list[str], prediction: list[str]) -> float:
    return float(classification_metrics(truth, prediction)["macro_f1"])


def evaluate_run(
    claims: list[dict[str, object]],
    prediction_path: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    target_coverage: float,
) -> dict[str, object]:
    prediction_rows = read_jsonl(prediction_path)
    by_id: dict[str, dict[str, object]] = {}
    for row in prediction_rows:
        claim_id = str(row["claim_id"])
        if claim_id in by_id:
            raise ValueError(f"Duplicate prediction for {claim_id} in {prediction_path}")
        by_id[claim_id] = row

    claim_ids = [str(row["claim_id"]) for row in claims]
    missing = sorted(set(claim_ids) - set(by_id))
    extra = sorted(set(by_id) - set(claim_ids))
    if missing or extra:
        raise ValueError(
            f"Prediction/claim mismatch for {prediction_path}: missing={missing[:5]} extra={extra[:5]}"
        )

    truth = [str(row["normalized_label"]) for row in claims]
    event_ids = [str(row["event_id"]) for row in claims]
    predicted = [str(by_id[claim_id]["predicted_label"]) for claim_id in claim_ids]
    probabilities = np.asarray(
        [
            [float(by_id[claim_id]["class_probabilities"][label]) for label in LABELS]
            for claim_id in claim_ids
        ],
        dtype=float,
    )
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"Probabilities do not sum to one in {prediction_path}")
    confidences = probabilities.max(axis=1)
    result = classification_metrics(truth, predicted)
    result.update(
        {
            "prediction_file": str(prediction_path.resolve()),
            "macro_ovr_auroc": macro_ovr_auroc(truth, probabilities),
            "brier": multiclass_brier(truth, probabilities),
            "nll": negative_log_likelihood(truth, probabilities),
            "ece": expected_calibration_error(truth, probabilities),
            "aurc": area_under_risk_coverage(truth, predicted, confidences),
            "risk_at_target_coverage": risk_at_coverage(
                truth, predicted, confidences, target_coverage
            ),
            "target_coverage": target_coverage,
            "macro_f1_event_bootstrap": event_bootstrap_ci(
                truth,
                predicted,
                event_ids,
                macro_f1,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    if args.expected_seeds is not None and len(args.predictions) != args.expected_seeds:
        raise ValueError(
            f"Expected {args.expected_seeds} prediction files, received {len(args.predictions)}"
        )
    claims = read_jsonl(args.claims)
    if args.split:
        claims = [
            row
            for row in claims
            if split_name(ClaimRecord.from_dict(row)) == args.split
        ]
    if not claims:
        raise ValueError("Claim manifest is empty")
    runs = [
        evaluate_run(
            claims,
            Path(path),
            args.bootstrap_replicates,
            args.bootstrap_seed + index,
            args.target_coverage,
        )
        for index, path in enumerate(args.predictions)
    ]
    scalar_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_ovr_auroc",
        "brier",
        "nll",
        "ece",
        "aurc",
        "risk_at_target_coverage",
    )
    aggregate = {
        key: {
            "mean": float(np.mean([float(run[key]) for run in runs])),
            "sample_std": float(np.std([float(run[key]) for run in runs], ddof=1))
            if len(runs) > 1
            else 0.0,
        }
        for key in scalar_keys
    }
    report = {
        "status": "measured_from_supplied_prediction_artifacts",
        "claim_manifest": str(Path(args.claims).resolve()),
        "evaluated_split": args.split or "all_rows",
        "seed_count": len(runs),
        "runs": runs,
        "aggregate": aggregate,
    }
    write_json(args.output, report)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
