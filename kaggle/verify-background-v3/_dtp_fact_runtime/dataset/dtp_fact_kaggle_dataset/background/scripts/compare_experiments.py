from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.io import read_jsonl, write_json
from data.schema import ClaimRecord, LABELS
from data.splits import split_name
from evaluation.bootstrap import holm_adjust, paired_event_bootstrap
from evaluation.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the completed experiment matrix and compare models by event"
    )
    parser.add_argument("--claims", required=True)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-variant", default="full")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def _prediction_files(root: Path, variant: str) -> list[Path]:
    if variant == "majority":
        path = root / variant / "predictions.jsonl"
        return [path] if path.exists() else []
    return sorted((root / variant).glob("seed_*/predictions.jsonl"))


def _ensemble_predictions(
    files: list[Path], claim_ids: list[str]
) -> tuple[list[str], np.ndarray]:
    if not files:
        raise ValueError("At least one prediction file is required")
    matrices: list[np.ndarray] = []
    for path in files:
        rows = {str(row["claim_id"]): row for row in read_jsonl(path)}
        missing = sorted(set(claim_ids) - set(rows))
        extra = sorted(set(rows) - set(claim_ids))
        if missing or extra:
            raise ValueError(
                f"Prediction mismatch in {path}: missing={missing[:5]} extra={extra[:5]}"
            )
        matrices.append(
            np.asarray(
                [
                    [float(rows[claim_id]["class_probabilities"][label]) for label in LABELS]
                    for claim_id in claim_ids
                ],
                dtype=float,
            )
        )
    probabilities = np.mean(matrices, axis=0)
    predicted = [LABELS[index] for index in probabilities.argmax(axis=1)]
    return predicted, probabilities


def main() -> None:
    args = parse_args()
    root = Path(args.matrix_root).resolve()
    manifest_path = root / "matrix_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Experiment matrix is not complete")
    test_claims = [
        row
        for row in read_jsonl(args.claims)
        if split_name(ClaimRecord.from_dict(row)) == "test"
    ]
    claim_ids = [str(row["claim_id"]) for row in test_claims]
    truth = [str(row["normalized_label"]) for row in test_claims]
    event_ids = [str(row["event_id"]) for row in test_claims]
    variants = ["majority", *[str(value) for value in manifest["variants"]]]
    evaluations: list[dict[str, object]] = []
    predictions: dict[str, list[str]] = {}
    for variant in variants:
        evaluation_path = root / variant / "evaluation.json"
        if not evaluation_path.exists():
            raise FileNotFoundError(evaluation_path)
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        predicted, _ = _ensemble_predictions(_prediction_files(root, variant), claim_ids)
        predictions[variant] = predicted
        evaluations.append(
            {
                "variant": variant,
                "seed_count": evaluation["seed_count"],
                "aggregate": evaluation["aggregate"],
                "ensemble": classification_metrics(truth, predicted),
                "evaluation": str(evaluation_path),
            }
        )
    reference = args.reference_variant
    if reference not in predictions:
        raise ValueError(f"Reference variant is absent: {reference}")
    comparisons: list[dict[str, object]] = []
    p_values: list[float] = []
    for index, variant in enumerate(variants):
        if variant == reference:
            continue
        result = paired_event_bootstrap(
            truth,
            predictions[reference],
            predictions[variant],
            event_ids,
            lambda y_true, y_pred: float(
                classification_metrics(y_true, y_pred)["macro_f1"]
            ),
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + index,
        )
        p_values.append(float(result["p_value_two_sided"]))
        comparisons.append(
            {
                "reference": reference,
                "comparison": variant,
                "metric": "macro_f1",
                **result,
            }
        )
    adjusted = holm_adjust(p_values)
    for row, value in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_p"] = value
    report = {
        "status": "checked_matrix_aggregation",
        "claim_manifest": str(Path(args.claims).resolve()),
        "matrix_manifest": str(manifest_path),
        "test_claims": len(test_claims),
        "test_events": len(set(event_ids)),
        "reference_variant": reference,
        "evaluations": evaluations,
        "paired_event_bootstrap": comparisons,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
