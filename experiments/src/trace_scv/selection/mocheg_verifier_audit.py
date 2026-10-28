from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

import numpy as np

from trace_scv.data.io import sha256_file


LABELS = ("nei", "refuted", "supported")
METHODS = ("cefes", "fixed", "threshold", "uncalibrated_cefes")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _macro_f1(gold: list[str], predicted: list[str]) -> float:
    values: list[float] = []
    for label in LABELS:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.0)
    return fmean(values)


def _bootstrap(gold: list[str], left: list[str], right: list[str], *, seed: int, samples: int) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    gold_array = np.asarray(gold, dtype=object)
    left_array = np.asarray(left, dtype=object)
    right_array = np.asarray(right, dtype=object)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample = generator.integers(0, len(gold), size=len(gold))
        values[index] = _macro_f1(gold_array[sample].tolist(), left_array[sample].tolist()) - _macro_f1(
            gold_array[sample].tolist(), right_array[sample].tolist()
        )
    return {
        "mean": _macro_f1(gold, left) - _macro_f1(gold, right),
        "lower": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 0.975)),
        "sample_count": int(samples),
        "samples_sha256": __import__("hashlib").sha256(values.tobytes()).hexdigest(),
    }


def audit_matched_mocheg_verifier(output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    discrepancies: list[str] = []
    recomputed_seeds: list[dict[str, object]] = []
    effects: list[float] = []
    tolerance = 1e-12
    for seed_row in summary["seeds"]:
        seed = int(seed_row["seed"])
        checkpoint_path = output_dir / f"seed_{seed}_verifier.json"
        record_path = output_dir / f"seed_{seed}_verdict_records.jsonl"
        checkpoint_hash = sha256_file(checkpoint_path)
        record_hash = sha256_file(record_path)
        if checkpoint_hash != seed_row["checkpoint"]["sha256"]:
            discrepancies.append(f"seed {seed}: checkpoint hash mismatch")
        if record_hash != seed_row["predictions"]["sha256"]:
            discrepancies.append(f"seed {seed}: prediction hash mismatch")
        rows = _read_jsonl(record_path)
        claim_ids = [str(row["claim_id"]) for row in rows]
        if len(rows) != 200 or len(set(claim_ids)) != len(rows):
            discrepancies.append(f"seed {seed}: expected 200 unique validation claims")
        if claim_ids != sorted(claim_ids):
            discrepancies.append(f"seed {seed}: validation claims are not stably ordered")
        gold = [str(row["gold_label"]) for row in rows]
        predicted: dict[str, list[str]] = {method: [] for method in METHODS}
        for row in rows:
            if row["checkpoint_sha256"] != checkpoint_hash:
                discrepancies.append(f"seed {seed}: record checkpoint hash mismatch")
                break
            for method in METHODS:
                result = row["methods"][method]
                probabilities = result["probabilities"]
                values = [float(probabilities[label]) for label in LABELS]
                if abs(sum(values) - 1.0) > tolerance:
                    discrepancies.append(f"seed {seed}: {method} probabilities do not sum to one")
                predicted_label = str(result["predicted_label"])
                if predicted_label != LABELS[int(np.argmax(values))]:
                    discrepancies.append(f"seed {seed}: {method} label is not probability argmax")
                if not result["selected_evidence_ids"]:
                    discrepancies.append(f"seed {seed}: {method} has an empty selection")
                predicted[method].append(predicted_label)
        metrics: dict[str, dict[str, float]] = {}
        for method in METHODS:
            macro_f1 = _macro_f1(gold, predicted[method])
            accuracy = fmean(float(g == p) for g, p in zip(gold, predicted[method], strict=True))
            metrics[method] = {"macro_f1": macro_f1, "accuracy": accuracy}
            stored = seed_row["validation"][method]
            if abs(macro_f1 - float(stored["macro_f1"])) > tolerance or abs(accuracy - float(stored["accuracy"])) > tolerance:
                discrepancies.append(f"seed {seed}: {method} metric mismatch")
        stored_bootstrap = seed_row["cefes_minus_fixed_macro_f1_bootstrap"]
        bootstrap = _bootstrap(
            gold,
            predicted["cefes"],
            predicted["fixed"],
            seed=seed,
            samples=int(stored_bootstrap["sample_count"]),
        )
        stored_samples = np.asarray(stored_bootstrap["samples"], dtype=np.float64)
        stored_samples_hash = __import__("hashlib").sha256(stored_samples.tobytes()).hexdigest()
        if bootstrap["samples_sha256"] != stored_samples_hash:
            discrepancies.append(f"seed {seed}: bootstrap samples mismatch")
        for key in ("mean", "lower", "median", "upper"):
            if abs(float(bootstrap[key]) - float(stored_bootstrap[key])) > tolerance:
                discrepancies.append(f"seed {seed}: bootstrap {key} mismatch")
        effect = metrics["cefes"]["macro_f1"] - metrics["fixed"]["macro_f1"]
        effects.append(effect)
        recomputed_seeds.append(
            {
                "seed": seed,
                "claims": len(rows),
                "checkpoint_sha256": checkpoint_hash,
                "predictions_sha256": record_hash,
                "validation": metrics,
                "cefes_minus_fixed_macro_f1_bootstrap": bootstrap,
            }
        )
    mean_effect = fmean(effects)
    if abs(mean_effect - float(summary["mean_cefes_minus_fixed_macro_f1"])) > tolerance:
        discrepancies.append("mean verdict effect mismatch")
    expected_noninferiority = mean_effect >= float(summary["config"]["noninferiority_margin"])
    if expected_noninferiority != bool(summary["verdict_noninferiority_point_pass"]):
        discrepancies.append("verdict noninferiority gate mismatch")
    if bool(summary["formal_test_annotations_accessed"]):
        discrepancies.append("formal-test access flag is true")
    result = {
        "profile": "mocheg_matched_verifier_independent_audit",
        "status": "pass" if not discrepancies else "fail",
        "summary_sha256": sha256_file(summary_path),
        "discrepancies": discrepancies,
        "recomputed": {
            "seeds": recomputed_seeds,
            "mean_cefes_minus_fixed_macro_f1": mean_effect,
            "verdict_noninferiority_point_pass": expected_noninferiority,
            "shared_checkpoint_each_seed": all(bool(row["checkpoint"]["shared_across_methods"]) for row in summary["seeds"]),
            "formal_test_annotations_accessed": False,
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
