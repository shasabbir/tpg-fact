from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import numpy as np

from trace_scv.cefes import select_cefes_prefix
from trace_scv.data.io import sha256_file


VERDICT_LABELS = ("nei", "refuted", "supported")


@dataclass(frozen=True, slots=True)
class MochegVerifierConfig:
    learning_rate: float = 0.05
    epochs: int = 3000
    l2: float = 1e-3
    bootstrap_samples: int = 10_000
    noninferiority_margin: float = -0.01


@dataclass(frozen=True, slots=True)
class MulticlassLogisticVerifier:
    labels: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: Sequence[str],
        *,
        learning_rate: float,
        epochs: int,
        l2: float,
    ) -> "MulticlassLogisticVerifier":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or not len(values) or len(values) != len(labels):
            raise ValueError("verifier features and labels must be non-empty and aligned")
        unknown = sorted(set(labels) - set(VERDICT_LABELS))
        if unknown:
            raise ValueError(f"unknown verdict labels: {unknown}")
        targets = np.asarray([VERDICT_LABELS.index(label) for label in labels])
        if set(np.unique(targets)) != set(range(len(VERDICT_LABELS))):
            raise ValueError("verifier training requires every verdict class")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (values - mean) / scale
        weights = np.zeros((values.shape[1], len(VERDICT_LABELS)), dtype=np.float64)
        bias = np.zeros(len(VERDICT_LABELS), dtype=np.float64)
        counts = np.bincount(targets, minlength=len(VERDICT_LABELS)).astype(np.float64)
        sample_weight = len(targets) / (len(VERDICT_LABELS) * counts[targets])
        denominator = float(sample_weight.sum())
        one_hot = np.eye(len(VERDICT_LABELS), dtype=np.float64)[targets]
        for epoch in range(epochs):
            logits = standardized @ weights + bias
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(np.clip(logits, -40.0, 40.0))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            residual = (probabilities - one_hot) * sample_weight[:, None]
            gradient = standardized.T @ residual / denominator + l2 * weights
            bias_gradient = residual.sum(axis=0) / denominator
            step = learning_rate / math.sqrt(1.0 + epoch / 300.0)
            weights -= step * gradient
            bias -= step * bias_gradient
        return cls(
            labels=VERDICT_LABELS,
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
            weights=tuple(tuple(float(value) for value in row) for row in weights),
            bias=tuple(float(value) for value in bias),
        )

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        standardized = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        logits = standardized @ np.asarray(self.weights) + np.asarray(self.bias)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(np.clip(logits, -40.0, 40.0))
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_claim_labels(processed_root: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for split in ("train", "validation"):
        for row in _read_jsonl(processed_root / f"{split}_claims.jsonl"):
            result[(split, str(row["claim_id"]))] = str(row["label"])
    return result


def _feature_index(feature_path: Path) -> dict[tuple[str, str, str], np.ndarray]:
    arrays = np.load(feature_path, allow_pickle=False)
    features = np.asarray(arrays["features"], dtype=np.float64)
    sidecar = json.loads(feature_path.with_suffix(".json").read_text(encoding="utf-8"))
    records = list(sidecar["records"])
    if len(records) != len(features):
        raise ValueError("feature records are misaligned")
    return {
        (str(row["split"]), str(row["claim_id"]), str(row["evidence_id"])): features[index]
        for index, row in enumerate(records)
    }


def _selected_rows(
    rows: Sequence[dict[str, object]],
    method: str,
    *,
    fixed_k: int,
    training: bool,
) -> list[dict[str, object]]:
    if not training:
        field = {
            "cefes": "selected_cefes",
            "fixed": "selected_fixed",
            "threshold": "selected_threshold_0_5",
            "uncalibrated_cefes": "selected_uncalibrated_cefes",
        }[method]
        selected = [row for row in rows if bool(row[field])]
        if not selected:
            raise ValueError(f"saved {method} selection is empty")
        return selected
    probability_field = (
        "raw_probability" if method == "uncalibrated_cefes" else "calibrated_probability"
    )
    probabilities = [float(row[probability_field]) for row in rows]
    order = sorted(range(len(rows)), key=lambda index: (-probabilities[index], index))
    if method == "fixed":
        selected_indices = order[:fixed_k]
    elif method == "threshold":
        selected_indices = [index for index in order if probabilities[index] >= 0.5]
        if not selected_indices:
            selected_indices = order[:1]
    elif method in {"cefes", "uncalibrated_cefes"}:
        selected_indices = list(
            select_cefes_prefix(probabilities, allow_empty=False, max_items=60).selected_indices
        )
    else:
        raise ValueError(f"unknown verifier selection method: {method}")
    return [rows[index] for index in selected_indices]


def _aggregate_selection(
    rows: Sequence[dict[str, object]],
    feature_index: dict[tuple[str, str, str], np.ndarray],
) -> np.ndarray:
    pair_features = np.stack(
        [
            feature_index[
                (str(row["split"]), str(row["claim_id"]), str(row["evidence_id"]))
            ]
            for row in rows
        ]
    )
    calibrated = np.asarray(
        [float(row["calibrated_probability"]) for row in rows], dtype=np.float64
    )
    raw = np.asarray([float(row["raw_probability"]) for row in rows], dtype=np.float64)
    modalities = [str(row["modality"]) for row in rows]
    return np.concatenate(
        [
            pair_features.mean(axis=0),
            pair_features.max(axis=0),
            pair_features.std(axis=0),
            np.asarray(
                [
                    math.log1p(len(rows)) / 10.0,
                    modalities.count("text") / len(rows),
                    modalities.count("image") / len(rows),
                    calibrated.mean(),
                    calibrated.max(),
                    raw.mean(),
                    raw.max(),
                ],
                dtype=np.float64,
            ),
        ]
    )


def _macro_f1(gold: Sequence[str], predicted: Sequence[str]) -> float:
    scores: list[float] = []
    for label in VERDICT_LABELS:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return fmean(scores)


def _paired_macro_f1_bootstrap(
    gold: Sequence[str],
    left: Sequence[str],
    right: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    generator = np.random.default_rng(seed)
    values: list[float] = []
    gold_array = np.asarray(gold, dtype=object)
    left_array = np.asarray(left, dtype=object)
    right_array = np.asarray(right, dtype=object)
    for _ in range(samples):
        indices = generator.integers(0, len(gold), size=len(gold))
        values.append(
            _macro_f1(gold_array[indices], left_array[indices])
            - _macro_f1(gold_array[indices], right_array[indices])
        )
    return {
        "mean": _macro_f1(gold, left) - _macro_f1(gold, right),
        "lower": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 0.975)),
        "sample_count": samples,
        "resampling_seed": seed,
        "resampling_unit": "claim",
        "samples": values,
    }


def run_matched_mocheg_verifier(
    feature_path: Path,
    selector_dir: Path,
    processed_root: Path,
    output_dir: Path,
    *,
    config: MochegVerifierConfig = MochegVerifierConfig(),
) -> dict[str, object]:
    started = time.perf_counter()
    selector_summary_path = selector_dir / "summary.json"
    selector_summary = json.loads(selector_summary_path.read_text(encoding="utf-8"))
    if not selector_summary["selector_numeric_criteria_met"]:
        raise RuntimeError("matched verifier requires a passing selector numeric gate")
    if selector_summary["formal_test_annotations_accessed"]:
        raise RuntimeError("refusing selector outputs that accessed formal-test annotations")
    feature_index = _feature_index(feature_path)
    claim_labels = _load_claim_labels(processed_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, object]] = []
    verdict_effects: list[float] = []
    methods = ("cefes", "fixed", "threshold", "uncalibrated_cefes")
    for selector_seed in selector_summary["seeds"]:
        seed = int(selector_seed["seed"])
        fixed_k = int(selector_seed["selected_fixed_budget"])
        candidate_path = selector_dir / f"seed_{seed}_candidate_records.jsonl"
        candidates = _read_jsonl(candidate_path)
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in candidates:
            grouped.setdefault((str(row["split"]), str(row["claim_id"])), []).append(row)

        training_features: list[np.ndarray] = []
        training_labels: list[str] = []
        training_claims: set[str] = set()
        for (split, claim_id), rows in sorted(grouped.items()):
            if split != "train" or str(rows[0]["role"]) != "fit":
                continue
            training_claims.add(claim_id)
            seen_selections: set[tuple[str, ...]] = set()
            for method in methods:
                selected = _selected_rows(rows, method, fixed_k=fixed_k, training=True)
                identity = tuple(sorted(str(row["evidence_id"]) for row in selected))
                if identity in seen_selections:
                    continue
                seen_selections.add(identity)
                training_features.append(_aggregate_selection(selected, feature_index))
                training_labels.append(claim_labels[(split, claim_id)])
        verifier = MulticlassLogisticVerifier.fit(
            np.stack(training_features),
            training_labels,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            l2=config.l2,
        )
        checkpoint_path = output_dir / f"seed_{seed}_verifier.json"
        checkpoint_path.write_text(
            json.dumps({"config": asdict(config), "model": asdict(verifier)}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        checkpoint_sha256 = sha256_file(checkpoint_path)

        validation_keys = sorted(key for key in grouped if key[0] == "validation")
        gold = [claim_labels[key] for key in validation_keys]
        predictions: dict[str, list[str]] = {}
        probabilities_by_method: dict[str, np.ndarray] = {}
        selected_ids_by_method: dict[str, list[list[str]]] = {}
        for method in methods:
            aggregate_rows: list[np.ndarray] = []
            selected_ids: list[list[str]] = []
            for key in validation_keys:
                selected = _selected_rows(
                    grouped[key], method, fixed_k=fixed_k, training=False
                )
                aggregate_rows.append(_aggregate_selection(selected, feature_index))
                selected_ids.append(sorted(str(row["evidence_id"]) for row in selected))
            probabilities = verifier.predict_probabilities(np.stack(aggregate_rows))
            probabilities_by_method[method] = probabilities
            predictions[method] = [
                VERDICT_LABELS[int(index)] for index in probabilities.argmax(axis=1)
            ]
            selected_ids_by_method[method] = selected_ids

        method_metrics = {
            method: {
                "macro_f1": _macro_f1(gold, predictions[method]),
                "accuracy": fmean(
                    float(gold_label == predicted_label)
                    for gold_label, predicted_label in zip(
                        gold, predictions[method], strict=True
                    )
                ),
            }
            for method in methods
        }
        paired = _paired_macro_f1_bootstrap(
            gold,
            predictions["cefes"],
            predictions["fixed"],
            samples=config.bootstrap_samples,
            seed=seed,
        )
        verdict_effects.append(
            method_metrics["cefes"]["macro_f1"] - method_metrics["fixed"]["macro_f1"]
        )
        prediction_path = output_dir / f"seed_{seed}_verdict_records.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for index, (_, claim_id) in enumerate(validation_keys):
                handle.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "claim_id": claim_id,
                            "gold_label": gold[index],
                            "checkpoint_sha256": checkpoint_sha256,
                            "methods": {
                                method: {
                                    "predicted_label": predictions[method][index],
                                    "probabilities": {
                                        label: float(probabilities_by_method[method][index, label_index])
                                        for label_index, label in enumerate(VERDICT_LABELS)
                                    },
                                    "selected_evidence_ids": selected_ids_by_method[method][index],
                                }
                                for method in methods
                            },
                        }
                    )
                    + "\n"
                )
        seed_results.append(
            {
                "seed": seed,
                "selected_fixed_budget": fixed_k,
                "training_claims": len(training_claims),
                "training_examples": len(training_features),
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                    "sha256": checkpoint_sha256,
                    "shared_across_methods": True,
                },
                "predictions": {
                    "path": str(prediction_path),
                    "bytes": prediction_path.stat().st_size,
                    "sha256": sha256_file(prediction_path),
                },
                "validation": method_metrics,
                "cefes_minus_fixed_macro_f1_bootstrap": paired,
            }
        )

    mean_effect = fmean(verdict_effects)
    result = {
        "profile": "mocheg_matched_verifier_development",
        "config": asdict(config),
        "feature_sha256": sha256_file(feature_path),
        "selector_summary_sha256": sha256_file(selector_summary_path),
        "seeds": seed_results,
        "mean_cefes_minus_fixed_macro_f1": mean_effect,
        "verdict_noninferiority_point_pass": mean_effect >= config.noninferiority_margin,
        "shared_checkpoint_each_seed": all(
            row["checkpoint"]["shared_across_methods"] for row in seed_results
        ),
        "formal_test_annotations_accessed": False,
        "runtime": {"wall_seconds": time.perf_counter() - started},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
