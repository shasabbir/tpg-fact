from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Mapping, Sequence

import numpy as np

from trace_scv.data.averimatec import OFFICIAL_LABELS


@dataclass(frozen=True, slots=True)
class ExternalVerifierModel:
    labels: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        standardized = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        logits = standardized @ np.asarray(self.weights) + np.asarray(self.bias)
        logits -= logits.max(axis=1, keepdims=True)
        exponential = np.exp(np.clip(logits, -40.0, 40.0))
        return exponential / exponential.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict[str, object]:
        return {"architecture": "linear", **asdict(self)}


@dataclass(frozen=True, slots=True)
class ExternalMLPVerifierModel:
    labels: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    input_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        standardized = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        hidden = np.maximum(
            0.0,
            standardized @ np.asarray(self.input_weights) + np.asarray(self.hidden_bias),
        )
        logits = hidden @ np.asarray(self.output_weights) + np.asarray(self.output_bias)
        logits -= logits.max(axis=1, keepdims=True)
        exponential = np.exp(np.clip(logits, -40.0, 40.0))
        return exponential / exponential.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict[str, object]:
        return {"architecture": "two_layer_mlp", **asdict(self)}


def verifier_from_dict(
    row: Mapping[str, object],
) -> ExternalVerifierModel | ExternalMLPVerifierModel:
    if row.get("architecture") == "two_layer_mlp":
        return ExternalMLPVerifierModel(
            labels=tuple(map(str, row["labels"])),
            mean=tuple(map(float, row["mean"])),
            scale=tuple(map(float, row["scale"])),
            input_weights=tuple(tuple(map(float, values)) for values in row["input_weights"]),
            hidden_bias=tuple(map(float, row["hidden_bias"])),
            output_weights=tuple(tuple(map(float, values)) for values in row["output_weights"]),
            output_bias=tuple(map(float, row["output_bias"])),
        )
    return ExternalVerifierModel(
        labels=tuple(map(str, row["labels"])),
        mean=tuple(map(float, row["mean"])),
        scale=tuple(map(float, row["scale"])),
        weights=tuple(tuple(map(float, values)) for values in row["weights"]),
        bias=tuple(map(float, row["bias"])),
    )


def fit_external_mlp_verifier(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    seed: int,
    hidden_units: int = 64,
    learning_rate: float = 0.01,
    epochs: int = 300,
    l2: float = 1e-3,
) -> ExternalMLPVerifierModel:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray([OFFICIAL_LABELS.index(str(label)) for label in labels], dtype=np.int64)
    if values.ndim != 2 or len(values) != len(targets) or not len(targets):
        raise ValueError("verifier features and labels must be non-empty and aligned")
    present = set(map(int, np.unique(targets)))
    if present != set(range(len(OFFICIAL_LABELS))):
        missing = [OFFICIAL_LABELS[index] for index in range(len(OFFICIAL_LABELS)) if index not in present]
        raise ValueError(f"external verifier training is missing verdict classes: {missing}")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (values - mean) / scale
    class_count = len(OFFICIAL_LABELS)
    generator = np.random.default_rng(20_261_111 + seed)
    input_weights = generator.normal(
        0.0, math.sqrt(2.0 / values.shape[1]), size=(values.shape[1], hidden_units)
    )
    hidden_bias = np.zeros(hidden_units, dtype=np.float64)
    output_weights = generator.normal(
        0.0, math.sqrt(2.0 / hidden_units), size=(hidden_units, class_count)
    )
    output_bias = np.zeros(class_count, dtype=np.float64)
    counts = np.bincount(targets, minlength=class_count).astype(np.float64)
    sample_weights = len(targets) / (class_count * counts[targets])
    one_hot = np.eye(class_count, dtype=np.float64)[targets]
    denominator = float(sample_weights.sum())
    for epoch in range(epochs):
        hidden_pre = standardized @ input_weights + hidden_bias
        hidden = np.maximum(0.0, hidden_pre)
        logits = hidden @ output_weights + output_bias
        logits -= logits.max(axis=1, keepdims=True)
        exponential = np.exp(np.clip(logits, -40.0, 40.0))
        probabilities = exponential / exponential.sum(axis=1, keepdims=True)
        residual = (probabilities - one_hot) * sample_weights[:, None] / denominator
        output_gradient = hidden.T @ residual + l2 * output_weights
        output_bias_gradient = residual.sum(axis=0)
        hidden_gradient = (residual @ output_weights.T) * (hidden_pre > 0.0)
        input_gradient = standardized.T @ hidden_gradient + l2 * input_weights
        hidden_bias_gradient = hidden_gradient.sum(axis=0)
        step = learning_rate / math.sqrt(1.0 + epoch / 250.0)
        output_weights -= step * output_gradient
        output_bias -= step * output_bias_gradient
        input_weights -= step * input_gradient
        hidden_bias -= step * hidden_bias_gradient
    return ExternalMLPVerifierModel(
        labels=tuple(OFFICIAL_LABELS),
        mean=tuple(map(float, mean)),
        scale=tuple(map(float, scale)),
        input_weights=tuple(tuple(map(float, row)) for row in input_weights),
        hidden_bias=tuple(map(float, hidden_bias)),
        output_weights=tuple(tuple(map(float, row)) for row in output_weights),
        output_bias=tuple(map(float, output_bias)),
    )


def aggregate_evidence_features(
    rows: Sequence[Mapping[str, object]], selected_evidence_ids: Sequence[str]
) -> np.ndarray:
    by_id = {str(row["evidence_id"]): row for row in rows}
    selected = [by_id[str(value)] for value in selected_evidence_ids]
    if not selected:
        raise ValueError("verifier requires at least one selected evidence item")
    features = np.asarray([row["features"] for row in selected], dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("verifier evidence features must be finite and rectangular")
    text_count = sum(row["modality"] == "text" for row in selected)
    image_count = sum(row["modality"] == "image" for row in selected)
    domains = {
        str(row.get("source_domain", ""))
        for row in selected
        if str(row.get("source_domain", ""))
    }
    count = len(selected)
    return np.concatenate(
        (
            features.mean(axis=0),
            features.max(axis=0),
            features.std(axis=0),
            np.asarray(
                [
                    math.log1p(count),
                    text_count / count,
                    image_count / count,
                    math.log1p(len(domains)),
                ],
                dtype=np.float64,
            ),
        )
    )


def fit_external_verifier(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    learning_rate: float = 0.05,
    epochs: int = 500,
    l2: float = 1e-3,
) -> ExternalVerifierModel:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray([OFFICIAL_LABELS.index(str(label)) for label in labels], dtype=np.int64)
    if values.ndim != 2 or len(values) != len(targets) or not len(targets):
        raise ValueError("verifier features and labels must be non-empty and aligned")
    present = set(map(int, np.unique(targets)))
    if present != set(range(len(OFFICIAL_LABELS))):
        missing = [OFFICIAL_LABELS[index] for index in range(len(OFFICIAL_LABELS)) if index not in present]
        raise ValueError(f"external verifier training is missing verdict classes: {missing}")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (values - mean) / scale
    class_count = len(OFFICIAL_LABELS)
    weights = np.zeros((values.shape[1], class_count), dtype=np.float64)
    bias = np.zeros(class_count, dtype=np.float64)
    counts = np.bincount(targets, minlength=class_count).astype(np.float64)
    sample_weights = len(targets) / (class_count * counts[targets])
    one_hot = np.eye(class_count, dtype=np.float64)[targets]
    denominator = float(sample_weights.sum())
    for epoch in range(epochs):
        logits = standardized @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        exponential = np.exp(np.clip(logits, -40.0, 40.0))
        probabilities = exponential / exponential.sum(axis=1, keepdims=True)
        residual = (probabilities - one_hot) * sample_weights[:, None]
        step = learning_rate / math.sqrt(1.0 + epoch / 250.0)
        weights -= step * (standardized.T @ residual / denominator + l2 * weights)
        bias -= step * residual.sum(axis=0) / denominator
    return ExternalVerifierModel(
        labels=tuple(OFFICIAL_LABELS),
        mean=tuple(map(float, mean)),
        scale=tuple(map(float, scale)),
        weights=tuple(tuple(map(float, row)) for row in weights),
        bias=tuple(map(float, bias)),
    )


def classification_metrics(
    gold: Sequence[str], predictions: Sequence[str]
) -> dict[str, object]:
    if len(gold) != len(predictions) or not gold:
        raise ValueError("verifier metrics require aligned, non-empty labels")
    per_class: dict[str, dict[str, float]] = {}
    for label in OFFICIAL_LABELS:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predictions, strict=True))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predictions, strict=True))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predictions, strict=True))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(value == label for value in gold)),
        }
    return {
        "accuracy": sum(left == right for left, right in zip(gold, predictions, strict=True)) / len(gold),
        "macro_f1": fmean(row["f1"] for row in per_class.values()),
        "per_class": per_class,
        "claims": len(gold),
    }


def paired_macro_f1_bootstrap(
    gold: Sequence[str],
    left: Sequence[str],
    right: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    if not (len(gold) == len(left) == len(right)) or not gold:
        raise ValueError("paired verifier bootstrap requires aligned, non-empty predictions")
    gold_values = np.asarray(gold, dtype=object)
    left_values = np.asarray(left, dtype=object)
    right_values = np.asarray(right, dtype=object)

    def effect(indices: np.ndarray) -> float:
        selected_gold = gold_values[indices].tolist()
        return float(
            classification_metrics(selected_gold, left_values[indices].tolist())["macro_f1"]
            - classification_metrics(selected_gold, right_values[indices].tolist())["macro_f1"]
        )

    actual = effect(np.arange(len(gold_values)))
    generator = np.random.default_rng(seed)
    values = np.asarray(
        [effect(generator.integers(0, len(gold_values), size=len(gold_values))) for _ in range(samples)],
        dtype=np.float64,
    )
    return {
        "mean": actual,
        "lower": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 0.975)),
        "sample_count": samples,
        "resampling_seed": seed,
    }


def seed_aligned_macro_f1_bootstrap(
    runs: Sequence[tuple[Sequence[str], Sequence[str], Sequence[str]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    if not runs:
        raise ValueError("seed-aligned verifier bootstrap requires at least one run")
    claim_count = len(runs[0][0])
    if claim_count < 1 or any(not (len(gold) == len(left) == len(right) == claim_count) for gold, left, right in runs):
        raise ValueError("seed-aligned verifier runs must have identical lengths")
    arrays = [
        (
            np.asarray(gold, dtype=object),
            np.asarray(left, dtype=object),
            np.asarray(right, dtype=object),
        )
        for gold, left, right in runs
    ]

    def effect(indices: np.ndarray) -> float:
        effects = []
        for gold, left, right in arrays:
            selected_gold = gold[indices].tolist()
            effects.append(
                float(classification_metrics(selected_gold, left[indices].tolist())["macro_f1"])
                - float(classification_metrics(selected_gold, right[indices].tolist())["macro_f1"])
            )
        return fmean(effects)

    actual = effect(np.arange(claim_count))
    generator = np.random.default_rng(seed)
    values = np.asarray(
        [effect(generator.integers(0, claim_count, size=claim_count)) for _ in range(samples)],
        dtype=np.float64,
    )
    return {
        "mean": actual,
        "lower": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 0.975)),
        "sample_count": samples,
        "resampling_seed": seed,
        "seeds": len(runs),
    }
