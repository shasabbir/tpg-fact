from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_LABELS = ("supported", "refuted", "insufficient_evidence")


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> dict[str, object]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    per_class: dict[str, dict[str, float]] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    correct = sum(truth == prediction for truth, prediction in zip(y_true, y_pred))
    for label in labels:
        true_positive = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        false_positive = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        false_negative = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        recalls.append(recall)
        f1_values.append(f1)
    return {
        "accuracy": _safe_divide(correct, len(y_true)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "sample_count": len(y_true),
    }


def multiclass_brier(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str] = DEFAULT_LABELS,
) -> float:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.shape != (len(y_true), len(labels)):
        raise ValueError("probability matrix has the wrong shape")
    label_index = {label: index for index, label in enumerate(labels)}
    target = np.zeros_like(matrix)
    for row, label in enumerate(y_true):
        target[row, label_index[label]] = 1.0
    return float(np.mean(np.sum((matrix - target) ** 2, axis=1)))


def negative_log_likelihood(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str] = DEFAULT_LABELS,
) -> float:
    matrix = np.asarray(probabilities, dtype=float)
    label_index = {label: index for index, label in enumerate(labels)}
    losses = [
        -np.log(matrix[row, label_index[label]].clip(1e-12, 1.0))
        for row, label in enumerate(y_true)
    ]
    return float(np.mean(losses))


def _binary_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC with average ranks for tied scores."""
    truth = np.asarray(targets, dtype=bool)
    values = np.asarray(scores, dtype=float)
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[truth].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def macro_ovr_auroc(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str] = DEFAULT_LABELS,
) -> float:
    """Return macro one-vs-rest AUROC over classes represented in the sample."""
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.shape != (len(y_true), len(labels)):
        raise ValueError("probability matrix has the wrong shape")
    scores = [
        _binary_auroc(np.asarray(y_true) == label, matrix[:, index])
        for index, label in enumerate(labels)
    ]
    finite = [score for score in scores if np.isfinite(score)]
    if not finite:
        raise ValueError("AUROC requires at least one class with positive and negative examples")
    return float(np.mean(finite))


def expected_calibration_error(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str] = DEFAULT_LABELS,
    bins: int = 10,
) -> float:
    matrix = np.asarray(probabilities, dtype=float)
    predictions = matrix.argmax(axis=1)
    confidences = matrix.max(axis=1)
    label_index = {label: index for index, label in enumerate(labels)}
    targets = np.array([label_index[label] for label in y_true])
    correctness = predictions == targets
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        error += mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean())
    return float(error)


def risk_coverage_curve(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    confidence_array = np.asarray(confidences, dtype=float)
    order = np.argsort(-confidence_array)
    errors = np.array([t != p for t, p in zip(y_true, y_pred)], dtype=float)[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return coverage, cumulative_risk


def area_under_risk_coverage(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
) -> float:
    coverage, risk = risk_coverage_curve(y_true, y_pred, confidences)
    return float(np.trapezoid(risk, coverage))


def risk_at_coverage(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
    target_coverage: float = 0.80,
) -> float:
    coverage, risk = risk_coverage_curve(y_true, y_pred, confidences)
    index = int(np.searchsorted(coverage, target_coverage, side="left"))
    return float(risk[min(index, len(risk) - 1)])

