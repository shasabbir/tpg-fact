from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Iterable, Mapping, Sequence


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def citation_macro_prf(
    predicted: Sequence[Iterable[str]],
    gold: Sequence[Iterable[str]],
) -> dict[str, float]:
    """Macro-average set precision, recall, and F1 over claim packets."""
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must have the same length")
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    rows: list[tuple[float, float, float]] = []
    for predicted_ids, gold_ids in zip(predicted, gold, strict=True):
        p_set, g_set = set(predicted_ids), set(gold_ids)
        overlap = len(p_set & g_set)
        precision = _safe_div(overlap, len(p_set))
        recall = _safe_div(overlap, len(g_set))
        f1 = _safe_div(2 * precision * recall, precision + recall)
        rows.append((precision, recall, f1))
    return {
        "precision": fmean(row[0] for row in rows),
        "recall": fmean(row[1] for row in rows),
        "f1": fmean(row[2] for row in rows),
    }


def label_macro_f1(gold: Sequence[str], predicted: Sequence[str]) -> float:
    """Dependency-free macro-F1 over labels present in gold or predictions."""
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    labels = sorted(set(gold) | set(predicted))
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        scores.append(_safe_div(2 * precision * recall, precision + recall))
    return fmean(scores)


def comprehensiveness(
    clean_probabilities: Sequence[Mapping[str, float]],
    deleted_probabilities: Sequence[Mapping[str, float]],
    gold_labels: Sequence[str],
) -> float:
    """Average drop in gold-label probability after cited evidence deletion."""
    if not (
        len(clean_probabilities) == len(deleted_probabilities) == len(gold_labels)
    ):
        raise ValueError("all inputs must have the same length")
    if not gold_labels:
        return 0.0
    drops = [
        float(clean.get(label, 0.0)) - float(deleted.get(label, 0.0))
        for clean, deleted, label in zip(
            clean_probabilities, deleted_probabilities, gold_labels, strict=True
        )
    ]
    return fmean(drops)


def probability_invariance(
    clean: Sequence[Mapping[str, float]],
    perturbed: Sequence[Mapping[str, float]],
) -> float:
    """One minus mean total-variation distance; one is perfectly invariant."""
    if len(clean) != len(perturbed):
        raise ValueError("clean and perturbed must have the same length")
    if not clean:
        return 0.0
    similarities: list[float] = []
    for left, right in zip(clean, perturbed, strict=True):
        labels = set(left) | set(right)
        total_variation = 0.5 * sum(
            abs(float(left.get(label, 0.0)) - float(right.get(label, 0.0)))
            for label in labels
        )
        similarities.append(1.0 - total_variation)
    return fmean(similarities)
