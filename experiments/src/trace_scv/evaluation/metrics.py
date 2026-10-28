from __future__ import annotations

from collections import defaultdict
import math
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


def citation_cardinality_metrics(
    predicted: Sequence[Iterable[str]], gold: Sequence[Iterable[str]]
) -> dict[str, float]:
    """Measure whether a system predicts the size of the required evidence set."""
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must have the same length")
    if not predicted:
        return {
            "mae": 0.0,
            "exact_match": 0.0,
            "mean_predicted": 0.0,
            "mean_gold": 0.0,
        }
    predicted_counts = [len(set(row)) for row in predicted]
    gold_counts = [len(set(row)) for row in gold]
    return {
        "mae": fmean(
            abs(left - right)
            for left, right in zip(predicted_counts, gold_counts, strict=True)
        ),
        "exact_match": fmean(
            float(left == right)
            for left, right in zip(predicted_counts, gold_counts, strict=True)
        ),
        "mean_predicted": fmean(predicted_counts),
        "mean_gold": fmean(gold_counts),
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


def classification_report(gold: Sequence[str], predicted: Sequence[str]) -> dict:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    labels = sorted(set(gold) | set(predicted))
    per_class = {}
    recalls = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        recalls.append(recall)
    return {
        "accuracy": _safe_div(sum(g == p for g, p in zip(gold, predicted, strict=True)), len(gold)),
        "macro_f1": label_macro_f1(gold, predicted),
        "balanced_accuracy": fmean(recalls) if recalls else 0.0,
        "per_class": per_class,
    }


def ranked_evidence_metrics(
    ranked: Sequence[Sequence[str]], gold: Sequence[Iterable[str]], k: int = 10
) -> dict[str, float]:
    if len(ranked) != len(gold):
        raise ValueError("ranked and gold must have the same length")
    recalls, average_precisions, ndcgs = [], [], []
    for ranking, gold_ids in zip(ranked, gold, strict=True):
        target = set(gold_ids)
        hits = [1 if item in target else 0 for item in list(ranking[:k])]
        recalls.append(_safe_div(sum(hits), len(target)))
        precisions = [sum(hits[: index + 1]) / (index + 1) for index, hit in enumerate(hits) if hit]
        average_precisions.append(_safe_div(sum(precisions), len(target)))
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(target), k)))
        ndcgs.append(_safe_div(dcg, ideal))
    return {
        f"recall@{k}": fmean(recalls) if recalls else 0.0,
        f"map@{k}": fmean(average_precisions) if average_precisions else 0.0,
        f"ndcg@{k}": fmean(ndcgs) if ndcgs else 0.0,
    }


def probabilistic_metrics(
    probabilities: Sequence[Mapping[str, float]], gold: Sequence[str], bins: int = 10
) -> dict[str, float]:
    if len(probabilities) != len(gold):
        raise ValueError("probabilities and gold must have the same length")
    if not gold:
        return {"brier": 0.0, "nll": 0.0, "ece": 0.0}
    labels = sorted({label for row in probabilities for label in row} | set(gold))
    brier_rows, nll_rows = [], []
    confidences, correctness = [], []
    for row, target in zip(probabilities, gold, strict=True):
        brier_rows.append(sum((float(row.get(label, 0.0)) - float(label == target)) ** 2 for label in labels))
        nll_rows.append(-math.log(max(float(row.get(target, 0.0)), 1e-12)))
        predicted = max(labels, key=lambda label: float(row.get(label, 0.0)))
        confidences.append(float(row.get(predicted, 0.0)))
        correctness.append(float(predicted == target))
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            i for i, value in enumerate(confidences)
            if lower <= value and (value < upper or (index == bins - 1 and value <= upper))
        ]
        if members:
            accuracy = fmean(correctness[i] for i in members)
            confidence = fmean(confidences[i] for i in members)
            ece += len(members) / len(gold) * abs(accuracy - confidence)
    return {"brier": fmean(brier_rows), "nll": fmean(nll_rows), "ece": ece}


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
