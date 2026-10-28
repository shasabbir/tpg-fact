from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class CEFESDecision:
    selected_indices: tuple[int, ...]
    selected_count: int
    expected_f1: float
    sorted_indices: tuple[int, ...]
    expected_f1_by_count: tuple[float, ...]


def _validated_probabilities(values: Iterable[float]) -> tuple[float, ...]:
    probabilities = tuple(float(value) for value in values)
    for index, value in enumerate(probabilities):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"probability at index {index} must be finite and in [0, 1], got {value}"
            )
    return probabilities


def poisson_binomial_distribution(probabilities: Iterable[float]) -> tuple[float, ...]:
    """Return P(X=j) for a sum of independent, non-identical Bernoulli variables."""
    probabilities = _validated_probabilities(probabilities)
    distribution = [1.0]
    for probability in probabilities:
        updated = [0.0] * (len(distribution) + 1)
        for successes, mass in enumerate(distribution):
            updated[successes] += mass * (1.0 - probability)
            updated[successes + 1] += mass * probability
        distribution = updated
    return tuple(distribution)


def _prefix_distributions(probabilities: tuple[float, ...]) -> list[tuple[float, ...]]:
    distributions: list[tuple[float, ...]] = [(1.0,)]
    for probability in probabilities:
        previous = distributions[-1]
        current = [0.0] * (len(previous) + 1)
        for successes, mass in enumerate(previous):
            current[successes] += mass * (1.0 - probability)
            current[successes + 1] += mass * probability
        distributions.append(tuple(current))
    return distributions


def _suffix_distributions(probabilities: tuple[float, ...]) -> list[tuple[float, ...]]:
    distributions: list[tuple[float, ...]] = [()] * (len(probabilities) + 1)
    distributions[len(probabilities)] = (1.0,)
    for index in range(len(probabilities) - 1, -1, -1):
        probability = probabilities[index]
        following = distributions[index + 1]
        current = [0.0] * (len(following) + 1)
        for successes, mass in enumerate(following):
            current[successes] += mass * (1.0 - probability)
            current[successes + 1] += mass * probability
        distributions[index] = tuple(current)
    return distributions


def expected_prefix_f1_curve(
    ranked_probabilities: Iterable[float],
    *,
    empty_gold_and_prediction_score: float = 1.0,
) -> tuple[float, ...]:
    """Compute exact expected set-F1 for every prefix under Bernoulli independence.

    The value at index k is E[2 TP_k / (k + G)] for selecting the first k
    ranked items. When both the selected and gold sets are empty, the explicit
    dataset-policy score is used instead of evaluating 0/0.
    """
    probabilities = _validated_probabilities(ranked_probabilities)
    if not 0.0 <= empty_gold_and_prediction_score <= 1.0:
        raise ValueError("empty-set score must be in [0, 1]")
    prefixes = _prefix_distributions(probabilities)
    suffixes = _suffix_distributions(probabilities)
    curve: list[float] = []
    for selected_count in range(len(probabilities) + 1):
        expected = 0.0
        for true_positive_count, inside_mass in enumerate(prefixes[selected_count]):
            for outside_gold_count, outside_mass in enumerate(suffixes[selected_count]):
                gold_count = true_positive_count + outside_gold_count
                probability_mass = inside_mass * outside_mass
                denominator = selected_count + gold_count
                if denominator == 0:
                    score = empty_gold_and_prediction_score
                else:
                    score = 2.0 * true_positive_count / denominator
                expected += probability_mass * score
        curve.append(expected)
    return tuple(curve)


def select_cefes_prefix(
    probabilities: Iterable[float],
    *,
    allow_empty: bool = True,
    max_items: int | None = None,
    empty_gold_and_prediction_score: float = 1.0,
) -> CEFESDecision:
    """Sort by calibrated inclusion probability and select maximum expected F1.

    Ties are resolved toward the smaller evidence set, as frozen in the study
    protocol. Returned indices refer to the caller's original candidate order.
    """
    probabilities = _validated_probabilities(probabilities)
    if not probabilities and not allow_empty:
        raise ValueError("cannot require non-empty selection with no candidates")
    if max_items is not None and max_items < 0:
        raise ValueError("max_items must be non-negative")

    order = tuple(sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index)))
    ranked = tuple(probabilities[index] for index in order)
    curve = expected_prefix_f1_curve(
        ranked,
        empty_gold_and_prediction_score=empty_gold_and_prediction_score,
    )
    upper = len(ranked) if max_items is None else min(max_items, len(ranked))
    lower = 0 if allow_empty else 1
    if upper < lower:
        raise ValueError("max_items excludes every admissible evidence set")

    best_count = lower
    best_score = curve[lower]
    for selected_count in range(lower + 1, upper + 1):
        score = curve[selected_count]
        if score > best_score + 1e-15:
            best_count = selected_count
            best_score = score
    return CEFESDecision(
        selected_indices=order[:best_count],
        selected_count=best_count,
        expected_f1=best_score,
        sorted_indices=order,
        expected_f1_by_count=curve,
    )
