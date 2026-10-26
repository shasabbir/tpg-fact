from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np


def event_bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    event_ids: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    replicates: int = 10000,
    seed: int = 2026,
) -> dict[str, float]:
    """Estimate a percentile interval by resampling whole events."""
    if not (len(y_true) == len(y_pred) == len(event_ids)):
        raise ValueError("All inputs must have equal length")
    by_event: dict[str, list[int]] = defaultdict(list)
    for index, event_id in enumerate(event_ids):
        by_event[event_id].append(index)
    events = sorted(by_event)
    if not events:
        raise ValueError("At least one event is required")
    random = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_events = random.choice(events, size=len(events), replace=True)
        indices = [index for event_id in sampled_events for index in by_event[event_id]]
        values[replicate] = metric(
            [y_true[index] for index in indices],
            [y_pred[index] for index in indices],
        )
    return {
        "estimate": float(metric(y_true, y_pred)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def paired_event_bootstrap(
    y_true: Sequence[str],
    first: Sequence[str],
    second: Sequence[str],
    event_ids: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    replicates: int = 10000,
    seed: int = 2026,
) -> dict[str, float]:
    if not (len(y_true) == len(first) == len(second) == len(event_ids)):
        raise ValueError("All inputs must have equal length")
    by_event: dict[str, list[int]] = defaultdict(list)
    for index, event_id in enumerate(event_ids):
        by_event[event_id].append(index)
    events = sorted(by_event)
    random = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_events = random.choice(events, size=len(events), replace=True)
        indices = [index for event_id in sampled_events for index in by_event[event_id]]
        truth = [y_true[index] for index in indices]
        first_values = [first[index] for index in indices]
        second_values = [second[index] for index in indices]
        differences[replicate] = metric(truth, first_values) - metric(truth, second_values)
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "p_value_two_sided": float(
            min(
                1.0,
                2.0
                * min(
                    (np.sum(differences <= 0.0) + 1) / (replicates + 1),
                    (np.sum(differences >= 0.0) + 1) / (replicates + 1),
                ),
            )
        ),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()

