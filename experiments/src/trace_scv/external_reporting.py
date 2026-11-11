from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean

import numpy as np

from trace_scv.evaluation.statistics import holm_adjust


def set_metrics(predicted: set[str], gold: set[str]) -> dict[str, float]:
    overlap = len(predicted & gold)
    denominator = len(predicted) + len(gold)
    return {
        "precision": overlap / len(predicted) if predicted else 0.0,
        "recall": overlap / len(gold) if gold else 0.0,
        "f1": 2.0 * overlap / denominator if denominator else 1.0,
        "exact": float(predicted == gold),
        "selected_count": float(len(predicted)),
    }


def summarize_claim_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in ("precision", "recall", "f1", "exact", "selected_count")}
    return {
        key: fmean(float(row[key]) for row in rows)
        for key in ("precision", "recall", "f1", "exact", "selected_count")
    }


def paired_claim_bootstrap(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    *,
    metric: str = "f1",
    samples: int = 10_000,
    seed: int = 13,
) -> dict[str, float | int]:
    right_by_id = {str(row["claim_id"]): float(row[metric]) for row in right}
    left_ids = [str(row["claim_id"]) for row in left]
    if set(left_ids) != set(right_by_id):
        raise ValueError("paired bootstrap requires identical claim identifiers")
    differences = np.asarray(
        [float(row[metric]) - right_by_id[str(row["claim_id"])] for row in left],
        dtype=np.float64,
    )
    if not len(differences):
        raise ValueError("paired bootstrap requires at least one claim")
    generator = np.random.default_rng(seed)
    bootstrapped: list[np.ndarray] = []
    remaining = samples
    while remaining:
        current = min(512, remaining)
        indices = generator.integers(0, len(differences), size=(current, len(differences)))
        bootstrapped.append(differences[indices].mean(axis=1))
        remaining -= current
    values = np.concatenate(bootstrapped)
    p_raw = min(
        1.0,
        2.0
        * min(
            (np.count_nonzero(values <= 0.0) + 1) / (len(values) + 1),
            (np.count_nonzero(values >= 0.0) + 1) / (len(values) + 1),
        ),
    )
    return {
        "mean": float(differences.mean()),
        "lower": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 0.975)),
        "p_raw": float(p_raw),
        "sample_count": int(samples),
        "resampling_seed": int(seed),
    }


def apply_holm_to_intervals(intervals: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    adjusted = holm_adjust({key: float(value["p_raw"]) for key, value in intervals.items()})
    return {
        key: {**dict(value), "p_holm": float(adjusted[key])}
        for key, value in intervals.items()
    }
