from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return _softmax(np.asarray(logits, dtype=float) / float(temperature))


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    values = np.asarray(logits, dtype=float)
    labels = np.asarray(targets, dtype=int)
    candidates = grid if grid is not None else np.geomspace(0.25, 4.0, 161)
    best_temperature, best_loss = 1.0, float("inf")
    for temperature in candidates:
        probabilities = apply_temperature(values, float(temperature))
        loss = -np.log(probabilities[np.arange(len(labels)), labels].clip(1e-12, 1.0)).mean()
        if loss < best_loss:
            best_temperature, best_loss = float(temperature), float(loss)
    return best_temperature


def select_threshold(confidences: np.ndarray, target_coverage: float = 0.80) -> float:
    values = np.asarray(confidences, dtype=float)
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    if values.size == 0:
        raise ValueError("confidences cannot be empty")
    return float(np.quantile(values, 1.0 - target_coverage, method="lower"))

