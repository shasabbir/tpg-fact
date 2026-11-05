from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


CALIBRATOR_KINDS = (
    "identity",
    "temperature",
    "prior_shift",
    "platt",
    "beta",
    "isotonic",
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


@dataclass(frozen=True, slots=True)
class ProbabilityCalibrator:
    kind: str
    coefficients: tuple[float, ...] = ()
    thresholds: tuple[float, ...] = ()
    values: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_calibrator(
    probabilities: Sequence[float] | np.ndarray,
    calibrator: ProbabilityCalibrator,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if calibrator.kind == "identity":
        calibrated = values
    elif calibrator.kind == "temperature":
        temperature = calibrator.coefficients[0]
        calibrated = _sigmoid(_logit(values) / temperature)
    elif calibrator.kind == "prior_shift":
        intercept = calibrator.coefficients[0]
        calibrated = _sigmoid(_logit(values) + intercept)
    elif calibrator.kind == "platt":
        slope, intercept = calibrator.coefficients
        calibrated = _sigmoid(slope * _logit(values) + intercept)
    elif calibrator.kind == "beta":
        a, b, intercept = calibrator.coefficients
        clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
        calibrated = _sigmoid(
            a * np.log(clipped) + b * np.log1p(-clipped) + intercept
        )
    elif calibrator.kind == "isotonic":
        thresholds = np.asarray(calibrator.thresholds, dtype=np.float64)
        fitted = np.asarray(calibrator.values, dtype=np.float64)
        indices = np.searchsorted(thresholds, values, side="left")
        indices = np.clip(indices, 0, len(fitted) - 1)
        calibrated = fitted[indices]
    else:
        raise ValueError(f"unknown calibrator kind: {calibrator.kind}")
    return np.clip(calibrated, 1e-7, 1.0 - 1e-7)


def _fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
    candidates = np.exp(np.linspace(math.log(0.05), math.log(20.0), 161))
    best_temperature = 1.0
    best_nll = math.inf
    logits = _logit(probabilities)
    for temperature in candidates:
        calibrated = _sigmoid(logits / float(temperature))
        nll = _binary_nll(calibrated, labels)
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return ProbabilityCalibrator("temperature", (best_temperature,))


def _fit_prior_shift(probabilities: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
    target = float(np.mean(labels))
    logits = _logit(probabilities)
    lower, upper = -30.0, 30.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if float(np.mean(_sigmoid(logits + middle))) > target:
            upper = middle
        else:
            lower = middle
    return ProbabilityCalibrator("prior_shift", ((lower + upper) / 2.0,))


def _fit_logistic_map(
    design: np.ndarray,
    labels: np.ndarray,
    initial: np.ndarray,
    *,
    l2: float = 1e-6,
    iterations: int = 100,
) -> np.ndarray:
    weights = np.asarray(initial, dtype=np.float64).copy()
    regularizer = np.full(len(weights), l2, dtype=np.float64)
    regularizer[-1] = 0.0
    for _ in range(iterations):
        predicted = _sigmoid(design @ weights)
        gradient = design.T @ (predicted - labels) / len(labels)
        gradient += regularizer * weights
        variance = np.maximum(predicted * (1.0 - predicted), 1e-8)
        hessian = (design.T * variance) @ design / len(labels)
        hessian += np.diag(regularizer + 1e-8)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        weights -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return weights


def _fit_platt(probabilities: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
    design = np.column_stack([_logit(probabilities), np.ones(len(probabilities))])
    weights = _fit_logistic_map(design, labels, np.asarray([1.0, 0.0]))
    return ProbabilityCalibrator("platt", tuple(map(float, weights)))


def _fit_beta(probabilities: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    design = np.column_stack(
        [np.log(clipped), np.log1p(-clipped), np.ones(len(clipped))]
    )
    weights = _fit_logistic_map(design, labels, np.asarray([1.0, -1.0, 0.0]))
    return ProbabilityCalibrator("beta", tuple(map(float, weights)))


def _fit_isotonic(probabilities: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
    order = np.argsort(probabilities, kind="stable")
    ordered_p = probabilities[order]
    ordered_y = labels[order]
    unique_p, starts, counts = np.unique(
        ordered_p, return_index=True, return_counts=True
    )
    sums = np.add.reduceat(ordered_y, starts).astype(np.float64)
    blocks: list[list[float]] = []
    for probability, positive, count in zip(unique_p, sums, counts, strict=True):
        blocks.append([float(probability), float(positive), float(count)])
        while len(blocks) >= 2:
            previous = blocks[-2][1] / blocks[-2][2]
            current = blocks[-1][1] / blocks[-1][2]
            if previous <= current:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([right[0], left[1] + right[1], left[2] + right[2]])
    thresholds = tuple(block[0] for block in blocks)
    fitted = tuple(block[1] / block[2] for block in blocks)
    return ProbabilityCalibrator("isotonic", thresholds=thresholds, values=fitted)


def fit_calibrator(
    kind: str,
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> ProbabilityCalibrator:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if kind not in CALIBRATOR_KINDS:
        raise ValueError(f"unknown calibrator kind: {kind}")
    if len(values) != len(targets) or not len(values):
        raise ValueError("calibrator fitting requires aligned non-empty inputs")
    if len(np.unique(targets)) < 2:
        return ProbabilityCalibrator("identity")
    if kind == "identity":
        return ProbabilityCalibrator("identity")
    if kind == "temperature":
        return _fit_temperature(values, targets)
    if kind == "prior_shift":
        return _fit_prior_shift(values, targets)
    if kind == "platt":
        return _fit_platt(values, targets)
    if kind == "beta":
        return _fit_beta(values, targets)
    return _fit_isotonic(values, targets)


def _binary_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    )
