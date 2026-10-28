from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


_EPSILON = 1e-7


def _validated_binary_inputs(
    probabilities: Iterable[float], labels: Iterable[int]
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    probabilities = tuple(float(value) for value in probabilities)
    labels = tuple(int(value) for value in labels)
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must be non-empty and aligned")
    for index, probability in enumerate(probabilities):
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid probability at index {index}: {probability}")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("binary calibration labels must be 0 or 1")
    return probabilities, labels


def _clamp_probability(probability: float) -> float:
    return min(1.0 - _EPSILON, max(_EPSILON, probability))


def _logit(probability: float) -> float:
    probability = _clamp_probability(probability)
    return math.log(probability) - math.log1p(-probability)


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        inverse = math.exp(-logit)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def binary_nll(probabilities: Iterable[float], labels: Iterable[int]) -> float:
    probabilities, labels = _validated_binary_inputs(probabilities, labels)
    return -sum(
        label * math.log(_clamp_probability(probability))
        + (1 - label) * math.log1p(-_clamp_probability(probability))
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)


def binary_brier_score(probabilities: Iterable[float], labels: Iterable[int]) -> float:
    probabilities, labels = _validated_binary_inputs(probabilities, labels)
    return sum(
        (probability - label) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)


def binary_expected_calibration_error(
    probabilities: Iterable[float], labels: Iterable[int], *, bins: int = 10
) -> float:
    probabilities, labels = _validated_binary_inputs(probabilities, labels)
    if bins <= 0:
        raise ValueError("bins must be positive")
    bucket_probabilities: list[list[float]] = [[] for _ in range(bins)]
    bucket_labels: list[list[int]] = [[] for _ in range(bins)]
    for probability, label in zip(probabilities, labels, strict=True):
        index = min(bins - 1, int(probability * bins))
        bucket_probabilities[index].append(probability)
        bucket_labels[index].append(label)
    total = len(labels)
    return sum(
        len(bucket) / total
        * abs(
            sum(bucket) / len(bucket)
            - sum(bucket_labels[index]) / len(bucket_labels[index])
        )
        for index, bucket in enumerate(bucket_probabilities)
        if bucket
    )


@dataclass(frozen=True, slots=True)
class BinaryTemperatureScaler:
    temperature: float
    fitted_examples: int
    base_nll: float
    calibrated_nll: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

    def transform(self, probabilities: Iterable[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in probabilities)
        for index, value in enumerate(values):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid probability at index {index}: {value}")
        return tuple(_sigmoid(_logit(value) / self.temperature) for value in values)

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def fit(
        cls,
        probabilities: Iterable[float],
        labels: Iterable[int],
        *,
        minimum_temperature: float = 0.05,
        maximum_temperature: float = 20.0,
        grid_points: int = 401,
    ) -> "BinaryTemperatureScaler":
        probabilities, labels = _validated_binary_inputs(probabilities, labels)
        if len(set(labels)) < 2:
            raise ValueError("temperature fitting requires both binary classes")
        if (
            minimum_temperature <= 0.0
            or maximum_temperature <= minimum_temperature
            or grid_points < 3
        ):
            raise ValueError("invalid temperature search range")

        logits = tuple(_logit(value) for value in probabilities)

        def objective(log_temperature: float) -> float:
            temperature = math.exp(log_temperature)
            calibrated = tuple(_sigmoid(value / temperature) for value in logits)
            return binary_nll(calibrated, labels)

        low = math.log(minimum_temperature)
        high = math.log(maximum_temperature)
        step = (high - low) / (grid_points - 1)
        grid = [low + index * step for index in range(grid_points)]
        losses = [objective(value) for value in grid]
        best_index = min(range(grid_points), key=losses.__getitem__)

        # Refine only inside the neighboring grid interval. Golden-section search
        # is deterministic and avoids an additional SciPy dependency.
        left = grid[max(0, best_index - 1)]
        right = grid[min(grid_points - 1, best_index + 1)]
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        inner_left = right - ratio * (right - left)
        inner_right = left + ratio * (right - left)
        left_loss = objective(inner_left)
        right_loss = objective(inner_right)
        for _ in range(80):
            if left_loss <= right_loss:
                right = inner_right
                inner_right = inner_left
                right_loss = left_loss
                inner_left = right - ratio * (right - left)
                left_loss = objective(inner_left)
            else:
                left = inner_left
                inner_left = inner_right
                left_loss = right_loss
                inner_right = left + ratio * (right - left)
                right_loss = objective(inner_right)
        best_log_temperature = (left + right) / 2.0
        temperature = math.exp(best_log_temperature)
        calibrated = tuple(_sigmoid(value / temperature) for value in logits)
        return cls(
            temperature=temperature,
            fitted_examples=len(labels),
            base_nll=binary_nll(probabilities, labels),
            calibrated_nll=binary_nll(calibrated, labels),
        )
