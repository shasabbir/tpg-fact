from __future__ import annotations

import numpy as np

from trace_scv.selection.probability_calibration import (
    CALIBRATOR_KINDS,
    apply_calibrator,
    fit_calibrator,
)


def test_prior_shift_restores_rare_positive_prevalence() -> None:
    probabilities = np.asarray([0.8] * 10 + [0.3] * 90)
    labels = np.asarray([1] * 5 + [0] * 95)
    calibrator = fit_calibrator("prior_shift", probabilities, labels)
    corrected = apply_calibrator(probabilities, calibrator)
    assert abs(float(corrected.mean()) - float(labels.mean())) < 1e-8


def test_declared_calibrators_return_bounded_probabilities() -> None:
    probabilities = np.asarray([0.05, 0.1, 0.2, 0.7, 0.8, 0.95])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    for kind in CALIBRATOR_KINDS:
        fitted = fit_calibrator(kind, probabilities, labels)
        calibrated = apply_calibrator(probabilities, fitted)
        assert calibrated.shape == probabilities.shape
        assert np.all((calibrated > 0.0) & (calibrated < 1.0))


def test_isotonic_output_is_monotone() -> None:
    probabilities = np.linspace(0.01, 0.99, 40)
    labels = np.asarray(([0, 1, 0, 0, 1] * 8), dtype=np.int8)
    fitted = fit_calibrator("isotonic", probabilities, labels)
    calibrated = apply_calibrator(probabilities, fitted)
    assert np.all(np.diff(calibrated) >= -1e-12)
