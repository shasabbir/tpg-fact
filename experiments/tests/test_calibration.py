import pytest

from trace_scv.selection.calibration import (
    BinaryTemperatureScaler,
    binary_brier_score,
    binary_expected_calibration_error,
    binary_nll,
)


def test_temperature_fit_reduces_overconfident_nll() -> None:
    probabilities = [0.99, 0.95, 0.8, 0.2, 0.05, 0.01]
    labels = [1, 0, 1, 0, 1, 0]

    scaler = BinaryTemperatureScaler.fit(probabilities, labels)
    calibrated = scaler.transform(probabilities)

    assert scaler.temperature > 1.0
    assert binary_nll(calibrated, labels) < binary_nll(probabilities, labels)
    assert scaler.calibrated_nll == pytest.approx(binary_nll(calibrated, labels))


def test_temperature_scaling_preserves_probability_order() -> None:
    probabilities = [0.05, 0.2, 0.7, 0.95]
    scaler = BinaryTemperatureScaler(2.0, 4, 0.5, 0.4)

    calibrated = scaler.transform(probabilities)

    assert calibrated == tuple(sorted(calibrated))


def test_binary_metrics_have_known_values() -> None:
    probabilities = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 0, 1]

    assert binary_brier_score(probabilities, labels) == pytest.approx(0.175)
    assert binary_expected_calibration_error(probabilities, labels, bins=2) == pytest.approx(0.25)


def test_temperature_fit_rejects_single_class_fold() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        BinaryTemperatureScaler.fit([0.2, 0.3, 0.4], [0, 0, 0])


def test_scaler_serialization_is_complete() -> None:
    scaler = BinaryTemperatureScaler(1.5, 20, 0.6, 0.5)

    assert scaler.to_dict() == {
        "temperature": 1.5,
        "fitted_examples": 20,
        "base_nll": 0.6,
        "calibrated_nll": 0.5,
    }
