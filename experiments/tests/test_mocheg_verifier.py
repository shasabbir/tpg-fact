from __future__ import annotations

import numpy as np

from trace_scv.selection.mocheg_verifier import (
    MulticlassLogisticVerifier,
    _macro_f1,
    _paired_macro_f1_bootstrap,
)


def test_multiclass_verifier_learns_separable_classes() -> None:
    features = np.asarray(
        [
            [-3.0, 0.0],
            [-2.0, 0.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [2.0, -2.0],
            [3.0, -3.0],
        ]
    )
    labels = ["nei", "nei", "refuted", "refuted", "supported", "supported"]
    model = MulticlassLogisticVerifier.fit(
        features, labels, learning_rate=0.1, epochs=1500, l2=1e-4
    )
    predictions = np.asarray(model.labels)[model.predict_probabilities(features).argmax(1)]
    assert predictions.tolist() == labels


def test_macro_f1_and_paired_bootstrap_are_preserved() -> None:
    gold = ["nei", "refuted", "supported", "supported"]
    left = ["nei", "refuted", "supported", "supported"]
    right = ["nei", "nei", "supported", "refuted"]
    assert _macro_f1(gold, left) == 1.0
    result = _paired_macro_f1_bootstrap(gold, left, right, samples=19, seed=13)
    assert result["sample_count"] == 19
    assert len(result["samples"]) == 19
    assert result["mean"] > 0.0
