from __future__ import annotations

import numpy as np

from trace_scv.selection.mocheg_dacs import DacsCountClassifier, DacsCountConfig


def test_dacs_count_classifier_learns_separable_counts() -> None:
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    model = DacsCountClassifier.fit(
        features,
        [1, 1, 3, 3],
        config=DacsCountConfig(maximum_count=4, epochs=1000, bootstrap_samples=10),
    )
    assert model.predict(features).tolist() == [1, 1, 3, 3]
    assert model.observed_counts == (1, 3)
