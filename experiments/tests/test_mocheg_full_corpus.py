from __future__ import annotations

import numpy as np

from trace_scv.mocheg_full_corpus import _ranking_metrics, _top_indices


def test_top_indices_are_descending_and_stable() -> None:
    values = np.asarray([0.1, 0.9, 0.9, 0.2])
    assert _top_indices(values, 3).tolist() == [1, 2, 3]


def test_ranking_metrics_penalize_unretrieved_gold() -> None:
    result = _ranking_metrics(["a", "x", "b"], {"a", "b", "missing"})
    assert result["recall_at_5"] == 2.0 / 3.0
    assert result["map_at_60"] < 1.0
    assert result["ndcg_at_60"] < 1.0


def test_ranking_metrics_use_full_gold_denominator() -> None:
    result = _ranking_metrics(["a", "b"], {"a", "b", "c", "d"})
    assert result["recall_at_50"] == 0.5
    assert result["map_at_60"] == 0.5
