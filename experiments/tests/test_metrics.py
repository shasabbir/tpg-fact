import pytest

from trace_scv.evaluation.metrics import (
    citation_macro_prf,
    comprehensiveness,
    label_macro_f1,
    probability_invariance,
)


def test_citation_macro_prf() -> None:
    result = citation_macro_prf([{"a"}, {"x"}], [{"a"}, {"y"}])
    assert result == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_label_macro_f1_perfect() -> None:
    assert label_macro_f1(["a", "b"], ["a", "b"]) == 1.0


def test_comprehensiveness_is_probability_drop() -> None:
    score = comprehensiveness(
        [{"support": 0.9}, {"refute": 0.8}],
        [{"support": 0.4}, {"refute": 0.5}],
        ["support", "refute"],
    )
    assert score == pytest.approx(0.4)


def test_probability_invariance() -> None:
    assert probability_invariance([{"a": 0.7, "b": 0.3}], [{"a": 0.7, "b": 0.3}]) == 1.0
