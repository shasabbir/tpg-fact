import pytest

from trace_scv.evaluation.metrics import (
    citation_cardinality_metrics,
    citation_macro_prf,
    classification_report,
    comprehensiveness,
    label_macro_f1,
    probabilistic_metrics,
    probability_invariance,
    ranked_evidence_metrics,
)


def test_citation_cardinality_metrics() -> None:
    result = citation_cardinality_metrics(
        [["a"], ["b", "c"]], [["a", "z"], ["b", "c"]]
    )
    assert result == {
        "mae": 0.5,
        "exact_match": 0.5,
        "mean_predicted": 1.5,
        "mean_gold": 2.0,
    }


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


def test_classification_and_ranked_metrics() -> None:
    report = classification_report(["a", "b"], ["a", "b"])
    ranking = ranked_evidence_metrics([["e1", "x"], ["e2"]], [{"e1"}, {"e2"}], k=2)
    assert report["balanced_accuracy"] == 1.0
    assert ranking["recall@2"] == 1.0
    assert ranking["map@2"] == 1.0


def test_probabilistic_metrics_perfect() -> None:
    result = probabilistic_metrics([{"a": 1.0, "b": 0.0}], ["a"])
    assert result["brier"] == 0.0
    assert result["nll"] == 0.0
    assert result["ece"] == 0.0
