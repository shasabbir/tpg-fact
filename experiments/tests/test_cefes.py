from itertools import product

import pytest

from trace_scv.cefes import (
    expected_prefix_f1_curve,
    poisson_binomial_distribution,
    select_cefes_prefix,
)


def _brute_expected_f1(probabilities: list[float], selected_count: int) -> float:
    expected = 0.0
    for outcomes in product((0, 1), repeat=len(probabilities)):
        mass = 1.0
        for probability, outcome in zip(probabilities, outcomes, strict=True):
            mass *= probability if outcome else 1.0 - probability
        true_positive = sum(outcomes[:selected_count])
        gold_count = sum(outcomes)
        denominator = selected_count + gold_count
        score = 1.0 if denominator == 0 else 2.0 * true_positive / denominator
        expected += mass * score
    return expected


def test_poisson_binomial_distribution() -> None:
    distribution = poisson_binomial_distribution([0.2, 0.6])

    assert distribution == pytest.approx((0.32, 0.56, 0.12))
    assert sum(distribution) == pytest.approx(1.0)


def test_exact_curve_matches_brute_force_enumeration() -> None:
    probabilities = [0.85, 0.45, 0.2, 0.05]

    curve = expected_prefix_f1_curve(probabilities)

    expected = [
        _brute_expected_f1(probabilities, selected_count)
        for selected_count in range(len(probabilities) + 1)
    ]
    assert curve == pytest.approx(expected, abs=1e-12)


def test_all_zero_probabilities_select_empty_set() -> None:
    decision = select_cefes_prefix([0.0, 0.0, 0.0])

    assert decision.selected_count == 0
    assert decision.expected_f1 == pytest.approx(1.0)


def test_all_one_probabilities_select_every_item() -> None:
    decision = select_cefes_prefix([1.0, 1.0, 1.0])

    assert decision.selected_count == 3
    assert decision.expected_f1 == pytest.approx(1.0)


def test_tie_prefers_smaller_set() -> None:
    decision = select_cefes_prefix([0.5])

    assert decision.expected_f1_by_count == pytest.approx((0.5, 0.5))
    assert decision.selected_count == 0


def test_nonempty_policy_resolves_empty_tie() -> None:
    decision = select_cefes_prefix([0.5], allow_empty=False)

    assert decision.selected_count == 1


def test_selection_sorts_probabilities_and_returns_original_indices() -> None:
    decision = select_cefes_prefix([0.1, 0.9, 0.8], allow_empty=False, max_items=2)

    assert decision.sorted_indices == (1, 2, 0)
    assert decision.selected_indices == decision.sorted_indices[: decision.selected_count]
    assert decision.selected_count <= 2


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_probabilities_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        select_cefes_prefix([value])
