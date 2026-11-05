from __future__ import annotations

from trace_scv.q1_phase2_reporting import bootstrap_two_sided_p, holm_adjust


def test_bootstrap_p_uses_finite_sample_correction() -> None:
    assert bootstrap_two_sided_p([1.0, 2.0, 3.0]) == 0.5


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adjusted["a"] == 0.03
    assert adjusted["b"] == 0.04
    assert adjusted["c"] == 0.5
