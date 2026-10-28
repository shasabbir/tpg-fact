from __future__ import annotations

from trace_scv.selection.mocheg_verifier_audit import _bootstrap, _macro_f1


def test_verifier_audit_metrics_are_deterministic() -> None:
    gold = ["nei", "refuted", "supported"] * 4
    left = list(gold)
    right = ["nei"] * len(gold)
    assert _macro_f1(gold, left) == 1.0
    first = _bootstrap(gold, left, right, seed=13, samples=100)
    second = _bootstrap(gold, left, right, seed=13, samples=100)
    assert first == second
    assert first["mean"] > 0.0
