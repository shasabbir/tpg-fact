from __future__ import annotations

from trace_scv.selection.mocheg_reproduction import _numeric_close


def test_reproduction_comparison_ignores_runtime_and_artifact_paths() -> None:
    left = {"metric": 0.5, "runtime": {"wall_seconds": 1}, "artifacts": {"path": "a"}}
    right = {"metric": 0.50000001, "runtime": {"wall_seconds": 2}, "artifacts": {"path": "b"}}
    assert _numeric_close(left, right, tolerance=1e-6)


def test_reproduction_comparison_rejects_metric_drift() -> None:
    assert not _numeric_close({"metric": 0.5}, {"metric": 0.6}, tolerance=1e-6)
