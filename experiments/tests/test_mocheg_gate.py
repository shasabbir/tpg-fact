from __future__ import annotations

from statistics import fmean


def test_declared_point_noninferiority_uses_mean_seed_effect() -> None:
    effects = [-0.015, 0.009, 0.006]
    assert fmean(effects) >= -0.01
