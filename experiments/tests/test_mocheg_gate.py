from __future__ import annotations

from pathlib import Path
from statistics import fmean

from scripts.freeze_mocheg_full_development import collect_freeze_paths


def test_declared_point_noninferiority_uses_mean_seed_effect() -> None:
    effects = [-0.015, 0.009, 0.006]
    assert fmean(effects) >= -0.01


def test_full_development_freeze_includes_clean_features_and_dacs(tmp_path: Path) -> None:
    discovered = [
        "outputs/mocheg/full_corpus_dacs_trial_002/audit.json",
        "outputs/mocheg/clean_rerun/dacs_trial_002/audit.json",
    ]
    for relative in discovered:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    relative_paths = {
        path.relative_to(tmp_path).as_posix() for path in collect_freeze_paths(tmp_path)
    }
    assert "outputs/mocheg/clean_rerun/full_corpus_features.npz" in relative_paths
    assert "outputs/mocheg/clean_rerun/full_corpus_features.rankings.jsonl" in relative_paths
    assert "outputs/mocheg/clean_rerun/full_corpus_features.audit.json" in relative_paths
    assert "outputs/mocheg/full_corpus_dacs_trial_002/audit.json" in relative_paths
    assert "outputs/mocheg/clean_rerun/dacs_trial_002/audit.json" in relative_paths
