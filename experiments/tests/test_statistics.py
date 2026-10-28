from trace_scv.evaluation.statistics import (
    holm_adjust,
    paired_group_bootstrap_comparisons,
    seed_level_comparisons,
)


def test_holm_adjust_is_bounded_and_monotone() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adjusted["a"] == 0.03
    assert adjusted["b"] >= adjusted["a"]
    assert adjusted["c"] <= 1.0


def test_seed_level_comparison() -> None:
    rows = []
    for seed in (1, 2):
        rows.extend(
            [
                {"variant": "trace_scv", "seed": seed, "verdict_macro_f1": 0.8, "citation_macro": {"f1": 0.7}},
                {"variant": "flat", "seed": seed, "verdict_macro_f1": 0.7, "citation_macro": {"f1": 0.5}},
            ]
        )
    result = seed_level_comparisons(rows)
    assert result["flat"]["verdict_macro_f1_difference"]["mean"] > 0
    assert "p_holm" in result["flat"]["citation_macro_f1_difference"]


def test_paired_group_bootstrap_comparison() -> None:
    gold = ["supported", "refuted", "supported", "refuted"]
    groups = ["a", "a", "b", "b"]
    rows = []
    for variant, predicted, cited in (
        ("trace_scv", gold, [["e1"], ["e1"], ["e1"], ["e1"]]),
        ("flat", ["supported"] * 4, [[], [], [], []]),
    ):
        records = [
            {
                "claim_id": f"c{index}",
                "group_id": groups[index],
                "gold_label": gold[index],
                "predicted_label": predicted[index],
                "gold_citations": ["e1"],
                "predicted_citations": cited[index],
            }
            for index in range(4)
        ]
        rows.append({"variant": variant, "seed": 13, "_comparison_records": records})
    result = paired_group_bootstrap_comparisons(rows, samples=50, seed=7, batch_size=10)
    assert result["flat"]["verdict_macro_f1_difference"]["mean"] > 0
    assert result["flat"]["citation_macro_f1_difference"]["lower"] > 0
    assert "p_holm" in result["flat"]["citation_macro_f1_difference"]
