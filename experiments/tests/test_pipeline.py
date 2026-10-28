from pathlib import Path

from trace_scv.pipeline import _pilot_readiness_gate, run_pipeline


def test_pilot_readiness_gate_requires_adaptive_citation_gain_and_data_safety() -> None:
    baseline = {
        "verdict_macro_f1_mean": 0.60,
        "citation_macro_f1_mean": 0.50,
        "citation_count_mae_mean": 1.27,
    }
    full = {
        "verdict_macro_f1_mean": 0.60,
        "citation_macro_f1_mean": 0.52,
        "citation_count_mae_mean": 0.60,
    }
    data_gate = {"pass": True}
    passed = _pilot_readiness_gate(
        {"dacs_scv": full, "trace_no_cf": baseline}, data_gate
    )
    assert passed["passed"] is True

    full["citation_macro_f1_mean"] = 0.505
    failed = _pilot_readiness_gate(
        {"dacs_scv": full, "trace_no_cf": baseline}, data_gate
    )
    assert failed["passed"] is False
    assert failed["checks"]["citation_macro_f1_gain_at_least_one_point"] is False


def test_complete_smoke_pipeline(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()
    result = run_pipeline(tmp_path, profile="smoke", device="cpu")
    assert result["status"] == "complete"
    assert set(result["summary"]) == {
        "majority", "claim_only", "all_pool", "relevance", "flat", "trace_no_cf",
        "fixed_top1", "fixed_top2", "dacs_scv", "dacs_no_valid_count",
        "dacs_no_visual_count"
    }
    output = tmp_path / "experiments" / "outputs" / "smoke"
    assert (output / "summary.json").exists()
    assert (output / "audit" / "run_manifest.json").exists()
    assert (output / "audit" / "artifact_inventory.json").exists()
    assert (output / "majority" / "evaluation" / "comparison_records.jsonl").exists()
    assert (
        output / "dacs_scv" / "seed-13" / "evaluation" / "comparison_records.jsonl"
    ).exists()

    # Identical reruns must safely reuse checkpoints bound to the same features/configuration.
    rerun = run_pipeline(tmp_path, profile="smoke", device="cpu")
    assert rerun["status"] == "complete"
