from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_full_run_readiness(
    experiment_root: str | Path,
    *,
    api_status: str = "not_checked",
    conflict_manifest_status: str = "not_checked",
) -> dict[str, Any]:
    root = Path(experiment_root).resolve()
    config = yaml.safe_load((root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    claims_path = root / "data" / "processed" / "pilot_claims_with_evidence.jsonl"
    evidence_path = root / "data" / "processed" / "pilot_evidence.jsonl"
    pilot_report_path = root / "outputs" / "pilot" / "pilot_assessment.json"
    pilot_report = _json(pilot_report_path)
    revisions = {
        "dense_text_model": config["retrieval"].get("dense_text_revision"),
        "reranker_model": config["retrieval"].get("reranker_revision"),
        "visual_model": config["retrieval"].get("visual_model_revision"),
        "comparison_mllm": config["model"].get("comparison_mllm_revision"),
    }
    torch_available = importlib.util.find_spec("torch") is not None
    cuda_available = False
    cuda_devices = 0
    if torch_available:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_devices = int(torch.cuda.device_count())

    blockers: list[str] = []
    if api_status not in {"ok", "frozen_snapshot"}:
        blockers.append("Fact Check Tools API preflight has not passed")
    if _jsonl_count(claims_path) < config["data"]["audit_sample_size"]:
        blockers.append("200-case pilot claim manifest is incomplete")
    if _jsonl_count(evidence_path) == 0:
        blockers.append("audited pilot evidence manifest is missing")
    if pilot_report.get("phase_status") != "pilot_complete_ready_for_scale_decision":
        blockers.append("pilot feasibility gate has not passed")
    if conflict_manifest_status != "valid":
        blockers.append("audited semantic-conflict donor manifest is missing or invalid")
    if any(not revision for revision in revisions.values()):
        blockers.append("exact pretrained model revisions are not frozen")
    required_scripts = (
        "train.py",
        "build_feature_store.py",
        "retrieve_evidence.py",
        "tune.py",
        "run_final_matrix.py",
        "evaluate_interventions.py",
        "compare_experiments.py",
        "export_paper_artifacts.py",
        "run_full_study.py",
    )
    missing_scripts = [name for name in required_scripts if not (root / "scripts" / name).exists()]
    if missing_scripts:
        blockers.append(f"final-run scripts are missing: {missing_scripts}")
    if not torch_available:
        blockers.append("PyTorch is not installed in this environment")
    if not cuda_available:
        blockers.append("no CUDA GPU is available in this environment")

    return {
        "status": "ready" if not blockers else "not_ready",
        "api_status": api_status,
        "conflict_manifest_status": conflict_manifest_status,
        "data": {
            "pilot_claims": _jsonl_count(claims_path),
            "pilot_evidence": _jsonl_count(evidence_path),
            "pilot_phase_status": pilot_report.get("phase_status", "missing"),
        },
        "models": {"revisions": revisions},
        "runtime": {
            "torch_available": torch_available,
            "cuda_available": cuda_available,
            "cuda_devices": cuda_devices,
        },
        "training_entry_point": str(root / "scripts" / "train.py"),
        "execution_scripts": {
            name: str(root / "scripts" / name) for name in required_scripts
        },
        "blockers": blockers,
        "interpretation": (
            "Ready means that the pilot gate, frozen model identities, real trainer, "
            "and GPU runtime all exist. It is not a claim of scientific validity."
        ),
    }
