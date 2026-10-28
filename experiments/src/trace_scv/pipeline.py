from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

from trace_scv.cardinality import (
    CardinalityConfig,
    cardinality_checkpoint_matches,
    evaluate_decoupled_cardinality,
    evaluate_fixed_cardinality,
    train_cardinality_head,
)
from trace_scv.data.download import download_huggingface_snapshot, download_sciver_subset
from trace_scv.data.io import sha256_file, write_packets
from trace_scv.data.musciclaims import prepare_musciclaims
from trace_scv.data.sciver import prepare_sciver
from trace_scv.evaluation.study import evaluate_checkpoint, evaluate_majority
from trace_scv.evaluation.statistics import (
    paired_group_bootstrap_comparisons,
    seed_level_comparisons,
)
from trace_scv.features import build_feature_store, make_embedder
from trace_scv.schema import ClaimPacket, EvidenceItem
from trace_scv.training import (
    TrainConfig,
    checkpoint_matches,
    train_model,
    training_input_fingerprint,
)


PROFILES = {
    "smoke": {
        "epochs": 2,
        "seeds": [13],
        "experiments": ["claim_only", "all_pool", "relevance", "flat", "trace_no_cf"],
        "backend": "hash",
        "dim": 64,
        "max_candidates": 6,
        "bootstrap": 50,
        "evaluation_split": "test",
        "external_musciclaims": False,
    },
    "pilot": {
        "epochs": 12,
        "seeds": [13],
        "experiments": ["claim_only", "all_pool", "relevance", "flat", "trace_no_cf"],
        "backend": "hf",
        "dim": 512,
        "max_candidates": 12,
        "bootstrap": 1000,
        "limits": (600, 150, 0),
        "evaluation_split": "validation",
        "external_musciclaims": False,
    },
    "formal": {
        "epochs": 25,
        "seeds": [13, 37, 73, 101, 137],
        "experiments": ["claim_only", "all_pool", "relevance", "flat", "trace_no_cf"],
        "backend": "hf",
        "dim": 512,
        "max_candidates": 12,
        "bootstrap": 10000,
        "limits": (None, None, None),
        "evaluation_split": "test",
        "external_musciclaims": False,
    },
}


def _experiment_config(name: str) -> tuple[str, dict[str, float]]:
    variant = {
        "claim_only": "claim_only",
        "all_pool": "all_pool",
        "relevance": "relevance",
        "flat": "flat",
    }.get(name, "trace_scv")
    weights = {
        "deletion_weight": 0.25,
        "irrelevant_weight": 0.15,
        "duplicate_weight": 0.15,
        "swap_weight": 0.15,
        "sparsity_weight": 0.02,
    }
    if name in {"claim_only", "all_pool", "relevance", "flat", "trace_no_cf"}:
        weights.update(
            deletion_weight=0.0,
            irrelevant_weight=0.0,
            duplicate_weight=0.0,
            swap_weight=0.0,
        )
    if name == "no_deletion":
        weights["deletion_weight"] = 0.0
    elif name == "no_irrelevant":
        weights["irrelevant_weight"] = 0.0
    elif name == "no_duplicate":
        weights["duplicate_weight"] = 0.0
    elif name == "no_swap":
        weights["swap_weight"] = 0.0
    elif name == "no_sparsity":
        weights["sparsity_weight"] = 0.0
    return variant, weights


def _synthetic_packets(split: str, count: int) -> list[ClaimPacket]:
    packets = []
    for index in range(count):
        supported = index % 2 == 0
        value_a, value_b = ((91, 84) if supported else (77, 84))
        packets.append(
            ClaimPacket(
                claim_id=f"synthetic:{split}:{index}",
                claim_text="Model A scores higher than Model B in the reported table.",
                label="supported" if supported else "refuted",
                split=split,
                dataset="synthetic",
                group_id=f"paper-{split}-{index // 2}",
                evidence=(
                    EvidenceItem(
                        "table:1",
                        "table",
                        f"Model A: {value_a}; Model B: {value_b}.",
                        gold=True,
                    ),
                    EvidenceItem("text:background", "text", "The evaluation uses five seeds."),
                    EvidenceItem("figure:2", "figure", "Accuracy by training epoch."),
                ),
            )
        )
    return packets


def _aggregate(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(result["variant"], []).append(result)
    summary = {}
    for variant, rows in grouped.items():
        comprehensiveness = [
            row["comprehensiveness"]
            for row in rows
            if row.get("comprehensiveness") is not None
        ]
        summary[variant] = {
            "runs": len(rows),
            "verdict_macro_f1_mean": sum(row["verdict_macro_f1"] for row in rows) / len(rows),
            "citation_macro_f1_mean": sum(row["citation_macro"]["f1"] for row in rows) / len(rows),
            "citation_count_mae_mean": sum(
                row["citation_cardinality"]["mae"] for row in rows
            )
            / len(rows),
            "citation_count_exact_match_mean": sum(
                row["citation_cardinality"]["exact_match"] for row in rows
            )
            / len(rows),
            "comprehensiveness_mean": (
                sum(comprehensiveness) / len(comprehensiveness)
                if comprehensiveness
                else None
            ),
            "gold_evidence_comprehensiveness_mean": sum(
                row["gold_evidence_comprehensiveness"] for row in rows
            )
            / len(rows),
            "duplicate_invariance_mean": sum(row["duplicate_invariance"] for row in rows) / len(rows),
            "swap_sensitivity_mean": sum(row["swap_sensitivity"] for row in rows) / len(rows),
            "swap_coverage_mean": sum(row["swap_coverage"] for row in rows) / len(rows),
            "visual_removed_macro_f1_mean": sum(
                row["visual_removed_macro_f1"] for row in rows
            )
            / len(rows),
            "visual_grounded_count_mean": sum(
                row["visual_grounded_subset"]["count"] for row in rows
            )
            / len(rows),
            "visual_grounded_clean_verdict_macro_f1_mean": sum(
                row["visual_grounded_subset"]["clean_verdict_macro_f1"]
                for row in rows
            )
            / len(rows),
            "visual_grounded_visual_removed_macro_f1_mean": sum(
                row["visual_grounded_subset"]["visual_removed_macro_f1"]
                for row in rows
            )
            / len(rows),
            "visual_grounded_swap_sensitivity_mean": sum(
                row["visual_grounded_subset"]["swap_sensitivity"] for row in rows
            )
            / len(rows),
        }
    return summary


def _pilot_readiness_gate(
    summary: dict[str, dict], data_release_gate: dict | None = None
) -> dict:
    """Apply the prespecified one-seed go/no-go checks before a formal run."""
    required = {"dacs_scv", "trace_no_cf"}
    missing = sorted(required - set(summary))
    if missing:
        return {"status": "not_evaluable", "missing_variants": missing, "passed": False}

    full = summary["dacs_scv"]
    baseline = summary["trace_no_cf"]
    checks = {
        "citation_macro_f1_gain_at_least_one_point": full[
            "citation_macro_f1_mean"
        ]
        >= baseline["citation_macro_f1_mean"] + 0.01,
        "verdict_predictions_preserved": abs(
            full["verdict_macro_f1_mean"] - baseline["verdict_macro_f1_mean"]
        )
        <= 1e-12,
        "data_release_gate_passes": bool(
            data_release_gate and data_release_gate.get("pass")
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "passed": all(checks.values()),
        "checks": checks,
        "dacs_minus_fixed_top3": {
            "verdict_macro_f1": full["verdict_macro_f1_mean"]
            - baseline["verdict_macro_f1_mean"],
            "citation_macro_f1": full["citation_macro_f1_mean"]
            - baseline["citation_macro_f1_mean"],
            "citation_count_mae": full["citation_count_mae_mean"]
            - baseline["citation_count_mae_mean"],
        },
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _hashed_files(paths: list[Path], root: Path) -> dict[str, str]:
    return {
        _relative(path, root): sha256_file(path)
        for path in sorted({item.resolve() for item in paths})
        if path.is_file()
    }


def _write_run_audit(
    project_root: Path,
    profile: str,
    settings: dict,
    processed: Path,
    features: Path,
    outputs: Path,
    external_processed: Path | None,
    external_features: Path | None,
) -> Path:
    """Persist the exact data/code/config provenance needed to audit a run."""
    audit_dir = outputs / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    data_files = list(processed.glob("*.jsonl"))
    feature_files = list(features.glob("*.npz")) + list(features.glob("*.json"))
    manifest_paths = [processed / "manifest.json"]
    if external_processed is not None:
        data_files.append(external_processed)
        manifest_paths.append(external_processed.with_name("external_manifest.json"))
    if external_features is not None:
        feature_files.extend([external_features, external_features.with_suffix(".json")])
    code_paths = list((project_root / "experiments" / "src").rglob("*.py"))
    code_paths.extend(
        [
            project_root / "experiments" / "pyproject.toml",
            project_root / "experiments" / "configs" / "study.yaml",
            project_root / "experiments" / "scripts" / "run_study.py",
            project_root / "kaggle" / "TRACE_SCV_Kaggle.ipynb",
        ]
    )
    dataset_manifests = {}
    for path in manifest_paths:
        if path.is_file():
            dataset_manifests[path.name] = json.loads(path.read_text(encoding="utf-8"))
    config_path = project_root / "experiments" / "configs" / "study.yaml"
    if config_path.is_file():
        shutil.copy2(config_path, audit_dir / "study.yaml")
    payload = {
        "artifact_schema_version": 5,
        "profile": profile,
        "effective_profile_settings": settings,
        "dataset_manifests": dataset_manifests,
        "data_hashes": _hashed_files(data_files, project_root),
        "feature_hashes": _hashed_files(feature_files, project_root),
        "code_hashes": _hashed_files(code_paths, project_root),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "notes": [
            "comparison_records.jsonl stores labels, source-paper groups, citations, all intervention probabilities, and swap/visual-grounding flags for independent recomputation.",
            "Feature arrays are reproducible from the pinned public release and recorded code/configuration; their hashes bind checkpoints to those exact arrays.",
            "DACS-SCV is trained after the matched verifier is frozen; its checkpoint records the base-checkpoint and development-feature hashes.",
            "DACS-SCV evaluation rejects any changed evidence ranking and copies the matched verifier's verdict records unchanged.",
        ],
    }
    path = audit_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_artifact_inventory(outputs: Path) -> Path:
    """Write a final hash inventory for every exported artifact except itself."""
    target = outputs / "audit" / "artifact_inventory.json"
    files = [path for path in outputs.rglob("*") if path.is_file() and path != target]
    payload = {
        "file_count": len(files),
        "files": {
            path.relative_to(outputs).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(files)
        },
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def run_pipeline(
    project_root: Path,
    *,
    profile: str,
    download: bool = False,
    device: str = "auto",
) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    settings = PROFILES[profile]
    experiments = project_root / "experiments"
    configured_data_root = os.environ.get("TRACE_SCV_DATA_ROOT")
    data_root = (
        Path(configured_data_root).expanduser().resolve()
        if configured_data_root
        else experiments / "data"
    )
    processed = data_root / "processed" / (
        "synthetic" if profile == "smoke" else f"sciver_{profile}"
    )
    features = data_root / "features" / profile
    configured_output_root = os.environ.get("TRACE_SCV_OUTPUT_ROOT")
    outputs = (
        Path(configured_output_root).expanduser().resolve() / profile
        if configured_output_root
        else experiments / "outputs" / profile
    )
    external_features = None
    external_processed = None
    prepared_manifest: dict | None = None

    if profile == "smoke":
        for split, count in (("train", 24), ("validation", 8), ("test", 12)):
            write_packets(processed / f"{split}.jsonl", _synthetic_packets(split, count))
    else:
        limits = settings["limits"]
        public_root = data_root / "public"
        dataset_root = public_root / "sciver"
        if download:
            if profile == "pilot":
                dataset_root = download_sciver_subset(
                    public_root,
                    max_train=limits[0],
                    max_validation=limits[1],
                    max_test=limits[2],
                    include_images=True,
                )
            else:
                dataset_root = download_huggingface_snapshot("sciver", public_root)
        prepared_manifest = prepare_sciver(
            dataset_root,
            processed,
            max_candidates=settings["max_candidates"],
            max_train=limits[0],
            max_validation=limits[1],
            max_test=limits[2],
        )
        data_release_gate = prepared_manifest["data_quality"]["release_gate"]
        if profile == "formal" and not data_release_gate["pass"]:
            raise RuntimeError(
                "Formal execution blocked by the SciVer cross-split release gate; "
                "inspect processed/sciver_formal/manifest.json and repair the split."
            )

        if profile == "formal" and settings["external_musciclaims"]:
            musci_root = data_root / "public" / "musciclaims"
            if download:
                musci_root = download_huggingface_snapshot("musciclaims", data_root / "public")
            external_processed = data_root / "processed" / "musciclaims" / "external_test.jsonl"
            prepare_musciclaims(musci_root, external_processed, max_candidates=settings["max_candidates"])
            external_features = data_root / "features" / profile / "musciclaims_external.npz"

    feature_embedder = make_embedder(settings["backend"], settings["dim"])
    required_splits = {"train", "validation", settings["evaluation_split"]}
    for split in sorted(required_splits):
        build_feature_store(
            processed / f"{split}.jsonl",
            features / f"{split}.npz",
            backend=settings["backend"],
            dim=settings["dim"],
            max_candidates=settings["max_candidates"],
            embedder=feature_embedder,
        )
    if external_features is not None and external_processed is not None:
        build_feature_store(
            external_processed,
            external_features,
            backend=settings["backend"],
            dim=settings["dim"],
            max_candidates=settings["max_candidates"],
            embedder=feature_embedder,
        )

    audit_manifest = _write_run_audit(
        project_root,
        profile,
        settings,
        processed,
        features,
        outputs,
        external_processed,
        external_features,
    )

    evaluation_features = features / f"{settings['evaluation_split']}.npz"
    evaluations = [
        evaluate_majority(
            features / "train.npz",
            evaluation_features,
            outputs / "majority" / "evaluation",
            bootstrap_samples=settings["bootstrap"],
        )
    ]
    external_evaluations = []
    if external_features is not None:
        external_evaluations.append(
            evaluate_majority(
                features / "train.npz",
                external_features,
                outputs / "majority" / "external_musciclaims",
                bootstrap_samples=settings["bootstrap"],
                dataset_name="musciclaims",
            )
        )
    for experiment_name in settings["experiments"]:
        variant, loss_weights = _experiment_config(experiment_name)
        for seed in settings["seeds"]:
            run_dir = outputs / experiment_name / f"seed-{seed}"
            config = TrainConfig(
                seed=seed,
                variant=variant,
                experiment_name=experiment_name,
                epochs=settings["epochs"],
                batch_size=16 if profile == "smoke" else 32,
                hidden_dim=64 if profile == "smoke" else 192,
                graph_layers=1 if profile == "smoke" else 2,
                citation_budget=2 if profile == "smoke" else 3,
                patience=2 if profile == "smoke" else 4,
                device=device,
                **loss_weights,
            )
            checkpoint_path = run_dir / "model.pt"
            input_fingerprint = training_input_fingerprint(
                features / "train.npz", features / "validation.npz", config
            )
            if checkpoint_path.exists() and not checkpoint_matches(
                checkpoint_path, config, input_fingerprint
            ):
                raise RuntimeError(
                    f"Refusing to reuse stale checkpoint: {checkpoint_path}. "
                    "Its feature/configuration fingerprint does not match this run; "
                    "start with an empty outputs directory."
                )
            if not checkpoint_path.exists():
                train_model(
                    features / "train.npz",
                    features / "validation.npz",
                    run_dir,
                    config,
                    input_fingerprint=input_fingerprint,
                )
            evaluation = evaluate_checkpoint(
                checkpoint_path,
                evaluation_features,
                run_dir / "evaluation",
                batch_size=config.batch_size,
                bootstrap_samples=settings["bootstrap"],
                device=device,
            )
            evaluations.append(evaluation)
            if external_features is not None:
                external_evaluations.append(
                    evaluate_checkpoint(
                        run_dir / "model.pt",
                        external_features,
                        run_dir / "external_musciclaims",
                        batch_size=config.batch_size,
                        bootstrap_samples=settings["bootstrap"],
                        device=device,
                        dataset_name="musciclaims",
                    )
                )

    for seed in settings["seeds"]:
        base_run = outputs / "trace_no_cf" / f"seed-{seed}"
        for citation_count in (1, 2):
            evaluations.append(
                evaluate_fixed_cardinality(
                    base_run / "evaluation",
                    evaluation_features,
                    outputs / f"fixed_top{citation_count}" / f"seed-{seed}" / "evaluation",
                    citation_count=citation_count,
                    bootstrap_samples=settings["bootstrap"],
                )
            )
        cardinality_variants = {
            "dacs_scv": (True, True),
            "dacs_no_valid_count": (False, True),
            "dacs_no_visual_count": (True, False),
        }
        for cardinality_name, (include_valid, include_visual) in cardinality_variants.items():
            adaptive_run = outputs / cardinality_name / f"seed-{seed}"
            cardinality_config = CardinalityConfig(
                seed=seed,
                experiment_name=cardinality_name,
                hidden_dim=16 if profile == "smoke" else 32,
                epochs=8 if profile == "smoke" else 100,
                patience=2 if profile == "smoke" else 12,
                include_valid_count=include_valid,
                include_visual_count=include_visual,
                device=device,
            )
            head_checkpoint = adaptive_run / "cardinality_head.pt"
            if head_checkpoint.exists() and not cardinality_checkpoint_matches(
                head_checkpoint,
                base_run / "model.pt",
                features / "train.npz",
                features / "validation.npz",
                cardinality_config,
            ):
                raise RuntimeError(
                    f"Refusing to reuse stale cardinality checkpoint: {head_checkpoint}"
                )
            if not head_checkpoint.exists():
                train_cardinality_head(
                    base_run / "model.pt",
                    features / "train.npz",
                    features / "validation.npz",
                    adaptive_run,
                    cardinality_config,
                )
            evaluations.append(
                evaluate_decoupled_cardinality(
                    base_run / "model.pt",
                    base_run / "evaluation",
                    head_checkpoint,
                    evaluation_features,
                    adaptive_run / "evaluation",
                    bootstrap_samples=settings["bootstrap"],
                    device=device,
                )
            )

    summary = _aggregate(evaluations)
    reference = "dacs_scv" if any(
        row["variant"] == "dacs_scv" for row in evaluations
    ) else "trace_scv"
    result = {
        "status": "complete",
        "profile": profile,
        "settings": settings,
        "evaluation_split": settings["evaluation_split"],
        "summary": summary,
        "pilot_readiness_gate": (
            _pilot_readiness_gate(
                summary,
                prepared_manifest["data_quality"]["release_gate"]
                if prepared_manifest
                else None,
            )
            if profile == "pilot"
            else None
        ),
        "external_summary": _aggregate(external_evaluations) if external_evaluations else {},
        "paired_seed_comparisons": seed_level_comparisons(
            evaluations, reference=reference
        ),
        "paired_group_bootstrap_comparisons": paired_group_bootstrap_comparisons(
            evaluations, reference=reference, samples=settings["bootstrap"]
        ),
        "external_paired_group_bootstrap_comparisons": (
            paired_group_bootstrap_comparisons(
                external_evaluations,
                reference=(
                    "dacs_scv"
                    if any(row["variant"] == "dacs_scv" for row in external_evaluations)
                    else "trace_scv"
                ),
                samples=settings["bootstrap"],
            )
            if external_evaluations
            else {}
        ),
        "formal_run_completed": profile == "formal",
        "scientific_results_validated": False,
        "audit_manifest": audit_manifest.relative_to(outputs).as_posix(),
        "interpretation": (
            "Formal public-data experiment completed; manuscript claims still require audit."
            if profile == "formal"
            else "Software/pilot output only; not a final scientific result."
        ),
    }
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_artifact_inventory(outputs)
    return result
