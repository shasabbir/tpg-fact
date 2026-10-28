from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trace_scv.data.io import sha256_file
from trace_scv.evaluation.study import evaluate_checkpoint
from trace_scv.features import load_feature_store
from trace_scv.training import TrainConfig, train_model, training_input_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the predeclared adaptive citation-set validation pilot."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    experiments = project_root / "experiments"
    features = experiments / "data" / "features" / "pilot"
    train_path = features / "train.npz"
    validation_path = features / "validation.npz"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            "Corrected pilot feature stores are required; run the v9 pilot first."
        )

    train = load_feature_store(train_path)
    validation = load_feature_store(validation_path)
    maximum_gold = int(np.max(train.gold_mask.sum(axis=1)))
    citation_budget = min(maximum_gold, int(train.evidence_mask.shape[1]))
    baseline_path = (
        experiments
        / "outputs"
        / "pilot"
        / "trace_no_cf"
        / "seed-13"
        / "evaluation"
        / "metrics.json"
    )
    if not baseline_path.is_file():
        raise FileNotFoundError("The audited fixed-budget graph baseline is required.")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    outputs = experiments / "outputs" / "adaptive_citation_pilot"
    outputs.mkdir(parents=True, exist_ok=True)

    train_gold_counts = train.gold_mask.sum(axis=1).astype(int)
    protocol = {
        "status": "validation_only",
        "official_test_evaluated": False,
        "train_count": len(train.labels),
        "validation_count": len(validation.labels),
        "train_gold_count_distribution": {
            str(value): int(np.sum(train_gold_counts == value))
            for value in sorted(set(train_gold_counts))
        },
        "citation_budget_from_training_maximum": citation_budget,
        "selection_rule": (
            "highest validation citation macro-F1 subject to verdict macro-F1 no more "
            "than 0.01 below the audited fixed-budget graph baseline"
        ),
        "feature_hashes": {
            path.name: sha256_file(path)
            for path in (
                train_path,
                train_path.with_suffix(".json"),
                validation_path,
                validation_path.with_suffix(".json"),
            )
        },
    }
    (outputs / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    results = {}
    for name, balanced in (
        ("adaptive_unbalanced", False),
        ("adaptive_balanced", True),
    ):
        run_dir = outputs / name / "seed-13"
        config = TrainConfig(
            seed=13,
            variant="adaptive",
            experiment_name=name,
            epochs=20,
            patience=5,
            batch_size=32,
            hidden_dim=192,
            graph_layers=2,
            citation_budget=citation_budget,
            balanced_selection=balanced,
            cardinality_weight=0.25,
            deletion_weight=0.0,
            irrelevant_weight=0.0,
            duplicate_weight=0.0,
            swap_weight=0.0,
            device=args.device,
        )
        fingerprint = training_input_fingerprint(train_path, validation_path, config)
        train_model(
            train_path,
            validation_path,
            run_dir,
            config,
            input_fingerprint=fingerprint,
        )
        results[name] = evaluate_checkpoint(
            run_dir / "model.pt",
            validation_path,
            run_dir / "evaluation",
            batch_size=32,
            bootstrap_samples=1000,
            device=args.device,
        )

    baseline_verdict = float(baseline["verdict_macro_f1"])
    eligible = {
        name: result
        for name, result in results.items()
        if result["verdict_macro_f1"] >= baseline_verdict - 0.01
    }
    selected = (
        max(eligible, key=lambda name: results[name]["citation_macro"]["f1"])
        if eligible
        else None
    )
    decision = {
        "status": "pass" if selected else "fail",
        "selected_variant": selected,
        "fixed_budget_graph_baseline": {
            "verdict_macro_f1": baseline_verdict,
            "citation_macro_f1": baseline["citation_macro"]["f1"],
            "citation_cardinality": baseline.get("citation_cardinality"),
        },
        "variants": {
            name: {
                "verdict_macro_f1": result["verdict_macro_f1"],
                "citation_macro_f1": result["citation_macro"]["f1"],
                "citation_cardinality": result["citation_cardinality"],
                "verdict_safety_pass": name in eligible,
            }
            for name, result in results.items()
        },
        "scientific_results_validated": False,
    }
    (outputs / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
