from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data.io import read_jsonl
from data.schema import ClaimRecord, EvidenceRecord
from data.splits import assert_no_group_leakage, partition
from training.dataset import EmbeddingStore, build_examples
from training.engine import GRAPH_VARIANTS, TrainingSettings, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one frozen DTP-Fact model variant and export test predictions"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--text-features", required=True)
    parser.add_argument("--image-features")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--hidden-dimension", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--graph-layers", type=int)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Train/select on train/validation without reading test examples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    claims = [ClaimRecord.from_dict(row) for row in read_jsonl(args.claims)]
    evidence = [EvidenceRecord.from_dict(row) for row in read_jsonl(args.evidence)]
    if not claims:
        raise ValueError("Claim manifest is empty")
    assert_no_group_leakage(claims)
    splits = partition(claims)
    if any(not rows for rows in splits.values()):
        raise ValueError(f"Every temporal split must be non-empty: { {k: len(v) for k, v in splits.items()} }")
    store = EmbeddingStore.from_npz(args.text_features, args.image_features)
    enforce_time_filter = args.variant not in {"vanilla_rag", "no_temporal_filter"}
    use_provenance = args.variant not in {"vanilla_rag", "time_filtered_rag", "no_provenance"}
    use_evidence_text = args.variant != "no_text_evidence"
    use_visual_evidence = args.variant != "no_visual_evidence"
    use_temporal_features = args.variant != "no_temporal_features"
    active_splits = {
        split: records
        for split, records in splits.items()
        if not (args.selection_only and split == "test")
    }
    examples = {
        split: build_examples(
            records,
            evidence,
            store,
            enforce_time_filter=enforce_time_filter,
            use_provenance=use_provenance,
            use_evidence_text=use_evidence_text,
            use_visual_evidence=use_visual_evidence,
            use_temporal_features=use_temporal_features,
        )
        for split, records in active_splits.items()
    }
    training = config.get("training", {})
    model = config.get("model", {})
    settings = TrainingSettings(
        variant=args.variant,
        seed=args.seed,
        hidden_dimension=args.hidden_dimension or int(model.get("hidden_dimension", 256)),
        graph_layers=args.graph_layers or int(model.get("graph_layers", 2)),
        dropout=args.dropout if args.dropout is not None else float(model.get("dropout", 0.2)),
        batch_size=args.batch_size or int(training.get("batch_size", 16)),
        epochs=args.epochs or int(training.get("epochs", 30)),
        learning_rate=args.learning_rate or float(training.get("learning_rate", 2e-4)),
        weight_decay=(
            args.weight_decay
            if args.weight_decay is not None
            else float(training.get("weight_decay", 1e-4))
        ),
        sufficiency_loss_weight=float(training.get("sufficiency_loss_weight", 0.25)),
        early_stopping_patience=int(training.get("early_stopping_patience", 5)),
        target_coverage=float(model.get("target_answer_coverage", 0.80)),
        num_workers=int(training.get("num_workers", 0)),
        precision=args.precision or str(training.get("precision", "fp32")),
        evidence_dropout=float(training.get("evidence_dropout", 0.10)),
    )
    result = train_model(
        examples["train"],
        examples["validation"],
        None if args.selection_only else examples["test"],
        settings,
        args.output_dir,
        device_name=args.device,
    )
    run_manifest = {
        "config": str(config_path),
        "claims": str(Path(args.claims).resolve()),
        "evidence": str(Path(args.evidence).resolve()),
        "text_features": str(Path(args.text_features).resolve()),
        "image_features": str(Path(args.image_features).resolve()) if args.image_features else None,
        "variant": args.variant,
        "seed": args.seed,
        "result": result,
    }
    Path(args.output_dir, "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "variant": args.variant,
                "seed": args.seed,
                "best_validation_macro_f1": result["best_validation_macro_f1"],
                "predictions": str(
                    Path(
                        args.output_dir,
                        "validation_predictions.jsonl" if args.selection_only else "predictions.jsonl",
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
