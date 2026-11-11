from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from trace_scv.data.io import sha256_file
from trace_scv.external_study import freeze_train_only_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze all AVerImaTeC train-only decisions.")
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/prepared_train"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/external_confirmation/averimatec.yaml"),
    )
    parser.add_argument(
        "--access-audit",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/access_audit.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/external_confirmation/averimatec/train_only.freeze.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    candidate_path = args.prepared_root / "candidate_records.jsonl"
    gold_path = args.prepared_root / "full_gold_sources.json"
    result = freeze_train_only_models(
        candidate_path,
        gold_path,
        seeds=(13, 37, 73, 101, 137),
        fixed_budgets=(1, 3, 5, 10),
        thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        text_widths=(50, 100),
        image_widths=(10, 20),
        maximum_items=10,
        maximum_fit_negatives_per_claim=40,
        learning_rate=0.05,
        epochs=400,
        l2=1e-3,
    )
    result["inputs"] = {
        "candidate_records": {"path": candidate_path.as_posix(), "sha256": sha256_file(candidate_path)},
        "full_gold_sources": {"path": gold_path.as_posix(), "sha256": sha256_file(gold_path)},
        "prejoin_rankings": {
            "path": (args.prepared_root / "rankings.prejoin.jsonl").as_posix(),
            "sha256": sha256_file(args.prepared_root / "rankings.prejoin.jsonl"),
        },
        "preparation_audit": {
            "path": (args.prepared_root / "preparation.audit.json").as_posix(),
            "sha256": sha256_file(args.prepared_root / "preparation.audit.json"),
        },
        "reachability_audit": {
            "path": (args.prepared_root / "reachability.audit.json").as_posix(),
            "sha256": sha256_file(args.prepared_root / "reachability.audit.json"),
        },
        "access_audit": {
            "path": args.access_audit.as_posix(),
            "sha256": sha256_file(args.access_audit),
        },
    }
    code_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    tracked_changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        text=True,
        encoding="utf-8",
    ).splitlines()
    result["protocol_identity"] = {
        "code_revision": code_revision,
        "tracked_worktree_clean": not tracked_changes,
        "config": {
            "path": args.config.as_posix(),
            "sha256": sha256_file(args.config),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "seeds": len(result["seeds"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
