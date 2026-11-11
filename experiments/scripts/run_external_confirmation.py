from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.data.io import sha256_file
from trace_scv.external_study import run_locked_development, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one locked AVerImaTeC development evaluation.")
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/external_confirmation/averimatec/prepared"))
    parser.add_argument("--freeze", type=Path, default=Path("outputs/external_confirmation/averimatec/train_only.freeze.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/external_confirmation/averimatec/locked_development"))
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"locked development output already exists: {args.output_root}")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_train_only" or freeze.get("development_outcomes_accessed"):
        raise RuntimeError("invalid train-only freeze")
    candidate_path = args.prepared_root / "candidate_records.jsonl"
    gold_path = args.prepared_root / "full_gold_sources.json"
    if sha256_file(candidate_path) != freeze["inputs"]["candidate_records"]["sha256"]:
        raise RuntimeError("candidate records changed after the train-only freeze")
    if sha256_file(gold_path) != freeze["inputs"]["full_gold_sources"]["sha256"]:
        raise RuntimeError("gold-source mapping changed after the train-only freeze")
    summary, rows = run_locked_development(
        candidate_path,
        gold_path,
        freeze,
        maximum_items=10,
        bootstrap_samples=10_000,
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    records_path = args.output_root / "claim_records.jsonl"
    write_jsonl(records_path, rows)
    summary["artifacts"] = {
        "claim_records": {"path": records_path.as_posix(), "sha256": sha256_file(records_path), "rows": len(rows)},
        "train_only_freeze": {"path": args.freeze.as_posix(), "sha256": sha256_file(args.freeze)},
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "claims": summary["claims"], "output": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
