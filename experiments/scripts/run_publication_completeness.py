from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.publication_completeness import (
    audit_completeness_switch,
    run_completeness_switch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen retrieval-completeness-aware selector."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/configs/publication_confirmation.yaml"),
    )
    parser.add_argument(
        "--selector",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/selector_base"),
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/retrieval_candidates"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/completeness_switch"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    resolve = lambda value: value if value.is_absolute() else project_root / value
    output = resolve(args.output).resolve()
    summary = run_completeness_switch(
        resolve(args.selector).resolve(),
        resolve(args.retrieval).resolve(),
        resolve(args.protocol).resolve(),
        output,
    )
    audit = audit_completeness_switch(output)
    (output / "completeness.audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    compact_seeds = [
        {
            "seed": row["seed"],
            "cutoff": row["selected_completeness_cutoff"],
            "expected_use_share": row["validation_expected_use_share"],
            "threshold": row["validation"]["threshold"],
            "expected_f1": row["validation"]["expected_f1"],
            "uncalibrated_expected_f1": row["validation"]["uncalibrated_expected_f1"],
            "switch": row["validation"]["completeness_switch"],
            "effect": row["paired_switch_vs_threshold"]["mean"],
        }
        for row in summary["seeds"]
    ]
    print(
        json.dumps(
            {
                "status": summary["status"],
                "seeds": compact_seeds,
                "aggregate_switch_vs_threshold": {
                    key: value
                    for key, value in summary["aggregate_switch_vs_threshold"].items()
                    if key != "samples"
                },
                "superiority_gate": summary["superiority_gate"],
                "audit": audit,
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

