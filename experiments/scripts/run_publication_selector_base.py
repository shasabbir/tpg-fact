from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from trace_scv.publication_selector import run_publication_selector_base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the matched publication selector baselines on repaired packets."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/configs/publication_confirmation.yaml"),
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/retrieval_candidates"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/selector_base"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    resolve = lambda value: value if value.is_absolute() else project_root / value
    output = resolve(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary, audit = run_publication_selector_base(
        resolve(args.retrieval).resolve(),
        output,
        resolve(args.protocol).resolve(),
    )
    method_means = {
        method: fmean(float(seed["validation"][method]["f1"]) for seed in summary["seeds"])
        for method in ("fixed", "threshold", "count", "cefes")
    }
    selected_counts = {
        method: fmean(
            float(seed["validation"][method]["selected_count"]) for seed in summary["seeds"]
        )
        for method in ("fixed", "threshold", "count", "cefes")
    }
    print(
        json.dumps(
            {
                "status": summary["status"],
                "method_mean_f1": method_means,
                "method_mean_selected_count": selected_counts,
                "aggregate_vs_fixed": summary["aggregate_seed_aligned_vs_fixed"],
                "audit": audit,
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

