from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from trace_scv.publication_verifier import (
    audit_stronger_verifier,
    run_stronger_verifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen stronger shared verifier safeguard."
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
        "--selector",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/selector_base"),
    )
    parser.add_argument(
        "--completeness",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/completeness_switch"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/verifier"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    resolve = lambda value: value if value.is_absolute() else project_root / value
    output = resolve(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = run_stronger_verifier(
        resolve(args.retrieval).resolve(),
        resolve(args.selector).resolve(),
        resolve(args.completeness).resolve(),
        resolve(args.protocol).resolve(),
        output,
    )
    audit = audit_stronger_verifier(output)
    (output / "verifier.audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    metric_means = {
        name: fmean(float(seed["metrics"][name]["macro_f1"]) for seed in summary["seeds"])
        for name in ("mlp_threshold", "mlp_switch", "linear_threshold", "linear_switch")
    }
    print(
        json.dumps(
            {
                "status": summary["status"],
                "mean_macro_f1": metric_means,
                "aggregate_switch_minus_threshold": {
                    key: value
                    for key, value in summary[
                        "aggregate_seed_aligned_switch_minus_threshold"
                    ].items()
                    if key != "samples"
                },
                "noninferiority_margin": summary["noninferiority_margin"],
                "aggregate_noninferiority_pass": summary["aggregate_noninferiority_pass"],
                "each_seed_noninferiority_pass": summary["each_seed_noninferiority_pass"],
                "audit": audit,
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

