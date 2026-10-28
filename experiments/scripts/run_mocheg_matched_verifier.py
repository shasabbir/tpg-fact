from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.selection.mocheg_verifier import (
    MochegVerifierConfig,
    run_matched_mocheg_verifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shared MOCHEG verifier.")
    parser.add_argument(
        "--features", type=Path, default=Path("outputs/mocheg/pilot_features.npz")
    )
    parser.add_argument(
        "--selector-dir",
        type=Path,
        default=Path("outputs/mocheg/selector_pilot_trial_002"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/mocheg_development"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mocheg/matched_verifier_trial_002"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    result = run_matched_mocheg_verifier(
        args.features,
        args.selector_dir,
        args.processed_root,
        args.output_dir,
        config=MochegVerifierConfig(bootstrap_samples=args.bootstrap_samples),
    )
    print(
        json.dumps(
            {
                "profile": result["profile"],
                "mean_cefes_minus_fixed_macro_f1": result[
                    "mean_cefes_minus_fixed_macro_f1"
                ],
                "verdict_noninferiority_point_pass": result[
                    "verdict_noninferiority_point_pass"
                ],
                "formal_test_annotations_accessed": result[
                    "formal_test_annotations_accessed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
