from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.selection.mocheg_study import (
    MochegSelectorConfig,
    run_mocheg_selector_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sealed MOCHEG selector pilot.")
    parser.add_argument(
        "--features", type=Path, default=Path("outputs/mocheg/pilot_features.npz")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/mocheg/selector_pilot")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--calibration-safety-guard", action="store_true")
    args = parser.parse_args()
    config = MochegSelectorConfig(
        bootstrap_samples=args.bootstrap_samples,
        calibration_safety_guard=args.calibration_safety_guard,
    )
    result = run_mocheg_selector_pilot(
        args.features, args.output_dir, config=config
    )
    print(
        json.dumps(
            {
                "profile": result["profile"],
                "mean_cefes_minus_fixed_f1": result[
                    "mean_cefes_minus_fixed_f1"
                ],
                "selector_numeric_criteria_met": result[
                    "selector_numeric_criteria_met"
                ],
                "full_development_gate_pass": result[
                    "full_development_gate_pass"
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
