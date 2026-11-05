from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.calibration_corrected_reporting import (
    build_calibration_corrected_reporting,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corrected study reporting artifacts.")
    parser.add_argument("--retrieval-root", type=Path, default=Path("outputs/q1_phase2/retrieval"))
    parser.add_argument("--selector-root", type=Path, default=Path("outputs/calibration_corrected_study/selector"))
    parser.add_argument("--verifier-root", type=Path, default=Path("outputs/calibration_corrected_study/verifier"))
    parser.add_argument("--robustness-root", type=Path, default=Path("outputs/calibration_corrected_study/robustness"))
    parser.add_argument("--research-root", type=Path, default=Path("../research/calibration_corrected_study"))
    parser.add_argument("--generated-root", type=Path, default=Path("../research/calibration_corrected_study/generated"))
    parser.add_argument("--figure-root", type=Path, default=Path("../research/calibration_corrected_study/figures"))
    args = parser.parse_args()
    result = build_calibration_corrected_reporting(
        args.retrieval_root,
        args.selector_root,
        args.verifier_root,
        args.robustness_root,
        args.research_root,
        args.generated_root,
        args.figure_root,
    )
    print(json.dumps({"status": result["status"], "profile": result["profile"]}, indent=2))


if __name__ == "__main__":
    main()
