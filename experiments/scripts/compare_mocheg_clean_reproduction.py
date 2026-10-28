from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.selection.mocheg_reproduction import compare_mocheg_clean_reproduction


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare independent MOCHEG development rerun artifacts.")
    parser.add_argument("--reference-feature", type=Path, default=Path("outputs/mocheg/full_corpus_features.npz"))
    parser.add_argument("--rerun-feature", type=Path, default=Path("outputs/mocheg/clean_rerun/full_corpus_features.npz"))
    parser.add_argument("--reference-selector", type=Path, default=Path("outputs/mocheg/full_corpus_selector_trial_002"))
    parser.add_argument("--rerun-selector", type=Path, default=Path("outputs/mocheg/clean_rerun/selector_trial_002"))
    parser.add_argument("--reference-verifier", type=Path, default=Path("outputs/mocheg/full_corpus_verifier_trial_002"))
    parser.add_argument("--rerun-verifier", type=Path, default=Path("outputs/mocheg/clean_rerun/verifier_trial_002"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mocheg/clean_rerun/audit.json"))
    args = parser.parse_args()
    result = compare_mocheg_clean_reproduction(
        args.reference_feature, args.rerun_feature,
        args.reference_selector, args.rerun_selector,
        args.reference_verifier, args.rerun_verifier, args.output,
    )
    print(json.dumps({"status": result["status"], "discrepancies": result["discrepancies"]}, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
