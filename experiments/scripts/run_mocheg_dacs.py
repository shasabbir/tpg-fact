from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.selection.mocheg_dacs import run_mocheg_dacs_count_control


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the matched MOCHEG DACS count-classification control.")
    parser.add_argument("--features", type=Path, default=Path("outputs/mocheg/full_corpus_features.npz"))
    parser.add_argument("--selector-dir", type=Path, default=Path("outputs/mocheg/full_corpus_selector_trial_002"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mocheg/full_corpus_dacs_trial_002"))
    args = parser.parse_args()
    result = run_mocheg_dacs_count_control(args.features, args.selector_dir, args.output_dir)
    print(json.dumps({"profile": result["profile"], "mean_dacs_evidence_f1": result["mean_dacs_evidence_f1"], "mean_cefes_minus_dacs_f1": result["mean_cefes_minus_dacs_f1"]}, indent=2))


if __name__ == "__main__":
    main()
