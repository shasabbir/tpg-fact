from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.selection.mocheg_dacs_audit import audit_mocheg_dacs


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit the MOCHEG DACS control.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mocheg/full_corpus_dacs_trial_002"))
    args = parser.parse_args()
    result = audit_mocheg_dacs(args.output_dir)
    print(json.dumps({"status": result["status"], "discrepancies": result["discrepancies"]}, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
