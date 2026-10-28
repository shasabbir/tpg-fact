from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.selection.mocheg_verifier_audit import audit_matched_mocheg_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit MOCHEG matched-verifier outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mocheg/matched_verifier_trial_002"),
    )
    args = parser.parse_args()
    result = audit_matched_mocheg_verifier(args.output_dir)
    print(json.dumps({"status": result["status"], "discrepancies": result["discrepancies"]}, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
