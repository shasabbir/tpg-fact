from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.q1_phase2_robustness import (
    audit_q1_phase2_robustness,
    run_q1_phase2_robustness,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory Phase-3 robustness diagnostics.")
    parser.add_argument("--retrieval-root", type=Path, default=Path("outputs/q1_phase2/retrieval"))
    parser.add_argument("--selector-root", type=Path, default=Path("outputs/q1_phase2/selector"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/q1_phase2/robustness"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        result = audit_q1_phase2_robustness(args.selector_root, args.output_root)
        print(json.dumps({"status": result["status"], "profile": result["profile"]}, indent=2))
    else:
        result = run_q1_phase2_robustness(
            args.retrieval_root,
            args.selector_root,
            args.output_root,
            bootstrap_samples=args.bootstrap_samples,
        )
        print(json.dumps({"status": result["status"], "confirmatory": result["confirmatory"]}, indent=2))


if __name__ == "__main__":
    main()
