from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from trace_scv.q1_phase2_study import (
    Phase2StudyConfig,
    audit_phase2_selector_study,
    run_phase2_selector_study,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Q1 Phase-2 matched selector study.")
    parser.add_argument("action", choices=("run", "audit"))
    parser.add_argument(
        "--retrieval-root", type=Path, default=Path("outputs/q1_phase2/retrieval")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/q1_phase2/selector")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/q1_phase2/protocol.yaml")
    )
    parser.add_argument("--logistic-epochs", type=int, default=400)
    parser.add_argument("--count-epochs", type=int, default=300)
    args = parser.parse_args()
    config = replace(
        Phase2StudyConfig.from_protocol(args.protocol),
        logistic_epochs=args.logistic_epochs,
        count_epochs=args.count_epochs,
    )
    if args.action == "run":
        result = run_phase2_selector_study(
            args.retrieval_root,
            args.output_root,
            args.protocol,
            config=config,
        )
    else:
        result = audit_phase2_selector_study(args.output_root)
    print(json.dumps({"status": result["status"], "profile": result["profile"]}, indent=2))


if __name__ == "__main__":
    main()
