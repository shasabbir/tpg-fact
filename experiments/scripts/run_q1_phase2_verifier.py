from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.q1_phase2_verifier import (
    Phase2VerifierConfig,
    audit_phase2_verifier,
    run_phase2_verifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase-2 shared verifier.")
    parser.add_argument("action", choices=("run", "audit"))
    parser.add_argument(
        "--retrieval-root", type=Path, default=Path("outputs/q1_phase2/retrieval")
    )
    parser.add_argument(
        "--selector-root", type=Path, default=Path("outputs/q1_phase2/selector")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/q1_phase2/verifier")
    )
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    if args.action == "run":
        result = run_phase2_verifier(
            args.retrieval_root,
            args.selector_root,
            args.output_root,
            config=Phase2VerifierConfig(epochs=args.epochs),
        )
    else:
        result = audit_phase2_verifier(args.output_root)
    print(json.dumps({"status": result["status"], "profile": result["profile"]}, indent=2))


if __name__ == "__main__":
    main()
