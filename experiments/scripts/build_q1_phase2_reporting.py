from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.q1_phase2_reporting import build_q1_phase2_reporting


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited Q1 Phase-2 reporting artifacts.")
    parser.add_argument("--original-root", type=Path, default=Path("outputs/q1_phase2"))
    parser.add_argument("--clean-root", type=Path, default=Path("outputs/q1_phase2/clean_rerun"))
    parser.add_argument("--research-root", type=Path, default=Path("../research/q1_phase2"))
    parser.add_argument("--generated-root", type=Path, default=Path("../research/q1_phase2/generated"))
    parser.add_argument("--figure-root", type=Path, default=Path("../research/q1_phase2/figures"))
    args = parser.parse_args()
    result = build_q1_phase2_reporting(
        args.original_root,
        args.clean_root,
        args.research_root,
        args.generated_root,
        args.figure_root,
    )
    print(json.dumps({"status": result["status"], "final_gate_pass": result["final_gate_pass"]}, indent=2))


if __name__ == "__main__":
    main()
