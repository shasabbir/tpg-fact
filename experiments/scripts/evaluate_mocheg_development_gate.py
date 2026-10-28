from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.selection.mocheg_gate import evaluate_mocheg_development_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every MOCHEG Section 13 development condition.")
    parser.add_argument("--selector-audit", type=Path, default=Path("outputs/mocheg/full_corpus_selector_trial_002/audit.json"))
    parser.add_argument("--verifier-audit", type=Path, default=Path("outputs/mocheg/full_corpus_verifier_trial_002/audit.json"))
    parser.add_argument("--retrieval-audit", type=Path, default=Path("outputs/mocheg/full_corpus_features.audit.json"))
    parser.add_argument("--reproduction-audit", type=Path, default=Path("outputs/mocheg/clean_rerun/audit.json"))
    parser.add_argument("--test-record", type=Path, default=Path("outputs/tests/full_suite.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mocheg/development_gate.json"))
    args = parser.parse_args()
    result = evaluate_mocheg_development_gate(
        args.selector_audit, args.verifier_audit, args.retrieval_audit,
        args.reproduction_audit, args.test_record, args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
