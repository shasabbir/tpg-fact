from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.mocheg_full_corpus_audit import audit_full_corpus_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit MOCHEG full-corpus retrieval.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/public/mocheg"))
    parser.add_argument("--features", type=Path, default=Path("outputs/mocheg/full_corpus_features.npz"))
    args = parser.parse_args()
    result = audit_full_corpus_features(args.dataset_root, args.features)
    print(json.dumps({"status": result["status"], "discrepancies": result["discrepancies"]}, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
