from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.features import build_feature_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed/sciver"))
    parser.add_argument("--output", type=Path, default=Path("data/features/hash"))
    parser.add_argument("--backend", choices=("hash", "hf"), default="hash")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=12)
    args = parser.parse_args()
    results = {}
    for split in ("train", "validation", "test"):
        results[split] = build_feature_store(
            args.processed / f"{split}.jsonl",
            args.output / f"{split}.npz",
            backend=args.backend,
            dim=args.dim,
            max_candidates=args.max_candidates,
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
