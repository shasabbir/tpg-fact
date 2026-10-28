from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.data.sciver import prepare_sciver


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize SciVer without touching its test labels.")
    parser.add_argument("--data", type=Path, default=Path("data/public/sciver"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/sciver"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    parser.add_argument("--max-test", type=int)
    args = parser.parse_args()
    result = prepare_sciver(
        args.data,
        args.output,
        seed=args.seed,
        max_candidates=args.max_candidates,
        max_train=args.max_train,
        max_validation=args.max_validation,
        max_test=args.max_test,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
