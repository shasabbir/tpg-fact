from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.evaluation.study import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=Path("data/features/hash/test.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_checkpoint(
                args.checkpoint,
                args.features,
                args.output,
                batch_size=args.batch_size,
                bootstrap_samples=args.bootstrap,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
