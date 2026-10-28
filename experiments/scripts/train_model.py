from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.training import TrainConfig, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/features/hash"))
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--variant", choices=("claim_only", "relevance", "flat", "trace_scv"), default="trace_scv")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--citation-budget", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-counterfactual", action="store_true")
    args = parser.parse_args()
    config = TrainConfig(
        seed=args.seed,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        citation_budget=args.citation_budget,
        device=args.device,
        deletion_weight=0.0 if args.no_counterfactual else 0.25,
        irrelevant_weight=0.0 if args.no_counterfactual else 0.15,
        duplicate_weight=0.0 if args.no_counterfactual else 0.15,
        swap_weight=0.0 if args.no_counterfactual else 0.15,
    )
    print(
        json.dumps(
            train_model(
                args.features / "train.npz",
                args.features / "validation.npz",
                args.output,
                config,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
