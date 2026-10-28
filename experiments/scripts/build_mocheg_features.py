from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.mocheg_features import (
    MochegEncoderConfig,
    HashMochegEncoder,
    TransformersMochegEncoder,
    build_mocheg_pair_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pinned MOCHEG pair features.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/mocheg_development"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/mocheg/pilot_features.npz")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=("transformers", "hash"), default="transformers")
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=64)
    args = parser.parse_args()
    config = MochegEncoderConfig(
        device=args.device,
        text_batch_size=args.text_batch_size,
        image_batch_size=args.image_batch_size,
    )
    encoder = (
        TransformersMochegEncoder(config)
        if args.backend == "transformers"
        else HashMochegEncoder()
    )
    result = build_mocheg_pair_features(
        [
            args.processed_root / "train_pilot_packets.jsonl",
            args.processed_root / "validation_pilot_packets.jsonl",
        ],
        args.output,
        encoder=encoder,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
