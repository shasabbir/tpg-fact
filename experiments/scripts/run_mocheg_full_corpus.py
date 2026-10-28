from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.mocheg_features import MochegEncoderConfig, TransformersMochegEncoder
from trace_scv.mocheg_full_corpus import (
    FullCorpusRetrievalConfig,
    acquire_full_development_image_corpus,
    build_full_corpus_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sealed MOCHEG full-corpus development retrieval.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/public/mocheg"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/mocheg_development"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mocheg/full_corpus_features.npz"))
    parser.add_argument("--download-media", action="store_true")
    parser.add_argument("--build-features", action="store_true")
    parser.add_argument("--max-download-workers", type=int, default=12)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=64)
    args = parser.parse_args()
    result: dict[str, object] = {}
    if args.download_media:
        result["media"] = acquire_full_development_image_corpus(
            args.dataset_root, max_workers=args.max_download_workers
        )
    if args.build_features:
        result["features"] = build_full_corpus_features(
            args.dataset_root,
            args.processed_root,
            args.output,
            encoder=TransformersMochegEncoder(
                MochegEncoderConfig(
                    device="cuda",
                    text_batch_size=args.text_batch_size,
                    image_batch_size=args.image_batch_size,
                )
            ),
            config=FullCorpusRetrievalConfig(),
        )
    print(json.dumps({key: {"status": value["status"]} for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
