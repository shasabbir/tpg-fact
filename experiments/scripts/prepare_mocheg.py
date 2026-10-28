from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.data.mocheg import (
    build_mocheg_pilot_packets,
    download_mocheg_development_metadata,
    download_mocheg_pilot_media,
    prepare_mocheg_sentence_text_candidates,
    run_mocheg_development_preflight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sealed MOCHEG train/development data without test access."
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/public/mocheg")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/mocheg_development")
    )
    parser.add_argument("--download-metadata", action="store_true")
    parser.add_argument("--download-pilot-media", action="store_true")
    parser.add_argument("--build-sentence-text", action="store_true")
    parser.add_argument("--build-pilot-packets", action="store_true")
    parser.add_argument("--nltk-data-root", type=Path)
    parser.add_argument("--max-train-media-claims", type=int, default=200)
    parser.add_argument("--max-validation-media-claims", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-download-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_metadata:
        download_mocheg_development_metadata(args.dataset_root)
    preflight = run_mocheg_development_preflight(
        args.dataset_root, args.output_root
    )
    result: dict[str, object] = {"preflight": preflight}
    if args.download_pilot_media:
        result["media"] = download_mocheg_pilot_media(
            args.dataset_root,
            max_train_claims=args.max_train_media_claims,
            max_validation_claims=args.max_validation_media_claims,
            seed=args.seed,
            max_workers=args.max_download_workers,
        )
    if args.build_sentence_text:
        result["sentence_text"] = prepare_mocheg_sentence_text_candidates(
            args.dataset_root,
            args.output_root,
            nltk_data_root=args.nltk_data_root,
        )
    if args.build_pilot_packets:
        result["pilot_packets"] = build_mocheg_pilot_packets(
            args.dataset_root, args.output_root
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
