from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.q1_phase2_retrieval import (
    Phase2RetrievalConfig,
    audit_phase2_retrieval,
    build_phase2_embedding_cache,
    run_phase2_retrieval,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the test-sealed Q1 Phase-2 retrieval experiment."
    )
    parser.add_argument(
        "action", choices=("build-cache", "retrieve", "audit")
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/public/mocheg")
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/mocheg_development"),
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("cache/q1_phase2")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/q1_phase2/retrieval")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/q1_phase2/protocol.yaml"),
    )
    parser.add_argument("--claim-limit", type=int)
    parser.add_argument("--skip-cross-encoder", action="store_true")
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--dense-query-batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    args = parser.parse_args()
    config = Phase2RetrievalConfig(
        text_batch_size=args.text_batch_size,
        image_batch_size=args.image_batch_size,
        dense_query_batch_size=args.dense_query_batch_size,
        reranker_batch_size=args.reranker_batch_size,
    )
    if args.action == "build-cache":
        result = build_phase2_embedding_cache(
            args.dataset_root,
            args.processed_root,
            args.cache_root,
            args.protocol,
            config=config,
        )
    elif args.action == "retrieve":
        result = run_phase2_retrieval(
            args.dataset_root,
            args.processed_root,
            args.cache_root,
            args.output_root,
            args.protocol,
            include_cross_encoder=not args.skip_cross_encoder,
            claim_limit=args.claim_limit,
            config=config,
        )
    else:
        result = audit_phase2_retrieval(args.output_root)
    print(json.dumps({"status": result["status"], "profile": result["profile"]}, indent=2))


if __name__ == "__main__":
    main()
