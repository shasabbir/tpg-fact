from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from data.io import read_jsonl, write_json, write_jsonl
from data.schema import ClaimRecord, EvidenceRecord
from data.splits import split_name
from data.temporal import is_time_valid
from evaluation.retrieval import ndcg_at_k, recall_at_k
from retrieval.dense import BgeReranker, cosine_rank
from retrieval.hybrid import bm25_rank, reciprocal_rank_fusion
from training.dataset import EmbeddingStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve and rerank a frozen evidence corpus for every claim"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--text-features", required=True)
    parser.add_argument("--image-features")
    parser.add_argument("--evidence-text")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Diagnostic only: preserve fused order instead of running the pinned reranker",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_claim(
    claim: ClaimRecord,
    evidence: list[EvidenceRecord],
    store: EmbeddingStore,
    evidence_text: dict[str, str],
    *,
    rrf_k: int,
    candidate_limit: int,
    retained_limit: int,
    reranker: BgeReranker | None,
    reranker_batch_size: int,
) -> tuple[list[str], dict[str, object]]:
    identifiers = [item.evidence_id for item in evidence]
    documents = {item.evidence_id: evidence_text[item.evidence_id] for item in evidence}
    lexical = bm25_rank(claim.claim_text, documents)
    dense_matrix = np.stack([store.text[item_id] for item_id in identifiers])
    dense = [
        item_id
        for item_id, _ in cosine_rank(
            store.text[claim.claim_id], dense_matrix, identifiers
        )
    ]
    rankings: list[list[str]] = [lexical, dense]
    visual_candidates = [item_id for item_id in identifiers if item_id in store.image]
    if claim.claim_id in store.image and visual_candidates:
        visual_matrix = np.stack([store.image[item_id] for item_id in visual_candidates])
        rankings.append(
            [
                item_id
                for item_id, _ in cosine_rank(
                    store.image[claim.claim_id], visual_matrix, visual_candidates
                )
            ]
        )
    fused = reciprocal_rank_fusion(rankings, k=rrf_k)
    candidates = [item_id for item_id, _ in fused[:candidate_limit]]
    if reranker is None:
        ranked = candidates
    else:
        ranked = [
            item_id
            for item_id, _ in reranker.rank(
                claim.claim_text,
                [(item_id, evidence_text[item_id]) for item_id in candidates],
                batch_size=reranker_batch_size,
            )
        ]
    evidence_by_id = {item.evidence_id: item for item in evidence}
    time_valid_ranked = [
        item_id
        for item_id in ranked
        if is_time_valid(evidence_by_id[item_id], claim.decision_cutoff)
    ]
    selected = time_valid_ranked[:retained_limit]
    relevant = set(claim.evidence_ids)
    log = {
        "claim_id": claim.claim_id,
        "candidate_count": len(evidence),
        "fused_candidate_ids": candidates,
        "reranked_ids": ranked,
        "time_valid_ranked_ids": time_valid_ranked,
        "selected_evidence_ids": selected,
        "gold_relevant_evidence_ids": sorted(relevant),
        "recall_at_10": recall_at_k(time_valid_ranked, relevant, 10),
        "ndcg_at_10": ndcg_at_k(time_valid_ranked, relevant, 10),
    }
    return selected, log


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    claims_path = Path(args.claims).resolve()
    evidence_path = Path(args.evidence).resolve()
    claims = [ClaimRecord.from_dict(row) for row in read_jsonl(claims_path)]
    evidence = [EvidenceRecord.from_dict(row) for row in read_jsonl(evidence_path)]
    if not claims or not evidence:
        raise ValueError("Claims and evidence must be non-empty")
    store = EmbeddingStore.from_npz(args.text_features, args.image_features)
    missing_text = [
        identifier
        for identifier in [
            *(claim.claim_id for claim in claims),
            *(item.evidence_id for item in evidence),
        ]
        if identifier not in store.text
    ]
    if missing_text:
        raise ValueError(f"Missing text embeddings: {missing_text[:5]}")
    text_by_id = {item.evidence_id: item.title for item in evidence}
    evidence_text_path = Path(args.evidence_text).resolve() if args.evidence_text else None
    if evidence_text_path:
        for row in read_jsonl(evidence_text_path):
            identifier = str(row["evidence_id"])
            if identifier in text_by_id:
                text_by_id[identifier] = str(row.get("text") or row.get("title") or "")
    retrieval = config["retrieval"]
    reranker = None
    if not args.skip_reranker:
        reranker = BgeReranker(
            retrieval["reranker_model"], revision=retrieval["reranker_revision"]
        )
    output_root = Path(args.output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, object]] = []
    retrieved_claims: list[ClaimRecord] = []
    for claim in claims:
        selected, log = _rank_claim(
            claim,
            evidence,
            store,
            text_by_id,
            rrf_k=int(retrieval["rrf_k"]),
            candidate_limit=int(retrieval["candidate_limit"]),
            retained_limit=int(retrieval["retained_limit"]),
            reranker=reranker,
            reranker_batch_size=args.reranker_batch_size,
        )
        retrieved_claims.append(replace(claim, evidence_ids=tuple(selected)))
        logs.append(log)
    claims_output = output_root / "retrieved_claims.jsonl"
    log_output = output_root / "retrieval_log.jsonl"
    write_jsonl(claims_output, (claim.to_dict() for claim in retrieved_claims))
    write_jsonl(log_output, logs)
    recalls = [float(row["recall_at_10"]) for row in logs]
    ndcgs = [float(row["ndcg_at_10"]) for row in logs]
    claim_by_id = {claim.claim_id: claim for claim in claims}
    metrics_by_split: dict[str, dict[str, float | int]] = {}
    for split in ("train", "validation", "test"):
        split_logs = [
            row
            for row in logs
            if split_name(claim_by_id[str(row["claim_id"])]) == split
        ]
        metrics_by_split[split] = {
            "claim_count": len(split_logs),
            "recall_at_10": float(np.mean([float(row["recall_at_10"]) for row in split_logs]))
            if split_logs
            else 0.0,
            "ndcg_at_10": float(np.mean([float(row["ndcg_at_10"]) for row in split_logs]))
            if split_logs
            else 0.0,
        }
    report = {
        "status": "completed_frozen_retrieval",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": str(config_path),
        "claims": str(claims_path),
        "claims_sha256": _sha256(claims_path),
        "evidence": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "text_features": str(Path(args.text_features).resolve()),
        "image_features": str(Path(args.image_features).resolve()) if args.image_features else None,
        "reranker": None if args.skip_reranker else retrieval["reranker_model"],
        "reranker_revision": None if args.skip_reranker else retrieval["reranker_revision"],
        "diagnostic_skip_reranker": bool(args.skip_reranker),
        "claim_count": len(claims),
        "evidence_count": len(evidence),
        "selected_count_distribution": dict(
            sorted(Counter(len(claim.evidence_ids) for claim in retrieved_claims).items())
        ),
        "retrieval_recall_at_10": float(np.mean(recalls)),
        "retrieval_ndcg_at_10": float(np.mean(ndcgs)),
        "metrics_by_split": metrics_by_split,
        "retrieved_claims": str(claims_output),
        "retrieved_claims_sha256": _sha256(claims_output),
        "retrieval_log": str(log_output),
        "retrieval_log_sha256": _sha256(log_output),
    }
    write_json(output_root / "retrieval_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
