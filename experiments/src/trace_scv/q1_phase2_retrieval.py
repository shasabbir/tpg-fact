from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import numpy as np

from trace_scv.data.io import sha256_file
from trace_scv.data.mocheg import assert_mocheg_development_sealed, load_mocheg_qrels
from trace_scv.mocheg_features import MochegEncoderConfig, TransformersMochegEncoder


@dataclass(frozen=True, slots=True)
class Phase2RetrievalConfig:
    audit_depth: int = 100
    retained_text_candidates: int = 40
    retained_image_candidates: int = 20
    rrf_constant: int = 60
    text_batch_size: int = 128
    image_batch_size: int = 128
    dense_query_batch_size: int = 32
    reranker_batch_size: int = 64
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    reranker_max_tokens: int = 512


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _claim_role(claim_id: str, seed: int) -> str:
    """Assign a claim to a deterministic 55/15/15/15 training partition."""
    digest = hashlib.sha256(f"{seed}:{claim_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < 0.55:
        return "fit"
    if value < 0.70:
        return "calibration"
    if value < 0.85:
        return "model_selection"
    return "internal_holdout"


def _load_claims(processed_root: Path) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    for filename, split in (
        ("train_claims.jsonl", "train"),
        ("validation_claims.jsonl", "validation"),
    ):
        for row in _read_jsonl(processed_root / filename):
            claims.append(
                {
                    "claim_id": str(row["claim_id"]),
                    "claim": str(row["claim"]),
                    "label": str(row["label"]),
                    "split": split,
                }
            )
    claims.sort(key=lambda row: (row["split"], row["claim_id"]))
    return claims


def _load_sentence_rows(processed_root: Path) -> list[dict[str, object]]:
    rows = _read_jsonl(processed_root / "sentence_text.jsonl")
    for row in rows:
        row["evidence_id"] = f"text:{row['document_id']}"
        row["source_group"] = (
            f"text:{row['source_claim_id']}-{row['source_document_id']}"
        )
    return rows


_IMAGE_PREFIX = re.compile(r"^(?:\d{3,6}[-_])+(?:\d{1,8}[-_])?")
_IMAGE_SUFFIX = re.compile(r"\.(?:jpe?g|png|gif|webp|bmp|tiff?)$", re.IGNORECASE)


def image_filename_text(filename: str) -> str:
    """Convert a release filename into a conservative label-blind text query target."""
    value = _IMAGE_SUFFIX.sub("", Path(filename).name)
    value = _IMAGE_PREFIX.sub("", value)
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"\b\d{4,}\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or "image"


def _load_image_rows(dataset_root: Path) -> list[dict[str, object]]:
    manifest_path = dataset_root / "media" / "full_development_media_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise RuntimeError("full development media manifest has not passed")
    basename_hashes: dict[str, set[str]] = defaultdict(set)
    basename_rows: dict[str, dict[str, object]] = {}
    for row in manifest["downloaded"]:
        basename = str(row["basename"])
        basename_hashes[basename].add(str(row["sha256"]))
        basename_rows.setdefault(basename, row)
    conflicts = [name for name, values in basename_hashes.items() if len(values) > 1]
    if conflicts:
        raise RuntimeError(f"conflicting image basenames: {len(conflicts)}")
    hash_counts: dict[str, int] = defaultdict(int)
    for row in basename_rows.values():
        hash_counts[str(row["sha256"])] += 1
    output: list[dict[str, object]] = []
    for basename in sorted(basename_rows):
        row = basename_rows[basename]
        path = (
            dataset_root
            / "media"
            / str(row["mirror"])
            / str(row["relative_path"])
        )
        output.append(
            {
                "evidence_id": f"image:{basename}",
                "basename": basename,
                "filename_text": image_filename_text(basename),
                "path": str(path),
                "sha256": str(row["sha256"]),
                "source_group": f"image-sha256:{row['sha256']}",
                "duplicate_group_size": hash_counts[str(row["sha256"])],
            }
        )
    return output


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, constant: int, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    if constant <= 0 or depth <= 0:
        raise ValueError("constant and depth must be positive")
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[int] = set()
        for rank, raw_index in enumerate(ranking, start=1):
            index = int(raw_index)
            if index in seen:
                continue
            seen.add(index)
            scores[index] += 1.0 / (constant + rank)
    ordered = sorted(scores, key=lambda index: (-scores[index], index))[:depth]
    return (
        np.asarray(ordered, dtype=np.int64),
        np.asarray([scores[index] for index in ordered], dtype=np.float32),
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def build_phase2_embedding_cache(
    dataset_root: Path,
    processed_root: Path,
    cache_root: Path,
    protocol_path: Path,
    *,
    config: Phase2RetrievalConfig = Phase2RetrievalConfig(),
) -> dict[str, object]:
    """Build reusable label-blind corpus, image, and claim embedding caches."""
    assert_mocheg_development_sealed(dataset_root)
    started = time.perf_counter()
    cache_root.mkdir(parents=True, exist_ok=True)
    claims = _load_claims(processed_root)
    sentences = _load_sentence_rows(processed_root)
    images = _load_image_rows(dataset_root)
    claim_texts = [row["claim"] for row in claims]
    sentence_texts = [str(row["content"]) for row in sentences]
    filename_texts = [str(row["filename_text"]) for row in images]

    encoder_config = MochegEncoderConfig(
        device="cuda",
        text_batch_size=config.text_batch_size,
        image_batch_size=config.image_batch_size,
    )
    encoder = TransformersMochegEncoder(encoder_config)
    bge = encoder.encode_bge([*claim_texts, *sentence_texts, *filename_texts])
    claim_end = len(claims)
    sentence_end = claim_end + len(sentences)
    claim_bge = bge[:claim_end]
    sentence_bge = bge[claim_end:sentence_end]
    filename_bge = bge[sentence_end:]
    del bge

    claim_siglip, image_siglip = encoder.encode_siglip(
        claim_texts, [str(row["path"]) for row in images]
    )

    arrays = {
        "claim_bge": claim_bge,
        "sentence_bge": sentence_bge,
        "filename_bge": filename_bge,
        "claim_siglip": claim_siglip,
        "image_siglip": image_siglip,
    }
    array_artifacts: dict[str, dict[str, object]] = {}
    for name, values in arrays.items():
        path = cache_root / f"{name}.npy"
        np.save(path, _normalize_rows(values).astype(np.float16))
        array_artifacts[name] = {
            "path": str(path),
            "shape": list(values.shape),
            "dtype": "float16",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    claims_path = cache_root / "claims.jsonl"
    images_path = cache_root / "images.jsonl"
    _write_jsonl(claims_path, claims)
    _write_jsonl(images_path, images)
    sentence_path = processed_root / "sentence_text.jsonl"
    source_paths = {
        "train_claims": processed_root / "train_claims.jsonl",
        "validation_claims": processed_root / "validation_claims.jsonl",
        "sentences": sentence_path,
        "media_manifest": dataset_root / "media" / "full_development_media_manifest.json",
        "protocol": protocol_path,
    }
    manifest: dict[str, object] = {
        "profile": "q1_phase2_embedding_cache",
        "status": "pass",
        "config": asdict(config),
        "encoder": encoder.identity(),
        "counts": {
            "claims": len(claims),
            "train_claims": sum(row["split"] == "train" for row in claims),
            "validation_claims": sum(row["split"] == "validation" for row in claims),
            "sentences": len(sentences),
            "images": len(images),
        },
        "arrays": array_artifacts,
        "sidecars": {
            "claims": {"path": str(claims_path), "sha256": sha256_file(claims_path)},
            "images": {"path": str(images_path), "sha256": sha256_file(images_path)},
        },
        "inputs": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source_commit": _source_commit(),
        "gold_labels_used": False,
        "formal_test_annotations_accessed": False,
    }
    manifest_path = cache_root / "embedding_cache.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    assert_mocheg_development_sealed(dataset_root)
    return manifest


def _dense_topk(
    queries: np.ndarray,
    corpus: np.ndarray,
    *,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Phase-2 dense retrieval requires CUDA")
    corpus_tensor = torch.as_tensor(np.asarray(corpus), device="cuda", dtype=torch.float16)
    index_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    for start in range(0, len(queries), batch_size):
        query_tensor = torch.as_tensor(
            np.asarray(queries[start : start + batch_size]),
            device="cuda",
            dtype=torch.float16,
        )
        with torch.inference_mode():
            scores = query_tensor @ corpus_tensor.T
            top_scores, top_indices = torch.topk(
                scores, k=min(k, corpus_tensor.shape[0]), dim=1, sorted=True
            )
        index_rows.append(top_indices.cpu().numpy().astype(np.int64))
        score_rows.append(top_scores.float().cpu().numpy().astype(np.float32))
        del scores, top_scores, top_indices, query_tensor
    del corpus_tensor
    torch.cuda.empty_cache()
    return np.concatenate(index_rows), np.concatenate(score_rows)


def _bm25_topk(
    texts: Sequence[str],
    queries: Sequence[str],
    index_root: Path,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    import bm25s

    index_root.mkdir(parents=True, exist_ok=True)
    params_path = index_root / "params.index.json"
    if params_path.is_file():
        retriever = bm25s.BM25.load(
            index_root,
            load_corpus=False,
            mmap=True,
            show_progress=False,
            verbosity=0,
        )
    else:
        corpus_tokens = bm25s.tokenize(
            list(texts), stopwords="en", show_progress=False
        )
        retriever = bm25s.BM25(method="lucene")
        retriever.index(corpus_tokens, show_progress=False)
        retriever.save(index_root, show_progress=False)
    query_tokens = bm25s.tokenize(list(queries), stopwords="en", show_progress=False)
    indices, scores = retriever.retrieve(
        query_tokens,
        k=min(k, len(texts)),
        show_progress=False,
        n_threads=0,
    )
    return np.asarray(indices, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def _rrf_matrix(
    left: np.ndarray,
    right: np.ndarray,
    *,
    constant: int,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for left_row, right_row in zip(left, right, strict=True):
        row_indices, row_scores = reciprocal_rank_fusion(
            (left_row, right_row), constant=constant, depth=depth
        )
        indices.append(row_indices)
        scores.append(row_scores)
    return np.stack(indices), np.stack(scores)


def _cross_encoder_scores(
    claims: Sequence[str],
    sentence_texts: Sequence[str],
    candidate_indices: np.ndarray,
    *,
    config: Phase2RetrievalConfig,
) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.reranker_model, revision=config.reranker_revision
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.reranker_model,
        revision=config.reranker_revision,
        torch_dtype=torch.float16,
    ).to("cuda").eval()
    flat_claims: list[str] = []
    flat_sentences: list[str] = []
    for claim, indices in zip(claims, candidate_indices, strict=True):
        flat_claims.extend([claim] * len(indices))
        flat_sentences.extend(sentence_texts[int(index)] for index in indices)
    scores: list[np.ndarray] = []
    for start in range(0, len(flat_claims), config.reranker_batch_size):
        batch = tokenizer(
            flat_claims[start : start + config.reranker_batch_size],
            flat_sentences[start : start + config.reranker_batch_size],
            padding=True,
            truncation=True,
            max_length=config.reranker_max_tokens,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            logits = model(**batch).logits.reshape(-1)
        scores.append(logits.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(scores).reshape(candidate_indices.shape).astype(np.float32)


def _sort_candidate_matrix(
    indices: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sorted_indices = np.empty_like(indices)
    sorted_scores = np.empty_like(scores)
    for row_index in range(len(indices)):
        order = np.lexsort((indices[row_index], -scores[row_index]))
        sorted_indices[row_index] = indices[row_index][order]
        sorted_scores[row_index] = scores[row_index][order]
    return sorted_indices, sorted_scores


def _ranking_metrics(ranking: Sequence[str], gold: set[str]) -> dict[str, float]:
    if not gold:
        return {
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "recall_at_50": 0.0,
            "map_at_100": 0.0,
            "ndcg_at_100": 0.0,
        }
    recalls = {
        f"recall_at_{depth}": len(set(ranking[:depth]) & gold) / len(gold)
        for depth in (5, 10, 20, 50)
    }
    hits = 0
    precision_sum = 0.0
    dcg = 0.0
    for rank, evidence_id in enumerate(ranking[:100], start=1):
        if evidence_id in gold:
            hits += 1
            precision_sum += hits / rank
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), 100) + 1))
    return {
        **recalls,
        "map_at_100": precision_sum / len(gold),
        "ndcg_at_100": dcg / ideal if ideal else 0.0,
    }


def _gold_lookup(dataset_root: Path, claims: Sequence[dict[str, str]]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for split, folder in (("train", "train"), ("validation", "val")):
        claim_ids = {row["claim_id"] for row in claims if row["split"] == split}
        for modality, filename in (
            ("text", "text_evidence_qrels_sentence_level.csv"),
            ("image", "img_evidence_qrels.csv"),
        ):
            rows = load_mocheg_qrels(dataset_root / "metadata" / folder / filename)
            for row in rows:
                if row.relevant and row.claim_id in claim_ids:
                    output[f"{split}:{row.claim_id}"].add(
                        f"{modality}:{row.document_id}"
                    )
    return output


def _mean_metric_rows(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys() if rows else ()
    return {key: fmean(row[key] for row in rows) for key in keys}


def _packet_ceiling(oracle_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    eligible = [row for row in oracle_rows if int(row["gold_count"]) > 0]
    if not eligible:
        raise ValueError("packet-ceiling evaluation requires at least one complete-gold claim")
    return {
        "mean_oracle_f1": fmean(float(row["oracle_f1"]) for row in eligible),
        "complete_gold_claim_coverage": fmean(
            float(row["complete_gold_reachable"]) for row in eligible
        ),
        "eligible_claims": len(eligible),
        "claims_without_released_gold": len(oracle_rows) - len(eligible),
        "eligible_claims_with_zero_reachable_gold": sum(
            int(row["reachable_gold_count"]) == 0 for row in eligible
        ),
        "population_policy": "claims with a released non-empty complete-gold evidence set",
    }


def run_phase2_retrieval(
    dataset_root: Path,
    processed_root: Path,
    cache_root: Path,
    output_root: Path,
    protocol_path: Path,
    *,
    include_cross_encoder: bool = True,
    claim_limit: int | None = None,
    config: Phase2RetrievalConfig = Phase2RetrievalConfig(),
) -> dict[str, object]:
    """Run all frozen Phase-2 label-blind retrieval variants and build packets."""
    assert_mocheg_development_sealed(dataset_root)
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache_root / "embedding_cache.manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("status") != "pass":
        raise RuntimeError("Phase-2 embedding cache has not passed")
    claims = _read_jsonl(cache_root / "claims.jsonl")
    images = _read_jsonl(cache_root / "images.jsonl")
    sentences = _load_sentence_rows(processed_root)
    if claim_limit is not None:
        if claim_limit <= 0:
            raise ValueError("claim_limit must be positive")
        claims = claims[:claim_limit]
    claim_texts = [str(row["claim"]) for row in claims]
    sentence_texts = [str(row["content"]) for row in sentences]
    claim_count = len(claims)

    claim_bge = np.load(cache_root / "claim_bge.npy", mmap_mode="r")[:claim_count]
    sentence_bge = np.load(cache_root / "sentence_bge.npy", mmap_mode="r")
    filename_bge = np.load(cache_root / "filename_bge.npy", mmap_mode="r")
    claim_siglip = np.load(cache_root / "claim_siglip.npy", mmap_mode="r")[:claim_count]
    image_siglip = np.load(cache_root / "image_siglip.npy", mmap_mode="r")

    text_dense_indices, text_dense_scores = _dense_topk(
        claim_bge,
        sentence_bge,
        k=config.audit_depth,
        batch_size=config.dense_query_batch_size,
    )
    text_bm25_indices, text_bm25_scores = _bm25_topk(
        sentence_texts,
        claim_texts,
        cache_root / "bm25_sentences",
        k=config.audit_depth,
    )
    text_rrf_indices, text_rrf_scores = _rrf_matrix(
        text_dense_indices,
        text_bm25_indices,
        constant=config.rrf_constant,
        depth=config.audit_depth,
    )
    if include_cross_encoder:
        text_cross_scores = _cross_encoder_scores(
            claim_texts, sentence_texts, text_rrf_indices, config=config
        )
        text_cross_indices, text_cross_scores = _sort_candidate_matrix(
            text_rrf_indices, text_cross_scores
        )
    else:
        text_cross_indices = text_rrf_indices.copy()
        text_cross_scores = text_rrf_scores.copy()

    image_siglip_indices, image_siglip_scores = _dense_topk(
        claim_siglip,
        image_siglip,
        k=config.audit_depth,
        batch_size=config.dense_query_batch_size,
    )
    image_filename_indices, image_filename_scores = _dense_topk(
        claim_bge,
        filename_bge,
        k=config.audit_depth,
        batch_size=config.dense_query_batch_size,
    )
    image_rrf_indices, image_rrf_scores = _rrf_matrix(
        image_siglip_indices,
        image_filename_indices,
        constant=config.rrf_constant,
        depth=config.audit_depth,
    )

    ranking_path = output_root / "rankings.npz"
    np.savez_compressed(
        ranking_path,
        text_dense_indices=text_dense_indices,
        text_dense_scores=text_dense_scores,
        text_bm25_indices=text_bm25_indices,
        text_bm25_scores=text_bm25_scores,
        text_rrf_indices=text_rrf_indices,
        text_rrf_scores=text_rrf_scores,
        text_cross_indices=text_cross_indices,
        text_cross_scores=text_cross_scores,
        image_siglip_indices=image_siglip_indices,
        image_siglip_scores=image_siglip_scores,
        image_filename_indices=image_filename_indices,
        image_filename_scores=image_filename_scores,
        image_rrf_indices=image_rrf_indices,
        image_rrf_scores=image_rrf_scores,
    )
    claim_sidecar_path = output_root / "claims.jsonl"
    _write_jsonl(claim_sidecar_path, claims)
    prejoin_manifest = {
        "profile": "q1_phase2_label_blind_rankings",
        "status": "rankings_frozen_before_gold_join",
        "config": asdict(config),
        "include_cross_encoder": include_cross_encoder,
        "claim_limit": claim_limit,
        "claims": len(claims),
        "sentences": len(sentences),
        "images": len(images),
        "rankings": {
            "path": str(ranking_path),
            "bytes": ranking_path.stat().st_size,
            "sha256": sha256_file(ranking_path),
        },
        "claims_sidecar": {
            "path": str(claim_sidecar_path),
            "bytes": claim_sidecar_path.stat().st_size,
            "sha256": sha256_file(claim_sidecar_path),
        },
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "protocol_sha256": sha256_file(protocol_path),
        "source_commit": _source_commit(),
        "gold_labels_used": False,
        "formal_test_annotations_accessed": False,
    }
    prejoin_path = output_root / "rankings.prejoin.manifest.json"
    prejoin_path.write_text(
        json.dumps(prejoin_manifest, indent=2) + "\n", encoding="utf-8"
    )

    # The gold join happens only after the complete ranking artifact is durable.
    full_gold = _gold_lookup(dataset_root, claims)
    sentence_ids = [str(row["evidence_id"]) for row in sentences]
    image_ids = [str(row["evidence_id"]) for row in images]
    variants = {
        "text": {
            "dense_sentence": text_dense_indices,
            "bm25_sentence": text_bm25_indices,
            "dense_bm25_rrf": text_rrf_indices,
            "dense_bm25_rrf_cross_encoder": text_cross_indices,
        },
        "image": {
            "siglip": image_siglip_indices,
            "filename_dense": image_filename_indices,
            "siglip_filename_rrf": image_rrf_indices,
        },
    }
    ids_by_modality = {"text": sentence_ids, "image": image_ids}
    metric_rows: list[dict[str, object]] = []
    for claim_index, claim in enumerate(claims):
        key = f"{claim['split']}:{claim['claim_id']}"
        gold = full_gold.get(key, set())
        for modality, modality_variants in variants.items():
            modality_gold = {
                value for value in gold if value.startswith(f"{modality}:")
            }
            if not modality_gold:
                continue
            for variant, indices in modality_variants.items():
                ranking = [
                    ids_by_modality[modality][int(index)]
                    for index in indices[claim_index]
                ]
                metric_rows.append(
                    {
                        "split": str(claim["split"]),
                        "claim_id": str(claim["claim_id"]),
                        "modality": modality,
                        "variant": variant,
                        "gold_count": len(modality_gold),
                        **_ranking_metrics(ranking, modality_gold),
                    }
                )
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for modality, modality_variants in variants.items():
        metrics[modality] = {}
        for variant in modality_variants:
            rows = [
                row
                for row in metric_rows
                if row["modality"] == modality and row["variant"] == variant
            ]
            numeric_rows = [
                {
                    key: float(row[key])
                    for key in (
                        "recall_at_5",
                        "recall_at_10",
                        "recall_at_20",
                        "recall_at_50",
                        "map_at_100",
                        "ndcg_at_100",
                    )
                }
                for row in rows
            ]
            metrics[modality][variant] = {
                **_mean_metric_rows(numeric_rows),
                "eligible_claims": float(len(rows)),
            }

    def rank_maps(indices: np.ndarray, scores: np.ndarray, claim_index: int) -> dict[int, tuple[int, float]]:
        return {
            int(index): (rank, float(score))
            for rank, (index, score) in enumerate(
                zip(indices[claim_index], scores[claim_index], strict=True), start=1
            )
        }

    features: list[list[float]] = []
    labels: list[int] = []
    candidate_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    for claim_index, claim in enumerate(claims):
        key = f"{claim['split']}:{claim['claim_id']}"
        gold = full_gold.get(key, set())
        dense_text = rank_maps(text_dense_indices, text_dense_scores, claim_index)
        lexical_text = rank_maps(text_bm25_indices, text_bm25_scores, claim_index)
        rrf_text = rank_maps(text_rrf_indices, text_rrf_scores, claim_index)
        cross_text = rank_maps(text_cross_indices, text_cross_scores, claim_index)
        siglip_image = rank_maps(image_siglip_indices, image_siglip_scores, claim_index)
        filename_image = rank_maps(
            image_filename_indices, image_filename_scores, claim_index
        )
        rrf_image = rank_maps(image_rrf_indices, image_rrf_scores, claim_index)
        retained_text = list(text_cross_indices[claim_index][: config.retained_text_candidates])
        retained_images = list(image_rrf_indices[claim_index][: config.retained_image_candidates])
        retained_ids: set[str] = set()
        for modality, retained in (("text", retained_text), ("image", retained_images)):
            for raw_index in retained:
                corpus_index = int(raw_index)
                if modality == "text":
                    row = sentences[corpus_index]
                    evidence_id = sentence_ids[corpus_index]
                    source_group = str(row["source_group"])
                    token_count = len(str(row["content"]).split())
                    final_rank, final_score = cross_text[corpus_index]
                    dense_rank, dense_score = dense_text.get(corpus_index, (0, 0.0))
                    lexical_rank, lexical_score = lexical_text.get(corpus_index, (0, 0.0))
                    _, rrf_score = rrf_text.get(corpus_index, (0, 0.0))
                    cross_score = final_score
                    duplicate_group_size = 1
                else:
                    row = images[corpus_index]
                    evidence_id = image_ids[corpus_index]
                    source_group = str(row["source_group"])
                    token_count = 0
                    final_rank, final_score = rrf_image[corpus_index]
                    dense_rank, dense_score = siglip_image.get(corpus_index, (0, 0.0))
                    lexical_rank, lexical_score = filename_image.get(corpus_index, (0, 0.0))
                    rrf_score = final_score
                    cross_score = 0.0
                    duplicate_group_size = int(row["duplicate_group_size"])
                retained_ids.add(evidence_id)
                candidate_count = (
                    config.retained_text_candidates + config.retained_image_candidates
                )
                feature_row = [
                    final_score,
                    final_score * final_score,
                    dense_score,
                    lexical_score,
                    rrf_score,
                    cross_score,
                    1.0 / dense_rank if dense_rank else 0.0,
                    1.0 / lexical_rank if lexical_rank else 0.0,
                    1.0 / final_rank if final_rank else 0.0,
                    float(modality == "text"),
                    float(modality == "image"),
                    math.log1p(token_count) / 10.0,
                    math.log1p(len(str(claim["claim"]).split())) / 10.0,
                    math.log1p(candidate_count) / 10.0,
                    math.log1p(duplicate_group_size) / 10.0,
                ]
                features.append(feature_row)
                labels.append(int(evidence_id in gold))
                candidate_rows.append(
                    {
                        "claim_index": claim_index,
                        "claim_id": str(claim["claim_id"]),
                        "split": str(claim["split"]),
                        "claim_label": str(claim["label"]),
                        "evidence_id": evidence_id,
                        "modality": modality,
                        "source_group": source_group,
                        "gold": evidence_id in gold,
                        "final_rank": final_rank,
                        "dense_rank": dense_rank or None,
                        "lexical_rank": lexical_rank or None,
                    }
                )
        reachable = retained_ids & gold
        oracle_f1 = 2.0 * len(reachable) / (len(reachable) + len(gold)) if gold else 1.0
        oracle_rows.append(
            {
                "claim_id": str(claim["claim_id"]),
                "split": str(claim["split"]),
                "gold_count": len(gold),
                "reachable_gold_count": len(reachable),
                "complete_gold_reachable": bool(gold and reachable == gold),
                "oracle_f1": oracle_f1,
            }
        )

    feature_path = output_root / "candidate_features.npz"
    np.savez_compressed(
        feature_path,
        features=np.asarray(features, dtype=np.float32),
        gold=np.asarray(labels, dtype=np.int8),
    )
    candidate_path = output_root / "candidate_records.jsonl"
    oracle_path = output_root / "oracle_records.jsonl"
    metrics_path = output_root / "retrieval_metrics.json"
    full_gold_path = output_root / "full_gold.json"
    _write_jsonl(candidate_path, candidate_rows)
    _write_jsonl(oracle_path, oracle_rows)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    full_gold_path.write_text(
        json.dumps({key: sorted(values) for key, values in sorted(full_gold.items())}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    packet_ceiling = _packet_ceiling(oracle_rows)
    feature_names = [
        "final_score",
        "final_score_squared",
        "dense_score",
        "lexical_or_filename_score",
        "rrf_score",
        "cross_encoder_score",
        "reciprocal_dense_rank",
        "reciprocal_lexical_rank",
        "reciprocal_final_rank",
        "is_text",
        "is_image",
        "log_text_token_count",
        "log_claim_token_count",
        "log_available_candidate_count",
        "log_duplicate_group_size",
    ]
    manifest: dict[str, object] = {
        "profile": "q1_phase2_retrieval_and_candidate_features",
        "status": "pass",
        "config": asdict(config),
        "include_cross_encoder": include_cross_encoder,
        "claim_limit": claim_limit,
        "counts": {
            "claims": len(claims),
            "candidate_rows": len(candidate_rows),
            "gold_candidate_rows": sum(labels),
        },
        "feature_names": feature_names,
        "retrieval_metrics": metrics,
        "packet_ceiling": packet_ceiling,
        "artifacts": {
            "rankings": {"path": str(ranking_path), "sha256": sha256_file(ranking_path)},
            "prejoin_manifest": {"path": str(prejoin_path), "sha256": sha256_file(prejoin_path)},
            "features": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
            "candidates": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "full_gold": {"path": str(full_gold_path), "sha256": sha256_file(full_gold_path)},
        },
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source_commit": _source_commit(),
        "gold_loaded_after_rankings_written": True,
        "formal_test_annotations_accessed": False,
    }
    manifest_path = output_root / "retrieval.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    assert_mocheg_development_sealed(dataset_root)
    return manifest


def audit_phase2_retrieval(output_root: Path) -> dict[str, object]:
    manifest_path = output_root / "retrieval.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("status") != "pass":
        errors.append("retrieval manifest is not passing")
    if not manifest.get("gold_loaded_after_rankings_written"):
        errors.append("rankings were not frozen before the gold join")
    if manifest.get("formal_test_annotations_accessed"):
        errors.append("formal-test annotations were accessed")
    for name, artifact in manifest.get("artifacts", {}).items():
        path = Path(str(artifact["path"]))
        if not path.is_file():
            errors.append(f"missing artifact: {name}")
        elif sha256_file(path) != artifact["sha256"]:
            errors.append(f"hash mismatch: {name}")
    arrays = np.load(output_root / "candidate_features.npz")
    candidates = _read_jsonl(output_root / "candidate_records.jsonl")
    if len(arrays["features"]) != len(candidates) or len(arrays["gold"]) != len(candidates):
        errors.append("candidate feature alignment failure")
    if any(int(label) != int(bool(row["gold"])) for label, row in zip(arrays["gold"], candidates, strict=True)):
        errors.append("candidate label alignment failure")
    prejoin = json.loads((output_root / "rankings.prejoin.manifest.json").read_text(encoding="utf-8"))
    if prejoin.get("gold_labels_used"):
        errors.append("prejoin ranking manifest reports gold use")
    audit = {
        "profile": "q1_phase2_retrieval_independent_audit",
        "status": "pass" if not errors else "failed",
        "errors": errors,
        "checked_candidate_rows": len(candidates),
        "formal_test_annotations_accessed": False,
    }
    audit_path = output_root / "retrieval.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError(f"Phase-2 retrieval audit failed: {errors}")
    return audit

