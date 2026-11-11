from __future__ import annotations

import gc
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np


TOKEN_PATTERN = re.compile(r"[\w'-]+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class ExternalRetrievalConfig:
    text_depth: int = 100
    image_depth: int = 20
    rerank_depth: int = 100
    rrf_constant: int = 60
    dense_batch_size: int = 16
    reranker_batch_size: int = 32
    reranker_max_tokens: int = 512
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class ExternalEncoder(Protocol):
    def encode_bge(self, values: Sequence[str]) -> np.ndarray: ...

    def encode_siglip(
        self, texts: Sequence[str], image_paths: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def identity(self) -> dict[str, object]: ...


def tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(value)]


def bm25_scores(
    query: str,
    documents: Sequence[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> np.ndarray:
    """Compute deterministic Lucene-style BM25 scores for one claim-local pool."""
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("invalid BM25 parameters")
    if not documents:
        return np.empty(0, dtype=np.float32)
    document_tokens = [tokens(value) for value in documents]
    average_length = max(1.0, float(np.mean([len(value) for value in document_tokens])))
    query_terms = list(dict.fromkeys(tokens(query)))
    document_frequency = {
        term: sum(term in set(value) for value in document_tokens) for term in query_terms
    }
    count = len(document_tokens)
    output = np.zeros(count, dtype=np.float64)
    for index, values in enumerate(document_tokens):
        frequencies: dict[str, int] = {}
        for value in values:
            if value in document_frequency:
                frequencies[value] = frequencies.get(value, 0) + 1
        length_normalizer = k1 * (1.0 - b + b * len(values) / average_length)
        for term, frequency in frequencies.items():
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (count - frequency_in_documents + 0.5) / (frequency_in_documents + 0.5)
            )
            output[index] += inverse_document_frequency * (
                frequency * (k1 + 1.0) / (frequency + length_normalizer)
            )
    return output.astype(np.float32)


def rank_indices(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    return np.asarray(sorted(range(len(values)), key=lambda index: (-values[index], index)))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, constant: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, raw_index in enumerate(ranking, start=1):
            index = int(raw_index)
            if index in seen:
                continue
            seen.add(index)
            scores[index] = scores.get(index, 0.0) + 1.0 / (constant + rank)
    ordered = sorted(scores, key=lambda index: (-scores[index], index))
    return (
        np.asarray(ordered, dtype=np.int64),
        np.asarray([scores[index] for index in ordered], dtype=np.float32),
    )


def image_filename_text(path: str | Path) -> str:
    stem = Path(path).stem
    value = re.sub(r"[_\-.]+", " ", stem)
    value = re.sub(r"\b[0-9a-f]{12,}\b", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip() or "image"


def _rank_map(order: Sequence[int], scores: Sequence[float]) -> tuple[dict[int, int], dict[int, float]]:
    return (
        {int(index): rank for rank, index in enumerate(order, start=1)},
        {int(index): float(score) for index, score in zip(order, scores, strict=True)},
    )


def _cross_encoder_scores(
    pairs: Sequence[tuple[str, str]], *, config: ExternalRetrievalConfig
) -> np.ndarray:
    if not pairs:
        return np.empty(0, dtype=np.float32)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("the pinned cross-encoder retrieval path requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(
        config.reranker_model, revision=config.reranker_revision
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.reranker_model,
        revision=config.reranker_revision,
        torch_dtype=torch.float16,
    ).to("cuda").eval()
    output: list[np.ndarray] = []
    for start in range(0, len(pairs), config.reranker_batch_size):
        batch_pairs = pairs[start : start + config.reranker_batch_size]
        batch = tokenizer(
            [left for left, _ in batch_pairs],
            [right for _, right in batch_pairs],
            padding=True,
            truncation=True,
            max_length=config.reranker_max_tokens,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**batch).logits.reshape(-1)
        output.append(logits.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(output).astype(np.float32)


def _safe_log_length(row: Mapping[str, object]) -> float:
    return math.log1p(int(row.get("content_word_count", 0))) / 10.0


def rank_external_candidates(
    claim_inputs: Sequence[Mapping[str, object]],
    candidates_by_claim: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    encoder: ExternalEncoder,
    config: ExternalRetrievalConfig,
    include_cross_encoder: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Rank label-blind, claim-local text and image candidates.

    `claim_inputs` must contain only split, claim_id, claim_text, and optional
    transcription fields. Gold questions, answers, verdicts, and justifications
    are deliberately outside this interface.
    """
    if config.text_depth < 1 or config.image_depth < 0 or config.rerank_depth < 1:
        raise ValueError("retrieval depths are invalid")
    allowed_claim_fields = {"split", "claim_id", "claim_text", "transcription"}
    for claim in claim_inputs:
        forbidden = set(claim) - allowed_claim_fields
        if forbidden:
            raise ValueError(f"outcome-bearing or unsupported claim fields: {sorted(forbidden)}")

    claim_by_key = {
        f"{claim['split']}:{claim['claim_id']}": dict(claim) for claim in claim_inputs
    }
    text_values: list[str] = []
    image_paths: list[str] = []
    for key in sorted(claim_by_key):
        for row in candidates_by_claim.get(key, ()):
            modality = str(row["modality"])
            if modality == "text":
                text_values.append(str(row["content"]))
            elif modality == "image":
                image_paths.append(str(row["media_path"]))
                text_values.append(image_filename_text(str(row["media_path"])))
            else:
                raise ValueError(f"unsupported external candidate modality: {modality}")

    claim_texts = {
        key: " ".join(
            value
            for value in (
                str(claim_by_key[key]["claim_text"]).strip(),
                str(claim_by_key[key].get("transcription", "")).strip(),
            )
            if value
        )
        for key in claim_by_key
    }
    bge_values = list(dict.fromkeys([*claim_texts.values(), *text_values]))
    if not bge_values:
        raise ValueError("external retrieval received no text to encode")
    bge_matrix = encoder.encode_bge(bge_values)
    bge = dict(zip(bge_values, bge_matrix, strict=True))

    text_state: dict[str, dict[str, object]] = {}
    cross_pairs: list[tuple[str, str]] = []
    cross_locations: list[tuple[str, int]] = []
    for key in sorted(claim_by_key):
        rows = [dict(row) for row in candidates_by_claim.get(key, ()) if row["modality"] == "text"]
        if not rows:
            continue
        query = claim_texts[key]
        documents = [str(row["content"]) for row in rows]
        lexical_scores = bm25_scores(query, documents)
        dense_scores = np.asarray([float(np.dot(bge[query], bge[value])) for value in documents])
        lexical_order = rank_indices(lexical_scores)
        dense_order = rank_indices(dense_scores)
        fused_order, fused_scores = reciprocal_rank_fusion(
            (lexical_order, dense_order), constant=config.rrf_constant
        )
        rerank_indices = fused_order[: min(config.rerank_depth, len(fused_order))]
        if include_cross_encoder:
            for index in rerank_indices:
                cross_pairs.append((query, documents[int(index)]))
                cross_locations.append((key, int(index)))
        text_state[key] = {
            "rows": rows,
            "lexical_scores": lexical_scores,
            "dense_scores": dense_scores,
            "lexical_order": lexical_order,
            "dense_order": dense_order,
            "fused_order": fused_order,
            "fused_scores": fused_scores,
            "rerank_indices": rerank_indices,
        }

    cross_by_key: dict[str, dict[int, float]] = {}
    if include_cross_encoder:
        values = _cross_encoder_scores(cross_pairs, config=config)
        for (key, index), value in zip(cross_locations, values, strict=True):
            cross_by_key.setdefault(key, {})[index] = float(value)

    ranked: list[dict[str, object]] = []
    for key in sorted(text_state):
        state = text_state[key]
        rows = state["rows"]
        fused_order = state["fused_order"]
        fused_scores = state["fused_scores"]
        if include_cross_encoder:
            reranked_head = sorted(
                map(int, state["rerank_indices"]),
                key=lambda index: (-cross_by_key[key][index], index),
            )
            head = set(reranked_head)
            final_order = [*reranked_head, *(int(index) for index in fused_order if int(index) not in head)]
        else:
            final_order = list(map(int, fused_order))
        lexical_ranks = {int(index): rank for rank, index in enumerate(state["lexical_order"], start=1)}
        dense_ranks = {int(index): rank for rank, index in enumerate(state["dense_order"], start=1)}
        _, fused_score_map = _rank_map(fused_order, fused_scores)
        for final_rank, index in enumerate(final_order[: config.text_depth], start=1):
            row = dict(rows[index])
            lexical_score = float(state["lexical_scores"][index])
            dense_score = float(state["dense_scores"][index])
            cross_score = float(cross_by_key.get(key, {}).get(index, 0.0))
            row.update(
                {
                    "retrieval_rank": final_rank,
                    "bm25_rank": lexical_ranks[index],
                    "dense_rank": dense_ranks[index],
                    "bm25_score": lexical_score,
                    "dense_score": dense_score,
                    "rrf_score": fused_score_map[index],
                    "cross_encoder_score": cross_score,
                    "siglip_score": 0.0,
                    "filename_dense_score": 0.0,
                    "features": [
                        lexical_score,
                        dense_score,
                        fused_score_map[index],
                        cross_score,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        float(row.get("branch") == "image_related"),
                        _safe_log_length(row),
                        1.0 / final_rank,
                    ],
                }
            )
            ranked.append(row)

    image_rows_all = [
        dict(row)
        for key in sorted(claim_by_key)
        for row in candidates_by_claim.get(key, ())
        if row["modality"] == "image"
    ]
    if image_rows_all:
        unique_paths = list(dict.fromkeys(str(row["media_path"]) for row in image_rows_all))
        siglip_claim_values = list(dict.fromkeys(claim_texts.values()))
        siglip_claim_matrix, siglip_image_matrix = encoder.encode_siglip(
            siglip_claim_values, unique_paths
        )
        siglip_claims = dict(zip(siglip_claim_values, siglip_claim_matrix, strict=True))
        siglip_images = dict(zip(unique_paths, siglip_image_matrix, strict=True))
        for key in sorted(claim_by_key):
            rows = [
                dict(row)
                for row in candidates_by_claim.get(key, ())
                if row["modality"] == "image"
            ]
            if not rows:
                continue
            query = claim_texts[key]
            siglip_scores = np.asarray(
                [float(np.dot(siglip_claims[query], siglip_images[str(row["media_path"])])) for row in rows]
            )
            filename_scores = np.asarray(
                [float(np.dot(bge[query], bge[image_filename_text(str(row["media_path"]))])) for row in rows]
            )
            siglip_order = rank_indices(siglip_scores)
            filename_order = rank_indices(filename_scores)
            fused_order, fused_scores = reciprocal_rank_fusion(
                (siglip_order, filename_order), constant=config.rrf_constant
            )
            siglip_ranks = {int(index): rank for rank, index in enumerate(siglip_order, start=1)}
            filename_ranks = {int(index): rank for rank, index in enumerate(filename_order, start=1)}
            _, fused_score_map = _rank_map(fused_order, fused_scores)
            for final_rank, raw_index in enumerate(fused_order[: config.image_depth], start=1):
                index = int(raw_index)
                row = dict(rows[index])
                siglip_score = float(siglip_scores[index])
                filename_score = float(filename_scores[index])
                row.update(
                    {
                        "retrieval_rank": final_rank,
                        "siglip_rank": siglip_ranks[index],
                        "filename_dense_rank": filename_ranks[index],
                        "bm25_score": 0.0,
                        "dense_score": 0.0,
                        "rrf_score": fused_score_map[index],
                        "cross_encoder_score": 0.0,
                        "siglip_score": siglip_score,
                        "filename_dense_score": filename_score,
                        "features": [
                            0.0,
                            0.0,
                            fused_score_map[index],
                            0.0,
                            siglip_score,
                            filename_score,
                            0.0,
                            1.0,
                            1.0,
                            _safe_log_length(row),
                            1.0 / final_rank,
                        ],
                    }
                )
                ranked.append(row)

    ranked.sort(
        key=lambda row: (
            str(row["split"]),
            int(row.get("claim_ordinal", 0)),
            str(row["modality"]),
            int(row["retrieval_rank"]),
            str(row["evidence_id"]),
        )
    )
    manifest = {
        "profile": "averimatec_label_blind_multimodal_retrieval",
        "config": asdict(config),
        "encoder": encoder.identity(),
        "cross_encoder_enabled": include_cross_encoder,
        "claims": len(claim_by_key),
        "input_text_candidates": sum(
            row["modality"] == "text"
            for rows in candidates_by_claim.values()
            for row in rows
        ),
        "input_image_candidates": sum(
            row["modality"] == "image"
            for rows in candidates_by_claim.values()
            for row in rows
        ),
        "retained_text_candidates": sum(row["modality"] == "text" for row in ranked),
        "retained_image_candidates": sum(row["modality"] == "image" for row in ranked),
        "gold_questions_used_for_retrieval": False,
        "verdict_labels_accessed": False,
        "test_annotations_accessed": False,
        "scientific_results_allowed": encoder.identity().get("scientific_results_allowed", True),
    }
    return ranked, manifest
