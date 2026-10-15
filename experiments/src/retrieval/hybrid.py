from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def bm25_scores(
    query: str,
    documents: Mapping[str, str],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    if not documents:
        return {}
    tokenized = {doc_id: tokenize(text) for doc_id, text in documents.items()}
    avg_len = sum(len(tokens) for tokens in tokenized.values()) / len(tokenized)
    avg_len = max(avg_len, 1.0)
    query_terms = set(tokenize(query))
    document_frequency = {
        term: sum(term in tokens for tokens in tokenized.values()) for term in query_terms
    }
    scores: dict[str, float] = {}
    count = len(tokenized)
    for doc_id, tokens in tokenized.items():
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * len(tokens) / avg_len)
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        scores[doc_id] = score
    return scores


def bm25_rank(query: str, documents: Mapping[str, str]) -> list[str]:
    scores = bm25_scores(query, documents)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))

