from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 10) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 10) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    gain = sum(
        1.0 / math.log2(rank + 2)
        for rank, item_id in enumerate(ranked_ids[:k])
        if item_id in relevant
    )
    ideal_count = min(k, len(relevant))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return gain / ideal if ideal else 0.0

