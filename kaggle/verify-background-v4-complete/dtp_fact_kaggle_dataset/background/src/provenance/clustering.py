from __future__ import annotations

from dataclasses import replace
from itertools import combinations

from data.schema import EvidenceRecord
from retrieval.hybrid import tokenize


class _DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(tokenize(left)), set(tokenize(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _hex_hamming(left: str, right: str) -> int:
    if not left or not right:
        return 10**9
    width = max(len(left), len(right))
    return (int(left, 16) ^ int(right, 16)).bit_count() + abs(len(left) - len(right)) * 4


def are_dependent(
    left: EvidenceRecord,
    right: EvidenceRecord,
    title_jaccard_threshold: float = 0.80,
    image_phash_hamming_threshold: int = 6,
) -> bool:
    if left.text_sha256 and left.text_sha256 == right.text_sha256:
        return True
    if _jaccard(left.title, right.title) >= title_jaccard_threshold:
        return True
    if _hex_hamming(left.image_phash, right.image_phash) <= image_phash_hamming_threshold:
        return True
    if left.url in right.outgoing_links or right.url in left.outgoing_links:
        return True
    return False


def dependency_reasons(
    left: EvidenceRecord,
    right: EvidenceRecord,
    title_jaccard_threshold: float = 0.80,
    image_phash_hamming_threshold: int = 6,
) -> tuple[str, ...]:
    """Return observable dependence signals without assigning credibility."""
    reasons: list[str] = []
    if left.text_sha256 and left.text_sha256 == right.text_sha256:
        reasons.append("exact_text_hash")
    if _jaccard(left.title, right.title) >= title_jaccard_threshold:
        reasons.append("near_duplicate_title")
    if _hex_hamming(left.image_phash, right.image_phash) <= image_phash_hamming_threshold:
        reasons.append("shared_or_near_duplicate_image")
    if left.url in right.outgoing_links or right.url in left.outgoing_links:
        reasons.append("direct_hyperlink")
    return tuple(reasons)


def assign_source_clusters(
    records: list[EvidenceRecord],
    title_jaccard_threshold: float = 0.80,
    image_phash_hamming_threshold: int = 6,
) -> list[EvidenceRecord]:
    sets = _DisjointSet([record.evidence_id for record in records])
    observed_signals: dict[str, set[str]] = {
        record.evidence_id: set(record.dependency_signals) for record in records
    }
    for left, right in combinations(records, 2):
        reasons = dependency_reasons(
            left,
            right,
            title_jaccard_threshold=title_jaccard_threshold,
            image_phash_hamming_threshold=image_phash_hamming_threshold,
        )
        if reasons:
            sets.union(left.evidence_id, right.evidence_id)
            observed_signals[left.evidence_id].update(reasons)
            observed_signals[right.evidence_id].update(reasons)

    roots = sorted({sets.find(record.evidence_id) for record in records})
    cluster_ids = {root: f"cluster-{index:04d}" for index, root in enumerate(roots, start=1)}
    return [
        replace(
            record,
            source_cluster_id=cluster_ids[sets.find(record.evidence_id)],
            dependency_signals=tuple(sorted(observed_signals[record.evidence_id])),
        )
        for record in records
    ]


def cluster_normalized_weights(records: list[EvidenceRecord]) -> dict[str, float]:
    """Give each source cluster one unit of mass, split across its documents."""
    sizes: dict[str, int] = {}
    for record in records:
        cluster_id = record.source_cluster_id or f"unassigned:{record.evidence_id}"
        sizes[cluster_id] = sizes.get(cluster_id, 0) + 1
    return {
        record.evidence_id: 1.0
        / sizes[record.source_cluster_id or f"unassigned:{record.evidence_id}"]
        for record in records
    }

