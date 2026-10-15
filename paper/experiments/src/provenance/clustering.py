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


def assign_source_clusters(
    records: list[EvidenceRecord],
    title_jaccard_threshold: float = 0.80,
    image_phash_hamming_threshold: int = 6,
) -> list[EvidenceRecord]:
    sets = _DisjointSet([record.evidence_id for record in records])
    by_id = {record.evidence_id: record for record in records}
    for left, right in combinations(records, 2):
        if are_dependent(
            left,
            right,
            title_jaccard_threshold=title_jaccard_threshold,
            image_phash_hamming_threshold=image_phash_hamming_threshold,
        ):
            sets.union(left.evidence_id, right.evidence_id)

    roots = sorted({sets.find(record.evidence_id) for record in records})
    cluster_ids = {root: f"cluster-{index:04d}" for index, root in enumerate(roots, start=1)}
    return [
        replace(record, source_cluster_id=cluster_ids[sets.find(record.evidence_id)])
        for record in records
    ]

