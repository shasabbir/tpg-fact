from __future__ import annotations

from collections import Counter

from data.schema import ClaimRecord, EvidenceRecord, PredictionRecord


class SmokeHierarchicalClassifier:
    """Deterministic integration baseline for synthetic records only.

    This class exists to test I/O and abstention flow. It is deliberately not a
    scientific model and must never be reported as a benchmark result.
    """

    def __init__(self, minimum_clusters: int = 1) -> None:
        self.minimum_clusters = minimum_clusters

    def predict(
        self,
        claim: ClaimRecord,
        evidence: list[EvidenceRecord],
    ) -> PredictionRecord:
        cluster_ids = sorted({item.source_cluster_id for item in evidence if item.source_cluster_id})
        sufficient_score = min(1.0, len(cluster_ids) / max(1, self.minimum_clusters))
        joined = " ".join(item.title.casefold() for item in evidence)

        if len(cluster_ids) < self.minimum_clusters or "no evidence" in joined:
            probabilities = {
                "supported": 0.10,
                "refuted": 0.10,
                "insufficient_evidence": 0.80,
            }
        elif any(token in joined for token in ("refutes", "does not", "false")):
            probabilities = {
                "supported": 0.10,
                "refuted": 0.80,
                "insufficient_evidence": 0.10,
            }
        elif any(token in joined for token in ("supports", "confirms", "official record")):
            probabilities = {
                "supported": 0.80,
                "refuted": 0.10,
                "insufficient_evidence": 0.10,
            }
        else:
            probabilities = {
                "supported": 0.15,
                "refuted": 0.15,
                "insufficient_evidence": 0.70,
            }

        predicted = max(probabilities, key=probabilities.get)
        return PredictionRecord(
            claim_id=claim.claim_id,
            predicted_label=predicted,
            class_probabilities=probabilities,
            sufficient_evidence_score=sufficient_score,
            selected_evidence_ids=tuple(item.evidence_id for item in evidence),
            source_cluster_ids=tuple(cluster_ids),
            decision_cutoff=claim.decision_cutoff,
        )

