from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from data.schema import ClaimRecord, EvidenceRecord
from data.temporal import is_time_valid
from provenance.graph import build_provenance_graph


NODE_TYPES = {
    "claim": 0,
    "image": 1,
    "document": 2,
    "publisher": 3,
    "source_cluster": 4,
}
BASE_RELATIONS = (
    "has_image",
    "retrieved_evidence",
    "published",
    "belongs_to_cluster",
    "contains_image",
    "links_to",
    "published_before",
)
RELATIONS = {
    name: index
    for index, name in enumerate(
        (*BASE_RELATIONS, *(f"reverse:{name}" for name in BASE_RELATIONS))
    )
}


@dataclass(frozen=True)
class EmbeddingStore:
    text: Mapping[str, np.ndarray]
    image: Mapping[str, np.ndarray]
    text_dim: int
    image_dim: int

    @classmethod
    def from_npz(
        cls,
        text_path: str | Path,
        image_path: str | Path | None = None,
    ) -> "EmbeddingStore":
        text = _load_npz(text_path)
        image = _load_npz(image_path) if image_path else {}
        text_dim = _embedding_dim(text, "text")
        image_dim = _embedding_dim(image, "image") if image else 0
        return cls(text=text, image=image, text_dim=text_dim, image_dim=image_dim)

    @property
    def common_dim(self) -> int:
        return self.text_dim + self.image_dim + 4


@dataclass(frozen=True)
class ModelExample:
    claim: ClaimRecord
    evidence: tuple[EvidenceRecord, ...]
    node_features: np.ndarray
    node_types: np.ndarray
    edge_index: np.ndarray
    edge_types: np.ndarray
    claim_node_index: int
    baseline_features: Mapping[str, np.ndarray]


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        identifiers = [str(value) for value in archive["ids"].tolist()]
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(identifiers) != embeddings.shape[0]:
        raise ValueError(f"Malformed embedding archive: {source}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate identifiers in embedding archive: {source}")
    return {identifier: embeddings[index] for index, identifier in enumerate(identifiers)}


def _embedding_dim(values: Mapping[str, np.ndarray], name: str) -> int:
    dimensions = {int(np.asarray(value).shape[0]) for value in values.values()}
    if not dimensions:
        raise ValueError(f"{name} embedding archive is empty")
    if len(dimensions) != 1:
        raise ValueError(f"{name} embeddings must share one dimension")
    return next(iter(dimensions))


def _zero(size: int) -> np.ndarray:
    return np.zeros(size, dtype=np.float32)


def _lookup(
    values: Mapping[str, np.ndarray],
    identifier: str,
    size: int,
    *,
    required: bool = False,
) -> np.ndarray:
    value = values.get(identifier)
    if value is None:
        if required:
            raise ValueError(f"Missing required embedding for {identifier}")
        return _zero(size)
    return np.asarray(value, dtype=np.float32)


def _timestamp_code(value: str) -> float:
    return {"uncertain": 0.0, "observed": 0.5, "confirmed": 1.0}.get(value, 0.0)


def _age_feature(published_at: datetime, cutoff: datetime) -> float:
    days = max(0.0, (cutoff - published_at).total_seconds() / 86400.0)
    return float(min(days / (365.25 * 5.0), 1.0))


def _cluster_weights(evidence: Iterable[EvidenceRecord]) -> dict[str, float]:
    rows = list(evidence)
    sizes: dict[str, int] = {}
    for row in rows:
        cluster = row.source_cluster_id or f"unassigned:{row.evidence_id}"
        sizes[cluster] = sizes.get(cluster, 0) + 1
    return {
        row.evidence_id: 1.0 / sizes[row.source_cluster_id or f"unassigned:{row.evidence_id}"]
        for row in rows
    }


def _document_feature(
    item: EvidenceRecord,
    claim: ClaimRecord,
    store: EmbeddingStore,
    cluster_weight: float,
    *,
    use_evidence_text: bool = True,
    use_visual_evidence: bool = True,
    use_temporal_features: bool = True,
) -> np.ndarray:
    text = (
        _lookup(store.text, item.evidence_id, store.text_dim, required=True)
        if use_evidence_text
        else _zero(store.text_dim)
    )
    image = (
        _lookup(store.image, item.evidence_id, store.image_dim)
        if use_visual_evidence
        else _zero(store.image_dim)
    )
    scalars = np.asarray(
        [
            _age_feature(item.published_at, claim.decision_cutoff)
            if use_temporal_features
            else 0.0,
            cluster_weight,
            float(is_time_valid(item, claim.decision_cutoff))
            if use_temporal_features
            else 0.0,
            _timestamp_code(item.timestamp_confidence) if use_temporal_features else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate((text, image, scalars))


def build_model_example(
    claim: ClaimRecord,
    evidence: Iterable[EvidenceRecord],
    store: EmbeddingStore,
    *,
    enforce_time_filter: bool = True,
    use_provenance: bool = True,
    use_evidence_text: bool = True,
    use_visual_evidence: bool = True,
    use_temporal_features: bool = True,
) -> ModelExample:
    selected = [
        item
        for item in evidence
        if item.evidence_id in claim.evidence_ids
        and (not enforce_time_filter or is_time_valid(item, claim.decision_cutoff))
    ]
    selected.sort(key=lambda item: (item.published_at, item.evidence_id))
    if not use_provenance:
        selected = [
            replace(
                item,
                source_cluster_id=f"unassigned:{item.evidence_id}",
                dependency_signals=(),
            )
            for item in selected
        ]
    weights = _cluster_weights(selected) if use_provenance else {
        item.evidence_id: 1.0 for item in selected
    }

    claim_text = _lookup(store.text, claim.claim_id, store.text_dim, required=True)
    claim_image = (
        _lookup(store.image, claim.claim_id, store.image_dim)
        if use_visual_evidence
        else _zero(store.image_dim)
    )
    claim_common = np.concatenate((claim_text, claim_image, _zero(4)))
    document_features = [
        _document_feature(
            item,
            claim,
            store,
            weights[item.evidence_id],
            use_evidence_text=use_evidence_text,
            use_visual_evidence=use_visual_evidence,
            use_temporal_features=use_temporal_features,
        )
        for item in selected
    ]
    if document_features:
        documents = np.stack(document_features)
        uniform_pool = documents.mean(axis=0)
        normalized = np.asarray(
            [weights[item.evidence_id] for item in selected], dtype=np.float32
        )
        normalized /= normalized.sum().clip(min=1e-12)
        provenance_pool = (documents * normalized[:, None]).sum(axis=0)
    else:
        uniform_pool = _zero(store.common_dim)
        provenance_pool = _zero(store.common_dim)

    graph = build_provenance_graph(claim, selected)
    evidence_by_id = {item.evidence_id: item for item in selected}
    graph_features: list[np.ndarray] = []
    graph_types: list[int] = []
    node_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
    for node in graph.nodes:
        graph_types.append(NODE_TYPES[node.node_type])
        if node.node_type == "claim":
            feature = claim_common
        elif node.node_type == "image":
            feature = np.concatenate(
                (_zero(store.text_dim), claim_image, _zero(4))
            )
        elif node.node_type == "document":
            feature = _document_feature(
                evidence_by_id[node.node_id],
                claim,
                store,
                weights[node.node_id],
                use_evidence_text=use_evidence_text,
                use_visual_evidence=use_visual_evidence,
                use_temporal_features=use_temporal_features,
            )
        else:
            feature = _zero(store.common_dim)
        graph_features.append(feature)

    edge_pairs: list[tuple[int, int]] = []
    edge_types: list[int] = []
    for edge in graph.edges:
        if not use_temporal_features and edge.relation == "published_before":
            continue
        source, target = node_index[edge.source], node_index[edge.target]
        edge_pairs.extend(((source, target), (target, source)))
        edge_types.extend(
            (RELATIONS[edge.relation], RELATIONS[f"reverse:{edge.relation}"])
        )
    edge_index = (
        np.asarray(edge_pairs, dtype=np.int64).T
        if edge_pairs
        else np.empty((2, 0), dtype=np.int64)
    )
    return ModelExample(
        claim=claim,
        evidence=tuple(selected),
        node_features=np.stack(graph_features).astype(np.float32),
        node_types=np.asarray(graph_types, dtype=np.int64),
        edge_index=edge_index,
        edge_types=np.asarray(edge_types, dtype=np.int64),
        claim_node_index=node_index[claim.claim_id],
        baseline_features={
            "text_only": claim_text,
            "image_text": np.concatenate((claim_text, claim_image)),
            "vanilla_rag": np.concatenate((claim_text, claim_image, uniform_pool)),
            "time_filtered_rag": np.concatenate((claim_text, claim_image, uniform_pool)),
            "provenance_pool": np.concatenate(
                (claim_text, claim_image, provenance_pool)
            ),
            "no_graph": np.concatenate((claim_text, claim_image, provenance_pool)),
        },
    )


def build_examples(
    claims: Iterable[ClaimRecord],
    evidence: Iterable[EvidenceRecord],
    store: EmbeddingStore,
    *,
    enforce_time_filter: bool = True,
    use_provenance: bool = True,
    use_evidence_text: bool = True,
    use_visual_evidence: bool = True,
    use_temporal_features: bool = True,
) -> list[ModelExample]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    return [
        build_model_example(
            claim,
            [evidence_by_id[item_id] for item_id in claim.evidence_ids if item_id in evidence_by_id],
            store,
            enforce_time_filter=enforce_time_filter,
            use_provenance=use_provenance,
            use_evidence_text=use_evidence_text,
            use_visual_evidence=use_visual_evidence,
            use_temporal_features=use_temporal_features,
        )
        for claim in claims
    ]
