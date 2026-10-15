from __future__ import annotations

from dataclasses import dataclass

from data.schema import ClaimRecord, EvidenceRecord


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    attributes: dict[str, object]


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class ProvenanceGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


def build_provenance_graph(
    claim: ClaimRecord,
    evidence: list[EvidenceRecord],
) -> ProvenanceGraph:
    nodes: list[GraphNode] = [
        GraphNode(
            node_id=claim.claim_id,
            node_type="claim",
            attributes={"text": claim.claim_text, "cutoff": claim.decision_cutoff.isoformat()},
        ),
        GraphNode(
            node_id=f"image:{claim.image_sha256 or claim.image_phash}",
            node_type="image",
            attributes={"phash": claim.image_phash, "url": claim.claim_image_url},
        ),
    ]
    edges: list[GraphEdge] = [
        GraphEdge(claim.claim_id, f"image:{claim.image_sha256 or claim.image_phash}", "has_image")
    ]
    publisher_nodes: set[str] = set()
    cluster_nodes: set[str] = set()

    for item in evidence:
        nodes.append(
            GraphNode(
                node_id=item.evidence_id,
                node_type="document",
                attributes={
                    "title": item.title,
                    "published_at": item.published_at.isoformat(),
                    "media_type": item.media_type,
                },
            )
        )
        edges.append(GraphEdge(claim.claim_id, item.evidence_id, "retrieved_evidence"))
        publisher_id = f"publisher:{item.domain}"
        if publisher_id not in publisher_nodes:
            publisher_nodes.add(publisher_id)
            nodes.append(GraphNode(publisher_id, "publisher", {"domain": item.domain}))
        edges.append(GraphEdge(publisher_id, item.evidence_id, "published"))

        cluster_id = item.source_cluster_id or f"cluster:unassigned:{item.evidence_id}"
        if cluster_id not in cluster_nodes:
            cluster_nodes.add(cluster_id)
            nodes.append(GraphNode(cluster_id, "source_cluster", {}))
        edges.append(GraphEdge(item.evidence_id, cluster_id, "belongs_to_cluster"))

        if item.image_phash and item.image_phash == claim.image_phash:
            edges.append(
                GraphEdge(item.evidence_id, f"image:{claim.image_sha256 or claim.image_phash}", "contains_image")
            )
        for linked_url in item.outgoing_links:
            linked = next((candidate for candidate in evidence if candidate.url == linked_url), None)
            if linked is not None:
                edges.append(GraphEdge(item.evidence_id, linked.evidence_id, "links_to"))

    ordered = sorted(evidence, key=lambda item: (item.published_at, item.evidence_id))
    for earlier, later in zip(ordered, ordered[1:]):
        edges.append(GraphEdge(earlier.evidence_id, later.evidence_id, "published_before"))

    return ProvenanceGraph(tuple(nodes), tuple(edges))

