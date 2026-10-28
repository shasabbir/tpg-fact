from __future__ import annotations

import math

import torch
from torch import nn


class GraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, values: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        query, key, value = self.query(values), self.key(values), self.value(values)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(values.shape[-1])
        allowed = adjacency.bool() & mask.unsqueeze(1) & mask.unsqueeze(2)
        scores = scores.masked_fill(~allowed, -1e4)
        weights = torch.softmax(scores, dim=-1)
        update = self.output(torch.matmul(weights, value))
        output = self.norm(values + self.dropout(update))
        return output * mask.unsqueeze(-1)


class TraceSCVModel(nn.Module):
    """Compact graph selector with a shared sparse evidence bottleneck."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        num_classes: int = 2,
        num_node_types: int = 6,
        graph_layers: int = 2,
        dropout: float = 0.15,
        citation_budget: int = 3,
        variant: str = "trace_scv",
    ) -> None:
        super().__init__()
        if variant not in {
            "claim_only",
            "all_pool",
            "relevance",
            "flat",
            "trace_scv",
            "adaptive",
        }:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.citation_budget = citation_budget
        self.claim_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.evidence_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.visual_fusion_gate = nn.Linear(hidden_dim * 3, 1)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.graph = nn.ModuleList(
            GraphLayer(hidden_dim, dropout)
            for _ in range(graph_layers if variant in {"trace_scv", "adaptive"} else 0)
        )
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.verdict = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        if variant == "adaptive":
            self.cardinality = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, citation_budget),
            )

    def forward(
        self,
        claim_features: torch.Tensor,
        evidence_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        node_types: torch.Tensor,
        adjacency: torch.Tensor,
        visual_features: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        claim = self.claim_projection(claim_features)
        if self.variant == "claim_only":
            zeros = torch.zeros_like(evidence_mask, dtype=claim.dtype)
            aggregate = torch.zeros_like(claim)
            return {
                "logits": self.verdict(torch.cat([claim, aggregate], dim=-1)),
                "selector_logits": zeros,
                "selection_probabilities": zeros,
                "hard_selection": zeros,
                "visual_fusion_gate": zeros,
                "predicted_citation_count": zeros.sum(dim=1).long(),
            }

        semantic = self.evidence_projection(evidence_features)
        if visual_features is None:
            visual_features = torch.zeros_like(evidence_features)
        if visual_mask is None:
            visual_mask = torch.zeros_like(evidence_mask)
        visual = self.visual_projection(visual_features)
        type_features = self.type_embedding(node_types)
        fusion_gate = torch.sigmoid(
            self.visual_fusion_gate(
                torch.cat([semantic, visual, semantic * visual], dim=-1)
            ).squeeze(-1)
        ) * visual_mask.to(semantic.dtype)
        evidence = self.fusion_norm(
            semantic + fusion_gate.unsqueeze(-1) * visual + type_features
        )
        evidence = evidence * evidence_mask.unsqueeze(-1)
        for layer in self.graph:
            evidence = layer(evidence, adjacency, evidence_mask)

        if self.variant == "all_pool":
            selected = evidence_mask.to(evidence.dtype)
            denominator = selected.sum(dim=1, keepdim=True).clamp_min(1.0)
            aggregate = (evidence * selected.unsqueeze(-1)).sum(dim=1) / denominator
            logits = self.verdict(torch.cat([claim, aggregate], dim=-1))
            return {
                "logits": logits,
                "selector_logits": selected,
                "selection_probabilities": selected,
                "hard_selection": selected,
                "visual_fusion_gate": fusion_gate,
                "predicted_citation_count": selected.sum(dim=1).long(),
            }

        expanded_claim = claim.unsqueeze(1).expand_as(evidence)
        selector_input = torch.cat(
            [expanded_claim, evidence, expanded_claim * evidence], dim=-1
        )
        if self.variant == "relevance":
            selector_logits = (expanded_claim * evidence).sum(dim=-1) / math.sqrt(evidence.shape[-1])
        else:
            selector_logits = self.selector(selector_input).squeeze(-1)
        selector_logits = selector_logits.masked_fill(~evidence_mask, -1e4)
        soft = torch.sigmoid(selector_logits) * evidence_mask

        budget = min(self.citation_budget, evidence.shape[1])
        top_indices = selector_logits.topk(budget, dim=1).indices
        if self.variant == "adaptive":
            valid_denominator = evidence_mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled_evidence = evidence.sum(dim=1) / valid_denominator
            cardinality_logits = self.cardinality(
                torch.cat([claim, pooled_evidence], dim=-1)
            )
            predicted_count = cardinality_logits.argmax(dim=1) + 1
            predicted_count = torch.minimum(predicted_count, evidence_mask.sum(dim=1))
            ranks = torch.arange(budget, device=evidence.device).unsqueeze(0)
            ranked_selection = (ranks < predicted_count.unsqueeze(1)).to(soft.dtype)
            hard = torch.zeros_like(soft).scatter(1, top_indices, ranked_selection)
            hard = hard * evidence_mask
        else:
            cardinality_logits = None
            hard = torch.zeros_like(soft).scatter(1, top_indices, 1.0) * evidence_mask
            predicted_count = hard.sum(dim=1).long()
        gate = hard + soft - soft.detach() if self.training else hard
        denominator = gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        aggregate = (evidence * gate.unsqueeze(-1)).sum(dim=1) / denominator
        logits = self.verdict(torch.cat([claim, aggregate], dim=-1))
        result = {
            "logits": logits,
            "selector_logits": selector_logits,
            "selection_probabilities": soft,
            "hard_selection": hard,
            "visual_fusion_gate": fusion_gate,
            "predicted_citation_count": predicted_count,
        }
        if cardinality_logits is not None:
            result["cardinality_logits"] = cardinality_logits
        return result
