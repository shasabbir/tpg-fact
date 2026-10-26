from __future__ import annotations

"""Trainable graph components.

The module is importable only when PyTorch is installed. It provides a compact
relation-aware graph layer and the preregistered sufficiency/veracity heads;
feature extraction and batching are intentionally kept in the training code.
"""

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - optional heavyweight dependency
    raise ImportError("Install the 'ml' extra to use the trainable graph model") from error


class RelationAwareGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, relation_count: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(relation_count)
        )
        self.value = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(relation_count)
        )
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim**-0.5

    def forward(self, node_states, edge_index, edge_type):
        source, target = edge_index
        messages = torch.zeros_like(node_states)
        normalizer = torch.zeros(node_states.size(0), device=node_states.device)
        query = self.query(node_states[target])
        for relation_id in range(len(self.key)):
            mask = edge_type == relation_id
            if not torch.any(mask):
                continue
            relation_source = source[mask]
            relation_target = target[mask]
            key = self.key[relation_id](node_states[relation_source])
            value = self.value[relation_id](node_states[relation_source])
            score = torch.sigmoid((query[mask] * key).sum(dim=-1) * self.scale)
            messages.index_add_(0, relation_target, value * score.unsqueeze(-1))
            normalizer.index_add_(0, relation_target, score)
        messages = messages / normalizer.clamp_min(1e-6).unsqueeze(-1)
        return self.norm(node_states + self.dropout(self.output(messages)))


class TemporalProvenanceVerdictModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        relation_count: int = 8,
        node_type_count: int = 5,
        graph_layers: int = 2,
        dropout: float = 0.2,
        hierarchical: bool = True,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.node_type_embedding = nn.Embedding(node_type_count, hidden_dim)
        self.layers = nn.ModuleList(
            RelationAwareGraphLayer(hidden_dim, relation_count, dropout)
            for _ in range(graph_layers)
        )
        self.hierarchical = hierarchical
        if hierarchical:
            self.sufficiency_head = nn.Linear(hidden_dim, 1)
            self.veracity_head = nn.Linear(hidden_dim, 2)
        else:
            self.direct_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        node_features,
        edge_index,
        edge_type,
        claim_node_index,
        node_type=None,
    ):
        state = self.project(node_features)
        if node_type is not None:
            state = state + self.node_type_embedding(node_type)
        for layer in self.layers:
            state = layer(state, edge_index, edge_type)
        claim_state = state[claim_node_index]
        if self.hierarchical:
            return {
                "sufficiency_logit": self.sufficiency_head(claim_state).squeeze(-1),
                "veracity_logits": self.veracity_head(claim_state),
            }
        return {"direct_logits": self.direct_head(claim_state)}


class HierarchicalFeatureModel(nn.Module):
    """Matched non-graph encoder used by controlled feature baselines."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sufficiency_head = nn.Linear(hidden_dim, 1)
        self.veracity_head = nn.Linear(hidden_dim, 2)

    def forward(self, features):
        state = self.encoder(features)
        return {
            "sufficiency_logit": self.sufficiency_head(state).squeeze(-1),
            "veracity_logits": self.veracity_head(state),
        }

