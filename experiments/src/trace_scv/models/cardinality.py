from __future__ import annotations

import torch
from torch import nn


def selector_profile(
    selector_logits: torch.Tensor,
    evidence_mask: torch.Tensor,
    visual_mask: torch.Tensor,
    *,
    include_valid_count: bool = True,
    include_visual_count: bool = True,
) -> torch.Tensor:
    """Create a fixed-size profile without changing the verifier or its ranking."""
    probabilities = torch.sigmoid(selector_logits).masked_fill(~evidence_mask, 0.0)
    sorted_probabilities = probabilities.sort(dim=1, descending=True).values
    scale = float(evidence_mask.shape[1])
    valid_count = evidence_mask.sum(dim=1, keepdim=True).to(probabilities.dtype) / scale
    visual_count = (
        (visual_mask & evidence_mask).sum(dim=1, keepdim=True).to(probabilities.dtype)
        / scale
    )
    parts = [sorted_probabilities]
    if include_valid_count:
        parts.append(valid_count)
    if include_visual_count:
        parts.append(visual_count)
    return torch.cat(parts, dim=1)


class CitationCardinalityHead(nn.Module):
    """Predict citation-set size from a frozen selector-score profile."""

    def __init__(
        self,
        maximum_candidates: int,
        maximum_citations: int,
        hidden_dim: int = 32,
        extra_features: int = 2,
    ) -> None:
        super().__init__()
        self.maximum_candidates = maximum_candidates
        self.maximum_citations = maximum_citations
        self.network = nn.Sequential(
            nn.Linear(maximum_candidates + extra_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, maximum_citations),
        )

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        return self.network(profile)
