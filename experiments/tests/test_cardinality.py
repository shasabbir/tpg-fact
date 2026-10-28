import torch

from trace_scv.models import CitationCardinalityHead, selector_profile


def test_selector_profile_masks_padding_and_tracks_available_modalities() -> None:
    profile = selector_profile(
        torch.tensor([[2.0, -1.0, 9.0, 9.0]]),
        torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        torch.tensor([[0, 1, 0, 0]], dtype=torch.bool),
    )
    assert profile.shape == (1, 6)
    assert profile[0, 0] > profile[0, 1] > profile[0, 2]
    assert profile[0, -2:].tolist() == [0.5, 0.25]


def test_cardinality_head_returns_one_logit_per_allowed_count() -> None:
    head = CitationCardinalityHead(12, 7, hidden_dim=8)
    assert head(torch.randn(3, 14)).shape == (3, 7)


def test_selector_profile_can_ablate_availability_counts() -> None:
    logits = torch.tensor([[2.0, -1.0, 9.0, 9.0]])
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    visual = torch.tensor([[0, 1, 0, 0]], dtype=torch.bool)
    no_valid = selector_profile(
        logits, mask, visual, include_valid_count=False
    )
    no_visual = selector_profile(
        logits, mask, visual, include_visual_count=False
    )
    assert no_valid.shape == no_visual.shape == (1, 5)
    assert no_valid[0, -1].item() == 0.25
    assert no_visual[0, -1].item() == 0.5
