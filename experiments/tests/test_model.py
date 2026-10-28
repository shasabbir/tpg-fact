import torch

from trace_scv.models import TraceSCVModel


def test_trace_model_forward_and_mask() -> None:
    model = TraceSCVModel(input_dim=16, hidden_dim=12, graph_layers=1, citation_budget=2)
    output = model(
        torch.randn(3, 16),
        torch.randn(3, 5, 16),
        torch.tensor([[1, 1, 1, 0, 0]] * 3, dtype=torch.bool),
        torch.tensor([[1, 3, 1, 0, 0]] * 3),
        torch.ones(3, 5, 5),
        torch.randn(3, 5, 16),
        torch.tensor([[0, 1, 0, 0, 0]] * 3, dtype=torch.bool),
    )
    assert output["logits"].shape == (3, 2)
    assert output["hard_selection"].sum(dim=1).tolist() == [2.0, 2.0, 2.0]
    assert output["hard_selection"][:, 3:].sum() == 0
    assert output["visual_fusion_gate"][:, 1].min() > 0
    assert output["visual_fusion_gate"][:, [0, 2, 3, 4]].sum() == 0


def test_all_pool_uses_every_valid_evidence_item() -> None:
    model = TraceSCVModel(
        input_dim=8, hidden_dim=8, graph_layers=0, citation_budget=1, variant="all_pool"
    )
    output = model(
        torch.randn(2, 8),
        torch.randn(2, 4, 8),
        torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
        torch.tensor([[1, 3, 0, 0], [1, 3, 4, 0]]),
        torch.ones(2, 4, 4),
    )
    assert output["hard_selection"].sum(dim=1).tolist() == [2.0, 3.0]


def test_adaptive_selector_predicts_a_bounded_per_claim_cardinality() -> None:
    model = TraceSCVModel(
        input_dim=8,
        hidden_dim=8,
        graph_layers=1,
        citation_budget=4,
        variant="adaptive",
        dropout=0.0,
    )
    with torch.no_grad():
        for parameter in model.cardinality.parameters():
            parameter.zero_()
        model.cardinality[-1].bias[1] = 10.0
    model.eval()
    output = model(
        torch.randn(2, 8),
        torch.randn(2, 4, 8),
        torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
        torch.tensor([[1, 0, 0, 0], [1, 3, 4, 0]]),
        torch.ones(2, 4, 4),
    )
    assert output["cardinality_logits"].shape == (2, 4)
    assert output["hard_selection"].sum(dim=1).tolist() == [1.0, 2.0]
    assert output["predicted_citation_count"].tolist() == [1, 2]
