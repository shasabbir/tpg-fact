import torch

from trace_scv.counterfactuals import (
    delete_selected_evidence,
    modality_matched_visual_swap,
)
from trace_scv.evaluation.study import _perturb_batch


def _batch() -> dict[str, torch.Tensor]:
    evidence = torch.zeros(3, 4, 2)
    evidence[0, 1] = torch.tensor([1.0, 1.0])  # target caption: must stay fixed
    evidence[1, 2] = torch.tensor([2.0, 2.0])  # source caption: must not be copied
    visual = torch.zeros(3, 4, 2)
    visual[0, 1] = torch.tensor([10.0, 10.0])  # target figure pixels
    visual[1, 0] = torch.tensor([20.0, 20.0])  # text: must not be used
    visual[1, 1] = torch.tensor([50.0, 50.0])  # table: valid source pixels
    visual[1, 2] = torch.tensor([30.0, 30.0])  # figure: valid source pixels
    visual[2, 1] = torch.tensor([40.0, 40.0])  # table: wrong visual type
    return {
        "claim": torch.zeros(3, 2),
        "evidence": evidence,
        "visual": visual,
        "visual_mask": torch.tensor(
            [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0]], dtype=torch.bool
        ),
        "mask": torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool
        ),
        "gold": torch.tensor(
            [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.float32
        ),
        "types": torch.tensor(
            [[1, 4, 1, 0], [1, 3, 4, 0], [1, 3, 0, 0]], dtype=torch.long
        ),
        "adjacency": torch.ones(3, 4, 4),
        "label": torch.tensor([1, 0, 1]),
    }


def test_selected_deletion_uses_citations_not_gold_mask() -> None:
    batch = _batch()
    selected = torch.tensor(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32
    )
    changed, selected_mask = delete_selected_evidence(batch, selected)
    assert selected_mask.bool().equal(selected.bool())
    assert changed["mask"][0].tolist() == [True, True, False, False]
    # The gold figure stays because it was not cited; the cited non-gold text is removed.
    assert changed["mask"][0, 1]
    assert not changed["mask"][0, 2]


def test_visual_swap_uses_same_modality_from_another_row() -> None:
    batch = _batch()
    changed, changed_rows = modality_matched_visual_swap(batch)
    assert changed_rows.tolist() == [True, False, True]
    assert changed["visual"][0, 1].tolist() == [30.0, 30.0]
    assert changed["evidence"][0, 1].tolist() == [1.0, 1.0]
    assert changed["types"][0, 1].item() == 4
    # The table in row two uses the table in row one, not row one's text item.
    assert changed["visual"][2, 1].tolist() == [50.0, 50.0]
    assert changed["types"][2, 1].item() == 3


def test_visual_swap_leaves_row_unchanged_without_same_type_source() -> None:
    batch = _batch()
    batch["types"][1, 2] = 5  # remove the only other figure source
    before = batch["visual"].clone()
    changed, changed_rows = modality_matched_visual_swap(batch)
    assert not changed_rows[0]
    assert torch.equal(changed["visual"][0], before[0])


def test_channel_removals_do_not_drop_evidence_nodes() -> None:
    batch = _batch()
    no_visual = _perturb_batch(batch, "no_visual")
    assert torch.equal(no_visual["mask"], batch["mask"])
    assert torch.equal(no_visual["evidence"], batch["evidence"])
    assert no_visual["visual"].count_nonzero() == 0
    assert not no_visual["visual_mask"].any()

    no_text = _perturb_batch(batch, "no_text")
    assert torch.equal(no_text["mask"], batch["mask"])
    assert no_text["evidence"].count_nonzero() == 0
    assert torch.equal(no_text["visual"], batch["visual"])
