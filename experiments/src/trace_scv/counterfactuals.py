from __future__ import annotations

import torch


def delete_selected_evidence(
    batch: dict[str, torch.Tensor], selected: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Mask the exact hard citation set used by the clean prediction."""
    changed = {key: value.clone() for key, value in batch.items()}
    selected_mask = selected.detach().bool() & batch["mask"].bool()
    changed["mask"] = batch["mask"].bool() & ~selected_mask
    return changed, selected_mask


def modality_matched_visual_swap(
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Replace only gold pixel features with same-type pixels from another row.

    Rows without an available same-modality source are left unchanged and marked
    false in the returned row mask. Semantic evidence, masks, and node types never
    change, so the intervention isolates the pixel channel.
    """
    changed = {key: value.clone() for key, value in batch.items()}
    mask = batch["mask"].bool()
    gold = batch["gold"].bool()
    node_types = batch["types"]
    visual_mask = batch["visual_mask"].bool()
    changed_rows = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)

    for row in range(mask.shape[0]):
        targets = torch.where(gold[row] & mask[row] & visual_mask[row])[0]
        for target_tensor in targets:
            target = int(target_tensor)
            target_type = int(node_types[row, target])
            source: tuple[int, int] | None = None
            for offset in range(1, mask.shape[0]):
                source_row = (row + offset) % mask.shape[0]
                candidates = torch.where(
                    mask[source_row]
                    & visual_mask[source_row]
                    & (node_types[source_row] == target_type)
                )[0]
                if len(candidates):
                    source = (source_row, int(candidates[0]))
                    break
            if source is None:
                continue
            source_row, source_position = source
            changed["visual"][row, target] = batch["visual"][
                source_row, source_position
            ]
            changed_rows[row] = True
    return changed, changed_rows
