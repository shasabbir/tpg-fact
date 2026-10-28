from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from trace_scv.counterfactuals import (
    delete_selected_evidence,
    modality_matched_visual_swap,
)
from trace_scv.evaluation.metrics import citation_macro_prf, label_macro_f1
from trace_scv.data.io import sha256_file
from trace_scv.features import FeatureStore, ID_TO_LABEL, load_feature_store
from trace_scv.models import TraceSCVModel


class ArrayDataset(Dataset):
    def __init__(self, store: FeatureStore) -> None:
        self.store = store

    def __len__(self) -> int:
        return len(self.store.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "claim": torch.from_numpy(self.store.claim_features[index]),
            "evidence": torch.from_numpy(self.store.evidence_features[index]),
            "visual": torch.from_numpy(self.store.visual_features[index]),
            "visual_mask": torch.from_numpy(self.store.visual_mask[index]),
            "mask": torch.from_numpy(self.store.evidence_mask[index]),
            "gold": torch.from_numpy(self.store.gold_mask[index]),
            "types": torch.from_numpy(self.store.node_types[index]),
            "adjacency": torch.from_numpy(self.store.adjacency[index]),
            "label": torch.tensor(int(self.store.labels[index]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 13
    variant: str = "trace_scv"
    experiment_name: str | None = None
    hidden_dim: int = 192
    graph_layers: int = 2
    citation_budget: int = 3
    dropout: float = 0.15
    batch_size: int = 32
    epochs: int = 20
    patience: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    verdict_weight: float = 1.0
    selection_weight: float = 0.5
    cardinality_weight: float = 0.25
    balanced_selection: bool = False
    sparsity_weight: float = 0.02
    deletion_weight: float = 0.25
    irrelevant_weight: float = 0.15
    duplicate_weight: float = 0.15
    swap_weight: float = 0.15
    intervention_margin: float = 0.15
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_input_fingerprint(
    train_features: Path,
    validation_features: Path,
    config: TrainConfig,
) -> str:
    """Bind a checkpoint to the exact feature stores and training configuration."""
    files = {}
    for role, path in (("train", train_features), ("validation", validation_features)):
        sidecar = path.with_suffix(".json")
        files[role] = {
            "features_sha256": sha256_file(path),
            "metadata_sha256": sha256_file(sidecar),
        }
    payload = {"files": files, "train_config": asdict(config)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def checkpoint_matches(
    checkpoint_path: Path,
    config: TrainConfig,
    input_fingerprint: str,
) -> bool:
    """Return true only when a resumable checkpoint matches all effective inputs."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        checkpoint.get("train_config") == asdict(config)
        and checkpoint.get("input_fingerprint") == input_fingerprint
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _forward(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        batch["claim"],
        batch["evidence"],
        batch["mask"],
        batch["types"],
        batch["adjacency"],
        batch.get("visual"),
        batch.get("visual_mask"),
    )


def _counterfactual_losses(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    clean: dict[str, torch.Tensor],
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = batch["mask"]
    gold = batch["gold"].bool()
    labels = batch["label"]
    clean_probability = clean["logits"].softmax(-1)
    clean_gold = clean_probability.gather(1, labels[:, None]).squeeze(1)

    deleted, _ = delete_selected_evidence(batch, clean["hard_selection"])
    deleted_out = _forward(model, deleted)
    deleted_gold = deleted_out["logits"].softmax(-1).gather(1, labels[:, None]).squeeze(1)
    deletion = F.relu(margin - (clean_gold - deleted_gold)).mean()

    irrelevant = {key: value.clone() for key, value in batch.items()}
    irrelevant_features = irrelevant["evidence"]
    for row in range(mask.shape[0]):
        source_row = (row + 1) % mask.shape[0]
        source_candidates = torch.where(mask[source_row] & ~gold[source_row])[0]
        source_position = int(source_candidates[0]) if len(source_candidates) else 0
        free = torch.where(~mask[row])[0]
        candidates = torch.where(mask[row] & ~gold[row])[0]
        position = int(free[0]) if len(free) else int(candidates[0]) if len(candidates) else int(torch.where(mask[row])[0][-1])
        irrelevant_features[row, position] = batch["evidence"][source_row, source_position]
        irrelevant["visual"][row, position] = batch["visual"][source_row, source_position]
        irrelevant["visual_mask"][row, position] = batch["visual_mask"][
            source_row, source_position
        ]
        irrelevant["types"][row, position] = batch["types"][source_row, source_position]
        irrelevant["mask"][row, position] = True
        irrelevant["adjacency"][row, position, irrelevant["mask"][row]] = 1.0
        irrelevant["adjacency"][row, irrelevant["mask"][row], position] = 1.0
    irrelevant["evidence"] = irrelevant_features
    irrelevant_probability = _forward(model, irrelevant)["logits"].softmax(-1)
    irrelevant_loss = F.mse_loss(irrelevant_probability, clean_probability.detach())

    duplicated = {key: value.clone() for key, value in batch.items()}
    duplicate_features = duplicated["evidence"]
    for row in range(mask.shape[0]):
        gold_positions = torch.where(gold[row] & mask[row])[0]
        other_positions = torch.where(mask[row] & ~gold[row])[0]
        free = torch.where(~mask[row])[0]
        if len(gold_positions):
            target = int(free[0]) if len(free) else int(other_positions[0]) if len(other_positions) else int(gold_positions[-1])
            source = int(gold_positions[0])
            duplicate_features[row, target] = batch["evidence"][row, source]
            duplicated["visual"][row, target] = batch["visual"][row, source]
            duplicated["visual_mask"][row, target] = batch["visual_mask"][row, source]
            duplicated["types"][row, target] = batch["types"][row, source]
            duplicated["mask"][row, target] = True
            duplicated["adjacency"][row, target, duplicated["mask"][row]] = 1.0
            duplicated["adjacency"][row, duplicated["mask"][row], target] = 1.0
    duplicated["evidence"] = duplicate_features
    duplicate_probability = _forward(model, duplicated)["logits"].softmax(-1)
    duplicate_loss = F.mse_loss(duplicate_probability, clean_probability.detach())

    swapped, visual_rows = modality_matched_visual_swap(batch)
    swapped_probability = _forward(model, swapped)["logits"].softmax(-1)
    swapped_gold = swapped_probability.gather(1, labels[:, None]).squeeze(1)
    if visual_rows.any():
        swap_loss = F.relu(margin - (clean_gold[visual_rows] - swapped_gold[visual_rows])).mean()
    else:
        swap_loss = torch.zeros((), device=labels.device)
    return deletion, irrelevant_loss, duplicate_loss, swap_loss


def _epoch_metrics(
    model: nn.Module, store: FeatureStore, batch_size: int, device: torch.device
) -> dict[str, float]:
    loader = DataLoader(ArrayDataset(store), batch_size=batch_size, shuffle=False)
    predicted_labels: list[str] = []
    gold_labels: list[str] = []
    predicted_citations: list[list[str]] = []
    gold_citations: list[list[str]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            indices = batch.pop("index").tolist()
            batch = {key: value.to(device) for key, value in batch.items()}
            output = _forward(model, batch)
            predictions = output["logits"].argmax(-1).cpu().tolist()
            hard = output["hard_selection"].cpu().numpy()
            gold = batch["gold"].cpu().numpy()
            for local, index in enumerate(indices):
                predicted_labels.append(ID_TO_LABEL[int(predictions[local])])
                gold_labels.append(ID_TO_LABEL[int(store.labels[index])])
                ids = store.evidence_ids[index]
                predicted_citations.append(
                    [ids[j] for j in range(min(len(ids), hard.shape[1])) if hard[local, j] > 0]
                )
                gold_citations.append(
                    [ids[j] for j in range(min(len(ids), gold.shape[1])) if gold[local, j] > 0]
                )
    citation = citation_macro_prf(predicted_citations, gold_citations)
    return {
        "verdict_macro_f1": label_macro_f1(gold_labels, predicted_labels),
        "citation_macro_f1": citation["f1"],
        "selection_score": 0.5 * label_macro_f1(gold_labels, predicted_labels)
        + 0.5 * citation["f1"],
    }


def train_model(
    train_features: Path,
    validation_features: Path,
    output_dir: Path,
    config: TrainConfig,
    *,
    input_fingerprint: str | None = None,
) -> dict:
    set_seed(config.seed)
    device = _device(config.device)
    train_store = load_feature_store(train_features)
    validation_store = load_feature_store(validation_features)
    input_dim = int(train_store.claim_features.shape[-1])
    model = TraceSCVModel(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        graph_layers=config.graph_layers,
        citation_budget=config.citation_budget,
        dropout=config.dropout,
        variant=config.variant,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        ArrayDataset(train_store),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict] = []
    best_score = -1.0
    stale = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch.pop("index")
            batch = {key: value.to(device) for key, value in batch.items()}
            clean = _forward(model, batch)
            verdict_loss = F.cross_entropy(clean["logits"], batch["label"])
            if config.variant in {"claim_only", "all_pool"}:
                selection_loss = torch.zeros((), device=device)
                cardinality_loss = torch.zeros((), device=device)
                sparsity = torch.zeros((), device=device)
                deletion = irrelevant = duplicate = swap = torch.zeros((), device=device)
            else:
                if config.variant == "relevance":
                    selection_loss = torch.zeros((), device=device)
                else:
                    valid_logits = clean["selector_logits"][batch["mask"]]
                    valid_gold = batch["gold"][batch["mask"]]
                    if config.balanced_selection:
                        positives = valid_gold.sum().clamp_min(1.0)
                        negatives = (1.0 - valid_gold).sum().clamp_min(1.0)
                        positive_weight = (negatives / positives).detach()
                    else:
                        positive_weight = None
                    selection_loss = F.binary_cross_entropy_with_logits(
                        valid_logits,
                        valid_gold,
                        pos_weight=positive_weight,
                    )
                if config.variant == "adaptive":
                    cardinality_target = (
                        batch["gold"].sum(dim=1).long().clamp(1, config.citation_budget)
                        - 1
                    )
                    cardinality_loss = F.cross_entropy(
                        clean["cardinality_logits"], cardinality_target
                    )
                else:
                    cardinality_loss = torch.zeros((), device=device)
                sparsity = clean["selection_probabilities"][batch["mask"]].mean()
                deletion, irrelevant, duplicate, swap = _counterfactual_losses(
                    model, batch, clean, config.intervention_margin
                )
            loss = (
                config.verdict_weight * verdict_loss
                + config.selection_weight * selection_loss
                + config.cardinality_weight * cardinality_loss
                + config.sparsity_weight * sparsity
                + config.deletion_weight * deletion
                + config.irrelevant_weight * irrelevant
                + config.duplicate_weight * duplicate
                + config.swap_weight * swap
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = _epoch_metrics(model, validation_store, config.batch_size, device)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(record)
        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": {
                        "input_dim": input_dim,
                        "hidden_dim": config.hidden_dim,
                        "graph_layers": config.graph_layers,
                        "citation_budget": config.citation_budget,
                        "dropout": config.dropout,
                        "variant": config.variant,
                    },
                    "train_config": asdict(config),
                    "input_fingerprint": input_fingerprint,
                    "epoch": epoch,
                    "validation": metrics,
                },
                output_dir / "model.pt",
            )
        else:
            stale += 1
            if stale >= config.patience:
                break

    result = {
        "status": "trained",
        "variant": config.experiment_name or config.variant,
        "seed": config.seed,
        "best_selection_score": best_score,
        "epochs_completed": len(history),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "history": history,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "input_fingerprint": input_fingerprint,
    }
    (output_dir / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_trained_model(checkpoint_path: Path, device: str = "auto") -> tuple[TraceSCVModel, torch.device, dict]:
    target = _device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target, weights_only=False)
    model = TraceSCVModel(**checkpoint["model"]).to(target)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, target, checkpoint
