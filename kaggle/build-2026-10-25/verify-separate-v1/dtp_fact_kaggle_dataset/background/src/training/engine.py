from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Iterable

import numpy as np

from data.schema import LABELS
from evaluation.metrics import classification_metrics
from models.temporal_provenance import (
    HierarchicalFeatureModel,
    TemporalProvenanceVerdictModel,
)
from training.calibration import apply_temperature, fit_temperature, select_threshold
from training.dataset import ModelExample, NODE_TYPES, RELATIONS

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
except ImportError as error:  # pragma: no cover - optional heavyweight dependency
    raise ImportError("Install the 'ml' extra to use the training engine") from error


GRAPH_VARIANTS = {
    "full",
    "no_temporal_filter",
    "no_provenance",
    "no_temporal_features",
    "no_text_evidence",
    "no_visual_evidence",
    "no_sufficiency_gate",
    "no_evidence_dropout",
}
FEATURE_VARIANTS = {
    "text_only",
    "image_text",
    "vanilla_rag",
    "time_filtered_rag",
    "provenance_pool",
    "no_graph",
}
SUPPORTED_VARIANTS = GRAPH_VARIANTS | FEATURE_VARIANTS
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}


@dataclass(frozen=True)
class TrainingSettings:
    variant: str
    seed: int
    hidden_dimension: int = 256
    graph_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    sufficiency_loss_weight: float = 0.25
    early_stopping_patience: int = 5
    target_coverage: float = 0.80
    num_workers: int = 0
    precision: str = "fp32"
    evidence_dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported variant: {self.variant}")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _target_arrays(examples: list[ModelExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor(
        [LABEL_INDEX[example.claim.normalized_label] for example in examples],
        dtype=torch.long,
    )
    sufficiency = []
    mask = []
    for example in examples:
        target = example.claim.sufficiency_target
        sufficiency.append(1.0 if target == "sufficient" else 0.0)
        mask.append(target in {"sufficient", "insufficient"})
    return (
        labels,
        torch.tensor(sufficiency, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
    )


def collate_examples(examples: list[ModelExample], variant: str) -> dict[str, Any]:
    labels, sufficiency, sufficiency_mask = _target_arrays(examples)
    batch: dict[str, Any] = {
        "examples": examples,
        "labels": labels,
        "sufficiency_targets": sufficiency,
        "sufficiency_mask": sufficiency_mask,
    }
    if variant in GRAPH_VARIANTS:
        features: list[torch.Tensor] = []
        node_types: list[torch.Tensor] = []
        edge_indices: list[torch.Tensor] = []
        edge_types: list[torch.Tensor] = []
        claim_indices: list[int] = []
        offset = 0
        for example in examples:
            node_count = int(example.node_features.shape[0])
            features.append(torch.from_numpy(example.node_features))
            node_types.append(torch.from_numpy(example.node_types))
            edge_indices.append(torch.from_numpy(example.edge_index) + offset)
            edge_types.append(torch.from_numpy(example.edge_types))
            claim_indices.append(offset + example.claim_node_index)
            offset += node_count
        batch.update(
            {
                "node_features": torch.cat(features, dim=0),
                "node_types": torch.cat(node_types, dim=0),
                "edge_index": torch.cat(edge_indices, dim=1),
                "edge_types": torch.cat(edge_types, dim=0),
                "claim_node_indices": torch.tensor(claim_indices, dtype=torch.long),
            }
        )
    else:
        batch["features"] = torch.from_numpy(
            np.stack([example.baseline_features[variant] for example in examples]).astype(
                np.float32
            )
        )
    return batch


def hierarchical_log_probabilities(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if "direct_logits" in outputs:
        return F.log_softmax(outputs["direct_logits"], dim=-1)
    sufficient = outputs["sufficiency_logit"]
    veracity = F.log_softmax(outputs["veracity_logits"], dim=-1)
    log_sufficient = F.logsigmoid(sufficient)
    return torch.stack(
        (
            log_sufficient + veracity[:, 0],
            log_sufficient + veracity[:, 1],
            F.logsigmoid(-sufficient),
        ),
        dim=-1,
    )


def _forward(
    model: nn.Module,
    batch: dict[str, Any],
    variant: str,
    evidence_dropout: float = 0.0,
) -> dict[str, torch.Tensor]:
    if variant in GRAPH_VARIANTS:
        node_features = batch["node_features"]
        if model.training and evidence_dropout > 0:
            document_mask = batch["node_types"] == NODE_TYPES["document"]
            drop_mask = document_mask & (
                torch.rand(node_features.shape[0], device=node_features.device)
                < evidence_dropout
            )
            if torch.any(drop_mask):
                node_features = node_features.clone()
                node_features[drop_mask] = 0.0
        return model(
            node_features,
            batch["edge_index"],
            batch["edge_types"],
            batch["claim_node_indices"],
            batch["node_types"],
        )
    return model(batch["features"])


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _class_weights(examples: Iterable[ModelExample], device: torch.device) -> torch.Tensor:
    counts = Counter(example.claim.normalized_label for example in examples)
    if set(counts) != set(LABELS):
        raise ValueError(f"Training split must contain every class; observed {dict(counts)}")
    weights = np.asarray([1.0 / counts[label] for label in LABELS], dtype=np.float32)
    weights /= weights.mean()
    return torch.tensor(weights, device=device)


def _batch_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    class_weights: torch.Tensor,
    sufficiency_loss_weight: float,
) -> torch.Tensor:
    log_probabilities = hierarchical_log_probabilities(outputs)
    loss = F.nll_loss(log_probabilities, batch["labels"], weight=class_weights)
    mask = batch["sufficiency_mask"]
    if "sufficiency_logit" in outputs and torch.any(mask):
        auxiliary = F.binary_cross_entropy_with_logits(
            outputs["sufficiency_logit"][mask], batch["sufficiency_targets"][mask]
        )
        loss = loss + sufficiency_loss_weight * auxiliary
    return loss


def _autocast(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _predict_logits(
    model: nn.Module,
    loader: DataLoader,
    variant: str,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, np.ndarray, list[ModelExample]]:
    model.eval()
    logits: list[np.ndarray] = []
    sufficiency: list[np.ndarray] = []
    ordered_examples: list[ModelExample] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            with _autocast(device, precision):
                outputs = _forward(model, batch, variant)
                log_probabilities = hierarchical_log_probabilities(outputs)
            logits.append(log_probabilities.detach().float().cpu().numpy())
            if "sufficiency_logit" in outputs:
                scores = torch.sigmoid(outputs["sufficiency_logit"])
            else:
                scores = 1.0 - torch.softmax(outputs["direct_logits"], dim=-1)[:, 2]
            sufficiency.append(scores.detach().float().cpu().numpy())
            ordered_examples.extend(raw_batch["examples"])
    return (
        np.concatenate(logits, axis=0),
        np.concatenate(sufficiency, axis=0),
        ordered_examples,
    )


def _loader(
    examples: list[ModelExample],
    settings: TrainingSettings,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(settings.seed)
    return DataLoader(
        examples,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.num_workers,
        collate_fn=lambda rows: collate_examples(rows, settings.variant),
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(examples: list[ModelExample], settings: TrainingSettings) -> nn.Module:
    if not examples:
        raise ValueError("At least one training example is required")
    if settings.variant in GRAPH_VARIANTS:
        return TemporalProvenanceVerdictModel(
            input_dim=int(examples[0].node_features.shape[1]),
            hidden_dim=settings.hidden_dimension,
            relation_count=len(RELATIONS),
            node_type_count=len(NODE_TYPES),
            graph_layers=settings.graph_layers,
            dropout=settings.dropout,
            hierarchical=settings.variant != "no_sufficiency_gate",
        )
    width = int(examples[0].baseline_features[settings.variant].shape[0])
    return HierarchicalFeatureModel(width, settings.hidden_dimension, settings.dropout)


def _state_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return metadata


def train_model(
    train_examples: list[ModelExample],
    validation_examples: list[ModelExample],
    test_examples: list[ModelExample] | None,
    settings: TrainingSettings,
    output_directory: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    seed_everything(settings.seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if settings.precision != "fp32" and device.type != "cuda":
        raise ValueError("fp16/bf16 training requires CUDA")
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    model = build_model(train_examples, settings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, settings.epochs)
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and settings.precision == "fp16"
    )
    weights = _class_weights(train_examples, device)
    train_loader = _loader(train_examples, settings, shuffle=True)
    validation_loader = _loader(validation_examples, settings, shuffle=False)
    test_loader = _loader(test_examples, settings, shuffle=False) if test_examples else None

    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    checkpoint = output_root / "best_model.pt"
    started = time.perf_counter()
    for epoch in range(1, settings.epochs + 1):
        model.train()
        losses: list[float] = []
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, settings.precision):
                outputs = _forward(
                    model,
                    batch,
                    settings.variant,
                    evidence_dropout=(
                        0.0
                        if settings.variant == "no_evidence_dropout"
                        else settings.evidence_dropout
                    ),
                )
                loss = _batch_loss(
                    outputs, batch, weights, settings.sufficiency_loss_weight
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation_logits, _, ordered = _predict_logits(
            model, validation_loader, settings.variant, device, settings.precision
        )
        predicted = [LABELS[index] for index in validation_logits.argmax(axis=1)]
        truth = [example.claim.normalized_label for example in ordered]
        macro_f1 = float(classification_metrics(truth, predicted)["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": macro_f1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if macro_f1 > best_macro_f1 + 1e-12:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= settings.early_stopping_patience:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation_logits, _, validation_order = _predict_logits(
        model, validation_loader, settings.variant, device, settings.precision
    )
    validation_targets = np.asarray(
        [LABEL_INDEX[example.claim.normalized_label] for example in validation_order],
        dtype=int,
    )
    temperature = fit_temperature(validation_logits, validation_targets)
    validation_probabilities = apply_temperature(validation_logits, temperature)
    abstention_threshold = select_threshold(
        validation_probabilities.max(axis=1), settings.target_coverage
    )
    if test_loader is None:
        prediction_logits, prediction_sufficiency, prediction_order = _predict_logits(
            model, validation_loader, settings.variant, device, settings.precision
        )
        prediction_name = "validation_predictions.jsonl"
        status = "completed_selection_run"
    else:
        prediction_logits, prediction_sufficiency, prediction_order = _predict_logits(
            model, test_loader, settings.variant, device, settings.precision
        )
        prediction_name = "predictions.jsonl"
        status = "completed_training_run"
    prediction_probabilities = apply_temperature(prediction_logits, temperature)
    predictions = prediction_rows(
        prediction_order,
        prediction_probabilities,
        prediction_sufficiency,
        abstention_threshold,
    )
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    result = {
        "status": status,
        "settings": asdict(settings),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "temperature": temperature,
        "abstention_threshold": abstention_threshold,
        "target_coverage": settings.target_coverage,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_memory,
        "runtime": runtime_metadata(device),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _state_sha256(checkpoint),
        "history": history,
        "counts": {
            "train": len(train_examples),
            "validation": len(validation_examples),
            "test": len(test_examples) if test_examples else 0,
        },
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_root / prediction_name).open("w", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def predict_from_checkpoint(
    examples: list[ModelExample],
    settings: TrainingSettings,
    checkpoint: str | Path,
    *,
    temperature: float,
    abstention_threshold: float,
    device_name: str = "auto",
) -> list[dict[str, Any]]:
    """Run a frozen checkpoint without refitting calibration or thresholds."""
    if not examples:
        return []
    seed_everything(settings.seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = build_model(examples, settings).to(device)
    model.load_state_dict(
        torch.load(Path(checkpoint), map_location=device, weights_only=True)
    )
    loader = _loader(examples, settings, shuffle=False)
    logits, sufficiency, ordered = _predict_logits(
        model, loader, settings.variant, device, settings.precision
    )
    probabilities = apply_temperature(logits, temperature)
    return prediction_rows(
        ordered, probabilities, sufficiency, abstention_threshold
    )


def prediction_rows(
    examples: list[ModelExample],
    probabilities: np.ndarray,
    sufficiency_scores: np.ndarray,
    abstention_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example, probability, sufficient_score in zip(
        examples, probabilities, sufficiency_scores, strict=True
    ):
        predicted_index = int(np.argmax(probability))
        confidence = float(np.max(probability))
        rows.append(
            {
                "claim_id": example.claim.claim_id,
                "predicted_label": LABELS[predicted_index],
                "class_probabilities": {
                    label: float(probability[index]) for index, label in enumerate(LABELS)
                },
                "sufficient_evidence_score": float(sufficient_score),
                "answer_confidence": confidence,
                "abstained": confidence < abstention_threshold,
                "abstention_threshold": float(abstention_threshold),
                "selected_evidence_ids": [item.evidence_id for item in example.evidence],
                "source_cluster_ids": sorted(
                    {
                        item.source_cluster_id
                        for item in example.evidence
                        if item.source_cluster_id
                    }
                ),
                "decision_cutoff": example.claim.decision_cutoff.isoformat().replace(
                    "+00:00", "Z"
                ),
            }
        )
    return rows
