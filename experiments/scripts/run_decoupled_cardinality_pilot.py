from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from trace_scv.data.io import sha256_file, write_jsonl
from trace_scv.evaluation.metrics import (
    citation_cardinality_metrics,
    citation_macro_prf,
)
from trace_scv.features import load_feature_store
from trace_scv.models import CitationCardinalityHead, selector_profile
from trace_scv.training import ArrayDataset, load_trained_model


@dataclass(frozen=True)
class HeadConfig:
    seed: int = 13
    hidden_dim: int = 32
    batch_size: int = 64
    epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_profiles(model, store, device: torch.device) -> dict:
    profiles = []
    rankings = []
    base_citations = []
    verdict_probabilities = []
    gold_counts = []
    loader = DataLoader(ArrayDataset(store), batch_size=64, shuffle=False)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            indices = batch.pop("index").tolist()
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(
                batch["claim"],
                batch["evidence"],
                batch["mask"],
                batch["types"],
                batch["adjacency"],
                batch["visual"],
                batch["visual_mask"],
            )
            profiles.append(
                selector_profile(
                    output["selector_logits"], batch["mask"], batch["visual_mask"]
                ).cpu()
            )
            verdict_probabilities.append(output["logits"].softmax(-1).cpu())
            scores = output["selector_logits"].cpu().numpy()
            hard = output["hard_selection"].cpu().numpy()
            gold_counts.extend(batch["gold"].sum(dim=1).long().cpu().tolist())
            for local, index in enumerate(indices):
                ids = store.evidence_ids[index]
                ranking = [ids[position] for position in np.argsort(-scores[local]) if position < len(ids)]
                rankings.append(ranking)
                base_citations.append(
                    [
                        evidence_id
                        for position, evidence_id in enumerate(ids)
                        if hard[local, position] > 0
                    ]
                )
    return {
        "profiles": torch.cat(profiles),
        "rankings": rankings,
        "base_citations": base_citations,
        "verdict_probabilities": torch.cat(verdict_probabilities),
        "gold_counts": torch.tensor(gold_counts, dtype=torch.long),
    }


def gold_citations(store) -> list[list[str]]:
    return [
        [
            evidence_id
            for position, evidence_id in enumerate(ids)
            if store.gold_mask[index, position] > 0
        ]
        for index, ids in enumerate(store.evidence_ids)
    ]


def select(rankings: list[list[str]], counts: torch.Tensor) -> list[list[str]]:
    return [
        ranking[: int(count)]
        for ranking, count in zip(rankings, counts.tolist(), strict=True)
    ]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    experiments = project_root / "experiments"
    feature_root = experiments / "data" / "features" / "pilot"
    train_path = feature_root / "train.npz"
    validation_path = feature_root / "validation.npz"
    base_checkpoint = (
        experiments / "outputs/pilot/trace_no_cf/seed-13/model.pt"
    )
    base_metrics_path = base_checkpoint.parent / "evaluation" / "metrics.json"
    if not all(
        path.is_file()
        for path in (train_path, validation_path, base_checkpoint, base_metrics_path)
    ):
        raise FileNotFoundError("Audited pilot features and fixed-budget checkpoint are required.")

    config = HeadConfig()
    set_seed(config.seed)
    device = torch.device("cpu")
    train_store = load_feature_store(train_path)
    validation_store = load_feature_store(validation_path)
    base_model, base_device, _ = load_trained_model(base_checkpoint, "cpu")
    train_data = extract_profiles(base_model, train_store, base_device)
    validation_data = extract_profiles(base_model, validation_store, base_device)
    maximum_citations = int(train_data["gold_counts"].max())
    train_targets = train_data["gold_counts"].clamp(1, maximum_citations) - 1
    validation_targets = validation_data["gold_counts"].clamp(1, maximum_citations) - 1
    validation_gold = gold_citations(validation_store)

    head = CitationCardinalityHead(
        maximum_candidates=train_store.evidence_mask.shape[1],
        maximum_citations=maximum_citations,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(train_data["profiles"], train_targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    output = experiments / "outputs" / "decoupled_cardinality_pilot"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "config": asdict(config),
        "selection_rule": "maximum validation citation macro-F1; final test untouched",
        "success_gate": "at least +0.01 citation macro-F1 with identical frozen verdicts",
        "input_fingerprint": hashlib.sha256(
            json.dumps(
                {
                    "base": sha256_file(base_checkpoint),
                    "train": sha256_file(train_path),
                    "validation": sha256_file(validation_path),
                    "config": asdict(config),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        head.train()
        losses = []
        for profiles, targets in loader:
            logits = head(profiles.to(device))
            loss = F.cross_entropy(logits, targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode():
            predicted_counts = head(validation_data["profiles"].to(device)).argmax(1).cpu() + 1
        predicted = select(validation_data["rankings"], predicted_counts)
        citation_f1 = citation_macro_prf(predicted, validation_gold)["f1"]
        cardinality = citation_cardinality_metrics(predicted, validation_gold)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "citation_macro_f1": citation_f1,
                "citation_count_mae": cardinality["mae"],
            }
        )
        if citation_f1 > best_f1 + 1e-12:
            best_f1 = citation_f1
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "head": {
                        "maximum_candidates": train_store.evidence_mask.shape[1],
                        "maximum_citations": maximum_citations,
                        "hidden_dim": config.hidden_dim,
                    },
                    "config": asdict(config),
                    "base_checkpoint_sha256": sha256_file(base_checkpoint),
                    "train_features_sha256": sha256_file(train_path),
                    "validation_features_sha256": sha256_file(validation_path),
                    "epoch": epoch,
                    "validation_citation_macro_f1": citation_f1,
                },
                output / "cardinality_head.pt",
            )
        else:
            stale += 1
            if stale >= config.patience:
                break

    checkpoint = torch.load(output / "cardinality_head.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    with torch.inference_mode():
        predicted_counts = head(validation_data["profiles"]).argmax(1) + 1
    adaptive_citations = select(validation_data["rankings"], predicted_counts)
    base_citations = validation_data["base_citations"]
    base_metric = citation_macro_prf(base_citations, validation_gold)
    adaptive_metric = citation_macro_prf(adaptive_citations, validation_gold)
    adaptive_cardinality = citation_cardinality_metrics(
        adaptive_citations, validation_gold
    )
    gate = {
        "citation_gain_at_least_one_point": (
            adaptive_metric["f1"] - base_metric["f1"] >= 0.01
        ),
        "verdict_predictions_identical_by_design": True,
        "official_test_untouched": True,
    }
    records = []
    for index, claim_id in enumerate(validation_store.claim_ids):
        records.append(
            {
                "claim_id": claim_id,
                "group_id": validation_store.group_ids[index],
                "gold_citations": validation_gold[index],
                "fixed_top3_citations": base_citations[index],
                "adaptive_citations": adaptive_citations[index],
                "ranked_citations": validation_data["rankings"][index],
                "predicted_citation_count": int(predicted_counts[index]),
                "gold_citation_count": len(validation_gold[index]),
                "verdict_probabilities_unchanged": validation_data[
                    "verdict_probabilities"
                ][index].tolist(),
            }
        )
    write_jsonl(output / "comparison_records.jsonl", records)
    result = {
        "status": "pass" if all(gate.values()) else "fail",
        "method": "frozen_verifier_plus_decoupled_adaptive_cardinality",
        "base_checkpoint": str(base_checkpoint.relative_to(project_root)),
        "base_citation_macro": base_metric,
        "adaptive_citation_macro": adaptive_metric,
        "citation_macro_f1_gain": adaptive_metric["f1"] - base_metric["f1"],
        "adaptive_citation_cardinality": adaptive_cardinality,
        "verdict_macro_f1": json.loads(
            base_metrics_path.read_text(encoding="utf-8")
        )["verdict_macro_f1"],
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "gate": gate,
        "scientific_results_validated": False,
        "interpretation": "Validation-only pilot; official test remains untouched.",
    }
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
