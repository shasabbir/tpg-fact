from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from trace_scv.data.io import sha256_file, write_jsonl, write_predictions
from trace_scv.evaluation.metrics import (
    citation_cardinality_metrics,
    citation_macro_prf,
)
from trace_scv.evaluation.study import _group_bootstrap
from trace_scv.features import FeatureStore, load_feature_store
from trace_scv.models import CitationCardinalityHead, selector_profile
from trace_scv.schema import Prediction
from trace_scv.training import ArrayDataset, load_trained_model


@dataclass(frozen=True)
class CardinalityConfig:
    seed: int = 13
    experiment_name: str = "dacs_scv"
    hidden_dim: int = 32
    batch_size: int = 64
    epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    include_valid_count: bool = True
    include_visual_count: bool = True
    device: str = "auto"


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _extract(
    model,
    store: FeatureStore,
    device: torch.device,
    *,
    include_valid_count: bool = True,
    include_visual_count: bool = True,
) -> dict:
    profiles = []
    rankings: list[list[str]] = []
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
                    output["selector_logits"],
                    batch["mask"],
                    batch["visual_mask"],
                    include_valid_count=include_valid_count,
                    include_visual_count=include_visual_count,
                ).cpu()
            )
            scores = output["selector_logits"].cpu().numpy()
            for local, index in enumerate(indices):
                ids = store.evidence_ids[index]
                rankings.append(
                    [ids[position] for position in np.argsort(-scores[local]) if position < len(ids)]
                )
    return {"profiles": torch.cat(profiles), "rankings": rankings}


def _gold_citations(store: FeatureStore) -> list[list[str]]:
    return [
        [
            evidence_id
            for position, evidence_id in enumerate(ids)
            if store.gold_mask[index, position] > 0
        ]
        for index, ids in enumerate(store.evidence_ids)
    ]


def _select(rankings: list[list[str]], counts: torch.Tensor) -> list[list[str]]:
    return [
        ranking[: int(count)]
        for ranking, count in zip(rankings, counts.tolist(), strict=True)
    ]


def _render_predictions(
    store: FeatureStore,
    base_records: list[dict],
    records: list[dict],
) -> list[Prediction]:
    predictions = []
    for index, row in enumerate(records):
        citations = row["predicted_citations"]
        texts = [
            store.evidence_text[index][store.evidence_ids[index].index(item)]
            for item in citations
        ]
        clauses = [
            f"{' '.join(content.split())[:180]} [{evidence_id}]"
            for evidence_id, content in zip(citations[:2], texts[:2], strict=False)
        ]
        explanation = (
            f"The claim is predicted {row['predicted_label']}. " + " ".join(clauses)
        )
        predictions.append(
            Prediction(
                claim_id=row["claim_id"],
                predicted_label=row["predicted_label"],
                probabilities=base_records[index]["probabilities"]["clean"],
                cited_evidence_ids=tuple(citations),
                explanation=explanation,
            )
        )
    return predictions


def _write_decoupled_evaluation(
    *,
    variant: str,
    store: FeatureStore,
    base_metrics: dict,
    base_records: list[dict],
    predicted_citations: list[list[str]],
    output_dir: Path,
    bootstrap_samples: int,
    elapsed_seconds: float,
) -> dict:
    gold_citations = _gold_citations(store)
    records = []
    for base, citations in zip(base_records, predicted_citations, strict=True):
        row = dict(base)
        row["predicted_citations"] = citations
        row["predicted_citation_count"] = len(citations)
        records.append(row)
    result = dict(base_metrics)
    result.update(
        {
            "variant": variant,
            "citation_macro": citation_macro_prf(predicted_citations, gold_citations),
            "citation_cardinality": citation_cardinality_metrics(
                predicted_citations, gold_citations
            ),
            "mean_selected_evidence": float(
                np.mean([len(row) for row in predicted_citations])
            ),
            "citation_coverage": float(np.mean([bool(row) for row in predicted_citations])),
            "comprehensiveness": None,
            "sufficiency_macro_f1": None,
            "confidence_intervals": _group_bootstrap(
                store,
                [row["predicted_label"] for row in records],
                predicted_citations,
                bootstrap_samples,
                seed=int(result["seed"]),
            ),
            "inference_seconds": elapsed_seconds,
            "scientific_results_validated": False,
        }
    )
    result["examples_per_second"] = (
        len(records) / elapsed_seconds if elapsed_seconds else None
    )
    result["_comparison_records"] = records
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        output_dir / "predictions.jsonl",
        _render_predictions(store, base_records, records),
    )
    write_jsonl(output_dir / "comparison_records.jsonl", records)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {key: value for key, value in result.items() if not key.startswith("_")},
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def evaluate_fixed_cardinality(
    base_evaluation_dir: Path,
    feature_path: Path,
    output_dir: Path,
    *,
    citation_count: int,
    bootstrap_samples: int,
) -> dict:
    """Cut a frozen ranking at a global citation count without changing verdicts."""
    if citation_count < 1:
        raise ValueError("citation_count must be positive")
    started = time.perf_counter()
    store = load_feature_store(feature_path)
    base_metrics = json.loads(
        (base_evaluation_dir / "metrics.json").read_text(encoding="utf-8")
    )
    base_records = _read_jsonl(base_evaluation_dir / "comparison_records.jsonl")
    if [row["claim_id"] for row in base_records] != store.claim_ids:
        raise ValueError("base evaluation and fixed-cardinality inputs are unaligned")
    predicted_citations = [
        row["ranked_citations"][:citation_count] for row in base_records
    ]
    return _write_decoupled_evaluation(
        variant=f"fixed_top{citation_count}",
        store=store,
        base_metrics=base_metrics,
        base_records=base_records,
        predicted_citations=predicted_citations,
        output_dir=output_dir,
        bootstrap_samples=bootstrap_samples,
        elapsed_seconds=time.perf_counter() - started,
    )


def _fingerprint(
    base_checkpoint: Path,
    train_features: Path,
    validation_features: Path,
    config: CardinalityConfig,
) -> str:
    payload = {
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "train_features_sha256": sha256_file(train_features),
        "validation_features_sha256": sha256_file(validation_features),
        "config": asdict(config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def cardinality_checkpoint_matches(
    path: Path,
    base_checkpoint: Path,
    train_features: Path,
    validation_features: Path,
    config: CardinalityConfig,
) -> bool:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return checkpoint.get("input_fingerprint") == _fingerprint(
        base_checkpoint, train_features, validation_features, config
    )


def train_cardinality_head(
    base_checkpoint: Path,
    train_features: Path,
    validation_features: Path,
    output_dir: Path,
    config: CardinalityConfig,
) -> dict:
    _seed(config.seed)
    target = torch.device(
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.device == "auto"
        else config.device
    )
    train_store = load_feature_store(train_features)
    validation_store = load_feature_store(validation_features)
    base_model, base_device, _ = load_trained_model(base_checkpoint, str(target))
    train_data = _extract(
        base_model,
        train_store,
        base_device,
        include_valid_count=config.include_valid_count,
        include_visual_count=config.include_visual_count,
    )
    validation_data = _extract(
        base_model,
        validation_store,
        base_device,
        include_valid_count=config.include_valid_count,
        include_visual_count=config.include_visual_count,
    )
    train_gold = _gold_citations(train_store)
    validation_gold = _gold_citations(validation_store)
    maximum_citations = max(len(row) for row in train_gold)
    train_targets = torch.tensor(
        [min(len(row), maximum_citations) - 1 for row in train_gold], dtype=torch.long
    )
    head = CitationCardinalityHead(
        maximum_candidates=train_store.evidence_mask.shape[1],
        maximum_citations=maximum_citations,
        hidden_dim=config.hidden_dim,
        extra_features=int(config.include_valid_count) + int(config.include_visual_count),
    ).to(target)
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
    fingerprint = _fingerprint(
        base_checkpoint, train_features, validation_features, config
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "input_fingerprint": fingerprint,
        "config": asdict(config),
        "selection_rule": "maximum development-validation citation macro-F1",
        "official_test_used_for_selection": False,
    }
    (output_dir / "cardinality_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    history = []
    best_f1 = -1.0
    stale = 0
    checkpoint_path = output_dir / "cardinality_head.pt"
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        head.train()
        losses = []
        for profiles, labels in loader:
            loss = F.cross_entropy(
                head(profiles.to(target)), labels.to(target)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        with torch.inference_mode():
            counts = (
                head(validation_data["profiles"].to(target)).argmax(dim=1).cpu() + 1
            )
        citations = _select(validation_data["rankings"], counts)
        score = citation_macro_prf(citations, validation_gold)["f1"]
        cardinality = citation_cardinality_metrics(citations, validation_gold)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "citation_macro_f1": score,
                "citation_count_mae": cardinality["mae"],
            }
        )
        if score > best_f1 + 1e-12:
            best_f1 = score
            stale = 0
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "head": {
                        "maximum_candidates": train_store.evidence_mask.shape[1],
                        "maximum_citations": maximum_citations,
                        "hidden_dim": config.hidden_dim,
                        "extra_features": int(config.include_valid_count)
                        + int(config.include_visual_count),
                    },
                    "config": asdict(config),
                    "input_fingerprint": fingerprint,
                    "base_checkpoint_sha256": sha256_file(base_checkpoint),
                    "train_features_sha256": sha256_file(train_features),
                    "validation_features_sha256": sha256_file(validation_features),
                    "epoch": epoch,
                    "validation_citation_macro_f1": score,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= config.patience:
                break
    result = {
        "status": "trained",
        "variant": config.experiment_name,
        "seed": config.seed,
        "best_validation_citation_macro_f1": best_f1,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(target),
        "history": history,
        "input_fingerprint": fingerprint,
    }
    (output_dir / "cardinality_run.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def evaluate_decoupled_cardinality(
    base_checkpoint: Path,
    base_evaluation_dir: Path,
    head_checkpoint: Path,
    feature_path: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int,
    device: str,
) -> dict:
    started = time.perf_counter()
    store = load_feature_store(feature_path)
    model, model_device, _ = load_trained_model(base_checkpoint, device)
    checkpoint = torch.load(head_checkpoint, map_location=model_device, weights_only=False)
    if checkpoint.get("base_checkpoint_sha256") != sha256_file(base_checkpoint):
        raise ValueError(
            "cardinality head and base verifier checkpoint do not match"
        )
    if checkpoint["head"]["maximum_candidates"] != store.evidence_mask.shape[1]:
        raise ValueError(
            "cardinality head and evaluation feature width do not match"
        )
    head_config = checkpoint.get("config", {})
    extracted = _extract(
        model,
        store,
        model_device,
        include_valid_count=head_config.get("include_valid_count", True),
        include_visual_count=head_config.get("include_visual_count", True),
    )
    head = CitationCardinalityHead(**checkpoint["head"]).to(model_device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    with torch.inference_mode():
        counts = head(extracted["profiles"].to(model_device)).argmax(dim=1).cpu() + 1
    predicted_citations = _select(extracted["rankings"], counts)
    gold_citations = _gold_citations(store)
    base_metrics = json.loads(
        (base_evaluation_dir / "metrics.json").read_text(encoding="utf-8")
    )
    base_records = _read_jsonl(base_evaluation_dir / "comparison_records.jsonl")
    if [row["claim_id"] for row in base_records] != store.claim_ids:
        raise ValueError("base evaluation and adaptive citation inputs are unaligned")
    if len(base_records) != len(extracted["rankings"]):
        raise ValueError("base evaluation and adaptive rankings have different lengths")
    records = []
    for index, (base, citations) in enumerate(
        zip(base_records, predicted_citations, strict=True)
    ):
        row = dict(base)
        if row.get("ranked_citations") != extracted["rankings"][index]:
            raise ValueError(
                f"frozen evidence ranking changed for claim {base['claim_id']}"
            )
        row["predicted_citations"] = citations
        row["ranked_citations"] = extracted["rankings"][index]
        row["predicted_citation_count"] = len(citations)
        records.append(row)
    result = _write_decoupled_evaluation(
        variant=head_config.get("experiment_name", "dacs_scv"),
        store=store,
        base_metrics=base_metrics,
        base_records=base_records,
        predicted_citations=predicted_citations,
        output_dir=output_dir,
        bootstrap_samples=bootstrap_samples,
        elapsed_seconds=time.perf_counter() - started,
    )
    oracle = _select(
        extracted["rankings"],
        torch.tensor([len(row) for row in gold_citations], dtype=torch.long),
    )
    result["oracle_cardinality_citation_macro"] = citation_macro_prf(
        oracle, gold_citations
    )
    public_result = {
        key: value for key, value in result.items() if not key.startswith("_")
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(public_result, indent=2), encoding="utf-8"
    )
    return result
