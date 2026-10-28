from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from trace_scv.counterfactuals import (
    delete_selected_evidence,
    modality_matched_visual_swap,
)
from trace_scv.data.io import write_jsonl, write_predictions
from trace_scv.evaluation.metrics import (
    citation_cardinality_metrics,
    citation_macro_prf,
    classification_report,
    comprehensiveness,
    label_macro_f1,
    probabilistic_metrics,
    probability_invariance,
    ranked_evidence_metrics,
)
from trace_scv.features import ID_TO_LABEL, FeatureStore, load_feature_store
from trace_scv.schema import Prediction
from trace_scv.training import ArrayDataset, load_trained_model


def _forward(model, batch):
    return model(
        batch["claim"],
        batch["evidence"],
        batch["mask"],
        batch["types"],
        batch["adjacency"],
        batch.get("visual"),
        batch.get("visual_mask"),
    )


def _probability_rows(values: np.ndarray) -> list[dict[str, float]]:
    return [
        {ID_TO_LABEL[index]: float(probability) for index, probability in enumerate(row)}
        for row in values
    ]


def _render_explanation(label: str, ids: list[str], texts: list[str]) -> str:
    if not ids:
        return f"The model predicts {label}; no evidence item was selected."
    clauses = []
    for evidence_id, content in zip(ids[:2], texts[:2], strict=False):
        snippet = " ".join(content.split())[:180]
        clauses.append(f"{snippet} [{evidence_id}]")
    return f"The claim is predicted {label}. " + " ".join(clauses)


def _perturb_batch(batch: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
    if mode == "swap":
        return modality_matched_visual_swap(batch)[0]
    perturbed = {key: value.clone() for key, value in batch.items() if key != "index"}
    mask = perturbed["mask"]
    gold = perturbed["gold"].bool()
    if mode == "delete":
        perturbed["mask"] = mask & ~gold
        return perturbed
    if mode == "no_text":
        perturbed["evidence"] = torch.zeros_like(batch["evidence"])
        return perturbed
    if mode == "no_visual":
        perturbed["visual"] = torch.zeros_like(batch["visual"])
        perturbed["visual_mask"] = torch.zeros_like(batch["visual_mask"])
        return perturbed
    features = perturbed["evidence"]
    for row in range(mask.shape[0]):
        gold_positions = torch.where(gold[row] & mask[row])[0]
        non_gold = torch.where(mask[row] & ~gold[row])[0]
        if mode == "irrelevant":
            source_row = (row + 1) % mask.shape[0]
            source_candidates = torch.where(mask[source_row] & ~gold[source_row])[0]
            source = int(source_candidates[0]) if len(source_candidates) else 0
            free = torch.where(~mask[row])[0]
            target = int(free[0]) if len(free) else int(non_gold[0]) if len(non_gold) else int(torch.where(mask[row])[0][-1])
            features[row, target] = batch["evidence"][source_row, source]
            perturbed["visual"][row, target] = batch["visual"][source_row, source]
            perturbed["visual_mask"][row, target] = batch["visual_mask"][
                source_row, source
            ]
            perturbed["types"][row, target] = batch["types"][source_row, source]
            perturbed["mask"][row, target] = True
            perturbed["adjacency"][row, target, perturbed["mask"][row]] = 1.0
            perturbed["adjacency"][row, perturbed["mask"][row], target] = 1.0
        elif mode == "duplicate" and len(gold_positions):
            free = torch.where(~mask[row])[0]
            target = int(free[0]) if len(free) else int(non_gold[0]) if len(non_gold) else int(gold_positions[-1])
            source = int(gold_positions[0])
            features[row, target] = batch["evidence"][row, source]
            perturbed["visual"][row, target] = batch["visual"][row, source]
            perturbed["visual_mask"][row, target] = batch["visual_mask"][row, source]
            perturbed["types"][row, target] = batch["types"][row, source]
            perturbed["mask"][row, target] = True
            perturbed["adjacency"][row, target, perturbed["mask"][row]] = 1.0
            perturbed["adjacency"][row, perturbed["mask"][row], target] = 1.0
        else:
            if mode not in {"irrelevant", "duplicate", "swap", "no_text", "no_visual"}:
                raise ValueError(mode)
    perturbed["evidence"] = features
    return perturbed


def _group_bootstrap(
    store: FeatureStore,
    predicted_labels: list[str],
    predicted_citations: list[list[str]],
    samples: int,
    seed: int = 13,
) -> dict[str, dict[str, float]]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(store.group_ids):
        by_group[group].append(index)
    groups = sorted(by_group)
    if not groups or samples <= 0:
        return {}
    generator = np.random.default_rng(seed)
    gold_labels = [ID_TO_LABEL[int(value)] for value in store.labels]
    gold_citations = [
        [item for item, flag in zip(ids, mask[: len(ids)], strict=True) if flag > 0]
        for ids, mask in zip(store.evidence_ids, store.gold_mask, strict=True)
    ]
    labels = sorted(set(gold_labels) | set(predicted_labels))
    label_index = {label: index for index, label in enumerate(labels)}
    group_index = {group: index for index, group in enumerate(groups)}
    confusion = np.zeros((len(groups), len(labels), len(labels)), dtype=np.float64)
    citation_sum = np.zeros(len(groups), dtype=np.float64)
    counts = np.zeros(len(groups), dtype=np.float64)
    for index, group in enumerate(store.group_ids):
        target = group_index[group]
        confusion[target, label_index[gold_labels[index]], label_index[predicted_labels[index]]] += 1.0
        citation_sum[target] += citation_macro_prf(
            [predicted_citations[index]], [gold_citations[index]]
        )["f1"]
        counts[target] += 1.0

    values = {"verdict_macro_f1": [], "citation_macro_f1": []}
    probabilities = np.full(len(groups), 1.0 / len(groups))
    remaining = samples
    while remaining:
        current = min(256, remaining)
        weights = generator.multinomial(len(groups), probabilities, size=current)
        matrices = np.einsum("bg,gij->bij", weights, confusion, optimize=True)
        true_positive = np.diagonal(matrices, axis1=-2, axis2=-1)
        false_positive = matrices.sum(axis=-2) - true_positive
        false_negative = matrices.sum(axis=-1) - true_positive
        denominator = 2.0 * true_positive + false_positive + false_negative
        class_f1 = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator > 0,
        )
        values["verdict_macro_f1"].extend(class_f1.mean(axis=1).tolist())
        sample_counts = weights @ counts
        citation = np.divide(
            weights @ citation_sum,
            sample_counts,
            out=np.zeros(current, dtype=np.float64),
            where=sample_counts > 0,
        )
        values["citation_macro_f1"].extend(citation.tolist())
        remaining -= current
    return {
        key: {
            "lower": float(np.quantile(rows, 0.025)),
            "median": float(np.quantile(rows, 0.5)),
            "upper": float(np.quantile(rows, 0.975)),
        }
        for key, rows in values.items()
    }


def evaluate_majority(
    train_feature_path: Path,
    feature_path: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int = 1000,
    dataset_name: str = "sciver",
) -> dict:
    """Evaluate a deterministic baseline fitted only to train-label prevalence."""
    train = load_feature_store(train_feature_path)
    store = load_feature_store(feature_path)
    counts = np.bincount(train.labels, minlength=len(ID_TO_LABEL)).astype(np.float64)
    probabilities = counts / counts.sum()
    predicted_label = ID_TO_LABEL[int(probabilities.argmax())]
    gold_labels = [ID_TO_LABEL[int(value)] for value in store.labels]
    predicted_labels = [predicted_label] * len(gold_labels)
    probability_row = _probability_rows(probabilities[None, :])[0]
    probability_rows = [dict(probability_row) for _ in gold_labels]
    predicted_citations: list[list[str]] = [[] for _ in gold_labels]
    gold_citations = [
        [item for item, flag in zip(ids, mask[: len(ids)], strict=True) if flag > 0]
        for ids, mask in zip(store.evidence_ids, store.gold_mask, strict=True)
    ]
    classification = classification_report(gold_labels, predicted_labels)
    visual_gold_rows = [
        bool(np.any((store.gold_mask[index] > 0) & store.visual_mask[index]))
        for index in range(len(gold_labels))
    ]
    visual_indices = [index for index, value in enumerate(visual_gold_rows) if value]
    visual_subset_f1 = (
        label_macro_f1(
            [gold_labels[index] for index in visual_indices],
            [predicted_labels[index] for index in visual_indices],
        )
        if visual_indices
        else 0.0
    )
    predictions = [
        Prediction(
            claim_id=claim_id,
            predicted_label=predicted_label,
            probabilities=probabilities_row,
            cited_evidence_ids=(),
            explanation=f"The train-prevalence majority baseline predicts {predicted_label}.",
        )
        for claim_id, probabilities_row in zip(store.claim_ids, probability_rows, strict=True)
    ]
    result = {
        "status": "evaluated",
        "dataset": dataset_name,
        "variant": "majority",
        "seed": -1,
        "count": len(gold_labels),
        "accuracy": classification["accuracy"],
        "balanced_accuracy": classification["balanced_accuracy"],
        "per_class": classification["per_class"],
        "verdict_macro_f1": classification["macro_f1"],
        "citation_macro": citation_macro_prf(predicted_citations, gold_citations),
        "citation_cardinality": citation_cardinality_metrics(
            predicted_citations, gold_citations
        ),
        "evidence_ranking": ranked_evidence_metrics(predicted_citations, gold_citations, k=10),
        "probabilistic": probabilistic_metrics(probability_rows, gold_labels),
        "comprehensiveness": 0.0,
        "gold_evidence_comprehensiveness": 0.0,
        "irrelevant_invariance": 1.0,
        "duplicate_invariance": 1.0,
        "swap_invariance": 1.0,
        "swap_sensitivity": 0.0,
        "swap_coverage": 0.0,
        "sufficiency_macro_f1": classification["macro_f1"],
        "text_removed_macro_f1": classification["macro_f1"],
        "visual_removed_macro_f1": classification["macro_f1"],
        "visual_grounded_subset": {
            "count": int(sum(visual_gold_rows)),
            "clean_verdict_macro_f1": visual_subset_f1,
            "visual_removed_macro_f1": visual_subset_f1,
            "visual_removal_difference": 0.0,
            "swap_sensitivity": 0.0,
            "swap_coverage": 0.0,
        },
        "confidence_intervals": _group_bootstrap(
            store,
            predicted_labels,
            predicted_citations,
            bootstrap_samples,
            seed=13,
        ),
        "mean_selected_evidence": 0.0,
        "citation_coverage": 0.0,
        "inference_seconds": 0.0,
        "examples_per_second": None,
        "scientific_results_validated": False,
    }
    result["_comparison_records"] = [
        {
            "claim_id": claim_id,
            "group_id": group_id,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "gold_citations": gold_ids,
            "predicted_citations": [],
            "ranked_citations": [],
            "probabilities": {
                "clean": probability_rows[index],
                "selected_deleted": probability_rows[index],
                "gold_deleted": probability_rows[index],
                "sufficient": probability_rows[index],
                "irrelevant": probability_rows[index],
                "duplicate": probability_rows[index],
                "visual_swap": probability_rows[index],
                "no_text": probability_rows[index],
                "no_visual": probability_rows[index],
            },
            "swap_applied": False,
            "has_visual_gold": visual_gold_rows[index],
        }
        for index, (claim_id, group_id, gold_label, predicted_label, gold_ids) in enumerate(zip(
            store.claim_ids,
            store.group_ids,
            gold_labels,
            predicted_labels,
            gold_citations,
            strict=True,
        ))
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "comparison_records.jsonl", result["_comparison_records"])
    public_result = {key: value for key, value in result.items() if not key.startswith("_")}
    (output_dir / "metrics.json").write_text(
        json.dumps(public_result, indent=2), encoding="utf-8"
    )
    return result


def evaluate_checkpoint(
    checkpoint_path: Path,
    feature_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 64,
    bootstrap_samples: int = 1000,
    device: str = "auto",
    dataset_name: str = "sciver",
) -> dict:
    store = load_feature_store(feature_path)
    model, target, checkpoint = load_trained_model(checkpoint_path, device)
    loader = DataLoader(ArrayDataset(store), batch_size=batch_size, shuffle=False)
    predicted_labels: list[str] = []
    gold_labels: list[str] = []
    predicted_citations: list[list[str]] = []
    ranked_citations: list[list[str]] = []
    gold_citations: list[list[str]] = []
    clean_probabilities: list[dict[str, float]] = []
    intervention_probabilities: dict[str, list[dict[str, float]]] = {
        mode: []
        for mode in (
            "delete",
            "gold_delete",
            "sufficient",
            "irrelevant",
            "duplicate",
            "swap",
            "no_text",
            "no_visual",
        )
    }
    swap_changed_rows: list[bool] = []
    predictions: list[Prediction] = []
    started = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            indices = batch["index"].tolist()
            device_batch = {
                key: value.to(target) for key, value in batch.items() if key != "index"
            }
            output = _forward(model, device_batch)
            probabilities = output["logits"].softmax(-1).cpu().numpy()
            hard = output["hard_selection"].cpu().numpy()
            selector_scores = output["selector_logits"].cpu().numpy()
            cited_mask = output["hard_selection"].bool()
            deleted, _ = delete_selected_evidence(device_batch, cited_mask)
            gold_deleted = {key: value.clone() for key, value in device_batch.items()}
            gold_deleted["mask"] = device_batch["mask"] & ~device_batch["gold"].bool()
            sufficient = {key: value.clone() for key, value in device_batch.items()}
            sufficient["mask"] = device_batch["mask"] & cited_mask
            for mode, changed in (
                ("delete", deleted),
                ("gold_delete", gold_deleted),
                ("sufficient", sufficient),
            ):
                rows = _forward(model, changed)["logits"].softmax(-1).cpu().numpy()
                intervention_probabilities[mode].extend(_probability_rows(rows))
            for mode in ("irrelevant", "duplicate", "no_text", "no_visual"):
                changed = _perturb_batch(device_batch, mode)
                rows = _forward(model, changed)["logits"].softmax(-1).cpu().numpy()
                intervention_probabilities[mode].extend(_probability_rows(rows))
            swapped, changed_rows = modality_matched_visual_swap(device_batch)
            rows = _forward(model, swapped)["logits"].softmax(-1).cpu().numpy()
            intervention_probabilities["swap"].extend(_probability_rows(rows))
            swap_changed_rows.extend(changed_rows.cpu().numpy().astype(bool).tolist())

            clean_probabilities.extend(_probability_rows(probabilities))
            for local, index in enumerate(indices):
                predicted_label = ID_TO_LABEL[int(probabilities[local].argmax())]
                gold_label = ID_TO_LABEL[int(store.labels[index])]
                ids = store.evidence_ids[index]
                order = np.argsort(-selector_scores[local])
                ranked = [] if model.variant == "claim_only" else [ids[j] for j in order if j < len(ids)]
                selected = [item for item in ranked if hard[local, ids.index(item)] > 0]
                selected_text = [
                    store.evidence_text[index][ids.index(item)] for item in selected
                ]
                gold_ids = [
                    ids[j]
                    for j in range(min(len(ids), store.gold_mask.shape[1]))
                    if store.gold_mask[index, j] > 0
                ]
                predicted_labels.append(predicted_label)
                gold_labels.append(gold_label)
                predicted_citations.append(selected)
                ranked_citations.append(ranked)
                gold_citations.append(gold_ids)
                predictions.append(
                    Prediction(
                        claim_id=store.claim_ids[index],
                        predicted_label=predicted_label,
                        probabilities=clean_probabilities[-len(indices) + local],
                        cited_evidence_ids=tuple(selected),
                        explanation=_render_explanation(predicted_label, selected, selected_text),
                    )
                )

    elapsed = time.perf_counter() - started
    citation = citation_macro_prf(predicted_citations, gold_citations)
    classification = classification_report(gold_labels, predicted_labels)
    valid_swap_clean = [
        row for row, changed in zip(clean_probabilities, swap_changed_rows, strict=True) if changed
    ]
    valid_swap_changed = [
        row
        for row, changed in zip(
            intervention_probabilities["swap"], swap_changed_rows, strict=True
        )
        if changed
    ]
    swap_invariance = (
        probability_invariance(valid_swap_clean, valid_swap_changed)
        if valid_swap_clean
        else 1.0
    )
    visual_gold_rows = [
        bool(np.any((store.gold_mask[index] > 0) & store.visual_mask[index]))
        for index in range(len(gold_labels))
    ]
    visual_indices = [index for index, value in enumerate(visual_gold_rows) if value]
    visual_removed_predictions = [
        max(row, key=row.get) for row in intervention_probabilities["no_visual"]
    ]
    visual_clean_f1 = (
        label_macro_f1(
            [gold_labels[index] for index in visual_indices],
            [predicted_labels[index] for index in visual_indices],
        )
        if visual_indices
        else 0.0
    )
    visual_removed_f1 = (
        label_macro_f1(
            [gold_labels[index] for index in visual_indices],
            [visual_removed_predictions[index] for index in visual_indices],
        )
        if visual_indices
        else 0.0
    )
    visual_swap_indices = [
        index for index in visual_indices if swap_changed_rows[index]
    ]
    visual_swap_invariance = (
        probability_invariance(
            [clean_probabilities[index] for index in visual_swap_indices],
            [intervention_probabilities["swap"][index] for index in visual_swap_indices],
        )
        if visual_swap_indices
        else 1.0
    )
    result = {
        "status": "evaluated",
        "dataset": dataset_name,
        "variant": checkpoint["train_config"].get("experiment_name") or checkpoint["model"]["variant"],
        "seed": checkpoint["train_config"]["seed"],
        "count": len(gold_labels),
        "accuracy": classification["accuracy"],
        "balanced_accuracy": classification["balanced_accuracy"],
        "per_class": classification["per_class"],
        "verdict_macro_f1": classification["macro_f1"],
        "citation_macro": citation,
        "citation_cardinality": citation_cardinality_metrics(
            predicted_citations, gold_citations
        ),
        "evidence_ranking": ranked_evidence_metrics(ranked_citations, gold_citations, k=10),
        "probabilistic": probabilistic_metrics(clean_probabilities, gold_labels),
        "comprehensiveness": comprehensiveness(
            clean_probabilities, intervention_probabilities["delete"], gold_labels
        ),
        "gold_evidence_comprehensiveness": comprehensiveness(
            clean_probabilities, intervention_probabilities["gold_delete"], gold_labels
        ),
        "irrelevant_invariance": probability_invariance(
            clean_probabilities, intervention_probabilities["irrelevant"]
        ),
        "duplicate_invariance": probability_invariance(
            clean_probabilities, intervention_probabilities["duplicate"]
        ),
        "swap_invariance": swap_invariance,
        "swap_sensitivity": 1.0 - swap_invariance,
        "swap_coverage": float(np.mean(swap_changed_rows)) if swap_changed_rows else 0.0,
        "sufficiency_macro_f1": label_macro_f1(
            gold_labels,
            [max(row, key=row.get) for row in intervention_probabilities["sufficient"]],
        ),
        "text_removed_macro_f1": label_macro_f1(
            gold_labels,
            [max(row, key=row.get) for row in intervention_probabilities["no_text"]],
        ),
        "visual_removed_macro_f1": label_macro_f1(
            gold_labels,
            visual_removed_predictions,
        ),
        "visual_grounded_subset": {
            "count": len(visual_indices),
            "clean_verdict_macro_f1": visual_clean_f1,
            "visual_removed_macro_f1": visual_removed_f1,
            "visual_removal_difference": visual_removed_f1 - visual_clean_f1,
            "swap_sensitivity": 1.0 - visual_swap_invariance,
            "swap_coverage": (
                len(visual_swap_indices) / len(visual_indices) if visual_indices else 0.0
            ),
        },
        "confidence_intervals": _group_bootstrap(
            store,
            predicted_labels,
            predicted_citations,
            bootstrap_samples,
            seed=int(checkpoint["train_config"]["seed"]),
        ),
        "mean_selected_evidence": float(np.mean([len(row) for row in predicted_citations])),
        "citation_coverage": float(np.mean([bool(row) for row in predicted_citations])),
        "inference_seconds": elapsed,
        "examples_per_second": len(gold_labels) / elapsed if elapsed else None,
        "scientific_results_validated": False,
    }
    result["_comparison_records"] = [
        {
            "claim_id": claim_id,
            "group_id": group_id,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "gold_citations": gold_ids,
            "predicted_citations": predicted_ids,
            "ranked_citations": ranked_citations[index],
            "probabilities": {
                "clean": clean_probabilities[index],
                "selected_deleted": intervention_probabilities["delete"][index],
                "gold_deleted": intervention_probabilities["gold_delete"][index],
                "sufficient": intervention_probabilities["sufficient"][index],
                "irrelevant": intervention_probabilities["irrelevant"][index],
                "duplicate": intervention_probabilities["duplicate"][index],
                "visual_swap": intervention_probabilities["swap"][index],
                "no_text": intervention_probabilities["no_text"][index],
                "no_visual": intervention_probabilities["no_visual"][index],
            },
            "swap_applied": swap_changed_rows[index],
            "has_visual_gold": visual_gold_rows[index],
        }
        for index, (claim_id, group_id, gold_label, predicted_label, gold_ids, predicted_ids) in enumerate(zip(
            store.claim_ids,
            store.group_ids,
            gold_labels,
            predicted_labels,
            gold_citations,
            predicted_citations,
            strict=True,
        ))
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "comparison_records.jsonl", result["_comparison_records"])
    public_result = {key: value for key, value in result.items() if not key.startswith("_")}
    (output_dir / "metrics.json").write_text(
        json.dumps(public_result, indent=2), encoding="utf-8"
    )
    return result
