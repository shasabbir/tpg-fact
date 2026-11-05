from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import numpy as np

from trace_scv.cefes import expected_prefix_f1_curve
from trace_scv.data.io import sha256_file
from trace_scv.q1_phase2_retrieval import _claim_role
from trace_scv.selection.calibration import (
    binary_brier_score,
    binary_expected_calibration_error,
    binary_nll,
)
from trace_scv.selection.mocheg_study import LogisticEvidenceScorer
from trace_scv.selection.probability_calibration import (
    CALIBRATOR_KINDS,
    apply_calibrator,
    fit_calibrator,
)


@dataclass(frozen=True, slots=True)
class Phase2StudyConfig:
    seeds: tuple[int, ...] = (13, 37, 73, 101, 137)
    partition_seed: int = 20_261_102
    fixed_budgets: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 60)
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    size_penalties: tuple[float, ...] = (
        0.0,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
    )
    redundancy_penalties: tuple[float, ...] = (0.0, 0.0025, 0.005)
    scorer_families: tuple[str, ...] = (
        "shared_logistic_balanced",
        "shared_logistic_unweighted",
        "modality_specific_logistic_balanced",
        "modality_specific_logistic_unweighted",
        "cross_encoder_augmented_logistic_balanced",
        "cross_encoder_augmented_logistic_unweighted",
    )
    calibrator_kinds: tuple[str, ...] = CALIBRATOR_KINDS
    baseline_text_recall: float = 0.3616018276517159
    baseline_image_recall: float = 0.21384297520661158
    maximum_selected_items: int = 60
    logistic_epochs: int = 400
    logistic_learning_rate: float = 0.08
    logistic_l2: float = 1e-3
    count_epochs: int = 300
    count_learning_rate: float = 0.05
    count_l2: float = 1e-3
    bootstrap_samples: int = 10_000

    @classmethod
    def from_protocol(cls, path: Path) -> "Phase2StudyConfig":
        import yaml

        protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
        scoring = protocol["scoring"]
        selection = protocol["selection"]
        evaluation = protocol["evaluation"]
        losses = tuple(scoring.get("losses", ("balanced",)))
        feature_families = tuple(scoring["controls"])
        scorer_families = tuple(
            f"{family}_{loss}" for family in feature_families for loss in losses
        )
        calibrators = tuple(scoring["calibration"]["candidates"])
        unknown = set(calibrators) - set(CALIBRATOR_KINDS)
        if unknown:
            raise ValueError(f"unknown protocol calibrators: {sorted(unknown)}")
        baselines = protocol["retrieval"].get("baselines", {})
        return cls(
            seeds=tuple(map(int, protocol["study"]["seeds"])),
            partition_seed=int(protocol["data"]["train_partition_seed"]),
            fixed_budgets=tuple(map(int, selection["fixed_budgets"])),
            thresholds=tuple(map(float, selection["thresholds"])),
            size_penalties=tuple(map(float, selection["size_penalties"])),
            redundancy_penalties=tuple(map(float, selection["redundancy_penalties"])),
            scorer_families=scorer_families,
            calibrator_kinds=calibrators,
            baseline_text_recall=float(
                baselines.get("text_recall_at_50", 0.3616018276517159)
            ),
            baseline_image_recall=float(
                baselines.get("image_recall_at_50", 0.21384297520661158)
            ),
            maximum_selected_items=int(selection["maximum_selected_items"]),
            bootstrap_samples=int(evaluation["bootstrap_samples"]),
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
    balanced: bool,
) -> LogisticEvidenceScorer:
    import torch

    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(targets) or len(np.unique(targets)) != 2:
        raise ValueError("logistic fitting requires aligned features and both classes")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (values - mean) / scale
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(standardized, device=device)
    y = torch.as_tensor(targets, device=device)
    weights = torch.zeros(values.shape[1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    positives = float(targets.sum())
    negatives = float(len(targets) - positives)
    sample_weights = (
        torch.where(
            y > 0.5,
            torch.full_like(y, len(targets) / (2.0 * positives)),
            torch.full_like(y, len(targets) / (2.0 * negatives)),
        )
        if balanced
        else None
    )
    optimizer = torch.optim.Adam([weights, bias], lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, weight=sample_weights
        ) + l2 * weights.square().mean()
        loss.backward()
        optimizer.step()
    result = LogisticEvidenceScorer(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in weights.detach().cpu().numpy()),
        bias=float(bias.detach().cpu()),
    )
    del x, y, weights, bias, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


@dataclass(frozen=True, slots=True)
class ScorerBundle:
    family: str
    feature_indices: tuple[int, ...]
    shared: LogisticEvidenceScorer | None = None
    text: LogisticEvidenceScorer | None = None
    image: LogisticEvidenceScorer | None = None

    def predict(self, features: np.ndarray, modalities: Sequence[str]) -> np.ndarray:
        values = features[:, self.feature_indices]
        if self.shared is not None:
            return self.shared.predict_probabilities(values)
        output = np.zeros(len(values), dtype=np.float64)
        modality_array = np.asarray(modalities)
        for modality, scorer in (("text", self.text), ("image", self.image)):
            indices = np.flatnonzero(modality_array == modality)
            if scorer is None or not len(indices):
                continue
            output[indices] = scorer.predict_probabilities(values[indices])
        return output

    def to_dict(self) -> dict[str, object]:
        def scorer_dict(value: LogisticEvidenceScorer | None) -> dict[str, object] | None:
            return asdict(value) if value is not None else None

        return {
            "family": self.family,
            "feature_indices": list(self.feature_indices),
            "shared": scorer_dict(self.shared),
            "text": scorer_dict(self.text),
            "image": scorer_dict(self.image),
        }


def _fit_scorer_family(
    family: str,
    features: np.ndarray,
    labels: np.ndarray,
    records: Sequence[dict[str, object]],
    fit_indices: np.ndarray,
    *,
    config: Phase2StudyConfig,
) -> ScorerBundle:
    old_like = (0, 1, 9, 10, 11, 12, 13)
    all_features = tuple(range(features.shape[1]))
    if family.endswith("_balanced"):
        base_family = family.removesuffix("_balanced")
        balanced = True
    elif family.endswith("_unweighted"):
        base_family = family.removesuffix("_unweighted")
        balanced = False
    else:
        base_family = family
        balanced = True
    selected = old_like if base_family == "shared_logistic" else all_features
    if base_family in {"shared_logistic", "cross_encoder_augmented_logistic"}:
        scorer = _fit_logistic(
            features[fit_indices][:, selected],
            labels[fit_indices],
            learning_rate=config.logistic_learning_rate,
            epochs=config.logistic_epochs,
            l2=config.logistic_l2,
            balanced=balanced,
        )
        return ScorerBundle(family=family, feature_indices=selected, shared=scorer)
    if base_family != "modality_specific_logistic":
        raise ValueError(f"unknown scorer family: {family}")
    modality_values = np.asarray([str(row["modality"]) for row in records])
    scorers: dict[str, LogisticEvidenceScorer] = {}
    for modality in ("text", "image"):
        indices = fit_indices[modality_values[fit_indices] == modality]
        scorers[modality] = _fit_logistic(
            features[indices][:, selected],
            labels[indices],
            learning_rate=config.logistic_learning_rate,
            epochs=config.logistic_epochs,
            l2=config.logistic_l2,
            balanced=balanced,
        )
    return ScorerBundle(
        family=family,
        feature_indices=selected,
        text=scorers["text"],
        image=scorers["image"],
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _temperature_transform(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    return _sigmoid(logits / temperature)


def _fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 1.0
    temperatures = np.exp(np.linspace(math.log(0.05), math.log(20.0), 161))
    best_temperature = 1.0
    best_nll = math.inf
    for temperature in temperatures:
        calibrated = _temperature_transform(probabilities, float(temperature))
        clipped = np.clip(calibrated, 1e-7, 1.0 - 1e-7)
        nll = float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        )
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return best_temperature


def _calibrate_by_modality(
    raw: np.ndarray,
    labels: np.ndarray,
    modalities: Sequence[str],
    calibration_indices: np.ndarray,
    guard_indices: np.ndarray,
    *,
    calibrator_kinds: Sequence[str],
) -> tuple[np.ndarray, dict[str, object]]:
    calibrated = np.asarray(raw, dtype=np.float64).copy()
    modality_array = np.asarray(modalities)
    report: dict[str, object] = {}
    for modality in ("text", "image"):
        fit = calibration_indices[modality_array[calibration_indices] == modality]
        guard = guard_indices[modality_array[guard_indices] == modality]
        candidates: list[dict[str, object]] = []
        fitted_models = {}
        for kind in calibrator_kinds:
            model = fit_calibrator(kind, raw[fit], labels[fit])
            fitted_models[kind] = model
            guard_values = apply_calibrator(raw[guard], model)
            candidates.append(
                {
                    "kind": kind,
                    "model": model.to_dict(),
                    "guard_brier": binary_brier_score(guard_values, labels[guard]),
                    "guard_nll": binary_nll(guard_values, labels[guard]),
                    "guard_mean_probability": float(np.mean(guard_values)),
                }
            )
        selected = min(
            candidates,
            key=lambda row: (float(row["guard_brier"]), str(row["kind"])),
        )
        selected_kind = str(selected["kind"])
        selected_model = fitted_models[selected_kind]
        indices = np.flatnonzero(modality_array == modality)
        calibrated[indices] = apply_calibrator(raw[indices], selected_model)
        guard_prevalence = float(np.mean(labels[guard]))
        guard_prevalence_brier = float(
            np.mean((labels[guard] - guard_prevalence) ** 2)
        )
        report[modality] = {
            "selected_kind": selected_kind,
            "selected_model": selected_model.to_dict(),
            "candidates": candidates,
            "fit_examples": len(fit),
            "guard_examples": len(guard),
            "guard_prevalence": guard_prevalence,
            "guard_prevalence_brier": guard_prevalence_brier,
            "guard_selected_brier": selected["guard_brier"],
            "guard_brier_skill": 1.0
            - float(selected["guard_brier"]) / guard_prevalence_brier,
        }
    return calibrated, report


def _group_indices(records: Sequence[dict[str, object]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        key = f"{row['split']}:{row['claim_id']}"
        grouped.setdefault(key, []).append(index)
    return grouped


def _evaluable_record_mask(
    records: Sequence[dict[str, object]],
    full_gold: dict[str, set[str]],
) -> np.ndarray:
    """Keep only claims with a released, non-empty complete-gold evidence set."""
    return np.asarray(
        [
            bool(full_gold.get(f"{row['split']}:{row['claim_id']}"))
            for row in records
        ],
        dtype=bool,
    )


def _retained_packet_metrics(
    records: Sequence[dict[str, object]],
    full_gold: dict[str, set[str]],
) -> dict[str, float]:
    grouped = _group_indices(records)
    text_recall: list[float] = []
    image_recall: list[float] = []
    oracle_f1: list[float] = []
    complete: list[float] = []
    zero_reachable = 0
    for key, indices in grouped.items():
        gold = full_gold[key]
        retained = {str(records[index]["evidence_id"]) for index in indices}
        reachable = retained & gold
        text_gold = {value for value in gold if value.startswith("text:")}
        image_gold = {value for value in gold if value.startswith("image:")}
        if text_gold:
            text_recall.append(len(retained & text_gold) / len(text_gold))
        if image_gold:
            image_recall.append(len(retained & image_gold) / len(image_gold))
        oracle_f1.append(2.0 * len(reachable) / (len(reachable) + len(gold)))
        complete.append(float(reachable == gold))
        zero_reachable += int(not reachable)
    return {
        "text_recall_at_40": fmean(text_recall),
        "text_eligible_claims": float(len(text_recall)),
        "image_recall_at_20": fmean(image_recall),
        "image_eligible_claims": float(len(image_recall)),
        "packet_oracle_f1": fmean(oracle_f1),
        "complete_gold_claim_coverage": fmean(complete),
        "eligible_claims": float(len(grouped)),
        "eligible_claims_with_zero_reachable_gold": float(zero_reachable),
    }


def _mean_average_precision(
    records: Sequence[dict[str, object]],
    probabilities: np.ndarray,
    indices: np.ndarray,
    full_gold: dict[str, set[str]],
) -> float:
    index_set = set(map(int, indices))
    grouped = _group_indices(records)
    values: list[float] = []
    for key, rows in grouped.items():
        rows = [index for index in rows if index in index_set]
        if not rows or not full_gold.get(key):
            continue
        rows.sort(key=lambda index: (-float(probabilities[index]), index))
        hits = 0
        precision_sum = 0.0
        for rank, index in enumerate(rows, start=1):
            if bool(records[index]["gold"]):
                hits += 1
                precision_sum += hits / rank
        values.append(precision_sum / len(full_gold[key]))
    return fmean(values) if values else 0.0


def _set_metrics(predicted: set[str], gold: set[str]) -> dict[str, float]:
    overlap = len(predicted & gold)
    denominator = len(predicted) + len(gold)
    return {
        "precision": overlap / len(predicted) if predicted else 0.0,
        "recall": overlap / len(gold) if gold else 0.0,
        "f1": 2.0 * overlap / denominator if denominator else 1.0,
        "jaccard": overlap / len(predicted | gold) if predicted or gold else 1.0,
        "exact": float(predicted == gold),
    }


def _penalized_prefix(
    probabilities: Sequence[float],
    source_groups: Sequence[str],
    *,
    maximum_items: int,
    size_penalty: float,
    redundancy_penalty: float,
) -> list[int]:
    order = sorted(range(len(probabilities)), key=lambda index: (-float(probabilities[index]), index))
    ranked = tuple(float(probabilities[index]) for index in order)
    curve = _cached_expected_curve(ranked)
    best_count = 1
    best_score = -math.inf
    seen: set[str] = set()
    redundant = 0
    for count in range(1, min(maximum_items, len(order)) + 1):
        group = source_groups[order[count - 1]]
        if group in seen:
            redundant += 1
        seen.add(group)
        score = curve[count] - size_penalty * count - redundancy_penalty * redundant
        if score > best_score + 1e-15:
            best_count = count
            best_score = score
    return order[:best_count]


@lru_cache(maxsize=100_000)
def _cached_expected_curve(ranked_probabilities: tuple[float, ...]) -> tuple[float, ...]:
    return expected_prefix_f1_curve(ranked_probabilities)


def _select_local(
    method: str,
    probabilities: Sequence[float],
    source_groups: Sequence[str],
    *,
    maximum_items: int,
    fixed_k: int | None = None,
    threshold: float | None = None,
    size_penalty: float = 0.0,
    redundancy_penalty: float = 0.0,
    predicted_count: int | None = None,
) -> list[int]:
    order = sorted(range(len(probabilities)), key=lambda index: (-float(probabilities[index]), index))
    if method == "fixed":
        if fixed_k is None:
            raise ValueError("fixed_k is required")
        return order[:fixed_k]
    if method == "threshold":
        if threshold is None:
            raise ValueError("threshold is required")
        selected = [index for index in order if probabilities[index] >= threshold]
        return (selected or order[:1])[:maximum_items]
    if method == "cefes":
        ranked = tuple(float(probabilities[index]) for index in order)
        curve = _cached_expected_curve(ranked)
        upper = min(maximum_items, len(order))
        best_count = min(range(1, upper + 1), key=lambda count: (-curve[count], count))
        return order[:best_count]
    if method == "penalized_cefes":
        return _penalized_prefix(
            probabilities,
            source_groups,
            maximum_items=maximum_items,
            size_penalty=size_penalty,
            redundancy_penalty=redundancy_penalty,
        )
    if method == "count":
        if predicted_count is None:
            raise ValueError("predicted_count is required")
        return order[: max(1, min(maximum_items, predicted_count))]
    raise ValueError(f"unknown method: {method}")


def _claim_profile(probabilities: Sequence[float], modalities: Sequence[str], maximum: int) -> np.ndarray:
    order = sorted(range(len(probabilities)), key=lambda index: (-float(probabilities[index]), index))
    ordered = np.asarray([float(probabilities[index]) for index in order], dtype=np.float32)
    padded = np.zeros(maximum, dtype=np.float32)
    padded[: min(maximum, len(ordered))] = ordered[:maximum]
    return np.concatenate(
        [
            padded,
            np.asarray(
                [
                    ordered.mean() if len(ordered) else 0.0,
                    ordered.max() if len(ordered) else 0.0,
                    ordered.std() if len(ordered) else 0.0,
                    len(ordered) / maximum,
                    sum(value == "text" for value in modalities) / max(1, len(modalities)),
                    sum(value == "image" for value in modalities) / max(1, len(modalities)),
                ],
                dtype=np.float32,
            ),
        ]
    )


@dataclass(frozen=True, slots=True)
class CountClassifier:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    maximum_count: int

    def predict(self, values: np.ndarray) -> np.ndarray:
        standardized = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        logits = standardized @ np.asarray(self.weights) + np.asarray(self.bias)
        return logits.argmax(axis=1) + 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _fit_count_classifier(
    features: np.ndarray,
    counts: Sequence[int],
    *,
    config: Phase2StudyConfig,
) -> CountClassifier:
    import torch

    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(counts, dtype=np.int64) - 1
    maximum = config.maximum_selected_items
    if np.any(targets < 0) or np.any(targets >= maximum):
        raise ValueError("count target outside declared range")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (values - mean) / scale
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(standardized, device=device)
    y = torch.as_tensor(targets, device=device)
    weights = torch.zeros((values.shape[1], maximum), device=device, requires_grad=True)
    bias = torch.zeros(maximum, device=device, requires_grad=True)
    counts_by_class = np.bincount(targets, minlength=maximum).astype(np.float32)
    class_weights = np.zeros(maximum, dtype=np.float32)
    present = counts_by_class > 0
    class_weights[present] = len(targets) / (present.sum() * counts_by_class[present])
    loss_weights = torch.as_tensor(class_weights, device=device)
    optimizer = torch.optim.Adam([weights, bias], lr=config.count_learning_rate)
    for _ in range(config.count_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weights + bias
        loss = torch.nn.functional.cross_entropy(logits, y, weight=loss_weights)
        loss = loss + config.count_l2 * weights.square().mean()
        loss.backward()
        optimizer.step()
    result = CountClassifier(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(tuple(float(value) for value in row) for row in weights.detach().cpu().numpy()),
        bias=tuple(float(value) for value in bias.detach().cpu().numpy()),
        maximum_count=maximum,
    )
    del x, y, weights, bias, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _claim_keys_for_role(
    records: Sequence[dict[str, object]], seed: int, role: str
) -> list[str]:
    keys = sorted(_group_indices(records))
    if role == "validation":
        return [key for key in keys if key.startswith("validation:")]
    return [
        key
        for key in keys
        if key.startswith("train:")
        and _claim_role(key.split(":", 1)[1], seed) == role
    ]


def _indices_for_keys(grouped: dict[str, list[int]], keys: Sequence[str]) -> np.ndarray:
    return np.asarray([index for key in keys for index in grouped[key]], dtype=np.int64)


def _evaluate_method(
    method: str,
    records: Sequence[dict[str, object]],
    probabilities: np.ndarray,
    full_gold: dict[str, set[str]],
    keys: Sequence[str],
    *,
    maximum_items: int,
    fixed_k: int | None = None,
    threshold: float | None = None,
    size_penalty: float = 0.0,
    redundancy_penalty: float = 0.0,
    count_predictions: dict[str, int] | None = None,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    grouped = _group_indices(records)
    claim_rows: list[dict[str, object]] = []
    for key in keys:
        indices = grouped[key]
        row_probabilities = [float(probabilities[index]) for index in indices]
        source_groups = [str(records[index]["source_group"]) for index in indices]
        selected_local = _select_local(
            method,
            row_probabilities,
            source_groups,
            maximum_items=maximum_items,
            fixed_k=fixed_k,
            threshold=threshold,
            size_penalty=size_penalty,
            redundancy_penalty=redundancy_penalty,
            predicted_count=(count_predictions or {}).get(key),
        )
        selected = {str(records[indices[index]]["evidence_id"]) for index in selected_local}
        gold = full_gold[key]
        scores = _set_metrics(selected, gold)
        claim_rows.append(
            {
                "key": key,
                "split": key.split(":", 1)[0],
                "claim_id": key.split(":", 1)[1],
                "method": method,
                "selected_evidence_ids": sorted(selected),
                "gold_evidence_ids": sorted(gold),
                "selected_count": len(selected),
                "gold_count": len(gold),
                **scores,
            }
        )
    metrics = {
        key: fmean(float(row[key]) for row in claim_rows)
        for key in ("precision", "recall", "f1", "jaccard", "exact", "selected_count")
    }
    metrics["claims"] = float(len(claim_rows))
    return metrics, claim_rows


def _paired_bootstrap(
    left: Sequence[dict[str, object]],
    right: Sequence[dict[str, object]],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    right_map = {str(row["key"]): float(row["f1"]) for row in right}
    differences = np.asarray(
        [float(row["f1"]) - right_map[str(row["key"])] for row in left],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    values: list[float] = []
    remaining = samples
    while remaining:
        current = min(512, remaining)
        indices = generator.integers(0, len(differences), size=(current, len(differences)))
        values.extend(differences[indices].mean(axis=1).tolist())
        remaining -= current
    return {
        "mean": float(differences.mean()),
        "lower": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 0.975)),
        "sample_count": samples,
        "resampling_seed": seed,
        "samples": values,
    }


def _calibration_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    prevalence = float(np.mean(labels))
    prevalence_brier = float(np.mean((labels - prevalence) ** 2))
    brier = binary_brier_score(probabilities, labels)
    return {
        "brier": brier,
        "nll": binary_nll(probabilities, labels),
        "ece": binary_expected_calibration_error(probabilities, labels),
        "prevalence": prevalence,
        "mean_probability": float(np.mean(probabilities)),
        "prevalence_brier": prevalence_brier,
        "brier_skill": 1.0 - brier / prevalence_brier,
    }


def _choose_count_predictions(
    records: Sequence[dict[str, object]],
    probabilities: np.ndarray,
    full_gold: dict[str, set[str]],
    fit_keys: Sequence[str],
    target_keys: Sequence[str],
    *,
    config: Phase2StudyConfig,
) -> tuple[CountClassifier, dict[str, int]]:
    grouped = _group_indices(records)

    def profile(key: str) -> np.ndarray:
        rows = grouped[key]
        return _claim_profile(
            [float(probabilities[index]) for index in rows],
            [str(records[index]["modality"]) for index in rows],
            config.maximum_selected_items,
        )

    fit_profiles = np.stack([profile(key) for key in fit_keys])
    fit_counts = [
        max(1, min(config.maximum_selected_items, len(full_gold[key])))
        for key in fit_keys
    ]
    model = _fit_count_classifier(fit_profiles, fit_counts, config=config)
    target_profiles = np.stack([profile(key) for key in target_keys])
    predicted = model.predict(target_profiles)
    return model, {key: int(value) for key, value in zip(target_keys, predicted, strict=True)}


def _best_parameter(
    values: Sequence[object],
    evaluate,
) -> tuple[object, dict[str, float]]:
    scored: list[tuple[object, dict[str, float]]] = [(value, evaluate(value)) for value in values]
    best_value, best_metrics = min(
        scored,
        key=lambda row: (-float(row[1]["f1"]), str(row[0])),
    )
    return best_value, best_metrics


def _aggregate_seed_aligned_bootstrap(
    seed_claim_rows: Sequence[tuple[int, Sequence[dict[str, object]], Sequence[dict[str, object]]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    effects: dict[str, list[float]] = {}
    for _, left, right in seed_claim_rows:
        right_map = {str(row["key"]): float(row["f1"]) for row in right}
        for row in left:
            key = str(row["key"])
            effects.setdefault(key, []).append(float(row["f1"]) - right_map[key])
    keys = sorted(effects)
    values = np.asarray([fmean(effects[key]) for key in keys], dtype=np.float64)
    generator = np.random.default_rng(seed)
    bootstrap: list[float] = []
    remaining = samples
    while remaining:
        current = min(512, remaining)
        indices = generator.integers(0, len(values), size=(current, len(values)))
        bootstrap.extend(values[indices].mean(axis=1).tolist())
        remaining -= current
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(bootstrap, 0.025)),
        "median": float(np.quantile(bootstrap, 0.5)),
        "upper": float(np.quantile(bootstrap, 0.975)),
        "claims": len(keys),
        "sample_count": samples,
        "resampling_seed": seed,
        "seed_alignment": "mean within claim across frozen seeds before resampling",
        "samples": bootstrap,
    }


def run_phase2_selector_study(
    retrieval_root: Path,
    output_root: Path,
    protocol_path: Path,
    *,
    config: Phase2StudyConfig = Phase2StudyConfig(),
) -> dict[str, object]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    retrieval_manifest_path = retrieval_root / "retrieval.manifest.json"
    retrieval_audit_path = retrieval_root / "retrieval.audit.json"
    retrieval_manifest = json.loads(retrieval_manifest_path.read_text(encoding="utf-8"))
    retrieval_audit = json.loads(retrieval_audit_path.read_text(encoding="utf-8"))
    if retrieval_manifest.get("status") != "pass" or retrieval_audit.get("status") != "pass":
        raise RuntimeError("Phase-2 retrieval and its independent audit must pass")
    arrays = np.load(retrieval_root / "candidate_features.npz")
    features = np.asarray(arrays["features"], dtype=np.float64)
    labels = np.asarray(arrays["gold"], dtype=np.int8)
    records = _read_jsonl(retrieval_root / "candidate_records.jsonl")
    if len(features) != len(labels) or len(features) != len(records):
        raise ValueError("Phase-2 feature and candidate records are misaligned")
    full_gold_raw = json.loads((retrieval_root / "full_gold.json").read_text(encoding="utf-8"))
    full_gold = {str(key): set(map(str, values)) for key, values in full_gold_raw.items()}
    original_record_count = len(records)
    original_claim_count = len(_group_indices(records))
    evaluable_mask = _evaluable_record_mask(records, full_gold)
    features = features[evaluable_mask]
    labels = labels[evaluable_mask]
    records = [row for row, keep in zip(records, evaluable_mask, strict=True) if keep]
    excluded_claims_without_gold = original_claim_count - len(_group_indices(records))
    grouped = _group_indices(records)
    retained_packet_metrics = _retained_packet_metrics(records, full_gold)
    modalities = [str(row["modality"]) for row in records]
    scorer_families = config.scorer_families

    seed_results: list[dict[str, object]] = []
    aggregate_inputs: dict[str, list[tuple[int, Sequence[dict[str, object]], Sequence[dict[str, object]]]]] = {
        method: []
        for method in ("threshold", "count", "cefes", "size_penalized", "rc_cefes")
    }
    for seed in config.seeds:
        seed_started = time.perf_counter()
        role_seed = config.partition_seed + seed
        fit_keys = _claim_keys_for_role(records, role_seed, "fit")
        calibration_keys = _claim_keys_for_role(records, role_seed, "calibration")
        model_selection_keys = _claim_keys_for_role(records, role_seed, "model_selection")
        holdout_keys = _claim_keys_for_role(records, role_seed, "internal_holdout")
        validation_keys = _claim_keys_for_role(records, role_seed, "validation")
        fit_indices = _indices_for_keys(grouped, fit_keys)
        calibration_indices = _indices_for_keys(grouped, calibration_keys)
        model_selection_indices = _indices_for_keys(grouped, model_selection_keys)
        holdout_indices = _indices_for_keys(grouped, holdout_keys)
        validation_indices = _indices_for_keys(grouped, validation_keys)

        family_rows: list[dict[str, object]] = []
        family_objects: dict[str, tuple[ScorerBundle, np.ndarray, np.ndarray, dict[str, object]]] = {}
        for family in scorer_families:
            scorer = _fit_scorer_family(
                family,
                features,
                labels,
                records,
                fit_indices,
                config=config,
            )
            raw = scorer.predict(features, modalities)
            calibrated, calibration_report = _calibrate_by_modality(
                raw,
                labels,
                modalities,
                calibration_indices,
                model_selection_indices,
                calibrator_kinds=config.calibrator_kinds,
            )
            selection_map = _mean_average_precision(
                records, calibrated, model_selection_indices, full_gold
            )
            raw_metrics = _calibration_metrics(raw[model_selection_indices], labels[model_selection_indices])
            calibrated_metrics = _calibration_metrics(
                calibrated[model_selection_indices], labels[model_selection_indices]
            )
            family_rows.append(
                {
                    "family": family,
                    "model_selection_map": selection_map,
                    "raw_calibration": raw_metrics,
                    "calibrated_calibration": calibrated_metrics,
                    "calibration_nonworse": calibrated_metrics["brier"] <= raw_metrics["brier"] + 1e-12,
                    "calibration_beats_prevalence": calibrated_metrics["brier_skill"] > 0.0,
                }
            )
            family_objects[family] = (scorer, raw, calibrated, calibration_report)
        eligible_families = [
            row
            for row in family_rows
            if row["calibration_nonworse"] and row["calibration_beats_prevalence"]
        ]
        selected_family_row = min(
            eligible_families or family_rows,
            key=lambda row: (-float(row["model_selection_map"]), str(row["family"])),
        )
        selected_family = str(selected_family_row["family"])
        scorer, raw_probabilities, probabilities, calibration_report = family_objects[selected_family]

        fixed_budget, fixed_selection_metrics = _best_parameter(
            config.fixed_budgets,
            lambda budget: _evaluate_method(
                "fixed", records, probabilities, full_gold, model_selection_keys,
                maximum_items=config.maximum_selected_items, fixed_k=int(budget)
            )[0],
        )
        threshold, threshold_selection_metrics = _best_parameter(
            config.thresholds,
            lambda value: _evaluate_method(
                "threshold", records, probabilities, full_gold, model_selection_keys,
                maximum_items=config.maximum_selected_items, threshold=float(value)
            )[0],
        )
        size_penalty, size_selection_metrics = _best_parameter(
            config.size_penalties,
            lambda value: _evaluate_method(
                "penalized_cefes", records, probabilities, full_gold, model_selection_keys,
                maximum_items=config.maximum_selected_items, size_penalty=float(value)
            )[0],
        )
        penalty_pairs = [
            (size_value, redundancy_value)
            for size_value in config.size_penalties
            for redundancy_value in config.redundancy_penalties
        ]
        rc_penalties, rc_selection_metrics = _best_parameter(
            penalty_pairs,
            lambda values: _evaluate_method(
                "penalized_cefes", records, probabilities, full_gold, model_selection_keys,
                maximum_items=config.maximum_selected_items,
                size_penalty=float(values[0]), redundancy_penalty=float(values[1])
            )[0],
        )
        count_model, count_predictions = _choose_count_predictions(
            records,
            probabilities,
            full_gold,
            fit_keys,
            [*model_selection_keys, *holdout_keys, *validation_keys],
            config=config,
        )

        method_parameters = {
            "fixed": {"fixed_k": int(fixed_budget)},
            "threshold": {"threshold": float(threshold)},
            "count": {},
            "cefes": {},
            "size_penalized": {"size_penalty": float(size_penalty)},
            "rc_cefes": {
                "size_penalty": float(rc_penalties[0]),
                "redundancy_penalty": float(rc_penalties[1]),
            },
        }

        def evaluate_split(keys: Sequence[str]) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, object]]]]:
            metrics: dict[str, dict[str, float]] = {}
            claim_rows: dict[str, list[dict[str, object]]] = {}
            for display, engine in (
                ("fixed", "fixed"),
                ("threshold", "threshold"),
                ("count", "count"),
                ("cefes", "cefes"),
                ("size_penalized", "penalized_cefes"),
                ("rc_cefes", "penalized_cefes"),
            ):
                params = method_parameters[display]
                metrics[display], claim_rows[display] = _evaluate_method(
                    engine,
                    records,
                    probabilities,
                    full_gold,
                    keys,
                    maximum_items=config.maximum_selected_items,
                    fixed_k=params.get("fixed_k"),
                    threshold=params.get("threshold"),
                    size_penalty=params.get("size_penalty", 0.0),
                    redundancy_penalty=params.get("redundancy_penalty", 0.0),
                    count_predictions=count_predictions,
                )
            return metrics, claim_rows

        holdout_metrics, holdout_rows = evaluate_split(holdout_keys)
        validation_metrics, validation_rows = evaluate_split(validation_keys)
        paired = {
            method: _paired_bootstrap(
                validation_rows[method],
                validation_rows["fixed"],
                samples=config.bootstrap_samples,
                seed=seed + index * 1009,
            )
            for index, method in enumerate(
                ("threshold", "count", "cefes", "size_penalized", "rc_cefes"), start=1
            )
        }
        for method in aggregate_inputs:
            aggregate_inputs[method].append(
                (seed, validation_rows[method], validation_rows["fixed"])
            )

        selected_sets = {
            method: {
                (str(row["claim_id"]), evidence_id)
                for row in validation_rows[method]
                for evidence_id in row["selected_evidence_ids"]
            }
            for method in validation_rows
        }
        candidate_output: list[dict[str, object]] = []
        for index in [*fit_indices, *calibration_indices, *model_selection_indices, *holdout_indices, *validation_indices]:
            row = records[int(index)]
            key = (str(row["claim_id"]), str(row["evidence_id"]))
            role = (
                "validation" if row["split"] == "validation" else _claim_role(str(row["claim_id"]), role_seed)
            )
            candidate_output.append(
                {
                    **row,
                    "seed": seed,
                    "role": role,
                    "raw_probability": float(raw_probabilities[index]),
                    "calibrated_probability": float(probabilities[index]),
                    **{
                        f"selected_{method}": key in selected_sets.get(method, set())
                        for method in validation_rows
                    },
                }
            )
        candidate_path = output_root / f"seed_{seed}_candidate_records.jsonl"
        claim_path = output_root / f"seed_{seed}_claim_records.jsonl"
        _write_jsonl(candidate_path, candidate_output)
        _write_jsonl(
            claim_path,
            (
                {**row, "seed": seed}
                for method in validation_rows
                for row in validation_rows[method]
            ),
        )
        validation_raw_calibration = _calibration_metrics(
            raw_probabilities[validation_indices], labels[validation_indices]
        )
        validation_calibrated_calibration = _calibration_metrics(
            probabilities[validation_indices], labels[validation_indices]
        )
        result: dict[str, object] = {
            "seed": seed,
            "role_seed": role_seed,
            "partition_claims": {
                "fit": len(fit_keys),
                "calibration": len(calibration_keys),
                "model_selection": len(model_selection_keys),
                "internal_holdout": len(holdout_keys),
                "validation": len(validation_keys),
            },
            "scorer_families": family_rows,
            "selected_scorer_family": selected_family,
            "selected_scorer": scorer.to_dict(),
            "calibration_report": calibration_report,
            "model_selection": {
                "fixed": fixed_selection_metrics,
                "threshold": threshold_selection_metrics,
                "size_penalized": size_selection_metrics,
                "rc_cefes": rc_selection_metrics,
            },
            "selected_parameters": method_parameters,
            "count_model": count_model.to_dict(),
            "internal_holdout": holdout_metrics,
            "validation": validation_metrics,
            "validation_calibration": {
                "raw": validation_raw_calibration,
                "calibrated": validation_calibrated_calibration,
            },
            "paired_vs_fixed": paired,
            "artifacts": {
                "candidates": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
                "claims": {"path": str(claim_path), "sha256": sha256_file(claim_path)},
            },
            "runtime": {"wall_seconds": time.perf_counter() - seed_started},
        }
        seed_path = output_root / f"seed_{seed}.json"
        seed_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        seed_results.append(result)

    aggregate = {
        method: _aggregate_seed_aligned_bootstrap(
            rows,
            samples=config.bootstrap_samples,
            seed=91_000 + index,
        )
        for index, (method, rows) in enumerate(aggregate_inputs.items())
    }
    baseline_recall = {
        "text": config.baseline_text_recall,
        "image": config.baseline_image_recall,
    }
    retrieval_metrics = retrieval_manifest["retrieval_metrics"]
    text_recall = float(
        retrieval_metrics["text"]["dense_bm25_rrf_cross_encoder"]["recall_at_50"]
    )
    image_recall = float(
        retrieval_metrics["image"]["siglip_filename_rrf"]["recall_at_50"]
    )
    retrieval_gate = {
        "text_gain_at_least_0_10": text_recall - baseline_recall["text"] >= 0.10,
        "image_gain_at_least_0_10": image_recall - baseline_recall["image"] >= 0.10,
        "text_recall_at_50": text_recall,
        "image_recall_at_50": image_recall,
        "text_gain": text_recall - baseline_recall["text"],
        "image_gain": image_recall - baseline_recall["image"],
    }
    selected_maps = {
        int(row["seed"]): {
            str(value["family"]): float(value["model_selection_map"])
            for value in row["scorer_families"]
        }
        for row in seed_results
    }
    shared_baselines = [
        family for family in scorer_families if family.startswith("shared_logistic")
    ]
    scorer_gate = {
        "positive_map_effect_each_seed": all(
            selected_maps[int(row["seed"])][str(row["selected_scorer_family"])]
            > max(
                selected_maps[int(row["seed"])][family]
                for family in shared_baselines
            )
            + 1e-12
            for row in seed_results
        ),
        "calibrated_brier_nonworse_each_seed": all(
            float(row["validation_calibration"]["calibrated"]["brier"])
            <= float(row["validation_calibration"]["raw"]["brier"]) + 1e-12
            for row in seed_results
        ),
        "calibrated_brier_better_than_prevalence_each_seed": all(
            float(row["validation_calibration"]["calibrated"]["brier_skill"]) > 0.0
            for row in seed_results
        ),
    }
    rc_gains = [
        float(row["validation"]["rc_cefes"]["f1"])
        - float(row["validation"]["fixed"]["f1"])
        for row in seed_results
    ]
    mean_selected_ratio = fmean(
        float(row["validation"]["rc_cefes"]["selected_count"])
        / max(
            1e-12,
            fmean(
                len(full_gold[key])
                for key in _claim_keys_for_role(records, int(row["role_seed"]), "validation")
            ),
        )
        for row in seed_results
    )
    superiority_gate = {
        "retrieval_gate_pass": all(
            value for key, value in retrieval_gate.items() if key.endswith("at_least_0_10")
        ),
        "scorer_gate_pass": all(scorer_gate.values()),
        "mean_gain_at_least_0_015": fmean(rc_gains) >= 0.015,
        "positive_each_seed": all(value > 0.0 for value in rc_gains),
        "aggregate_interval_lower_positive": float(aggregate["rc_cefes"]["lower"]) > 0.0,
        "calibrated_brier_nonworse_each_seed": scorer_gate["calibrated_brier_nonworse_each_seed"],
        "mean_selected_to_gold_ratio_no_more_than_2": mean_selected_ratio <= 2.0,
        "verifier_noninferiority_pending": True,
        "clean_reproduction_pending": True,
    }
    summary: dict[str, object] = {
        "profile": "q1_phase2_matched_selector_study",
        "status": "development_complete_verifier_and_reproduction_pending",
        "config": asdict(config),
        "retrieval_manifest_sha256": sha256_file(retrieval_manifest_path),
        "retrieval_audit_sha256": sha256_file(retrieval_audit_path),
        "protocol_sha256": sha256_file(protocol_path),
        "evaluation_population": {
            "candidate_rows_before_complete_gold_filter": original_record_count,
            "candidate_rows": len(records),
            "claims_before_complete_gold_filter": original_claim_count,
            "claims": len(grouped),
            "claims_excluded_without_released_gold": excluded_claims_without_gold,
            "policy": "exclude claims without a released non-empty complete-gold evidence set",
        },
        "retained_packet_metrics": retained_packet_metrics,
        "seeds": seed_results,
        "aggregate_seed_aligned_vs_fixed": aggregate,
        "retrieval_gate": retrieval_gate,
        "scorer_gate": scorer_gate,
        "mean_rc_cefes_minus_fixed_f1": fmean(rc_gains),
        "mean_selected_to_gold_ratio": mean_selected_ratio,
        "superiority_gate": superiority_gate,
        "selector_superiority_gate_pass": False,
        "reason_gate_not_final": "shared-verifier non-inferiority and clean reproduction are pending",
        "formal_test_unlocked": False,
        "formal_test_annotations_accessed": False,
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def audit_phase2_selector_study(output_root: Path) -> dict[str, object]:
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if summary.get("formal_test_unlocked") or summary.get("formal_test_annotations_accessed"):
        errors.append("formal-test seal is not intact")
    for seed_row in summary.get("seeds", []):
        for artifact in seed_row.get("artifacts", {}).values():
            path = Path(str(artifact["path"]))
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                errors.append(f"artifact mismatch: {path}")
        validation = seed_row["validation"]
        stored = seed_row["paired_vs_fixed"]["rc_cefes"]["mean"]
        recomputed = float(validation["rc_cefes"]["f1"]) - float(validation["fixed"]["f1"])
        if not math.isclose(stored, recomputed, abs_tol=1e-12):
            errors.append(f"paired mean mismatch for seed {seed_row['seed']}")
    audit = {
        "profile": "q1_phase2_selector_independent_audit",
        "status": "pass" if not errors else "failed",
        "errors": errors,
        "summary_sha256": sha256_file(summary_path),
        "formal_test_annotations_accessed": False,
    }
    path = output_root / "audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError(f"Phase-2 selector audit failed: {errors}")
    return audit


