from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import numpy as np

from trace_scv.cefes import select_cefes_prefix
from trace_scv.data.io import sha256_file
from trace_scv.selection.calibration import (
    BinaryTemperatureScaler,
    binary_brier_score,
    binary_expected_calibration_error,
    binary_nll,
)


@dataclass(frozen=True, slots=True)
class MochegSelectorConfig:
    seeds: tuple[int, ...] = (13, 37, 73)
    calibration_fraction: float = 0.2
    fixed_budgets: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 60)
    maximum_text_candidates: int = 40
    maximum_image_candidates: int = 20
    maximum_selected_items: int = 60
    learning_rate: float = 0.08
    epochs: int = 2500
    l2: float = 1e-3
    bootstrap_samples: int = 10_000
    minimum_mean_gain: float = 0.015
    calibration_safety_guard: bool = False
    calibration_guard_fraction: float = 0.5


@dataclass(frozen=True, slots=True)
class LogisticEvidenceScorer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
        epochs: int,
        l2: float,
    ) -> "LogisticEvidenceScorer":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(labels) or not len(labels):
            raise ValueError("features and labels must be non-empty and aligned")
        if set(np.unique(labels)) != {0.0, 1.0}:
            raise ValueError("scorer training requires both binary classes")
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        values = (features - mean) / scale
        weights = np.zeros(values.shape[1], dtype=np.float64)
        bias = 0.0
        positives = float(labels.sum())
        negatives = float(len(labels) - positives)
        sample_weight = np.where(
            labels > 0.5,
            len(labels) / (2.0 * positives),
            len(labels) / (2.0 * negatives),
        )
        denominator = float(sample_weight.sum())
        for epoch in range(epochs):
            logits = np.clip(values @ weights + bias, -40.0, 40.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            residual = (probabilities - labels) * sample_weight
            gradient = values.T @ residual / denominator + l2 * weights
            bias_gradient = float(residual.sum() / denominator)
            step = learning_rate / math.sqrt(1.0 + epoch / 250.0)
            weights -= step * gradient
            bias -= step * bias_gradient
        return cls(
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
            weights=tuple(float(value) for value in weights),
            bias=float(bias),
        )

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        mean = np.asarray(self.mean)
        scale = np.asarray(self.scale)
        weights = np.asarray(self.weights)
        logits = np.clip(((values - mean) / scale) @ weights + self.bias, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)


def _claim_partition(claim_ids: Iterable[str], seed: int, fraction: float) -> set[str]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration fraction must lie in (0, 1)")
    unique = sorted(set(claim_ids))
    if len(unique) < 2:
        raise ValueError("at least two training claims are required")
    ordered = sorted(
        unique,
        key=lambda claim_id: hashlib.sha256(
            f"{seed}:{claim_id}".encode("utf-8")
        ).hexdigest(),
    )
    count = min(len(unique) - 1, max(1, round(len(unique) * fraction)))
    return set(ordered[:count])


def _bounded_candidate_indices(
    records: Sequence[dict[str, object]],
    features: np.ndarray,
    *,
    maximum_text_candidates: int,
    maximum_image_candidates: int,
) -> list[int]:
    """Keep the highest semantic-similarity candidates without using gold labels."""
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(str(record["claim_id"]), []).append(index)
    selected: list[int] = []
    limits = {"text": maximum_text_candidates, "image": maximum_image_candidates}
    for claim_id in sorted(grouped):
        for modality, limit in limits.items():
            candidates = [
                index
                for index in grouped[claim_id]
                if records[index]["modality"] == modality
            ]
            candidates.sort(key=lambda index: (-float(features[index, 0]), index))
            selected.extend(candidates[:limit])
    return sorted(selected)


def _set_scores(predicted: set[str], gold: set[str]) -> dict[str, float]:
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    denominator = len(predicted) + len(gold)
    f1 = 2.0 * overlap / denominator if denominator else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": float(predicted == gold),
        "jaccard": (
            overlap / len(predicted | gold) if predicted or gold else 1.0
        ),
    }


def _evaluate_claims(
    records: Sequence[dict[str, object]],
    probabilities: Sequence[float],
    *,
    method: str,
    fixed_k: int | None = None,
    maximum_selected_items: int = 10,
    full_gold_evidence_ids_by_claim: dict[str, set[str]] | None = None,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(str(record["claim_id"]), []).append(index)
    claim_rows: list[dict[str, object]] = []
    for claim_id in sorted(grouped):
        indices = grouped[claim_id]
        row_probabilities = [float(probabilities[index]) for index in indices]
        order = sorted(
            range(len(indices)), key=lambda index: (-row_probabilities[index], index)
        )
        if method == "fixed":
            if fixed_k is None or fixed_k <= 0:
                raise ValueError("fixed selection requires a positive k")
            selected_local = order[:fixed_k]
        elif method == "threshold":
            selected_local = [index for index in order if row_probabilities[index] >= 0.5]
            if not selected_local:
                selected_local = order[:1]
            selected_local = selected_local[:maximum_selected_items]
        elif method == "cefes":
            decision = select_cefes_prefix(
                row_probabilities,
                allow_empty=False,
                max_items=maximum_selected_items,
            )
            selected_local = list(decision.selected_indices)
        else:
            raise ValueError(f"unknown selection method: {method}")
        evidence_ids = [str(records[index]["evidence_id"]) for index in indices]
        selected = {evidence_ids[index] for index in selected_local}
        gold = (
            set(full_gold_evidence_ids_by_claim[claim_id])
            if full_gold_evidence_ids_by_claim is not None
            else {
                evidence_ids[index]
                for index, global_index in enumerate(indices)
                if bool(records[global_index]["gold"])
            }
        )
        scores = _set_scores(selected, gold)
        claim_rows.append(
            {
                "claim_id": claim_id,
                "selected_evidence_ids": sorted(selected),
                "gold_evidence_ids": sorted(gold),
                "selected_count": len(selected),
                **scores,
            }
        )
    metrics = {
        key: fmean(float(row[key]) for row in claim_rows) if claim_rows else 0.0
        for key in ("precision", "recall", "f1", "exact", "jaccard")
    }
    metrics["mean_selected"] = (
        fmean(float(row["selected_count"]) for row in claim_rows)
        if claim_rows
        else 0.0
    )
    metrics["claims"] = float(len(claim_rows))
    return metrics, claim_rows


def _paired_bootstrap(
    left: Sequence[dict[str, object]],
    right: Sequence[dict[str, object]],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    right_by_id = {str(row["claim_id"]): row for row in right}
    differences = np.asarray(
        [
            float(row["f1"]) - float(right_by_id[str(row["claim_id"])]["f1"])
            for row in left
        ],
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
        "resampling_unit": "claim",
        "samples": values,
    }


def _rank_by_claim(
    records: Sequence[dict[str, object]], probabilities: Sequence[float]
) -> dict[int, int]:
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(str(record["claim_id"]), []).append(index)
    ranks: dict[int, int] = {}
    for indices in grouped.values():
        order = sorted(indices, key=lambda index: (-float(probabilities[index]), index))
        for rank, index in enumerate(order, start=1):
            ranks[index] = rank
    return ranks


def run_mocheg_selector_pilot(
    feature_path: Path,
    output_dir: Path,
    *,
    config: MochegSelectorConfig = MochegSelectorConfig(),
) -> dict[str, object]:
    run_started = time.perf_counter()
    arrays = np.load(feature_path)
    features = np.asarray(arrays["features"], dtype=np.float64)
    labels = np.asarray(arrays["gold"], dtype=np.int8)
    sidecar_path = feature_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    feature_manifest_path = feature_path.with_suffix(".manifest.json")
    feature_manifest = json.loads(
        feature_manifest_path.read_text(encoding="utf-8")
    )
    scientific_features = bool(
        feature_manifest.get("scientific_results_allowed", True)
    )
    records = list(sidecar["records"])
    full_gold_evidence_ids_by_claim = (
        {
            str(claim_id): {str(evidence_id) for evidence_id in evidence_ids}
            for claim_id, evidence_ids in sidecar[
                "full_gold_evidence_ids_by_claim"
            ].items()
        }
        if "full_gold_evidence_ids_by_claim" in sidecar
        else None
    )
    if len(records) != len(features) or len(labels) != len(features):
        raise ValueError("feature arrays and sidecar records are misaligned")
    bounded_indices = set(
        _bounded_candidate_indices(
            records,
            features,
            maximum_text_candidates=config.maximum_text_candidates,
            maximum_image_candidates=config.maximum_image_candidates,
        )
    )
    train_indices = [
        index
        for index, row in enumerate(records)
        if row["split"] == "train" and index in bounded_indices
    ]
    validation_indices = [
        index
        for index, row in enumerate(records)
        if row["split"] == "validation" and index in bounded_indices
    ]
    if not train_indices or not validation_indices:
        raise ValueError("both train and validation feature rows are required")

    seed_results: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in config.seeds:
        seed_started = time.perf_counter()
        calibration_claims = _claim_partition(
            (str(records[index]["claim_id"]) for index in train_indices),
            seed,
            config.calibration_fraction,
        )
        fit_indices = [
            index
            for index in train_indices
            if str(records[index]["claim_id"]) not in calibration_claims
        ]
        calibration_indices = [
            index
            for index in train_indices
            if str(records[index]["claim_id"]) in calibration_claims
        ]
        scorer = LogisticEvidenceScorer.fit(
            features[fit_indices],
            labels[fit_indices],
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            l2=config.l2,
        )
        raw_probabilities = scorer.predict_probabilities(features)
        calibration_guard: dict[str, object] | None = None
        if config.calibration_safety_guard:
            guard_claims = _claim_partition(
                calibration_claims,
                seed + 1_000_003,
                config.calibration_guard_fraction,
            )
            temperature_fit_indices = [
                index
                for index in calibration_indices
                if str(records[index]["claim_id"]) not in guard_claims
            ]
            guard_indices = [
                index
                for index in calibration_indices
                if str(records[index]["claim_id"]) in guard_claims
            ]
            fitted_scaler = BinaryTemperatureScaler.fit(
                raw_probabilities[temperature_fit_indices],
                labels[temperature_fit_indices],
            )
            guard_raw = raw_probabilities[guard_indices]
            guard_scaled = np.asarray(
                fitted_scaler.transform(guard_raw), dtype=np.float64
            )
            guard_labels = labels[guard_indices]
            raw_guard_brier = binary_brier_score(guard_raw, guard_labels)
            scaled_guard_brier = binary_brier_score(guard_scaled, guard_labels)
            temperature_retained = scaled_guard_brier <= raw_guard_brier + 1e-12
            scaler = (
                fitted_scaler
                if temperature_retained
                else BinaryTemperatureScaler(
                    temperature=1.0,
                    fitted_examples=len(guard_indices),
                    base_nll=binary_nll(guard_raw, guard_labels),
                    calibrated_nll=binary_nll(guard_raw, guard_labels),
                )
            )
            calibration_guard = {
                "strategy": "claim_grouped_temperature_or_identity_by_guard_brier",
                "temperature_fit_claims": len(calibration_claims - guard_claims),
                "guard_claims": len(guard_claims),
                "temperature_fit_examples": len(temperature_fit_indices),
                "guard_examples": len(guard_indices),
                "candidate_temperature": fitted_scaler.temperature,
                "guard_raw_brier": raw_guard_brier,
                "guard_temperature_brier": scaled_guard_brier,
                "temperature_retained": temperature_retained,
                "applied_temperature": scaler.temperature,
            }
        else:
            scaler = BinaryTemperatureScaler.fit(
                raw_probabilities[calibration_indices], labels[calibration_indices]
            )
        calibrated_probabilities = np.asarray(
            scaler.transform(raw_probabilities), dtype=np.float64
        )

        calibration_records = [records[index] for index in calibration_indices]
        calibration_values = calibrated_probabilities[calibration_indices]
        fixed_results: dict[int, dict[str, float]] = {}
        for budget in config.fixed_budgets:
            fixed_results[budget] = _evaluate_claims(
                calibration_records,
                calibration_values,
                method="fixed",
                fixed_k=budget,
                maximum_selected_items=config.maximum_selected_items,
                full_gold_evidence_ids_by_claim=full_gold_evidence_ids_by_claim,
            )[0]
        selected_budget = min(
            config.fixed_budgets,
            key=lambda budget: (-fixed_results[budget]["f1"], budget),
        )

        validation_records = [records[index] for index in validation_indices]
        validation_raw = raw_probabilities[validation_indices]
        validation_calibrated = calibrated_probabilities[validation_indices]
        fixed_metrics, fixed_claims = _evaluate_claims(
            validation_records,
            validation_calibrated,
            method="fixed",
            fixed_k=selected_budget,
            maximum_selected_items=config.maximum_selected_items,
            full_gold_evidence_ids_by_claim=full_gold_evidence_ids_by_claim,
        )
        cefes_metrics, cefes_claims = _evaluate_claims(
            validation_records,
            validation_calibrated,
            method="cefes",
            maximum_selected_items=config.maximum_selected_items,
            full_gold_evidence_ids_by_claim=full_gold_evidence_ids_by_claim,
        )
        raw_cefes_metrics, raw_cefes_claims = _evaluate_claims(
            validation_records,
            validation_raw,
            method="cefes",
            maximum_selected_items=config.maximum_selected_items,
            full_gold_evidence_ids_by_claim=full_gold_evidence_ids_by_claim,
        )
        threshold_metrics, threshold_claims = _evaluate_claims(
            validation_records,
            validation_calibrated,
            method="threshold",
            maximum_selected_items=config.maximum_selected_items,
            full_gold_evidence_ids_by_claim=full_gold_evidence_ids_by_claim,
        )
        calibration_metrics = {
            "raw": {
                "brier": binary_brier_score(validation_raw, labels[validation_indices]),
                "nll": binary_nll(validation_raw, labels[validation_indices]),
                "ece": binary_expected_calibration_error(
                    validation_raw, labels[validation_indices]
                ),
            },
            "calibrated": {
                "brier": binary_brier_score(
                    validation_calibrated, labels[validation_indices]
                ),
                "nll": binary_nll(
                    validation_calibrated, labels[validation_indices]
                ),
                "ece": binary_expected_calibration_error(
                    validation_calibrated, labels[validation_indices]
                ),
            },
        }
        paired = _paired_bootstrap(
            cefes_claims,
            fixed_claims,
            samples=config.bootstrap_samples,
            seed=seed,
        )
        result = {
            "seed": seed,
            "fit_claims": len(
                {str(records[index]["claim_id"]) for index in fit_indices}
            ),
            "calibration_claims": len(calibration_claims),
            "selected_fixed_budget": selected_budget,
            "fixed_budget_calibration_metrics": fixed_results,
            "temperature_scaler": scaler.to_dict(),
            "calibration_guard": calibration_guard,
            "scorer": asdict(scorer),
            "validation": {
                "cefes": cefes_metrics,
                "fixed": fixed_metrics,
                "uncalibrated_cefes": raw_cefes_metrics,
                "threshold_0_5": threshold_metrics,
                "calibration": calibration_metrics,
                "cefes_minus_fixed_bootstrap": paired,
            },
            "runtime": {"wall_seconds": time.perf_counter() - seed_started},
        }
        claim_records_path = output_dir / f"seed_{seed}_claim_records.jsonl"
        with claim_records_path.open(
            "w", encoding="utf-8"
        ) as handle:
            fixed_by_id = {str(row["claim_id"]): row for row in fixed_claims}
            for row in cefes_claims:
                claim_id = str(row["claim_id"])
                handle.write(
                    json.dumps(
                        {
                            **row,
                            "fixed_f1": fixed_by_id[claim_id]["f1"],
                            "cefes_minus_fixed_f1": (
                                float(row["f1"]) - float(fixed_by_id[claim_id]["f1"])
                            ),
                        }
                    )
                    + "\n"
                )

        validation_rank = _rank_by_claim(validation_records, validation_calibrated)
        validation_positions = {
            global_index: local_index
            for local_index, global_index in enumerate(validation_indices)
        }
        cefes_selected = {
            (str(row["claim_id"]), str(evidence_id))
            for row in cefes_claims
            for evidence_id in row["selected_evidence_ids"]
        }
        fixed_selected = {
            (str(row["claim_id"]), str(evidence_id))
            for row in fixed_claims
            for evidence_id in row["selected_evidence_ids"]
        }
        raw_cefes_selected = {
            (str(row["claim_id"]), str(evidence_id))
            for row in raw_cefes_claims
            for evidence_id in row["selected_evidence_ids"]
        }
        threshold_selected = {
            (str(row["claim_id"]), str(evidence_id))
            for row in threshold_claims
            for evidence_id in row["selected_evidence_ids"]
        }
        candidate_records_path = output_dir / f"seed_{seed}_candidate_records.jsonl"
        with candidate_records_path.open("w", encoding="utf-8") as handle:
            for global_index in [*train_indices, *validation_indices]:
                record = records[global_index]
                claim_id = str(record["claim_id"])
                evidence_id = str(record["evidence_id"])
                split = str(record["split"])
                role = (
                    "validation"
                    if split == "validation"
                    else "calibration"
                    if claim_id in calibration_claims
                    else "fit"
                )
                validation_local_index = (
                    validation_positions.get(global_index)
                )
                key = (claim_id, evidence_id)
                handle.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "role": role,
                            **record,
                            "raw_probability": float(raw_probabilities[global_index]),
                            "calibrated_probability": float(
                                calibrated_probabilities[global_index]
                            ),
                            "calibrated_rank": (
                                validation_rank[validation_local_index]
                                if validation_local_index is not None
                                else None
                            ),
                            "selected_cefes": key in cefes_selected,
                            "selected_fixed": key in fixed_selected,
                            "selected_uncalibrated_cefes": key in raw_cefes_selected,
                            "selected_threshold_0_5": key in threshold_selected,
                        }
                    )
                    + "\n"
                )
        result["artifacts"] = {
            "claim_records": {
                "path": str(claim_records_path),
                "bytes": claim_records_path.stat().st_size,
                "sha256": sha256_file(claim_records_path),
            },
            "candidate_records": {
                "path": str(candidate_records_path),
                "bytes": candidate_records_path.stat().st_size,
                "sha256": sha256_file(candidate_records_path),
            },
        }
        seed_results.append(result)
        (output_dir / f"seed_{seed}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    gains = [
        float(row["validation"]["cefes"]["f1"])
        - float(row["validation"]["fixed"]["f1"])
        for row in seed_results
    ]
    brier_not_worse = all(
        float(row["validation"]["calibration"]["calibrated"]["brier"])
        <= float(row["validation"]["calibration"]["raw"]["brier"]) + 1e-12
        for row in seed_results
    )
    selector_criteria_met = (
        fmean(gains) >= config.minimum_mean_gain
        and all(gain > 0.0 for gain in gains)
        and brier_not_worse
    )
    selector_gate = scientific_features and selector_criteria_met
    implementation_path = Path(__file__).resolve()
    package_root = implementation_path.parents[1]
    config_path = Path.cwd() / "configs" / "mocheg_development.yaml"
    implementation_files = {
        str(path): sha256_file(path)
        for path in (
            implementation_path,
            implementation_path.parent / "calibration.py",
            package_root / "cefes.py",
        )
    }
    summary: dict[str, object] = {
        "profile": (
            "mocheg_selector_development_pilot"
            if scientific_features
            else "mocheg_selector_software_smoke"
        ),
        "config": asdict(config),
        "feature_sha256": sha256_file(feature_path),
        "feature_sidecar_sha256": sha256_file(sidecar_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "implementation_files": implementation_files,
        "development_config": (
            {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            }
            if config_path.is_file()
            else None
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "runtime": {"wall_seconds": time.perf_counter() - run_started},
        "scientific_feature_backend": scientific_features,
        "bounded_candidate_rows": {
            "train": len(train_indices),
            "validation": len(validation_indices),
        },
        "seeds": seed_results,
        "mean_cefes_f1": fmean(
            float(row["validation"]["cefes"]["f1"]) for row in seed_results
        ),
        "mean_fixed_f1": fmean(
            float(row["validation"]["fixed"]["f1"]) for row in seed_results
        ),
        "mean_cefes_minus_fixed_f1": fmean(gains),
        "positive_effect_each_seed": all(gain > 0.0 for gain in gains),
        "calibrated_brier_not_worse_each_seed": brier_not_worse,
        "selector_numeric_criteria_met": selector_criteria_met,
        "selector_development_gate_pass": selector_gate,
        "full_development_gate_pass": False,
        "reason_full_gate_not_passed": (
            "full gate requires external audited aggregation of selector, verifier, "
            "full-corpus retrieval, and clean-rerun artifacts"
        ),
        "scientific_results_validated": False,
        "formal_test_annotations_accessed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
