from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

import numpy as np

from trace_scv.data.io import sha256_file


@dataclass(frozen=True, slots=True)
class DacsCountConfig:
    maximum_count: int = 60
    learning_rate: float = 0.05
    epochs: int = 3000
    l2: float = 1e-3
    bootstrap_samples: int = 10_000


@dataclass(frozen=True, slots=True)
class DacsCountClassifier:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    observed_counts: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        counts: Sequence[int],
        *,
        config: DacsCountConfig,
    ) -> "DacsCountClassifier":
        values = np.asarray(features, dtype=np.float64)
        targets = np.asarray(counts, dtype=np.int64) - 1
        if values.ndim != 2 or not len(values) or len(values) != len(targets):
            raise ValueError("DACS features and count targets must be aligned")
        if np.any(targets < 0) or np.any(targets >= config.maximum_count):
            raise ValueError("DACS count target lies outside the frozen range")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (values - mean) / scale
        weights = np.zeros((values.shape[1], config.maximum_count), dtype=np.float64)
        bias = np.full(config.maximum_count, -20.0, dtype=np.float64)
        observed = np.unique(targets)
        bias[observed] = 0.0
        class_counts = np.bincount(targets, minlength=config.maximum_count).astype(np.float64)
        sample_weight = len(targets) / (len(observed) * class_counts[targets])
        denominator = float(sample_weight.sum())
        one_hot = np.eye(config.maximum_count, dtype=np.float64)[targets]
        for epoch in range(config.epochs):
            logits = standardized @ weights + bias
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(np.clip(logits, -40.0, 40.0))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            residual = (probabilities - one_hot) * sample_weight[:, None]
            gradient = standardized.T @ residual / denominator + config.l2 * weights
            bias_gradient = residual.sum(axis=0) / denominator
            step = config.learning_rate / math.sqrt(1.0 + epoch / 300.0)
            weights -= step * gradient
            bias -= step * bias_gradient
        return cls(
            mean=tuple(map(float, mean)),
            scale=tuple(map(float, scale)),
            weights=tuple(tuple(map(float, row)) for row in weights),
            bias=tuple(map(float, bias)),
            observed_counts=tuple(int(index + 1) for index in observed),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features, dtype=np.float64) - np.asarray(self.mean)) / np.asarray(self.scale)
        return (values @ np.asarray(self.weights) + np.asarray(self.bias)).argmax(axis=1) + 1


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _profile(rows: Sequence[dict[str, object]], maximum_count: int) -> np.ndarray:
    ordered = sorted(rows, key=lambda row: (-float(row["calibrated_probability"]), str(row["evidence_id"])))
    calibrated = np.asarray([float(row["calibrated_probability"]) for row in ordered], dtype=np.float64)
    raw = np.asarray([float(row["raw_probability"]) for row in ordered], dtype=np.float64)
    def padded(values: np.ndarray) -> np.ndarray:
        result = np.zeros(maximum_count, dtype=np.float64)
        result[: min(maximum_count, len(values))] = values[:maximum_count]
        return result
    return np.concatenate(
        [
            padded(calibrated),
            padded(raw),
            np.asarray(
                [
                    calibrated.mean(), calibrated.max(), calibrated.std(),
                    raw.mean(), raw.max(), raw.std(),
                    len(rows) / maximum_count,
                    sum(str(row["modality"]) == "text" for row in rows) / len(rows),
                    sum(str(row["modality"]) == "image" for row in rows) / len(rows),
                ],
                dtype=np.float64,
            ),
        ]
    )


def _set_f1(predicted: set[str], gold: set[str]) -> float:
    denominator = len(predicted) + len(gold)
    return 2.0 * len(predicted & gold) / denominator if denominator else 1.0


def run_mocheg_dacs_count_control(
    feature_path: Path,
    selector_dir: Path,
    output_dir: Path,
    *,
    config: DacsCountConfig = DacsCountConfig(),
) -> dict[str, object]:
    started = time.perf_counter()
    selector_summary_path = selector_dir / "summary.json"
    selector_summary = json.loads(selector_summary_path.read_text(encoding="utf-8"))
    sidecar = json.loads(feature_path.with_suffix(".json").read_text(encoding="utf-8"))
    full_gold = {
        str(claim_id): set(map(str, ids))
        for claim_id, ids in sidecar.get("full_gold_evidence_ids_by_claim", {}).items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, object]] = []
    for seed_row in selector_summary["seeds"]:
        seed = int(seed_row["seed"])
        candidates = _read_jsonl(selector_dir / f"seed_{seed}_candidate_records.jsonl")
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in candidates:
            grouped.setdefault((str(row["role"]), str(row["claim_id"])), []).append(row)
        train_keys = sorted(key for key in grouped if key[0] == "fit")
        validation_keys = sorted(key for key in grouped if key[0] == "validation")
        train_profiles = np.stack([_profile(grouped[key], config.maximum_count) for key in train_keys])
        train_counts = [
            min(
                config.maximum_count,
                max(
                    1,
                    len(full_gold.get(key[1], {str(row["evidence_id"]) for row in grouped[key] if row["gold"]})),
                ),
            )
            for key in train_keys
        ]
        model = DacsCountClassifier.fit(train_profiles, train_counts, config=config)
        checkpoint_path = output_dir / f"seed_{seed}_dacs_count.json"
        checkpoint_path.write_text(json.dumps({"config": asdict(config), "model": asdict(model)}, indent=2) + "\n", encoding="utf-8")
        validation_profiles = np.stack([_profile(grouped[key], config.maximum_count) for key in validation_keys])
        predicted_counts = model.predict(validation_profiles)
        records: list[dict[str, object]] = []
        differences: list[float] = []
        count_errors: list[float] = []
        exact_counts: list[float] = []
        for key, predicted_count in zip(validation_keys, predicted_counts, strict=True):
            rows = grouped[key]
            ordered = sorted(rows, key=lambda row: (-float(row["calibrated_probability"]), str(row["evidence_id"])))
            selected = {str(row["evidence_id"]) for row in ordered[: int(predicted_count)]}
            gold = full_gold.get(key[1], {str(row["evidence_id"]) for row in rows if row["gold"]})
            cefes = {str(row["evidence_id"]) for row in rows if row["selected_cefes"]}
            dacs_f1 = _set_f1(selected, gold)
            cefes_f1 = _set_f1(cefes, gold)
            differences.append(cefes_f1 - dacs_f1)
            count_errors.append(abs(len(gold) - int(predicted_count)))
            exact_counts.append(float(len(gold) == int(predicted_count)))
            records.append(
                {
                    "seed": seed,
                    "claim_id": key[1],
                    "predicted_count": int(predicted_count),
                    "gold_count": len(gold),
                    "selected_evidence_ids": sorted(selected),
                    "gold_evidence_ids": sorted(gold),
                    "dacs_f1": dacs_f1,
                    "cefes_f1": cefes_f1,
                    "cefes_minus_dacs_f1": cefes_f1 - dacs_f1,
                }
            )
        generator = np.random.default_rng(seed)
        difference_array = np.asarray(differences, dtype=np.float64)
        samples = [
            float(difference_array[generator.integers(0, len(difference_array), size=len(difference_array))].mean())
            for _ in range(config.bootstrap_samples)
        ]
        records_path = output_dir / f"seed_{seed}_dacs_records.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
        seed_results.append(
            {
                "seed": seed,
                "training_claims": len(train_keys),
                "validation_claims": len(validation_keys),
                "observed_training_counts": list(model.observed_counts),
                "dacs_evidence_f1": fmean(float(row["dacs_f1"]) for row in records),
                "cefes_evidence_f1": fmean(float(row["cefes_f1"]) for row in records),
                "count_mae": fmean(count_errors),
                "count_exact_match": fmean(exact_counts),
                "cefes_minus_dacs_bootstrap": {
                    "mean": fmean(differences),
                    "lower": float(np.quantile(samples, 0.025)),
                    "median": float(np.quantile(samples, 0.5)),
                    "upper": float(np.quantile(samples, 0.975)),
                    "sample_count": config.bootstrap_samples,
                    "resampling_seed": seed,
                    "resampling_unit": "claim",
                    "samples": samples,
                },
                "checkpoint": {"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
                "records": {"path": str(records_path), "bytes": records_path.stat().st_size, "sha256": sha256_file(records_path)},
            }
        )
    result = {
        "profile": "mocheg_dacs_count_classification_control",
        "config": asdict(config),
        "feature_sha256": sha256_file(feature_path),
        "selector_summary_sha256": sha256_file(selector_summary_path),
        "seeds": seed_results,
        "mean_dacs_evidence_f1": fmean(float(row["dacs_evidence_f1"]) for row in seed_results),
        "mean_cefes_minus_dacs_f1": fmean(float(row["cefes_minus_dacs_bootstrap"]["mean"]) for row in seed_results),
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "formal_test_annotations_accessed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
