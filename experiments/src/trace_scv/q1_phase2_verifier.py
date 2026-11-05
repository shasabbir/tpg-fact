from __future__ import annotations

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

from trace_scv.data.io import sha256_file
from trace_scv.q1_phase2_study import _select_local


VERDICT_LABELS = ("nei", "refuted", "supported")


@dataclass(frozen=True, slots=True)
class Phase2VerifierConfig:
    epochs: int = 500
    learning_rate: float = 0.05
    l2: float = 1e-3
    bootstrap_samples: int = 10_000
    noninferiority_margin: float = -0.01


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(frozen=True, slots=True)
class VerifierModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        values = (features - np.asarray(self.mean)) / np.asarray(self.scale)
        logits = values @ np.asarray(self.weights) + np.asarray(self.bias)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(np.clip(logits, -40.0, 40.0))
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def _fit_verifier(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    config: Phase2VerifierConfig,
) -> VerifierModel:
    import torch

    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray([VERDICT_LABELS.index(label) for label in labels], dtype=np.int64)
    if set(np.unique(targets)) != set(range(len(VERDICT_LABELS))):
        raise ValueError("verifier training requires all verdict classes")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (values - mean) / scale
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(standardized, device=device)
    y = torch.as_tensor(targets, device=device)
    weights = torch.zeros(
        (values.shape[1], len(VERDICT_LABELS)), device=device, requires_grad=True
    )
    bias = torch.zeros(len(VERDICT_LABELS), device=device, requires_grad=True)
    counts = np.bincount(targets, minlength=len(VERDICT_LABELS)).astype(np.float32)
    class_weights = torch.as_tensor(len(targets) / (len(VERDICT_LABELS) * counts), device=device)
    optimizer = torch.optim.Adam([weights, bias], lr=config.learning_rate)
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weights + bias
        loss = torch.nn.functional.cross_entropy(logits, y, weight=class_weights)
        loss = loss + config.l2 * weights.square().mean()
        loss.backward()
        optimizer.step()
    result = VerifierModel(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(tuple(float(value) for value in row) for row in weights.detach().cpu().numpy()),
        bias=tuple(float(value) for value in bias.detach().cpu().numpy()),
    )
    del x, y, weights, bias, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _aggregate(rows: Sequence[dict[str, object]], feature_index: dict[tuple[str, str, str], np.ndarray]) -> np.ndarray:
    features = np.stack(
        [
            feature_index[(str(row["split"]), str(row["claim_id"]), str(row["evidence_id"]))]
            for row in rows
        ]
    )
    calibrated = np.asarray([float(row["calibrated_probability"]) for row in rows])
    raw = np.asarray([float(row["raw_probability"]) for row in rows])
    modalities = [str(row["modality"]) for row in rows]
    return np.concatenate(
        [
            features.mean(axis=0),
            features.max(axis=0),
            features.std(axis=0),
            np.asarray(
                [
                    math.log1p(len(rows)) / 10.0,
                    modalities.count("text") / len(rows),
                    modalities.count("image") / len(rows),
                    calibrated.mean(),
                    calibrated.max(),
                    raw.mean(),
                    raw.max(),
                ]
            ),
        ]
    )


def _selected_rows(
    rows: Sequence[dict[str, object]],
    method: str,
    parameters: dict[str, object],
    *,
    maximum_items: int,
) -> list[dict[str, object]]:
    probabilities = [float(row["calibrated_probability"]) for row in rows]
    source_groups = [str(row["source_group"]) for row in rows]
    engine = {
        "fixed": "fixed",
        "threshold": "threshold",
        "cefes": "cefes",
        "size_penalized": "penalized_cefes",
        "rc_cefes": "penalized_cefes",
    }[method]
    selected = _select_local(
        engine,
        probabilities,
        source_groups,
        maximum_items=maximum_items,
        fixed_k=parameters.get("fixed_k"),
        threshold=parameters.get("threshold"),
        size_penalty=float(parameters.get("size_penalty", 0.0)),
        redundancy_penalty=float(parameters.get("redundancy_penalty", 0.0)),
    )
    return [rows[index] for index in selected]


def _class_metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    scores: list[float] = []
    for label in VERDICT_LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        denominator = 2 * tp + fp + fn
        score = 2 * tp / denominator if denominator else 0.0
        result[f"{label}_f1"] = score
        scores.append(score)
    result["macro_f1"] = fmean(scores)
    result["accuracy"] = fmean(float(g == p) for g, p in zip(gold, predicted, strict=True))
    return result


def _paired_bootstrap(
    gold: Sequence[str],
    left: Sequence[str],
    right: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    generator = np.random.default_rng(seed)
    gold_values = np.asarray(gold, dtype=object)
    left_values = np.asarray(left, dtype=object)
    right_values = np.asarray(right, dtype=object)
    values: list[float] = []
    for _ in range(samples):
        indices = generator.integers(0, len(gold_values), size=len(gold_values))
        values.append(
            _class_metrics(gold_values[indices], left_values[indices])["macro_f1"]
            - _class_metrics(gold_values[indices], right_values[indices])["macro_f1"]
        )
    effect = _class_metrics(gold, left)["macro_f1"] - _class_metrics(gold, right)["macro_f1"]
    return {
        "mean": effect,
        "lower": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 0.975)),
        "sample_count": samples,
        "samples": values,
    }


def _seed_aligned_paired_bootstrap(
    rows: Sequence[tuple[Sequence[str], Sequence[str], Sequence[str]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """Resample claims once and average the paired macro-F1 effect over seeds."""
    if not rows:
        raise ValueError("seed-aligned bootstrap requires at least one seed")
    reference_gold = np.asarray(rows[0][0], dtype=object)
    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    effects: list[float] = []
    for gold, left, right in rows:
        gold_values = np.asarray(gold, dtype=object)
        if not np.array_equal(gold_values, reference_gold):
            raise ValueError("seed-aligned bootstrap requires identical claim order and labels")
        left_values = np.asarray(left, dtype=object)
        right_values = np.asarray(right, dtype=object)
        prepared.append((left_values, right_values))
        effects.append(
            _class_metrics(gold_values, left_values)["macro_f1"]
            - _class_metrics(gold_values, right_values)["macro_f1"]
        )
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = generator.integers(0, len(reference_gold), size=len(reference_gold))
        resampled_gold = reference_gold[indices]
        values.append(
            fmean(
                _class_metrics(resampled_gold, left[indices])["macro_f1"]
                - _class_metrics(resampled_gold, right[indices])["macro_f1"]
                for left, right in prepared
            )
        )
    return {
        "mean": fmean(effects),
        "lower": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 0.975)),
        "claims": len(reference_gold),
        "seed_count": len(rows),
        "sample_count": samples,
        "resampling_seed": seed,
        "samples": values,
    }


def run_phase2_verifier(
    retrieval_root: Path,
    selector_root: Path,
    output_root: Path,
    *,
    config: Phase2VerifierConfig = Phase2VerifierConfig(),
) -> dict[str, object]:
    started = time.perf_counter()
    selector_summary_path = selector_root / "summary.json"
    selector_summary = json.loads(selector_summary_path.read_text(encoding="utf-8"))
    if selector_summary.get("formal_test_annotations_accessed"):
        raise RuntimeError("selector outputs accessed formal-test annotations")
    retrieval_arrays = np.load(retrieval_root / "candidate_features.npz")
    retrieval_features = np.asarray(retrieval_arrays["features"], dtype=np.float64)
    retrieval_records = _read_jsonl(retrieval_root / "candidate_records.jsonl")
    feature_index = {
        (str(row["split"]), str(row["claim_id"]), str(row["evidence_id"])): retrieval_features[index]
        for index, row in enumerate(retrieval_records)
    }
    output_root.mkdir(parents=True, exist_ok=True)
    methods = ("fixed", "threshold", "cefes", "size_penalized", "rc_cefes")
    seed_results: list[dict[str, object]] = []
    aggregate_inputs: list[tuple[Sequence[str], Sequence[str], Sequence[str]]] = []
    for seed_row in selector_summary["seeds"]:
        seed = int(seed_row["seed"])
        parameters = seed_row["selected_parameters"]
        candidates = _read_jsonl(selector_root / f"seed_{seed}_candidate_records.jsonl")
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in candidates:
            grouped.setdefault((str(row["split"]), str(row["claim_id"])), []).append(row)
        training_features: list[np.ndarray] = []
        training_labels: list[str] = []
        for (split, _), rows in sorted(grouped.items()):
            if split != "train" or str(rows[0]["role"]) != "fit":
                continue
            seen: set[tuple[str, ...]] = set()
            for method in methods:
                selected = _selected_rows(
                    rows,
                    method,
                    parameters[method],
                    maximum_items=int(selector_summary["config"]["maximum_selected_items"]),
                )
                identity = tuple(sorted(str(row["evidence_id"]) for row in selected))
                if identity in seen:
                    continue
                seen.add(identity)
                training_features.append(_aggregate(selected, feature_index))
                training_labels.append(str(rows[0]["claim_label"]))
        verifier = _fit_verifier(np.stack(training_features), training_labels, config=config)
        checkpoint_path = output_root / f"seed_{seed}_verifier.json"
        checkpoint_path.write_text(
            json.dumps({"config": asdict(config), "model": asdict(verifier)}, indent=2) + "\n",
            encoding="utf-8",
        )
        validation_keys = sorted(key for key in grouped if key[0] == "validation")
        gold = [str(grouped[key][0]["claim_label"]) for key in validation_keys]
        predictions: dict[str, list[str]] = {}
        probabilities: dict[str, np.ndarray] = {}
        selected_ids: dict[str, list[list[str]]] = {}
        for method in methods:
            aggregates: list[np.ndarray] = []
            method_ids: list[list[str]] = []
            for key in validation_keys:
                selected = _selected_rows(
                    grouped[key],
                    method,
                    parameters[method],
                    maximum_items=int(selector_summary["config"]["maximum_selected_items"]),
                )
                aggregates.append(_aggregate(selected, feature_index))
                method_ids.append(sorted(str(row["evidence_id"]) for row in selected))
            method_probabilities = verifier.predict_probabilities(np.stack(aggregates))
            probabilities[method] = method_probabilities
            predictions[method] = [VERDICT_LABELS[int(index)] for index in method_probabilities.argmax(axis=1)]
            selected_ids[method] = method_ids
        metrics = {method: _class_metrics(gold, predictions[method]) for method in methods}
        paired = _paired_bootstrap(
            gold,
            predictions["rc_cefes"],
            predictions["fixed"],
            samples=config.bootstrap_samples,
            seed=seed,
        )
        aggregate_inputs.append((gold, predictions["rc_cefes"], predictions["fixed"]))
        records_path = output_root / f"seed_{seed}_verdict_records.jsonl"
        _write_jsonl(
            records_path,
            (
                {
                    "seed": seed,
                    "claim_id": claim_id,
                    "gold_label": gold[index],
                    "methods": {
                        method: {
                            "predicted_label": predictions[method][index],
                            "probabilities": {
                                label: float(probabilities[method][index, label_index])
                                for label_index, label in enumerate(VERDICT_LABELS)
                            },
                            "selected_evidence_ids": selected_ids[method][index],
                        }
                        for method in methods
                    },
                }
                for index, (_, claim_id) in enumerate(validation_keys)
            ),
        )
        seed_results.append(
            {
                "seed": seed,
                "training_examples": len(training_features),
                "validation_claims": len(validation_keys),
                "metrics": metrics,
                "rc_cefes_minus_fixed_bootstrap": paired,
                "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
                "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
            }
        )
    mean_effect = fmean(
        float(row["metrics"]["rc_cefes"]["macro_f1"])
        - float(row["metrics"]["fixed"]["macro_f1"])
        for row in seed_results
    )
    lower_bounds = [
        float(row["rc_cefes_minus_fixed_bootstrap"]["lower"]) for row in seed_results
    ]
    aggregate = _seed_aligned_paired_bootstrap(
        aggregate_inputs,
        samples=config.bootstrap_samples,
        seed=20261102,
    )
    summary: dict[str, object] = {
        "profile": "q1_phase2_shared_verifier",
        "status": "pass",
        "config": asdict(config),
        "selector_summary_sha256": sha256_file(selector_summary_path),
        "seeds": seed_results,
        "mean_rc_cefes_minus_fixed_macro_f1": mean_effect,
        "aggregate_seed_aligned_rc_cefes_minus_fixed": aggregate,
        "noninferiority_margin": config.noninferiority_margin,
        "aggregate_noninferiority_lower_bound_above_margin": (
            float(aggregate["lower"]) > config.noninferiority_margin
        ),
        "noninferiority_lower_bound_above_margin_each_seed": all(
            value > config.noninferiority_margin for value in lower_bounds
        ),
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


def audit_phase2_verifier(output_root: Path) -> dict[str, object]:
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if summary.get("formal_test_annotations_accessed"):
        errors.append("formal-test annotations were accessed")
    for seed_row in summary.get("seeds", []):
        for name in ("checkpoint", "records"):
            artifact = seed_row[name]
            path = Path(str(artifact["path"]))
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                errors.append(f"artifact mismatch: {path}")
        metrics = seed_row["metrics"]
        effect = float(metrics["rc_cefes"]["macro_f1"]) - float(metrics["fixed"]["macro_f1"])
        if not math.isclose(effect, float(seed_row["rc_cefes_minus_fixed_bootstrap"]["mean"]), abs_tol=1e-12):
            errors.append(f"effect mismatch for seed {seed_row['seed']}")
    audit = {
        "profile": "q1_phase2_verifier_independent_audit",
        "status": "pass" if not errors else "failed",
        "errors": errors,
        "summary_sha256": sha256_file(summary_path),
        "formal_test_annotations_accessed": False,
    }
    path = output_root / "audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError(f"Phase-2 verifier audit failed: {errors}")
    return audit
