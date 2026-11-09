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
import yaml

from trace_scv.data.io import sha256_file
from trace_scv.q1_phase2_verifier import (
    VERDICT_LABELS,
    Phase2VerifierConfig,
    _aggregate,
    _class_metrics,
    _fit_verifier,
    _paired_bootstrap,
    _seed_aligned_paired_bootstrap,
    _selected_rows,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(frozen=True, slots=True)
class MLPVerifierConfig:
    hidden_units: int = 64
    epochs: int = 300
    learning_rate: float = 0.01
    l2: float = 1e-3
    deterministic_seed_base: int = 20_261_109
    bootstrap_samples: int = 10_000
    noninferiority_margin: float = -0.01


@dataclass(frozen=True, slots=True)
class MLPVerifierModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    input_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        values = (features - np.asarray(self.mean)) / np.asarray(self.scale)
        hidden = np.maximum(
            0.0,
            values @ np.asarray(self.input_weights) + np.asarray(self.hidden_bias),
        )
        logits = hidden @ np.asarray(self.output_weights) + np.asarray(self.output_bias)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(np.clip(logits, -40.0, 40.0))
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_mlp_verifier(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    seed: int,
    config: MLPVerifierConfig,
) -> MLPVerifierModel:
    import torch

    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray([VERDICT_LABELS.index(label) for label in labels], dtype=np.int64)
    if set(np.unique(targets)) != set(range(len(VERDICT_LABELS))):
        raise ValueError("verifier training requires all verdict classes")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (values - mean) / scale
    torch.manual_seed(config.deterministic_seed_base + seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.deterministic_seed_base + seed)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(standardized, device=device)
    y = torch.as_tensor(targets, device=device)
    input_weights = torch.empty(
        (values.shape[1], config.hidden_units), device=device, requires_grad=True
    )
    hidden_bias = torch.zeros(config.hidden_units, device=device, requires_grad=True)
    output_weights = torch.empty(
        (config.hidden_units, len(VERDICT_LABELS)), device=device, requires_grad=True
    )
    output_bias = torch.zeros(len(VERDICT_LABELS), device=device, requires_grad=True)
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(input_weights)
        torch.nn.init.xavier_uniform_(output_weights)
    counts = np.bincount(targets, minlength=len(VERDICT_LABELS)).astype(np.float32)
    class_weights = torch.as_tensor(
        len(targets) / (len(VERDICT_LABELS) * counts), device=device
    )
    optimizer = torch.optim.Adam(
        [input_weights, hidden_bias, output_weights, output_bias],
        lr=config.learning_rate,
    )
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.relu(x @ input_weights + hidden_bias)
        logits = hidden @ output_weights + output_bias
        loss = torch.nn.functional.cross_entropy(logits, y, weight=class_weights)
        loss = loss + config.l2 * (
            input_weights.square().mean() + output_weights.square().mean()
        )
        loss.backward()
        optimizer.step()
    result = MLPVerifierModel(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        input_weights=tuple(
            tuple(float(value) for value in row)
            for row in input_weights.detach().cpu().numpy()
        ),
        hidden_bias=tuple(float(value) for value in hidden_bias.detach().cpu().numpy()),
        output_weights=tuple(
            tuple(float(value) for value in row)
            for row in output_weights.detach().cpu().numpy()
        ),
        output_bias=tuple(float(value) for value in output_bias.detach().cpu().numpy()),
    )
    del x, y, input_weights, hidden_bias, output_weights, output_bias, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def config_from_protocol(protocol: dict[str, object]) -> MLPVerifierConfig:
    model = protocol["verifier"]["model"]
    return MLPVerifierConfig(
        hidden_units=int(model["hidden_units"]),
        epochs=int(model["epochs"]),
        learning_rate=float(model["learning_rate"]),
        l2=float(model["l2"]),
        deterministic_seed_base=int(model["deterministic_seed_base"]),
        bootstrap_samples=int(protocol["statistics"]["bootstrap_samples"]),
        noninferiority_margin=float(protocol["verifier"]["noninferiority_margin"]),
    )


def run_stronger_verifier(
    retrieval_root: Path,
    selector_root: Path,
    completeness_root: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["study"]["formal_test_annotations_allowed"]:
        raise RuntimeError("formal-test annotations are not allowed")
    config = config_from_protocol(protocol)
    selector_summary = json.loads((selector_root / "summary.json").read_text(encoding="utf-8"))
    completeness_summary = json.loads(
        (completeness_root / "summary.json").read_text(encoding="utf-8")
    )
    if selector_summary.get("formal_test_annotations_accessed"):
        raise RuntimeError("base selector reports formal-test access")
    if completeness_summary.get("formal_test_annotations_accessed"):
        raise RuntimeError("completeness selector reports formal-test access")

    arrays = np.load(retrieval_root / "candidate_features.npz")
    retrieval_features = np.asarray(arrays["features"], dtype=np.float64)
    retrieval_records = _read_jsonl(retrieval_root / "candidate_records.jsonl")
    feature_index = {
        (str(row["split"]), str(row["claim_id"]), str(row["evidence_id"])): retrieval_features[index]
        for index, row in enumerate(retrieval_records)
    }
    output_root.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, object]] = []
    aggregate_inputs: list[tuple[Sequence[str], Sequence[str], Sequence[str]]] = []
    maximum_items = int(selector_summary["config"]["maximum_selected_items"])

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
            for method in ("fixed", "threshold", "cefes"):
                selected = _selected_rows(
                    rows,
                    method,
                    parameters[method],
                    maximum_items=maximum_items,
                )
                identity = tuple(sorted(str(row["evidence_id"]) for row in selected))
                if identity in seen:
                    continue
                seen.add(identity)
                training_features.append(_aggregate(selected, feature_index))
                training_labels.append(str(rows[0]["claim_label"]))
        training_array = np.stack(training_features)
        mlp = fit_mlp_verifier(training_array, training_labels, seed=seed, config=config)
        linear = _fit_verifier(
            training_array,
            training_labels,
            config=Phase2VerifierConfig(
                epochs=500,
                learning_rate=0.05,
                l2=1e-3,
                bootstrap_samples=config.bootstrap_samples,
                noninferiority_margin=config.noninferiority_margin,
            ),
        )
        checkpoint_path = output_root / f"seed_{seed}_verifier.json"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "mlp": asdict(mlp),
                    "linear_control": asdict(linear),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        switch_rows = {
            str(row["claim_id"]): bool(row["switch_used_expected_f1"])
            for row in _read_jsonl(completeness_root / f"seed_{seed}_claim_records.jsonl")
        }
        validation_keys = sorted(key for key in grouped if key[0] == "validation")
        gold = [str(grouped[key][0]["claim_label"]) for key in validation_keys]
        threshold_aggregates: list[np.ndarray] = []
        switch_aggregates: list[np.ndarray] = []
        threshold_ids: list[list[str]] = []
        switch_ids: list[list[str]] = []
        for key in validation_keys:
            rows = grouped[key]
            threshold_selected = _selected_rows(
                rows, "threshold", parameters["threshold"], maximum_items=maximum_items
            )
            expected_selected = _selected_rows(
                rows, "cefes", parameters["cefes"], maximum_items=maximum_items
            )
            use_expected = switch_rows[key[1]]
            switch_selected = expected_selected if use_expected else threshold_selected
            threshold_aggregates.append(_aggregate(threshold_selected, feature_index))
            switch_aggregates.append(_aggregate(switch_selected, feature_index))
            threshold_ids.append(sorted(str(row["evidence_id"]) for row in threshold_selected))
            switch_ids.append(sorted(str(row["evidence_id"]) for row in switch_selected))
        threshold_array = np.stack(threshold_aggregates)
        switch_array = np.stack(switch_aggregates)
        threshold_probabilities = mlp.predict_probabilities(threshold_array)
        switch_probabilities = mlp.predict_probabilities(switch_array)
        linear_threshold_probabilities = linear.predict_probabilities(threshold_array)
        linear_switch_probabilities = linear.predict_probabilities(switch_array)
        threshold_predictions = [
            VERDICT_LABELS[int(index)] for index in threshold_probabilities.argmax(axis=1)
        ]
        switch_predictions = [
            VERDICT_LABELS[int(index)] for index in switch_probabilities.argmax(axis=1)
        ]
        linear_threshold_predictions = [
            VERDICT_LABELS[int(index)]
            for index in linear_threshold_probabilities.argmax(axis=1)
        ]
        linear_switch_predictions = [
            VERDICT_LABELS[int(index)] for index in linear_switch_probabilities.argmax(axis=1)
        ]
        metrics = {
            "mlp_threshold": _class_metrics(gold, threshold_predictions),
            "mlp_switch": _class_metrics(gold, switch_predictions),
            "linear_threshold": _class_metrics(gold, linear_threshold_predictions),
            "linear_switch": _class_metrics(gold, linear_switch_predictions),
        }
        paired = _paired_bootstrap(
            gold,
            switch_predictions,
            threshold_predictions,
            samples=config.bootstrap_samples,
            seed=config.deterministic_seed_base + seed,
        )
        aggregate_inputs.append((gold, switch_predictions, threshold_predictions))
        records_path = output_root / f"seed_{seed}_verdict_records.jsonl"
        _write_jsonl(
            records_path,
            (
                {
                    "seed": seed,
                    "claim_id": claim_id,
                    "gold_label": gold[index],
                    "threshold": {
                        "predicted_label": threshold_predictions[index],
                        "probabilities": {
                            label: float(threshold_probabilities[index, label_index])
                            for label_index, label in enumerate(VERDICT_LABELS)
                        },
                        "selected_evidence_ids": threshold_ids[index],
                    },
                    "completeness_switch": {
                        "predicted_label": switch_predictions[index],
                        "probabilities": {
                            label: float(switch_probabilities[index, label_index])
                            for label_index, label in enumerate(VERDICT_LABELS)
                        },
                        "selected_evidence_ids": switch_ids[index],
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
                "switch_minus_threshold_bootstrap": paired,
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": sha256_file(checkpoint_path),
                },
                "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
            }
        )

    aggregate = _seed_aligned_paired_bootstrap(
        aggregate_inputs,
        samples=config.bootstrap_samples,
        seed=config.deterministic_seed_base,
    )
    seed_effects = [
        float(row["metrics"]["mlp_switch"]["macro_f1"])
        - float(row["metrics"]["mlp_threshold"]["macro_f1"])
        for row in seed_results
    ]
    lower_bounds = [
        float(row["switch_minus_threshold_bootstrap"]["lower"])
        for row in seed_results
    ]
    summary = {
        "profile": "publication_confirmation_stronger_shared_verifier",
        "status": "complete",
        "config": asdict(config),
        "selector_summary_sha256": sha256_file(selector_root / "summary.json"),
        "completeness_summary_sha256": sha256_file(completeness_root / "summary.json"),
        "seeds": seed_results,
        "mean_switch_minus_threshold_macro_f1": fmean(seed_effects),
        "aggregate_seed_aligned_switch_minus_threshold": aggregate,
        "noninferiority_margin": config.noninferiority_margin,
        "aggregate_noninferiority_pass": float(aggregate["lower"])
        > config.noninferiority_margin,
        "each_seed_noninferiority_pass": all(
            value > config.noninferiority_margin for value in lower_bounds
        ),
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "formal_test_annotations_accessed": False,
        "formal_test_unlocked": False,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def audit_stronger_verifier(output_root: Path) -> dict[str, object]:
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if summary.get("formal_test_annotations_accessed") or summary.get("formal_test_unlocked"):
        failures.append("formal-test lock is not intact")
    effects: list[float] = []
    for row in summary["seeds"]:
        for name in ("checkpoint", "records"):
            artifact = Path(str(row[name]["path"]))
            if not artifact.is_file() or sha256_file(artifact) != row[name]["sha256"]:
                failures.append(f"artifact mismatch: {name} seed {row['seed']}")
        effect = (
            float(row["metrics"]["mlp_switch"]["macro_f1"])
            - float(row["metrics"]["mlp_threshold"]["macro_f1"])
        )
        effects.append(effect)
        if not math.isclose(
            effect,
            float(row["switch_minus_threshold_bootstrap"]["mean"]),
            abs_tol=1e-12,
        ):
            failures.append(f"effect mismatch seed {row['seed']}")
    if not math.isclose(
        fmean(effects),
        float(summary["mean_switch_minus_threshold_macro_f1"]),
        abs_tol=1e-12,
    ):
        failures.append("mean effect mismatch")
    return {
        "profile": "publication_confirmation_stronger_verifier_audit",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "summary_sha256": sha256_file(summary_path),
        "formal_test_annotations_accessed": False,
    }

