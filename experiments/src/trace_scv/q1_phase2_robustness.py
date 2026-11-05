from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Sequence

import numpy as np

from trace_scv.q1_phase2_study import _select_local, _set_metrics, _temperature_transform


METHODS = ("fixed", "threshold", "cefes", "rc_cefes")
CONDITIONS = (
    "baseline",
    "width_10",
    "width_20",
    "width_40",
    "text_only",
    "image_only",
    "remove_gold_25",
    "remove_gold_50",
    "promote_5_distractors",
    "duplicate_top_3",
    "temperature_0.5",
    "temperature_2.0",
)


def _read_validation_records(path: Path) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "validation":
                grouped[f"validation:{row['claim_id']}"].append(row)
    return dict(grouped)


def _hash_fraction(value: str) -> float:
    raw = hashlib.sha256(value.encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, "big") / float(2**64)


def _transform_rows(
    rows: Sequence[dict[str, object]],
    condition: str,
    *,
    seed: int,
    key: str,
) -> tuple[list[dict[str, object]], np.ndarray]:
    copied = [dict(row) for row in rows]
    if condition.startswith("width_"):
        width = int(condition.split("_")[1])
        copied = [row for row in copied if int(row["final_rank"]) <= width]
    elif condition == "text_only":
        copied = [row for row in copied if row["modality"] == "text"]
    elif condition == "image_only":
        copied = [row for row in copied if row["modality"] == "image"]
    elif condition.startswith("remove_gold_"):
        fraction = int(condition.rsplit("_", 1)[1]) / 100.0
        copied = [
            row
            for row in copied
            if not bool(row["gold"])
            or _hash_fraction(f"{seed}:{key}:{row['evidence_id']}:{condition}") >= fraction
        ]
    probabilities = np.asarray(
        [float(row["calibrated_probability"]) for row in copied], dtype=np.float64
    )
    if condition == "promote_5_distractors":
        candidates = sorted(
            (index for index, row in enumerate(copied) if not bool(row["gold"])),
            key=lambda index: (int(copied[index]["final_rank"]), index),
        )[:5]
        for offset, index in enumerate(candidates):
            probabilities[index] = 0.999999 - offset * 1e-7
    elif condition == "duplicate_top_3":
        order = sorted(range(len(copied)), key=lambda index: (-probabilities[index], index))[:3]
        additions: list[dict[str, object]] = []
        addition_probabilities: list[float] = []
        for offset, index in enumerate(order):
            duplicate = dict(copied[index])
            duplicate["evidence_id"] = f"{duplicate['evidence_id']}#duplicate-{offset}"
            duplicate["gold"] = False
            additions.append(duplicate)
            addition_probabilities.append(float(probabilities[index]))
        copied.extend(additions)
        probabilities = np.concatenate((probabilities, np.asarray(addition_probabilities)))
    elif condition.startswith("temperature_"):
        temperature = float(condition.split("_")[1])
        probabilities = _temperature_transform(probabilities, temperature)
    if not copied:
        fallback = dict(rows[0])
        fallback["gold"] = False
        copied = [fallback]
        probabilities = np.asarray([float(fallback["calibrated_probability"])])
    return copied, probabilities


def _evaluate_claim(
    rows: Sequence[dict[str, object]],
    probabilities: np.ndarray,
    gold: set[str],
    method: str,
    parameters: dict[str, object],
    maximum_items: int,
) -> dict[str, float]:
    engine = "penalized_cefes" if method == "rc_cefes" else method
    selected_local = _select_local(
        engine,
        probabilities,
        [str(row["source_group"]) for row in rows],
        maximum_items=maximum_items,
        fixed_k=parameters.get("fixed_k"),
        threshold=parameters.get("threshold"),
        size_penalty=float(parameters.get("size_penalty", 0.0)),
        redundancy_penalty=float(parameters.get("redundancy_penalty", 0.0)),
    )
    selected = {str(rows[index]["evidence_id"]) for index in selected_local}
    metrics = _set_metrics(selected, gold)
    metrics["selected_count"] = float(len(selected))
    return metrics


def _aligned_bootstrap(
    differences: dict[int, dict[str, float]], *, samples: int, seed: int
) -> dict[str, float]:
    keys = sorted(next(iter(differences.values())))
    matrix = np.asarray(
        [[differences[current_seed][key] for key in keys] for current_seed in sorted(differences)]
    )
    claim_effects = matrix.mean(axis=0)
    generator = np.random.default_rng(seed)
    values: list[float] = []
    remaining = samples
    while remaining:
        current = min(512, remaining)
        indices = generator.integers(0, len(keys), size=(current, len(keys)))
        values.extend(claim_effects[indices].mean(axis=1).tolist())
        remaining -= current
    return {
        "mean": float(claim_effects.mean()),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "claims": float(len(keys)),
        "samples": float(samples),
    }


def run_q1_phase2_robustness(
    retrieval_root: Path,
    selector_root: Path,
    output_root: Path,
    *,
    bootstrap_samples: int = 5_000,
) -> dict[str, object]:
    selector = json.loads((selector_root / "summary.json").read_text(encoding="utf-8"))
    full_gold = {
        str(key): set(map(str, values))
        for key, values in json.loads(
            (retrieval_root / "full_gold.json").read_text(encoding="utf-8")
        ).items()
    }
    condition_rows: list[dict[str, object]] = []
    effects: dict[str, dict[int, dict[str, float]]] = {
        condition: {} for condition in CONDITIONS
    }
    for seed_row in selector["seeds"]:
        seed = int(seed_row["seed"])
        grouped = _read_validation_records(
            selector_root / f"seed_{seed}_candidate_records.jsonl"
        )
        parameters = seed_row["selected_parameters"]
        for condition in CONDITIONS:
            method_claim_metrics: dict[str, list[dict[str, float]]] = {
                method: [] for method in METHODS
            }
            condition_effects: dict[str, float] = {}
            for key in sorted(grouped):
                rows, probabilities = _transform_rows(
                    grouped[key], condition, seed=seed, key=key
                )
                claim_metrics = {
                    method: _evaluate_claim(
                        rows,
                        probabilities,
                        full_gold[key],
                        method,
                        parameters[method],
                        int(selector["config"]["maximum_selected_items"]),
                    )
                    for method in METHODS
                }
                for method in METHODS:
                    method_claim_metrics[method].append(claim_metrics[method])
                condition_effects[key] = (
                    claim_metrics["rc_cefes"]["f1"] - claim_metrics["fixed"]["f1"]
                )
            effects[condition][seed] = condition_effects
            for method in METHODS:
                rows = method_claim_metrics[method]
                condition_rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "method": method,
                        "f1": fmean(row["f1"] for row in rows),
                        "precision": fmean(row["precision"] for row in rows),
                        "recall": fmean(row["recall"] for row in rows),
                        "selected_count": fmean(row["selected_count"] for row in rows),
                    }
                )
    aggregates = {
        condition: _aligned_bootstrap(
            effects[condition], samples=bootstrap_samples, seed=77_000 + index
        )
        for index, condition in enumerate(CONDITIONS)
    }
    summary = {
        "profile": "q1_phase2_exploratory_robustness",
        "status": "complete",
        "confirmatory": False,
        "conditions": list(CONDITIONS),
        "methods": list(METHODS),
        "seed_condition_method_metrics": condition_rows,
        "rc_cefes_minus_fixed_seed_aligned": aggregates,
        "formal_test_annotations_accessed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def audit_q1_phase2_robustness(selector_root: Path, output_root: Path) -> dict[str, object]:
    selector = json.loads((selector_root / "summary.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if summary.get("confirmatory") is not False:
        errors.append("robustness analysis is not marked exploratory")
    if summary.get("formal_test_annotations_accessed"):
        errors.append("formal-test seal is not intact")
    baseline = {
        (int(row["seed"]), str(row["method"])): float(row["f1"])
        for row in summary.get("seed_condition_method_metrics", [])
        if row.get("condition") == "baseline"
    }
    for seed_row in selector["seeds"]:
        seed = int(seed_row["seed"])
        for method in METHODS:
            expected = float(seed_row["validation"][method]["f1"])
            actual = baseline.get((seed, method))
            if actual is None or not np.isclose(actual, expected, atol=1e-12):
                errors.append(f"baseline mismatch for seed {seed}, method {method}")
    audit = {
        "profile": "q1_phase2_exploratory_robustness_audit",
        "status": "pass" if not errors else "failed",
        "errors": errors,
        "formal_test_annotations_accessed": False,
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError(f"robustness audit failed: {errors}")
    return audit
