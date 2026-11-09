from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import numpy as np
import yaml

from trace_scv.data.io import sha256_file
from trace_scv.q1_phase2_study import (
    _aggregate_seed_aligned_bootstrap,
    _fit_logistic,
    _paired_bootstrap,
    _select_local,
    _set_metrics,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _group_claims(rows: Sequence[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = f"{row['split']}:{row['claim_id']}"
        grouped.setdefault(key, []).append(row)
    return grouped


def _selected_indices(
    probabilities: Sequence[float],
    rows: Sequence[dict[str, object]],
    method: str,
    *,
    threshold: float,
    maximum_items: int,
) -> list[int]:
    return _select_local(
        method,
        probabilities,
        [str(row["source_group"]) for row in rows],
        threshold=threshold if method == "threshold" else None,
        maximum_items=maximum_items,
    )


def completeness_feature_vector(
    rows: Sequence[dict[str, object]],
    *,
    threshold: float,
    maximum_items: int,
) -> np.ndarray:
    probabilities = np.asarray(
        [float(row["calibrated_probability"]) for row in rows], dtype=np.float64
    )
    order = np.argsort(-probabilities, kind="stable")
    ranked = probabilities[order]
    top5 = ranked[:5]
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    entropy = -np.mean(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped))
    threshold_indices = _selected_indices(
        probabilities, rows, "threshold", threshold=threshold, maximum_items=maximum_items
    )
    expected_indices = _selected_indices(
        probabilities, rows, "cefes", threshold=threshold, maximum_items=maximum_items
    )
    text_mass = sum(
        probability
        for probability, row in zip(probabilities, rows, strict=True)
        if row["modality"] == "text"
    )
    image_mass = float(probabilities.sum()) - text_mass
    agreements: list[float] = []
    for row in rows:
        dense = row.get("dense_rank")
        lexical = row.get("lexical_rank")
        if dense is not None and lexical is not None:
            agreements.append(1.0 / (1.0 + abs(int(dense) - int(lexical))))
    rank_agreement = fmean(agreements) if agreements else 0.0
    fifth = float(ranked[min(4, len(ranked) - 1)])
    return np.asarray(
        [
            float(probabilities.sum()),
            float(ranked[0]),
            float(top5.mean()),
            float(entropy),
            float(ranked[0] - fifth),
            float(len(threshold_indices)),
            float(len(expected_indices)),
            float(text_mass),
            float(image_mass),
            float(rank_agreement),
        ],
        dtype=np.float64,
    )


def _claim_policy_rows(
    key: str,
    rows: Sequence[dict[str, object]],
    gold: set[str],
    *,
    threshold: float,
    maximum_items: int,
) -> dict[str, object]:
    calibrated = [float(row["calibrated_probability"]) for row in rows]
    raw = [float(row["raw_probability"]) for row in rows]
    threshold_indices = _selected_indices(
        calibrated, rows, "threshold", threshold=threshold, maximum_items=maximum_items
    )
    expected_indices = _selected_indices(
        calibrated, rows, "cefes", threshold=threshold, maximum_items=maximum_items
    )
    raw_expected_indices = _selected_indices(
        raw, rows, "cefes", threshold=threshold, maximum_items=maximum_items
    )

    def selected_ids(indices: Sequence[int]) -> set[str]:
        return {str(rows[index]["evidence_id"]) for index in indices}

    packet_ids = {str(row["evidence_id"]) for row in rows}
    threshold_ids = selected_ids(threshold_indices)
    expected_ids = selected_ids(expected_indices)
    raw_expected_ids = selected_ids(raw_expected_indices)
    split, claim_id = key.split(":", 1)
    return {
        "split": split,
        "claim_id": claim_id,
        "role": str(rows[0]["role"]),
        "complete_gold_reachable": bool(gold and gold <= packet_ids),
        "features": completeness_feature_vector(
            rows, threshold=threshold, maximum_items=maximum_items
        ),
        "threshold_ids": threshold_ids,
        "expected_ids": expected_ids,
        "raw_expected_ids": raw_expected_ids,
        "threshold_metrics": _set_metrics(threshold_ids, gold),
        "expected_metrics": _set_metrics(expected_ids, gold),
        "raw_expected_metrics": _set_metrics(raw_expected_ids, gold),
    }


def _summarize(rows: Sequence[dict[str, object]], prefix: str) -> dict[str, float]:
    return {
        metric: fmean(float(row[f"{prefix}_{metric}"]) for row in rows)
        for metric in ("precision", "recall", "f1", "selected_count")
    }


def paired_bootstrap_rows(
    rows: Sequence[dict[str, object]], prefix: str
) -> list[dict[str, object]]:
    return [
        {"key": str(row["claim_id"]), "f1": float(row[f"{prefix}_f1"])}
        for row in rows
    ]


def serialize_completeness_model(model: object) -> dict[str, object]:
    return asdict(model)


def run_completeness_switch(
    selector_root: Path,
    retrieval_root: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["study"]["formal_test_annotations_allowed"]:
        raise RuntimeError("formal-test annotations are not allowed")
    selector_summary = json.loads((selector_root / "summary.json").read_text(encoding="utf-8"))
    selector_audit = json.loads(
        (selector_root / "selector.audit.json").read_text(encoding="utf-8")
    )
    if selector_audit.get("status") != "pass":
        raise RuntimeError("base selector audit must pass")
    full_gold_raw = json.loads((retrieval_root / "full_gold.json").read_text(encoding="utf-8"))
    full_gold = {str(key): set(map(str, value)) for key, value in full_gold_raw.items()}
    maximum_items = int(protocol["selection"]["maximum_selected_items"])
    switch_thresholds = tuple(
        map(float, protocol["selection"]["completeness_model"]["threshold_candidates"])
    )
    bootstrap_samples = int(protocol["statistics"]["bootstrap_samples"])
    output_root.mkdir(parents=True, exist_ok=True)

    seed_results: list[dict[str, object]] = []
    aggregate_inputs: list[
        tuple[int, Sequence[dict[str, object]], Sequence[dict[str, object]]]
    ] = []
    for seed_row in selector_summary["seeds"]:
        seed = int(seed_row["seed"])
        threshold = float(seed_row["selected_parameters"]["threshold"]["threshold"])
        candidates = _read_jsonl(selector_root / f"seed_{seed}_candidate_records.jsonl")
        grouped = _group_claims(candidates)
        profiles: list[dict[str, object]] = []
        for key, rows in grouped.items():
            gold = full_gold.get(key, set())
            if not gold:
                continue
            profiles.append(
                _claim_policy_rows(
                    key,
                    rows,
                    gold,
                    threshold=threshold,
                    maximum_items=maximum_items,
                )
            )
        fit_profiles = [row for row in profiles if row["role"] == "fit"]
        fit_x = np.vstack([row["features"] for row in fit_profiles])
        fit_y = np.asarray(
            [int(row["complete_gold_reachable"]) for row in fit_profiles], dtype=np.int8
        )
        model = _fit_logistic(
            fit_x,
            fit_y,
            learning_rate=0.05,
            epochs=400,
            l2=1e-3,
            balanced=True,
        )
        profile_x = np.vstack([row["features"] for row in profiles])
        predictions = model.predict_probabilities(profile_x)
        for row, probability in zip(profiles, predictions, strict=True):
            row["predicted_completeness"] = float(probability)

        model_selection = [row for row in profiles if row["role"] == "model_selection"]

        def switch_f1(cutoff: float, subset: Sequence[dict[str, object]]) -> float:
            return fmean(
                float(
                    row["expected_metrics"]["f1"]
                    if float(row["predicted_completeness"]) >= cutoff
                    else row["threshold_metrics"]["f1"]
                )
                for row in subset
            )

        selected_cutoff = max(
            switch_thresholds,
            key=lambda value: (switch_f1(value, model_selection), value),
        )
        validation_output: list[dict[str, object]] = []
        for profile in profiles:
            if profile["role"] != "validation":
                continue
            use_expected = float(profile["predicted_completeness"]) >= selected_cutoff
            switch_ids = profile["expected_ids"] if use_expected else profile["threshold_ids"]
            gold = full_gold[f"validation:{profile['claim_id']}"]
            switch_metrics = _set_metrics(switch_ids, gold)
            validation_output.append(
                {
                    "claim_id": str(profile["claim_id"]),
                    "complete_gold_reachable": bool(profile["complete_gold_reachable"]),
                    "predicted_completeness": float(profile["predicted_completeness"]),
                    "switch_used_expected_f1": use_expected,
                    **{
                        f"threshold_{metric}": float(value)
                        for metric, value in profile["threshold_metrics"].items()
                    },
                    "threshold_selected_count": len(profile["threshold_ids"]),
                    **{
                        f"expected_{metric}": float(value)
                        for metric, value in profile["expected_metrics"].items()
                    },
                    "expected_selected_count": len(profile["expected_ids"]),
                    **{
                        f"raw_expected_{metric}": float(value)
                        for metric, value in profile["raw_expected_metrics"].items()
                    },
                    "raw_expected_selected_count": len(profile["raw_expected_ids"]),
                    **{f"switch_{metric}": float(value) for metric, value in switch_metrics.items()},
                    "switch_selected_count": len(switch_ids),
                }
            )
        claim_path = output_root / f"seed_{seed}_claim_records.jsonl"
        _write_jsonl(claim_path, validation_output)
        switch_rows = paired_bootstrap_rows(validation_output, "switch")
        threshold_rows = paired_bootstrap_rows(validation_output, "threshold")
        paired = _paired_bootstrap(
            switch_rows,
            threshold_rows,
            samples=bootstrap_samples,
            seed=80_000 + seed,
        )
        aggregate_inputs.append((seed, switch_rows, threshold_rows))
        targets = np.asarray(
            [int(row["complete_gold_reachable"]) for row in profiles if row["role"] == "validation"],
            dtype=np.float64,
        )
        predicted = np.asarray(
            [float(row["predicted_completeness"]) for row in profiles if row["role"] == "validation"],
            dtype=np.float64,
        )
        seed_result = {
            "seed": seed,
            "base_threshold": threshold,
            "selected_completeness_cutoff": selected_cutoff,
            "fit_claims": len(fit_profiles),
            "fit_complete_prevalence": float(fit_y.mean()),
            "validation_complete_prevalence": float(targets.mean()),
            "validation_completeness_brier": float(np.mean((predicted - targets) ** 2)),
            "validation_expected_use_share": fmean(
                float(row["switch_used_expected_f1"]) for row in validation_output
            ),
            "model": serialize_completeness_model(model),
            "model_selection_switch_f1_by_cutoff": {
                str(value): switch_f1(value, model_selection) for value in switch_thresholds
            },
            "validation": {
                "threshold": _summarize(validation_output, "threshold"),
                "expected_f1": _summarize(validation_output, "expected"),
                "uncalibrated_expected_f1": _summarize(validation_output, "raw_expected"),
                "completeness_switch": _summarize(validation_output, "switch"),
            },
            "paired_switch_vs_threshold": paired,
            "artifact": {"path": str(claim_path), "sha256": sha256_file(claim_path)},
        }
        seed_path = output_root / f"seed_{seed}.json"
        seed_path.write_text(json.dumps(seed_result, indent=2) + "\n", encoding="utf-8")
        seed_results.append(seed_result)

    aggregate = _aggregate_seed_aligned_bootstrap(
        aggregate_inputs,
        samples=bootstrap_samples,
        seed=97_311,
    )
    effects = [
        float(row["validation"]["completeness_switch"]["f1"])
        - float(row["validation"]["threshold"]["f1"])
        for row in seed_results
    ]
    mean_switch_count = fmean(
        float(row["validation"]["completeness_switch"]["selected_count"])
        for row in seed_results
    )
    mean_threshold_count = fmean(
        float(row["validation"]["threshold"]["selected_count"])
        for row in seed_results
    )
    gate = {
        "minimum_gain_0_01": fmean(effects) >= 0.01,
        "positive_each_partition": all(value > 0.0 for value in effects),
        "aggregate_interval_lower_positive": float(aggregate["lower"]) > 0.0,
        "mean_selected_count_not_above_threshold": mean_switch_count <= mean_threshold_count,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "profile": "publication_confirmation_completeness_switch",
        "status": "complete",
        "feature_names": list(protocol["selection"]["completeness_model"]["features"]),
        "threshold_candidates": list(switch_thresholds),
        "seeds": seed_results,
        "aggregate_switch_vs_threshold": aggregate,
        "mean_switch_minus_threshold_f1": fmean(effects),
        "mean_switch_selected_count": mean_switch_count,
        "mean_threshold_selected_count": mean_threshold_count,
        "superiority_gate": gate,
        "base_selector_summary_sha256": sha256_file(selector_root / "summary.json"),
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "formal_test_annotations_accessed": False,
        "formal_test_unlocked": False,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def audit_completeness_switch(output_root: Path) -> dict[str, object]:
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if summary.get("formal_test_annotations_accessed") or summary.get("formal_test_unlocked"):
        failures.append("formal-test lock is not intact")
    effects: list[float] = []
    for seed in summary["seeds"]:
        artifact = Path(str(seed["artifact"]["path"]))
        if not artifact.is_file() or sha256_file(artifact) != seed["artifact"]["sha256"]:
            failures.append(f"artifact mismatch for seed {seed['seed']}")
        effect = (
            float(seed["validation"]["completeness_switch"]["f1"])
            - float(seed["validation"]["threshold"]["f1"])
        )
        effects.append(effect)
        if not math.isclose(effect, float(seed["paired_switch_vs_threshold"]["mean"]), abs_tol=1e-12):
            failures.append(f"paired effect mismatch for seed {seed['seed']}")
    if not math.isclose(fmean(effects), float(summary["mean_switch_minus_threshold_f1"]), abs_tol=1e-12):
        failures.append("mean effect mismatch")
    return {
        "profile": "publication_confirmation_completeness_switch_audit",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "summary_sha256": sha256_file(summary_path),
        "formal_test_annotations_accessed": False,
    }

