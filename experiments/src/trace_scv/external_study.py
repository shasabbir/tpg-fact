from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

import numpy as np

from trace_scv.data.averimatec import TRAIN_ROLES, train_role
from trace_scv.external_reporting import (
    apply_holm_to_intervals,
    paired_claim_bootstrap,
    set_metrics,
    summarize_claim_rows,
)
from trace_scv.external_selection import select_evidence_indices
from trace_scv.selection.mocheg_study import LogisticEvidenceScorer
from trace_scv.selection.probability_calibration import (
    CALIBRATOR_KINDS,
    ProbabilityCalibrator,
    apply_calibrator,
    fit_calibrator,
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _features(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    values = np.asarray([row["features"] for row in rows], dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("candidate features must be a finite, non-empty matrix")
    return values


def scorer_to_dict(model: LogisticEvidenceScorer) -> dict[str, object]:
    return asdict(model)


def scorer_from_dict(row: Mapping[str, object]) -> LogisticEvidenceScorer:
    return LogisticEvidenceScorer(
        mean=tuple(map(float, row["mean"])),
        scale=tuple(map(float, row["scale"])),
        weights=tuple(map(float, row["weights"])),
        bias=float(row["bias"]),
    )


def calibrator_from_dict(row: Mapping[str, object]) -> ProbabilityCalibrator:
    return ProbabilityCalibrator(
        kind=str(row["kind"]),
        coefficients=tuple(map(float, row.get("coefficients", ()))),
        thresholds=tuple(map(float, row.get("thresholds", ()))),
        values=tuple(map(float, row.get("values", ()))),
    )


def _group(rows: Sequence[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = f"{row['split']}:{row['claim_id']}"
        grouped.setdefault(key, []).append(row)
    return grouped


def _role_rows(rows: Sequence[dict[str, object]], seed: int, role: str) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["split"] == "train" and train_role(str(row["claim_id"]), seed) == role
    ]


def _balanced_fit_rows(
    rows: Sequence[dict[str, object]], seed: int, maximum_negatives_per_claim: int
) -> list[dict[str, object]]:
    grouped = _group(rows)
    selected: list[dict[str, object]] = []
    for key in sorted(grouped):
        claim_rows = grouped[key]
        positives = [row for row in claim_rows if bool(row["gold"])]
        negatives = [row for row in claim_rows if not bool(row["gold"])]
        negatives.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['evidence_id']}".encode("utf-8")
            ).hexdigest()
        )
        selected.extend(positives)
        selected.extend(negatives[:maximum_negatives_per_claim])
    return selected


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))


def _packet_rows(
    rows: Sequence[dict[str, object]],
    probabilities: Sequence[float],
    *,
    text_width: int,
    image_width: int,
) -> tuple[list[dict[str, object]], list[float]]:
    paired = list(zip(rows, map(float, probabilities), strict=True))
    selected: list[tuple[dict[str, object], float]] = []
    for modality, width in (("text", text_width), ("image", image_width)):
        branch = [item for item in paired if item[0]["modality"] == modality]
        branch.sort(key=lambda item: (-item[1], str(item[0]["evidence_id"])))
        selected.extend(branch[:width])
    selected.sort(key=lambda item: (-item[1], str(item[0]["evidence_id"])))
    return [item[0] for item in selected], [item[1] for item in selected]


def _gold_sources(full_gold: Mapping[str, Sequence[str]], key: str) -> set[str]:
    return set(map(str, full_gold.get(key, ())))


def _policy_claim_rows(
    grouped: Mapping[str, Sequence[dict[str, object]]],
    raw_probabilities: Mapping[str, Sequence[float]],
    calibrated_probabilities: Mapping[str, Sequence[float]],
    full_gold: Mapping[str, Sequence[str]],
    *,
    method: str,
    text_width: int,
    image_width: int,
    maximum_items: int,
    fixed_k: int | None = None,
    threshold: float | None = None,
    raw_expected: bool = False,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = list(grouped[key])
        packet, packet_raw = _packet_rows(
            rows,
            raw_probabilities[key],
            text_width=text_width,
            image_width=image_width,
        )
        calibrated_by_id = {
            str(row["evidence_id"]): probability
            for row, probability in zip(rows, calibrated_probabilities[key], strict=True)
        }
        packet_calibrated = [calibrated_by_id[str(row["evidence_id"])] for row in packet]
        values = packet_raw if raw_expected else packet_calibrated
        selected_indices = select_evidence_indices(
            values,
            method=method,
            maximum_items=maximum_items,
            fixed_k=fixed_k,
            threshold=threshold,
            require_nonempty=True,
        )
        predicted = {str(packet[index]["normalized_source_url"]) for index in selected_indices}
        gold = _gold_sources(full_gold, key)
        split, claim_id = key.split(":", 1)
        output.append(
            {
                "split": split,
                "claim_id": claim_id,
                "selected_source_urls": sorted(predicted),
                "gold_source_urls": sorted(gold),
                **set_metrics(predicted, gold),
            }
        )
    return output


def _predict_by_claim(
    grouped: Mapping[str, Sequence[dict[str, object]]],
    model: LogisticEvidenceScorer,
    calibrator: ProbabilityCalibrator,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    raw: dict[str, list[float]] = {}
    calibrated: dict[str, list[float]] = {}
    for key, rows in grouped.items():
        values = model.predict_probabilities(_features(rows))
        raw[key] = values.tolist()
        calibrated[key] = apply_calibrator(values, calibrator).tolist()
    return raw, calibrated


def _oracle_packet_score(
    grouped: Mapping[str, Sequence[dict[str, object]]],
    raw: Mapping[str, Sequence[float]],
    full_gold: Mapping[str, Sequence[str]],
    text_width: int,
    image_width: int,
) -> float:
    scores: list[float] = []
    for key, rows in grouped.items():
        packet, _ = _packet_rows(rows, raw[key], text_width=text_width, image_width=image_width)
        predicted = {str(row["normalized_source_url"]) for row in packet}
        scores.append(set_metrics(predicted, _gold_sources(full_gold, key))["f1"])
    return fmean(scores) if scores else 0.0


def freeze_train_only_models(
    candidate_path: Path,
    full_gold_path: Path,
    *,
    seeds: Sequence[int],
    fixed_budgets: Sequence[int],
    thresholds: Sequence[float],
    text_widths: Sequence[int],
    image_widths: Sequence[int],
    maximum_items: int,
    maximum_fit_negatives_per_claim: int,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> dict[str, object]:
    records = read_jsonl(candidate_path)
    if any(row["split"] == "test" for row in records):
        raise RuntimeError("test candidates are forbidden during train-only freezing")
    full_gold = json.loads(full_gold_path.read_text(encoding="utf-8"))
    seed_models: list[dict[str, object]] = []
    for seed in seeds:
        fit_rows = _balanced_fit_rows(
            _role_rows(records, seed, "fit"), seed, maximum_fit_negatives_per_claim
        )
        fit_labels = np.asarray([int(bool(row["gold"])) for row in fit_rows], dtype=np.int8)
        model = LogisticEvidenceScorer.fit(
            _features(fit_rows),
            fit_labels,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
        )
        calibration_rows = _role_rows(records, seed, "calibration")
        calibration_raw = model.predict_probabilities(_features(calibration_rows))
        calibration_labels = np.asarray(
            [int(bool(row["gold"])) for row in calibration_rows], dtype=np.int8
        )
        guard_rows = _role_rows(records, seed, "model_selection")
        guard_raw = model.predict_probabilities(_features(guard_rows))
        guard_labels = np.asarray([int(bool(row["gold"])) for row in guard_rows], dtype=np.int8)
        calibrator_trials: list[tuple[float, str, ProbabilityCalibrator]] = []
        for kind in CALIBRATOR_KINDS:
            fitted = fit_calibrator(kind, calibration_raw, calibration_labels)
            calibrator_trials.append(
                (_brier(apply_calibrator(guard_raw, fitted), guard_labels), kind, fitted)
            )
        guard_brier, calibrator_kind, calibrator = min(
            calibrator_trials, key=lambda row: (row[0], row[1])
        )
        guard_grouped = _group(guard_rows)
        guard_raw_map, guard_calibrated_map = _predict_by_claim(guard_grouped, model, calibrator)
        packet_trials = [
            (
                _oracle_packet_score(guard_grouped, guard_raw_map, full_gold, text_width, image_width),
                text_width,
                image_width,
            )
            for text_width in text_widths
            for image_width in image_widths
        ]
        packet_score, text_width, image_width = max(
            packet_trials, key=lambda row: (row[0], -row[1], -row[2])
        )

        def evaluate(method: str, *, fixed_k: int | None = None, threshold: float | None = None) -> float:
            rows = _policy_claim_rows(
                guard_grouped,
                guard_raw_map,
                guard_calibrated_map,
                full_gold,
                method=method,
                text_width=text_width,
                image_width=image_width,
                maximum_items=maximum_items,
                fixed_k=fixed_k,
                threshold=threshold,
            )
            return summarize_claim_rows(rows)["f1"]

        fixed_k = max(fixed_budgets, key=lambda value: (evaluate("fixed", fixed_k=value), -value))
        threshold = max(
            thresholds,
            key=lambda value: (evaluate("threshold", threshold=value), value),
        )
        internal_rows = _role_rows(records, seed, "internal_holdout")
        internal_grouped = _group(internal_rows)
        internal_raw, internal_calibrated = _predict_by_claim(internal_grouped, model, calibrator)
        internal_metrics: dict[str, object] = {}
        policy_specs = {
            "fixed": ("fixed", {"fixed_k": fixed_k}),
            "calibrated_threshold": ("threshold", {"threshold": threshold}),
            "expected_f1": ("expected_f1", {}),
            "uncalibrated_expected_f1": ("expected_f1", {"raw_expected": True}),
        }
        for name, (method, kwargs) in policy_specs.items():
            claim_rows = _policy_claim_rows(
                internal_grouped,
                internal_raw,
                internal_calibrated,
                full_gold,
                method=method,
                text_width=text_width,
                image_width=image_width,
                maximum_items=maximum_items,
                **kwargs,
            )
            internal_metrics[name] = summarize_claim_rows(claim_rows)
        seed_models.append(
            {
                "seed": seed,
                "role_claim_counts": {
                    role: len({str(row["claim_id"]) for row in _role_rows(records, seed, role)})
                    for role in TRAIN_ROLES
                },
                "fit_candidate_rows": len(fit_rows),
                "fit_positive_rows": int(fit_labels.sum()),
                "scorer": scorer_to_dict(model),
                "calibrator": calibrator.to_dict(),
                "calibrator_kind": calibrator_kind,
                "guard_brier": guard_brier,
                "calibrator_trials": [
                    {"kind": kind, "guard_brier": score}
                    for score, kind, _ in sorted(calibrator_trials)
                ],
                "text_packet_width": text_width,
                "image_packet_width": image_width,
                "model_selection_packet_oracle_f1": packet_score,
                "fixed_k": fixed_k,
                "threshold": threshold,
                "internal_holdout": internal_metrics,
            }
        )
    return {
        "status": "frozen_train_only",
        "test_annotations_accessed": False,
        "development_outcomes_accessed": False,
        "seeds": seed_models,
    }


def run_locked_development(
    candidate_path: Path,
    full_gold_path: Path,
    freeze: Mapping[str, object],
    *,
    maximum_items: int,
    bootstrap_samples: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = [row for row in read_jsonl(candidate_path) if row["split"] == "validation"]
    full_gold = json.loads(full_gold_path.read_text(encoding="utf-8"))
    grouped = _group(records)
    all_rows: list[dict[str, object]] = []
    seed_summaries: list[dict[str, object]] = []
    for seed_row in freeze["seeds"]:
        seed = int(seed_row["seed"])
        model = scorer_from_dict(seed_row["scorer"])
        calibrator = calibrator_from_dict(seed_row["calibrator"])
        raw, calibrated = _predict_by_claim(grouped, model, calibrator)
        specs = {
            "fixed": ("fixed", {"fixed_k": int(seed_row["fixed_k"])}),
            "calibrated_threshold": (
                "threshold",
                {"threshold": float(seed_row["threshold"])},
            ),
            "expected_f1": ("expected_f1", {}),
            "uncalibrated_expected_f1": ("expected_f1", {"raw_expected": True}),
        }
        metrics: dict[str, object] = {}
        for policy, (method, kwargs) in specs.items():
            claim_rows = _policy_claim_rows(
                grouped,
                raw,
                calibrated,
                full_gold,
                method=method,
                text_width=int(seed_row["text_packet_width"]),
                image_width=int(seed_row["image_packet_width"]),
                maximum_items=maximum_items,
                **kwargs,
            )
            for row in claim_rows:
                row["seed"] = seed
                row["policy"] = policy
            all_rows.extend(claim_rows)
            metrics[policy] = summarize_claim_rows(claim_rows)
        seed_summaries.append({"seed": seed, "metrics": metrics})

    averaged: dict[str, list[dict[str, object]]] = {}
    for policy in sorted({str(row["policy"]) for row in all_rows}):
        policy_rows = [row for row in all_rows if row["policy"] == policy]
        by_claim: dict[str, list[dict[str, object]]] = {}
        for row in policy_rows:
            by_claim.setdefault(str(row["claim_id"]), []).append(row)
        averaged[policy] = [
            {
                "claim_id": claim_id,
                **{
                    metric: fmean(float(row[metric]) for row in rows)
                    for metric in ("precision", "recall", "f1", "exact", "selected_count")
                },
            }
            for claim_id, rows in sorted(by_claim.items())
        ]
    comparisons = {
        "threshold_minus_fixed": paired_claim_bootstrap(
            averaged["calibrated_threshold"],
            averaged["fixed"],
            samples=bootstrap_samples,
            seed=91_013,
        ),
        "expected_f1_minus_threshold": paired_claim_bootstrap(
            averaged["expected_f1"],
            averaged["calibrated_threshold"],
            samples=bootstrap_samples,
            seed=91_037,
        ),
        "calibrated_minus_uncalibrated_expected_f1": paired_claim_bootstrap(
            averaged["expected_f1"],
            averaged["uncalibrated_expected_f1"],
            samples=bootstrap_samples,
            seed=91_073,
        ),
    }
    comparisons = apply_holm_to_intervals(comparisons)
    aggregate = {policy: summarize_claim_rows(rows) for policy, rows in averaged.items()}
    threshold_count = aggregate["calibrated_threshold"]["selected_count"]
    expected_count = aggregate["expected_f1"]["selected_count"]
    summary = {
        "status": "checked_locked_development",
        "claims": len(grouped),
        "seeds": seed_summaries,
        "aggregate_mean_over_seed_then_claim": aggregate,
        "paired_claim_bootstrap": comparisons,
        "expected_f1_size_reduction_vs_threshold": (
            (threshold_count - expected_count) / threshold_count if threshold_count else 0.0
        ),
        "official_semantic_evaluator": {
            "status": "not_run",
            "score": None,
            "reason": "requires a separately checked Gemma-3-27B execution",
        },
        "test_annotations_accessed": False,
        "test_submission_performed": False,
    }
    return summary, all_rows
