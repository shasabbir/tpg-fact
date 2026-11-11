from __future__ import annotations

import hashlib
import json
import math
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
from trace_scv.external_verifier import (
    aggregate_evidence_features,
    classification_metrics,
    fit_external_mlp_verifier,
    fit_external_verifier,
    paired_macro_f1_bootstrap,
    seed_aligned_macro_f1_bootstrap,
    verifier_from_dict,
)
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


def _evaluation_id(row: Mapping[str, object]) -> str:
    return str(row.get("evaluation_id", row.get("normalized_source_url", "")))


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
    switch_probabilities: Mapping[str, float] | None = None,
    switch_threshold: float | None = None,
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
        selected_method = method
        selected_threshold = threshold
        switch_probability: float | None = None
        if method == "completeness_switch":
            if switch_probabilities is None or switch_threshold is None or threshold is None:
                raise ValueError("completeness switch requires probabilities and thresholds")
            switch_probability = float(switch_probabilities[key])
            if switch_probability >= switch_threshold:
                selected_method = "expected_f1"
                selected_threshold = None
            else:
                selected_method = "threshold"
        selected_indices = select_evidence_indices(
            values,
            method=selected_method,
            maximum_items=maximum_items,
            fixed_k=fixed_k,
            threshold=selected_threshold,
            require_nonempty=True,
        )
        predicted = {_evaluation_id(packet[index]) for index in selected_indices}
        selected_evidence_ids = [str(packet[index]["evidence_id"]) for index in selected_indices]
        gold = _gold_sources(full_gold, key)
        split, claim_id = key.split(":", 1)
        output.append(
            {
                "split": split,
                "claim_id": claim_id,
                "selected_source_urls": sorted(predicted),
                "selected_evidence_ids": selected_evidence_ids,
                "gold_source_urls": sorted(gold),
                "selection_route": selected_method,
                "completeness_probability": switch_probability,
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
        predicted = {_evaluation_id(row) for row in packet}
        scores.append(set_metrics(predicted, _gold_sources(full_gold, key))["f1"])
    return fmean(scores) if scores else 0.0


def _completeness_features_by_claim(
    grouped: Mapping[str, Sequence[dict[str, object]]],
    raw: Mapping[str, Sequence[float]],
    calibrated: Mapping[str, Sequence[float]],
    *,
    text_width: int,
    image_width: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key, rows in grouped.items():
        packet, packet_raw = _packet_rows(
            rows, raw[key], text_width=text_width, image_width=image_width
        )
        calibrated_by_id = {
            str(row["evidence_id"]): float(probability)
            for row, probability in zip(rows, calibrated[key], strict=True)
        }
        packet_calibrated = np.asarray(
            [calibrated_by_id[str(row["evidence_id"])] for row in packet],
            dtype=np.float64,
        )
        packet_raw_array = np.asarray(packet_raw, dtype=np.float64)

        def summary(values: np.ndarray) -> tuple[float, float, float]:
            if not len(values):
                return 0.0, 0.0, 0.0
            return float(values.max()), float(values.mean()), float(values.std())

        raw_max, raw_mean, raw_std = summary(packet_raw_array)
        calibrated_max, calibrated_mean, calibrated_std = summary(packet_calibrated)
        ordered = np.sort(packet_calibrated)[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else calibrated_max
        text_count = sum(row["modality"] == "text" for row in packet)
        image_count = sum(row["modality"] == "image" for row in packet)
        domains = {
            str(row.get("source_domain", ""))
            for row in packet
            if str(row.get("source_domain", ""))
        }
        denominator = max(1, len(packet))
        output[key] = np.asarray(
            [
                math.log1p(len(packet)),
                text_count / denominator,
                image_count / denominator,
                math.log1p(len(domains)),
                raw_max,
                raw_mean,
                raw_std,
                calibrated_max,
                calibrated_mean,
                calibrated_std,
                margin,
                float(packet_calibrated.sum()) / denominator,
            ],
            dtype=np.float64,
        )
    return output


def _completeness_labels(
    grouped: Mapping[str, Sequence[dict[str, object]]],
    raw: Mapping[str, Sequence[float]],
    full_gold: Mapping[str, Sequence[str]],
    *,
    text_width: int,
    image_width: int,
) -> dict[str, int]:
    labels: dict[str, int] = {}
    for key, rows in grouped.items():
        gold = _gold_sources(full_gold, key)
        if not gold:
            continue
        packet, _ = _packet_rows(
            rows, raw[key], text_width=text_width, image_width=image_width
        )
        targets = {_evaluation_id(row) for row in packet}
        labels[key] = int(gold <= targets)
    return labels


def _fit_completeness_model(
    features: Mapping[str, np.ndarray],
    labels: Mapping[str, int],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> dict[str, object]:
    keys = sorted(set(features) & set(labels))
    if not keys:
        raise ValueError("no claims are available for completeness fitting")
    values = np.stack([features[key] for key in keys])
    targets = np.asarray([labels[key] for key in keys], dtype=np.int8)
    if len(np.unique(targets)) == 1:
        return {
            "kind": "constant",
            "probability": float(targets[0]),
            "training_claims": len(keys),
            "positive_claims": int(targets.sum()),
        }
    model = LogisticEvidenceScorer.fit(
        values,
        targets,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    return {
        "kind": "logistic",
        "scorer": scorer_to_dict(model),
        "training_claims": len(keys),
        "positive_claims": int(targets.sum()),
    }


def _predict_completeness(
    model: Mapping[str, object], features: Mapping[str, np.ndarray]
) -> dict[str, float]:
    if model["kind"] == "constant":
        return {key: float(model["probability"]) for key in features}
    scorer = scorer_from_dict(model["scorer"])
    keys = sorted(features)
    probabilities = scorer.predict_probabilities(np.stack([features[key] for key in keys]))
    return {key: float(value) for key, value in zip(keys, probabilities, strict=True)}


def _fit_shared_verifier_for_seed(
    grouped: Mapping[str, Sequence[dict[str, object]]],
    model: LogisticEvidenceScorer,
    calibrator: ProbabilityCalibrator,
    full_gold: Mapping[str, Sequence[str]],
    completeness_model: Mapping[str, object],
    *,
    text_width: int,
    image_width: int,
    maximum_items: int,
    fixed_k: int,
    threshold: float,
    switch_threshold: float,
    learning_rate: float,
    epochs: int,
    l2: float,
    seed: int,
) -> dict[str, object]:
    if any(not str(rows[0].get("claim_label", "")) for rows in grouped.values()):
        return {
            "status": "not_fitted",
            "reason": "candidate records do not contain released training verdict labels",
        }
    raw, calibrated = _predict_by_claim(grouped, model, calibrator)
    completeness_features = _completeness_features_by_claim(
        grouped,
        raw,
        calibrated,
        text_width=text_width,
        image_width=image_width,
    )
    completeness_probabilities = _predict_completeness(
        completeness_model, completeness_features
    )
    specs = {
        "fixed": ("fixed", {"fixed_k": fixed_k}),
        "calibrated_threshold": ("threshold", {"threshold": threshold}),
        "expected_f1": ("expected_f1", {}),
        "completeness_aware_switch": (
            "completeness_switch",
            {
                "threshold": threshold,
                "switch_probabilities": completeness_probabilities,
                "switch_threshold": switch_threshold,
            },
        ),
    }
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    seen: dict[str, set[tuple[str, ...]]] = {}
    for _, (method, kwargs) in specs.items():
        selections = _policy_claim_rows(
            grouped,
            raw,
            calibrated,
            full_gold,
            method=method,
            text_width=text_width,
            image_width=image_width,
            maximum_items=maximum_items,
            **kwargs,
        )
        for selection in selections:
            key = f"{selection['split']}:{selection['claim_id']}"
            identity = tuple(sorted(map(str, selection["selected_evidence_ids"])))
            if identity in seen.setdefault(key, set()):
                continue
            seen[key].add(identity)
            feature_rows.append(
                aggregate_evidence_features(grouped[key], selection["selected_evidence_ids"])
            )
            labels.append(str(grouped[key][0]["claim_label"]))
    try:
        linear_verifier = fit_external_verifier(
            np.stack(feature_rows),
            labels,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
        )
        verifier = fit_external_mlp_verifier(
            np.stack(feature_rows),
            labels,
            seed=seed,
            hidden_units=64,
            learning_rate=min(learning_rate, 0.01),
            epochs=max(300, epochs),
            l2=l2,
        )
    except ValueError as error:
        return {
            "status": "not_fitted",
            "reason": str(error),
            "training_examples": len(feature_rows),
        }
    return {
        "status": "frozen_train_only",
        "model": verifier.to_dict(),
        "linear_control": linear_verifier.to_dict(),
        "training_examples": len(feature_rows),
        "training_claims": len(grouped),
        "policies": sorted(specs),
    }


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
        calibration_grouped = _group(calibration_rows)
        calibration_raw_map, calibration_calibrated_map = _predict_by_claim(
            calibration_grouped, model, calibrator
        )
        completeness_training_features = _completeness_features_by_claim(
            calibration_grouped,
            calibration_raw_map,
            calibration_calibrated_map,
            text_width=text_width,
            image_width=image_width,
        )
        completeness_training_labels = _completeness_labels(
            calibration_grouped,
            calibration_raw_map,
            full_gold,
            text_width=text_width,
            image_width=image_width,
        )
        completeness_model = _fit_completeness_model(
            completeness_training_features,
            completeness_training_labels,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
        )
        guard_completeness_features = _completeness_features_by_claim(
            guard_grouped,
            guard_raw_map,
            guard_calibrated_map,
            text_width=text_width,
            image_width=image_width,
        )
        guard_completeness_probabilities = _predict_completeness(
            completeness_model, guard_completeness_features
        )

        def evaluate_switch(value: float) -> float:
            rows = _policy_claim_rows(
                guard_grouped,
                guard_raw_map,
                guard_calibrated_map,
                full_gold,
                method="completeness_switch",
                text_width=text_width,
                image_width=image_width,
                maximum_items=maximum_items,
                threshold=threshold,
                switch_probabilities=guard_completeness_probabilities,
                switch_threshold=value,
            )
            return summarize_claim_rows(rows)["f1"]

        switch_threshold = max(
            thresholds,
            key=lambda value: (evaluate_switch(value), value),
        )
        internal_rows = _role_rows(records, seed, "internal_holdout")
        internal_grouped = _group(internal_rows)
        internal_raw, internal_calibrated = _predict_by_claim(internal_grouped, model, calibrator)
        internal_completeness_features = _completeness_features_by_claim(
            internal_grouped,
            internal_raw,
            internal_calibrated,
            text_width=text_width,
            image_width=image_width,
        )
        internal_completeness_probabilities = _predict_completeness(
            completeness_model, internal_completeness_features
        )
        internal_metrics: dict[str, object] = {}
        policy_specs = {
            "fixed": ("fixed", {"fixed_k": fixed_k}),
            "calibrated_threshold": ("threshold", {"threshold": threshold}),
            "expected_f1": ("expected_f1", {}),
            "uncalibrated_expected_f1": ("expected_f1", {"raw_expected": True}),
            "completeness_aware_switch": (
                "completeness_switch",
                {
                    "threshold": threshold,
                    "switch_probabilities": internal_completeness_probabilities,
                    "switch_threshold": switch_threshold,
                },
            ),
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
        all_training_grouped = _group([row for row in records if row["split"] == "train"])
        shared_verifier = _fit_shared_verifier_for_seed(
            all_training_grouped,
            model,
            calibrator,
            full_gold,
            completeness_model,
            text_width=text_width,
            image_width=image_width,
            maximum_items=maximum_items,
            fixed_k=fixed_k,
            threshold=threshold,
            switch_threshold=switch_threshold,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
            seed=seed,
        )
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
                "completeness_model": completeness_model,
                "completeness_switch_threshold": switch_threshold,
                "shared_verifier": shared_verifier,
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
    verifier_runs: list[tuple[Sequence[str], Sequence[str], Sequence[str]]] = []
    for seed_row in freeze["seeds"]:
        seed = int(seed_row["seed"])
        model = scorer_from_dict(seed_row["scorer"])
        calibrator = calibrator_from_dict(seed_row["calibrator"])
        raw, calibrated = _predict_by_claim(grouped, model, calibrator)
        completeness_features = _completeness_features_by_claim(
            grouped,
            raw,
            calibrated,
            text_width=int(seed_row["text_packet_width"]),
            image_width=int(seed_row["image_packet_width"]),
        )
        completeness_probabilities = _predict_completeness(
            seed_row["completeness_model"], completeness_features
        )
        specs = {
            "fixed": ("fixed", {"fixed_k": int(seed_row["fixed_k"])}),
            "calibrated_threshold": (
                "threshold",
                {"threshold": float(seed_row["threshold"])},
            ),
            "expected_f1": ("expected_f1", {}),
            "uncalibrated_expected_f1": ("expected_f1", {"raw_expected": True}),
            "completeness_aware_switch": (
                "completeness_switch",
                {
                    "threshold": float(seed_row["threshold"]),
                    "switch_probabilities": completeness_probabilities,
                    "switch_threshold": float(seed_row["completeness_switch_threshold"]),
                },
            ),
        }
        metrics: dict[str, object] = {}
        policy_claim_rows: dict[str, list[dict[str, object]]] = {}
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
            policy_claim_rows[policy] = claim_rows
            metrics[policy] = summarize_claim_rows(claim_rows)
        verifier_summary: dict[str, object]
        verifier_freeze = seed_row.get("shared_verifier", {})
        if verifier_freeze.get("status") == "frozen_train_only":
            verifier = verifier_from_dict(verifier_freeze["model"])
            verdict_predictions: dict[str, list[str]] = {}
            verdict_probabilities: dict[str, np.ndarray] = {}
            verdict_gold: list[str] = []
            evaluated_policies = ("calibrated_threshold", "completeness_aware_switch")
            for policy in evaluated_policies:
                rows = policy_claim_rows[policy]
                aggregates: list[np.ndarray] = []
                current_gold: list[str] = []
                for row in rows:
                    key = f"{row['split']}:{row['claim_id']}"
                    aggregates.append(
                        aggregate_evidence_features(grouped[key], row["selected_evidence_ids"])
                    )
                    current_gold.append(str(grouped[key][0]["claim_label"]))
                probabilities = verifier.predict_probabilities(np.stack(aggregates))
                predictions = [verifier.labels[int(index)] for index in probabilities.argmax(axis=1)]
                verdict_predictions[policy] = predictions
                verdict_probabilities[policy] = probabilities
                if not verdict_gold:
                    verdict_gold = current_gold
                elif verdict_gold != current_gold:
                    raise RuntimeError("verifier policy rows are not claim-aligned")
                for index, row in enumerate(rows):
                    row["gold_verdict"] = current_gold[index]
                    row["predicted_verdict"] = predictions[index]
                    row["verdict_probabilities"] = {
                        label: float(probabilities[index, label_index])
                        for label_index, label in enumerate(verifier.labels)
                    }
            threshold_predictions = verdict_predictions["calibrated_threshold"]
            switch_predictions = verdict_predictions["completeness_aware_switch"]
            paired = paired_macro_f1_bootstrap(
                verdict_gold,
                switch_predictions,
                threshold_predictions,
                samples=bootstrap_samples,
                seed=191_000 + seed,
            )
            verifier_runs.append((verdict_gold, switch_predictions, threshold_predictions))
            verifier_summary = {
                "status": "checked_locked_development",
                "metrics": {
                    policy: classification_metrics(verdict_gold, predictions)
                    for policy, predictions in verdict_predictions.items()
                },
                "switch_minus_threshold_macro_f1": paired,
            }
        else:
            verifier_summary = {
                "status": "not_run",
                "reason": verifier_freeze.get("reason", "shared verifier was not frozen"),
            }
        seed_summaries.append(
            {"seed": seed, "metrics": metrics, "shared_verifier": verifier_summary}
        )

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
            family_size=4,
        ),
        "expected_f1_minus_threshold": paired_claim_bootstrap(
            averaged["expected_f1"],
            averaged["calibrated_threshold"],
            samples=bootstrap_samples,
            seed=91_037,
            family_size=4,
        ),
        "calibrated_minus_uncalibrated_expected_f1": paired_claim_bootstrap(
            averaged["expected_f1"],
            averaged["uncalibrated_expected_f1"],
            samples=bootstrap_samples,
            seed=91_073,
            family_size=4,
        ),
        "completeness_switch_minus_threshold": paired_claim_bootstrap(
            averaged["completeness_aware_switch"],
            averaged["calibrated_threshold"],
            samples=bootstrap_samples,
            seed=91_101,
            family_size=4,
        ),
    }
    comparisons = apply_holm_to_intervals(comparisons)
    aggregate = {policy: summarize_claim_rows(rows) for policy, rows in averaged.items()}
    threshold_count = aggregate["calibrated_threshold"]["selected_count"]
    expected_count = aggregate["expected_f1"]["selected_count"]
    switch_effects = [
        float(row["metrics"]["completeness_aware_switch"]["f1"])
        - float(row["metrics"]["calibrated_threshold"]["f1"])
        for row in seed_summaries
    ]
    switch_comparison = comparisons["completeness_switch_minus_threshold"]
    switch_gate = (
        float(switch_comparison["mean"]) >= 0.010
        and float(switch_comparison["lower_adjusted"]) > 0.0
        and all(value > 0.0 for value in switch_effects)
        and aggregate["completeness_aware_switch"]["selected_count"]
        <= aggregate["calibrated_threshold"]["selected_count"]
    )
    verifier_aggregate = (
        seed_aligned_macro_f1_bootstrap(
            verifier_runs,
            samples=bootstrap_samples,
            seed=291_013,
        )
        if verifier_runs
        else None
    )
    summary = {
        "status": "checked_locked_development",
        "claims": len(grouped),
        "seeds": seed_summaries,
        "aggregate_mean_over_seed_then_claim": aggregate,
        "paired_claim_bootstrap": comparisons,
        "expected_f1_size_reduction_vs_threshold": (
            (threshold_count - expected_count) / threshold_count if threshold_count else 0.0
        ),
        "completeness_switch_superiority_gate": switch_gate,
        "completeness_switch_seed_effects": switch_effects,
        "shared_verifier": {
            "status": "checked_locked_development" if verifier_aggregate else "not_run",
            "aggregate_switch_minus_threshold_macro_f1": verifier_aggregate,
            "noninferiority_margin": -0.010,
            "noninferiority_pass": (
                float(verifier_aggregate["lower"]) > -0.010
                if verifier_aggregate is not None
                else False
            ),
        },
        "official_semantic_evaluator": {
            "status": "not_run",
            "score": None,
            "reason": "requires a separately checked Gemma-3-27B execution",
        },
        "test_annotations_accessed": False,
        "test_submission_performed": False,
    }
    return summary, all_rows
