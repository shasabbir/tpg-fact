from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import numpy as np
import yaml

from trace_scv.data.io import sha256_file
from trace_scv.data.mocheg import assert_mocheg_development_sealed
from trace_scv.q1_phase2_retrieval import _claim_role, _load_sentence_rows


def _source_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def weighted_rank_fusion(
    first: Sequence[int],
    second: Sequence[int],
    *,
    first_weight: float,
    constant: int,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= first_weight <= 1.0:
        raise ValueError("first_weight must be in [0, 1]")
    if constant <= 0 or depth <= 0:
        raise ValueError("constant and depth must be positive")
    scores: dict[int, float] = defaultdict(float)
    for weight, ranking in ((first_weight, first), (1.0 - first_weight, second)):
        seen: set[int] = set()
        for rank, raw_index in enumerate(ranking, start=1):
            index = int(raw_index)
            if index in seen:
                continue
            seen.add(index)
            scores[index] += weight / (constant + rank)
    ordered = sorted(scores, key=lambda index: (-scores[index], index))[:depth]
    return (
        np.asarray(ordered, dtype=np.int64),
        np.asarray([scores[index] for index in ordered], dtype=np.float32),
    )


def weighted_rank_fusion_matrix(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_weight: float,
    constant: int,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    if first.shape[0] != second.shape[0]:
        raise ValueError("ranking matrices must contain the same claims")
    indices = np.empty((first.shape[0], depth), dtype=np.int64)
    scores = np.empty((first.shape[0], depth), dtype=np.float32)
    for claim_index in range(first.shape[0]):
        row_indices, row_scores = weighted_rank_fusion(
            first[claim_index],
            second[claim_index],
            first_weight=first_weight,
            constant=constant,
            depth=depth,
        )
        if len(row_indices) != depth:
            raise ValueError(f"fusion produced only {len(row_indices)} candidates")
        indices[claim_index] = row_indices
        scores[claim_index] = row_scores
    return indices, scores


def _average_precision(ranking: Sequence[str], gold: set[str]) -> float:
    hits = 0
    total = 0.0
    for rank, evidence_id in enumerate(ranking, start=1):
        if evidence_id in gold:
            hits += 1
            total += hits / rank
    return total / len(gold) if gold else 0.0


def _ndcg(ranking: Sequence[str], gold: set[str]) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, evidence_id in enumerate(ranking, start=1)
        if evidence_id in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), len(ranking)) + 1))
    return dcg / ideal if ideal else 0.0


def _population_matches(claim: dict[str, object], population: str, partition_seed: int) -> bool:
    split = str(claim["split"])
    if population == "all_eligible":
        return True
    if population == "validation":
        return split == "validation"
    return split == "train" and _claim_role(str(claim["claim_id"]), partition_seed) == population


def _modality_metrics(
    claims: Sequence[dict[str, object]],
    rankings: np.ndarray,
    evidence_ids: Sequence[str],
    full_gold: dict[str, set[str]],
    *,
    modality: str,
    depth: int,
    population: str,
    partition_seed: int,
) -> dict[str, float]:
    recalls: list[float] = []
    average_precisions: list[float] = []
    ndcgs: list[float] = []
    for claim_index, claim in enumerate(claims):
        if not _population_matches(claim, population, partition_seed):
            continue
        key = f"{claim['split']}:{claim['claim_id']}"
        gold = {item for item in full_gold.get(key, set()) if item.startswith(f"{modality}:")}
        if not gold:
            continue
        ranking = [evidence_ids[int(index)] for index in rankings[claim_index]]
        recalls.append(len(set(ranking[:depth]) & gold) / len(gold))
        average_precisions.append(_average_precision(ranking, gold))
        ndcgs.append(_ndcg(ranking, gold))
    return {
        "eligible_claims": float(len(recalls)),
        f"recall_at_{depth}": fmean(recalls) if recalls else 0.0,
        "map_at_100": fmean(average_precisions) if average_precisions else 0.0,
        "ndcg_at_100": fmean(ndcgs) if ndcgs else 0.0,
    }


def _packet_metrics(
    claims: Sequence[dict[str, object]],
    text_rankings: np.ndarray,
    image_rankings: np.ndarray,
    text_ids: Sequence[str],
    image_ids: Sequence[str],
    full_gold: dict[str, set[str]],
    *,
    text_depth: int,
    image_depth: int,
    population: str,
    partition_seed: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    text_recalls: list[float] = []
    image_recalls: list[float] = []
    oracle_f1s: list[float] = []
    complete: list[float] = []
    zero_reachable = 0
    detail: list[dict[str, object]] = []
    for claim_index, claim in enumerate(claims):
        if not _population_matches(claim, population, partition_seed):
            continue
        key = f"{claim['split']}:{claim['claim_id']}"
        gold = full_gold.get(key, set())
        if not gold:
            continue
        text_packet = {
            text_ids[int(index)] for index in text_rankings[claim_index][:text_depth]
        }
        image_packet = {
            image_ids[int(index)] for index in image_rankings[claim_index][:image_depth]
        }
        packet = text_packet | image_packet
        reachable = packet & gold
        text_gold = {item for item in gold if item.startswith("text:")}
        image_gold = {item for item in gold if item.startswith("image:")}
        if text_gold:
            text_recalls.append(len(text_packet & text_gold) / len(text_gold))
        if image_gold:
            image_recalls.append(len(image_packet & image_gold) / len(image_gold))
        oracle = 2.0 * len(reachable) / (len(reachable) + len(gold))
        is_complete = reachable == gold
        oracle_f1s.append(oracle)
        complete.append(float(is_complete))
        zero_reachable += int(not reachable)
        detail.append(
            {
                "split": str(claim["split"]),
                "claim_id": str(claim["claim_id"]),
                "gold_count": len(gold),
                "reachable_gold_count": len(reachable),
                "complete_gold_reachable": is_complete,
                "oracle_f1": oracle,
            }
        )
    return (
        {
            "eligible_claims": float(len(oracle_f1s)),
            "text_eligible_claims": float(len(text_recalls)),
            "image_eligible_claims": float(len(image_recalls)),
            f"text_recall_at_{text_depth}": fmean(text_recalls) if text_recalls else 0.0,
            f"image_recall_at_{image_depth}": fmean(image_recalls) if image_recalls else 0.0,
            "packet_oracle_f1": fmean(oracle_f1s) if oracle_f1s else 0.0,
            "complete_gold_claim_coverage": fmean(complete) if complete else 0.0,
            "eligible_claims_with_zero_reachable_gold": float(zero_reachable),
        },
        detail,
    )


def _weight_label(weight: float) -> str:
    return f"w{int(round(weight * 100)):03d}"


def metric_csv_fields(text_depth: int, image_depth: int) -> list[str]:
    return [
        "modality",
        "weight",
        "eligible_claims",
        f"recall_at_{text_depth}",
        f"recall_at_{image_depth}",
        "map_at_100",
        "ndcg_at_100",
    ]


def run_publication_retrieval(
    project_root: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["study"]["formal_test_annotations_allowed"]:
        raise RuntimeError("formal-test annotations are not allowed")
    experiment_root = project_root / "experiments"
    dataset_root = experiment_root / protocol["data"]["dataset_root"]
    assert_mocheg_development_sealed(dataset_root)
    frozen_root = experiment_root / protocol["frozen_inputs"]["retrieval_root"]
    cache_root = experiment_root / protocol["frozen_inputs"]["cache_root"]
    processed_root = dataset_root
    output_root.mkdir(parents=True, exist_ok=True)

    claims = _read_jsonl(frozen_root / "claims.jsonl")
    images = _read_jsonl(cache_root / "images.jsonl")
    sentences = _load_sentence_rows(processed_root)
    text_ids = [str(row["evidence_id"]) for row in sentences]
    image_ids = [str(row["evidence_id"]) for row in images]
    source = np.load(frozen_root / "rankings.npz", mmap_mode="r")
    weights = [float(value) for value in protocol["retrieval"]["train_only_weight_candidates"]]
    constant = int(protocol["retrieval"]["rrf_constant"])
    audit_depth = int(protocol["retrieval"]["audit_depth"])
    text_depth = int(protocol["retrieval"]["retained_text_candidates"])
    image_depth = int(protocol["retrieval"]["retained_image_candidates"])
    partition_seed = int(protocol["data"]["train_partition_seed"])

    fused: dict[str, np.ndarray] = {}
    fused_scores: dict[str, np.ndarray] = {}
    for weight in weights:
        label = _weight_label(weight)
        indices, scores = weighted_rank_fusion_matrix(
            source["text_rrf_indices"],
            source["text_cross_indices"],
            first_weight=weight,
            constant=constant,
            depth=audit_depth,
        )
        fused[f"text_{label}"] = indices
        fused_scores[f"text_{label}_scores"] = scores
        indices, scores = weighted_rank_fusion_matrix(
            source["image_siglip_indices"],
            source["image_filename_indices"],
            first_weight=weight,
            constant=constant,
            depth=audit_depth,
        )
        fused[f"image_{label}"] = indices
        fused_scores[f"image_{label}_scores"] = scores

    ranking_path = output_root / "weighted_rankings.npz"
    np.savez_compressed(ranking_path, **fused, **fused_scores)
    claim_path = output_root / "claims.jsonl"
    _write_jsonl(claim_path, claims)
    prejoin = {
        "profile": "publication_confirmation_label_blind_weighted_rankings",
        "status": "rankings_frozen_before_gold_join",
        "protocol_sha256": sha256_file(protocol_path),
        "source_rankings_sha256": sha256_file(frozen_root / "rankings.npz"),
        "weights": weights,
        "constant": constant,
        "claims": len(claims),
        "gold_labels_used": False,
        "formal_test_annotations_accessed": False,
        "artifacts": {
            "rankings": {"path": str(ranking_path), "sha256": sha256_file(ranking_path)},
            "claims": {"path": str(claim_path), "sha256": sha256_file(claim_path)},
        },
        "source_commit": _source_commit(project_root),
    }
    prejoin_path = output_root / "weighted_rankings.prejoin.manifest.json"
    prejoin_path.write_text(json.dumps(prejoin, indent=2) + "\n", encoding="utf-8")

    # Gold is loaded only after all candidate rankings are durable and hashed.
    full_gold_raw = json.loads((frozen_root / "full_gold.json").read_text(encoding="utf-8"))
    full_gold = {str(key): set(map(str, values)) for key, values in full_gold_raw.items()}

    tuning_rows: list[dict[str, object]] = []
    text_metrics: dict[float, dict[str, float]] = {}
    image_metrics: dict[float, dict[str, float]] = {}
    for weight in weights:
        label = _weight_label(weight)
        text_metrics[weight] = _modality_metrics(
            claims,
            fused[f"text_{label}"],
            text_ids,
            full_gold,
            modality="text",
            depth=text_depth,
            population="model_selection",
            partition_seed=partition_seed,
        )
        image_metrics[weight] = _modality_metrics(
            claims,
            fused[f"image_{label}"],
            image_ids,
            full_gold,
            modality="image",
            depth=image_depth,
            population="model_selection",
            partition_seed=partition_seed,
        )
        tuning_rows.append({"modality": "text", "weight": weight, **text_metrics[weight]})
        tuning_rows.append({"modality": "image", "weight": weight, **image_metrics[weight]})

    selected_text_weight = max(
        weights,
        key=lambda value: (
            text_metrics[value][f"recall_at_{text_depth}"],
            text_metrics[value]["map_at_100"],
            -abs(value - 0.5),
        ),
    )
    selected_image_weight = max(
        weights,
        key=lambda value: (
            image_metrics[value][f"recall_at_{image_depth}"],
            image_metrics[value]["map_at_100"],
            -abs(value - 0.5),
        ),
    )
    selected_text = fused[f"text_{_weight_label(selected_text_weight)}"]
    selected_image = fused[f"image_{_weight_label(selected_image_weight)}"]
    baseline_text = source["text_cross_indices"]
    baseline_image = source["image_rrf_indices"]

    populations = ["model_selection", "internal_holdout", "validation", "all_eligible"]
    population_results: dict[str, dict[str, dict[str, float]]] = {}
    validation_detail: list[dict[str, object]] = []
    for population in populations:
        selected_metrics, detail = _packet_metrics(
            claims,
            selected_text,
            selected_image,
            text_ids,
            image_ids,
            full_gold,
            text_depth=text_depth,
            image_depth=image_depth,
            population=population,
            partition_seed=partition_seed,
        )
        baseline_metrics, _ = _packet_metrics(
            claims,
            baseline_text,
            baseline_image,
            text_ids,
            image_ids,
            full_gold,
            text_depth=text_depth,
            image_depth=image_depth,
            population=population,
            partition_seed=partition_seed,
        )
        population_results[population] = {
            "baseline": baseline_metrics,
            "selected_weighted_fusion": selected_metrics,
            "difference": {
                key: selected_metrics[key] - baseline_metrics[key]
                for key in (
                    f"text_recall_at_{text_depth}",
                    f"image_recall_at_{image_depth}",
                    "packet_oracle_f1",
                    "complete_gold_claim_coverage",
                    "eligible_claims_with_zero_reachable_gold",
                )
            },
        }
        if population == "validation":
            validation_detail = detail

    validation = population_results["validation"]
    selected_validation = validation["selected_weighted_fusion"]
    baseline_validation = validation["baseline"]
    gate = {
        "packet_oracle_f1_improved": selected_validation["packet_oracle_f1"]
        > baseline_validation["packet_oracle_f1"],
        "text_recall_nonworse": selected_validation[f"text_recall_at_{text_depth}"]
        >= baseline_validation[f"text_recall_at_{text_depth}"],
        "image_recall_nonworse": selected_validation[f"image_recall_at_{image_depth}"]
        >= baseline_validation[f"image_recall_at_{image_depth}"],
        "complete_gold_coverage_nonworse": selected_validation["complete_gold_claim_coverage"]
        >= baseline_validation["complete_gold_claim_coverage"],
    }
    gate["pass"] = all(gate.values())
    engineering = protocol["retrieval"]["engineering_targets"]
    gate["engineering_target_pass"] = (
        validation["difference"]["packet_oracle_f1"]
        >= float(engineering["minimum_absolute_packet_oracle_f1_gain"])
        and selected_validation["complete_gold_claim_coverage"]
        >= float(engineering["complete_gold_claim_coverage"])
    )

    selected_path = output_root / "selected_rankings.npz"
    np.savez_compressed(
        selected_path,
        text_indices=selected_text,
        image_indices=selected_image,
    )
    _write_jsonl(output_root / "validation_packet_oracle.jsonl", validation_detail)
    csv_path = output_root / "weight_selection_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=metric_csv_fields(text_depth, image_depth),
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(tuning_rows)

    result: dict[str, object] = {
        "profile": "publication_confirmation_weighted_retrieval",
        "status": "complete",
        "selection_population": "train:model_selection",
        "selected_text_rrf_weight": selected_text_weight,
        "selected_image_siglip_weight": selected_image_weight,
        "tuning_rows": tuning_rows,
        "population_results": population_results,
        "gate": gate,
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "artifacts": {
            "prejoin_manifest": {"path": str(prejoin_path), "sha256": sha256_file(prejoin_path)},
            "weighted_rankings": {"path": str(ranking_path), "sha256": sha256_file(ranking_path)},
            "selected_rankings": {"path": str(selected_path), "sha256": sha256_file(selected_path)},
            "weight_metrics": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
        },
        "formal_test_annotations_accessed": False,
        "formal_test_unlocked": False,
    }
    result_path = output_root / "retrieval_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert_mocheg_development_sealed(dataset_root)
    return result


def audit_publication_retrieval(output_root: Path) -> dict[str, object]:
    result_path = output_root / "retrieval_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if result.get("formal_test_annotations_accessed"):
        failures.append("formal-test annotations reported accessed")
    if result.get("formal_test_unlocked"):
        failures.append("formal test unexpectedly unlocked")
    for key, artifact in result["artifacts"].items():
        path = Path(str(artifact["path"]))
        if not path.exists():
            failures.append(f"missing artifact {key}: {path}")
        elif sha256_file(path) != artifact["sha256"]:
            failures.append(f"hash mismatch for {key}")
    prejoin = json.loads(
        Path(str(result["artifacts"]["prejoin_manifest"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if prejoin.get("gold_labels_used"):
        failures.append("prejoin manifest reports gold-label use")
    if prejoin.get("formal_test_annotations_accessed"):
        failures.append("prejoin manifest reports test access")
    validation = result["population_results"]["validation"]
    for metric in (
        "text_recall_at_40",
        "image_recall_at_20",
        "packet_oracle_f1",
        "complete_gold_claim_coverage",
    ):
        observed = (
            float(validation["selected_weighted_fusion"][metric])
            - float(validation["baseline"][metric])
        )
        if not math.isclose(observed, float(validation["difference"][metric]), abs_tol=1e-12):
            failures.append(f"difference mismatch for {metric}")
    return {
        "profile": "publication_confirmation_retrieval_audit",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "result_sha256": sha256_file(result_path),
        "formal_test_annotations_accessed": False,
    }

