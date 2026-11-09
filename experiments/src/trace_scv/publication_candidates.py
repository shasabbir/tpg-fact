from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from trace_scv.data.io import sha256_file
from trace_scv.data.mocheg import assert_mocheg_development_sealed
from trace_scv.publication_retrieval import (
    _modality_metrics,
    _packet_metrics,
    _read_jsonl,
)
from trace_scv.q1_phase2_retrieval import _load_sentence_rows


def rank_map(
    indices: np.ndarray, scores: np.ndarray, claim_index: int
) -> dict[int, tuple[int, float]]:
    return {
        int(index): (rank, float(score))
        for rank, (index, score) in enumerate(
            zip(indices[claim_index], scores[claim_index], strict=True), start=1
        )
    }


def load_npz_arrays(path: Path, names: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]) for name in names}


def _source_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _weight_label(weight: float) -> str:
    return f"w{int(round(weight * 100)):03d}"


def build_publication_candidates(
    project_root: Path,
    protocol_path: Path,
    retrieval_result_root: Path,
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
    retrieval_result = json.loads(
        (retrieval_result_root / "retrieval_results.json").read_text(encoding="utf-8")
    )
    retrieval_audit = json.loads(
        (retrieval_result_root / "retrieval.audit.json").read_text(encoding="utf-8")
    )
    if retrieval_audit.get("status") != "pass":
        raise RuntimeError("weighted retrieval must pass its audit")
    if retrieval_result.get("formal_test_annotations_accessed"):
        raise RuntimeError("weighted retrieval reports formal-test access")

    output_root.mkdir(parents=True, exist_ok=True)
    claims = _read_jsonl(frozen_root / "claims.jsonl")
    images = _read_jsonl(cache_root / "images.jsonl")
    sentences = _load_sentence_rows(dataset_root)
    text_ids = [str(row["evidence_id"]) for row in sentences]
    image_ids = [str(row["evidence_id"]) for row in images]
    text_depth = int(protocol["retrieval"]["retained_text_candidates"])
    image_depth = int(protocol["retrieval"]["retained_image_candidates"])
    partition_seed = int(protocol["data"]["train_partition_seed"])

    source = load_npz_arrays(
        frozen_root / "rankings.npz",
        (
            "text_dense_indices",
            "text_dense_scores",
            "text_bm25_indices",
            "text_bm25_scores",
            "text_rrf_indices",
            "text_rrf_scores",
            "text_cross_indices",
            "text_cross_scores",
            "image_siglip_indices",
            "image_siglip_scores",
            "image_filename_indices",
            "image_filename_scores",
            "image_rrf_indices",
            "image_rrf_scores",
        ),
    )
    text_weight = float(retrieval_result["selected_text_rrf_weight"])
    image_weight = float(retrieval_result["selected_image_siglip_weight"])
    text_label = _weight_label(text_weight)
    image_label = _weight_label(image_weight)
    selected_arrays = load_npz_arrays(
        retrieval_result_root / "selected_rankings.npz",
        ("text_indices", "image_indices"),
    )
    weighted_arrays = load_npz_arrays(
        retrieval_result_root / "weighted_rankings.npz",
        (f"text_{text_label}_scores", f"image_{image_label}_scores"),
    )
    selected_text = selected_arrays["text_indices"]
    selected_image = selected_arrays["image_indices"]
    text_final_scores = weighted_arrays[f"text_{text_label}_scores"]
    image_final_scores = weighted_arrays[f"image_{image_label}_scores"]

    full_gold_raw = json.loads((frozen_root / "full_gold.json").read_text(encoding="utf-8"))
    full_gold = {str(key): set(map(str, values)) for key, values in full_gold_raw.items()}
    feature_names = [
        "final_score",
        "final_score_squared",
        "dense_score",
        "lexical_or_filename_score",
        "rrf_score",
        "cross_encoder_score",
        "reciprocal_dense_rank",
        "reciprocal_lexical_rank",
        "reciprocal_final_rank",
        "is_text",
        "is_image",
        "log_text_token_count",
        "log_claim_token_count",
        "log_available_candidate_count",
        "log_duplicate_group_size",
    ]
    features = np.empty(
        (len(claims) * (text_depth + image_depth), len(feature_names)), dtype=np.float32
    )
    labels = np.empty(len(features), dtype=np.int8)
    candidate_path = output_root / "candidate_records.jsonl"
    cursor = 0
    with candidate_path.open("w", encoding="utf-8", newline="\n") as handle:
        for claim_index, claim in enumerate(claims):
            key = f"{claim['split']}:{claim['claim_id']}"
            gold = full_gold.get(key, set())
            text_maps = {
                "dense": rank_map(source["text_dense_indices"], source["text_dense_scores"], claim_index),
                "lexical": rank_map(source["text_bm25_indices"], source["text_bm25_scores"], claim_index),
                "rrf": rank_map(source["text_rrf_indices"], source["text_rrf_scores"], claim_index),
                "cross": rank_map(source["text_cross_indices"], source["text_cross_scores"], claim_index),
                "final": rank_map(selected_text, text_final_scores, claim_index),
            }
            image_maps = {
                "dense": rank_map(source["image_siglip_indices"], source["image_siglip_scores"], claim_index),
                "lexical": rank_map(source["image_filename_indices"], source["image_filename_scores"], claim_index),
                "rrf": rank_map(source["image_rrf_indices"], source["image_rrf_scores"], claim_index),
                "final": rank_map(selected_image, image_final_scores, claim_index),
            }
            for modality, retained in (
                ("text", selected_text[claim_index][:text_depth]),
                ("image", selected_image[claim_index][:image_depth]),
            ):
                maps = text_maps if modality == "text" else image_maps
                for raw_index in retained:
                    corpus_index = int(raw_index)
                    final_rank, final_score = maps["final"][corpus_index]
                    dense_rank, dense_score = maps["dense"].get(corpus_index, (0, 0.0))
                    lexical_rank, lexical_score = maps["lexical"].get(corpus_index, (0, 0.0))
                    _, rrf_score = maps["rrf"].get(corpus_index, (0, 0.0))
                    if modality == "text":
                        row = sentences[corpus_index]
                        evidence_id = text_ids[corpus_index]
                        source_group = str(row["source_group"])
                        token_count = len(str(row["content"]).split())
                        _, cross_score = maps["cross"].get(corpus_index, (0, 0.0))
                        duplicate_group_size = 1
                    else:
                        row = images[corpus_index]
                        evidence_id = image_ids[corpus_index]
                        source_group = str(row["source_group"])
                        token_count = 0
                        cross_score = 0.0
                        duplicate_group_size = int(row["duplicate_group_size"])
                    features[cursor] = [
                        final_score,
                        final_score * final_score,
                        dense_score,
                        lexical_score,
                        rrf_score,
                        cross_score,
                        1.0 / dense_rank if dense_rank else 0.0,
                        1.0 / lexical_rank if lexical_rank else 0.0,
                        1.0 / final_rank if final_rank else 0.0,
                        float(modality == "text"),
                        float(modality == "image"),
                        math.log1p(token_count) / 10.0,
                        math.log1p(len(str(claim["claim"]).split())) / 10.0,
                        math.log1p(text_depth + image_depth) / 10.0,
                        math.log1p(duplicate_group_size) / 10.0,
                    ]
                    labels[cursor] = int(evidence_id in gold)
                    handle.write(
                        json.dumps(
                            {
                                "claim_index": claim_index,
                                "claim_id": str(claim["claim_id"]),
                                "split": str(claim["split"]),
                                "claim_label": str(claim["label"]),
                                "evidence_id": evidence_id,
                                "modality": modality,
                                "source_group": source_group,
                                "gold": evidence_id in gold,
                                "final_rank": final_rank,
                                "dense_rank": dense_rank or None,
                                "lexical_rank": lexical_rank or None,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    cursor += 1
    if cursor != len(features):
        raise RuntimeError(f"candidate cursor mismatch: {cursor} != {len(features)}")

    feature_path = output_root / "candidate_features.npz"
    np.savez_compressed(feature_path, features=features, gold=labels)
    claims_path = output_root / "claims.jsonl"
    full_gold_path = output_root / "full_gold.json"
    shutil.copyfile(frozen_root / "claims.jsonl", claims_path)
    shutil.copyfile(frozen_root / "full_gold.json", full_gold_path)

    text_metrics = _modality_metrics(
        claims,
        selected_text,
        text_ids,
        full_gold,
        modality="text",
        depth=50,
        population="all_eligible",
        partition_seed=partition_seed,
    )
    image_metrics = _modality_metrics(
        claims,
        selected_image,
        image_ids,
        full_gold,
        modality="image",
        depth=50,
        population="all_eligible",
        partition_seed=partition_seed,
    )
    packet_metrics, _ = _packet_metrics(
        claims,
        selected_text,
        selected_image,
        text_ids,
        image_ids,
        full_gold,
        text_depth=text_depth,
        image_depth=image_depth,
        population="all_eligible",
        partition_seed=partition_seed,
    )
    retrieval_metrics = {
        "text": {
            "dense_bm25_rrf_cross_encoder": text_metrics,
            "weighted_rrf_cross_fusion": text_metrics,
        },
        "image": {
            "siglip_filename_rrf": image_metrics,
            "weighted_siglip_filename_fusion": image_metrics,
        },
    }
    manifest = {
        "profile": "publication_confirmation_candidate_packet",
        "status": "pass",
        "selected_weights": {"text_rrf": text_weight, "image_siglip": image_weight},
        "counts": {
            "claims": len(claims),
            "candidate_rows": len(features),
            "gold_candidate_rows": int(labels.sum()),
        },
        "feature_names": feature_names,
        "retrieval_metrics": retrieval_metrics,
        "packet_ceiling": packet_metrics,
        "weighted_retrieval_result_sha256": sha256_file(
            retrieval_result_root / "retrieval_results.json"
        ),
        "prejoin_manifest_sha256": sha256_file(
            retrieval_result_root / "weighted_rankings.prejoin.manifest.json"
        ),
        "artifacts": {
            "features": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
            "candidates": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "claims": {"path": str(claims_path), "sha256": sha256_file(claims_path)},
            "full_gold": {"path": str(full_gold_path), "sha256": sha256_file(full_gold_path)},
        },
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "source_commit": _source_commit(project_root),
        "gold_loaded_only_after_prejoin_ranking_freeze": True,
        "formal_test_annotations_accessed": False,
    }
    manifest_path = output_root / "retrieval.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    assert_mocheg_development_sealed(dataset_root)
    return manifest


def audit_publication_candidates(output_root: Path) -> dict[str, object]:
    manifest_path = output_root / "retrieval.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = np.load(output_root / "candidate_features.npz", mmap_mode="r")
    features = arrays["features"]
    labels = arrays["gold"]
    full_gold_raw = json.loads((output_root / "full_gold.json").read_text(encoding="utf-8"))
    full_gold = {str(key): set(map(str, values)) for key, values in full_gold_raw.items()}
    failures: list[str] = []
    if manifest.get("formal_test_annotations_accessed"):
        failures.append("manifest reports formal-test access")
    if len(features) != len(labels) or len(features) != int(manifest["counts"]["candidate_rows"]):
        failures.append("feature, label, and manifest counts disagree")
    cursor = 0
    current_key: str | None = None
    current_ids: set[str] = set()
    current_count = 0
    expected_per_claim = 60
    with (output_root / "candidate_records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = f"{row['split']}:{row['claim_id']}"
            if current_key is not None and key != current_key:
                if current_count != expected_per_claim or len(current_ids) != expected_per_claim:
                    failures.append(f"invalid packet cardinality or duplicate in {current_key}")
                current_ids = set()
                current_count = 0
            current_key = key
            evidence_id = str(row["evidence_id"])
            current_ids.add(evidence_id)
            current_count += 1
            expected_gold = evidence_id in full_gold.get(key, set())
            if bool(row["gold"]) != expected_gold or int(labels[cursor]) != int(expected_gold):
                failures.append(f"gold mismatch at row {cursor}")
                break
            cursor += 1
    if current_key is not None and (
        current_count != expected_per_claim or len(current_ids) != expected_per_claim
    ):
        failures.append(f"invalid packet cardinality or duplicate in {current_key}")
    if cursor != len(features):
        failures.append(f"candidate rows read {cursor}, expected {len(features)}")
    for key, artifact in manifest["artifacts"].items():
        path = Path(str(artifact["path"]))
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            failures.append(f"artifact mismatch: {key}")
    return {
        "profile": "publication_confirmation_candidate_packet_audit",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "candidate_rows_checked": cursor,
        "manifest_sha256": sha256_file(manifest_path),
        "formal_test_annotations_accessed": False,
    }

