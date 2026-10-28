from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

import numpy as np

from trace_scv.data.io import sha256_file
from trace_scv.data.mocheg import assert_mocheg_development_sealed, load_mocheg_qrels
from trace_scv.mocheg_full_corpus import _ranking_metrics


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_full_corpus_features(
    dataset_root: Path, feature_path: Path
) -> dict[str, object]:
    assert_mocheg_development_sealed(dataset_root)
    manifest_path = feature_path.with_suffix(".manifest.json")
    sidecar_path = feature_path.with_suffix(".json")
    rankings_path = feature_path.with_suffix(".rankings.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    rankings = _read_jsonl(rankings_path)
    media_manifest_path = dataset_root / "media" / "full_development_media_manifest.json"
    media = json.loads(media_manifest_path.read_text(encoding="utf-8"))
    arrays = np.load(feature_path, allow_pickle=False)
    discrepancies: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            discrepancies.append(message)

    require(sha256_file(feature_path) == manifest["features_sha256"], "feature hash mismatch")
    require(sha256_file(sidecar_path) == manifest["sidecar_sha256"], "sidecar hash mismatch")
    require(sha256_file(rankings_path) == manifest["artifacts"]["rankings"]["sha256"], "ranking hash mismatch")
    require(sha256_file(media_manifest_path) == manifest["artifacts"]["media_manifest"]["sha256"], "media manifest hash mismatch")
    require(len(arrays["features"]) == len(arrays["gold"]) == len(sidecar["records"]), "feature alignment mismatch")
    require(manifest["corpus_documents"] == 91_822, "released text corpus is incomplete")
    require(media["status"] == "pass", "media acquisition did not pass")
    require(not media["failures"], "media download failures are present")
    require(not media["image_audit"]["unreadable"], "unreadable media are present")
    require(not manifest["claim_association_used_as_model_input"], "claim association was a model input")
    require(manifest["gold_loaded_after_rankings_computed"], "gold was not joined after ranking")
    require(not manifest["formal_test_repository_listed"], "test repository was listed")
    require(not manifest["formal_test_annotations_accessed"], "test annotations were accessed")

    ranking_by_key: dict[tuple[str, str, str], list[str]] = {}
    for row in rankings:
        key = (str(row["split"]), str(row["claim_id"]), str(row["modality"]))
        entries = list(row["ranking"])
        ids = [str(entry["evidence_id"]) for entry in entries]
        ranks = [int(entry["rank"]) for entry in entries]
        scores = [float(entry["score"]) for entry in entries]
        require(key not in ranking_by_key, f"duplicate ranking {key}")
        require(ranks == list(range(1, len(entries) + 1)), f"invalid ranks {key}")
        require(len(ids) == len(set(ids)), f"duplicate evidence in ranking {key}")
        require(all(left >= right for left, right in zip(scores, scores[1:])), f"unsorted scores {key}")
        require(len(ids) <= int(manifest["retrieval_config"]["audit_ranking_depth"]), f"ranking too deep {key}")
        ranking_by_key[key] = ids

    claim_keys = {(str(row["split"]), str(row["claim_id"])) for row in sidecar["records"]}
    require(len(claim_keys) == manifest["claims"], "claim count mismatch")
    require(len(ranking_by_key) == 2 * len(claim_keys), "missing modality rankings")
    qrel_gold: dict[tuple[str, str], set[str]] = {}
    for split, folder in (("train", "train"), ("validation", "val")):
        split_claims = {claim_id for row_split, claim_id in claim_keys if row_split == split}
        for modality, filename in (("text", "text_evidence_qrels_sentence_level.csv"), ("image", "img_evidence_qrels.csv")):
            for qrel in load_mocheg_qrels(dataset_root / "metadata" / folder / filename):
                if qrel.claim_id in split_claims and qrel.relevant:
                    qrel_gold.setdefault((split, qrel.claim_id), set()).add(f"{modality}:{qrel.document_id}")
    sidecar_gold = {
        claim_id: set(map(str, ids))
        for claim_id, ids in sidecar["full_gold_evidence_ids_by_claim"].items()
    }
    for split, claim_id in claim_keys:
        require(sidecar_gold[claim_id] == qrel_gold.get((split, claim_id), set()), f"full gold mismatch {split}:{claim_id}")

    metric_rows: list[dict[str, object]] = []
    for split, claim_id in sorted(claim_keys):
        for modality in ("text", "image"):
            gold = {item for item in sidecar_gold[claim_id] if item.startswith(f"{modality}:")}
            if gold:
                metric_rows.append({"modality": modality, **_ranking_metrics(ranking_by_key[(split, claim_id, modality)], gold)})
    recomputed_metrics = {
        modality: {
            key: fmean(float(row[key]) for row in metric_rows if row["modality"] == modality)
            for key in ("recall_at_5", "recall_at_10", "recall_at_50", "map_at_60", "ndcg_at_60")
        }
        for modality in ("text", "image")
    }
    for modality, values in recomputed_metrics.items():
        for key, value in values.items():
            require(abs(value - float(manifest["retrieval_metrics"][modality][key])) <= 1e-12, f"{modality} {key} mismatch")
    labels_by_record = [
        int(str(row["evidence_id"]) in sidecar_gold[str(row["claim_id"])])
        for row in sidecar["records"]
    ]
    require(labels_by_record == arrays["gold"].astype(int).tolist(), "feature gold labels mismatch")
    result = {
        "profile": "mocheg_full_corpus_independent_audit",
        "status": "pass" if not discrepancies else "fail",
        "discrepancies": discrepancies,
        "recomputed": {
            "claims": len(claim_keys),
            "feature_rows": len(sidecar["records"]),
            "gold_feature_rows": sum(labels_by_record),
            "retrieval_metrics": recomputed_metrics,
            "formal_test_repository_listed": False,
            "formal_test_annotations_accessed": False,
        },
        "artifact_hashes": {
            "features": sha256_file(feature_path),
            "sidecar": sha256_file(sidecar_path),
            "rankings": sha256_file(rankings_path),
            "manifest": sha256_file(manifest_path),
            "media_manifest": sha256_file(media_manifest_path),
        },
    }
    feature_path.with_suffix(".audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
