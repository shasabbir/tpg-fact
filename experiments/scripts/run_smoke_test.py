from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from data.io import write_json, write_jsonl
from data.schema import LABELS
from data.splits import assert_no_group_leakage, partition
from data.synthetic import build_synthetic_records
from data.temporal import filter_time_valid
from evaluation.metrics import (
    area_under_risk_coverage,
    classification_metrics,
    expected_calibration_error,
    macro_ovr_auroc,
    multiclass_brier,
    negative_log_likelihood,
    risk_at_coverage,
)
from models.smoke import SmokeHierarchicalClassifier
from provenance.clustering import assign_source_clusters
from provenance.graph import build_provenance_graph
from retrieval.hybrid import bm25_rank, reciprocal_rank_fusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic TPG-Fact integration check")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_root = config_path.parent.parent
    output_directory = experiment_root / config["outputs"]["smoke_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    claims, all_evidence = build_synthetic_records()
    assert_no_group_leakage(claims)
    split_records = partition(claims)
    evidence_by_id = {item.evidence_id: item for item in all_evidence}
    classifier = SmokeHierarchicalClassifier(minimum_clusters=1)

    predictions = []
    graph_summaries = []
    retrieval_summaries = []
    for claim in claims:
        candidates = [evidence_by_id[evidence_id] for evidence_id in claim.evidence_ids]
        valid = filter_time_valid(candidates, claim.decision_cutoff, {"factcheck.example"})
        clustered = assign_source_clusters(
            valid,
            title_jaccard_threshold=config["provenance"]["title_jaccard_threshold"],
            image_phash_hamming_threshold=config["provenance"]["image_phash_hamming_threshold"],
        )
        lexical = bm25_rank(claim.claim_text, {item.evidence_id: item.title for item in clustered})
        visual = [
            item.evidence_id
            for item in clustered
            if item.image_phash and item.image_phash == claim.image_phash
        ]
        fused = reciprocal_rank_fusion([lexical, visual], k=config["retrieval"]["rrf_k"])
        ranked_ids = [item_id for item_id, _ in fused][: config["retrieval"]["retained_limit"]]
        selected = [next(item for item in clustered if item.evidence_id == item_id) for item_id in ranked_ids]
        graph = build_provenance_graph(claim, selected)
        prediction = classifier.predict(claim, selected)
        predictions.append(prediction)
        graph_summaries.append(
            {
                "claim_id": claim.claim_id,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "source_cluster_count": len(set(prediction.source_cluster_ids)),
            }
        )
        retrieval_summaries.append(
            {
                "claim_id": claim.claim_id,
                "candidate_count": len(candidates),
                "time_valid_count": len(valid),
                "ranked_evidence_ids": ranked_ids,
            }
        )

    truth = [claim.normalized_label for claim in claims]
    predicted = [prediction.predicted_label for prediction in predictions]
    probabilities = np.array(
        [[prediction.class_probabilities[label] for label in LABELS] for prediction in predictions]
    )
    confidences = probabilities.max(axis=1)
    metrics = classification_metrics(truth, predicted)
    metrics.update(
        {
            "brier": multiclass_brier(truth, probabilities),
            "nll": negative_log_likelihood(truth, probabilities),
            "ece": expected_calibration_error(truth, probabilities, bins=5),
            "macro_ovr_auroc": macro_ovr_auroc(truth, probabilities),
            "aurc": area_under_risk_coverage(truth, predicted, confidences),
            "risk_at_80_coverage": risk_at_coverage(truth, predicted, confidences, 0.80),
            "split_counts": {key: len(value) for key, value in split_records.items()},
            "status": "synthetic_software_diagnostic_only",
        }
    )

    write_jsonl(output_directory / "claims.jsonl", (claim.to_dict() for claim in claims))
    write_jsonl(output_directory / "evidence.jsonl", (item.to_dict() for item in all_evidence))
    write_jsonl(output_directory / "predictions.jsonl", (item.to_dict() for item in predictions))
    write_json(output_directory / "metrics.json", metrics)
    write_json(output_directory / "graphs.json", graph_summaries)
    write_json(output_directory / "retrieval.json", retrieval_summaries)

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

